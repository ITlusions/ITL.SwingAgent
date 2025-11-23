"""Ross Cameron-style market scanner with notification hooks.

This module adds a Pydantic agent that evaluates symbols against five
"pillars" commonly used in Ross Cameron's momentum scanning approach.
It is deliberately notification-first: as soon as a symbol clears a
minimum number of pillars, the agent fires messages to multiple chat
channels so downstream automations can react quickly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Dict, List, Optional
import logging

import pandas as pd
import yfinance as yf
from pydantic import BaseModel, Field, validator

from .data import load_ohlcv

logger = logging.getLogger(__name__)


class PillarName(str, Enum):
    """Five Ross Cameron-inspired pillars used for screening."""

    GAP_MOMENTUM = "gap_momentum"
    RELATIVE_VOLUME = "relative_volume"
    LOW_FLOAT = "low_float"
    MOMENTUM_STACKING = "momentum_stacking"
    HIGH_TIGHT_FLAG = "high_tight_flag"


class PillarAssessment(BaseModel):
    """Per-pillar evaluation result."""

    name: PillarName
    met: bool
    score: float
    detail: str


class RossCameronScreeningResult(BaseModel):
    """Aggregate screening output for a single symbol."""

    symbol: str
    asof: datetime
    pillars: List[PillarAssessment]
    met_count: int
    triggered: bool

    @property
    def met_pillars(self) -> List[PillarAssessment]:
        return [pillar for pillar in self.pillars if pillar.met]


class NotificationChannel(str, Enum):
    WHATSAPP = "whatsapp"
    TEAMS = "teams"
    TELEGRAM = "telegram"
    SIGNAL = "signal"


class NotificationConfig(BaseModel):
    """Configures where and when alerts are published."""

    channels: List[NotificationChannel] = Field(
        default_factory=lambda: [
            NotificationChannel.WHATSAPP,
            NotificationChannel.TEAMS,
            NotificationChannel.TELEGRAM,
            NotificationChannel.SIGNAL,
        ]
    )
    min_pillars_to_alert: int = 2
    channel_senders: Dict[NotificationChannel, Callable[[str], None]] = Field(
        default_factory=dict
    )

    class Config:
        arbitrary_types_allowed = True


class NotificationDispatcher(BaseModel):
    """Sends alerts to chat systems without hard-coding transport logic."""

    config: NotificationConfig = Field(default_factory=NotificationConfig)

    class Config:
        arbitrary_types_allowed = True

    def _build_message(self, result: RossCameronScreeningResult) -> str:
        pillar_summary = ", ".join(
            f"{p.name.value}:{p.score:.2f}" for p in result.pillars
        )
        return (
            f"Ross Cameron scan | {result.symbol} | "
            f"{len(result.met_pillars)}/{len(result.pillars)} pillars | "
            f"met: {[p.name.value for p in result.met_pillars]} | "
            f"details: {pillar_summary}"
        )

    def dispatch(self, result: RossCameronScreeningResult) -> None:
        """Send a formatted alert to every configured channel."""

        if not result.triggered:
            return

        message = self._build_message(result)
        for channel in self.config.channels:
            sender = self.config.channel_senders.get(channel)
            if sender:
                try:
                    sender(message)
                except Exception:  # pragma: no cover - defensive logging only
                    logger.exception("Failed to send %s alert", channel.value)
            else:
                logger.info(
                    "No sender configured for %s; queued message: %s",
                    channel.value,
                    message,
                )


class RossCameronAgent(BaseModel):
    """Pydantic agent that screens symbols against Ross Cameron's pillars."""

    watchlist: List[str]
    lookback_days: int = 60
    data_loader: Callable[..., pd.DataFrame] = Field(default=load_ohlcv)
    gap_threshold: float = 0.03
    relative_volume_threshold: float = 1.5
    low_float_max_shares: float = 60_000_000
    momentum_fast_ema: int = 9
    momentum_slow_ema: int = 20
    high_tight_flag_lookback: int = 30
    min_pillars_to_trigger: int = 2
    dispatcher: NotificationDispatcher = Field(default_factory=NotificationDispatcher)
    benchmark_symbol: str = "SPY"

    class Config:
        arbitrary_types_allowed = True

    @validator("watchlist")
    def _upper_watchlist(cls, value: List[str]) -> List[str]:
        return [symbol.upper() for symbol in value]

    def scan(self) -> List[RossCameronScreeningResult]:
        """Scan the entire watchlist and fire notifications when warranted."""

        results: List[RossCameronScreeningResult] = []
        benchmark = self._safe_load(self.benchmark_symbol)
        for symbol in self.watchlist:
            df = self._safe_load(symbol)
            if df is None:
                continue
            missing_columns = self._missing_columns(df)
            if missing_columns:
                logger.warning(
                    "Skipping %s due to missing OHLCV columns: %s",
                    symbol,
                    ", ".join(sorted(missing_columns)),
                )
                results.append(self._missing_data_result(symbol, missing_columns))
                continue
            result = self.evaluate_symbol(symbol, df, benchmark)
            results.append(result)
            self.dispatcher.dispatch(result)
        return results

    def evaluate_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None,
        ) -> RossCameronScreeningResult:
        """Evaluate a single symbol against all five pillars."""

        missing_columns = self._missing_columns(df)
        if missing_columns:
            return self._missing_data_result(symbol, missing_columns)

        pillars = [
            self._pillar_gap(df),
            self._pillar_relative_volume(df),
            self._pillar_low_float(symbol),
            self._pillar_momentum(df),
            self._pillar_high_tight_flag(df, benchmark_df),
        ]

        met_count = sum(1 for pillar in pillars if pillar.met)
        triggered = met_count >= self.min_pillars_to_trigger
        return RossCameronScreeningResult(
            symbol=symbol,
            asof=datetime.now(timezone.utc),
            pillars=pillars,
            met_count=met_count,
            triggered=triggered,
        )

    def _pillar_gap(self, df: pd.DataFrame) -> PillarAssessment:
        if len(df) < 2:
            return PillarAssessment(
                name=PillarName.GAP_MOMENTUM,
                met=False,
                score=0.0,
                detail="Insufficient bars for gap calculation",
            )

        prev_close = float(df["close"].iloc[-2])
        open_price = float(df["open"].iloc[-1])
        gap_pct = (open_price - prev_close) / max(prev_close, 1e-9)
        met = gap_pct >= self.gap_threshold
        return PillarAssessment(
            name=PillarName.GAP_MOMENTUM,
            met=met,
            score=gap_pct,
            detail=f"Gap {gap_pct:.2%} vs threshold {self.gap_threshold:.0%}",
        )

    def _pillar_relative_volume(self, df: pd.DataFrame) -> PillarAssessment:
        if len(df) < 12:
            return PillarAssessment(
                name=PillarName.RELATIVE_VOLUME,
                met=False,
                score=0.0,
                detail="Need 12+ daily bars for relative volume",
            )

        trailing_volume = float(df["volume"].iloc[-11:-1].mean())
        today_volume = float(df["volume"].iloc[-1])
        relative_vol = today_volume / max(trailing_volume, 1e-9)
        met = relative_vol >= self.relative_volume_threshold
        return PillarAssessment(
            name=PillarName.RELATIVE_VOLUME,
            met=met,
            score=relative_vol,
            detail=(
                f"Relative volume {relative_vol:.2f}x vs "
                f"threshold {self.relative_volume_threshold:.1f}x"
            ),
        )

    def _pillar_low_float(self, symbol: str) -> PillarAssessment:
        try:
            ticker = yf.Ticker(symbol)
            float_data = getattr(ticker, "fast_info", {})
            float_shares = None
            if isinstance(float_data, dict):
                float_shares = float_data.get("float_shares") or float_data.get(
                    "floatShares"
                )
            if float_shares is None:
                info = getattr(ticker, "info", {})
                if isinstance(info, dict):
                    float_shares = info.get("floatShares")
        except Exception:  # pragma: no cover - external API access
            logger.exception("Failed to fetch float data for %s", symbol)
            float_shares = None

        if not float_shares:
            return PillarAssessment(
                name=PillarName.LOW_FLOAT,
                met=False,
                score=0.0,
                detail="Float data unavailable",
            )

        float_millions = float(float_shares) / 1_000_000
        met = float_shares <= self.low_float_max_shares
        return PillarAssessment(
            name=PillarName.LOW_FLOAT,
            met=met,
            score=float_millions,
            detail=(
                f"Float {float_millions:.1f}M vs cap {self.low_float_max_shares/1e6:.0f}M"
            ),
        )

    def _pillar_momentum(self, df: pd.DataFrame) -> PillarAssessment:
        if len(df) < max(self.momentum_fast_ema, self.momentum_slow_ema) + 5:
            return PillarAssessment(
                name=PillarName.MOMENTUM_STACKING,
                met=False,
                score=0.0,
                detail="Not enough data for EMA stack",
            )

        ema_fast = df["close"].ewm(span=self.momentum_fast_ema, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.momentum_slow_ema, adjust=False).mean()
        close = float(df["close"].iloc[-1])
        fast = float(ema_fast.iloc[-1])
        slow = float(ema_slow.iloc[-1])
        met = close > fast > slow
        momentum_score = (close - slow) / max(slow, 1e-9)
        return PillarAssessment(
            name=PillarName.MOMENTUM_STACKING,
            met=met,
            score=momentum_score,
            detail=(
                f"Close {close:.2f} > EMA{self.momentum_fast_ema} {fast:.2f} > "
                f"EMA{self.momentum_slow_ema} {slow:.2f}"
            ),
        )

    def _pillar_high_tight_flag(
        self, df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame]
    ) -> PillarAssessment:
        if len(df) < self.high_tight_flag_lookback + 1:
            return PillarAssessment(
                name=PillarName.HIGH_TIGHT_FLAG,
                met=False,
                score=0.0,
                detail="Need more history for high-tight flag",
            )

        recent_window = df.iloc[-self.high_tight_flag_lookback :]
        prior_high = float(recent_window["high"].iloc[:-1].max())
        last_close = float(df["close"].iloc[-1])
        breakout = last_close >= prior_high

        rs_score = 0.0
        if benchmark_df is not None and len(benchmark_df) >= len(recent_window):
            symbol_return = recent_window["close"].iloc[-1] / recent_window["close"].iloc[0]
            bench_window = benchmark_df.iloc[-self.high_tight_flag_lookback :]
            bench_return = bench_window["close"].iloc[-1] / bench_window["close"].iloc[0]
            if bench_return:
                rs_score = symbol_return / bench_return

        met = breakout and rs_score >= 1.05 if rs_score else breakout
        detail = (
            f"Breakout {'yes' if breakout else 'no'}; relative strength {rs_score:.2f}"
        )
        return PillarAssessment(
            name=PillarName.HIGH_TIGHT_FLAG,
            met=met,
            score=rs_score if rs_score else 0.0,
            detail=detail,
        )

    def _safe_load(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            return self.data_loader(
                symbol,
                interval="1d",
                lookback_days=self.lookback_days,
            )
        except Exception:
            logger.exception("Failed to load OHLCV for %s", symbol)
            return None

    def _missing_columns(self, df: pd.DataFrame) -> List[str]:
        required = {"open", "high", "low", "close", "volume"}
        return [col for col in required if col not in df.columns]

    def _missing_data_result(
        self, symbol: str, missing_columns: List[str]
    ) -> RossCameronScreeningResult:
        detail = f"Missing OHLCV columns: {', '.join(sorted(missing_columns))}"
        pillars = [
            PillarAssessment(name=name, met=False, score=0.0, detail=detail)
            for name in PillarName
        ]
        return RossCameronScreeningResult(
            symbol=symbol,
            asof=datetime.now(timezone.utc),
            pillars=pillars,
            met_count=0,
            triggered=False,
        )
