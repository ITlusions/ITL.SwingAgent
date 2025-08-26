from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from .calibration import calibrated_winrate
from .config import get_config
from .data import load_ohlcv
from .features import build_setup_vector, time_of_day_bucket, vol_regime_from_series
from .llm_predictor import llm_build_action_plan, llm_extra_prediction
from .models import EntryPlan, TradeSignal, TrendState
from .storage import record_signal
from .strategy import build_entry, label_trend
from .vectorstore import add_vector, extended_stats, filter_neighbors, knn


def _clip01(x: float) -> float:
    """Clip a value to the range [0, 1].
    
    Args:
        x: Value to clip.
        
    Returns:
        float: Value clipped to [0, 1] range.
    """
    return max(0.0, min(1.0, x))


def _context_from_df(df: pd.DataFrame) -> dict[str, Any]:
    """Extract market context from price data.
    
    Args:
        df: OHLCV DataFrame with at least 2 bars.
        
    Returns:
        Dict containing prev_range_pct, gap_pct, and atr_pct.
    """
    prev_range_pct = float(
        (df["high"].iloc[-2] - df["low"].iloc[-2]) /
        max(1e-9, df["close"].iloc[-2])
    )
    gap_pct = float(
        (df["open"].iloc[-1] - df["close"].iloc[-2]) /
        max(1e-9, df["close"].iloc[-2])
    )
    from .indicators import atr
    cfg = get_config()
    atr14 = float(atr(df, cfg.ATR_PERIOD).iloc[-1])
    atr_pct = atr14 / max(1e-9, df["close"].iloc[-1])
    return {
        "prev_range_pct": prev_range_pct,
        "gap_pct": gap_pct,
        "atr_pct": atr_pct
    }


def _rel_strength(df_sym: pd.DataFrame, df_bench: pd.DataFrame,
                 lookback: int = None) -> float:
    """Calculate relative strength ratio vs benchmark over lookback period.
    
    Measures how well a symbol has performed relative to a benchmark
    (sector ETF or SPY) over the specified period. Values > 1.0 indicate
    outperformance, < 1.0 indicate underperformance.
    
    Args:
        df_sym: Symbol OHLCV price DataFrame.
        df_bench: Benchmark OHLCV price DataFrame.
        lookback: Number of days to calculate relative performance over.
                 Uses config RS_LOOKBACK_DAYS if None.
        
    Returns:
        float: Relative strength ratio. 
               - > 1.0: Symbol outperforming benchmark
               - = 1.0: Symbol matching benchmark performance  
               - < 1.0: Symbol underperforming benchmark
               - NaN: Insufficient data or calculation error
               
    Example:
        >>> df_aapl = load_ohlcv("AAPL", "30m", 30)
        >>> df_spy = load_ohlcv("SPY", "30m", 30)
        >>> rs = _rel_strength(df_aapl, df_spy, lookback=20)
        >>> if rs > 1.05:
        ...     print("AAPL strongly outperforming SPY")
        >>> elif rs < 0.95:
        ...     print("AAPL underperforming SPY")
        >>> else:
        ...     print("AAPL performance in line with SPY")
        
    Note:
        Calculation: (symbol_return / benchmark_return) over lookback period.
        Used for sector rotation and momentum analysis in signal generation.
    """
    if df_bench is None:
        return float("nan")

    if lookback is None:
        lookback = get_config().RS_LOOKBACK_DAYS

    if len(df_sym) < lookback + 1 or len(df_bench) < lookback + 1:
        return float("nan")

    cs = df_sym["close"].iloc[-1] / df_sym["close"].iloc[-(lookback + 1)]
    cb = df_bench["close"].iloc[-1] / df_bench["close"].iloc[-(lookback + 1)]

    if cb == 0:
        return float("nan")

    return float(cs / cb)

class SwingAgent:
    """Main SwingAgent class for generating trading signals.
    
    Combines technical analysis, machine learning pattern matching, and LLM insights
    to generate comprehensive 1-2 day swing trading signals.
    
    Args:
        interval: Trading timeframe (e.g., "30m", "1h", "4h").
        lookback_days: Days of historical data to analyze.
        log_db: Path to signals database (uses centralized db if None).
        vec_db: Path to vector database (uses centralized db if None).
        use_llm: Whether to use LLM for additional insights.
        llm_extras: Whether to generate detailed LLM action plans.
        sector_symbol: Symbol to use for sector relative strength.
    """

    def __init__(self, interval: str = "30m", lookback_days: int = 30,
                 log_db: str | None = None, vec_db: str | None = None,
                 use_llm: bool = True, llm_extras: bool = True,
                 sector_symbol: str = "XLK"):
        self.interval = interval
        self.lookback_days = lookback_days

        # Use centralized database by default - both signals and vectors in same file
        if log_db is None and vec_db is None:
            # Default to centralized database
            self.log_db = "data/swing_agent.sqlite"
            self.vec_db = "data/swing_agent.sqlite"
        else:
            # Backward compatibility - use provided paths but ensure they point to centralized db
            self.log_db = log_db or "data/swing_agent.sqlite"
            self.vec_db = vec_db or "data/swing_agent.sqlite"

        self.use_llm = use_llm
        self.llm_extras = llm_extras
        self.sector_symbol = sector_symbol

    def analyze(self, symbol: str) -> TradeSignal:
        """Analyze a symbol and generate a trading signal.
        
        Args:
            symbol: Stock symbol to analyze.
            
        Returns:
            TradeSignal: Complete trading signal with entry plan and analysis.
            
        Raises:
            RuntimeError: If insufficient data is available for analysis.
        """
        df = load_ohlcv(symbol, interval=self.interval, lookback_days=self.lookback_days)
        return self.analyze_df(symbol, df)

    def analyze_df(self, symbol: str, df: pd.DataFrame) -> TradeSignal:
        """Analyze price data and generate a comprehensive trading signal.
        
        This is the main orchestration method that coordinates all analysis components:
        1. Market context building (gaps, volatility, time of day)
        2. Technical analysis (trend, entry plans)
        3. Multi-timeframe analysis
        4. ML expectations from vector similarity
        5. LLM insights and action plans
        6. Signal assembly and persistence
        
        Args:
            symbol: Stock symbol being analyzed.
            df: OHLCV price DataFrame with sufficient history.
            
        Returns:
            TradeSignal: Complete signal with all analysis components.
            
        Raises:
            RuntimeError: If insufficient data bars for analysis.
        """
        # Validate data quality
        df = df[~df.index.duplicated(keep="last")]
        cfg = get_config()
        if len(df) < cfg.MIN_DATA_BARS:
            raise RuntimeError("Not enough data bars for analysis")

        # 1. Build market context and perform technical analysis
        context = self._build_market_context(symbol, df)
        trend, entry = self._perform_technical_analysis(df)

        # 2. Get multi-timeframe analysis
        mtf_data = self._get_multitimeframe_analysis(symbol, trend)

        # 3. Calculate confidence score
        confidence = self._calculate_confidence(trend, entry, mtf_data, context)

        # 4. Get ML expectations from vector store
        ml_expectations = self._get_ml_expectations(symbol, df, trend, entry, context)
        confidence = self._adjust_confidence_with_priors(confidence, ml_expectations)

        # 5. Get LLM insights
        llm_insights = self._get_llm_insights(symbol, df, trend, entry, confidence, ml_expectations, context)
        confidence = self._adjust_confidence_with_llm(confidence, entry, llm_insights)

        # 6. Assemble and return signal
        return self._assemble_signal(symbol, df, trend, entry, confidence,
                                   mtf_data, ml_expectations, llm_insights, context)
    def _build_market_context(self, symbol: str, df: pd.DataFrame) -> dict[str, Any]:
        """Build comprehensive market context for analysis.
        
        Args:
            symbol: Stock symbol being analyzed.
            df: OHLCV price DataFrame.
            
        Returns:
            Dict containing market context data including volatility regime,
            time of day, relative strength, and price context.
        """
        # Basic price context (gaps, ranges, ATR)
        price_context = _context_from_df(df)

        # Relative strength vs sector and SPY
        rs_sector, rs_spy = self._calculate_relative_strength(symbol, df)

        # Time of day and volatility regime
        try:
            tod = time_of_day_bucket(df.index[-1].tz_convert("America/New_York"))
        except Exception:
            tod = "mid"

        vol_regime = vol_regime_from_series(df["close"])

        return {
            **price_context,
            "rs_sector": rs_sector,
            "rs_spy": rs_spy,
            "tod_bucket": tod,
            "vol_regime": vol_regime,
            "price": float(df['close'].iloc[-1]),
            "prev_high": float(df['high'].iloc[-2]),
            "prev_low": float(df['low'].iloc[-2])
        }

    def _calculate_relative_strength(self, symbol: str, df: pd.DataFrame) -> tuple[float, float]:
        """Calculate relative strength vs sector and SPY.
        
        Args:
            symbol: Stock symbol being analyzed.
            df: OHLCV price DataFrame.
            
        Returns:
            Tuple of (rs_sector, rs_spy) relative strength values.
        """
        cfg = get_config()

        # Sector relative strength
        try:
            df_sector = load_ohlcv(self.sector_symbol, interval=self.interval,
                                 lookback_days=self.lookback_days)
            rs_sector = _rel_strength(df, df_sector, cfg.RS_LOOKBACK_DAYS)
        except Exception:
            rs_sector = float("nan")

        # SPY relative strength
        try:
            df_spy = load_ohlcv("SPY", interval=self.interval,
                              lookback_days=self.lookback_days)
            rs_spy = _rel_strength(df, df_spy, cfg.RS_LOOKBACK_DAYS)
        except Exception:
            rs_spy = float("nan")

        return rs_sector, rs_spy

    def _perform_technical_analysis(self, df: pd.DataFrame) -> tuple[TrendState, EntryPlan | None]:
        """Perform core technical analysis.
        
        Args:
            df: OHLCV price DataFrame.
            
        Returns:
            Tuple of (trend_state, entry_plan).
        """
        trend = label_trend(df)
        entry = build_entry(df, trend)
        return trend, entry

    def _get_multitimeframe_analysis(self, symbol: str, trend: TrendState) -> dict[str, Any]:
        """Analyze multiple timeframes for trend alignment.
        
        Args:
            symbol: Stock symbol being analyzed.
            trend: Current timeframe trend state.
            
        Returns:
            Dict containing MTF analysis results.
        """
        # Multi-timeframe alignment
        try:
            df15 = load_ohlcv(symbol, interval="15m", lookback_days=self.lookback_days)
            df1h = load_ohlcv(symbol, interval="1h", lookback_days=self.lookback_days)
            t15 = label_trend(df15).label.value
            t1h = label_trend(df1h).label.value
        except Exception:
            t15 = t1h = None

        # Calculate alignment score
        mtf_align = 0
        for t in (t15, t1h):
            if not t:
                continue
            if "up" in t and "up" in trend.label.value:
                mtf_align += 1
            elif "down" in t and "down" in trend.label.value:
                mtf_align += 1
            elif "sideways" in t or "sideways" in trend.label.value:
                mtf_align += 0
            else:
                mtf_align -= 1

        return {
            "mtf_15m_trend": t15,
            "mtf_1h_trend": t1h,
            "mtf_alignment": mtf_align
        }

    def _calculate_confidence(self, trend: TrendState, entry: EntryPlan | None,
                            mtf_data: dict[str, Any], context: dict[str, Any]) -> float:
        """Calculate base confidence score.
        
        Args:
            trend: Current trend state.
            entry: Entry plan if available.
            mtf_data: Multi-timeframe analysis data.
            context: Market context data.
            
        Returns:
            float: Base confidence score [0, 1].
        """
        cfg = get_config()

        # Base confidence by trend strength
        base = cfg.BASE_CONFIDENCE.get(trend.label.value, 0.2)

        # R-multiple bonus
        r_add = 0.0
        if entry is not None:
            r_add = min(cfg.MAX_R_MULTIPLE_BONUS,
                       max(0.0, cfg.R_MULTIPLE_FACTOR * entry.r_multiple))

        # Multi-timeframe alignment bonus
        mtf_bonus = cfg.MTF_ALIGNMENT_BONUS * mtf_data["mtf_alignment"]

        # Relative strength bonus
        rs_bonus = 0.0
        rs_sector = context["rs_sector"]
        if not np.isnan(rs_sector) and rs_sector > cfg.RS_SECTOR_THRESHOLD:
            rs_bonus = cfg.RS_SECTOR_BONUS

        return base + r_add + mtf_bonus + rs_bonus

    def _get_ml_expectations(self, symbol: str, df: pd.DataFrame, trend: TrendState,
                           entry: EntryPlan | None, context: dict[str, Any]) -> dict[str, Any]:
        """Get ML-based expectations from vector similarity analysis.
        
        Args:
            symbol: Stock symbol being analyzed.
            df: OHLCV price DataFrame.
            trend: Current trend state.
            entry: Entry plan if available.
            context: Market context data.
            
        Returns:
            Dict containing ML expectations and statistics.
        """
        if self.vec_db is None:
            return {
                "expected_r": None,
                "expected_winrate": None,
                "expected_hold_bars": None,
                "expected_hold_days": None,
                "expected_win_hold_bars": None,
                "expected_loss_hold_bars": None,
                "prior_confidence": 0.5,
                "prior_description": ""
            }

        cfg = get_config()

        # Build feature vector
        llm_conf = 0.0  # Will be updated after LLM analysis
        session_bin = {"open": 0, "mid": 1, "close": 2}.get(context["tod_bucket"], 1)

        vec = build_setup_vector(
            price=context["price"],
            trend=trend,
            entry=entry,
            prev_range_pct=context["prev_range_pct"],
            gap_pct=context["gap_pct"],
            atr_pct=context["atr_pct"],
            session_bin=session_bin,
            llm_conf=llm_conf
        )

        # Store vector for future training
        try:
            add_vector(
                self.vec_db,
                vid=f"{symbol}-{df.index[-1].isoformat()}",
                ts_utc=df.index[-1].isoformat(),
                symbol=symbol,
                timeframe=self.interval,
                vec=vec,
                realized_r=None,
                exit_reason=None,
                payload={
                    "note": "pending",
                    "vector_version": cfg.VECTOR_VERSION,
                    "vol_regime": context["vol_regime"]
                }
            )
        except Exception:
            pass

        # Get similar patterns
        nbrs = (knn(self.vec_db, query_vec=vec, k=cfg.KNN_DEFAULT_K, symbol=symbol) or
                knn(self.vec_db, query_vec=vec, k=cfg.KNN_DEFAULT_K))

        # Filter by volatility regime
        nbrs = filter_neighbors(nbrs, vol_regime=context["vol_regime"])

        # Calculate statistics
        stats = extended_stats(nbrs)

        prior_conf = _clip01(0.5 + stats["avg_R"] / 4.0)

        prior_str = (
            f" Vector prior (k={stats['n']}) [{context['vol_regime']}] : "
            f"win={stats['p_win']*100:.0f}%, exp={stats['avg_R']:+.2f}R "
            f"(avg_win={stats['avg_win_R']:+.2f}R, avg_loss={stats['avg_loss_R']:+.2f}R), "
            f"PF={stats['profit_factor']}. "
            f"Median hold: {stats['median_hold_bars']} bars (~{stats['median_hold_days']} days); "
            f"win-median={stats['median_win_hold_bars']}, loss-median={stats['median_loss_hold_bars']}."
        )

        return {
            "expected_r": stats["p_win"] * stats["avg_win_R"] + (1 - stats["p_win"]) * stats["avg_loss_R"],
            "expected_winrate": stats["p_win"],
            "expected_hold_bars": stats["median_hold_bars"],
            "expected_hold_days": stats["median_hold_days"],
            "expected_win_hold_bars": stats["median_win_hold_bars"],
            "expected_loss_hold_bars": stats["median_loss_hold_bars"],
            "prior_confidence": prior_conf,
            "prior_description": prior_str
        }

    def _adjust_confidence_with_priors(self, confidence: float,
                                     ml_expectations: dict[str, Any]) -> float:
        """Adjust confidence based on ML priors.
        
        Args:
            confidence: Base confidence score.
            ml_expectations: ML expectations data.
            
        Returns:
            float: Adjusted confidence score.
        """
        prior_conf = ml_expectations["prior_confidence"]
        return _clip01(0.6 * confidence + 0.4 * prior_conf)

    def _get_llm_insights(self, symbol: str, df: pd.DataFrame, trend: TrendState,
                         entry: EntryPlan | None, confidence: float,
                         ml_expectations: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Get LLM-based insights and analysis.
        
        Args:
            symbol: Stock symbol being analyzed.
            df: OHLCV price DataFrame.
            trend: Current trend state.
            entry: Entry plan if available.
            confidence: Current confidence score.
            ml_expectations: ML expectations data.
            context: Market context data.
            
        Returns:
            Dict containing LLM insights and action plans.
        """
        if not self.use_llm:
            return {
                "llm_vote": None,
                "llm_explanation": None,
                "action_plan": None,
                "risk_notes": None,
                "scenarios": None
            }

        # Get LLM vote
        llm_vote = None
        llm_explanation = None

        try:
            from .indicators import fibonacci_range
            cfg = get_config()
            fib = fibonacci_range(df, lookback=cfg.FIB_LOOKBACK)

            vote = llm_extra_prediction(
                symbol=symbol,
                price=context["price"],
                ema20_slope=float(trend.ema_slope),
                rsi14=float(trend.rsi_14),
                price_above_ema=bool(trend.price_above_ema),
                prev_high=context["prev_high"],
                prev_low=context["prev_low"],
                fib_golden_low=float(fib.golden_low),
                fib_golden_high=float(fib.golden_high),
                fib_ext_1272=float(fib.levels["1.272"]),
                fib_ext_1618=float(fib.levels["1.618"]),
                atr14=float(context["atr_pct"] * context["price"])
            )

            llm_vote = {
                "trend_label": vote.trend_label,
                "entry_bias": vote.entry_bias,
                "entry_window_low": vote.entry_window_low,
                "entry_window_high": vote.entry_window_high,
                "confidence": vote.confidence
            }
            llm_explanation = vote.rationale

        except Exception:
            pass

        # Get detailed action plan
        action_plan = None
        risk_notes = None
        scenarios = None

        if self.llm_extras and entry is not None:
            try:
                sig_for_llm = {
                    "symbol": symbol,
                    "timeframe": self.interval,
                    "trend": {
                        "label": trend.label.value,
                        "ema_slope": float(trend.ema_slope),
                        "price_above_ema": bool(trend.price_above_ema),
                        "rsi_14": float(trend.rsi_14)
                    },
                    "entry": {
                        "side": entry.side.value,
                        "entry_price": entry.entry_price,
                        "stop_price": entry.stop_price,
                        "take_profit": entry.take_profit,
                        "r_multiple": entry.r_multiple,
                        "comment": entry.comment,
                        "fib_golden_low": entry.fib_golden_low,
                        "fib_golden_high": entry.fib_golden_high,
                        "fib_target_1": entry.fib_target_1,
                        "fib_target_2": entry.fib_target_2
                    },
                    "confidence": round(confidence, 2),
                    "expected_r": (round(ml_expectations["expected_r"], 3)
                                 if ml_expectations["expected_r"] is not None else None),
                    "expected_winrate": ml_expectations["expected_winrate"],
                    "expected_hold_bars": ml_expectations["expected_hold_bars"],
                    "expected_hold_days": ml_expectations["expected_hold_days"],
                    "expected_win_hold_bars": ml_expectations["expected_win_hold_bars"],
                    "expected_loss_hold_bars": ml_expectations["expected_loss_hold_bars"],
                    "enrich": {
                        "mtf_15m_trend": context.get("mtf_15m_trend"),
                        "mtf_1h_trend": context.get("mtf_1h_trend"),
                        "mtf_alignment": context.get("mtf_alignment"),
                        "rs_sector_20": context["rs_sector"],
                        "rs_spy_20": context["rs_spy"],
                        "tod_bucket": context["tod_bucket"],
                        "vol_regime": context["vol_regime"]
                    }
                }

                ap = llm_build_action_plan(signal_json=sig_for_llm, style="balanced")
                action_plan = ap.action_plan
                risk_notes = ap.risk_notes
                scenarios = ap.scenarios

            except Exception:
                pass

        return {
            "llm_vote": llm_vote,
            "llm_explanation": llm_explanation,
            "action_plan": action_plan,
            "risk_notes": risk_notes,
            "scenarios": scenarios
        }

    def _adjust_confidence_with_llm(self, confidence: float, entry: EntryPlan | None,
                                  llm_insights: dict[str, Any]) -> float:
        """Adjust confidence based on LLM agreement/disagreement.
        
        Args:
            confidence: Current confidence score.
            entry: Entry plan if available.
            llm_insights: LLM insights data.
            
        Returns:
            float: Final adjusted confidence score.
        """
        llm_vote = llm_insights.get("llm_vote")
        if not llm_vote or not entry:
            return confidence

        cfg = get_config()

        # Check LLM agreement with entry plan
        agrees = ((entry.side.value == "long" and llm_vote["entry_bias"] == "long") or
                 (entry.side.value == "short" and llm_vote["entry_bias"] == "short"))

        if agrees:
            confidence = min(1.0, confidence + cfg.LLM_AGREEMENT_BONUS * llm_vote["confidence"])
        else:
            confidence = max(0.0, confidence - cfg.LLM_DISAGREEMENT_PENALTY * llm_vote["confidence"])

        return confidence

    def _assemble_signal(self, symbol: str, df: pd.DataFrame, trend: TrendState,
                        entry: EntryPlan | None, confidence: float,
                        mtf_data: dict[str, Any], ml_expectations: dict[str, Any],
                        llm_insights: dict[str, Any], context: dict[str, Any]) -> TradeSignal:
        """Assemble the final trading signal from all analysis components.
        
        Args:
            symbol: Stock symbol being analyzed.
            df: OHLCV price DataFrame.
            trend: Current trend state.
            entry: Entry plan if available.
            confidence: Final confidence score.
            mtf_data: Multi-timeframe analysis data.
            ml_expectations: ML expectations data.
            llm_insights: LLM insights data.
            context: Market context data.
            
        Returns:
            TradeSignal: Complete assembled trading signal.
        """
        # Build reasoning narrative
        reasons = []

        # Technical analysis summary
        reasons.append(
            f"EMA20 slope {trend.ema_slope:.4f}, RSI14 {trend.rsi_14:.1f}. "
            f"Trend={trend.label.value}. MTF align={mtf_data['mtf_alignment']}, "
            f"RS_sector_20={(None if np.isnan(context['rs_sector']) else round(context['rs_sector'], 2))}"
        )

        # Entry plan summary
        if entry is None:
            reasons.append("No actionable setup per rules.")
        else:
            reasons.append(
                f"Proposed {entry.side.value.upper()} at {entry.entry_price:.2f}, "
                f"SL {entry.stop_price:.2f}, TP {entry.take_profit:.2f} "
                f"(R={entry.r_multiple:.2f}). {entry.comment}."
            )

        # ML prior summary
        if ml_expectations["prior_description"]:
            reasons.append(ml_expectations["prior_description"])

        # LLM summary
        llm_vote = llm_insights.get("llm_vote")
        llm_explanation = llm_insights.get("llm_explanation")
        if llm_vote and llm_explanation:
            reasons.append(
                f"LLM vote: trend={llm_vote['trend_label']}, "
                f"bias={llm_vote['entry_bias']}, conf={llm_vote['confidence']:.2f}. "
                f"{llm_explanation}"
            )

        # Create and store the signal
        signal = TradeSignal(
            symbol=symbol,
            timeframe=self.interval,
            asof=datetime.now(UTC).isoformat(),
            trend=trend,
            entry=entry,
            confidence=round(confidence, 2),
            reasoning=" ".join([s for s in reasons if s]),
            llm_vote=llm_insights.get("llm_vote"),
            llm_explanation=llm_insights.get("llm_explanation"),
            expected_r=(round(ml_expectations["expected_r"], 3)
                       if ml_expectations["expected_r"] is not None else None),
            expected_winrate=(
                calibrated_winrate(
                    ml_expectations["expected_winrate"], self.log_db
                )
                if ml_expectations["expected_winrate"] is not None
                else None
            ),
            expected_source=("vector_knn/v1.6.1" if self.vec_db is not None else None),
            expected_notes=("E[R]=p*avg_win+(1-p)*avg_loss; med hold from KNN neighbors (filtered by vol_regime)."
                          if self.vec_db is not None else None),
            expected_hold_bars=ml_expectations["expected_hold_bars"],
            expected_hold_days=ml_expectations["expected_hold_days"],
            expected_win_hold_bars=ml_expectations["expected_win_hold_bars"],
            expected_loss_hold_bars=ml_expectations["expected_loss_hold_bars"],
            action_plan=llm_insights.get("action_plan"),
            risk_notes=llm_insights.get("risk_notes"),
            scenarios=llm_insights.get("scenarios"),
            mtf_15m_trend=mtf_data.get("mtf_15m_trend"),
            mtf_1h_trend=mtf_data.get("mtf_1h_trend"),
            mtf_alignment=mtf_data.get("mtf_alignment"),
            rs_sector_20=(None if np.isnan(context["rs_sector"]) else float(context["rs_sector"])),
            rs_spy_20=(None if np.isnan(context["rs_spy"]) else float(context["rs_spy"])),
            sector_symbol=self.sector_symbol,
            tod_bucket=context["tod_bucket"],
            atr_pct=float(context["atr_pct"]),
            vol_regime=context["vol_regime"]
        )

        # Store signal in database
        if self.log_db:
            try:
                record_signal(signal, self.log_db)
            except Exception:
                pass  # Graceful degradation if storage fails

        return signal
