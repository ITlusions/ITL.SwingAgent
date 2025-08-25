from __future__ import annotations
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from .data import load_ohlcv
from .strategy import label_trend, build_entry
from .models import TradeSignal
from .llm_predictor import llm_extra_prediction, llm_build_action_plan
from .features import build_setup_vector, time_of_day_bucket, vol_regime_from_series
from .vectorstore import add_vector, knn, extended_stats, filter_neighbors
from .storage import record_signal

def _clip01(x: float) -> float: return max(0.0, min(1.0, x))

def _context_from_df(df: pd.DataFrame) -> dict:
    prev_range_pct = float((df["high"].iloc[-2] - df["low"].iloc[-2]) / max(1e-9, df["close"].iloc[-2]))
    gap_pct = float((df["open"].iloc[-1] - df["close"].iloc[-2]) / max(1e-9, df["close"].iloc[-2]))
    from .indicators import atr
    atr14 = float(atr(df, 14).iloc[-1])
    atr_pct = atr14 / max(1e-9, df["close"].iloc[-1])
    return {"prev_range_pct": prev_range_pct, "gap_pct": gap_pct, "atr_pct": atr_pct}

def _rel_strength(df_sym: pd.DataFrame, df_bench: pd.DataFrame, lookback: int = 20) -> float:
    if df_bench is None: return float("nan")
    if len(df_sym) < lookback+1 or len(df_bench) < lookback+1: return float("nan")
    cs = df_sym["close"].iloc[-1] / df_sym["close"].iloc[-(lookback+1)]
    cb = df_bench["close"].iloc[-1] / df_bench["close"].iloc[-(lookback+1)]
    if cb == 0: return float("nan")
    return float(cs / cb)

class SwingAgent:
    def __init__(self, interval: str="30m", lookback_days: int=30, log_db: str|None=None, vec_db: str|None=None, use_llm: bool=True, llm_extras: bool=True, sector_symbol: str="XLK"):
        self.interval=interval; self.lookback_days=lookback_days
        self.log_db=log_db; self.vec_db=vec_db
        self.use_llm=use_llm; self.llm_extras=llm_extras
        self.sector_symbol = sector_symbol

    def analyze(self, symbol: str) -> TradeSignal:
        df = load_ohlcv(symbol, interval=self.interval, lookback_days=self.lookback_days)
        return self.analyze_df(symbol, df)

    def analyze_df(self, symbol: str, df: pd.DataFrame) -> TradeSignal:
        df = df[~df.index.duplicated(keep="last")]
        if len(df) < 60: raise RuntimeError("Not enough data bars for analysis")

        trend = label_trend(df); plan = build_entry(df, trend)

        # Multi-timeframe alignment
        try:
            df15 = load_ohlcv(symbol, interval="15m", lookback_days=self.lookback_days)
            df1h = load_ohlcv(symbol, interval="1h", lookback_days=self.lookback_days)
            t15 = label_trend(df15).label.value; t1h = label_trend(df1h).label.value
        except Exception:
            t15 = t1h = None
        mtf_align = 0
        for t in (t15, t1h):
            if not t: continue
            if "up" in t and "up" in trend.label.value: mtf_align += 1
            elif "down" in t and "down" in trend.label.value: mtf_align += 1
            elif "sideways" in t or "sideways" in trend.label.value: mtf_align += 0
            else: mtf_align -= 1

        # Relative strength vs sector & SPY
        try: df_sector = load_ohlcv(self.sector_symbol, interval=self.interval, lookback_days=self.lookback_days)
        except Exception: df_sector = None
        try: df_spy = load_ohlcv("SPY", interval=self.interval, lookback_days=self.lookback_days)
        except Exception: df_spy = None
        rs_sector = _rel_strength(df, df_sector, 20) if df_sector is not None else float("nan")
        rs_spy = _rel_strength(df, df_spy, 20) if df_spy is not None else float("nan")

        # TOD & volatility regime
        try:
            tod = time_of_day_bucket(df.index[-1].tz_convert("America/New_York"))
        except Exception:
            tod = "mid"
        vol_reg = vol_regime_from_series(df["close"])
        ctx = _context_from_df(df)

        # Confidence
        base={"strong_up":0.35,"up":0.25,"sideways":0.15,"down":0.25,"strong_down":0.35}.get(trend.label.value,0.2)
        r_add=0.0 if plan is None else min(0.4, max(0.0, 0.2*plan.r_multiple))
        conf=base+r_add + 0.05*mtf_align + (0.05 if (not np.isnan(rs_sector) and rs_sector>1.0) else 0.0)

        price=float(df['close'].iloc[-1]); prev_hi=float(df['high'].iloc[-2]); prev_lo=float(df['low'].iloc[-2])
        reasons=[f"EMA20 slope {trend.ema_slope:.4f}, RSI14 {trend.rsi_14:.1f}. Trend={trend.label.value}. MTF align={mtf_align}, RS_sector_20={(None if np.isnan(rs_sector) else round(rs_sector,2))}"]

        # LLM vote
        llm_payload=None; llm_expl=None
        if self.use_llm:
            try:
                from .indicators import fibonacci_range
                fib=fibonacci_range(df, lookback=40)
                vote=llm_extra_prediction(symbol=symbol, price=price, ema20_slope=float(trend.ema_slope), rsi14=float(trend.rsi_14), price_above_ema=bool(trend.price_above_ema), prev_high=prev_hi, prev_low=prev_lo, fib_golden_low=float(fib.golden_low), fib_golden_high=float(fib.golden_high), fib_ext_1272=float(fib.levels["1.272"]), fib_ext_1618=float(fib.levels["1.618"]), atr14=float(ctx["atr_pct"]*price))
                agrees=(plan is not None) and ((plan.side.value=="long" and vote.entry_bias=="long") or (plan.side.value=="short" and vote.entry_bias=="short"))
                conf = min(1.0, conf + 0.15*vote.confidence) if agrees else max(0.0, conf - 0.1*vote.confidence)
                llm_payload={"trend_label":vote.trend_label,"entry_bias":vote.entry_bias,"entry_window_low":vote.entry_window_low,"entry_window_high":vote.entry_window_high,"confidence":vote.confidence}
                llm_expl=vote.rationale
                reasons.append(f"LLM vote: trend={vote.trend_label}, bias={vote.entry_bias}, conf={vote.confidence:.2f}. {vote.rationale}")
            except Exception:
                pass

        # Priors (KNN) with vol_reg filter
        expected_r=None; expected_winrate=None
        exp_hold_bars=None; exp_hold_days=None; exp_win_hold_bars=None; exp_loss_hold_bars=None
        prior_str=""
        if self.vec_db is not None:
            llm_conf=(llm_payload or {}).get("confidence",0.0)
            session_bin = {"open":0, "mid":1, "close":2}.get(tod, 1)
            vec=build_setup_vector(price=price, trend=trend, entry=plan, prev_range_pct=ctx["prev_range_pct"], gap_pct=ctx["gap_pct"], atr_pct=ctx["atr_pct"], session_bin=session_bin, llm_conf=llm_conf)
            try: add_vector(self.vec_db, vid=f"{symbol}-{df.index[-1].isoformat()}", ts_utc=df.index[-1].isoformat(), symbol=symbol, timeframe=self.interval, vec=vec, realized_r=None, exit_reason=None, payload={"note":"pending","vector_version":"v1.6.1","vol_regime":vol_reg})
            except Exception: pass
            nbrs=knn(self.vec_db, query_vec=vec, k=60, symbol=symbol) or knn(self.vec_db, query_vec=vec, k=60)
            from .vectorstore import filter_neighbors, extended_stats
            nbrs = filter_neighbors(nbrs, vol_regime=vol_reg)
            stats=extended_stats(nbrs)
            expected_winrate=stats["p_win"]
            expected_r=stats["p_win"]*stats["avg_win_R"]+(1-stats["p_win"])*stats["avg_loss_R"]
            exp_hold_bars=stats["median_hold_bars"]; exp_hold_days=stats["median_hold_days"]
            exp_win_hold_bars=stats["median_win_hold_bars"]; exp_loss_hold_bars=stats["median_loss_hold_bars"]
            prior=_clip01(0.5+stats["avg_R"]/4.0); conf=_clip01(0.6*conf+0.4*prior)
            prior_str=(f" Vector prior (k={stats['n']}) [{vol_reg}] : win={stats['p_win']*100:.0f}%, exp={stats['avg_R']:+.2f}R "
                       f"(avg_win={stats['avg_win_R']:+.2f}R, avg_loss={stats['avg_loss_R']:+.2f}R), PF={stats['profit_factor']}. "
                       f"Median hold: {exp_hold_bars} bars (~{exp_hold_days} days); win-median={exp_win_hold_bars}, loss-median={exp_loss_hold_bars}.")

        # Rule reasoning
        if plan is None: reasons.append("No actionable setup per rules.")
        else: reasons.append(f"Proposed {plan.side.value.upper()} at {plan.entry_price:.2f}, SL {plan.stop_price:.2f}, TP {plan.take_profit:.2f} (R={plan.r_multiple:.2f}). {plan.comment}.")
        reasons.append(prior_str)

        # LLM action plan
        action_plan=None; risk_notes=None; scenarios=None
        if self.llm_extras and self.use_llm and plan is not None:
            sig_for_llm = {
                "symbol": symbol, "timeframe": self.interval,
                "trend": {"label": trend.label.value, "ema_slope": float(trend.ema_slope), "price_above_ema": bool(trend.price_above_ema), "rsi_14": float(trend.rsi_14)},
                "entry": {"side": plan.side.value, "entry_price": plan.entry_price, "stop_price": plan.stop_price, "take_profit": plan.take_profit, "r_multiple": plan.r_multiple, "comment": plan.comment, "fib_golden_low": plan.fib_golden_low, "fib_golden_high": plan.fib_golden_high, "fib_target_1": plan.fib_target_1, "fib_target_2": plan.fib_target_2},
                "confidence": round(conf, 2),
                "expected_r": (round(expected_r, 3) if expected_r is not None else None),
                "expected_winrate": expected_winrate,
                "expected_hold_bars": exp_hold_bars, "expected_hold_days": exp_hold_days,
                "expected_win_hold_bars": exp_win_hold_bars, "expected_loss_hold_bars": exp_loss_hold_bars,
                "enrich": {"mtf_15m_trend": t15, "mtf_1h_trend": t1h, "mtf_alignment": mtf_align, "rs_sector_20": rs_sector, "rs_spy_20": rs_spy, "tod_bucket": tod, "vol_regime": vol_reg}
            }
            try:
                ap = llm_build_action_plan(signal_json=sig_for_llm, style="balanced")
                action_plan, risk_notes, scenarios = ap.action_plan, ap.risk_notes, ap.scenarios
            except Exception:
                pass

        signal = TradeSignal(
            symbol=symbol, timeframe=self.interval, asof=datetime.now(timezone.utc).isoformat(),
            trend=trend, entry=plan, confidence=round(conf,2),
            reasoning=" ".join([s for s in reasons if s]),
            llm_vote=llm_payload, llm_explanation=llm_expl,
            expected_r=(round(expected_r,3) if expected_r is not None else None),
            expected_winrate=expected_winrate, expected_source=("vector_knn/v1.6.1" if self.vec_db is not None else None),
            expected_notes=("E[R]=p*avg_win+(1-p)*avg_loss; med hold from KNN neighbors (filtered by vol_regime)." if self.vec_db is not None else None),
            expected_hold_bars=exp_hold_bars, expected_hold_days=exp_hold_days,
            expected_win_hold_bars=exp_win_hold_bars, expected_loss_hold_bars=exp_loss_hold_bars,
            action_plan=action_plan, risk_notes=risk_notes, scenarios=scenarios,
            mtf_15m_trend=t15, mtf_1h_trend=t1h, mtf_alignment=mtf_align,
            rs_sector_20=(None if np.isnan(rs_sector) else float(rs_sector)),
            rs_spy_20=(None if np.isnan(rs_spy) else float(rs_spy)),
            sector_symbol=self.sector_symbol, tod_bucket=tod,
            atr_pct=float(ctx["atr_pct"]), vol_regime=vol_reg
        )

        if self.log_db:
            try: record_signal(signal, self.log_db)
            except Exception: pass

        return signal
