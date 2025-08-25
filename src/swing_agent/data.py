from __future__ import annotations
from datetime import datetime, timedelta, timezone
import pandas as pd
import yfinance as yf

VALID_INTERVALS = {"15m": "15m", "30m": "30m", "1h": "60m", "1d": "1d"}

def load_ohlcv(symbol: str, interval: str = "30m", lookback_days: int = 30) -> pd.DataFrame:
    if interval not in VALID_INTERVALS:
        raise ValueError(f"Unsupported interval: {interval}")
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    df = yf.download(
        tickers=symbol,
        interval=VALID_INTERVALS[interval],
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        prepost=False,
        threads=True,
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data for {symbol} @ {interval}")
    df = df.rename(columns=str.lower)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="last")]
    for col in ["open","high","low","close","volume"]:
        if col not in df.columns:
            raise RuntimeError(f"Missing column '{col}' in data for {symbol}")
    return df
