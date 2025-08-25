from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Dict, Any
import pandas as pd
import yfinance as yf


class SwingAgentDataError(Exception):
    """Custom exception for data-related errors in SwingAgent."""
    
    def __init__(self, message: str, symbol: str = None, interval: str = None):
        self.symbol = symbol
        self.interval = interval
        super().__init__(message)


VALID_INTERVALS: Dict[str, str] = {
    "15m": "15m", 
    "30m": "30m", 
    "1h": "60m", 
    "1d": "1d"
}


def load_ohlcv(symbol: str, interval: str = "30m", lookback_days: int = 30) -> pd.DataFrame:
    """Load OHLCV data from Yahoo Finance with enhanced error handling.
    
    Args:
        symbol: Stock symbol to download (e.g., "AAPL", "SPY").
        interval: Trading timeframe ("15m", "30m", "1h", "1d").
        lookback_days: Number of days of historical data to retrieve.
        
    Returns:
        pd.DataFrame: OHLCV data with datetime index in UTC.
        
    Raises:
        SwingAgentDataError: If data cannot be retrieved or is invalid.
        ValueError: If interval is not supported.
        
    Examples:
        >>> df = load_ohlcv("AAPL", "30m", 30)
        >>> print(f"Loaded {len(df)} bars for AAPL")
    """
    # Validate inputs
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol must be a non-empty string")
    
    if interval not in VALID_INTERVALS:
        raise ValueError(f"Unsupported interval: {interval}. "
                        f"Supported intervals: {list(VALID_INTERVALS.keys())}")
    
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    
    # Clean symbol (remove whitespace, convert to uppercase)
    symbol = symbol.strip().upper()
    
    try:
        # Calculate date range
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days)
        
        # Download data with error handling
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
        
        # Handle various error conditions
        if df is None:
            raise SwingAgentDataError(
                f"yfinance returned None for symbol {symbol}",
                symbol=symbol, interval=interval
            )
        
        if df.empty:
            raise SwingAgentDataError(
                f"No data available for {symbol} in the last {lookback_days} days. "
                f"Symbol may be delisted, invalid, or markets may be closed.",
                symbol=symbol, interval=interval
            )
        
        # Clean and validate data
        df = df.rename(columns=str.lower)
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[~df.index.duplicated(keep="last")]
        
        # Ensure required columns exist
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise SwingAgentDataError(
                f"Missing required columns {missing_columns} in data for {symbol}",
                symbol=symbol, interval=interval
            )
        
        # Check for sufficient data points
        if len(df) < 20:
            raise SwingAgentDataError(
                f"Insufficient data for {symbol}: only {len(df)} bars available. "
                f"Need at least 20 bars for analysis.",
                symbol=symbol, interval=interval
            )
        
        # Validate data quality (no all-zero or all-NaN data)
        if df[["open", "high", "low", "close"]].isna().all().any():
            raise SwingAgentDataError(
                f"Invalid data for {symbol}: contains all-NaN columns",
                symbol=symbol, interval=interval
            )
        
        return df
        
    except SwingAgentDataError:
        # Re-raise our custom errors
        raise
    except Exception as e:
        # Wrap other exceptions in our custom exception
        error_msg = str(e).lower()
        
        if "404" in error_msg or "not found" in error_msg:
            raise SwingAgentDataError(
                f"Symbol {symbol} not found. Please check if the symbol is valid.",
                symbol=symbol, interval=interval
            ) from e
        elif "network" in error_msg or "connection" in error_msg:
            raise SwingAgentDataError(
                f"Network error while fetching data for {symbol}. "
                f"Please check your internet connection.",
                symbol=symbol, interval=interval
            ) from e
        elif "rate limit" in error_msg or "too many requests" in error_msg:
            raise SwingAgentDataError(
                f"Rate limit exceeded while fetching data for {symbol}. "
                f"Please wait before retrying.",
                symbol=symbol, interval=interval
            ) from e
        else:
            raise SwingAgentDataError(
                f"Unexpected error while fetching data for {symbol}: {e}",
                symbol=symbol, interval=interval
            ) from e
