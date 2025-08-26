"""Configuration management for SwingAgent trading parameters.

This module centralizes all magic numbers and configuration values
used throughout the SwingAgent system for better maintainability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TradingConfig:
    """Central configuration class for all trading parameters.
    
    This class eliminates magic numbers throughout the codebase and provides
    a single place to tune trading parameters and thresholds.
    """

    # Trend detection thresholds
    EMA_SLOPE_THRESHOLD_UP: float = 0.01
    EMA_SLOPE_THRESHOLD_STRONG: float = 0.02
    EMA_SLOPE_THRESHOLD_DOWN: float = -0.01
    EMA_SLOPE_THRESHOLD_STRONG_DOWN: float = -0.02
    EMA_SLOPE_LOOKBACK: int = 6

    # RSI thresholds for trend confirmation
    RSI_TREND_UP_MIN: float = 60.0
    RSI_TREND_DOWN_MAX: float = 40.0
    RSI_OVERSOLD_THRESHOLD: float = 35.0
    RSI_OVERBOUGHT_THRESHOLD: float = 65.0
    RSI_PERIOD: int = 14

    # Risk management parameters
    ATR_STOP_BUFFER: float = 0.2
    ATR_STOP_MULTIPLIER: float = 1.2
    ATR_TARGET_MULTIPLIER: float = 2.0
    ATR_MEAN_REVERSION_STOP: float = 1.0
    ATR_MEAN_REVERSION_TARGET: float = 1.5
    ATR_PERIOD: int = 14

    # Volatility regime analysis
    VOL_REGIME_LOOKBACK: int = 60
    VOL_LOW_PERCENTILE: float = 0.33
    VOL_HIGH_PERCENTILE: float = 0.66

    # Fibonacci analysis
    FIB_LOOKBACK: int = 40
    GOLDEN_POCKET_LOW: float = 0.618
    GOLDEN_POCKET_HIGH: float = 0.65

    # EMA parameters
    EMA20_PERIOD: int = 20

    # Confidence scoring
    BASE_CONFIDENCE: dict[str, float] = None
    MAX_R_MULTIPLE_BONUS: float = 0.4
    R_MULTIPLE_FACTOR: float = 0.2
    MTF_ALIGNMENT_BONUS: float = 0.05
    RS_SECTOR_BONUS: float = 0.05
    RS_SECTOR_THRESHOLD: float = 1.0
    LLM_AGREEMENT_BONUS: float = 0.15
    LLM_DISAGREEMENT_PENALTY: float = 0.1

    # Vector store parameters
    KNN_DEFAULT_K: int = 60
    VECTOR_VERSION: str = "v1.6.1"

    # Data requirements
    MIN_DATA_BARS: int = 60

    # Relative strength parameters
    RS_LOOKBACK_DAYS: int = 20

    def __post_init__(self):
        """Initialize default confidence values."""
        if self.BASE_CONFIDENCE is None:
            self.BASE_CONFIDENCE = {
                "strong_up": 0.35,
                "up": 0.25,
                "sideways": 0.15,
                "down": 0.25,
                "strong_down": 0.35
            }


# Global configuration instance
config = TradingConfig()


def get_config() -> TradingConfig:
    """Get the global trading configuration instance.
    
    Returns:
        TradingConfig: The global configuration object.
    """
    return config


def update_config(**kwargs: Any) -> None:
    """Update configuration parameters.
    
    Args:
        **kwargs: Configuration parameters to update.
        
    Example:
        >>> update_config(EMA_SLOPE_THRESHOLD_UP=0.015, RSI_PERIOD=21)
    """
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")
