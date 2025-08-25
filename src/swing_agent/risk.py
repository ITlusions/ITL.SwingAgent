"""Risk and position sizing utilities for SwingAgent.

This module contains helper functions for sizing positions based on
account equity and stop distance, along with simple portfolio level
risk checks.
"""

from __future__ import annotations

from typing import Iterable, Optional


def sized_quantity(
    equity: float,
    entry: float,
    stop: float,
    max_risk_pct: float = 0.01,
    atr: Optional[float] = None,
    k_atr: float = 0.5,
) -> int:
    """Calculate position size given account equity and stop distance.

    Parameters
    ----------
    equity: float
        Current account equity.
    entry: float
        Proposed entry price.
    stop: float
        Stop-loss price.
    max_risk_pct: float, default 0.01
        Maximum fraction of equity to risk on this trade.
    atr: float, optional
        Average True Range of the instrument.  When provided the stop
        distance will be at least ``k_atr * atr``.
    k_atr: float, default 0.5
        Multiplier for ``atr`` when enforcing a minimum stop distance.

    Returns
    -------
    int
        Whole-number quantity of shares/contracts to trade.
    """
    risk_eur = equity * max_risk_pct
    risk_per_share = max(entry - stop, 0.01)
    if atr is not None:
        risk_per_share = max(risk_per_share, k_atr * atr)
    qty = int(risk_eur / risk_per_share)
    return max(qty, 0)


def total_risk(at_risk: Iterable[float]) -> float:
    """Return the total euro amount currently at risk."""
    return sum(at_risk)


def can_open(
    at_risk: Iterable[float],
    equity: float,
    new_risk: float,
    max_total_risk_pct: float = 0.04,
) -> bool:
    """Check if a new position can be opened without breaching portfolio risk.

    Parameters
    ----------
    at_risk: Iterable[float]
        Collection of existing risk amounts (in euros) for open positions.
    equity: float
        Current account equity.
    new_risk: float
        Risk amount of the proposed trade.
    max_total_risk_pct: float, default 0.04
        Maximum total portfolio risk allowed as a fraction of equity.
    """
    return total_risk(at_risk) + new_risk <= equity * max_total_risk_pct


__all__ = ["sized_quantity", "total_risk", "can_open"]
