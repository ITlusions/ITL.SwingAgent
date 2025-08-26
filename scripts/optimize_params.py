#!/usr/bin/env python
"""Parameter optimization for SwingAgent backtests.

This script evaluates multiple parameter combinations using
``portfolio_backtest.run_backtest`` and selects the best configuration
based on average realized R multiple or Sharpe ratio.  Results are
written to JSON and the global ``TradingConfig`` is updated accordingly.
"""

import argparse
import itertools
import json
import sys
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.portfolio_backtest import run_backtest
from swing_agent.config import update_config


def _parse_ints(values: str) -> Iterable[int]:
    return [int(v) for v in values.split(",") if v]


def grid_search(signals: Path, ema_vals: Iterable[int], rsi_vals: Iterable[int], metric: str):
    best_score = float("-inf")
    best_params: dict[str, int] | None = None
    for ema, rsi in itertools.product(ema_vals, rsi_vals):
        update_config(EMA20_PERIOD=ema, RSI_PERIOD=rsi)
        result = run_backtest(signals)
        score = result["avg_r"] if metric == "realized_r" else result["sharpe"]
        if score > best_score:
            best_score = score
            best_params = {"EMA20_PERIOD": ema, "RSI_PERIOD": rsi}
    if best_params is None:  # no trades
        best_params = {"EMA20_PERIOD": ema_vals[0], "RSI_PERIOD": rsi_vals[0]}
        best_score = 0.0
    return best_params, best_score


def optuna_search(
    signals: Path,
    ema_vals: Iterable[int],
    rsi_vals: Iterable[int],
    metric: str,
    trials: int,
):
    try:
        import optuna
    except Exception as exc:  # pragma: no cover - optional dependency
        raise SystemExit("optuna is required for --use-optuna") from exc

    def objective(trial: "optuna.Trial") -> float:
        ema = trial.suggest_categorical("EMA20_PERIOD", ema_vals)
        rsi = trial.suggest_categorical("RSI_PERIOD", rsi_vals)
        update_config(EMA20_PERIOD=ema, RSI_PERIOD=rsi)
        result = run_backtest(signals)
        return result["avg_r"] if metric == "realized_r" else result["sharpe"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)
    return study.best_params, study.best_value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", required=True, type=Path, help="CSV file with generated signals")
    ap.add_argument("--ema-periods", default="20", help="Comma separated EMA lengths")
    ap.add_argument("--rsi-periods", default="14", help="Comma separated RSI periods")
    ap.add_argument("--metric", choices=["realized_r", "sharpe"], default="realized_r")
    ap.add_argument("--output", type=Path, default=Path("best_params.json"))
    ap.add_argument("--use-optuna", action="store_true", help="Use optuna instead of grid search")
    ap.add_argument("--trials", type=int, default=20, help="Number of optuna trials")
    args = ap.parse_args()

    ema_vals = list(_parse_ints(args.ema_periods))
    rsi_vals = list(_parse_ints(args.rsi_periods))

    if args.use_optuna:
        best_params, best_score = optuna_search(
            args.signals, ema_vals, rsi_vals, args.metric, args.trials
        )
    else:
        best_params, best_score = grid_search(args.signals, ema_vals, rsi_vals, args.metric)

    # Update global config and persist
    update_config(**best_params)
    best_params["score"] = best_score
    args.output.write_text(json.dumps(best_params, indent=2))
    print(f"Best parameters: {best_params}")


if __name__ == "__main__":
    main()
