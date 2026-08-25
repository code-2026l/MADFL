"""Entry point for the 28-fold walk-forward validation.

Reproduces the paper's experimental protocol on real OHLCV data:

    python experiments/run_walkforward.py \
        --config experiments/configs/csi500.yaml \
        --data path/to/ohlcv.csv

The CSV must contain columns ``date,code,open,high,low,close,volume`` for at
least ``n_stocks`` instruments over the evaluation window.  Results (metric
summary + per-fold output) are written to ``outputs/walkforward/`` and can be
formatted into the paper's tables with:

    python scripts/reproduce_tables.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from madfl.config import FactorConfig, GateConfig, PortfolioConfig, \
    RegimeConfig, DistillationConfig, MemoryAlphaConfig
from madfl.pipeline import run_walkforward


def load_config(path: str) -> MemoryAlphaConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return MemoryAlphaConfig(
        gate=GateConfig(**raw.get("gate", {})),
        regime=RegimeConfig(**raw.get("regime", {})),
        factors=FactorConfig(**raw.get("factors", {})),
        portfolio=PortfolioConfig(**raw.get("portfolio", {})),
        distillation=DistillationConfig(**raw.get("distillation", {})),
        train_days=raw.get("train_days", 60),
        test_days=raw.get("test_days", 21),
        n_folds=raw.get("n_folds", 28),
        seed=raw.get("seed", 42),
        instrument_universe=raw.get("instrument_universe", "csi500"),
        n_stocks=raw.get("n_stocks", 200),
        debate_rounds=raw.get("debate_rounds", 3),
        consensus_gate=raw.get("consensus_gate", 0.5),
    )


def load_ohlcv(path: str, n_stocks: int):
    """Load a long-format OHLCV CSV into (returns, volume, market).

    Only names covering the full evaluation window are kept (>= 1200 price
    points; the panel median is 1699), so recently-listed stocks cannot
    truncate the panel via the ``T = min(...)`` alignment below.
    """
    import pandas as pd
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    all_codes = df["code"].unique()
    returns, volume, kept = [], [], []
    for code in all_codes:
        sub = df[df["code"] == code].sort_values("date")
        px = sub["close"].values
        if px.size >= 1200:  # require full-window coverage
            returns.append(np.diff(np.log(px)))
            volume.append(sub["volume"].values[:-1])
            kept.append(code)
        if len(kept) >= n_stocks:
            break
    print(f"[MADFL] kept {len(kept)}/{len(all_codes)} stocks with full-window coverage")
    T = min(len(r) for r in returns)
    returns = np.stack([r[:T] for r in returns], axis=1)
    volume = np.stack([v[:T] for v in volume], axis=1)
    market = np.nanmean(returns, axis=1)
    return returns, volume, market


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the 28-fold walk-forward validation on real OHLCV data.")
    ap.add_argument("--config", default="experiments/configs/csi500.yaml")
    ap.add_argument("--data", required=True,
                    help="CSV of OHLCV data (date,code,open,high,low,close,volume)")
    ap.add_argument("--n-stocks", type=int, default=None)
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--fold-start", type=int, default=0)
    ap.add_argument("--fold-end", type=int, default=None)
    ap.add_argument("--out", type=str, default="outputs/walkforward",
                    help="Output directory for results")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        print(f"[MADFL] data file not found: {args.data}")
        sys.exit(1)

    cfg = load_config(args.config)
    if args.n_stocks:
        cfg.n_stocks = args.n_stocks
    if args.folds:
        cfg.n_folds = args.folds

    returns, volume, market = load_ohlcv(args.data, cfg.n_stocks)
    if returns.shape[0] < cfg.train_days + cfg.test_days:
        print(f"[MADFL] not enough history: {returns.shape[0]} days "
              f"< train({cfg.train_days}) + test({cfg.test_days})")
        sys.exit(1)

    print(f"[MADFL] universe={cfg.instrument_universe} "
          f"stocks={returns.shape[1]} folds={cfg.n_folds} "
          f"days={returns.shape[0]} slice=[{args.fold_start},{args.fold_end})")

    result = run_walkforward(returns, volume, market, cfg,
                             fold_start=args.fold_start,
                             fold_end=args.fold_end)

    print("\n=== Walk-forward results ===")
    for key in ("mean_ic", "icir", "annualized_return", "sharpe",
                "max_drawdown", "turnover"):
        val = result[key]
        fmt = f"{val:.4f}" if isinstance(val, float) else val
        print(f"  {key:20s}: {fmt}")

    # persist machine-readable results so reproduce_tables can format them
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"_f{args.fold_start}-{args.fold_end}" if (args.fold_start or args.fold_end) else ""
    serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in result.items()}
    with open(out / f"run_results{suffix}.json", "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\n[Step 1] Results written to {out / 'run_results{suffix}.json'}")
    print(f"[Step 2] Format tables:  python scripts/reproduce_tables.py "
          f"--results {out / 'run_results.json'}")


if __name__ == "__main__":
    main()