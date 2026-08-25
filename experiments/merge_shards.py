#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Merge sharded walk-forward results into a single run_results.json.

Each worker writes run_results_f{a}-{b}.json with full-length arrays where
only its fold slice is populated. This script stitches them and recomputes
the aggregate metrics from the merged arrays.
"""
import argparse
import json
import glob
import os

import numpy as np

from madfl.config import MemoryAlphaConfig
from madfl.utils import (annualized_return, icir, max_drawdown, mean_ic,
                         sharpe, turnover)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="output directory with shards")
    ap.add_argument("--train-days", type=int, default=504)
    ap.add_argument("--test-days", type=int, default=21)
    ap.add_argument("--out", required=True, help="merged json path")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "run_results_f*.json")))
    if not files:
        print("no shards found in", args.dir)
        raise SystemExit(1)
    print("shards:", len(files))

    shards = [json.load(open(f, encoding="utf-8")) for f in files]
    T = len(shards[0]["oos_returns"])
    n_stocks = len(shards[0]["oos_weights"][0])

    oos_returns = np.zeros(T)
    oos_weights = np.zeros((T, n_stocks))
    per_day_ic = np.full(T, np.nan)
    weight_rows = np.zeros(T, dtype=bool)

    for sh in shards:
        oos_returns += np.asarray(sh["oos_returns"], dtype=float)
        w = np.asarray(sh["oos_weights"], dtype=float)
        row_active = w.any(axis=1)
        oos_weights[row_active] = w[row_active]
        weight_rows |= row_active
        ic = np.asarray(sh["per_day_ic"], dtype=float)
        good = ~np.isnan(ic)
        per_day_ic[good] = ic[good]

    # per-fold IC: fold k's OOS days are [train+test*k, train+test*k+test-1)
    train = args.train_days
    test = args.test_days
    per_fold = []
    for k in range(0, (T - train) // test):
        seg = per_day_ic[train + test * k: train + test * (k + 1)]
        seg = seg[~np.isnan(seg)]
        if seg.size:
            per_fold.append(float(np.mean(seg)))
    per_fold_ic = np.array(per_fold)

    valid = weight_rows
    rets = oos_returns[valid]
    wealth = np.cumprod(1.0 + rets)

    out = {
        "oos_returns": oos_returns.tolist(),
        "oos_weights": oos_weights.tolist(),
        "per_fold_ic": per_fold_ic.tolist(),
        "per_day_ic": per_day_ic.tolist(),
        "mean_ic": mean_ic(per_fold_ic),
        "icir": icir(per_fold_ic),
        "annualized_return": annualized_return(rets),
        "sharpe": sharpe(rets),
        "max_drawdown": max_drawdown(wealth),
        "turnover": turnover(oos_weights[valid]),
        "regime_states": shards[0].get("regime_states"),
        "regime_names": shards[0].get("regime_names"),
        "factor_half_life": shards[0].get("factor_half_life"),
        "factor_families": shards[0].get("factor_families"),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\n=== Merged walk-forward results ===")
    for key in ("mean_ic", "icir", "annualized_return", "sharpe",
                "max_drawdown", "turnover"):
        print(f"  {key:20s}: {out[key]:.4f}")
    print("  n_folds:", len(per_fold_ic), "| valid days:", int(valid.sum()))
    print("saved:", args.out)


if __name__ == "__main__":
    main()
