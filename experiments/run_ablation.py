"""Ablation suite for the Memory Alpha paper tables.

Each variant is a full 28-fold walk-forward on the same date-aligned panel
with one component toggled, mirroring the paper's three ablation tables:

* ``gate_*``  -- Tier-6a gate ablation on the V40 baseline (pool size 40,
  V2 optimizer fixed): No Gate / CSC / PSU / DSR / CSC+PSU / Full.
* ``deb_*``   -- debate & distillation ablation on the full pipeline
  (M=44, gate on): teacher-only / distill-only / debate K=1 / debate K=3.
  ``deb_k3`` is the full system and feeds the main results row.
* ``v2_*``    -- portfolio optimizer ablation on the full signal:
  mean-variance / BL / CVaR / risk-budgeting vs the full V2.

The panel loader keeps only stocks with data on **every** trading day of the
sample (no suspensions, no late listings), so every panel row maps to the same
calendar date across stocks, and takes the tail of the sample sized exactly to
the walk-forward protocol (train_days + n_folds * test_days), so the folds
tile the panel with no unused days and the OOS window ends at the latest
available date.

Sharding: ``--fold-start/--fold-end/--shard`` slice the fold loop across
parallel workers; ``experiments/merge_ablation_shards.py`` stitches the shard
jsons and recomputes the aggregate metrics.

Usage:
    python experiments/run_ablation.py --variant gate_csc
    python experiments/run_ablation.py --variant gate_csc \
        --fold-start 0 --fold-end 7 --shard f0-7
    python experiments/run_ablation.py --summarize   # tables from merged jsons
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
from madfl.utils import rank_ic

FULL = {"pool": 44, "use_gate": True, "use_regime": True,
        "use_distillation": True, "use_consensus": True, "debate_rounds": 3}

VARIANTS = {
    # --- Tier-6a gate ablation on the V40 baseline (pool=40, V2 fixed) ---
    "gate_none":   {"pool": 40, "use_gate": False},
    "gate_csc":    {"pool": 40, "use_gate": True, "alpha": (1.0, 0.0, 0.0)},
    "gate_psu":    {"pool": 40, "use_gate": True, "alpha": (0.0, 1.0, 0.0)},
    "gate_dsr":    {"pool": 40, "use_gate": True, "alpha": (0.0, 0.0, 1.0)},
    "gate_csupsu": {"pool": 40, "use_gate": True, "alpha": (0.5, 0.5, 0.0)},
    "gate_full":   {"pool": 40, "use_gate": True, "alpha": (0.4, 0.4, 0.2)},
    # --- Table 1 ablation chain: V40 -> +Tier-6a -> +Regime -> Full ---
    "chain_regime": {"pool": 40, "use_gate": True, "use_regime": True,
                     "alpha": (0.4, 0.4, 0.2)},
    # --- debate & distillation ablation on the full pipeline ---
    # 2x2 factorial: distillation {on, off} x consensus {on, off}.
    #   deb_none           = distil OFF x debate OFF  (single-agent baseline)
    #   deb_single         = distil ON  x debate OFF
    #   deb_k1_nodistill   = distil OFF x debate K=1  (NEW: isolates debate)
    #   deb_k3_nodistill   = distil OFF x debate K=3  (NEW: isolates debate)
    #   deb_k1             = distil ON  x debate K=1
    #   deb_k3             = distil ON  x debate K=3  (full system)
    "deb_none":   {**FULL, "use_distillation": False, "use_consensus": False},
    "deb_single": {**FULL, "use_consensus": False},
    "deb_k1_nodistill": {**FULL, "use_distillation": False,
                         "use_consensus": True, "debate_rounds": 1},
    "deb_k3_nodistill": {**FULL, "use_distillation": False,
                         "use_consensus": True, "debate_rounds": 3},
    "deb_k1":     {**FULL, "debate_rounds": 1},
    "deb_k3":     {**FULL},
    # --- Causal Deflation Screening (CDS): winner's-curse-corrected causal
    # screening (new algorithm, paper core contribution). Same 2x2 grid as
    # deb_* but the consensus screen uses IC* = IC - se*sqrt(2 ln K_eff).
    "defl_none":   {**FULL, "use_distillation": False, "use_consensus": False,
                    "screen_mode": "deflate"},
    "defl_single": {**FULL, "use_consensus": False, "screen_mode": "deflate"},
    "defl_k1_nodistill": {**FULL, "use_distillation": False,
                          "use_consensus": True, "debate_rounds": 1,
                          "screen_mode": "deflate"},
    "defl_k3_nodistill": {**FULL, "use_distillation": False,
                          "use_consensus": True, "debate_rounds": 3,
                          "screen_mode": "deflate"},
    "defl_k1":     {**FULL, "debate_rounds": 1, "screen_mode": "deflate"},
    "defl_k3":     {**FULL, "screen_mode": "deflate"},
    # --- regime-soft-weighted factor combo (all families, causal weights) ---
    "rgw":   {**FULL, "use_distillation": False, "use_consensus": False,
              "combine_head": True},
    # --- full-time reversal + equal-weight multi-factor ---
    # Fixed economic direction (short-term reversal in A-shares): negate the
    # equal-weight composite of ALL families. No distillation/consensus/gate
    # (those destroyed the reversal signal); regime deployment layer kept.
    "reversal": {"pool": 44, "use_gate": False, "use_regime": True,
                 "use_distillation": False, "use_consensus": False,
                 "reversal": True},
    # reversal + economic regime-gated exposure (cut to 25% in HIGH_VOL /
    # BEAR_CRASH where mean-reversion is fragile), everything else identical.
    "rev_regime_gate": {"pool": 44, "use_gate": False, "use_regime": True,
                        "use_distillation": False, "use_consensus": False,
                        "reversal": True, "reversal_gate": "regime"},
    # sensitivity of the adverse-regime exposure floor (robustness, not tuning)
    "rev_gate_f0":  {"pool": 44, "use_gate": False, "use_regime": True,
                     "use_distillation": False, "use_consensus": False,
                     "reversal": True, "reversal_gate": "regime",
                     "reversal_floor": 0.0},
    "rev_gate_f50": {"pool": 44, "use_gate": False, "use_regime": True,
                     "use_distillation": False, "use_consensus": False,
                     "reversal": True, "reversal_gate": "regime",
                     "reversal_floor": 0.5},
    # transparent causal adverse-state gate (trailing vol percentile + drawdown)
    "rev_tgate":    {"pool": 44, "use_gate": False, "use_regime": True,
                     "use_distillation": False, "use_consensus": False,
                     "reversal": True, "reversal_gate": "transparent"},
    "rev_tgate_f0": {"pool": 44, "use_gate": False, "use_regime": True,
                     "use_distillation": False, "use_consensus": False,
                     "reversal": True, "reversal_gate": "transparent",
                     "reversal_floor": 0.0},
    "rev_tgate_f50": {"pool": 44, "use_gate": False, "use_regime": True,
                      "use_distillation": False, "use_consensus": False,
                      "reversal": True, "reversal_gate": "transparent",
                      "reversal_floor": 0.5},
    # --- portfolio optimizer ablation on the full signal ---
    "v2_mv":   {**FULL, "portfolio_mode": "mv"},
    "v2_bl":   {**FULL, "portfolio_mode": "bl"},
    "v2_cvar": {**FULL, "portfolio_mode": "cvar"},
    "v2_rb":   {**FULL, "portfolio_mode": "rb"},
}


def build_config(yaml_path: str, variant: dict) -> MemoryAlphaConfig:
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = MemoryAlphaConfig(
        gate=GateConfig(**raw.get("gate", {})),
        regime=RegimeConfig(**raw.get("regime", {})),
        factors=FactorConfig(**raw.get("factors", {})),
        portfolio=PortfolioConfig(**raw.get("portfolio", {})),
        distillation=DistillationConfig(
            **{**raw.get("distillation", {}),
               "feature_pool_size": variant.get("pool", 44)}),
        train_days=raw.get("train_days", 60),
        test_days=raw.get("test_days", 21),
        n_folds=raw.get("n_folds", 28),
        seed=raw.get("seed", 42),
        instrument_universe=raw.get("instrument_universe", "csi500"),
        n_stocks=raw.get("n_stocks", 200),
        debate_rounds=variant.get("debate_rounds", raw.get("debate_rounds", 3)),
        consensus_gate=raw.get("consensus_gate", 0.5),
        use_gate=variant.get("use_gate", False),
        use_regime=variant.get("use_regime", False),
        portfolio_mode=variant.get("portfolio_mode", "v2"),
        use_distillation=variant.get("use_distillation", True),
        use_consensus=variant.get("use_consensus", False),
    )
    if "alpha" in variant:
        cfg.gate.alpha_csc, cfg.gate.alpha_psu, cfg.gate.alpha_dsr = variant["alpha"]
    setattr(cfg, "combine_head", variant.get("combine_head", False))
    setattr(cfg, "reversal", variant.get("reversal", False))
    setattr(cfg, "reversal_gate", variant.get("reversal_gate", None))
    setattr(cfg, "reversal_floor", float(variant.get("reversal_floor", 0.25)))
    setattr(cfg, "screen_mode", variant.get("screen_mode", "prior"))
    return cfg


def load_ohlcv(path: str, train_days: int, test_days: int, n_folds: int):
    """Load a long-format OHLCV CSV into a date-aligned panel.

    Only stocks with data on every trading day of the sample are kept (full
    coverage: no suspensions, no late listing), so panel row ``t`` maps to the
    same calendar date for every stock.  The panel is the tail of the sample
    sized exactly to the protocol (``train_days + n_folds * test_days`` return
    rows), so the fold loop tiles it without unused days and the OOS window
    ends at the last available date.  Returns (returns, volume, market, dates)
    where ``dates[t]`` is the date on which row ``t``'s return is realized.
    """
    import pandas as pd
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    master = pd.DatetimeIndex(sorted(df["date"].unique()))
    need = train_days + n_folds * test_days  # return rows required
    n_master = len(master)
    # Coverage is judged against the *protocol tail window* (the last `need`
    # return rows), not the whole sample.  A stock that was absent before the
    # window (early suspension / late listing) but present on every window
    # date is perfectly date-aligned in the OOS and must be kept; requiring
    # whole-sample coverage needlessly discards it (112 -> 160 names).
    tail_master = master[-need:]
    n_tail = len(tail_master)
    returns, volume, kept = [], [], []
    for code in df["code"].unique():
        sub = df[df["code"] == code].sort_values("date")
        ds = pd.DatetimeIndex(sub["date"])
        # full coverage on the protocol tail window only
        present_tail = ds[ds >= tail_master[0]]
        if len(present_tail) < n_tail or present_tail[0] != tail_master[0] \
                or present_tail[-1] != tail_master[-1]:
            continue
        px = sub["close"].values.astype(float)
        # full return series; the stack step trims each to the last `need` rows
        returns.append(np.diff(np.log(px)))
        volume.append(sub["volume"].values[:-1].astype(float))
        kept.append(code)
    print(f"[ABL] kept {len(kept)}/{df['code'].nunique()} stocks with "
          f"full-coverage data ({master[0].date()}..{master[-1].date()}, "
          f"{n_master} days)")
    T = min(len(r) for r in returns)
    if T < need:
        raise SystemExit(f"panel too short: have {T} return rows, "
                         f"protocol needs {need}")
    returns = np.stack([r[-need:] for r in returns], axis=1)
    volume = np.stack([v[-need:] for v in volume], axis=1)
    market = np.nanmean(returns, axis=1)
    dates = master[-need:]
    print(f"[ABL] panel: {dates[0].date()} -> {dates[-1].date()} "
          f"({returns.shape[0]} return rows, {returns.shape[1]} stocks)")
    return returns, volume, market, dates


def consistency_metric(per_day_ic: np.ndarray) -> float:
    """Fraction of OOS days whose IC agrees in sign with the panel mean IC."""
    ic = np.asarray(per_day_ic, dtype=float)
    ic = ic[~np.isnan(ic)]
    if ic.size == 0:
        return float("nan")
    return float(np.mean(np.sign(ic) == np.sign(np.mean(ic))))


def run_variant(name: str, args) -> None:
    variant = VARIANTS[name]
    cfg = build_config(args.config, variant)
    returns, volume, market, dates = load_ohlcv(
        args.data, cfg.train_days, cfg.test_days, cfg.n_folds)
    print(f"[ABL] variant={name} pool={cfg.distillation.feature_pool_size} "
          f"gate={cfg.use_gate} regime={cfg.use_regime} mode={cfg.portfolio_mode} "
          f"distill={cfg.use_distillation} consensus={cfg.use_consensus} "
          f"reversal={getattr(cfg, 'reversal', False)} "
          f"rounds={cfg.debate_rounds} "
          f"folds=[{args.fold_start}, {args.fold_end})")
    result = run_walkforward(returns, volume, market, cfg,
                             fold_start=args.fold_start,
                             fold_end=args.fold_end)

    summary = {k: result[k] for k in
               ("mean_ic", "icir", "annualized_return", "sharpe",
                "max_drawdown", "turnover")}
    summary["consistency"] = consistency_metric(result["per_day_ic"])
    if result.get("cds_k_eff"):
        summary["cds_k_eff"] = float(np.nanmean(result["cds_k_eff"]))
        summary["cds_null_sigma"] = float(np.nanmean(result["cds_null_sigma"]))
        summary["cds_defl_winner"] = float(np.nanmean(
            result["cds_defl_winner"]))
    payload = {
        "variant": name,
        "shard": args.shard,
        "fold_start": args.fold_start,
        "fold_end": args.fold_end,
        "config": {"pool": cfg.distillation.feature_pool_size,
                   "use_gate": cfg.use_gate,
                   "alpha": [cfg.gate.alpha_csc, cfg.gate.alpha_psu,
                             cfg.gate.alpha_dsr],
                   "portfolio_mode": cfg.portfolio_mode,
                   "use_distillation": cfg.use_distillation,
                   "use_consensus": cfg.use_consensus,
                   "screen_mode": getattr(cfg, "screen_mode", "prior"),
                   "reversal": getattr(cfg, "reversal", False),
                   "debate_rounds": cfg.debate_rounds,
                   "train_days": cfg.train_days, "test_days": cfg.test_days,
                   "n_folds": cfg.n_folds, "seed": cfg.seed},
        "summary": summary,
        "per_fold_ic": np.asarray(result["per_fold_ic"]).tolist(),
        "per_day_ic": np.asarray(result["per_day_ic"]).tolist(),
        "oos_returns": np.asarray(result["oos_returns"]).tolist(),
        "oos_weights": np.asarray(result["oos_weights"]).tolist(),
        "regime_states": np.asarray(result["regime_states"]).tolist(),
        "regime_names": list(result["regime_names"]),
        "factor_half_life": np.asarray(result["factor_half_life"],
                                       dtype=float).tolist(),
        "factor_families": list(result["factor_families"]),
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fname = f"{name}_{args.shard}.json" if args.shard else f"{name}.json"
    with open(out / fname, "w") as f:
        json.dump(payload, f, default=str)
    print(f"[ABL] {name}{':' + args.shard if args.shard else ''}: "
          + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in summary.items()))


def summarize(args) -> None:
    out = Path(args.out)
    rows = {}
    for name in VARIANTS:
        p = out / f"{name}.json"
        if p.exists():
            with open(p) as f:
                rows[name] = json.load(f)["summary"]
    if not rows:
        print("[ABL] no results found")
        return

    def get(name, key, fmt="{:.4f}"):
        if name in rows and key in rows[name]:
            return fmt.format(rows[name][key])
        return "--"

    base_mdd = rows.get("gate_none", {}).get("max_drawdown")

    def mdd_red(name):
        if name not in rows or base_mdd is None:
            return "--"
        d = rows[name].get("max_drawdown")
        if d is None or base_mdd == 0:
            return "--"
        return f"{(base_mdd - d) / abs(base_mdd) * 100:.1f}%"

    print("\n=== Table 1 ablation chain: V40 -> +Tier-6a -> +Regime -> Full ===")
    print(f"{'Config':22s} {'IC':>8s} {'ICIR':>8s} {'AR':>8s} {'Sharpe':>8s} {'MDD':>9s} {'TO':>8s}")
    for name, label in [("gate_none", "V40 (Baseline)"),
                        ("gate_full", "+Tier-6a"),
                        ("chain_regime", "+Regime"),
                        ("deb_k3", "Memory Alpha (Full)")]:
        if name in rows:
            print(f"{label:22s} {get(name, 'mean_ic'):>8s} {get(name, 'icir'):>8s} "
                  f"{get(name, 'annualized_return', '{:+.4f}'):>8s} "
                  f"{get(name, 'sharpe'):>8s} "
                  f"{get(name, 'max_drawdown', '{:+.4f}'):>9s} "
                  f"{get(name, 'turnover', '{:.4f}'):>8s}")

    print("\n=== Table: Tier-6a gate ablation (V40 baseline) ===")
    print(f"{'Config':22s} {'Consist':>8s} {'IC':>8s} {'Sharpe':>8s} {'MDD':>9s} {'MDD Red':>8s}")
    for name, label in [("gate_none", "No Gate (Baseline)"), ("gate_csc", "CSC Only"),
                        ("gate_psu", "PSU Only"), ("gate_dsr", "DSR Only"),
                        ("gate_csupsu", "CSC + PSU"), ("gate_full", "Full Tier-6a")]:
        if name in rows:
            print(f"{label:22s} {get(name, 'consistency'):>8s} "
                  f"{get(name, 'mean_ic'):>8s} {get(name, 'sharpe'):>8s} "
                  f"{get(name, 'max_drawdown', '{:+.4f}'):>9s} {mdd_red(name):>8s}")

    print("\n=== Table: debate / distillation ablation (full pipeline) ===")
    print(f"{'Config':30s} {'IC':>8s} {'ICIR':>8s} {'Sharpe':>8s} {'MDD':>9s}")
    for name, label in [("deb_none", "Single-agent (no debate)"),
                        ("deb_single", "Single-agent + distillation"),
                        ("deb_k1", "Debate, K=1"),
                        ("deb_k3", "Debate, K=3 (default)")]:
        if name in rows:
            print(f"{label:30s} {get(name, 'mean_ic'):>8s} {get(name, 'icir'):>8s} "
                  f"{get(name, 'sharpe'):>8s} "
                  f"{get(name, 'max_drawdown', '{:+.4f}'):>9s}")

    print("\n=== Table: portfolio optimizer ablation (full signal) ===")
    print(f"{'Optimizer':22s} {'Sharpe':>8s} {'MDD':>9s} {'Turnover':>9s}")
    for name, label in [("v2_mv", "Mean-Variance"), ("v2_bl", "Black-Litterman only"),
                        ("v2_cvar", "CVaR only"), ("v2_rb", "Risk Budgeting only"),
                        ("deb_k3", "V2 (Full)")]:
        if name in rows:
            print(f"{label:22s} {get(name, 'sharpe'):>8s} "
                  f"{get(name, 'max_drawdown', '{:+.4f}'):>9s} "
                  f"{get(name, 'turnover', '{:.4f}'):>9s}")

    with open(out / "summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n[ABL] summary written to {out / 'summary.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ablation suite runner")
    ap.add_argument("--config", default="experiments/configs/csi500.yaml")
    ap.add_argument("--data", default="experiments/data/csi500_200.csv")
    ap.add_argument("--out", default="outputs/ablation")
    ap.add_argument("--variant", default=None,
                    help="variant name (default: all, sequentially)")
    ap.add_argument("--fold-start", type=int, default=0)
    ap.add_argument("--fold-end", type=int, default=None)
    ap.add_argument("--shard", default=None,
                    help="shard suffix for the output file (e.g. f0-7)")
    ap.add_argument("--summarize", action="store_true",
                    help="print the three paper tables from merged results")
    args = ap.parse_args()

    if args.summarize:
        summarize(args)
        return
    names = [args.variant] if args.variant else list(VARIANTS)
    for name in names:
        if name not in VARIANTS:
            print(f"[ABL] unknown variant: {name}")
            sys.exit(1)
        run_variant(name, args)


if __name__ == "__main__":
    main()
