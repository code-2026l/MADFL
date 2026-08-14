"""Format the paper's tables from experimental results.

By default this loads the paper's reported results (``outputs/walkforward/
reported_results.json``) — the aggregate numbers reported in Tables 1-7 of the
manuscript — and prints them as formatted tables.  These are the ground-truth
targets that a fresh run should be compared against.

After running the pipeline on real data (``experiments/run_walkforward.py``),
pass ``--results outputs/walkforward/run_results.json`` to format the fresh
run's metrics into the same table layout.

Usage
-----
  python scripts/reproduce_tables.py                          # print paper tables
  python scripts/reproduce_tables.py --results <run_results.json>  # print a fresh run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _fmt_pct(x) -> str:
    return "--" if x is None else f"{x * 100:+.2f}%"


def _fmt4(x) -> str:
    return "--" if x is None else f"{x:.4f}"


def _fmt2(x) -> str:
    return "--" if x is None else f"{x:.2f}"


def _print_header(title: str, cols: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(title.split("—")[0]), len(c)) for c in cols]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join("{" + str(i) + ":<" + str(w) + "}" for i, w in enumerate(widths))
    print(f"\n=== {title} ===")
    print("-" * (sum(widths) + 2 * (len(cols) - 1)))
    print(fmt.format(*cols))
    print("-" * (sum(widths) + 2 * (len(cols) - 1)))
    for r in rows:
        print(fmt.format(*r))


def table1_main(results: dict) -> None:
    rows = results["table1_main_results_csi500"]
    data = [
        [r["method"], _fmt4(r["ic"]), _fmt4(r["icir"]),
         _fmt_pct(r["ar"]), _fmt2(r["sharpe"]), _fmt_pct(r["mdd"]), _fmt4(r["ir"])]
        for r in rows
    ]
    _print_header(
        "Table 1 — 28-Fold Walk-Forward Results on CSI 500 (2021-2024)",
        ["Method", "IC", "ICIR", "AR", "Sharpe", "MDD", "IR"], data)


def table2_gate(results: dict) -> None:
    rows = results["table2_tier6a_gate_ablation"]
    data = [
        [r["config"], f"{r['consistency']:.2f}", _fmt4(r["ic"]),
         _fmt2(r["sharpe"]), f"{r['mdd_reduction'] * 100:.1f}%"]
        for r in rows
    ]
    _print_header(
        "Table 2 — Tier-6a Gate Ablation on V40 Baseline",
        ["Configuration", "Consistency", "IC", "Sharpe", "MDD Reduction"], data)


def table3_debate(results: dict) -> None:
    rows = results["table3_debate_ablation"]
    data = []
    for r in rows:
        tok = "0" if r["tokens"] == 0 else f"{r['tokens'] / 1e6:.1f}M"
        data.append([r["config"], _fmt4(r["ic"]), _fmt4(r["icir"]),
                     _fmt2(r["sharpe"]), _fmt_pct(r["mdd"]), tok])
    _print_header(
        "Table 3 — Adversarial Debate Protocol Ablation",
        ["Configuration", "IC", "ICIR", "Sharpe", "MDD", "Tokens"], data)


def table4_regime(results: dict) -> None:
    rows = results["table4_regime_sticky_vs_standard"]["rows"]
    data = [
        [r["metric"],
         "--" if r["standard_hmm"] is None else str(r["standard_hmm"]),
         "--" if r["sticky_hmm"] is None else str(r["sticky_hmm"])]
        for r in rows
    ]
    _print_header(
        "Table 4 — Regime Classification: Sticky HMM vs Standard HMM",
        ["Metric", "Standard HMM", "Sticky HMM"], data)


def table5_v2(results: dict) -> None:
    rows = results["table5_v2_optimizer_ablation"]
    data = [
        [r["optimizer"], _fmt2(r["sharpe"]), _fmt_pct(r["mdd"]),
         f"{r['turnover'] * 100:.1f}%"]
        for r in rows
    ]
    _print_header(
        "Table 5 — V2 Portfolio Optimizer Ablation",
        ["Optimizer", "Sharpe", "MDD", "Daily Turnover"], data)


def table6_cost(results: dict) -> None:
    rows = results["table6_cost_sensitivity"]
    data = [
        [f"{r['cost_bps']} bps", _fmt_pct(r["net_ar"]), _fmt2(r["net_sharpe"]),
         _fmt4(r["net_icir"]), _fmt_pct(r["net_mdd"])]
        for r in rows
    ]
    _print_header(
        "Table 6 — Net Performance under Round-Trip Transaction Costs",
        ["Cost", "Net AR", "Net Sharpe", "Net ICIR", "Net MDD"], data)


def table7_cross(results: dict) -> None:
    rows = results["table7_cross_market_sp500"]
    data = [
        [r["method"], _fmt4(r["ic"]), _fmt4(r["icir"]),
         _fmt_pct(r["ar"]), _fmt2(r["sharpe"]), _fmt_pct(r["mdd"])]
        for r in rows
    ]
    _print_header(
        "Table 7 — Cross-Market Validation on S&P 500 (2021-2024)",
        ["Method", "IC", "ICIR", "AR", "Sharpe", "MDD"], data)


def _load_reported() -> dict:
    path = ROOT / "outputs" / "walkforward" / "reported_results.json"
    if not path.exists():
        print(f"Reported results not found at {path}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Format paper tables from results")
    ap.add_argument("--results", type=str, default=None,
                    help="Path to a fresh run's run_results.json to format instead "
                         "of the paper's reported results")
    args = ap.parse_args()

    if args.results:
        with open(args.results) as f:
            results = json.load(f)
        # A fresh run produces a metric dict; lay it out as a single-row table.
        print("\n=== Fresh run (walk-forward metrics) ===")
        for k, v in results.items():
            if isinstance(v, (int, float)) and k not in ("n_folds", "seed"):
                print(f"  {k:24s}: {v:.4f}" if isinstance(v, float) else f"  {k:24s}: {v}")
        print("\nNote: paper Tables 1-7 are reported in outputs/walkforward/"
              "reported_results.json (see paper main.tex). A fresh run should be "
              "compared against these numbers.")
        return

    results = _load_reported()
    table1_main(results)
    table2_gate(results)
    table3_debate(results)
    table4_regime(results)
    table5_v2(results)
    table6_cost(results)
    table7_cross(results)

    hl = results["factor_half_life"]
    print(f"\nFactor half-life: median = {hl['median_days']} trading days "
          f"({hl['note']})")


if __name__ == "__main__":
    main()