"""Format the paper's tables from the authoritative experimental results.

Loads ``outputs/walkforward/reported_results.json`` — the exact numbers
reported in the WSDM'27 paper (Tables 1-9 + CDS diagnostics, Section S10.1) —
and prints them as formatted tables. These are the ground-truth targets a
fresh run should be compared against.

After re-running the pipeline yourself (``experiments/run_ablation.py``), pass
``--results <run_results.json>`` to format a fresh run's metrics.

Usage
-----
  python scripts/reproduce_tables.py                         # print paper tables
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
    widths = [len(c) for c in cols]
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
    data = []
    for r in rows:
        to = r.get("to") if r.get("to") is not None else r.get("turnover")
        data.append([r["config"], _fmt4(r.get("ic")), _fmt4(r.get("icir")),
                     _fmt_pct(r.get("ar")), _fmt2(r.get("sharpe")),
                     _fmt_pct(r.get("mdd")),
                     "--" if to is None else f"{to * 100:.1f}%"])
    _print_header(
        "Table 1 - 28-Fold Walk-Forward Results on CSI 500 (OOS 2022-08..2024-12)",
        ["Configuration", "IC", "ICIR", "AR", "Sharpe", "MDD", "TO"], data)


def table2_debate(results: dict) -> None:
    rows = results["table2_debate_2x2_cds"]
    data = [
        [r["config"], _fmt4(r["ic"]), _fmt4(r["icir"]),
         _fmt2(r["sharpe"]), _fmt_pct(r["mdd"]), str(r["tokens"])]
        for r in rows
    ]
    _print_header(
        "Table 2 - Distillation x Debate 2x2 Grid (CDS screening)",
        ["Configuration", "IC", "ICIR", "Sharpe", "MDD", "Tokens"], data)


def table3_cds(results: dict) -> None:
    rows = results["table3_cds_screening_comparison"]
    data = [
        [r["rule"],
         "--" if r["k_eff"] is None else f"{r['k_eff']:.2f}",
         "--" if r["winner_ic"] is None else f"{r['winner_ic']:+.3f}",
         "--" if r["ic_star"] is None else f"{r['ic_star']:+.4f}",
         _fmt4(r["oos_ic"]), _fmt4(r["icir"]),
         "--" if r["nw_t"] is None else f"{r['nw_t']:+.2f}"]
        for r in rows
    ]
    _print_header(
        "Table 3 - CDS Screening-Rule Comparison (teacher-only rows)",
        ["Rule", "K_eff", "Winner IC", "IC*", "OOS IC", "ICIR", "t(NW)"], data)


def table4_gate(results: dict) -> None:
    g = results["table4_tier6a_gate_ablation"]
    names = [("baseline", "No Gate (Baseline)"), ("csc", "CSC Only"),
             ("psu", "PSU Only"), ("dsr", "DSR Only"),
             ("csc_psu", "CSC + PSU"), ("full", "Full Tier-6a")]
    data = []
    for key, label in names:
        r = g.get(key) or {}
        data.append([label, _fmt4(r.get("consistency")), _fmt4(r.get("mean_ic")),
                     _fmt2(r.get("sharpe")), _fmt_pct(r.get("max_drawdown"))])
    _print_header(
        "Table 4 - Tier-6a Gate Ablation on V40 Baseline",
        ["Configuration", "Consistency", "IC", "Sharpe", "MDD"], data)


def table5_sign(results: dict) -> None:
    rows = results["table5_sign_direction"]
    data = [
        [r["direction"], _fmt4(r["ic"]), f"{r['nw_t']:+.2f}"]
        for r in rows
    ]
    _print_header(
        "Table 5 - Sign-Direction Ablation of the Raw Multi-Family Composite",
        ["Signal direction", "Mean OOS IC", "t (Newey-West)"], data)


def table6_rev_gate(results: dict) -> None:
    rg = results["table6_rev_gate"]
    rows = [
        ["Full-time reversal (gross)", _fmt_pct(rg["reversal_gross"]["ar"]),
         _fmt2(rg["reversal_gross"]["sharpe"]), _fmt_pct(rg["reversal_gross"]["mdd"])],
        ["+ HMM gate, floor 0.50", _fmt_pct(rg["hmm_gate"]["f0_50"]["ar"]),
         _fmt2(rg["hmm_gate"]["f0_50"]["sharpe"]), _fmt_pct(rg["hmm_gate"]["f0_50"]["mdd"])],
        ["+ HMM gate, floor 0.25", _fmt_pct(rg["hmm_gate"]["f0_25"]["ar"]),
         _fmt2(rg["hmm_gate"]["f0_25"]["sharpe"]), _fmt_pct(rg["hmm_gate"]["f0_25"]["mdd"])],
        ["+ HMM gate, floor 0.00", _fmt_pct(rg["hmm_gate"]["f0_00"]["ar"]),
         _fmt2(rg["hmm_gate"]["f0_00"]["sharpe"]), _fmt_pct(rg["hmm_gate"]["f0_00"]["mdd"])],
        ["+ Transparent gate, floor 0.50", _fmt_pct(rg["transparent_gate"]["f0_50"]["ar"]),
         _fmt2(rg["transparent_gate"]["f0_50"]["sharpe"]), _fmt_pct(rg["transparent_gate"]["f0_50"]["mdd"])],
        ["+ Transparent gate, floor 0.25", _fmt_pct(rg["transparent_gate"]["f0_25"]["ar"]),
         _fmt2(rg["transparent_gate"]["f0_25"]["sharpe"]), _fmt_pct(rg["transparent_gate"]["f0_25"]["mdd"])],
        ["+ Transparent gate, floor 0.00", _fmt_pct(rg["transparent_gate"]["f0_00"]["ar"]),
         _fmt2(rg["transparent_gate"]["f0_00"]["sharpe"]), _fmt_pct(rg["transparent_gate"]["f0_00"]["mdd"])],
    ]
    _print_header(
        "Table 6 - Adverse-Regime Gating of the Reversal Signal",
        ["Deployment (reversal signal)", "AR", "Sharpe", "MDD"], rows)


def table7_v2(results: dict) -> None:
    v2 = results["table7_v2_optimizer"]
    names = [("mv", "Mean-Variance"), ("bl", "Black-Litterman only"),
             ("cvar", "CVaR only"), ("rb", "Risk Budgeting only"),
             ("v2_full", "V2 (Full)")]
    data = []
    for key, label in names:
        r = v2.get(key) or {}
        data.append([label, _fmt_pct(r.get("ar")), _fmt2(r.get("sharpe")),
                     _fmt_pct(r.get("mdd")),
                     "--" if r.get("to") is None else f"{r['to'] * 100:.1f}%"])
    _print_header(
        "Table 7 - V2 Portfolio-Optimizer Ablation",
        ["Optimizer", "AR", "Sharpe", "MDD", "Turnover"], data)


def table8_cross(results: dict) -> None:
    spx = results["table8_cross_market"]
    if not isinstance(spx, dict) or "rows" not in spx:
        print("\n=== Table 8 - Cross-Market (S&P 500) ===\n  see "
              "outputs/walkforward/results_legacy/spx_reversal_fast.json")
        return
    data = [
        [r["metric"], "--" if r.get("csi") is None else str(r["csi"]),
         "--" if r.get("spx") is None else str(r["spx"])]
        for r in spx["rows"]
    ]
    _print_header(
        "Table 8 - Cross-Market Comparison (identical protocol)",
        ["Metric", "CSI 500 (CN)", "S&P 500 (US)"], data)


def table9_stream(results: dict) -> None:
    st = results["table9_stream"]
    mn = st.get("mnist") or {}
    ml = st.get("movielens") or {}
    print("\n=== Table 9 - Non-Financial Streaming Transfer ===")
    print(f"  rotating MNIST: avg accuracy = {mn.get('avg_accuracy', '--')}, "
          f"forgetting rate = {mn.get('forgetting_rate', '--')}")
    print(f"  MovieLens final-slice hit-rate = {ml.get('final_hit_rate', '--')}")


def _load_reported() -> dict:
    path = ROOT / "outputs" / "walkforward" / "reported_results.json"
    if not path.exists():
        print(f"Reported results not found at {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Format paper tables from results")
    ap.add_argument("--results", type=str, default=None,
                    help="Path to a fresh run's results json to format instead "
                         "of the paper's reported results")
    args = ap.parse_args()

    if args.results:
        with open(args.results, encoding="utf-8") as f:
            results = json.load(f)
        print("\n=== Fresh run (walk-forward metrics) ===")
        for k, v in results.items():
            if isinstance(v, (int, float)) and k not in ("n_folds", "seed"):
                print(f"  {k:24s}: {v:.4f}" if isinstance(v, float)
                      else f"  {k:24s}: {v}")
        print("\nNote: the paper's Tables 1-9 are in outputs/walkforward/"
              "reported_results.json; a fresh run should be compared against "
              "those numbers.")
        return

    results = _load_reported()
    p = results["protocol"]
    print(f"Protocol: {p['universe']}, {p['validation']}, "
          f"train {p['train_days']}d / test {p['test_days']}d, seed {p['seed']}")
    print(f"CDS diagnostics: K_eff = {p['cds_k_eff_mean']:.3f}, "
          f"null sigma = {p['cds_null_sigma_mean']:.5f}, "
          f"deflated winner IC* = {p['cds_defl_winner_mean']:+.4f}")
    table1_main(results)
    table2_debate(results)
    table3_cds(results)
    table4_gate(results)
    table5_sign(results)
    table6_rev_gate(results)
    table7_v2(results)
    table8_cross(results)
    table9_stream(results)
    hl = results.get("factor_half_life_days")
    if isinstance(hl, dict) and "median_days" in hl:
        print(f"\nFactor half-life: median = {hl['median_days']} trading days")
    else:
        print(f"\nFactor half-life: median = 12.8 trading days (Section 4.2)")


if __name__ == "__main__":
    main()
