# MADFL — Memory Alpha: What Survives Honest Walk-Forward Evaluation?

**WSDM 2027 submission · double-blind review period · anonymous repository**

Memory Alpha is a full-stack, agentic alpha-factor-mining framework released
together with a rigorous, fully reproducible evaluation protocol. The paper
asks a pointed question: *when an LLM-driven agentic factor-mining pipeline is
evaluated honestly — date-aligned, end-aligned, intent-to-treat walk-forward —
what actually survives?*

The answer, backed by the code and data in this repository (28-fold
walk-forward, 160 CSI 500 stocks, seed 42, OOS 2022-08 .. 2024-12):

1. **The honest cross-sectional signal is short-term reversal, not momentum**
   (OOS IC +0.018, Newey–West t = +2.63, direction fixed).
2. **Search bias — not weak analysis — inflates agentic claims.** With the
   legacy in-sample screen the agent's "best" family reports IC **+0.1697**
   (ICIR +5.24); after our correction the same search yields **+0.0009**
   (t = 0.38).
3. **Once the search is corrected, the two advertised mechanisms separate
   cleanly:** cross-generation distillation *improves* OOS IC
   (**+0.0028 → +0.0116**, paired t = 4.69), while LLM debate contributes
   nothing (debate-only: **+0.0009**, t = 0.38; debate on top of distillation
   *lowers* IC to +0.0079).
4. **The reversal edge is economically immaterial** (gross Sharpe ≈ 0.005,
   negative net of costs); the "regime-gated" Sharpe lift (0.49–0.84) is a
   latent-state mislabeling artifact.

## Contributions

- **HAE — Honest Agentic Evaluation protocol** (paper §4.1): formal
  definitions of *panel selection bias* and *latent-state arbitrage*, a
  label-permutation bound on gated Sharpe, and an intent-to-treat 28-fold
  walk-forward procedure (Algorithm 1).
- **CDS — Causal Deflation Screening** (paper §4.6): the first screening
  *algorithm* that prices an agent's search. CDS treats the search as a
  stochastic hypothesis-generating process, measures its effective size
  K_eff on a strictly-causal prior window (spectral participation ratio),
  corrects the winner's-curse inflation of every reported IC via a
  block-permutation null, and serves as both a deployable selection rule and
  an honest reporting metric (the deflated IC, IC\*).
- **Full-stack agentic miner**: multi-family factor synthesis (5 anomaly
  families), adversarial multi-agent consensus (bull/bear debate + fusion),
  Tier-6a signal gate (CSC + PSU + DSR), sticky-HMM + BOCPD regime detection,
  and a V2 portfolio optimizer (Black–Litterman + CVaR + risk budgeting +
  drawdown circuit breaker).

## Repository structure

```
MADFL/
├── madfl/                  # core package
│   ├── agents/             # mining / screening / debate / fusion agents
│   ├── distillation/       # cross-generation teacher-student distillation
│   ├── factors/            # multi-family factor synthesis (5 families)
│   ├── gate/               # Tier-6a signal gate (CSC, PSU, DSR)
│   ├── portfolio/          # V2 optimizer (BL + CVaR + risk budget + CB)
│   ├── regime/             # sticky HMM + BOCPD regime detection
│   ├── pipeline.py         # walk-forward orchestration + CDS screening
│   ├── config.py           # hyperparameters
│   └── utils.py            # IC/ICIR, Newey-West, drawdown analytics
├── experiments/            # reproduction entry points
│   ├── run_ablation.py     # the 12-variant distillation x debate grid + CDS
│   ├── run_walkforward.py  # single-pipeline walk-forward entry point
│   ├── merge_shards.py     # stitch sharded fold runs
│   ├── configs/csi500.yaml # experiment hyperparameters
│   └── data/               # CSI 500 OHLCV panel used in the paper (public)
├── scripts/
│   └── reproduce_tables.py # format all paper tables from the results
├── figures/                # paper figures (architecture / results / regime)
├── outputs/
│   └── walkforward/
│       ├── reported_results.json   # authoritative numbers for Tables 1-9
│       ├── results_cds/            # per-fold series, 12 debate/distillation variants
│       └── results_legacy/         # gate / reversal / V2 / cross-market outputs
├── tests/
│   └── test_smoke.py       # smoke test (runs 1 fold)
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

The manuscript source is kept private during the double-blind review period
(`paper/` is git-ignored); everything needed to reproduce and verify the
experimental results is public here.

## Installation

```bash
git clone <anonymous-repo-url> MADFL
cd MADFL
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt     # or: pip install -e .
```

## Reproduce the paper

### 1. Print the paper's tables (ground-truth targets)

```bash
python scripts/reproduce_tables.py
```

This reads `outputs/walkforward/reported_results.json` (the authoritative
numbers for Tables 1–9 and the CDS diagnostics, Section S10.1) and prints the
formatted tables. The raw per-fold series for the 12 debate/distillation
variants are in `outputs/walkforward/results_cds/*.json`.

### 2. Re-run the pipeline from raw OHLCV

```bash
# the full 12-variant distillation x debate grid (28 folds each, ~1-2 h total
# on a multi-core machine; each variant can be sharded with --fold-start/--fold-end)
python experiments/run_ablation.py --variant deb_none --out outputs/walkforward/results_cds
python experiments/run_ablation.py --variant deb_single --out outputs/walkforward/results_cds
python experiments/run_ablation.py --variant deb_k1 --out outputs/walkforward/results_cds
python experiments/run_ablation.py --variant deb_k3 --out outputs/walkforward/results_cds
python experiments/run_ablation.py --variant deb_k1_nodistill --out outputs/walkforward/results_cds
python experiments/run_ablation.py --variant deb_k3_nodistill --out outputs/walkforward/results_cds
python experiments/run_ablation.py --variant defl_none --out outputs/walkforward/results_cds
python experiments/run_ablation.py --variant defl_single --out outputs/walkforward/results_cds
python experiments/run_ablation.py --variant defl_k1 --out outputs/walkforward/results_cds
python experiments/run_ablation.py --variant defl_k3 --out outputs/walkforward/results_cds
python experiments/run_ablation.py --variant defl_k1_nodistill --out outputs/walkforward/results_cds
python experiments/run_ablation.py --variant defl_k3_nodistill --out outputs/walkforward/results_cds

# format a fresh run's metrics into the same layout
python scripts/reproduce_tables.py --results outputs/walkforward/results_cds/defl_k3.json
```

Variants:
- `deb_*` — honest walk-forward with **causal-prior-window** consensus
  screening (no in-sample selection leak);
- `defl_*` — the same grid under **Causal Deflation Screening (CDS)**,
  i.e. the paper's headline numbers (Table 2);
- distillation `{on, off}` × debate `{off, K=1, K=3}` — the full 2×2 grid;
- `defl_k3` is the full system; `defl_k1_nodistill` isolates the debate
  (the reviewer-facing question "debate on, distillation off").

> **Reproducibility note.** The reported numbers were produced with the exact
> dependency versions pinned in `requirements.txt` (Python 3.11, Linux,
> numpy 1.26.4 / pandas 2.1.4 / xgboost 2.0.3 / lightgbm 3.3.5 /
> scikit-learn 1.7.2 / scipy 1.16.3). Gradient-boosted models are
> deterministic for a fixed environment, but per-fold IC can shift by
> ~1e-3–5e-3 across platforms/versions (BLAS, xgboost/lightgbm point
> releases). For verification of the reported *tables*, pin the exact
> versions and compare against `outputs/walkforward/reported_results.json`;
> the aggregate metrics (mean IC, ICIR, Sharpe) are stable to well within
> ±1e-3, and the CDS diagnostics (K_eff, σ̂, IC\*) are version-robust.

### 3. Run the smoke test

```bash
python -m pytest tests/ -q          # or: python tests/test_smoke.py
```

## Data

- `experiments/data/csi500_200.csv` — the CSI 500 OHLCV panel used in the
  paper: 200 candidate names, full daily coverage 2019-01-02 .. 2024-12-31,
  sourced from public market data (akshare). The protocol keeps the 160 names
  with full coverage on the tail window (2020-07-06 .. 2024-12-31) and
  date-/end-aligns the 28-fold walk-forward on it.
- `outputs/walkforward/results_legacy/spx_reversal_fast.json` — the S&P 500
  cross-market run (Table 8 / Section S6), on the same protocol.
- The MNIST / MovieLens streaming-transfer numbers (Table 9 / Section S18)
  are summarized in `reported_results.json`; the adaptation scripts follow
  the same `madfl` modules with hard voting replacing the LLM debate.

All data are public prices and volumes; no non-public, insider, or proprietary
data are used.

## Double-blind note

During the double-blind review period this repository is anonymous: no author
identifiers appear in code, configuration, or metadata, and the manuscript
source (`paper/`) is git-ignored. After acceptance the manuscript will be
published and the repository will be linked.

## Citation

```bibtex
@inproceedings{madfl2027,
  title     = {Memory Alpha: What Survives Honest Walk-Forward Evaluation?},
  author    = {Anonymous, WSDM 2027 submission},
  booktitle = {Proceedings of the 30th ACM International Conference on Web
               Search and Data Mining (WSDM)},
  year      = {2027},
  note      = {Under double-blind review; anonymized for submission}
}
```

## License

MIT — see [LICENSE](LICENSE).
