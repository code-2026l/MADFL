# MADFL — Memory Alpha: Taming the Discover-and-Forget Loop with Multi-Agent Consensus and Distillation

Memory Alpha is a self-evolving, multi-agent framework for systematic alpha factor mining. It treats factor mining as a **discover-and-forget loop** — signals are discovered, decay, and are re-learned from scratch — and manages factors across their full lifecycle: discovery, validation, calibration, deployment, and cross-generation distillation. This repository is the codebase behind the paper of the same name (WSDM 2027), and provides the code needed to reproduce the paper's experiments.

The framework is built on two orchestration protocols:

- **Adversarial multi-agent consensus** — mining, screening, debate (bull vs. bear), and fusion agents argue for and against every candidate signal, so a factor must survive structured scrutiny before entering the library.
- **Cross-generation knowledge distillation** — a teacher–student architecture carries knowledge from a growing factor library into a fixed-capacity model, cutting factor rediscovery by 67% and acting as a replay buffer against catastrophic forgetting.

Around these protocols the pipeline integrates a Tier-6a signal gate (CSC + PSU + DSR), sticky-HMM + BOCPD regime detection, multi-family factor synthesis, and a V2 portfolio optimizer (Black-Litterman + CVaR + risk budgeting + drawdown circuit breaker).

## Repository structure

The manuscript reports a 28-fold walk-forward study on CSI 500 (2021–2024) with cross-market validation on S&P 500. The paper's reported numbers are stored in `outputs/walkforward/reported_results.json` so the tables and figures can be reproduced, and the same pipeline can be re-run on your own OHLCV data.

```
MADFL/
├── madfl/                  # core package
│   ├── agents/             # mining / screening / debate / fusion agents
│   ├── gate/               # Tier-6a signal gate (CSC, PSU, DSR)
│   ├── regime/             # sticky HMM + BOCPD regime detection
│   ├── factors/            # multi-family factor synthesis (5 families)
│   ├── portfolio/          # V2 optimizer (BL + CVaR + risk budget + CB)
│   ├── distillation/       # cross-generation teacher-student distillation
│   ├── config.py           # hyperparameters
│   ├── utils.py            # IC/ICIR, Newey-West, drawdown analytics
│   ├── baselines.py        # comparison baselines (LightGBM, LSTM, AlphaForge)
│   └── pipeline.py         # 28-fold walk-forward orchestration
├── experiments/            # run the 28-fold walk-forward on real OHLCV data
│   ├── run_walkforward.py  # entry point: data -> walk-forward results
│   └── configs/csi500.yaml # experiment hyperparameters
├── scripts/
│   └── reproduce_tables.py # format reported / fresh-run tables
├── figures/                # original figure files from the paper
│   ├── architecture.png   (Figure 1: framework overview)
│   ├── results_panel.png  (Figure 2: 4-panel main results)
│   └── regime_timeline.png (Figure 3: regime assignments timeline)
├── outputs/
│   └── walkforward/        # reported results (ground-truth targets)
├── tests/                  # unit + smoke tests
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

The original figure files used in the paper (Figures 1-3) are committed under
`figures/`, together with `case_study_debate.png`, an illustration of
the September-2022 debate case study: bull/bear arguments, the fusion rule, and the
factor-weight trajectory 0.31 → 0.19 around fold 11. The manuscript source
(`paper/`) is kept private during the double-blind review period (see
`.gitignore`), and the repository otherwise contains only the code needed to
reproduce the experimental results.

## Installation

```bash
git clone https://github.com/code-2026l/MADFL.git
cd MADFL
pip install -r requirements.txt     # or: pip install -e .
```

Python 3.9+; core deps: `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`, `lightgbm`, `numba`, `hmmlearn`, `matplotlib`.

## Reproducing the paper

There are two complementary steps. The first exposes the paper's reported results; the second re-runs the pipeline on a data file of your own.

### 1. Format the paper's reported results

The aggregate numbers in the manuscript (Tables 1-7) are stored verbatim in `outputs/walkforward/reported_results.json`. This is the ground truth a fresh run should be compared against.

```bash
# Print Tables 1-7 exactly as reported in the paper
python scripts/reproduce_tables.py
```

### 2. Re-run the 28-fold walk-forward on real data

The pipeline reproduces the paper's protocol on real OHLCV data. Prepare a CSV with columns `date, code, open, high, low, close, volume` for your universe and run:

```bash
python experiments/run_walkforward.py \
    --config experiments/configs/csi500.yaml \
    --data path/to/your_ohlcv.csv

# Then format the fresh run's metrics
python scripts/reproduce_tables.py --results outputs/walkforward/run_results.json
```

The walks forward in 21-day test steps over 28 folds with a 504-day training window, matching the paper's protocol. Compare the fresh run's OOS IC / ICIR / Sharpe against the reported values in `outputs/walkforward/reported_results.json`.

> **About the data.** The paper's experiments used CSI 500 / S&P 500 OHLCV data that we cannot redistribute. The reported numbers in the paper are the surviving record of those experiments; this repository provides the code to repeat the protocol on any licensed or freely available OHLCV panel.

## Key results

The table below is the output of `python scripts/reproduce_tables.py` — these are the numbers reported in Table 1 of the paper.

| Method | IC | ICIR | AR | Sharpe | MDD |
|---|---|---|---|---|---|
| CSI 500 Index (buy-and-hold) | — | — | –2.20% | –0.12 | –33.00% |
| LightGBM | 0.0120 | 0.1209 | –1.18% | 0.21 | –18.97% |
| LSTM | 0.0175 | 0.1521 | 4.96% | 0.62 | –9.68% |
| AlphaForge | 0.0146 | 0.1299 | 3.45% | 0.33 | –17.67% |
| AlphaAgent | 0.0212 | 0.1938 | 11.00% | 0.72 | –9.36% |
| V40 (Baseline) | 0.0080 | 0.0721 | 1.20% | 0.21 | –28.40% |
| V40 + Tier-6a | 0.0142 | 0.1285 | 6.50% | 0.55 | –21.70% |
| V40 + Tier-6a + Regime | 0.0186 | 0.1682 | 9.80% | 0.72 | –15.30% |
| **Memory Alpha (Full)** | **0.0245** | **0.2248** | **13.20%** | **0.88** | **–10.80%** |

21 of 28 folds (75%) are positive; the positive mean IC is significant against the index (Newey-West corrected, `p < 0.001`). Full breakdowns — gate ablation, debate ablation, V2 optimizer ablation, cost sensitivity, and S&P 500 cross-market results — are reproduced by `scripts/reproduce_tables.py`.

## Tests

```bash
python -m pytest tests/ -q
```

The test suite exercises all core components (factor synthesis, regime detection, distillation, portfolio optimization, walk-forward pipeline) with 14 smoke tests.

## Using the LLM debate

The adversarial consensus module ships with a deterministic fallback (`RuleBasedLLMClient`) so the pipeline runs without a proprietary model. To use a frontier reasoning model, subclass `LLMClient`:

```python
from madfl.agents.base import LLMClient
from madfl.pipeline import run_walkforward

class MyLLM(LLMClient):
    def complete(self, prompt: str) -> str:
        ...  # call your model and return its response

result = run_walkforward(returns, volume, market, llm=MyLLM())
```

## License

MIT License. See [LICENSE](LICENSE).

## Citation

```bibtex
@inproceedings{madfl2027,
  title={Memory Alpha: Taming the Discover-and-Forget Loop with Multi-Agent Consensus and Distillation},
  booktitle={Proceedings of the 20th ACM International Conference on Web Search and Data Mining (WSDM)},
  year={2027},
  note={Anonymized version during review}
}
```

> **Note for the review period.** Per the venue's double-blind policy, please keep this repository private (or publish only after acceptance / as explicitly permitted) until the review process completes.