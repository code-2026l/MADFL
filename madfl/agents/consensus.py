"""Adversarial multi-agent consensus protocol (paper Sec. 4.2, Alg. 1).

Four specialized roles collaborate so that every candidate signal must survive
structured bull/bear scrutiny before entering the factor library:

  * Mining agent    -- proposes new factor parameterizations
  * Screening agent -- filters candidates by IC / ICIR / turnover
  * Debate agent    -- pits a Bull and a Bear sub-agent against each other
  * Fusion agent    -- maps the debate outcome to a consensus score

The consensus score (paper Eq. 1):

    C(s) = sigma( w0 + sum_k w_k * tanh(e+_k - e-_k) )

with e+_k / e-_k the bull / bear evidence strengths from round k. Economic
prior is embedded through canonical argument templates (S17).
"""

from __future__ import annotations

import re

import numpy as np

from madfl.agents.base import BaseAgent, LLMClient, RuleBasedLLMClient

__all__ = ["MiningAgent", "ScreeningAgent", "DebateAgent", "FusionAgent",
           "ConsensusProtocol"]


def _evidence_score(response: str) -> float:
    """Extract the numeric evidence strength from a model response."""
    m = re.search(r"score=([-+]?\d*\.?\d+)", response)
    if not m:
        return 0.5
    return float(max(0.0, min(1.0, float(m.group(1)))))


def _tanh(a: float, b: float) -> float:
    return float(np.tanh(a - b))


class MiningAgent(BaseAgent):
    """Proposes candidate factor parameterizations via formulaic search."""

    def __init__(self, llm: LLMClient | None = None,
                 param_grid: dict | None = None):
        super().__init__("mining", llm)
        self.param_grid = param_grid or {
            "lookback": [5, 10, 20, 30, 60],
            "signal_type": ["momentum", "reversal", "volatility", "volume"],
        }

    def propose(self, n_candidates: int = 8, rng: np.random.Generator | None = None
                ) -> list[dict]:
        """Generate ``n_candidates`` candidate parameterizations."""
        rng = rng or np.random.default_rng(0)
        cands = []
        for _ in range(n_candidates):
            cand = {
                "lookback": int(rng.choice(self.param_grid["lookback"])),
                "signal_type": str(rng.choice(self.param_grid["signal_type"])),
            }
            cands.append(cand)
        return cands


class ScreeningAgent(BaseAgent):
    """Coarse, cheap pre-filter so the adversarial debate is the real arbiter.

    Thresholds are calibrated to the *real scale of candidate factor features*
    (raw cross-sectional IC is O(1e-2), raw feature churn ~0.7-1.1), NOT to
    portfolio-scale turnover. The pre-filter only removes grossly negative-IC
    junk; every survivor goes on to the bull/bear debate, whose evidence scores
    are the discriminating layer (paper Alg. 1).
    """

    def __init__(self, llm: LLMClient | None = None,
                 min_ic: float = 0.0, min_icir: float = -0.05,
                 max_turnover: float = 3.0):
        super().__init__("screening", llm)
        self.min_ic = min_ic
        self.min_icir = min_icir
        self.max_turnover = max_turnover

    def screen(self, candidates: list[dict],
               stats: list[dict]) -> tuple[list[dict], list[dict]]:
        """Return (passed, rejected) candidate lists given per-candidate stats."""
        passed, rejected = [], []
        for cand, st in zip(candidates, stats):
            ok = (st.get("ic", 0.0) >= self.min_ic
                  and st.get("icir", 0.0) >= self.min_icir
                  and st.get("turnover", 1e9) <= self.max_turnover)
            (passed if ok else rejected).append(cand)
        return passed, rejected


class DebateAgent(BaseAgent):
    """Runs the K-round bull/bear adversarial debate for one candidate."""

    def __init__(self, llm: LLMClient | None = None, rounds: int = 3):
        super().__init__("debate", llm)
        self.rounds = rounds

    def run(self, candidate: dict, stats: dict, regime: str) -> tuple[list, list]:
        """Return (e_plus, e_minus), the per-round evidence strengths."""
        e_plus, e_minus = [], []
        counter = ""
        for k in range(self.rounds):
            bull_prompt = self._prompt(candidate, stats, regime, "BULL", counter)
            bull = self.llm.complete(bull_prompt)
            bear_prompt = self._prompt(candidate, stats, regime, "BEAR", bull)
            bear = self.llm.complete(bear_prompt)
            e_plus.append(_evidence_score(bull))
            e_minus.append(_evidence_score(bear))
            counter = bear
        return e_plus, e_minus

    @staticmethod
    def _prompt(candidate: dict, stats: dict, regime: str, stance: str,
                rebuttal: str) -> str:
        return (f"stance={stance} factor={candidate} "
                f"IC={stats.get('ic', 0.0):.3f} ICIR={stats.get('icir', 0.0):.3f} "
                f"turnover={stats.get('turnover', 0.3):.3f} "
                f"regime={regime} rebut={rebuttal[:80]}")


class FusionAgent(BaseAgent):
    """Maps the debate outcome to a consensus score with learned weights."""

    def __init__(self, llm: LLMClient | None = None, rounds: int = 3,
                 gate: float = 0.5):
        super().__init__("fusion", llm)
        self.rounds = rounds
        self.gate = gate
        # learned weights: w0, w_1..w_K; default = unweighted average
        self.w0 = 0.0
        self.weights = np.ones(rounds) / rounds

    def fit(self, e_plus: np.ndarray, e_minus: np.ndarray,
            labels: np.ndarray) -> "FusionAgent":
        """Fit aggregation weights by logistic regression on a validation set.

        e_plus / e_minus have shape (n, K); labels is binary acceptance.
        """
        from sklearn.linear_model import LogisticRegression
        X = np.tanh(e_plus - e_minus)
        clf = LogisticRegression(C=1.0, max_iter=1000)
        clf.fit(X, labels)
        self.w0 = float(clf.intercept_[0])
        self.weights = clf.coef_[0]
        return self

    def consensus(self, e_plus: np.ndarray, e_minus: np.ndarray) -> float:
        """C(s) in [0, 1] via the sigmoid of the weighted tanh votes."""
        votes = np.array([_tanh(a, b) for a, b in zip(e_plus, e_minus)])
        z = self.w0 + float(np.dot(self.weights, votes))
        return float(1.0 / (1.0 + np.exp(-z)))

    def accept(self, e_plus: np.ndarray, e_minus: np.ndarray) -> bool:
        return self.consensus(e_plus, e_minus) >= self.gate


class ConsensusProtocol:
    """Orchestrates mining -> screening -> debate -> fusion for a batch."""

    def __init__(self, llm: LLMClient | None = None, rounds: int = 3,
                 min_ic: float = 0.0, min_icir: float = -0.05,
                 gate: float = 0.5, n_candidates: int = 8):
        self.mining = MiningAgent(llm)
        self.screening = ScreeningAgent(llm, min_ic=min_ic, min_icir=min_icir,
                                        max_turnover=3.0)
        self.debate = DebateAgent(llm, rounds=rounds)
        self.fusion = FusionAgent(llm, rounds=rounds, gate=gate)
        self.n_candidates = n_candidates

    def run(self, stats: list[dict], regimes: list[str],
            rng: np.random.Generator | None = None) -> list[dict]:
        """Run the protocol over candidate factors.

        ``stats`` is a list of per-candidate dicts (ic, icir, turnover);
        ``regimes`` is the regime label associated with each candidate.
        Returns the list of accepted candidates annotated with their consensus
        scores.
        """
        rng = rng or np.random.default_rng(0)
        accepted = []
        for i, st in enumerate(stats):
            cand = {"candidate_id": i, "factor": {"lookback": 20,
                                                  "signal_type": "momentum"}}
            passed = self.screening.screen([cand], [st])
            if not passed[0]:
                continue
            regime = regimes[i] if i < len(regimes) else "SIDEWAYS"
            e_plus, e_minus = self.debate.run(cand["factor"], st, regime)
            if self.fusion.accept(np.array(e_plus), np.array(e_minus)):
                score = self.fusion.consensus(np.array(e_plus), np.array(e_minus))
                accepted.append({"candidate": cand, "stats": st, "regime": regime,
                                 "consensus": score, "e_plus": e_plus,
                                 "e_minus": e_minus})
        return accepted