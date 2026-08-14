"""Smoke tests exercising every module of the Memory Alpha framework.

Run with:
    python -m pytest tests/ -q
    (or)  python tests/test_smoke.py
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from madfl.config import MemoryAlphaConfig
from madfl.gate import Tier6aGate, cross_sectional_consensus, \
    predictive_stability, deflated_sharpe
from madfl.regime import StickyHMM, BayesianOnlineChangepoint, RegimeDetector
from madfl.factors.synthesizer import FactorSynthesizer
from madfl.portfolio.v2 import V2Optimizer, cvar, risk_budget_deviation
from madfl.agents.consensus import ConsensusProtocol
from madfl.distillation.memory_alpha import FeaturePool, MemoryAlphaDistillation
from madfl.pipeline import run_walkforward
from madfl.utils import rank_ic, icir, max_drawdown, sharpe


def _random_panel(n_days: int, n_stocks: int, seed: int = 0):
    """Tiny random panel to exercise the pipeline end-to-end in tests.

    This is a unit-test fixture only — it is not experimental data and is not
    used for any reported result.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.02, (n_days, n_stocks))
    volume = np.abs(rng.normal(1e6, 2e5, (n_days, n_stocks)))
    market = returns.mean(axis=1)
    return returns, volume, market


class TestGate(unittest.TestCase):
    def test_csc(self):
        self.assertAlmostEqual(cross_sectional_consensus(np.array([1, 1, 1])), 1.0, places=4)

    def test_psu(self):
        s = np.ones(25)
        self.assertEqual(predictive_stability(s, 20), 1.0)

    def test_dsr_bounds(self):
        d = deflated_sharpe(1.0, 250, 0.0, 3.0)
        self.assertTrue(0.0 <= d <= 1.0)

    def test_calibrate(self):
        gate = Tier6aGate()
        p = gate.calibrate(80.0, 0.2)
        self.assertLess(p, 80.0)


class TestRegime(unittest.TestCase):
    def test_hmm(self):
        rng = np.random.default_rng(0)
        X = np.concatenate([rng.normal(0, 1, (200, 2)), rng.normal(3, 1, (200, 2))])
        hmm = StickyHMM(n_states=2, beta=3.0)
        hmm.fit(X)
        states = hmm.predict(X)
        self.assertEqual(states.shape, (X.shape[0],))

    def test_bocpd(self):
        data = np.concatenate([np.random.normal(0, 1, 100),
                               np.random.normal(5, 1, 100)])
        probs = BayesianOnlineChangepoint().run(data)
        self.assertEqual(probs.size, data.size)
        self.assertTrue(np.all(probs >= 0))

    def test_detector(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (50, 3))
        det = RegimeDetector()
        det.fit(X)
        scores = det.detect(X, np.ones(50))
        self.assertEqual(scores.shape, (50,))


class TestFactors(unittest.TestCase):
    def test_synthesizer(self):
        rng = np.random.default_rng(2)
        returns = rng.normal(0, 0.01, (300, 12))
        volume = np.abs(rng.normal(1e6, 1, (300, 12)))
        market = returns.mean(axis=1)
        out = FactorSynthesizer().synthesize(returns, volume, market)
        self.assertEqual(out.shape, (300, 12))
        self.assertTrue(np.isfinite(out).all())


class TestPortfolio(unittest.TestCase):
    def test_v2(self):
        rng = np.random.default_rng(3)
        returns = rng.normal(0.0005, 0.02, (200, 8))
        mu = returns.mean(axis=0)
        cov = np.cov(returns.T)
        w = V2Optimizer().optimize(returns, mu, cov)
        self.assertAlmostEqual(w.sum(), 1.0, places=4)
        self.assertTrue(np.all(w >= 0))


class TestConsensus(unittest.TestCase):
    def test_protocol(self):
        stats = [{"ic": 0.03, "icir": 0.3, "turnover": 0.2},
                 {"ic": 0.001, "icir": 0.01, "turnover": 0.9}]
        accepted = ConsensusProtocol().run(stats, ["BULL", "BEAR_CRASH"])
        self.assertIsInstance(accepted, list)


class TestDistillation(unittest.TestCase):
    def test_feature_pool(self):
        pool = FeaturePool()
        pool.set_initial(["a", "b", "c"], {"a": 0.3, "b": 0.2, "c": 0.1})
        self.assertEqual(pool.size, 3)
        self.assertTrue(pool.consider("d", 0.5, age=100))

    def test_distill(self):
        rng = np.random.default_rng(4)
        X = rng.normal(0, 1, (200, 6))
        y = X[:, 0] * 0.5 + rng.normal(0, 0.1, 200)
        md = MemoryAlphaDistillation()
        md.init_pool(X, y)
        md.fit_teacher(X, y)
        md.distill(X, y, X)
        pred = md.predict_student(X)
        self.assertEqual(pred.shape, (200,))


class TestPipeline(unittest.TestCase):
    def test_walkforward(self):
        returns, volume, market = _random_panel(400, 30, seed=7)
        cfg = MemoryAlphaConfig(n_folds=3, train_days=60, test_days=21,
                                n_stocks=30)
        result = run_walkforward(returns, volume, market, cfg)
        for key in ("mean_ic", "icir", "sharpe", "max_drawdown"):
            self.assertIn(key, result)


class TestUtils(unittest.TestCase):
    def test_rank_ic(self):
        ic = rank_ic(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]))
        self.assertGreater(ic, 0.99)


if __name__ == "__main__":
    unittest.main(verbosity=2)