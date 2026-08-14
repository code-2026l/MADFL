"""Tier-6a signal gate.

A triple-gating mechanism that dynamically adjusts signal conviction based on
three complementary quality dimensions (paper Eq. 5):

    G(s_t) = alpha_csc * CSC(s_t) + alpha_psu * PSU(s_t) + alpha_dsr * DSR(s_t)

When G(s_t) < damping_threshold the signal is damped toward the neutral 50%
level (paper Eq. 10):

    p^{cal}_t = p_t + (50 - p_t) * (1 - G(s_t))
"""

from __future__ import annotations

import numpy as np

from madfl.config import GateConfig

__all__ = ["cross_sectional_consensus", "predictive_stability",
           "deflated_sharpe", "Tier6aGate"]


def cross_sectional_consensus(signals: np.ndarray) -> float:
    """CSC: fraction of stocks agreeing with the cross-sectional median sign."""
    s = np.asarray(signals, dtype=float)
    if s.size == 0:
        return 0.5
    median = np.median(s)
    if abs(median) < 1e-12:
        return 0.5
    agree = np.mean(np.sign(s) == np.sign(median))
    return float(agree)


def predictive_stability(series: np.ndarray, window: int = 20,
                         eps: float = 1e-8) -> float:
    """PSU: 1 - std/range over a trailing window (paper Eq. 7)."""
    x = np.asarray(series, dtype=float)
    if x.size == 0:
        return 0.5
    tail = x[-window:]
    rng = np.ptp(tail)
    if rng < eps:
        return 1.0
    return float(1.0 - np.std(tail, ddof=1) / (rng + eps))


def deflated_sharpe(sharpe: float, n_obs: int, skewness: float,
                    kurtosis: float, expected_sharpe: float = 0.0) -> float:
    """DSR: probability-adjusted Sharpe corrected for non-normality and
    multiple testing (Bailey & Lopez de Prado, paper Eq. 8)."""
    if n_obs <= 1:
        return 0.5
    num = (sharpe - expected_sharpe) * np.sqrt(n_obs - 1)
    denom = np.sqrt(1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe ** 2)
    if denom <= 1e-12:
        return 0.5
    from scipy.stats import norm
    return float(norm.cdf(num / denom))


class Tier6aGate:
    """Online, per-fold signal gate with grid re-searched alpha weights."""

    def __init__(self, config: GateConfig | None = None):
        self.cfg = config or GateConfig()

    def _score(self, signals: np.ndarray, sharpe: float, n_obs: int,
               skewness: float, kurtosis: float) -> float:
        csc = cross_sectional_consensus(signals)
        psu = predictive_stability(signals, self.cfg.psu_window, self.cfg.psu_eps)
        dsr = deflated_sharpe(sharpe, n_obs, skewness, kurtosis)
        return (self.cfg.alpha_csc * csc
                + self.cfg.alpha_psu * psu
                + self.cfg.alpha_dsr * dsr)

    def gate(self, signals: np.ndarray, returns: np.ndarray) -> float:
        """Compute the composite gate score G(s_t) in [0, 1]."""
        r = np.asarray(returns, dtype=float)
        r = r[~np.isnan(r)]
        sr = float(np.mean(r) / (np.std(r) if np.std(r) > 1e-12 else 1.0))
        skew = float(__import__("scipy.stats", fromlist=["skew"]).skew(r)) if r.size > 2 else 0.0
        kurt = float(__import__("scipy.stats", fromlist=["kurtosis"]).kurtosis(r, fisher=True)) if r.size > 2 else 0.0
        return self._score(signals, sr, max(1, r.size), skew, kurt)

    def calibrate(self, p: float, g: float) -> float:
        """Damp a raw probability signal toward neutrality when G is low."""
        if g >= self.cfg.damping_threshold:
            return float(p)
        return float(p + (50.0 - p) * (1.0 - g))

    def __call__(self, signals: np.ndarray, returns: np.ndarray,
                 p: float) -> tuple[float, float]:
        """Convenience: returns (gate_score, calibrated_probability)."""
        g = self.gate(signals, returns)
        return g, self.calibrate(p, g)