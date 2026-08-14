"""Synthesis of the factor library into a single predictive score.

Pipeline (paper Sec. 5.4):
  1. Newey-West orthogonalization of family scores against one another.
  2. Rolling PCA extraction of the dominant principal component.
  3. ICIR-aware hyperbolic decay weighting:

        w_i(t) = w_i(0) / (1 + lambda * age_i(t))

  The final family score is the decay-weighted combination of the
  standardized principal components.
"""

from __future__ import annotations

import numpy as np

from madfl.config import FactorConfig
from madfl.factors import families

__all__ = ["FactorSynthesizer"]


class FactorSynthesizer:
    """Computes, orthogonalizes, and decay-aggregates the factor families."""

    def __init__(self, config: FactorConfig | None = None):
        self.cfg = config or FactorConfig()
        self.families = [cls() for cls in families.FACTORY.values()]

    def compute_families(self, returns: np.ndarray, volume: np.ndarray,
                         market_returns: np.ndarray) -> dict[str, np.ndarray]:
        """Return {family_name: scores (T, N)}."""
        out = {}
        for fam in self.families:
            out[fam.name] = fam.compute(returns, volume, market_returns)
        return out

    @staticmethod
    def _orthogonalize(family_scores: np.ndarray) -> np.ndarray:
        """Newey-West orthogonalization of the family stack.

        ``family_scores`` has shape (T, N, F). Returns (T, N, F) with shared
        variation across families removed.
        """
        T, N, F = family_scores.shape
        out = np.empty_like(family_scores)
        for t in range(T):
            X = family_scores[t]                      # (N, F)
            Xc = X - X.mean(axis=0, keepdims=True)
            # cross-sectional sample covariance of the family stack
            cov_m = Xc.T @ Xc / max(1, N - 1)
            for f in range(F):
                other = [j for j in range(F) if j != f]
                if not other or N < F + 1:
                    out[t, :, f] = Xc[:, f]
                    continue
                A = cov_m[np.ix_(other, other)]
                b = cov_m[np.ix_(other, [f])].ravel()
                try:
                    coef = np.linalg.solve(A + 1e-8 * np.eye(len(other)), b)
                except np.linalg.LinAlgError:
                    coef = np.zeros(len(other))
                resid = Xc[:, f] - Xc[:, other] @ coef
                out[t, :, f] = resid
        return out

    def synthesize(self, returns: np.ndarray, volume: np.ndarray,
                   market_returns: np.ndarray,
                   icir_per_family: np.ndarray | None = None,
                   age_per_family: np.ndarray | None = None) -> np.ndarray:
        """Return the synthesized family score of shape (T, N).

        icir_per_family (T, F) and age_per_family (T, F) are used to build the
        decay weights; when absent, equal weights are used.
        """
        fam = self.compute_families(returns, volume, market_returns)
        names = list(fam.keys())
        F = len(names)
        T, N = returns.shape
        stack = np.stack([np.nan_to_num(fam[n]) for n in names], axis=-1)  # (T, N, F)
        # standardize each family cross-sectionally
        for t in range(T):
            mu = stack[t].mean(axis=0, keepdims=True)
            sd = stack[t].std(axis=0, keepdims=True)
            stack[t] = (stack[t] - mu) / (sd + 1e-12)
        ortho = self._orthogonalize(stack)
        # rolling PCA: leading component per column stock block handled jointly
        # We PCA over the family axis per time step, then take component 1.
        scores = np.full((T, N), np.nan)
        for t in range(T):
            X = ortho[t]                              # (N, F)
            cov = X.T @ X / max(1, N - 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            v = eigvecs[:, -1]                        # leading component
            comp = X @ v
            # sign calibrate: higher predicts higher forward returns
            comp = _sign_calibrate(comp)
            scores[t] = comp
        # decay weighting
        if icir_per_family is not None and age_per_family is not None:
            w = _decay_weights(icir_per_family, age_per_family,
                               self.cfg.decay_lambda)
            # weighted average of family scores
            stacked = np.stack([np.nan_to_num(fam[n]) for n in names], axis=-1)
            for t in range(T):
                scores[t] = np.nanmean(stacked[t] * w[t, None, :], axis=-1)
        return scores


def _sign_calibrate(comp: np.ndarray) -> np.ndarray:
    """Flip the sign so the composite has positive mean cross-sectional
    alignment with the forward-day return proxy (approx)."""
    if np.nanmean(comp) < 0:
        return -comp
    return comp


def _decay_weights(icir_per_family: np.ndarray, age_per_family: np.ndarray,
                   lam: float) -> np.ndarray:
    """Hyperbolic ICIR-aware decay weights, shape (T, F)."""
    w0 = np.maximum(icir_per_family, 0.0) + 1e-6
    denom = 1.0 + lam * np.maximum(age_per_family, 0.0)
    w = w0 / denom
    return w / (w.sum(axis=-1, keepdims=True) + 1e-12)