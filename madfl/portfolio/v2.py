"""V2 portfolio optimizer.

Folds Black-Litterman views, CVaR, and risk budgeting into one scalarized
objective (paper Eq. 13), each term repairing a distinct weakness of plain
mean-variance, plus a drawdown circuit breaker (paper Eq. 14):

    min_w  lambda1 * BL(w) + lambda2 * CVaR_alpha(w) + lambda3 * RB(w)

    w^{adj}_t = w_t * min(1, (D_max - D_t) / (D_max - D_min))
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from madfl.config import PortfolioConfig

__all__ = ["black_litterman", "cvar", "risk_budget_deviation",
           "drawdown_circuit_breaker", "V2Optimizer"]


def black_litterman(expected_returns: np.ndarray, cov: np.ndarray,
                    views: np.ndarray, view_cov: np.ndarray | None = None,
                    tau: float = 0.05) -> np.ndarray:
    """Black-Litterman posterior expected returns (Idzorek 2005)."""
    n = expected_returns.size
    inv_tau_cov = np.linalg.inv(tau * cov + 1e-10 * np.eye(n))
    if views is None or views.size == 0:
        return expected_returns
    P = np.eye(n)  # views expressed directly on each asset's expected return
    Omega = view_cov if view_cov is not None else np.diag(np.full(n, 0.01))
    A = inv_tau_cov + P.T @ np.linalg.inv(Omega) @ P
    b = inv_tau_cov @ expected_returns + P.T @ np.linalg.inv(Omega) @ views
    return np.linalg.solve(A, b)


def cvar(returns: np.ndarray, weights: np.ndarray, alpha: float = 0.95) -> float:
    """Conditional value-at-risk of a portfolio return sample."""
    r = np.asarray(returns, dtype=float)
    if r.ndim == 1:
        r = r[:, None]
    port = r @ weights
    var = np.quantile(port, 1.0 - alpha)
    tail = port[port <= var]
    if tail.size == 0:
        return float(-var)
    return float(-np.mean(tail))


def risk_budget_deviation(weights: np.ndarray, cov: np.ndarray,
                          budget: np.ndarray | None = None) -> float:
    """Quadratic deviation from target risk contributions (Maillard et al.)."""
    w = np.asarray(weights, dtype=float)
    n = w.size
    if budget is None:
        budget = np.full(n, 1.0 / n)
    port_var = w @ cov @ w
    if port_var <= 1e-12:
        return 0.0
    mrc = (cov @ w) / np.sqrt(port_var)
    rc = w * mrc
    contrib = rc / (rc.sum() + 1e-12)
    return float(np.sum((contrib - budget) ** 2))


def drawdown_circuit_breaker(weights: np.ndarray, current_drawdown: float,
                             d_min: float = 0.12, d_max: float = 0.25
                             ) -> np.ndarray:
    """Scale exposure by the remaining drawdown headroom."""
    if current_drawdown <= d_min:
        scale = 1.0
    elif current_drawdown >= d_max:
        scale = 0.0
    else:
        scale = (d_max - current_drawdown) / (d_max - d_min)
    return weights * scale


class V2Optimizer:
    """Scalarized V2 optimizer with a drawdown circuit breaker."""

    def __init__(self, config: PortfolioConfig | None = None, mode: str = "v2"):
        # `mode` accepted for compatibility with the staged pipeline's
        # V2Optimizer(..., mode=...) call; the local optimizer implements the
        # full scalarized V2 (BL + CVaR + risk-budget) objective regardless.
        self.cfg = config or PortfolioConfig()
        self.mode = mode

    def _objective(self, w: np.ndarray, returns: np.ndarray, mu: np.ndarray,
                   cov: np.ndarray, views: np.ndarray) -> float:
        bl = float(np.sum((w * mu - views) ** 2))
        cv = cvar(returns, w, self.cfg.cvar_alpha)
        rb = risk_budget_deviation(w, cov)
        return (self.cfg.lambda_bl * bl
                + self.cfg.lambda_cvar * cv
                + self.cfg.lambda_rb * rb)

    def optimize(self, returns: np.ndarray, expected_returns: np.ndarray,
                 cov: np.ndarray, views: np.ndarray | None = None
                 ) -> np.ndarray:
        """Return optimal portfolio weights (sum to 1, long-only).

        ``returns`` is the (T, N) historical return matrix used for CVaR and
        risk budgeting; ``expected_returns`` is the (N,) forecast vector.
        """
        n = returns.shape[1]
        if views is None:
            views = np.zeros(n)
        mu = black_litterman(expected_returns, cov, views)
        x0 = np.full(n, 1.0 / n)
        bounds = [(0.0, min(self.cfg.max_weight, 1.0))] * n
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        res = minimize(self._objective, x0, args=(returns, mu, cov, views),
                       method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-6})
        w = np.clip(res.x, 0.0, None)
        s = w.sum()
        if s > 1e-12:
            w = w / s
        return w

    def apply_circuit_breaker(self, weights: np.ndarray,
                              current_drawdown: float) -> np.ndarray:
        return drawdown_circuit_breaker(weights, current_drawdown,
                                        self.cfg.d_min, self.cfg.d_max)