"""Shared numerical utilities.

Implements the information-coefficient machinery, Newey-West
heteroskedasticity- and autocorrelation-consistent covariance estimation,
rolling statistics, and drawdown helpers used across the framework.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


# --------------------------------------------------------------------------
# Information coefficient (IC) and IC information ratio (ICIR)
# --------------------------------------------------------------------------
def rank_ic(signal: np.ndarray, forward_return: np.ndarray) -> float:
    """Cross-sectional Spearman IC between a signal and the forward return.

    Assumes 1-D arrays of equal length (one value per stock on a given day).
    Returns NaN when the input is degenerate.
    """
    s = np.asarray(signal, dtype=float)
    r = np.asarray(forward_return, dtype=float)
    if s.size < 2 or r.size < 2:
        return float("nan")
    if np.std(s) < EPS or np.std(r) < EPS:
        return float("nan")
    return float(np.corrcoef(_rank(s), _rank(r))[0, 1])


def mean_ic(ic_series: np.ndarray) -> float:
    """Mean IC over a walk-forward window (NaN-aware)."""
    ics = np.asarray(ic_series, dtype=float)
    ics = ics[~np.isnan(ics)]
    if ics.size == 0:
        return float("nan")
    return float(np.nanmean(ics))


def icir(ic_series: np.ndarray) -> float:
    """IC information ratio = mean(IC) / std(IC) over the window."""
    ics = np.asarray(ic_series, dtype=float)
    valid = ics[~np.isnan(ics)]
    if valid.size < 2:
        return float("nan")
    sd = np.std(valid, ddof=1)
    if sd < EPS:
        return float("nan")
    return float(np.mean(valid) / sd)


def _rank(x: np.ndarray) -> np.ndarray:
    """Rank with average ties, scaled to [0, 1]."""
    order = np.argsort(np.argsort(x))
    return order.astype(float) / max(1, len(x) - 1)


# --------------------------------------------------------------------------
# Newey-West covariance estimation
# --------------------------------------------------------------------------
def newey_west_cov(
    returns: np.ndarray, lags: int | None = None
) -> np.ndarray:
    """Newey-West HAC covariance of a (T, N) return matrix.

    The cross-sectional covariance is estimated with autocorrelation
    correction up to ``lags`` (default: floor(4 * (T/100)^(2/9))).
    """
    returns = np.asarray(returns, dtype=float)
    T, N = returns.shape
    if T < 2:
        return np.eye(N)
    if lags is None:
        lags = max(1, int(4 * (T / 100.0) ** (2.0 / 9.0)))
    demeaned = returns - returns.mean(axis=0, keepdims=True)
    gamma0 = demeaned.T @ demeaned / T
    cov = gamma0.copy()
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1)  # Bartlett kernel
        gamma_l = demeaned[l:].T @ demeaned[:-l] / T
        cov += w * (gamma_l + gamma_l.T)
    return cov


# --------------------------------------------------------------------------
# Rolling statistics
# --------------------------------------------------------------------------
def rolling_pca(
    X: np.ndarray, window: int, n_components: int = 1
) -> np.ndarray:
    """Rolling PCA on feature matrix (T, N).

    Returns the leading principal component scores of shape (T,) with the
    sign calibrated so that higher values predict higher forward returns is
    left to the caller.
    """
    X = np.asarray(X, dtype=float)
    T, N = X.shape
    scores = np.full(T, np.nan)
    for t in range(window - 1, T):
        block = X[t - window + 1: t + 1]
        block_c = block - block.mean(axis=0, keepdims=True)
        cov = (block_c.T @ block_c) / max(1, window - 1)
        # first eigenvector via power iteration (robust, no LAPACK dependency)
        v = np.random.default_rng(0).normal(size=N)
        v /= np.linalg.norm(v)
        for _ in range(20):
            nv = cov @ v
            nv /= np.linalg.norm(nv) + EPS
            v = nv
        scores[t] = block_c[-1] @ v
    return scores


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation with NaN padding at the head."""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    if x.size < window:
        return out
    for t in range(window - 1, x.size):
        out[t] = np.std(x[t - window + 1: t + 1], ddof=1)
    return out


def rolling_icir(signal: np.ndarray, forward_return: np.ndarray,
                 window: int) -> np.ndarray:
    """Rolling ICIR of a signal against forward returns over ``window`` days."""
    signal = np.asarray(signal, dtype=float)
    fwd = np.asarray(forward_return, dtype=float)
    n = min(signal.size, fwd.size)
    out = np.full(n, np.nan)
    for t in range(window, n):
        ics = [
            rank_ic(signal[max(0, i - window + 1): i + 1],
                    fwd[max(0, i - window + 1): i + 1])
            for i in range(t - window + 1, t)
        ]
        out[t] = icir(np.array(ics))
    return out


# --------------------------------------------------------------------------
# Drawdown and portfolio analytics
# --------------------------------------------------------------------------
def max_drawdown(cum_wealth: np.ndarray) -> float:
    """Maximum drawdown of a cumulative wealth series (positive fraction)."""
    cum = np.asarray(cum_wealth, dtype=float)
    if cum.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / (running_max + EPS)
    return float(np.min(dd))


def sharpe(returns: np.ndarray, risk_free: float = 0.0, periods: int = 252) -> float:
    """Annualized Sharpe ratio from a per-period return series."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size < 2 or np.std(r) < EPS:
        return float("nan")
    return float(np.sqrt(periods) * (np.mean(r) - risk_free / periods) / np.std(r, ddof=1))


def annualized_return(returns: np.ndarray, periods: int = 252) -> float:
    """Annualized compounding return."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return float("nan")
    return float(np.prod(1.0 + r) ** (periods / len(r)) - 1.0)


def turnover(weights: np.ndarray) -> float:
    """One-sided portfolio turnover across consecutive rebalances."""
    w = np.asarray(weights, dtype=float)
    if w.shape[0] < 2:
        return 0.0
    return float(np.mean(np.sum(np.abs(np.diff(w, axis=0)), axis=1)))


def to_wealth(returns: np.ndarray) -> np.ndarray:
    """Cumulative wealth series from per-period returns."""
    r = np.asarray(returns, dtype=float)
    r = np.nan_to_num(r, nan=0.0)
    return np.cumprod(1.0 + r)