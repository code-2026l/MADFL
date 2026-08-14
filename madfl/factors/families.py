"""Multi-family factor synthesis.

Five established academic anomaly families (paper Table 2) are computed from
publicly available OHLCV data and synthesized into a single predictive score
through Newey-West orthogonalization, rolling PCA, and ICIR-aware decay.

Families:
  F35 anomaly composite   -- beta, size, 12m momentum, 1m reversal, idio. vol
  F36 mispricing          -- MGMT / PERF / FINAN / OTHER proxies
  F37 q-factor            -- investment, profitability, ROE proxies
  F38 accruals            -- total-accruals proxy
  F39 PEAD                -- earnings-surprise proxy

Accounting-based families are computed from price-volume proxies so the whole
pipeline runs on public OHLCV data; a user may supply their own fundamental
series by subclassing :class:`FactorFamily`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["FactorFamily", "AnomalyComposite", "Mispricing", "QFactor",
           "Accruals", "Pead", "FACTORY"]


class FactorFamily(ABC):
    """Base class for a factor family operating on a returns matrix."""

    name: str = "base"

    @abstractmethod
    def compute(self, returns: np.ndarray, volume: np.ndarray,
                market_returns: np.ndarray) -> np.ndarray:
        """Return per-stock factor values of shape (T, N)."""


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    for i in range(w - 1, x.shape[0]):
        out[i] = x[i - w + 1: i + 1].mean(axis=0)
    return out


class AnomalyComposite(FactorFamily):
    """Beta, size, 12m momentum, 1m reversal, and idiosyncratic volatility."""

    name = "F35_anomaly"

    def compute(self, returns, volume, market_returns):
        T, N = returns.shape
        beta = np.full_like(returns, np.nan)
        for i in range(N):
            r = returns[:, i]
            m = market_returns
            cov_ = np.cov(r, m)[0, 1]
            var_ = np.var(m)
            beta[:, i] = cov_ / var_ if var_ > 1e-12 else 0.0
        log_cap = np.nan_to_num(np.log(np.cumsum(volume, axis=0) + 1.0))
        mom12 = np.full_like(returns, np.nan)
        mom12[251:] = np.prod(1.0 + returns[1:251], axis=0) - 1.0
        rev1 = np.full_like(returns, np.nan)
        rev1[1:] = -returns[:-1]
        idio = np.full_like(returns, np.nan)
        for i in range(N):
            resid = returns[:, i] - beta[:, i] * market_returns
            idio[:, i] = _rolling_mean(resid ** 2, 60)
        score = (0.2 * np.nan_to_num(beta) + 0.2 * np.nan_to_num(log_cap)
                 + 0.2 * np.nan_to_num(mom12) + 0.2 * np.nan_to_num(rev1)
                 - 0.2 * np.nan_to_num(idio))
        return score


class Mispricing(FactorFamily):
    """Mispricing proxies (management / performance / financing / other)."""

    name = "F36_mispricing"

    def compute(self, returns, volume, market_returns):
        T, N = returns.shape
        # PERF proxy: short-term and medium-term return strength
        perf = np.full_like(returns, np.nan)
        perf[1:] = np.nan_to_num(returns[1:]) + np.nan_to_num(_rolling_mean(returns, 20)[1:])
        # MGMT / FINAN / OTHER: volume-flow proxies
        mgmt = np.nan_to_num(np.log(volume + 1.0) - np.log(volume + 1.0).mean(axis=0, keepdims=True))
        financ = np.full_like(returns, np.nan)
        financ[1:] = -_rolling_mean(np.nan_to_num(volume), 20)[1:]
        other = np.nan_to_num(returns)
        return 0.4 * perf + 0.2 * mgmt + 0.2 * financ + 0.2 * other


class QFactor(FactorFamily):
    """q-factor proxies: investment, profitability, and ROE.

    On pure OHLCV data we use cross-sectional proxies:
      * ROE          -- 60-day mean excess return over the market
      * profitability -- 20-day mean excess return
      * investment   -- change in volume trend (60d vs 20d) as a proxy for
                        asset growth / issuance
    All three are z-scored cross-sectionally before blending so the composite
    has genuine cross-sectional dispersion.
    """

    name = "F37_qfactor"

    @staticmethod
    def _cs_zscore(x: np.ndarray) -> np.ndarray:
        mu = np.nanmean(x, axis=1, keepdims=True)
        sd = np.nanstd(x, axis=1, keepdims=True)
        return (x - mu) / (sd + 1e-12)

    def compute(self, returns, volume, market_returns):
        T, N = returns.shape
        excess = np.nan_to_num(returns - market_returns[:, None])
        # ROE proxy: longer-horizon excess return (cross-sectionally persistent)
        roe = _rolling_mean(excess, 60)
        # profitability proxy: shorter-horizon excess return
        profit = _rolling_mean(excess, 20)
        # investment proxy: volume-trend change (asset-growth stand-in)
        vol_log = np.log(np.nan_to_num(volume) + 1.0)
        invest = _rolling_mean(vol_log, 60) - _rolling_mean(vol_log, 20)
        return (0.33 * self._cs_zscore(np.nan_to_num(roe))
                + 0.33 * self._cs_zscore(np.nan_to_num(profit))
                + 0.34 * self._cs_zscore(np.nan_to_num(invest)))


class Accruals(FactorFamily):
    """Total-accruals proxy from price-volume divergence.

    We use backward differences (strictly causal) so the signal uses only
    information available up to the current day; a naive ``np.gradient``
    would introduce forward-looking bias via its central-difference stencil.
    """

    name = "F38_accruals"

    def compute(self, returns, volume, market_returns):
        price_flow = np.nan_to_num(returns)
        volume_flow = np.nan_to_num(volume) / (np.nan_to_num(volume) + 1e-12)
        # strictly causal (backward) differences
        dp = np.zeros_like(price_flow)
        dv = np.zeros_like(volume_flow)
        dp[1:] = price_flow[1:] - price_flow[:-1]
        dv[1:] = volume_flow[1:] - volume_flow[:-1]
        accrual = dp - dv
        return accrual


class Pead(FactorFamily):
    """PEAD / earnings-surprise proxy from abnormal return drift."""

    name = "F39_pead"

    def compute(self, returns, volume, market_returns):
        T, N = returns.shape
        abn = np.nan_to_num(returns - market_returns[:, None])
        drift = np.full_like(returns, np.nan)
        drift[1:] = _rolling_mean(abn, 5)[:-1]
        return np.nan_to_num(drift)


FACTORY = {
    "F35_anomaly": AnomalyComposite,
    "F36_mispricing": Mispricing,
    "F37_qfactor": QFactor,
    "F38_accruals": Accruals,
    "F39_pead": Pead,
}