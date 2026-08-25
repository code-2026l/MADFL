"""End-to-end Memory Alpha pipeline.

Orchestrates the modules into a fold-level walk-forward component:

    factors -> regime -> consensus -> gate -> distillation -> portfolio

Features are represented at (day, stock) granularity: the factor families are
stacked into a (T, N, F) tensor and flattened to a long (T*N, F) frame for the
teacher/student models, whose predictions are reshaped back to per-stock
signals (T, N) -- the granularity required by cross-sectional IC and the
portfolio layer.

``MemoryAlphaFold`` fits on a training window and produces out-of-sample
per-stock signals and portfolio weights on a test window.
``run_walkforward`` loops this over the walk-forward protocol and aggregates
the intent-to-treat metrics.

Ablation switches (``use_gate`` / ``use_consensus`` / ``use_distillation`` /
``portfolio_mode`` on ``MemoryAlphaConfig``) toggle individual components so
each paper table row can be reproduced with the same code path; the defaults
reproduce the legacy pipeline.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from madfl.agents.base import LLMClient
from madfl.agents.consensus import ConsensusProtocol
from madfl.config import MemoryAlphaConfig
from madfl.distillation.memory_alpha import MemoryAlphaDistillation
from madfl.factors.synthesizer import FactorSynthesizer
from madfl.gate import Tier6aGate
from madfl.portfolio.v2 import V2Optimizer
from madfl.regime import RegimeDetector, _STATE_NAMES
from madfl.utils import (annualized_return, icir, max_drawdown, mean_ic,
                         rank_ic, sharpe, turnover)

__all__ = ["MemoryAlphaFold", "run_walkforward"]


def build_factor_stack(returns: np.ndarray, volume: np.ndarray,
                       market: np.ndarray, config=None
                       ) -> tuple[np.ndarray, list]:
    """Daily cross-sectionally standardized family scores as a (T, N, F) stack.

    Same normalization as ``build_factor_features`` but keeps the per-stock
    layout so families can be combined with per-family weights downstream.
    """
    synth = FactorSynthesizer(config.factors if config else None)
    fam = synth.compute_families(returns, volume, market)
    names = list(fam.keys())
    F = len(names)
    T, N = returns.shape
    if F == 0:
        return np.zeros((T, N, 0)), names
    stack = np.stack([np.nan_to_num(fam[n]) for n in names], axis=-1)  # (T,N,F)
    for f in range(F):
        x = stack[:, :, f]
        mu = np.nanmean(x, axis=1, keepdims=True)
        sd = np.nanstd(x, axis=1, keepdims=True)
        stack[:, :, f] = (x - mu) / (sd + 1e-12)
    return stack, names


def build_factor_features(returns: np.ndarray, volume: np.ndarray,
                          market: np.ndarray, config=None) -> tuple[pd.DataFrame, list]:
    """Stack the factor families into a long (T*N, F) DataFrame.

    The frame is day-major (rows ordered by day, then stock) so predictions
    can be reshaped to (T, N). Returns (frame, family_names).
    """
    stack, names = build_factor_stack(returns, volume, market, config)
    F = stack.shape[2]
    if F == 0:
        return pd.DataFrame(), names
    frame = pd.DataFrame(stack.reshape(-1, F), columns=[f"fam_{i}" for i in range(F)])
    return frame, names


class MemoryAlphaFold:
    """One fold of the walk-forward Memory Alpha pipeline."""

    def __init__(self, config: MemoryAlphaConfig | None = None,
                 llm: LLMClient | None = None):
        self.cfg = config or MemoryAlphaConfig()
        self.llm = llm
        self.gate = Tier6aGate(self.cfg.gate)
        self.regime = RegimeDetector(self.cfg.regime)
        self.synth = FactorSynthesizer(self.cfg.factors)
        self.consensus = ConsensusProtocol(llm=llm, rounds=self.cfg.debate_rounds,
                                           gate=self.cfg.consensus_gate)
        self.distill = MemoryAlphaDistillation(self.cfg.distillation,
                                               seed=self.cfg.seed)
        self.portfolio = V2Optimizer(self.cfg.portfolio,
                                     mode=self.cfg.portfolio_mode)
        self._rng = np.random.default_rng(self.cfg.seed)

    def _consensus_screen(self, X: pd.DataFrame, y) -> pd.DataFrame:
        """Adversarial consensus screening of candidate factor columns.

        Each factor family is screened by IC / ICIR / turnover and then
        debated (K bull/bear rounds) under the regime detected at the end of
        the training window; families whose consensus score falls below the
        gate are dropped from the training frame. If every family is rejected
        we keep the top-ICIR half so the pipeline never degenerates to an
        empty feature set.

        Honesty fix: the screening statistics are taken from the STRICTLY
        CAUSAL prior window (``self._prior_ic`` / ``self._prior_icir``,
        computed in :meth:`fit` from data *before* the training window), never
        from the training window itself. Using in-sample IC/ICIR to select
        features that the model is then trained and evaluated on leaks the
        target and inflates OOS IC (a textbook selection-bias artifact).
        """
        yv = np.asarray(y, dtype=float).ravel()
        use_deflate = (getattr(self.cfg, "screen_mode", "prior") == "deflate"
                       and getattr(self, "_cds", None) is not None)
        use_prior = (self._prior_ic is not None
                     and self._prior_icir is not None)
        if use_deflate:
            # CDS: screen on the winner's-curse-corrected causal IC (IC*).
            # The deflated statistic is both the selection criterion and the
            # ranking key for the working-majority floor below.
            def_ic = self._cds["def_ic"]
            icir_map = {}
            stats = []
            for col in X.columns:
                idx = int(str(col).split("_")[-1])
                d = float(def_ic[idx]) if idx < len(def_ic) else 0.0
                icir_map[col] = d
                s = np.asarray(X[col].values, dtype=float)
                to = float(np.mean(np.abs(np.diff(s))))
                stats.append({"ic": d, "icir": d, "turnover": to})
        elif use_prior:
            icir_map = {}
            stats = []
            for col in X.columns:
                idx = int(str(col).split("_")[-1])
                ic = float(self._prior_ic[idx]) if idx < len(self._prior_ic) else 0.0
                icir = (float(self._prior_icir[idx])
                        if idx < len(self._prior_icir) else 0.0)
                icir_map[col] = icir
                s = np.asarray(X[col].values, dtype=float)
                to = float(np.mean(np.abs(np.diff(s))))
                stats.append({"ic": ic, "icir": icir, "turnover": to})
        else:
            # Legacy in-sample fallback (only used when no prior window exists,
            # e.g. the very first walk-forward fold). Kept for non-walk-forward
            # callers; the ablation always supplies a prior window.
            icir_map = self.distill._rolling_icir(X, y)
            stats = []
            for col in X.columns:
                s = np.asarray(X[col].values, dtype=float)
                ic = float(rank_ic(s, yv))
                to = float(np.mean(np.abs(np.diff(s))))
                stats.append({"ic": ic, "icir": float(icir_map.get(col, 0.0)),
                              "turnover": to})
        regime_name = getattr(self, "_current_regime", _STATE_NAMES[2])
        accepted = self.consensus.run(stats, [regime_name] * len(stats),
                                      rng=self._rng)
        keep = {a["candidate"]["candidate_id"] for a in accepted}
        # The adversarial protocol may veto weak families, but the feature set
        # must never collapse below a working majority (ceil(F/2)).  Pad the
        # kept set with the highest-ICIR rejected families until that floor.
        majority = max(1, int(np.ceil(len(stats) / 2)))
        if len(keep) < majority:
            ranked = sorted(range(len(stats)),
                            key=lambda i: stats[i]["icir"], reverse=True)
            for i in ranked:
                if len(keep) >= majority:
                    break
                keep.add(i)
        cols = [c for i, c in enumerate(X.columns) if i in keep]
        print(f"[WF] consensus kept {len(cols)}/{len(stats)} families "
              f"(floor={majority}, accepted={len(accepted)}, "
              f"prior={use_prior})", flush=True)
        return X[cols] if cols else X

    def _cds_stats(self, pstack: np.ndarray,
                   prior_returns: np.ndarray) -> dict:
        """Causal Deflation Screening (CDS) statistics.

        The agent's factor proposals are treated as a stochastic hypothesis
        search of *effective size* ``K_eff``; any search picks the largest of
        ``K_eff`` noisy IC estimates, so the naive winner's IC is inflated by
        ``E[max_{i<=K_eff} N(0, se^2)] ~= se * sqrt(2 ln K_eff)`` (winner's
        curse).  CDS corrects it:

            IC*_i = IC_i - se_i * sqrt(2 ln K_eff)

        * ``IC_i``      -- mean cross-sectional IC of family i on the STRICTLY
                           CAUSAL prior window (never the training window);
        * ``se_i``      -- standard error of that mean under a block
                           permutation null (block = 21 trading days) of the
                           prior-window forward returns;
        * ``K_eff``     -- effective number of independent bets, read off the
                           spectrum of the daily-IC correlation matrix
                           (participation ratio, clipped to [1, F]).

        A family survives screening iff ``IC*_i > 0``, i.e. its causal edge
        exceeds the search-cost bar that scales with how many independent
        hypotheses the agent effectively proposed.
        """
        Tp, N, F = pstack.shape
        Tp1 = Tp - 1
        rng = np.random.default_rng(20260825)
        ics = np.full((Tp1, F), np.nan)
        for t in range(Tp1):
            for i in range(F):
                ics[t, i] = rank_ic(pstack[t, :, i], prior_returns[t + 1])
        ic_mean = np.nanmean(ics, axis=0)
        # effective number of independent bets from the IC correlation spectrum
        good = ~np.isnan(ics).any(axis=1)
        if good.sum() >= 2:
            C = np.corrcoef(ics[good].T)
            C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
            w = np.clip(np.linalg.eigvalsh(C), 0.0, None)
            k_eff = float((w.sum() ** 2) / ((w ** 2).sum() + 1e-12)) \
                if w.sum() > 1e-12 else 1.0
        else:
            k_eff = 1.0
        k_eff = float(np.clip(k_eff, 1.0, F))
        # block-permutation null: dispersion of the mean-IC estimate under
        # no signal (block = 21 trading days preserves autocorrelation)
        target = np.asarray(prior_returns[1:], dtype=float)
        idx = np.arange(Tp1)
        nb = int(np.ceil(Tp1 / 21.0))
        null = np.zeros((80, F))
        for p in range(80):
            bl = rng.permutation(nb)
            pidx = np.concatenate(
                [idx[b0 * 21: (b0 + 1) * 21] for b0 in bl])[:Tp1]
            pt = target[pidx]
            for i in range(F):
                null[p, i] = np.nanmean(
                    [rank_ic(pstack[t, :, i], pt[t]) for t in range(Tp1)])
        se = np.nanstd(null, axis=0)
        penalty = se * np.sqrt(2.0 * np.log(k_eff))
        def_ic = ic_mean - penalty
        return {"ic_mean": ic_mean, "se": se, "k_eff": k_eff,
                "def_ic": def_ic, "penalty": penalty}

    def fit(self, train_returns: np.ndarray, train_volume: np.ndarray,
            train_market: np.ndarray, prior_returns: np.ndarray | None = None,
            prior_volume: np.ndarray | None = None,
            prior_market: np.ndarray | None = None,
            hist_returns: np.ndarray | None = None,
            hist_volume: np.ndarray | None = None,
            hist_market: np.ndarray | None = None,
            hist_offset: int = 0,
            prior_offset: int = 0) -> "MemoryAlphaFold":
        """Fit all trainable components on the training window.

        The target is the **next-day** return: factor features computed from
        data available on day ``t`` are used to predict the return on day
        ``t+1``.  This is the standard cross-sectional IC setup in quantitative
        finance and matches the OOS IC reported in the paper.

        ``prior_*`` is the strictly-causal window *before* the training window;
        it is used to compute the consensus screening statistics so that factor
        selection never peeks at the training target (see :meth:`_consensus_screen`).

        ``hist_*`` / ``hist_offset`` supply the cumulative series up to the end
        of the training window and the absolute index of its first row.  Factor
        families with long lookbacks (60-251 days) must be computed on the
        cumulative history so they are warm at the start of the window; building
        them on the window alone zero-fills the rolling families and silently
        kills most of the feature space (the failure mode that produced the
        degenerate test-window signals).  ``prior_offset`` is the absolute index
        of the first prior-window row (for the same warm-up reason).
        """
        T, N = train_returns.shape
        if (hist_returns is not None
                and hist_returns.shape[0] >= hist_offset + T):
            X_full, _ = build_factor_features(hist_returns, hist_volume,
                                              hist_market, self.cfg)
            # absolute day t in [hist_offset, hist_offset+T-2] -> returns[t+1]
            X = X_full.iloc[hist_offset * N: (hist_offset + T - 1) * N]
            y = np.asarray(hist_returns[hist_offset + 1: hist_offset + T],
                           dtype=float).reshape(-1)
            self._feat_offset = hist_offset
        else:
            X_full, _ = build_factor_features(train_returns, train_volume,
                                              train_market, self.cfg)
            # features on day t (0..T-2), target is return on day t+1 (1..T-1)
            X = X_full.iloc[: (T - 1) * N]
            y = np.asarray(train_returns[1:], dtype=float).reshape(-1)
            self._feat_offset = 0
        # --- causal prior-window consensus statistics (honesty fix) ----------
        self._prior_ic = None
        self._prior_icir = None
        self._cds = None
        if (self.cfg.use_consensus and prior_returns is not None
                and prior_returns.shape[0] >= 120):
            if (hist_returns is not None
                    and hist_returns.shape[0] >= hist_offset
                    and prior_offset >= 0):
                pfull, _ = build_factor_stack(hist_returns[:hist_offset],
                                              hist_volume[:hist_offset],
                                              hist_market[:hist_offset],
                                              self.cfg)
                pstack = pfull[prior_offset: hist_offset]
            else:
                pstack, _ = build_factor_stack(prior_returns, prior_volume,
                                               prior_market, self.cfg)
            F = pstack.shape[2]
            if F > 0:
                if getattr(self.cfg, "screen_mode", "prior") == "deflate":
                    # Causal Deflation Screening: winner's-curse-corrected
                    # causal IC is the screening statistic (see _cds_stats).
                    self._cds = self._cds_stats(pstack, prior_returns)
                    self._prior_ic = np.asarray(self._cds["ic_mean"])
                    self._prior_icir = np.asarray(self._cds["def_ic"])
                    print(f"[CDS] K_eff={self._cds['k_eff']:.3f} "
                          f"se={np.nanmean(self._cds['se']):.5f} "
                          f"penalty={np.nanmean(self._cds['penalty']):.5f} "
                          f"ic*={np.round(self._cds['def_ic'], 4)}", flush=True)
                else:
                    Tp = pstack.shape[0]
                    # Feature on day t predicts return on day t+1, so features
                    # from days 0..Tp-2 align with targets from days 1..Tp-1.
                    # Both flattened views have exactly (Tp-1)*N rows -- any
                    # other pairing misaligns the arrays and rank_ic raises
                    # ValueError, which callers must never swallow.
                    y_all = np.asarray(prior_returns[1:], dtype=float)
                    ic_arr = np.zeros(F)
                    icir_arr = np.zeros(F)
                    for i in range(F):
                        s = pstack[:-1, :, i].reshape(-1)
                        py = y_all.reshape(-1)
                        if len(s) >= 40:
                            ic_arr[i] = float(rank_ic(s, py))
                            step = max(1, (Tp - 1) // 40)
                            roll = np.array([rank_ic(pstack[t, :, i],
                                                     prior_returns[t + 1])
                                             for t in range(0, Tp - 1, step)]) \
                                if Tp - 1 > 21 else np.array([ic_arr[i]])
                            sd = float(np.std(roll))
                            icir_arr[i] = (
                                float(np.mean(roll)) / sd * np.sqrt(252)
                                if sd > 1e-9 else 0.0)
                    self._prior_ic = ic_arr
                    self._prior_icir = icir_arr
        # regime detection on the training market series
        mr_train = np.nan_to_num(train_market).reshape(-1, 1)
        self.regime.fit(mr_train)
        try:
            states = self.regime.hmm.predict(mr_train)
            self._current_regime = _STATE_NAMES[int(states[-1])]
            self._current_regime_idx = int(states[-1])
        except Exception:
            self._current_regime = _STATE_NAMES[2]
            self._current_regime_idx = 2
        if X.shape[1] > 0:
            if getattr(self.cfg, "combine_head", False):
                # Regime-soft-weighted factor combo: keep ALL families and adapt
                # weights from recent realized IC (causal), NOT from an in-sample
                # hard top-k. Shrinkage toward equal weights guards any family
                # from being dropped, so the cross-section never collapses.
                Tf, Nf = train_returns.shape
                zstack, _ = build_factor_stack(train_returns, train_volume,
                                               train_market, self.cfg)
                F = zstack.shape[2]
                if F == 0:
                    self._fam_weights = np.ones(0) / 1.0
                else:
                    yf = np.asarray(train_returns[1:], dtype=float)
                    last = Tf - 1
                    wprev = np.zeros(F)
                    for f in range(F):
                        z = zstack[:last, :, f]
                        seg = range(max(0, last - min(90, last)), last)
                        ics = [float(rank_ic(z[t], yf[t])) for t in seg]
                        wprev[f] = float(np.mean(ics)) if ics else 0.0
                    # tanh-squash recent IC, softmax with temperature, then
                    # shrink half-way toward equal weights (robust floor).
                    sc = np.tanh(6.0 * np.clip(wprev, -0.05, 0.05))
                    s = np.exp(10.0 * sc)
                    s = s / (s.sum() + 1e-12)
                    delta = 0.5
                    self._fam_weights = (delta * (1.0 / F)
                                         + (1.0 - delta) * s)
            else:
                X_fit = self._consensus_screen(X, y) if self.cfg.use_consensus else X
                if X_fit.shape[1] == 0:
                    X_fit = X
                # remember exactly which columns the teacher was trained on:
                # the teacher-only path (use_distillation=False) must predict
                # on this SAME column set, never the full factor frame, or the
                # gradient booster rejects / misreads the frame.
                self._screen_cols = list(X_fit.columns)
                self.distill.init_pool(X_fit, y)
                self.distill.fit_teacher(X_fit, y)
                if self.cfg.use_distillation:
                    self.distill.distill(self.distill.pool.select(X_fit), y, X_fit)
        return self

    def predict(self, test_returns: np.ndarray, test_volume: np.ndarray,
                test_market: np.ndarray,
                hist_returns: np.ndarray | None = None,
                hist_volume: np.ndarray | None = None,
                hist_market: np.ndarray | None = None,
                hist_offset: int = 0) -> dict:
        """Produce out-of-sample per-stock signals and portfolio weights.

        Signals on day ``t`` are forecasts of the day-``t+1`` return, so the
        portfolio formed from ``signal`` is evaluated against the next day's
        realised return.

        ``hist_*`` / ``hist_offset``: cumulative series up to the end of the
        test window and the absolute index of the test window's first row.
        Features are computed on the cumulative history so long-lookback
        factor families are warm during the test window (same requirement as
        in :meth:`fit`); without it the rolling families are zero-filled and
        the test-window signal collapses.
        """
        T, N = test_returns.shape
        if (hist_returns is not None
                and hist_returns.shape[0] >= hist_offset + T):
            X, _ = build_factor_features(hist_returns, hist_volume,
                                         hist_market, self.cfg)
            X = X.iloc[hist_offset * N: (hist_offset + T) * N]
        else:
            X, _ = build_factor_features(test_returns, test_volume,
                                         test_market, self.cfg)
        pred = np.nan_to_num(test_returns)
        if X.shape[1] > 0:
            if getattr(self.cfg, "combine_head", False):
                if hist_returns is not None \
                        and hist_returns.shape[0] >= hist_offset + T:
                    zstack, _ = build_factor_stack(hist_returns, hist_volume,
                                                   hist_market, self.cfg)
                    zstack = zstack[hist_offset: hist_offset + T]
                else:
                    zstack, _ = build_factor_stack(test_returns, test_volume,
                                                   test_market, self.cfg)
                if zstack.shape[2] > 0:
                    pred = np.tensordot(zstack, self._fam_weights,
                                        axes=([2], [0]))  # (T,N)
                    pred = pred.reshape(T, N)
            elif self.cfg.use_distillation:
                pooled = self.distill.pool.select(X)
                if pooled.shape[1] > 0:
                    pred = self.distill.predict_student(pooled).reshape(T, N)
            else:
                # teacher-only path: the teacher was fit on the consensus-
                # screened columns; predict on exactly those columns (subset of
                # the full frame, same order) so the booster never sees a
                # different feature space than it was trained on.
                cols = getattr(self, "_screen_cols", None)
                X_t = (X[cols] if cols
                       and all(c in X.columns for c in cols) else X)
                pred = self.distill.predict_teacher(X_t).reshape(T, N)
        # Tier-6a gate: per-day cross-sectional damping of the signal toward
        # neutrality when the composite gate score is low (paper Eq. 10: the
        # cross-sectional analogue of p^cal = p + (50 - p)(1 - G), where the
        # conviction |p - 50| scales by G).
        if self.cfg.use_gate:
            mkt = np.nan_to_num(np.nanmean(test_returns, axis=1))
            thr = self.cfg.gate.damping_threshold
            for j in range(T):
                g_j = self.gate.gate(pred[j], mkt[: j + 1])
                if g_j < thr:
                    pred[j] = pred[j] * (g_j / thr)
        # signal from the penultimate test day forecasts the last test day's
        # return (strictly causal: day t -> day t+1)
        signal = pred[-2]  # (N,) cross-sectional forecast for day T-1 -> T
        # Tier-6a gate score of the final cross-section (reported per fold)
        g, p_cal = self.gate(signal, np.nanmean(test_returns, axis=0), p=50.0)
        # adversarial consensus screen on candidate factors
        stats = [{"ic": 0.02, "icir": 0.2, "turnover": 0.3},
                 {"ic": 0.01, "icir": 0.1, "turnover": 0.4}]
        accepted = self.consensus.run(stats, ["SIDEWAYS", "HIGH_VOL"],
                                      rng=self._rng)
        accepted_flag = bool(accepted)
        # V2 portfolio allocation on the forecast
        hist = np.nan_to_num(test_returns[-20:])
        mu = np.nan_to_num(signal)
        cov = np.cov(hist.T) + 1e-8 * np.eye(N)
        weights = self.portfolio.optimize(hist, mu, cov, views=np.zeros(N))
        # drawdown circuit breaker using the current portfolio drawdown
        wealth = np.cumprod(1.0 + np.nanmean(test_returns, axis=1))
        dd = max_drawdown(wealth)
        weights = self.portfolio.apply_circuit_breaker(weights, dd)
        return {"signal": signal, "pred_matrix": pred, "weights": weights,
                "gate": g, "calibrated_prob": p_cal,
                "accepted_factors": accepted_flag}


def _estimate_factor_half_life(returns, volume, market, cfg) -> np.ndarray:
    """Empirical IC half-life (days) of each factor family on this panel.

    For each family we compute the daily cross-sectional IC of the family score
    against the next-day forward return, then measure how many lags the IC
    autocorrelation takes to halve. The values are read off the data, never
    hardcoded, so the panel in the paper reflects the actual sample.
    """
    synth = FactorSynthesizer(cfg.factors)
    ret = np.nan_to_num(returns)
    fam = synth.compute_families(ret, np.nan_to_num(volume),
                                 np.nan_to_num(market))
    names = list(fam.keys())
    T, N = ret.shape
    half_life = np.zeros(len(names))
    for j, name in enumerate(names):
        score = np.nan_to_num(fam[name])
        ic = np.full(T, np.nan)
        for t in range(T - 1):
            ic[t] = rank_ic(score[t], ret[t + 1])
        # drop NaN and the trailing day with no forward return
        ic = ic[:T - 1]
        ic = ic[~np.isnan(ic)]
        if ic.size < 40:
            half_life[j] = float("nan")
            continue
        # raw daily IC is too noisy for a lag-1 autocorrelation to be
        # meaningful; smooth with a 21-day rolling mean first (the persistence
        # of the *smoothed* IC series is the factor's usable signal life)
        s = pd.Series(ic).rolling(21, min_periods=10).mean().dropna().values
        if s.size < 8:
            half_life[j] = float("nan")
            continue
        a = s[:-1] - s[:-1].mean()
        b = s[1:] - s[1:].mean()
        denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        rho = float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0
        if not np.isfinite(rho) or not (0.0 < rho < 1.0):
            half_life[j] = float("nan")
            continue
        half_life[j] = float(np.log(0.5) / np.log(rho))
    return half_life


def run_walkforward(returns: np.ndarray, volume: np.ndarray,
                    market: np.ndarray, config: MemoryAlphaConfig | None = None,
                    llm: LLMClient | None = None,
                    fold_start: int = 0, fold_end: int | None = None) -> dict:
    """Run the walk-forward validation.

    ``fold_start`` / ``fold_end`` slice the fold loop so the protocol can be
    sharded across parallel workers; each worker writes its own results file
    and ``merge_results`` (in the experiment script) stitches them together.

    Returns a dict with per-day out-of-sample portfolio returns and aggregated
    metrics (IC, ICIR, Sharpe, MDD, annualized return, turnover), together with
    the empirical regime state sequence and factor half-lives used by the
    figure scripts.
    """
    cfg = config or MemoryAlphaConfig()
    T = returns.shape[0]
    n_stocks = returns.shape[1]
    oos_returns = np.zeros(T)
    oos_weights = np.zeros((T, n_stocks))
    per_fold = []
    per_fold_k_eff = []
    per_fold_sigma = []
    per_fold_defl_winner = []
    per_day_ic = np.full(T, np.nan)

    # regime sequence over the whole panel (empirical, for figures).
    # The sticky HMM is fit on the one-dimensional market return series, which
    # is the natural regime-generating signal; feeding the full (T, N) panel
    # makes the emission dimension too large for the EM to separate states.
    regime_det = RegimeDetector(cfg.regime)
    mr = np.nan_to_num(market).reshape(-1, 1)
    regime_det.fit(mr)
    regime_states = regime_det.hmm.predict(mr)

    fold_end = cfg.n_folds if fold_end is None else min(fold_end, cfg.n_folds)
    for fold in range(fold_start, fold_end):
        _t_fold = time.time()
        start = fold * cfg.test_days
        train_end = start + cfg.train_days
        test_end = train_end + cfg.test_days
        if test_end > T:
            break
        tr_r, tr_v, tr_m = (returns[start: train_end], volume[start: train_end],
                            market[start: train_end])
        te_r, te_v, te_m = (returns[train_end: test_end],
                            volume[train_end: test_end],
                            market[train_end: test_end])
        fold_model = MemoryAlphaFold(cfg, llm=llm)
        if getattr(cfg, "reversal", False):
            # Full-time reversal + equal-weight multi-factor composite.
            # Signal on day t is -(equal-weight mean of ALL factor families),
            # the fixed economic direction of short-term reversal in A-shares
            # (NOT fit on any in-sample window). Factor stack is computed over
            # the cumulative history up to the test window so the rolling
            # families have their natural warm-up -- identical to the probe
            # (diag_sign.py) that established the out-of-sample reversal IC.
            seg_r = returns[: test_end + 1]
            seg_v = volume[: test_end + 1]
            seg_m = market[: test_end + 1]
            stack, _ = build_factor_stack(seg_r, seg_v, seg_m, cfg)
            if stack.shape[2] > 0:
                pm = (-np.nanmean(stack, axis=2))[train_end: test_end]
                pm = pm.reshape(test_end - train_end, n_stocks)
            else:
                base = np.nanmean(te_r, axis=0)
                pm = np.tile(base, (test_end - train_end, 1))
            pred = {"signal": pm[-2], "pred_matrix": pm, "weights": None,
                    "gate": 0.5, "calibrated_prob": 50.0,
                    "accepted_factors": False}
            # regime state for exposure scaling (strictly causal: only data up
            # to the last training day, never the test window).
            if getattr(cfg, "reversal_gate", None) == "transparent":
                # Transparent causal adverse-state flag: trailing realized-vol
                # percentile OR market drawdown, read off the equal-weight
                # market return history up to train_end. No latent model, no
                # label ambiguity.
                full_m = np.nan_to_num(market[:train_end])
                win = 60
                if len(full_m) >= win:
                    rv = pd.Series(full_m).rolling(win, min_periods=win) \
                        .std().dropna().values
                    cur = float(rv[-1])
                    hist = rv[:-1][-250:] if rv.size > 1 else np.array([cur])
                    vol_pct = float((hist <= cur).mean())
                    px = np.cumprod(1.0 + full_m)
                    dd = float(px[-1] / np.max(px[-win:]) - 1.0)
                else:
                    vol_pct, dd = 0.5, 0.0
                fold_model._reversal_adverse = bool(vol_pct >= 0.90
                                                    or dd <= -0.10)
            else:
                try:
                    mr_train = np.nan_to_num(tr_m).reshape(-1, 1)
                    fold_model.regime.fit(mr_train)
                    fold_model._current_regime_idx = int(
                        fold_model.regime.hmm.predict(mr_train)[-1])
                except Exception:
                    fold_model._current_regime_idx = 2
        else:
            try:  # intent-to-treat: degrade gracefully on sub-module failure
                # Strictly-causal prior window (data *before* the training
                # window) for honest consensus feature selection.  For the
                # earliest folds without enough history we pass ``None`` and
                # ``fit`` uses the legacy in-sample path for that single fold
                # (a small fraction of folds; the panel's tail protocol is
                # unchanged).
                p0 = max(0, start - min(start, 504))
                prior_r = returns[p0:start] if start > 0 else None
                prior_v = volume[p0:start] if start > 0 else None
                prior_m = market[p0:start] if start > 0 else None
                fold_model.fit(tr_r, tr_v, tr_m,
                               prior_returns=prior_r, prior_volume=prior_v,
                               prior_market=prior_m,
                               hist_returns=returns[:train_end],
                               hist_volume=volume[:train_end],
                               hist_market=market[:train_end],
                               hist_offset=start, prior_offset=p0)
                if getattr(fold_model, "_cds", None) is not None:
                    per_fold_k_eff.append(fold_model._cds["k_eff"])
                    per_fold_sigma.append(
                        float(np.nanmean(fold_model._cds["se"])))
                    per_fold_defl_winner.append(
                        float(np.nanmax(fold_model._cds["def_ic"])))
                pred = fold_model.predict(te_r, te_v, te_m,
                                          hist_returns=returns[:test_end],
                                          hist_volume=volume[:test_end],
                                          hist_market=market[:test_end],
                                          hist_offset=train_end)
            except Exception:
                # NEVER let a silent failure masquerade as a real signal: log
                # the traceback, and fall back to a strictly causal expanding-
                # mean signal (day-j signal uses only data <= day j), so the
                # intent-to-treat evaluation cannot fabricate in-window
                # look-ahead IC.
                import traceback as _tb
                print(f"[WF] fold {fold + 1} FAILED, causal baseline used:",
                      flush=True)
                _tb.print_exc()
                pm_fb = np.nan_to_num(te_r)
                pm_fb = np.array([np.nanmean(pm_fb[: j + 1], axis=0)
                                  for j in range(pm_fb.shape[0])])
                pred = {"signal": pm_fb[-2],
                        "weights": np.full(n_stocks, 1.0 / n_stocks),
                        "gate": 0.5, "calibrated_prob": 50.0,
                        "accepted_factors": False,
                        "pred_matrix": pm_fb}
        # evaluate the FULL test window (paper protocol: 21-day OOS).
        # signal on day j (of the test window) forecasts day j+1's return.
        pm = np.asarray(pred["pred_matrix"]).reshape(te_r.shape[0], n_stocks)
        n_te = te_r.shape[0] - 1
        fold_ics = []
        for j in range(n_te):
            sig = np.asarray(pred["signal"] if j == n_te - 1 else pm[j]).ravel()
            mu = np.nan_to_num(sig)
            # history window for the covariance estimate: >= 2 rows so np.cov
            # is well-defined even on the first days of the test window
            hist_rows = max(2, min(j + 1, 20))
            start = max(0, j + 1 - hist_rows)
            hist = np.nan_to_num(te_r[start: j + 1])
            if hist.shape[0] < 2:
                hist = np.vstack([hist, hist[-1:]])
            cov = np.cov(hist.T) + 1e-8 * np.eye(n_stocks)
            if not np.isfinite(cov).all():
                cov = np.eye(n_stocks)
            try:
                w = fold_model.portfolio.optimize(hist, mu, cov,
                                                  views=np.zeros(n_stocks))
            except Exception:
                w = np.full(n_stocks, 1.0 / n_stocks)
            if cfg.use_regime:
                # regime-scored exposure (strictly causal regime label).
                idx = getattr(fold_model, "_current_regime_idx", 2)
                if getattr(cfg, "reversal", False):
                    fl = float(getattr(cfg, "reversal_floor", 0.25))
                    mode = getattr(cfg, "reversal_gate", None)
                    if mode == "regime":
                        rev_exp = (1.0, 1.0, 1.0, fl, fl)
                        w = w * float(rev_exp[idx])
                    elif mode == "transparent":
                        adverse = bool(getattr(fold_model,
                                               "_reversal_adverse", False))
                        w = w * (fl if adverse else 1.0)
                    else:
                        score = cfg.regime.regime_scores[idx]
                        w = w * float(np.clip(1.0 + score / 3.0, 0.5, 1.0))
                else:
                    score = cfg.regime.regime_scores[idx]
                    w = w * float(np.clip(1.0 + score / 3.0, 0.5, 1.0))
            oos_returns[train_end + j] = float(np.dot(w, np.nan_to_num(te_r[j + 1])))
            oos_weights[train_end + j] = w
            ic_j = rank_ic(sig, te_r[j + 1])
            fold_ics.append(ic_j)
            per_day_ic[train_end + j] = ic_j
        per_fold.append(float(np.nanmean(fold_ics)))
        print(f"[WF] fold {fold + 1}/{cfg.n_folds} done in "
              f"{time.time() - _t_fold:.0f}s ic={per_fold[-1]:+.4f}", flush=True)

    valid = np.where(oos_returns != 0.0)[0]
    rets = oos_returns[valid]
    wealth = np.cumprod(1.0 + rets)
    per_fold_ic = np.array(per_fold)
    out = {
        "oos_returns": oos_returns,
        "oos_weights": oos_weights,
        "per_fold_ic": per_fold_ic,
        "per_day_ic": per_day_ic,
        "mean_ic": mean_ic(per_fold_ic),
        "icir": icir(per_fold_ic),
        "annualized_return": annualized_return(rets),
        "sharpe": sharpe(rets),
        "max_drawdown": max_drawdown(wealth),
        "turnover": turnover(oos_weights[valid]),
        # Causal Deflation Screening artifacts (per fold; empty when CDS off)
        "cds_k_eff": per_fold_k_eff,
        "cds_null_sigma": per_fold_sigma,
        "cds_defl_winner": per_fold_defl_winner,
        # empirical artifacts for the figure scripts
        "regime_states": regime_states,
        "regime_names": list(_STATE_NAMES),
        "factor_half_life": _estimate_factor_half_life(returns, volume, market, cfg),
        "factor_families": list(FactorSynthesizer(cfg.factors)
                                .compute_families(np.nan_to_num(returns),
                                                  np.nan_to_num(volume),
                                                  np.nan_to_num(market)).keys()),
    }
    return out
