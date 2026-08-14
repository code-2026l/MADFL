r"""Sticky HMM regime detection with Bayesian Online Change Point Detection.

The sticky HMM (Fox et al., 2011) uses a self-transition prior that suppresses
the spurious state flicker of standard HMMs:

    P_{ij} \propto exp(beta * delta_ij + theta_ij)

BOCPD (Adams & MacKay, 2007) runs alongside and re-initialises the HMM with a
flat prior when the posterior probability of a fresh segment exceeds a
threshold. Regime scores are then mapped to a continuous signal and smoothed
with a soft mixture (70% current HMM score + 30% previous day's score).
"""

from __future__ import annotations

import numpy as np

from madfl.config import RegimeConfig

__all__ = ["StickyHMM", "BayesianOnlineChangepoint", "RegimeDetector"]

_STATE_NAMES = ("BULL_TREND", "BULL", "SIDEWAYS", "HIGH_VOL", "BEAR_CRASH")


class StickyHMM:
    """Gaussian-emission HMM with a sticky self-transition prior."""

    def __init__(self, n_states: int = 5, beta: float = 3.0,
                 n_gauss: int = 2, seed: int = 42):
        self.K = n_states
        self.beta = beta
        self.n_gauss = n_gauss
        self.rng = np.random.default_rng(seed)
        self.pi: np.ndarray | None = None
        self.A: np.ndarray | None = None
        self.means: np.ndarray | None = None
        self.covs: np.ndarray | None = None

    def _init_params(self, X: np.ndarray, D: int) -> None:
        self.pi = np.full(self.K, 1.0 / self.K)
        # sticky prior makes self-transitions more likely
        base = np.full((self.K, self.K), 0.05 / (self.K - 1))
        np.fill_diagonal(base, self.beta)
        m = base.max()
        self.A = np.exp(base - m)
        self.A /= self.A.sum(axis=1, keepdims=True)
        self.means = self.rng.normal(size=(self.K, D))
        self.covs = np.tile(np.eye(D), (self.K, 1, 1))

    def _emission_logp(self, X: np.ndarray) -> np.ndarray:
        """Log emission probabilities, shape (T, K)."""
        T = X.shape[0]
        logp = np.zeros((T, self.K))
        for k in range(self.K):
            d = X - self.means[k]
            cov = self.covs[k] + 1e-6 * np.eye(X.shape[1])
            sign, logdet = np.linalg.slogdet(cov)
            inv = np.linalg.inv(cov)
            logp[:, k] = -0.5 * (logdet
                                 + np.einsum("ti,ij,tj->t", d, inv, d)
                                 + X.shape[1] * np.log(2 * np.pi))
        return logp

    @staticmethod
    def _forward_backward(logp: np.ndarray, A: np.ndarray, pi: np.ndarray):
        T, K = logp.shape
        logA = np.log(A + 1e-300)
        logpi = np.log(pi + 1e-300)
        alpha = np.zeros((T, K))
        alpha[0] = logpi + logp[0]
        for t in range(1, T):
            alpha[t] = logp[t] + np.logaddexp.reduce(logA.T + alpha[t - 1, :, None], axis=0)
        loglik = np.logaddexp.reduce(alpha[-1])
        # backward / gamma
        beta = np.zeros((T, K))
        beta[-1] = 0.0
        for t in range(T - 2, -1, -1):
            beta[t] = np.logaddexp.reduce(logA + (logp[t + 1] + beta[t + 1])[None, :], axis=1)
        gamma = np.exp(alpha + beta - loglik)
        gamma /= gamma.sum(axis=1, keepdims=True)
        # xi (transition posterior) via forward recurrence
        xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            m = alpha[t, :, None] + logA + (logp[t + 1] + beta[t + 1])[None, :] - loglik
            xi[t] = np.exp(m - np.logaddexp.reduce(m))
        return loglik, gamma, xi

    def fit(self, X: np.ndarray, max_iter: int = 50, tol: float = 1e-4) -> "StickyHMM":
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        T, D = X.shape
        n_init = max(1, min(self.K, T))
        self._init_params(X, D)
        prev = -np.inf
        for it in range(max_iter):
            logp = self._emission_logp(X)
            loglik, gamma, xi = self._forward_backward(logp, self.A, self.pi)
            if abs(loglik - prev) < tol:
                break
            prev = loglik
            Nk = gamma.sum(axis=0) + 1e-12
            self.pi = Nk / Nk.sum()
            self.A = (xi.sum(axis=0) + 1e-12)
            self.A /= self.A.sum(axis=1, keepdims=True)
            for k in range(self.K):
                w = gamma[:, k][:, None]
                self.means[k] = (w * X).sum(axis=0) / (w.sum() + 1e-12)
                d = X - self.means[k]
                self.covs[k] = (w * d).T @ (w * d) / (w.sum() + 1e-12)
                self.covs[k] += 1e-6 * np.eye(D)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi-style maximum-likelihood state sequence."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        logp = self._emission_logp(X)
        T, K = logp.shape
        logA = np.log(self.A + 1e-300)
        delta = np.zeros((T, K))
        psi = np.zeros((T, K), dtype=int)
        delta[0] = np.log(self.pi + 1e-300) + logp[0]
        for t in range(1, T):
            for k in range(K):
                cand = delta[t - 1] + logA[:, k]
                psi[t, k] = int(np.argmax(cand))
                delta[t, k] = cand[psi[t, k]] + logp[t, k]
        states = np.zeros(T, dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states


class BayesianOnlineChangepoint:
    """Adams-MacKay online changepoint detection (Gaussian observation model)."""

    def __init__(self, hazard: float = 1.0 / 200.0, mu0: float = 0.0,
                 kappa0: float = 1.0, alpha0: float = 1.0, beta0: float = 1.0):
        self.hazard = hazard
        self.mu0, self.kappa0, self.alpha0, self.beta0 = mu0, kappa0, alpha0, beta0

    def _predictive_logp(self, x: float, mu: float, kappa: float,
                         alpha: float, beta: float) -> float:
        """Student-t predictive density log-probability."""
        from scipy.stats import t as student_t
        df = 2.0 * alpha
        scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
        return float(student_t.logpdf(x, df=df, loc=mu, scale=scale))

    def run(self, data: np.ndarray) -> np.ndarray:
        """Return the run-length posterior P(run=0 | x_1:t) at each step.

        Implements the Adams-MacKay online algorithm; the returned array is
        the probability that a changepoint occurred at each observation.
        """
        data = np.asarray(data, dtype=float)
        T = data.size
        run_probs = np.zeros(T)
        # run-length distribution R[r] = P(run length = r)
        dist = np.array([1.0])
        mu = np.array([self.mu0])
        kappa = np.array([self.kappa0])
        alpha = np.array([self.alpha0])
        beta = np.array([self.beta0])
        for t in range(T):
            x = data[t]
            predprobs = np.array(
                [self._predictive_logp(x, mu[i], kappa[i], alpha[i], beta[i])
                 for i in range(len(mu))])
            # growth component: extend each existing run by one observation
            growth = np.exp(predprobs + np.log(dist + 1e-300))
            # changepoint component: start a fresh run at length 0
            cp = self.hazard * np.sum(np.exp(predprobs) * dist)
            new = np.zeros(len(growth) + 1)
            new[0] = cp
            new[1:] = (1.0 - self.hazard) * growth
            s = new.sum()
            new = new / s if s > 1e-300 else new
            run_probs[t] = float(new[0])
            # update conjugate sufficient statistics for the grown runs
            new_mu = np.concatenate([[(self.kappa0 * self.mu0 + x) / (self.kappa0 + 1.0)],
                                     (kappa * mu + x) / (kappa + 1.0)])
            new_kappa = np.concatenate([[self.kappa0 + 1.0], kappa + 1.0])
            new_alpha = np.concatenate([[self.alpha0 + 0.5], alpha + 0.5])
            new_beta = np.concatenate([
                [self.beta0 + 0.5 * self.kappa0 / (self.kappa0 + 1.0)
                 * (x - self.mu0) ** 2],
                beta + 0.5 * kappa / (kappa + 1.0) * (x - mu) ** 2])
            dist = new
            mu = new_mu
            kappa = new_kappa
            alpha = new_alpha
            beta = new_beta
            # prune low-probability runs to keep the representation bounded
            keep = dist > 1e-12
            if keep.sum() < 1:
                keep = np.ones_like(keep, dtype=bool)
            if keep.sum() < dist.size:
                dist = dist[keep]
                mu = mu[keep]
                kappa = kappa[keep]
                alpha = alpha[keep]
                beta = beta[keep]
                dist /= dist.sum()
        return run_probs


class RegimeDetector:
    """Sticky HMM + BOCPD integrated detector with soft regime scoring."""

    def __init__(self, config: RegimeConfig | None = None):
        self.cfg = config or RegimeConfig()
        self.hmm = StickyHMM(n_states=self.cfg.n_states,
                             beta=self.cfg.sticky_beta,
                             seed=42)
        self.bocpd = BayesianOnlineChangepoint(hazard=self.cfg.bocpd_hazard)
        self._prev_score: float | None = None

    @staticmethod
    def _score_for_state(state: int, scores: tuple) -> float:
        idx = int(np.clip(state, 0, len(scores) - 1))
        return float(scores[idx])

    def fit(self, features: np.ndarray) -> "RegimeDetector":
        """Fit the sticky HMM on the feature matrix (T, D)."""
        self.hmm.fit(features)
        return self

    def detect(self, features: np.ndarray, observations: np.ndarray) -> np.ndarray:
        """Return continuous regime scores for each timestep.

        BOCPD re-initialisation: when P(run=0) > threshold at a step, the HMM
        is re-fit with a flat prior on the window since the last changepoint.
        """
        features = np.asarray(features, dtype=float)
        obs = np.asarray(observations, dtype=float)
        T = features.shape[0]
        raw_scores = np.zeros(T)
        run_probs = self.bocpd.run(obs)
        seg_start = 0
        for t in range(T):
            if run_probs[t] > self.cfg.bocpd_switch_prob and t > seg_start:
                # re-initialise HMM on the fresh segment
                self.hmm = StickyHMM(n_states=self.cfg.n_states,
                                     beta=self.cfg.sticky_beta, seed=42)
                self.hmm.fit(features[seg_start: t + 1])
                seg_start = t
            states = self.hmm.predict(features[seg_start: t + 1])
            state = int(states[-1])
            raw_scores[t] = self._score_for_state(state, self.cfg.regime_scores)
        # soft mixture smoothing
        smooth = np.zeros(T)
        for t in range(T):
            if t == 0 or self._prev_score is None:
                smooth[t] = raw_scores[t]
            else:
                smooth[t] = (self.cfg.soft_mix * raw_scores[t]
                             + (1.0 - self.cfg.soft_mix) * self._prev_score)
            self._prev_score = smooth[t]
        return smooth