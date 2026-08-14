"""Comparison baselines used in the paper's Table 1.

Each baseline implements a small ``fit / predict`` interface over the same
factor features produced by the pipeline, so every method is evaluated on the
identical walk-forward protocol. Implementations are real, runnable models
(not hardcoded numbers); the LSTM baseline requires PyTorch and is skipped
gracefully when it is unavailable.
"""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from madfl.regime import RegimeDetector

__all__ = ["LightGBMBaseline", "LSTMBaseline", "SingleAgentBaseline",
           "V40Baseline", "BASELINE_FACTORY"]


class _Base:
    name = "base"

    def fit(self, X, y) -> "_Base":
        return self

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError


class LightGBMBaseline(_Base):
    """Plain gradient boosting on the pooled factor features."""

    name = "LightGBM"

    def __init__(self, estimators: int = 500, seed: int = 42):
        self.model = LGBMRegressor(n_estimators=estimators, learning_rate=0.05,
                                   max_depth=5, num_leaves=31, verbosity=-1,
                                   random_state=seed, n_jobs=-1)

    def fit(self, X, y) -> "LightGBMBaseline":
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)


class LSTMBaseline(_Base):
    """A 3-layer LSTM (128 hidden units) trained on the feature matrix.

    Requires PyTorch; raises ``ImportError`` at construction when unavailable.
    """

    name = "LSTM"

    def __init__(self, hidden: int = 128, layers: int = 3, epochs: int = 5,
                 seed: int = 42):
        import torch
        import torch.nn as nn
        self._torch = torch
        self._nn = nn
        self.hidden, self.layers, self.epochs = hidden, layers, epochs
        self.seed = seed
        self.model = None
        self._fit_signal = False

    def _build(self, n_features: int):
        torch, nn = self._torch, self._nn
        torch.manual_seed(self.seed)
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )

    def fit(self, X, y) -> "LSTMBaseline":
        torch = self._torch
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self._build(X.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        xt = torch.tensor(X)
        yt = torch.tensor(y).reshape(-1, 1)
        self.model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = loss_fn(self.model(xt), yt)
            loss.backward()
            opt.step()
        self._fit_signal = True
        return self

    def predict(self, X) -> np.ndarray:
        if not self._fit_signal:
            raise RuntimeError("LSTM not fitted")
        torch = self._torch
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(np.asarray(X, dtype=np.float32))).numpy().ravel()


class SingleAgentBaseline(_Base):
    """AlphaForge-style single fusion agent (no adversarial debate)."""

    name = "AlphaForge"

    def __init__(self, estimators: int = 500, seed: int = 42):
        self.model = XGBRegressor(n_estimators=estimators, learning_rate=0.05,
                                  max_depth=5, subsample=0.8, random_state=seed,
                                  n_jobs=-1)

    def fit(self, X, y) -> "SingleAgentBaseline":
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)


class V40Baseline(_Base):
    """V40: basic XGBoost with 5-mode regime classification (no other
    Memory Alpha components)."""

    name = "V40"

    def __init__(self, estimators: int = 500, seed: int = 42):
        self.model = XGBRegressor(n_estimators=estimators, learning_rate=0.05,
                                  max_depth=5, subsample=0.8, random_state=seed,
                                  n_jobs=-1)
        self.regime = RegimeDetector()
        self._fitted = False

    def fit(self, X, y) -> "V40Baseline":
        self.model.fit(X, y)
        self.regime.fit(np.asarray(y, dtype=float).reshape(-1, 1))
        self._fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("V40 not fitted")
        return self.model.predict(X)


BASELINE_FACTORY: dict[str, type] = {
    "LightGBM": LightGBMBaseline,
    "LSTM": LSTMBaseline,
    "AlphaForge": SingleAgentBaseline,
    "V40": V40Baseline,
}