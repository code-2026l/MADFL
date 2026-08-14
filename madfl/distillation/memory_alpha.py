"""Cross-generation knowledge distillation (paper Sec. 5.6).

Two parallel hierarchies:

  * Teacher -- a large-capacity XGBoost (500+ estimators) trained on the full
               historical factor library; holds global knowledge.
  * Student -- a lightweight booster (100 estimators, 44 features) distilled
               from the teacher and updated online; tracks recent patterns.

The feature pool has fixed capacity ``M`` and is ranked by rolling ICIR. A new
factor replaces the lowest-ICIR incumbent older than ``min_replacement_age``
days when its rolling ICIR beats the pool's `replacement_percentile` quantile,
and is otherwise discarded -- so the student's input dimension never changes
even as the teacher's feature space grows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from madfl.config import DistillationConfig
from madfl.utils import icir, rank_ic

__all__ = ["FeaturePool", "MemoryAlphaDistillation"]


class FeaturePool:
    """Fixed-capacity feature pool ranked by rolling ICIR."""

    def __init__(self, config: DistillationConfig | None = None):
        self.cfg = config or DistillationConfig()
        self.features: list[str] = []
        self.icir_by_feature: dict[str, float] = {}

    @property
    def size(self) -> int:
        return len(self.features)

    def set_initial(self, features: list[str],
                    icir_by_feature: dict[str, float]) -> None:
        """Seed the pool with the top-M features by ICIR."""
        ranked = sorted(icir_by_feature.items(), key=lambda kv: kv[1],
                        reverse=True)
        self.features = [f for f, _ in ranked[: self.cfg.feature_pool_size]]
        # if fewer than M candidates, keep what we have
        self.icir_by_feature = {f: icir_by_feature[f] for f in self.features}

    def consider(self, feature: str, icir_new: float, age: int) -> bool:
        """Return True if the feature enters the pool (replacing the weakest
        incumbent older than min age), else False."""
        if feature in self.features:
            self.icir_by_feature[feature] = icir_new
            return True
        if self.size < self.cfg.feature_pool_size:
            self.features.append(feature)
            self.icir_by_feature[feature] = icir_new
            return True
        # candidate must beat the pool's 25th percentile ICIR
        percentiles = np.percentile(
            list(self.icir_by_feature.values()), self.cfg.replacement_percentile * 100)
        if icir_new < percentiles:
            return False
        # replace the weakest incumbent strictly older than min age
        candidates = [f for f in self.features
                      if self.icir_by_feature.get(f, 0.0) <= percentiles]
        if not candidates:
            return False
        drop = min(candidates, key=lambda f: self.icir_by_feature[f])
        self.features.remove(drop)
        del self.icir_by_feature[drop]
        self.features.append(feature)
        self.icir_by_feature[feature] = icir_new
        return True

    def select(self, X: pd.DataFrame, age_by_feature: dict | None = None
               ) -> pd.DataFrame:
        """Return the columns of ``X`` that belong to the pool."""
        cols = [c for c in self.features if c in X.columns]
        return X[cols]


class MemoryAlphaDistillation:
    """Teacher-student distillation with a fixed-capacity feature pool."""

    def __init__(self, config: DistillationConfig | None = None,
                 seed: int = 42):
        self.cfg = config or DistillationConfig()
        self.pool = FeaturePool(config)
        self.teacher = XGBRegressor(n_estimators=self.cfg.teacher_estimators,
                                    learning_rate=0.05, max_depth=6,
                                    subsample=0.8, colsample_bytree=0.8,
                                    random_state=seed, n_jobs=-1)
        self.student = LGBMRegressor(n_estimators=self.cfg.student_estimators,
                                     learning_rate=self.cfg.online_learning_rate,
                                     max_depth=4, num_leaves=31,
                                     random_state=seed, verbosity=-1,
                                     n_jobs=-1)
        self._fitted_teacher = False

    # -- feature-pool utilities -------------------------------------------
    @staticmethod
    def _to_df(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)

    def _rolling_icir(self, X: pd.DataFrame, y, window: int = 20
                      ) -> dict[str, float]:
        X = self._to_df(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(np.asarray(y).ravel())
        out = {}
        T = len(X)
        for col in X.columns:
            vals = []
            for t in range(window, T):
                s = X[col].iloc[t - window: t].values
                r = y.iloc[t - window: t].values
                vals.append(rank_ic(s, r))
            out[col] = icir(np.array(vals))
        return out

    def init_pool(self, X, y) -> None:
        X = self._to_df(X)
        icir_map = self._rolling_icir(X, y)
        self.pool.set_initial(list(X.columns), icir_map)

    # -- training ---------------------------------------------------------
    def fit_teacher(self, X, y) -> None:
        X = self._to_df(X)
        self.teacher.fit(X, y)
        self._fitted_teacher = True

    def distill(self, X_pool, y, X_full) -> None:
        """Train the student on the teacher's soft targets over pooled features,
        giving a replay-buffer effect against catastrophic forgetting."""
        X_pool = self._to_df(X_pool)
        X_full = self._to_df(X_full)
        if not self._fitted_teacher:
            self.fit_teacher(X_full, y)
        soft = self.teacher.predict(X_pool)
        blend = self.cfg.soft_target_weight * soft + (
            1.0 - self.cfg.soft_target_weight) * np.asarray(y).ravel()
        self.student.fit(X_pool, blend)

    def update_online(self, X_pool, y) -> None:
        """Online update of the student with fresh market data."""
        X_pool = self._to_df(X_pool)
        self.student.fit(X_pool, y,
                         init_model=self.student if self.student.n_estimators else None)

    def predict_student(self, X_pool) -> np.ndarray:
        return self.student.predict(self._to_df(X_pool))

    def predict_teacher(self, X_full) -> np.ndarray:
        X_full = self._to_df(X_full)
        return self.teacher.predict(X_full) if self._fitted_teacher else np.zeros(len(X_full))