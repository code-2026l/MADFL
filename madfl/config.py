"""Central configuration for the Memory Alpha framework.

All hyperparameters below mirror the values reported in the paper
(28-fold walk-forward validation on CSI 500, 2021--2024).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GateConfig:
    """Tier-6a signal gate (CSC + PSU + DSR)."""

    alpha_csc: float = 0.4
    alpha_psu: float = 0.4
    alpha_dsr: float = 0.2
    damping_threshold: float = 0.4  # flat for thresholds in [0.3, 0.5]
    psu_window: int = 20
    psu_eps: float = 1e-8


@dataclass
class RegimeConfig:
    """Sticky HMM + BOCPD regime detection."""

    n_states: int = 5                      # BULL_TREND, BULL, SIDEWAYS, HIGH_VOL, BEAR_CRASH
    sticky_beta: float = 3.0
    n_components: int = 2                  # GMM components per observation
    bocpd_hazard: float = 1.0 / 200.0
    bocpd_switch_prob: float = 0.5         # re-init threshold P(run=0) > 0.5
    soft_mix: float = 0.7                  # 70% current HMM score + 30% previous
    regime_scores: tuple = (+1.5, +0.8, 0.0, -0.5, -1.5)


@dataclass
class FactorConfig:
    """Multi-family factor synthesis."""

    n_families: int = 5
    pca_window: int = 60
    decay_lambda: float = 0.05
    min_icir_peak_age: int = 60
    neweywest_lags: int = 5


@dataclass
class PortfolioConfig:
    """V2 portfolio optimizer (BL + CVaR + risk budgets + drawdown breaker)."""

    lambda_bl: float = 0.3
    lambda_cvar: float = 0.4
    lambda_rb: float = 0.3
    cvar_alpha: float = 0.95
    risk_budget: str = "equal"             # equal risk contribution
    d_min: float = 0.12                    # drawdown floor (keep full exposure)
    d_max: float = 0.25                    # drawdown ceiling (reduce to zero)
    max_leverage: float = 1.0
    max_weight: float = 0.05
    risk_free: float = 0.0


@dataclass
class DistillationConfig:
    """Cross-generation knowledge distillation."""

    teacher_estimators: int = 500
    student_estimators: int = 100
    feature_pool_size: int = 44            # M in the paper
    online_learning_rate: float = 0.1
    replacement_icir_window: int = 20
    replacement_percentile: float = 0.25
    min_replacement_age: int = 60
    soft_target_weight: float = 0.5


@dataclass
class MemoryAlphaConfig:
    """Top-level configuration for the full pipeline."""

    gate: GateConfig = field(default_factory=GateConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    factors: FactorConfig = field(default_factory=FactorConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)

    # Walk-forward protocol
    train_days: int = 60
    test_days: int = 21
    n_folds: int = 28
    seed: int = 42
    top_k_features: int = 44
    # Consensus / debate
    debate_rounds: int = 3
    consensus_gate: float = 0.5
    # Data
    instrument_universe: str = "csi500"
    n_stocks: int = 200