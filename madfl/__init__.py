"""Memory Alpha (MADFL).

A self-evolving, multi-agent framework for alpha factor mining that manages
factors across their full lifecycle -- discovery, validation, calibration,
deployment, and cross-generation distillation.

Two core orchestration protocols:
  * Adversarial multi-agent consensus  (agents/consensus.py)
  * Cross-generation knowledge distillation (distillation/memory_alpha.py)

Reference:
  "Memory Alpha: Taming the Discover-and-Forget Loop with Multi-Agent
   Consensus and Distillation" (WSDM 2027).
"""

from madfl.config import MemoryAlphaConfig, GateConfig, RegimeConfig, \
    FactorConfig, PortfolioConfig, DistillationConfig

__version__ = "1.0.0"
__all__ = [
    "MemoryAlphaConfig",
    "GateConfig",
    "RegimeConfig",
    "FactorConfig",
    "PortfolioConfig",
    "DistillationConfig",
    "__version__",
]