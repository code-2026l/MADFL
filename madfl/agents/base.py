"""Base agent and pluggable LLM client.

The framework ships a deterministic rule-based fallback so the whole pipeline
runs without a proprietary LLM. To use a frontier reasoning model, subclass
:class:`LLMClient` and install it via ``ConsensusProtocol(llm=...)``.
"""

from __future__ import annotations

import abc

__all__ = ["LLMClient", "RuleBasedLLMClient", "BaseAgent"]


class LLMClient(abc.ABC):
    """Interface for the reasoning model used in the adversarial debate."""

    @abc.abstractmethod
    def complete(self, prompt: str) -> str:
        """Return the model's free-text response to ``prompt``."""


class RuleBasedLLMClient(LLMClient):
    """Deterministic fallback that scores factors from statistics + regime.

    This is the no-LLM configuration studied in the paper's debate ablation
    (Table 6). It produces structured bull/bear arguments and numeric scores
    without any external API.
    """

    def complete(self, prompt: str) -> str:
        if "BEAR" in prompt:
            return _bear_arguments(prompt)
        return _bull_arguments(prompt)


def _bull_arguments(prompt: str) -> str:
    ic = _extract(prompt, "IC")
    icir = _extract(prompt, "ICIR")
    regime = _extract(prompt, "regime")
    score = 0.5 + 0.25 * _clip01(ic) + 0.15 * _clip01(icir)
    if regime in ("BULL_TREND", "BULL"):
        score = min(1.0, score + 0.1)
    return (f"[BULL] momentum persistence is high (IC={ic:.3f}); "
            f"ICIR={icir:.3f} indicates stability; regime {regime} is favourable. "
            f"score={score:.3f}")


def _bear_arguments(prompt: str) -> str:
    ic = _extract(prompt, "IC")
    icir = _extract(prompt, "ICIR")
    regime = _extract(prompt, "regime")
    decay = 1.0 - _clip01(ic)
    score = 0.3 + 0.3 * decay + 0.2 * (1.0 - _clip01(icir))
    if regime in ("HIGH_VOL", "BEAR_CRASH"):
        score = min(1.0, score + 0.15)
    return (f"[BEAR] crowding and mean-reversion risk; IC={ic:.3f} may decay; "
            f"regime {regime} is hostile. score={score:.3f}")


def _extract(prompt: str, key: str) -> float:
    import re
    m = re.search(rf"{key}=([-+]?\d*\.?\d+)", prompt)
    return float(m.group(1)) if m else 0.0


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


class BaseAgent:
    """Base class for the specialized multi-agent roles."""

    def __init__(self, name: str, llm: LLMClient | None = None):
        self.name = name
        self.llm = llm or RuleBasedLLMClient()

    def speak(self, prompt: str) -> str:
        return self.llm.complete(prompt)