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
        # IMPORTANT: route by the explicit stance= role, NOT by whether the word
        # "BEAR"/"BULL" appears anywhere in the prompt. The debate prompt embeds
        # the opponent's rebuttal text, which itself contains the stance marker
        # word; substring matching made the bull agent emit bear arguments and
        # collapsed the debate to bull == bear (e+ ~= e- ~ 1.0 => consensus ~0.5).
        if "stance=BEAR" in prompt:
            return _bear_arguments(prompt)
        if "stance=BULL" in prompt:
            return _bull_arguments(prompt)
        return _bull_arguments(prompt)


def _bull_arguments(prompt: str) -> str:
    ic = _extract(prompt, "IC")
    icir = _extract(prompt, "ICIR")
    to = _extract(prompt, "turnover")
    regime = _extract_str(prompt, "regime")
    # Re-scale IC/ICIR to their *realistic* magnitudes (raw values are O(1e-2)):
    #   wi = 8*IC   -> ic=0.02 gives +0.16;  ic=-0.01 gives -0.08
    #   wi2 = 2*ICIR -> icir=0.10 gives +0.20
    #   excess turnover (raw feature churn ~0.7-1.0) erodes the bull case
    wi = 8.0 * ic
    wi2 = 2.0 * icir
    wt = 0.6 * _clip01(to - 0.5) if to > 0 else 0.0
    z = 0.5 + wi + wi2 - wt
    if regime in ("BULL_TREND", "BULL"):
        z += 0.05
    score = max(0.05, min(0.98, z))
    return (f"[BULL] momentum persistence is high (IC={ic:.4f}); "
            f"ICIR={icir:.4f} indicates stability; churn={to:.2f}; "
            f"regime {regime} is favourable. score={score:.3f}")


def _bear_arguments(prompt: str) -> str:
    ic = _extract(prompt, "IC")
    icir = _extract(prompt, "ICIR")
    to = _extract(prompt, "turnover")
    regime = _extract_str(prompt, "regime")
    wi = 8.0 * ic
    wi2 = 2.0 * icir
    wt = 0.6 * _clip01(to - 0.5) if to > 0 else 0.0
    z = 0.5 - wi - wi2 + wt
    if regime in ("HIGH_VOL", "BEAR_CRASH"):
        z += 0.05
    score = max(0.05, min(0.98, z))
    return (f"[BEAR] crowding and mean-reversion risk; IC={ic:.4f} may decay; "
            f"churn={to:.2f}; regime {regime} is hostile. score={score:.3f}")


def _extract(prompt: str, key: str) -> float:
    import re
    m = re.search(rf"{key}=([-+]?\d*\.?\d+)", prompt)
    return float(m.group(1)) if m else 0.0


def _extract_str(prompt: str, key: str) -> str:
    import re
    m = re.search(rf"{key}=([A-Za-z_]+)", prompt)
    return m.group(1) if m else ""


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


class BaseAgent:
    """Base class for the specialized multi-agent roles."""

    def __init__(self, name: str, llm: LLMClient | None = None):
        self.name = name
        self.llm = llm or RuleBasedLLMClient()

    def speak(self, prompt: str) -> str:
        return self.llm.complete(prompt)