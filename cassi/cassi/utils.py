"""Utility helpers shared across CASSI modules."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("cassi")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class Token:
    """A single token with its probability and any metadata."""

    id: int
    text: str = ""
    logprob: float = 0.0
    accepted: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Container for the output of :meth:`CassiEngine.generate`."""

    text: str
    tokens: list[Token]
    tokens_generated: int
    draft_tokens_proposed: int
    draft_tokens_accepted: int
    forward_passes: int
    elapsed_sec: float
    acceptance_rate: float
    final_k: int

    @property
    def speedup_vs_naive(self) -> float:
        """Naive autoregressive baseline takes ``tokens_generated`` forward passes.

        We measure speedup as ``baseline_passes / actual_passes``.
        """
        if self.forward_passes == 0:
            return 0.0
        return self.tokens_generated / max(self.forward_passes, 1)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
class Timer:
    """A tiny context-manager style timer used in benchmarks and engine."""

    def __init__(self) -> None:
        self._t0: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._t0


# ---------------------------------------------------------------------------
# Stats primitives
# ---------------------------------------------------------------------------
class EMA:
    """Exponential moving average used by the adaptive-k controller."""

    def __init__(self, alpha: float = 0.2) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.value: float | None = None

    def update(self, sample: float) -> float:
        if self.value is None:
            self.value = float(sample)
        else:
            self.value = (1.0 - self.alpha) * self.value + self.alpha * float(sample)
        return self.value


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def safe_div(num: float, denom: float, default: float = 0.0) -> float:
    """Safe division that returns ``default`` when denominator is zero."""
    return num / denom if denom else default


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotent logging setup for the CASSI logger."""
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[cassi] %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)


__all__ = [
    "Token",
    "GenerationResult",
    "Timer",
    "EMA",
    "safe_div",
    "configure_logging",
    "logger",
]
