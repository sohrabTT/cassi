"""Abstract backend interface for CASSI.

A backend exposes the minimal surface required by the engine:

1. ``draft_next``   - propose a candidate next token from the *draft* model.
2. ``target_score`` - score a candidate token sequence from the *target*
   model in a single batched forward pass (this is where speculative
   decoding wins).

Subclasses are free to wrap any runtime they want (PyTorch, llama.cpp,
MLC-LLM, a remote API, ...). The engine never touches the runtime directly,
which keeps CASSI truly backend-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from cassi.utils import Token


@dataclass
class DraftProposal:
    """A draft proposal: token id, decoded text, and the draft's log-prob."""

    token_id: int
    text: str
    logprob: float


@dataclass
class TargetScores:
    """Output of a target verification pass.

    ``token_ids[i]`` is the most likely token at position ``i`` according to
    the target model, and ``logprobs[i]`` is its log-probability.
    """

    token_ids: list[int]
    logprobs: list[float]


class BaseBackend(ABC):
    """Backend contract used by :class:`cassi.engine.CassiEngine`.

    Implementations must be deterministic given the same prompt unless a seed
    is explicitly provided. This keeps tests reproducible and lets the
    acceptance-rate controller compare apples to apples.
    """

    # Subclasses may override to advertise their name.
    name: str = "base"

    # --- lifecycle ---------------------------------------------------------
    def setup(self, prompt: str) -> None:
        """Called once at the start of generation to prime any state.

        Default implementation is a no-op so simple backends can ignore it.
        """
        return None

    def reset(self) -> None:
        """Reset any per-generation state (e.g. KV cache)."""
        return None

    # --- draft model ------------------------------------------------------
    @abstractmethod
    def draft_next(self, context_ids: list[int]) -> DraftProposal:
        """Propose the next token id from the draft model.

        ``context_ids`` is the current committed prefix (already-accepted
        tokens). Implementations should be fast and may use a small/quantized
        model, a heuristic, or any cheap predictor.
        """

    # --- target model -----------------------------------------------------
    @abstractmethod
    def target_score(
        self, context_ids: list[int], candidate_ids: list[int]
    ) -> TargetScores:
        """Score ``candidate_ids`` against the target model in one batched pass.

        ``context_ids`` is the committed prefix; ``candidate_ids`` is the draft
        proposal sequence. The implementation returns the *target's* preferred
        token id (and its log-prob) at each candidate position.
        """

    # --- tokenizer shim ---------------------------------------------------
    def encode(self, text: str) -> list[int]:
        """Encode text to token ids. Default is a byte-level fallback."""
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to text. Default is a byte-level fallback."""
        return bytes(ids).decode("utf-8", errors="ignore")

    # --- introspection ----------------------------------------------------
    def describe(self) -> dict[str, str]:
        """Return a short human-readable description for logs/benchmarks."""
        return {"backend": self.name}


__all__ = ["BaseBackend", "DraftProposal", "TargetScores"]
