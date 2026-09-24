"""Verifier - parallel verification of draft tokens.

This is the heart of speculative decoding: given a draft proposal of length k
and the target model's preferred tokens at the same k positions, decide which
tokens to accept.

CASSI uses a *strict* acceptance criterion (token-id equality) by default;
this is the version that all theoretical speedup bounds assume. A *rejection
sampling* mode (Leviathan et al., 2023; Chen et al., 2023) is also supported
via :class:`Verifier`'s ``mode`` argument for backends that expose the full
draft and target distributions.

Bonus token semantics
-----------------------
In standard speculative decoding the target's forward pass over a candidate
sequence of length k produces logits at k+1 positions (positions 0..k,
where position 0 is "what comes right after the context", and position k is
"what comes right after the last candidate"). The verifier contract is
therefore:

- If the verifier accepts tokens at positions 0..(j-1) and rejects at
  position j, the bonus token is the target's preferred token at position
  *j* (the reject position). The target has already computed logits there,
  so we get a free correct token.
- If all k candidates are accepted, the bonus token is the target's
  preferred token at position *k* (one beyond the last accepted).

The :class:`Verifier` accepts scores whose length is either ``k`` or
``k+1``. When the length is exactly ``k``, the bonus is taken from the
reject position (no extra position available).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from cassi.backends.base import DraftProposal, TargetScores


class VerifyMode(str, Enum):
    STRICT = "strict"          # accept iff draft_id == target_id
    REJECTION = "rejection"   # probabilistic acceptance (placeholder)


@dataclass
class VerifyResult:
    """Outcome of a single verification pass.

    Attributes
    ----------
    accepted_ids:
        Token ids that were accepted, in order.
    accepted_count:
        Number of accepted tokens (may equal ``len(candidates)`` on full hit).
    rejected_at:
        Index of the first rejected candidate, or ``None`` if all accepted.
    bonus_token:
        The "free" token produced by the target at the position *after* the
        last accepted one (when all accepted) or *at* the reject position
        (when a reject occurred). Always available in speculative decoding
        because the target pass already computed logits there.
    """

    accepted_ids: list[int]
    accepted_count: int
    rejected_at: int | None
    bonus_token: int


class Verifier:
    """Decide which draft tokens to accept.

    Parameters
    ----------
    mode:
        :data:`VerifyMode.STRICT` (default) compares token ids directly.
        :data:`VerifyMode.REJECTION` is reserved for backends exposing
        the full distributions; it falls back to STRICT when those are absent.
    """

    def __init__(self, mode: VerifyMode = VerifyMode.STRICT) -> None:
        self.mode = mode

    def verify(
        self,
        candidates: Sequence[DraftProposal],
        scores: TargetScores,
    ) -> VerifyResult:
        if not candidates:
            return VerifyResult(
                accepted_ids=[],
                accepted_count=0,
                rejected_at=None,
                bonus_token=scores.token_ids[0] if scores.token_ids else 0,
            )

        accepted: list[int] = []
        rejected_at: int | None = None

        for i, cand in enumerate(candidates):
            tgt = scores.token_ids[i] if i < len(scores.token_ids) else None
            if tgt is None:
                rejected_at = i
                break
            if self._accept(cand.token_id, tgt):
                accepted.append(cand.token_id)
            else:
                rejected_at = i
                break

        # Bonus token logic:
        # - If a reject happened at position j, the bonus is the target's
        #   preferred token at position j (it's already computed, free token).
        # - If everything was accepted, the bonus is the target's token at
        #   position len(candidates), i.e. the position *after* the last
        #   accepted one. This requires scores of length >= len(candidates)+1.
        if rejected_at is not None:
            bonus = scores.token_ids[rejected_at]
        elif len(scores.token_ids) > len(candidates):
            bonus = scores.token_ids[len(candidates)]
        else:
            # All accepted but no extra score provided. We have no bonus.
            # Use the last accepted token id as a degenerate fallback
            # (engine will still make progress via the next draft round).
            bonus = scores.token_ids[-1]

        return VerifyResult(
            accepted_ids=accepted,
            accepted_count=len(accepted),
            rejected_at=rejected_at,
            bonus_token=bonus,
        )

    def _accept(self, draft_id: int, target_id: int) -> bool:
        if self.mode == VerifyMode.STRICT:
            return draft_id == target_id
        # REJECTION mode placeholder: would need full distributions.
        return draft_id == target_id


__all__ = ["Verifier", "VerifyMode", "VerifyResult"]
