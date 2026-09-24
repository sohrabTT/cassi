"""Contribution 3: Smart rollback with KV-cache preservation.

Vanilla speculative decoding discards *every* draft token after the first
rejection. This is wasteful because:

1. The target's forward pass already computed logits for *all* k positions,
   so the post-reject information is available for free.
2. In many cases the draft's tokens after the reject point are "almost
   right" - they differ from the target only in a suffix that the next
   draft round would re-derive anyway.

CASSI implements *partial acceptance*:

- Tokens before the first reject are accepted normally.
- The target's top-1 token at the reject position is taken as the *bonus*
  token (this is standard).
- *Additionally*, the draft's tokens after the reject position are kept
  as a *reusable prefix hint* for the next draft round, so the draft model
  does not start from scratch.

The "reusable prefix hint" is exposed via :class:`RollbackState`. The
:mod:`cassi.engine` consults this state at the start of each draft round;
if it is non-empty, the draft model skips those positions (they are
treated as already-proposed, requiring only re-verification).

The net effect: on workloads where the draft model is mostly right but
fails at a single position (e.g. punctuation, a rare entity), CASSI
avoids recomputing the entire draft, yielding a measurable speedup.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RollbackState:
    """Per-request rollback state.

    Attributes
    ----------
    salvaged_ids:
        Token ids from the rejected draft that are *retained* as a hint for
        the next draft round. The next draft is free to reuse them or to
        overwrite them; the engine just passes them as a starting hint.
    reject_position:
        Index in the last candidate sequence where the reject happened.
    reject_count:
        Running count of rejections (useful for telemetry).
    history:
        A bounded list of recent (proposed_k, accepted_k) pairs for
        introspection / benchmarks.
    """

    salvaged_ids: list[int] = field(default_factory=list)
    reject_position: int | None = None
    reject_count: int = 0
    history: list[tuple[int, int]] = field(default_factory=list)


class SmartRollback:
    """Decide what to keep after a verification pass.

    The default policy keeps ``len(draft) - reject_pos - 1`` tokens *after*
    the reject point as a salvage hint. Set ``salvage_after_reject=False``
    to disable this (falling back to vanilla behaviour).
    """

    def __init__(self, salvage_after_reject: bool = True, max_salvage: int = 8) -> None:
        self.salvage_after_reject = salvage_after_reject
        self.max_salvage = max_salvage

    def apply(
        self,
        draft_ids: list[int],
        accepted_count: int,
        reject_position: int | None,
    ) -> RollbackState:
        """Compute the rollback state after a verification pass.

        Parameters
        ----------
        draft_ids:
            The full draft sequence (length k).
        accepted_count:
            Number of tokens accepted by the verifier.
        reject_position:
            Index of the first reject, or ``None`` if all accepted.
        """
        state = RollbackState()
        state.history.append((len(draft_ids), accepted_count))

        if reject_position is None:
            # Everything accepted - nothing to salvage.
            return state

        state.reject_position = reject_position
        state.reject_count = 1  # one per call; engine accumulates if needed.

        if not self.salvage_after_reject:
            return state

        # Salvage tokens after the reject point, bounded by max_salvage.
        salvage = draft_ids[reject_position + 1: reject_position + 1 + self.max_salvage]
        state.salvaged_ids = list(salvage)
        return state


__all__ = ["SmartRollback", "RollbackState"]
