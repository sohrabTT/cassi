"""Draft model wrapper.

The draft model is responsible for proposing the next k tokens autoregressively.
This module wraps any :class:`~cassi.backends.base.BaseBackend` and exposes a
clean interface to produce a draft of length ``k``.
"""

from __future__ import annotations

from cassi.backends.base import BaseBackend, DraftProposal


class DraftModel:
    """Wrap a backend's ``draft_next`` to produce a sequence of k proposals.

    The draft model is *stateless* across calls: each call to
    :meth:`propose` must be primed with the current committed context.
    """

    def __init__(self, backend: BaseBackend) -> None:
        self.backend = backend

    def propose(self, context_ids: list[int], k: int) -> list[DraftProposal]:
        """Produce ``k`` draft tokens by repeatedly querying the draft model.

        Each draft step extends the *local* context by the previously proposed
        token; this mimics the autoregressive behaviour of a real small LLM.
        """
        if k <= 0:
            return []
        proposals: list[DraftProposal] = []
        local_ctx = list(context_ids)
        for _ in range(k):
            p = self.backend.draft_next(local_ctx)
            proposals.append(p)
            local_ctx.append(p.token_id)
        return proposals


__all__ = ["DraftModel"]
