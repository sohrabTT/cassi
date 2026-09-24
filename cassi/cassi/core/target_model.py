"""Target model wrapper.

The target model is responsible for *verifying* a proposed sequence in a
single batched forward pass. This module wraps a
:class:`~cassi.backends.base.BaseBackend` and exposes a clean verify interface.
"""

from __future__ import annotations

from cassi.backends.base import BaseBackend, TargetScores


class TargetModel:
    """Wrap a backend's ``target_score`` to verify a draft sequence."""

    def __init__(self, backend: BaseBackend) -> None:
        self.backend = backend

    def verify(
        self, context_ids: list[int], candidate_ids: list[int]
    ) -> TargetScores:
        """Score ``candidate_ids`` under the target distribution.

        Returns the target's preferred token id and its log-prob at each
        candidate position.
        """
        return self.backend.target_score(context_ids, candidate_ids)


__all__ = ["TargetModel"]
