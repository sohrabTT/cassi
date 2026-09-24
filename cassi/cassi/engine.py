"""CASSI main engine.

The :class:`CassiEngine` ties together the draft model, target model,
verifier, adaptive-k controller, and smart-rollback into a single
speculative decoding loop.

Algorithm
---------
1. Prime the backend with the prompt.
2. While more tokens are needed:
   a. Ask the adaptive-k controller for the current ``k``.
   b. Optionally prepend the salvaged prefix from the previous round
      (Contribution 3) to the draft.
   c. Have the draft model propose ``k`` tokens.
   d. Verify them in one batched target pass.
   e. Accept the longest prefix of matches; emit a *bonus* token if the
      target pass produced one.
   f. Feed (accepted, proposed) back to the adaptive-k controller
      (Contribution 1).
   g. Update rollback state (Contribution 3).
3. Return the result with full telemetry.
"""

from __future__ import annotations

import logging
from typing import Any

from cassi.backends.base import BaseBackend, DraftProposal
from cassi.core.adaptive_k import AdaptiveK
from cassi.core.cross_request import CrossRequestPipeline
from cassi.core.draft_model import DraftModel
from cassi.core.rollback import RollbackState, SmartRollback
from cassi.core.target_model import TargetModel
from cassi.core.verifier import Verifier, VerifyMode
from cassi.utils import GenerationResult, Token, Timer, configure_logging, logger


class CassiEngine:
    """Speculative decoding engine with CASSI's three contributions.

    Parameters
    ----------
    backend:
        Any :class:`~cassi.backends.base.BaseBackend` implementation. The
        default :class:`~cassi.backends.toy.ToyBackend` lets you develop and
        test without any ML runtime installed.
    adaptive_k:
        Optional pre-configured :class:`AdaptiveK`. If ``None``, a default
        controller with ``k_min=2, k_max=8`` is used.
    verifier:
        Optional pre-configured :class:`Verifier`. Defaults to STRICT mode.
    rollback:
        Optional pre-configured :class:`SmartRollback`. Defaults to salvaging
        up to 8 tokens after a reject.
    log_level:
        Logging level for the ``cassi`` logger.
    """

    def __init__(
        self,
        backend: BaseBackend,
        adaptive_k: AdaptiveK | None = None,
        verifier: Verifier | None = None,
        rollback: SmartRollback | None = None,
        log_level: int = logging.WARNING,
    ) -> None:
        configure_logging(log_level)
        self.backend = backend
        self.draft = DraftModel(backend)
        self.target = TargetModel(backend)
        self.adaptive_k = adaptive_k or AdaptiveK(k_min=2, k_max=8, k_init=4)
        self.verifier = verifier or Verifier(mode=VerifyMode.STRICT)
        self.rollback = rollback or SmartRollback()
        self._rollback_state = RollbackState()

    # ----------------------------------------------------------------- API
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        max_k: int | None = None,
    ) -> GenerationResult:
        """Generate up to ``max_new_tokens`` tokens for ``prompt``.

        Parameters
        ----------
        prompt:
            The input string.
        max_new_tokens:
            Maximum number of new tokens to produce.
        max_k:
            Optional per-call cap on the draft length. Useful when the
            adaptive controller is warming up.
        """
        self.backend.reset()
        self.backend.setup(prompt)
        self._rollback_state = RollbackState()

        context_ids: list[int] = list(self.backend.encode(prompt))
        emitted: list[Token] = []
        draft_proposed_total = 0
        draft_accepted_total = 0
        forward_passes = 0

        with Timer() as timer:
            while len(emitted) < max_new_tokens:
                # ---- 1. current k from the adaptive controller ----
                k = self.adaptive_k.current_k()
                if max_k is not None:
                    k = min(k, max_k)

                # ---- 2. (Contribution 3) reuse salvaged prefix hint ----
                salvage = self._rollback_state.salvaged_ids
                salvaged_count = min(len(salvage), k)
                reused_ids = list(salvage[:salvaged_count])

                # ---- 3. draft proposes (k - salvaged_count) more tokens ----
                extra_proposals = self.draft.propose(
                    context_ids + reused_ids,
                    k - salvaged_count,
                )
                proposals: list[DraftProposal] = [
                    DraftProposal(token_id=tid, text="", logprob=-0.5)
                    for tid in reused_ids
                ] + extra_proposals
                full_draft_ids = [p.token_id for p in proposals]
                draft_proposed_total += len(full_draft_ids)

                # ---- 4. verify in a single batched target pass ----
                scores = self.target.verify(context_ids, full_draft_ids)
                forward_passes += 1
                verdict = self.verifier.verify(proposals, scores)

                accepted = list(verdict.accepted_ids)

                # ---- 5. emit accepted tokens + bonus token ----
                # Only emit up to max_new_tokens tokens total. The bonus
                # token is emitted last and only if we still have budget.
                remaining = max_new_tokens - len(emitted)
                accepted_to_emit = accepted[:remaining]
                for tid in accepted_to_emit:
                    emitted.append(Token(id=tid, accepted=True))
                # Bonus token fills the next slot if there is one.
                if len(emitted) < max_new_tokens:
                    emitted.append(Token(id=verdict.bonus_token, accepted=True))
                    accepted = accepted_to_emit + [verdict.bonus_token]
                else:
                    accepted = accepted_to_emit

                draft_accepted_total += verdict.accepted_count
                context_ids.extend(accepted)

                # ---- 6. (Contribution 1) update adaptive k ----
                self.adaptive_k.update(verdict.accepted_count, len(full_draft_ids))

                # ---- 7. (Contribution 3) compute rollback state ----
                self._rollback_state = self.rollback.apply(
                    draft_ids=full_draft_ids,
                    accepted_count=verdict.accepted_count,
                    reject_position=verdict.rejected_at,
                )

                logger.debug(
                    "step: k=%d proposed=%d accepted=%d bonus=%s",
                    k, len(full_draft_ids), verdict.accepted_count,
                    verdict.bonus_token,
                )

                if not accepted:
                    # Safety: avoid infinite loop if backend misbehaves.
                    logger.warning("no progress in step - breaking")
                    break

        text = self.backend.decode([t.id for t in emitted])
        acceptance_rate = (
            draft_accepted_total / draft_proposed_total
            if draft_proposed_total
            else 0.0
        )
        return GenerationResult(
            text=text,
            tokens=emitted,
            tokens_generated=len(emitted),
            draft_tokens_proposed=draft_proposed_total,
            draft_tokens_accepted=draft_accepted_total,
            forward_passes=forward_passes,
            elapsed_sec=timer.elapsed,
            acceptance_rate=acceptance_rate,
            final_k=self.adaptive_k.current_k(),
        )

    # ------------------------------------------------- batch convenience
    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 32,
        pipeline: bool = True,
    ) -> list[GenerationResult]:
        """Generate for multiple prompts.

        When ``pipeline=True``, requests are routed through
        :class:`CrossRequestPipeline` (Contribution 2) to demonstrate the
        cross-request overlap. When ``False``, requests are processed
        sequentially as a baseline.
        """
        if not pipeline:
            return [self.generate(p, max_new_tokens) for p in prompts]

        # Pipeline path: a single pipeline processes all prompts. The
        # generate_fn closure stashes the GenerationResult by request_id
        # via a side-channel so the engine only runs once per prompt.
        results_by_id: dict[str, GenerationResult] = {}

        def _gen(prompt: str, n: int, request_id: str) -> str:
            r = self.generate(prompt, max_new_tokens=n)
            results_by_id[request_id] = r
            return r.text

        pipe = CrossRequestPipeline(generate_fn=_gen)
        for i, p in enumerate(prompts):
            req_id = f"req-{i:03d}"
            pipe.submit(req_id, p, max_new_tokens, delay_ms=i * 1.0)
        pipe.drain()

        # Return results in input order.
        return [results_by_id[f"req-{i:03d}"] for i in range(len(prompts))]

    # ------------------------------------------------------- introspection
    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.backend.describe(),
            "k_controller": {
                "k_min": self.adaptive_k.k_min,
                "k_max": self.adaptive_k.k_max,
                "current_k": self.adaptive_k.current_k(),
            },
            "rollback": {
                "salvage_after_reject": self.rollback.salvage_after_reject,
                "max_salvage": self.rollback.max_salvage,
            },
        }


__all__ = ["CassiEngine"]
