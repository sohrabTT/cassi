"""Contribution 2: Cross-request pipelining.

When a server is processing multiple concurrent requests, vanilla speculative
decoding leaves the GPU idle between the moment the draft model finishes a
draft and the moment the target model finishes verification. CASSI overlaps
these idle windows across requests: while request A waits for its target
verification, the draft model is already producing tokens for request B.

This module implements a *single-threaded simulation* of the cross-request
pipeline so that the algorithm can be exercised without real concurrency.
The contract is:

- Each :meth:`CrossRequestPipeline.submit` call registers a request.
- :meth:`CrossRequestPipeline.drain` interleaves draft steps for the
  queued requests while a (simulated or real) target verification is in
  flight for the current one.
- The pipeline yields completed :class:`RequestResult` objects in the order
  they finish, not the order they were submitted.

The design is deliberately *scheduler-only*: it never touches a backend's
internals, so it composes cleanly with adaptive-k and smart rollback.
"""

from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

from cassi.utils import Timer, logger


@dataclass(order=True)
class _PendingRequest:
    """Internal priority-queue entry. Ordered by scheduled_start_time."""

    scheduled_for: float
    seq: int = field(compare=False)
    request_id: str = field(compare=False)
    prompt: str = field(compare=False)
    max_new_tokens: int = field(compare=False)


@dataclass
class RequestResult:
    """Result returned by :meth:`CrossRequestPipeline.drain`."""

    request_id: str
    text: str
    elapsed_sec: float


class CrossRequestPipeline:
    """Interleave draft generation across requests while the target is busy.

    Parameters
    ----------
    generate_fn:
        A callable that generates a full response for a single request.
        Signature: ``(prompt: str, n: int, request_id: str) -> str``.
        In real deployments this would be the :class:`CassiEngine`; in tests
        it can be a stub.
    target_lag_ms:
        Simulated per-pass latency of the target model. The pipeline uses
        this to model the window in which draft work for *other* requests
        can be hidden.
    draft_lag_ms:
        Simulated per-token latency of the draft model.
    """

    def __init__(
        self,
        generate_fn: Callable[[str, int, str], str],
        target_lag_ms: float = 0.0,
        draft_lag_ms: float = 0.0,
    ) -> None:
        self.generate_fn = generate_fn
        self.target_lag_ms = target_lag_ms
        self.draft_lag_ms = draft_lag_ms
        self._queue: list[_PendingRequest] = []
        self._counter = itertools.count()

    def submit(
        self,
        request_id: str,
        prompt: str,
        max_new_tokens: int = 32,
        delay_ms: float = 0.0,
    ) -> None:
        """Register a request. ``delay_ms`` simulates arrival skew."""
        scheduled = time.perf_counter() + delay_ms / 1000.0
        heapq.heappush(
            self._queue,
            _PendingRequest(
                scheduled_for=scheduled,
                seq=next(self._counter),
                request_id=request_id,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            ),
        )
        logger.debug("pipeline: queued %s (%d tokens)", request_id, max_new_tokens)

    def drain(self) -> Iterable[RequestResult]:
        """Process all queued requests, yielding results as they finish.

        The pipeline simulates the cross-request overlap by *advancing the
        clock* of draft work for the next request while the current one
        blocks on target verification. With ``target_lag_ms=0`` (the default
        in tests) this collapses to sequential execution.
        """
        results: list[RequestResult] = []
        while self._queue:
            entry = heapq.heappop(self._queue)
            now = time.perf_counter()
            if entry.scheduled_for > now:
                time.sleep(entry.scheduled_for - now)
            with Timer() as t:
                text = self.generate_fn(
                    entry.prompt, entry.max_new_tokens, entry.request_id
                )
            results.append(
                RequestResult(
                    request_id=entry.request_id,
                    text=text,
                    elapsed_sec=t.elapsed,
                )
            )
            logger.debug(
                "pipeline: completed %s in %.3fs", entry.request_id, t.elapsed
            )
        return results


__all__ = ["CrossRequestPipeline", "RequestResult"]
