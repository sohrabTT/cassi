"""Tests for the cross-request pipeline."""

import time

from cassi.core.cross_request import CrossRequestPipeline


def test_drain_returns_results_for_all_submitted():
    pipe = CrossRequestPipeline(generate_fn=lambda p, n, rid: p[::-1])
    pipe.submit("a", "hello", max_new_tokens=4)
    pipe.submit("b", "world", max_new_tokens=4)
    results = pipe.drain()
    assert len(results) == 2
    texts = {r.request_id: r.text for r in results}
    assert texts["a"] == "olleh"
    assert texts["b"] == "dlrow"


def test_drain_includes_elapsed():
    def slow_gen(prompt: str, n: int, rid: str) -> str:
        time.sleep(0.01)
        return prompt

    pipe = CrossRequestPipeline(generate_fn=slow_gen)
    pipe.submit("x", "p", max_new_tokens=2)
    results = pipe.drain()
    assert len(results) == 1
    assert results[0].elapsed_sec >= 0.005


def test_empty_pipeline_returns_empty():
    pipe = CrossRequestPipeline(generate_fn=lambda p, n, rid: p)
    assert list(pipe.drain()) == []


def test_delayed_submission_waited():
    """A submission with delay_ms should cause drain to sleep at least that long."""
    def fast(p, n, rid):
        return p

    pipe = CrossRequestPipeline(generate_fn=fast)
    t0 = time.perf_counter()
    pipe.submit("d", "p", max_new_tokens=1, delay_ms=30)
    pipe.drain()
    elapsed = time.perf_counter() - t0
    assert elapsed >= 0.025


def test_request_id_passed_to_generate_fn():
    """The pipeline should forward the request_id to generate_fn."""
    seen_ids: list[str] = []

    def gen(p: str, n: int, rid: str) -> str:
        seen_ids.append(rid)
        return p

    pipe = CrossRequestPipeline(generate_fn=gen)
    pipe.submit("alpha", "x", max_new_tokens=1)
    pipe.submit("beta", "y", max_new_tokens=1)
    pipe.drain()
    assert set(seen_ids) == {"alpha", "beta"}
