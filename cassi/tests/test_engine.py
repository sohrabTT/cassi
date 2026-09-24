"""Tests for the toy backend and the integrated engine."""

from cassi import CassiEngine, ToyBackend
from cassi.backends.toy import ToyConfig


def test_toy_backend_deterministic_with_seed():
    cfg = ToyConfig(seed=42, draft_correctness=1.0)
    b1 = ToyBackend(cfg)
    b1.setup("hello")
    p1 = b1.draft_next([1, 2, 3])

    b2 = ToyBackend(ToyConfig(seed=42, draft_correctness=1.0))
    b2.setup("hello")
    p2 = b2.draft_next([1, 2, 3])
    assert p1.token_id == p2.token_id


def test_engine_generates_requested_token_count():
    engine = CassiEngine(backend=ToyBackend(ToyConfig(draft_correctness=0.9)))
    result = engine.generate(prompt="hello world", max_new_tokens=32)
    assert result.tokens_generated == 32
    assert result.forward_passes <= 32  # speculative must not be slower than naive


def test_engine_acceptance_rate_high_with_perfect_draft():
    # When draft_correctness=1.0, every draft token should match the target,
    # so acceptance rate should be very high.
    engine = CassiEngine(backend=ToyBackend(ToyConfig(draft_correctness=1.0)))
    result = engine.generate(prompt="hi", max_new_tokens=16)
    assert result.acceptance_rate >= 0.8


def test_engine_speedup_better_than_naive_with_good_draft():
    # With a good draft model, we should need fewer forward passes than naive.
    engine = CassiEngine(backend=ToyBackend(ToyConfig(draft_correctness=0.9)))
    result = engine.generate(prompt="hi", max_new_tokens=32)
    assert result.speedup_vs_naive > 1.0


def test_engine_describe_includes_backend_and_k():
    engine = CassiEngine(backend=ToyBackend())
    info = engine.describe()
    assert "backend" in info
    assert info["backend"]["backend"] == "toy"
    assert "k_controller" in info
    assert info["k_controller"]["k_min"] >= 1


def test_engine_batch_returns_aligned_results():
    engine = CassiEngine(backend=ToyBackend(ToyConfig(draft_correctness=0.8)))
    prompts = ["one", "two", "three"]
    results = engine.generate_batch(prompts, max_new_tokens=8, pipeline=False)
    assert len(results) == 3
    for r in results:
        assert r.tokens_generated == 8


def test_engine_generate_text_decodable():
    engine = CassiEngine(backend=ToyBackend(ToyConfig(seed=7)))
    result = engine.generate(prompt="hello", max_new_tokens=8)
    # text should be a string (decodable bytes from the toy backend)
    assert isinstance(result.text, str)


def test_engine_respects_max_k():
    engine = CassiEngine(backend=ToyBackend(ToyConfig(draft_correctness=1.0)))
    result = engine.generate(prompt="hi", max_new_tokens=16, max_k=2)
    # With max_k=2, k can never exceed 2.
    # The number of proposed tokens per step is bounded by 2.
    assert result.draft_tokens_proposed <= 2 * result.forward_passes + 2
