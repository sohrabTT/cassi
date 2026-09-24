"""Tests for the EAGLE-style draft head backend.

Skipped automatically if torch/transformers are not installed.

Run with:
    pip install -e ".[torch,dev]"
    pytest tests/test_eagle_backend.py -v
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from cassi import CassiEngine
from cassi.backends.eagle import EagleDraftBackend, EagleDraftHead


@pytest.fixture(scope="module")
def backend():
    """Module-scoped backend to avoid re-loading models per test."""
    return EagleDraftBackend(
        target_model_name="gpt2",
        device="cpu",
        dtype="float32",
    )


def test_eagle_draft_head_module():
    head = EagleDraftHead(hidden_dim=768, vocab_size=50257)
    assert head.hidden_dim == 768
    assert head.vocab_size == 50257
    # Test forward pass
    h = torch.randn(1, 768)
    logits = head(h)
    assert logits.shape == (1, 50257)


def test_backend_describe(backend):
    info = backend.describe()
    assert info["backend"] == "eagle"
    assert info["target"] == "gpt2"
    assert info["kv_cache"] == "enabled"
    assert info["draft_head"] == "eagle-mlp"


def test_setup_primes_cache(backend):
    backend.setup("hello")
    assert backend._draft_head is not None
    assert backend._target_past is not None
    assert backend._target_ctx_len == len(backend._prompt_ids)
    assert backend._last_hidden is not None
    assert backend._last_hidden.shape == (1, 768)


def test_draft_next_returns_valid_token(backend):
    backend.setup("hello world")
    proposal = backend.draft_next(backend._prompt_ids)
    assert isinstance(proposal.token_id, int)
    assert 0 <= proposal.token_id < backend._tokenizer.vocab_size


def test_target_score_returns_k_plus_1_positions(backend):
    backend.setup("hello world")
    candidates = [1, 2, 3, 4]
    scores = backend.target_score(backend._prompt_ids, candidates)
    assert len(scores.token_ids) == len(candidates) + 1
    assert len(scores.logprobs) == len(candidates) + 1


def test_engine_generates_text_with_eagle_backend(backend):
    engine = CassiEngine(backend=backend)
    result = engine.generate(
        prompt="The future of AI is",
        max_new_tokens=8,
        max_k=2,
    )
    assert result.tokens_generated == 8
    assert isinstance(result.text, str)
    assert len(result.text) > 0
