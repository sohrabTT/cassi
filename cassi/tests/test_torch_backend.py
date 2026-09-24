"""Tests for the real TorchBackend.

These tests require PyTorch and HuggingFace transformers. They are skipped
automatically if the optional 'torch' extra is not installed, so the default
test suite stays fast and dependency-free.

Run with:
    pip install -e ".[torch,dev]"
    pytest tests/test_torch_backend.py -v
"""

import os
import sys

import pytest

# Skip the entire module if torch/transformers are not installed.
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from cassi import CassiEngine
from cassi.backends.torch_backend import TorchBackend


@pytest.fixture(scope="module")
def backend():
    """Module-scoped backend to avoid re-loading models per test."""
    return TorchBackend(
        target_model_name="gpt2",
        draft_model_name="gpt2",
        device="cpu",
        dtype="float32",
    )


def test_backend_describe(backend):
    info = backend.describe()
    assert info["backend"] == "torch"
    assert info["target"] == "gpt2"
    assert info["draft"] == "gpt2"
    assert info["kv_cache"] == "enabled"


def test_encode_decode_round_trip(backend):
    ids = backend.encode("hello world")
    assert isinstance(ids, list)
    assert len(ids) > 0
    text = backend.decode(ids)
    assert "hello" in text.lower()


def test_setup_primes_cache(backend):
    backend.setup("hello")
    assert backend._draft_ctx_len == len(backend._prompt_ids)
    assert backend._target_ctx_len == len(backend._prompt_ids)
    assert backend._draft_past is not None
    assert backend._target_past is not None


def test_draft_next_returns_valid_token(backend):
    backend.setup("hello world")
    proposal = backend.draft_next(backend._prompt_ids)
    assert isinstance(proposal.token_id, int)
    assert 0 <= proposal.token_id < backend._tokenizer.vocab_size


def test_target_score_returns_k_plus_1_positions(backend):
    backend.setup("hello world")
    candidates = [1, 2, 3, 4]  # dummy candidate ids
    scores = backend.target_score(backend._prompt_ids, candidates)
    # Should return len(candidates) + 1 positions (for bonus token)
    assert len(scores.token_ids) == len(candidates) + 1
    assert len(scores.logprobs) == len(candidates) + 1


def test_engine_generates_text_with_torch_backend(backend):
    """End-to-end: engine + TorchBackend produces text."""
    engine = CassiEngine(backend=backend)
    result = engine.generate(
        prompt="The future of AI is",
        max_new_tokens=8,
    )
    assert result.tokens_generated == 8
    assert isinstance(result.text, str)
    assert len(result.text) > 0


def test_engine_output_matches_naive(backend):
    """Critical correctness guarantee: spec output == naive autoregressive output.

    This is the core invariant of speculative decoding: it can never produce
    wrong output, only slower output. We verify this by generating the same
    prompt both ways and checking the text matches exactly.
    """
    prompt = "The future of AI is"
    n_tokens = 8

    # Speculative
    engine = CassiEngine(backend=backend)
    spec_result = engine.generate(prompt=prompt, max_new_tokens=n_tokens)

    # Naive (with KV cache via the backend)
    backend.reset()
    backend.setup(prompt)
    ids = list(backend._prompt_ids)
    prompt_len = len(ids)
    with torch.inference_mode():
        for _ in range(n_tokens):
            logits = backend._target_last_logits
            next_tok = torch.argmax(logits, dim=-1)[0].item()
            ids.append(next_tok)
            next_tensor = torch.tensor(
                [[next_tok]], dtype=torch.long, device=backend.device
            )
            out = backend._target(
                input_ids=next_tensor,
                past_key_values=backend._target_past,
                use_cache=True,
            )
            backend._target_past = out.past_key_values
            backend._target_last_logits = out.logits[:, -1, :]
    naive_text = backend.decode(ids[prompt_len:])

    assert spec_result.text == naive_text, (
        f"Speculative output {spec_result.text!r} != naive output {naive_text!r}"
    )
