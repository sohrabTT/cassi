"""Quick smoke test for the real TorchBackend with GPT-2."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import torch

from cassi import CassiEngine
from cassi.backends.torch_backend import TorchBackend


def main():
    print("=== CASSI TorchBackend smoke test ===")
    print("Loading models (first call may take ~30s to download)...")
    t0 = time.perf_counter()
    backend = TorchBackend(
        target_model_name="gpt2-medium",
        draft_model_name="gpt2",
        device="cpu",
        dtype="float32",
    )
    print(f"Models loaded in {time.perf_counter()-t0:.1f}s")

    engine = CassiEngine(backend=backend)

    prompt = "The future of artificial intelligence is"
    n_tokens = 20
    print(f"\nPrompt: {prompt!r}")
    print(f"Generating {n_tokens} tokens...\n")

    # Speculative
    t0 = time.perf_counter()
    result = engine.generate(prompt=prompt, max_new_tokens=n_tokens)
    elapsed_spec = time.perf_counter() - t0
    print(f"Speculative:  {elapsed_spec:.2f}s, {result.forward_passes} passes, "
          f"accept={result.acceptance_rate:.1%}, "
          f"speedup_passes={result.speedup_vs_naive:.2f}x")
    print(f"  Output: {result.text!r}")

    # Naive baseline WITH KV cache (fair comparison)
    t0 = time.perf_counter()
    naive_text, naive_passes = naive_generate_cached(backend, prompt, n_tokens)
    elapsed_naive = time.perf_counter() - t0
    print(f"\nNaive (cached): {elapsed_naive:.2f}s, {naive_passes} passes")
    print(f"  Output: {naive_text!r}")

    print(f"\nReal wall-clock speedup: {elapsed_naive/elapsed_spec:.2f}x")
    print(f"Algorithm correctness (texts match): {result.text == naive_text}")


def naive_generate_cached(backend: TorchBackend, prompt: str, max_new_tokens: int) -> tuple[str, int]:
    """Autoregressive with KV cache: one cheap forward pass per token."""
    backend.setup(prompt)  # Re-prime the cache (overwrites spec state)
    ids = list(backend._prompt_ids)
    prompt_len = len(ids)
    # Use the saved last_logits from setup() to predict token 1
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            # Use the cached last_logits to predict next token
            logits = backend._target_last_logits  # (1, vocab)
            next_tok = torch.argmax(logits, dim=-1)[0].item()
            ids.append(next_tok)
            # Feed the new token through the target model to update cache
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
    text = backend.decode(ids[prompt_len:])
    return text, max_new_tokens


if __name__ == "__main__":
    main()
