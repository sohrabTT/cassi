"""Benchmark: real speculative decoding with GPT-2 on TorchBackend.

Runs CASSI with real GPT-2 models and compares against a fair KV-cached
naive autoregressive baseline. Saves results to ``results/benchmark_real.json``
and ``results/benchmark_real.png``.

The naive baseline ALSO uses KV caching, so the comparison isolates the
*speculative-decoding algorithm* from the *KV-cache optimization*. Without
this, naive would re-process the entire context every step and speculative
would always win, which is misleading.

Usage
-----
    python benchmarks/real_gpt2.py
    python benchmarks/real_gpt2.py --tokens 16 --num-prompts 3
    python benchmarks/real_gpt2.py --draft gpt2 --target gpt2-xl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import torch

from cassi import CassiEngine
from cassi.backends.torch_backend import TorchBackend


PROMPTS = [
    "The future of artificial intelligence is",
    "Once upon a time, in a galaxy far far away,",
    "The most important thing about machine learning is",
    "In the year 2050, humanity will finally",
    "Python is a programming language that",
]


@dataclass
class BenchmarkRow:
    combo_name: str
    prompt: str
    prompt_len: int
    tokens: int
    spec_time_sec: float
    naive_time_sec: float
    spec_passes: int
    acceptance_rate: float
    speedup_passes: float
    speedup_wallclock: float
    final_k: int
    draft_params_m: float
    target_params_m: float
    spec_text: str
    naive_text: str
    texts_match: bool


def naive_generate_cached(
    backend: TorchBackend, prompt: str, max_new_tokens: int
) -> tuple[str, int]:
    """Fair naive baseline: KV-cached autoregressive, one pass per token.

    This is the same baseline that production inference engines (vLLM, TGI)
    use. Comparing speculative decoding against this isolates the algorithmic
    benefit from the KV-cache benefit.
    """
    backend.setup(prompt)  # Re-prime the cache (overwrites spec state)
    ids = list(backend._prompt_ids)
    prompt_len = len(ids)
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = backend._target_last_logits  # (1, vocab)
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
    text = backend.decode(ids[prompt_len:])
    return text, max_new_tokens


def run_combo(
    draft: str,
    target: str,
    prompts: list[str],
    tokens: int,
    device: str,
) -> list[BenchmarkRow]:
    combo = f"{draft}+{target}"
    print(f"\n=== {combo} ===")
    print(f"Loading models on {device}...")
    t0 = time.perf_counter()
    backend = TorchBackend(
        target_model_name=target,
        draft_model_name=draft,
        device=device,
        dtype="float32",
    )
    print(f"Models loaded in {time.perf_counter()-t0:.1f}s")
    print(f"  target: {backend.target_model_name} "
          f"({sum(p.numel() for p in backend._target.parameters())/1e6:.0f}M params)")
    print(f"  draft : {backend.draft_model_name} "
          f"({sum(p.numel() for p in backend._draft.parameters())/1e6:.0f}M params)")

    engine = CassiEngine(backend=backend)
    draft_params = sum(p.numel() for p in backend._draft.parameters()) / 1e6
    target_params = sum(p.numel() for p in backend._target.parameters()) / 1e6

    rows: list[BenchmarkRow] = []
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] {prompt[:60]!r}...")

        # Speculative
        t0 = time.perf_counter()
        spec_result = engine.generate(prompt=prompt, max_new_tokens=tokens)
        spec_time = time.perf_counter() - t0

        # Naive (KV-cached)
        t0 = time.perf_counter()
        naive_text, naive_passes = naive_generate_cached(backend, prompt, tokens)
        naive_time = time.perf_counter() - t0

        row = BenchmarkRow(
            combo_name=combo,
            prompt=prompt,
            prompt_len=len(backend.encode(prompt)),
            tokens=tokens,
            spec_time_sec=spec_time,
            naive_time_sec=naive_time,
            spec_passes=spec_result.forward_passes,
            acceptance_rate=spec_result.acceptance_rate,
            speedup_passes=spec_result.speedup_vs_naive,
            speedup_wallclock=naive_time / spec_time if spec_time > 0 else 0,
            final_k=spec_result.final_k,
            draft_params_m=draft_params,
            target_params_m=target_params,
            spec_text=spec_result.text,
            naive_text=naive_text,
            texts_match=spec_result.text == naive_text,
        )
        rows.append(row)
        print(f"  spec : {spec_time:6.2f}s  passes={spec_result.forward_passes:3d}  "
              f"accept={spec_result.acceptance_rate:5.1%}  "
              f"speedup_wall={row.speedup_wallclock:.2f}x  "
              f"texts_match={row.texts_match}")
        print(f"  naive: {naive_time:6.2f}s  passes={naive_passes:3d}  (KV-cached)")
    return rows


def save_results(rows: list[BenchmarkRow], output_json: str, output_plot: str) -> None:
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump([asdict(r) for r in rows], f, indent=2)
    print(f"\nResults saved to {output_json}")

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

        # Left: wall-clock time per prompt
        ax = axes[0]
        combos = sorted({r.combo_name for r in rows})
        n_combos = len(combos)
        x = np.arange(len(rows) // max(1, n_combos) if n_combos else 1)
        width = 0.4

        idx = 0
        for ci, combo in enumerate(combos):
            combo_rows = [r for r in rows if r.combo_name == combo]
            xs = np.arange(len(combo_rows)) + ci * (len(combo_rows) + 1)
            spec_t = [r.spec_time_sec for r in combo_rows]
            naive_t = [r.naive_time_sec for r in combo_rows]
            ax.bar(xs - width / 2, spec_t, width, label=f"{combo} spec", alpha=0.85)
            ax.bar(xs + width / 2, naive_t, width, label=f"{combo} naive", alpha=0.55)
        ax.set_xlabel("Prompt index (grouped by combo)")
        ax.set_ylabel("Wall-clock time (s)")
        ax.set_title("Speculative vs KV-cached naive (real GPT-2 on CPU)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

        # Right: acceptance rate per prompt
        ax = axes[1]
        for ci, combo in enumerate(combos):
            combo_rows = [r for r in rows if r.combo_name == combo]
            acc = [r.acceptance_rate * 100 for r in combo_rows]
            ax.plot(range(len(acc)), acc, "o-", label=combo)
        ax.set_xlabel("Prompt index")
        ax.set_ylabel("Acceptance rate (%)")
        ax.set_title("Draft acceptance rate per prompt")
        ax.legend()
        ax.grid(alpha=0.3)

        fig.savefig(output_plot, dpi=120)
        print(f"Plot saved to {output_plot}")
    except ImportError:
        print("matplotlib not installed; skipping plot")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", default=None,
                        help="Override draft model (e.g. gpt2)")
    parser.add_argument("--target", default=None,
                        help="Override target model (e.g. gpt2-large)")
    parser.add_argument("--tokens", type=int, default=16,
                        help="Tokens to generate per prompt (default 16)")
    parser.add_argument("--num-prompts", type=int, default=3,
                        help="Number of prompts to use (default 3)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", default="results/benchmark_real.json")
    parser.add_argument("--output-plot", default="results/benchmark_real.png")
    args = parser.parse_args()

    prompts = PROMPTS[: args.num_prompts]

    if args.draft and args.target:
        combos = [(args.draft, args.target)]
    else:
        combos = [
            ("gpt2", "gpt2-medium"),
        ]

    all_rows: list[BenchmarkRow] = []
    for draft, target in combos:
        try:
            rows = run_combo(draft, target, prompts, args.tokens, args.device)
            all_rows.extend(rows)
        except Exception as e:
            print(f"  Combo {draft}+{target} failed: {e}")
            import traceback; traceback.print_exc()
            continue

    print("\n=== Summary ===")
    for r in all_rows:
        print(f"{r.combo_name:30s}  prompt={r.prompt_len:3d}  "
              f"spec={r.spec_time_sec:.2f}s  naive={r.naive_time_sec:.2f}s  "
              f"speedup={r.speedup_wallclock:.2f}x  accept={r.acceptance_rate:.1%}  "
              f"match={r.texts_match}")

    save_results(all_rows, args.output_json, args.output_plot)


if __name__ == "__main__":
    main()
