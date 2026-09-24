"""Benchmark: CASSI with EAGLE-trained draft head vs vanilla (untrained) vs naive.

Compares three configurations on real GPT-2:
  1. Naive (KV-cached autoregressive, no speculation)
  2. CASSI + untrained EAGLE draft head (baseline, expect ~0% acceptance)
  3. CASSI + trained EAGLE draft head (expect high acceptance + speedup)

Saves results to ``results/benchmark_eagle.json`` and
``results/benchmark_eagle.png``.

Usage
-----
    python benchmarks/real_eagle.py --target gpt2-medium --tokens 16
    python benchmarks/real_eagle.py --target gpt2 --tokens 16 --num-prompts 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import torch

from cassi import CassiEngine
from cassi.backends.eagle import EagleDraftBackend


PROMPTS = [
    "The future of artificial intelligence is",
    "Once upon a time, in a galaxy far far away,",
    "The most important thing about machine learning is",
    "In the year 2050, humanity will finally",
    "Python is a programming language that",
]


@dataclass
class BenchmarkRow:
    config: str
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
    spec_text: str
    naive_text: str
    texts_match: bool


def naive_generate_cached(
    backend: EagleDraftBackend, prompt: str, max_new_tokens: int,
) -> tuple[str, int]:
    """KV-cached naive baseline: one forward pass per token."""
    backend.setup(prompt)  # Re-prime cache
    ids = list(backend._prompt_ids)
    prompt_len = len(ids)
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            # Use the cached last hidden state -> target's LM head logits
            # We need a forward pass on the cached state to get logits.
            # Simpler: just use the target model directly.
            last_tensor = torch.tensor(
                [[ids[-1]]], dtype=torch.long, device=backend.device
            )
            out = backend._target(
                input_ids=last_tensor,
                past_key_values=backend._target_past,
                use_cache=True,
                output_hidden_states=True,
            )
            backend._target_past = out.past_key_values
            next_tok = torch.argmax(out.logits[0, -1, :]).item()
            ids.append(next_tok)
            backend._last_hidden = out.hidden_states[-1][:, -1, :]
    text = backend.decode(ids[prompt_len:])
    return text, max_new_tokens


def run_config(
    backend: EagleDraftBackend,
    config_name: str,
    prompts: list[str],
    tokens: int,
) -> list[BenchmarkRow]:
    engine = CassiEngine(backend=backend)
    rows: list[BenchmarkRow] = []
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] ({config_name}) {prompt[:60]!r}...")

        # Speculative
        t0 = time.perf_counter()
        spec_result = engine.generate(prompt=prompt, max_new_tokens=tokens)
        spec_time = time.perf_counter() - t0

        # Naive
        t0 = time.perf_counter()
        naive_text, naive_passes = naive_generate_cached(backend, prompt, tokens)
        naive_time = time.perf_counter() - t0

        row = BenchmarkRow(
            config=config_name,
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
            spec_text=spec_result.text,
            naive_text=naive_text,
            texts_match=spec_result.text == naive_text,
        )
        rows.append(row)
        print(f"  spec : {spec_time:6.2f}s  passes={spec_result.forward_passes:3d}  "
              f"accept={spec_result.acceptance_rate:5.1%}  "
              f"speedup_wall={row.speedup_wallclock:.2f}x  "
              f"texts_match={row.texts_match}")
        print(f"  naive: {naive_time:6.2f}s  passes={naive_passes:3d}")
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

        # Left: wall-clock speedup per prompt, grouped by config
        ax = axes[0]
        configs = sorted({r.config for r in rows})
        width = 0.35
        x = np.arange(len(rows) // max(1, len(configs)))
        for ci, cfg in enumerate(configs):
            cfg_rows = [r for r in rows if r.config == cfg]
            speedups = [r.speedup_wallclock for r in cfg_rows]
            ax.bar(x + ci * width, speedups, width, label=cfg)
        ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5,
                   label="break-even (1.0x)")
        ax.set_xlabel("Prompt index")
        ax.set_ylabel("Wall-clock speedup (×)")
        ax.set_title("Real speedup vs naive (KV-cached)")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")

        # Right: acceptance rate per prompt
        ax = axes[1]
        for ci, cfg in enumerate(configs):
            cfg_rows = [r for r in rows if r.config == cfg]
            acc = [r.acceptance_rate * 100 for r in cfg_rows]
            ax.plot(range(len(acc)), acc, "o-", label=cfg)
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
    parser.add_argument("--target", default="gpt2-medium",
                        help="Target model name (default gpt2-medium)")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--output-json", default="results/benchmark_eagle.json")
    parser.add_argument("--output-plot", default="results/benchmark_eagle.png")
    args = parser.parse_args()

    prompts = PROMPTS[: args.num_prompts]
    safe_name = args.target.replace("/", "_")
    ckpt_path = Path(args.checkpoint_dir) / f"draft_{safe_name}.pt"
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint {ckpt_path} not found. Run "
              f"`python scripts/train_eagle_head.py --target {args.target}` first.")
        sys.exit(1)

    all_rows: list[BenchmarkRow] = []

    # Config 1: trained EAGLE draft head
    print(f"\n=== Config 1: trained EAGLE draft head ({ckpt_path}) ===")
    backend_trained = EagleDraftBackend(
        target_model_name=args.target,
        device=args.device,
        dtype="float32",
        draft_head_path=ckpt_path,
    )
    rows = run_config(backend_trained, "eagle-trained", prompts, args.tokens)
    all_rows.extend(rows)

    # Config 2: untrained EAGLE draft head (baseline)
    print(f"\n=== Config 2: untrained EAGLE draft head (baseline) ===")
    backend_untrained = EagleDraftBackend(
        target_model_name=args.target,
        device=args.device,
        dtype="float32",
        draft_head_path=None,  # random init
    )
    rows = run_config(backend_untrained, "eagle-untrained", prompts, args.tokens)
    all_rows.extend(rows)

    print("\n=== Summary ===")
    for r in all_rows:
        print(f"{r.config:20s}  prompt={r.prompt_len:3d}  "
              f"spec={r.spec_time_sec:.2f}s  naive={r.naive_time_sec:.2f}s  "
              f"speedup={r.speedup_wallclock:.2f}x  accept={r.acceptance_rate:.1%}  "
              f"match={r.texts_match}")

    save_results(all_rows, args.output_json, args.output_plot)


if __name__ == "__main__":
    main()
