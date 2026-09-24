"""Benchmark: sweep draft_correctness and measure speedup vs naive.

This script runs the CASSI engine across a range of draft-model accuracies
and reports the resulting speedup, acceptance rate, and forward-pass count.
It produces both a textual table and (optionally) a matplotlib chart saved
to ``results/benchmark_speedup.png``.

Usage
-----
    python benchmarks/speedup.py
    python benchmarks/speedup.py --output results/speedup.png
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

# Allow running the script from the repo root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from cassi import CassiEngine, ToyBackend
from cassi.backends.toy import ToyConfig


@dataclass
class Row:
    draft_correctness: float
    tokens_generated: int
    forward_passes: int
    acceptance_rate: float
    speedup: float
    final_k: int


def run_sweep(prompt: str, max_new_tokens: int, accuracies: list[float]) -> list[Row]:
    rows: list[Row] = []
    for acc in accuracies:
        engine = CassiEngine(
            backend=ToyBackend(ToyConfig(draft_correctness=acc, seed=42)),
        )
        r = engine.generate(prompt=prompt, max_new_tokens=max_new_tokens)
        rows.append(
            Row(
                draft_correctness=acc,
                tokens_generated=r.tokens_generated,
                forward_passes=r.forward_passes,
                acceptance_rate=r.acceptance_rate,
                speedup=r.speedup_vs_naive,
                final_k=r.final_k,
            )
        )
    return rows


def print_table(rows: list[Row]) -> None:
    header = f"{'draft_acc':>10} {'tokens':>8} {'passes':>8} {'accept%':>9} {'speedup':>9} {'k_final':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.draft_correctness:>10.2f} "
            f"{r.tokens_generated:>8d} "
            f"{r.forward_passes:>8d} "
            f"{r.acceptance_rate*100:>8.1f}% "
            f"{r.speedup:>8.2f}x "
            f"{r.final_k:>8d}"
        )


def plot(rows: list[Row], output: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return
    xs = [r.draft_correctness for r in rows]
    speedups = [r.speedup for r in rows]
    accept = [r.acceptance_rate for r in rows]

    fig, ax1 = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax1.plot(xs, speedups, "o-", color="#1f77b4", label="Speedup vs naive")
    ax1.set_xlabel("Draft model accuracy")
    ax1.set_ylabel("Speedup (×)", color="#1f77b4")
    ax1.set_ylim(0, max(speedups) * 1.2 + 0.5)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(xs, accept, "s--", color="#d62728", label="Acceptance rate")
    ax2.set_ylabel("Acceptance rate", color="#d62728")
    ax2.set_ylim(0, 1.05)

    ax1.set_title("CASSI: speculative decoding speedup vs draft accuracy")
    fig.savefig(output, dpi=120)
    print(f"Plot saved to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="The quick brown fox jumps over")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument(
        "--accuracies",
        nargs="+",
        type=float,
        default=[0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    parser.add_argument("--output", default="results/benchmark_speedup.png")
    args = parser.parse_args()

    rows = run_sweep(args.prompt, args.tokens, args.accuracies)
    print_table(rows)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plot(rows, args.output)


if __name__ == "__main__":
    main()
