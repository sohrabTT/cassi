"""Benchmark: throughput with cross-request pipelining vs sequential.

This script measures how cross-request pipelining (Contribution 2) changes
total wall-clock time when serving a batch of prompts. Two configurations
are compared:

- ``--mode sequential``  : serve prompts one by one (no overlap).
- ``--mode pipelined``   : route requests through the cross-request pipeline.

By default the ToyBackend has zero lag, so pipelining adds scheduler overhead
without overlap benefit. Pass ``--target-lag-ms`` and ``--draft-lag-ms`` to
simulate realistic per-pass latencies; the pipelined mode will then hide
draft-model work behind target-model verification.

Usage
-----
    python benchmarks/throughput.py
    python benchmarks/throughput.py --target-lag-ms 10 --draft-lag-ms 2
    python benchmarks/throughput.py --output results/throughput.png
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from cassi import CassiEngine, ToyBackend
from cassi.backends.toy import ToyConfig


def make_engine(target_lag_ms: float, draft_lag_ms: float) -> CassiEngine:
    return CassiEngine(
        backend=ToyBackend(
            ToyConfig(
                draft_correctness=0.8,
                target_lag_ms=target_lag_ms,
                draft_lag_ms=draft_lag_ms,
            )
        )
    )


def run_sequential(prompts: list[str], max_new_tokens: int,
                   target_lag_ms: float, draft_lag_ms: float) -> float:
    engine = make_engine(target_lag_ms, draft_lag_ms)
    t0 = time.perf_counter()
    for p in prompts:
        engine.generate(p, max_new_tokens=max_new_tokens)
    return time.perf_counter() - t0


def run_pipelined(prompts: list[str], max_new_tokens: int,
                  target_lag_ms: float, draft_lag_ms: float) -> float:
    engine = make_engine(target_lag_ms, draft_lag_ms)
    t0 = time.perf_counter()
    engine.generate_batch(prompts, max_new_tokens=max_new_tokens, pipeline=True)
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--target-lag-ms", type=float, default=0.0)
    parser.add_argument("--draft-lag-ms", type=float, default=0.0)
    parser.add_argument("--output", default="results/throughput.png")
    args = parser.parse_args()

    prompts = [f"prompt number {i:03d}" for i in range(args.num_prompts)]

    seq_t = run_sequential(prompts, args.tokens,
                          args.target_lag_ms, args.draft_lag_ms)
    pipe_t = run_pipelined(prompts, args.tokens,
                           args.target_lag_ms, args.draft_lag_ms)

    print(f"Prompts: {len(prompts)} x {args.tokens} tokens "
          f"(target_lag={args.target_lag_ms}ms, draft_lag={args.draft_lag_ms}ms)")
    print(f"Sequential:  {seq_t*1000:8.2f} ms  ({len(prompts)*args.tokens/seq_t:.1f} tok/s)")
    print(f"Pipelined:   {pipe_t*1000:8.2f} ms  ({len(prompts)*args.tokens/pipe_t:.1f} tok/s)")
    print(f"Throughput ratio: {seq_t/pipe_t:.2f}x")

    # Optional plot
    try:
        import matplotlib.pyplot as plt

        labels = ["Sequential", "Pipelined"]
        times = [seq_t * 1000, pipe_t * 1000]
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.bar(labels, times, color=["#888", "#1f77b4"])
        ax.set_ylabel("Wall-clock time (ms)")
        ax.set_title(
            f"Throughput on {len(prompts)} prompts × {args.tokens} tokens\n"
            f"(target_lag={args.target_lag_ms}ms, draft_lag={args.draft_lag_ms}ms)"
        )
        for i, t in enumerate(times):
            ax.text(i, t + max(times) * 0.02, f"{t:.1f} ms", ha="center")
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        fig.savefig(args.output, dpi=120)
        print(f"Plot saved to {args.output}")
    except ImportError:
        print("matplotlib not installed; skipping plot")


if __name__ == "__main__":
    main()
