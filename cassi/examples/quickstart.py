"""Minimal end-to-end example: run CASSI on the toy backend and print stats."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from cassi import CassiEngine, ToyBackend
from cassi.backends.toy import ToyConfig


def main() -> None:
    # High draft accuracy -> high acceptance -> big speedup.
    engine = CassiEngine(backend=ToyBackend(ToyConfig(draft_correctness=0.9, seed=42)))

    result = engine.generate(prompt="The quick brown fox jumps over", max_new_tokens=64)

    print("CASSI quickstart")
    print("=" * 50)
    print(f"tokens generated : {result.tokens_generated}")
    print(f"forward passes   : {result.forward_passes}")
    print(f"draft proposed    : {result.draft_tokens_proposed}")
    print(f"draft accepted    : {result.draft_tokens_accepted}")
    print(f"acceptance rate  : {result.acceptance_rate*100:.1f}%")
    print(f"speedup vs naive : {result.speedup_vs_naive:.2f}x")
    print(f"final k          : {result.final_k}")
    print(f"engine config    : {engine.describe()}")


if __name__ == "__main__":
    main()
