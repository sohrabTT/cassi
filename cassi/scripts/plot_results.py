"""Standalone script to generate the architecture diagram used in the README.

We avoid a heavy dependency on graphviz; instead we use matplotlib to draw
a clean, publication-style block diagram of the CASSI pipeline.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))


def main(output: str = "docs/architecture.png") -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("matplotlib not installed; skipping architecture diagram")
        return

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def block(x: float, y: float, w: float, h: float, text: str, color: str) -> None:
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05",
            linewidth=1.2, edgecolor="#222", facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=10, family="DejaVu Sans",
        )

    # Top row: draft + target
    block(0.5, 4.5, 2.5, 1.0, "Draft Model\n(small, fast)", "#e8f0fe")
    block(8.0, 4.5, 2.5, 1.0, "Target Model\n(large, slow)", "#fce8e8")

    # Middle row: verifier + adaptive-k + rollback
    block(0.5, 2.5, 2.5, 1.0, "Adaptive-K\nController", "#fef7e0")
    block(4.0, 2.5, 3.0, 1.0, "Verifier\n(parallel, batched)", "#e6f4ea")
    block(8.0, 2.5, 2.5, 1.0, "Smart Rollback\n(KV salvage)", "#f3e8fd")

    # Bottom row: engine + cross-request pipeline
    block(2.5, 0.5, 6.0, 1.0, "CassiEngine\n(orchestrates the loop)", "#fff3e0")

    # Arrows
    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#444", lw=1.2),
        )

    arrow(3.0, 5.0, 4.0, 3.5)   # draft -> verifier
    arrow(8.0, 5.0, 7.0, 3.5)   # target -> verifier
    arrow(1.75, 4.5, 1.75, 3.5)  # adaptive-k self loop (visual)
    arrow(1.75, 2.5, 4.0, 2.0)   # adaptive-k -> engine (label via text)
    arrow(5.5, 2.5, 5.5, 1.5)   # verifier -> engine
    arrow(9.25, 2.5, 7.0, 1.5)   # rollback -> engine
    arrow(8.0, 5.5, 4.5, 5.5)   # prompt-flow arrow top
    ax.text(6.0, 5.7, "draft proposes k tokens →", ha="center", fontsize=9, color="#555")
    ax.text(6.0, 4.0, "target verifies all k in 1 pass", ha="center", fontsize=9, color="#555")

    ax.set_title("CASSI Architecture: Confidence-Adaptive Speculative Inference", fontsize=13)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=120, bbox_inches=None)
    print(f"Architecture diagram saved to {output}")


if __name__ == "__main__":
    main()
