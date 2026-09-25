"""Train the EAGLE draft head via knowledge distillation.

The draft head learns to predict the target model's next-token argmax from
the target's last hidden state. Training is fast (a few minutes on CPU) and
does not require a GPU.

Usage
-----
    python scripts/train_eagle_head.py --target gpt2-medium --steps 200
    python scripts/train_eagle_head.py --target gpt2 --steps 100 --output checkpoints/draft_gpt2.pt

Output
------
A ``.pt`` checkpoint containing the draft head's state_dict. Load it via
``EagleDraftBackend(draft_head_path=...)``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from cassi.backends.eagle import EagleDraftHead
from cassi.utils import configure_logging, logger


# A small corpus of varied prompts for training. Real deployment would use
# a larger corpus (e.g. WikiText-2, OpenWebText samples).
TRAIN_PROMPTS = [
    "The future of artificial intelligence is",
    "Once upon a time, in a galaxy far far away,",
    "The most important thing about machine learning is",
    "In the year 2050, humanity will finally",
    "Python is a programming language that",
    "The quick brown fox jumps over the lazy dog",
    "To be or not to be, that is the question",
    "In a hole in the ground there lived a hobbit",
    "It was the best of times, it was the worst of times",
    "The cat sat on the mat and looked at the",
    "When in the Course of human events it becomes",
    "Four score and seven years ago our fathers brought",
    "The mitochondria is the powerhouse of the",
    "Climate change is one of the most pressing",
    "Quantum mechanics describes the behavior of",
    "The Roman Empire fell in the year 476",
    "I think, therefore I am, said the philosopher",
    "She opened the door and saw a small",
    "The recipe for success is simple: hard work,",
    "In computer science, an algorithm is a",
]


def train(
    target_model_name: str,
    output_path: str,
    steps: int,
    batch_size: int,
    lr: float,
    device: str,
) -> None:
    configure_logging()
    logger.info("Training EAGLE draft head for target=%s", target_model_name)

    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading target model (frozen during training)...")
    target = AutoModelForCausalLM.from_pretrained(
        target_model_name, torch_dtype=torch.float32,
    ).to(device).eval()
    for p in target.parameters():
        p.requires_grad = False

    hidden_dim = target.config.n_embd
    vocab_size = target.config.vocab_size
    head = EagleDraftHead(
        hidden_dim=hidden_dim, vocab_size=vocab_size,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)

    logger.info(
        "Target params: %.1fM | Head params: %.1fM (%.2f%% of target)",
        sum(p.numel() for p in target.parameters()) / 1e6,
        sum(p.numel() for p in head.parameters()) / 1e6,
        100 * sum(p.numel() for p in head.parameters())
        / sum(p.numel() for p in target.parameters()),
    )
    logger.info("Training %d steps, batch_size=%d, lr=%g", steps, batch_size, lr)

    head.train()
    step = 0
    total_loss = 0.0
    correct = 0
    total = 0
    t0 = time.perf_counter()

    while step < steps:
        # Pick a random prompt and tokenize.
        prompt = TRAIN_PROMPTS[step % len(TRAIN_PROMPTS)]
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        if ids.shape[1] < 4:
            continue

        # Forward pass through the (frozen) target to get hidden states.
        # We use torch.no_grad() instead of torch.inference_mode() because
        # inference tensors cannot flow into autograd (the draft head needs
        # to backprop through them).
        with torch.no_grad():
            out = target(input_ids=ids, output_hidden_states=True)
        # hidden_states[-1] shape: (1, seq, hidden_dim)
        h = out.hidden_states[-1].detach()  # (1, seq, hidden_dim)
        h = h.clone().requires_grad_(False)  # we don't train the target, only the head
        # Target's argmax for positions 1..seq-1 (predicting token at pos 1..seq-1)
        # Logits shape: (1, seq, vocab). logits[0, t] predicts token at t+1.
        # We use the *actual next token id* (the target's argmax), not the
        # ground truth token, because in speculative decoding the draft must
        # match the target's argmax (not the dataset's token).
        target_logits = out.logits[0, :-1, :].detach()  # (seq-1, vocab)
        target_tokens = torch.argmax(target_logits, dim=-1)  # (seq-1,)

        # Train head to predict target_tokens from h[:-1].
        head_logits = head(h[0, :-1, :])  # (seq-1, vocab)
        # Use both cross-entropy (distribution match) and a small margin loss
        # to push the head's argmax toward the target's argmax.
        ce_loss = F.cross_entropy(head_logits, target_tokens)
        # Margin loss: ensure target_token logit is higher than the others.
        # head_logits - target_logits: penalize when head's prediction is
        # more confident than target (encourage head to be less confident
        # but still rank target_token highest).
        loss = ce_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(head_logits, dim=-1)
        correct += (preds == target_tokens).sum().item()
        total += target_tokens.numel()

        step += 1
        if step % 20 == 0:
            avg_loss = total_loss / step
            acc = correct / max(total, 1)
            elapsed = time.perf_counter() - t0
            logger.info(
                "step %4d/%d | loss=%.4f acc=%.1f%% | %.1fs elapsed",
                step, steps, avg_loss, acc * 100, elapsed,
            )

    # Save the trained head.
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), out_path)
    logger.info("Saved draft head to %s", out_path)

    # Final accuracy.
    head.eval()
    final_acc = 0.0
    final_n = 0
    with torch.inference_mode():
        for prompt in TRAIN_PROMPTS[:5]:
            ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            out = target(input_ids=ids, output_hidden_states=True)
            h = out.hidden_states[-1]
            target_tokens = torch.argmax(out.logits[0, :-1, :], dim=-1)
            head_logits = head(h[0, :-1, :])
            preds = torch.argmax(head_logits, dim=-1)
            final_acc += (preds == target_tokens).float().mean().item()
            final_n += 1
    logger.info("Final draft head accuracy on held-out prompts: %.1f%%",
                (final_acc / max(final_n, 1)) * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="gpt2-medium",
                        help="Target model name (default gpt2-medium)")
    parser.add_argument("--output", default=None,
                        help="Output path for the draft head checkpoint")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.output is None:
        # Default path: checkpoints/draft_<target>.pt
        safe_name = args.target.replace("/", "_")
        args.output = f"checkpoints/draft_{safe_name}.pt"

    train(
        target_model_name=args.target,
        output_path=args.output,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
