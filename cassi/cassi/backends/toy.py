"""Pure-Python :class:`ToyBackend` for development and testing.

The toy backend implements a *tiny* next-token predictor over a synthetic
vocabulary. It is intentionally deterministic and self-contained so that
the algorithmic core of CASSI (adaptive k, cross-request pipelining, smart
rollback) can be developed, profiled, and unit-tested without a single ML
dependency.

Behaviour
---------
- Vocabulary: the 256 byte values.
- Position-based ground truth: the "true" token at any position ``p`` in the
  sequence is computed deterministically from the prompt seed and ``p``.
  This stands in for a real transformer target model whose top-1 token the
  draft is trying to match.
- Draft model: a cheap predictor that, given the current position, returns
  the *correct* ground-truth token with probability ``draft_correctness``
  and a random *noise* token otherwise. This models a real draft model
  that is sometimes right, sometimes wrong, in a controllable way.

Tuning knobs
------------
- ``draft_correctness`` controls the probability that the draft's token
  matches the target's top-1 token. Useful for benchmarks that sweep over
  acceptance rates.
- ``target_lag_ms`` / ``draft_lag_ms`` simulate wall-clock latency of the
  two models so the cross-request pipeline controller can be exercised
  meaningfully.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from cassi.backends.base import BaseBackend, DraftProposal, TargetScores


VOCAB_SIZE = 256  # byte-level


def _position_token(seed: int, position: int) -> int:
    """Deterministic ground-truth token at ``position`` for a given ``seed``.

    The mapping is stable across calls and processes, so the target's notion
    of "the correct token at position p" is always well-defined regardless
    of how many forward passes produced it.
    """
    h = hashlib.blake2b(
        f"{seed}:{position}".encode("utf-8"), digest_size=4
    ).digest()
    return int.from_bytes(h, "big") % VOCAB_SIZE


def _noise_token(seed: int, position: int, attempt: int) -> int:
    """A token *different* from the ground truth at ``position``."""
    h = hashlib.blake2b(
        f"{seed}:{position}:noise:{attempt}".encode("utf-8"), digest_size=4
    ).digest()
    return int.from_bytes(h, "big") % VOCAB_SIZE


@dataclass
class ToyConfig:
    """Tunables for the :class:`ToyBackend`."""

    draft_correctness: float = 0.7
    target_lag_ms: float = 0.0
    draft_lag_ms: float = 0.0
    seed: int | None = None


class ToyBackend(BaseBackend):
    """Self-contained backend with no external dependencies.

    The backend is *position-based*: the ground-truth token at position
    ``p`` in the sequence is a deterministic hash of the seed and ``p``.
    The draft model returns the correct token with probability
    ``draft_correctness`` and a noise token otherwise.

    This design means:
    - The same prompt + seed always produces the same target sequence.
    - The draft's correctness is independent of position, which makes
      sweeping ``draft_correctness`` in benchmarks straightforward.
    - The target's verification pass can return ``k+1`` positions (so the
      verifier can always extract a bonus token) by scoring the candidate
      sequence plus one extra position.
    """

    name = "toy"

    def __init__(self, config: ToyConfig | None = None) -> None:
        self.config = config or ToyConfig()
        self._seed: int = 0
        self._prompt: str = ""
        self._prompt_ids: list[int] = []
        # Each draft_next() call advances a per-position RNG counter so
        # successive wrong guesses don't all collide.
        self._noise_attempts: dict[int, int] = {}

    # --- lifecycle --------------------------------------------------------
    def setup(self, prompt: str) -> None:
        if self.config.seed is not None:
            self._seed = int(self.config.seed)
        else:
            digest = hashlib.blake2b(
                prompt.encode("utf-8"), digest_size=8
            ).digest()
            self._seed = int.from_bytes(digest, "big")
        self._prompt = prompt
        self._prompt_ids = self.encode(prompt)
        self._noise_attempts.clear()

    def reset(self) -> None:
        self._seed = 0
        self._prompt = ""
        self._prompt_ids = []
        self._noise_attempts.clear()

    # --- draft model ------------------------------------------------------
    def draft_next(self, context_ids: list[int]) -> DraftProposal:
        position = len(self._prompt_ids) + len(context_ids)
        if self.config.draft_lag_ms:
            time.sleep(self.config.draft_lag_ms / 1000.0)

        # Use a cheap deterministic "roll" to decide correct vs noise.
        # The roll is keyed by (seed, position, attempt) so successive
        # draft_next() calls at the same position get fresh rolls.
        attempt = self._noise_attempts.get(position, 0)
        self._noise_attempts[position] = attempt + 1
        roll = self._roll(position, attempt)
        if roll < self.config.draft_correctness:
            token_id = _position_token(self._seed, position)
        else:
            true_tok = _position_token(self._seed, position)
            token_id = _noise_token(self._seed, position, attempt)
            # Avoid accidentally matching.
            while token_id == true_tok:
                attempt += 1
                token_id = _noise_token(self._seed, position, attempt)

        return DraftProposal(
            token_id=token_id,
            text=chr(token_id) if 32 <= token_id < 127 else "·",
            logprob=-0.5,
        )

    # --- target model -----------------------------------------------------
    def target_score(
        self, context_ids: list[int], candidate_ids: list[int]
    ) -> TargetScores:
        if self.config.target_lag_ms:
            time.sleep(self.config.target_lag_ms / 1000.0)

        base_position = len(self._prompt_ids) + len(context_ids)
        # Return k+1 positions so the verifier always has a bonus token
        # available when all candidates are accepted.
        positions = list(range(base_position, base_position + len(candidate_ids) + 1))
        token_ids = [_position_token(self._seed, p) for p in positions]
        logprobs = [-0.05] * len(token_ids)
        return TargetScores(token_ids=token_ids, logprobs=logprobs)

    # --- helpers ----------------------------------------------------------
    def _roll(self, position: int, attempt: int) -> float:
        """A deterministic uniform [0, 1) roll, independent of the truth."""
        h = hashlib.blake2b(
            f"{self._seed}:{position}:roll:{attempt}".encode("utf-8"),
            digest_size=4,
        ).digest()
        return int.from_bytes(h, "big") / (1 << 32)

    def describe(self) -> dict[str, str]:
        return {
            "backend": self.name,
            "draft_correctness": f"{self.config.draft_correctness:.2f}",
            "vocab_size": str(VOCAB_SIZE),
        }


__all__ = ["ToyBackend", "ToyConfig", "VOCAB_SIZE"]
