"""Contribution 1: Adaptive draft length (k) controller.

Vanilla speculative decoding fixes the draft length ``k`` (typically 4 or 8)
regardless of workload. This is suboptimal: when the draft model is confident
(acceptance rate is high) we *waste* an opportunity by stopping the draft
early; when the draft model is lost (acceptance rate is low) we waste compute
producing tokens that are certain to be rejected.

CASSI ships an online controller that tracks the recent acceptance rate via
an exponential moving average (EMA) and adjusts ``k`` multiplicatively.
This is inspired by additive-increase/multiplicative-decrease (AIMD) in TCP
congestion control, and is the simplest provably-stable policy for a
feedback signal that is monotone in the controlled variable.

Reference
---------
S. Leviathan et al., "Fast Inference from Transformers via Speculative
Decoding" (2023) notes that the optimal ``k`` is a function of the
acceptance rate ``α``: ``k* ≈ α / (1 - α)``. Our controller converges
toward this value via feedback rather than an analytic formula.
"""

from __future__ import annotations

from cassi.utils import EMA, logger


class AdaptiveK:
    """Online controller for the draft length ``k``.

    Parameters
    ----------
    k_min, k_max:
        Bounds on ``k``. Defaults of 1 and 16 cover typical use.
    k_init:
        Initial value of ``k``. Defaults to ``k_min``.
    ema_alpha:
        Smoothing factor for the acceptance-rate EMA. Higher means more
        reactive to recent changes.
    inc_factor:
        Multiplicative factor used when acceptance rate is high (default 1.5).
    dec_factor:
        Multiplicative factor used when acceptance rate is low (default 0.5).
    high_threshold, low_threshold:
        Acceptance-rate thresholds above which ``k`` increases and below
        which ``k`` decreases, respectively.
    """

    def __init__(
        self,
        k_min: int = 1,
        k_max: int = 16,
        k_init: int | None = None,
        ema_alpha: float = 0.3,
        inc_factor: float = 1.5,
        dec_factor: float = 0.5,
        high_threshold: float = 0.8,
        low_threshold: float = 0.4,
    ) -> None:
        if k_min < 1:
            raise ValueError("k_min must be >= 1")
        if k_max <= k_min:
            raise ValueError("k_max must be > k_min")
        self.k_min = k_min
        self.k_max = k_max
        self.k = k_init if k_init is not None else k_min
        self.ema = EMA(ema_alpha)
        self.inc_factor = inc_factor
        self.dec_factor = dec_factor
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self._history: list[tuple[int, float, int]] = []  # (k_used, alpha, k_next)

    def current_k(self) -> int:
        return self.k

    def update(self, accepted: int, proposed: int) -> int:
        """Feed back the result of the last verification pass.

        Returns the new value of ``k`` to use for the next draft.
        """
        if proposed <= 0:
            return self.k
        alpha = accepted / proposed
        smoothed = self.ema.update(alpha)
        old_k = self.k
        if smoothed >= self.high_threshold:
            self.k = max(self.k_min, min(self.k_max, int(self.k * self.inc_factor) + 1))
        elif smoothed <= self.low_threshold:
            self.k = max(self.k_min, int(self.k * self.dec_factor))
        # else: hold steady
        if self.k != old_k:
            logger.debug(
                "adaptive-k: alpha=%.2f ema=%.2f k %d -> %d",
                alpha, smoothed, old_k, self.k,
            )
        self._history.append((old_k, alpha, self.k))
        return self.k

    def history(self) -> list[tuple[int, float, int]]:
        """Return the (k_used, alpha, k_next) trace; useful for benchmarks."""
        return list(self._history)


__all__ = ["AdaptiveK"]
