"""Tests for the adaptive-k controller."""

from cassi.core.adaptive_k import AdaptiveK


def test_starts_at_k_min():
    c = AdaptiveK(k_min=2, k_max=16)
    assert c.current_k() == 2


def test_starts_at_k_init():
    c = AdaptiveK(k_min=1, k_max=16, k_init=8)
    assert c.current_k() == 8


def test_increases_when_acceptance_high():
    c = AdaptiveK(k_min=1, k_max=16, k_init=4, high_threshold=0.8)
    # Simulate high acceptance rate (8/8 accepted)
    for _ in range(5):
        c.update(accepted=8, proposed=8)
    assert c.current_k() > 4


def test_decreases_when_acceptance_low():
    c = AdaptiveK(k_min=1, k_max=16, k_init=8, low_threshold=0.4)
    # Simulate low acceptance rate (1/8)
    for _ in range(5):
        c.update(accepted=1, proposed=8)
    assert c.current_k() < 8


def test_respects_bounds():
    c = AdaptiveK(k_min=1, k_max=8, k_init=1)
    # Push k upward as much as possible
    for _ in range(20):
        c.update(accepted=8, proposed=8)
    assert c.current_k() <= 8
    # Push k downward as much as possible
    for _ in range(20):
        c.update(accepted=0, proposed=8)
    assert c.current_k() >= 1


def test_holds_in_between():
    c = AdaptiveK(k_min=1, k_max=16, k_init=4,
                  high_threshold=0.8, low_threshold=0.4)
    # Acceptance rate ~0.5 should keep k steady.
    for _ in range(5):
        c.update(accepted=4, proposed=8)
    assert c.current_k() == 4


def test_history_trace():
    c = AdaptiveK(k_min=1, k_max=16, k_init=4)
    c.update(8, 8)
    c.update(1, 8)
    h = c.history()
    assert len(h) == 2
    assert all(isinstance(row, tuple) and len(row) == 3 for row in h)
