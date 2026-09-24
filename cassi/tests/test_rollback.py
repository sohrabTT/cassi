"""Tests for the smart rollback."""

from cassi.core.rollback import SmartRollback


def test_no_rollback_when_all_accepted():
    rb = SmartRollback()
    state = rb.apply(
        draft_ids=[1, 2, 3, 4],
        accepted_count=4,
        reject_position=None,
    )
    assert state.salvaged_ids == []
    assert state.reject_position is None


def test_salvage_tokens_after_reject():
    rb = SmartRollback(salvage_after_reject=True, max_salvage=4)
    state = rb.apply(
        draft_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        accepted_count=2,
        reject_position=2,
    )
    # tokens at indices 3..6 should be salvaged (up to max_salvage=4)
    assert state.salvaged_ids == [4, 5, 6, 7]
    assert state.reject_position == 2


def test_salvage_disabled():
    rb = SmartRollback(salvage_after_reject=False)
    state = rb.apply(
        draft_ids=[1, 2, 3, 4],
        accepted_count=2,
        reject_position=2,
    )
    assert state.salvaged_ids == []


def test_history_recorded_per_call():
    rb = SmartRollback()
    s1 = rb.apply(draft_ids=[1, 2], accepted_count=2, reject_position=None)
    s2 = rb.apply(draft_ids=[1, 2, 3], accepted_count=1, reject_position=1)
    s3 = rb.apply(draft_ids=[1], accepted_count=0, reject_position=0)
    # Each call returns a fresh state with one history entry.
    assert len(s1.history) == 1
    assert len(s2.history) == 1
    assert len(s3.history) == 1
    # The entry is (proposed, accepted)
    assert s2.history[0] == (3, 1)


def test_max_salvage_respected():
    rb = SmartRollback(max_salvage=2)
    state = rb.apply(
        draft_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        accepted_count=1,
        reject_position=1,
    )
    assert len(state.salvaged_ids) == 2


def test_reject_position_none_does_not_set():
    rb = SmartRollback()
    state = rb.apply(draft_ids=[1, 2, 3], accepted_count=3, reject_position=None)
    assert state.reject_position is None
    assert state.reject_count == 0


def test_reject_position_set():
    rb = SmartRollback()
    state = rb.apply(draft_ids=[1, 2, 3], accepted_count=1, reject_position=1)
    assert state.reject_position == 1
    assert state.reject_count == 1
