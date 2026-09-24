"""Tests for the speculative verifier."""

from cassi.backends.base import DraftProposal, TargetScores
from cassi.core.verifier import Verifier, VerifyMode


def _prop(tids: list[int]) -> list[DraftProposal]:
    return [DraftProposal(token_id=t, text="", logprob=-0.5) for t in tids]


def _scores(tids: list[int]) -> TargetScores:
    return TargetScores(token_ids=list(tids), logprobs=[-0.1] * len(tids))


def test_full_acceptance():
    v = Verifier()
    res = v.verify(_prop([1, 2, 3, 4]), _scores([1, 2, 3, 4, 99]))
    assert res.accepted_ids == [1, 2, 3, 4]
    assert res.accepted_count == 4
    assert res.rejected_at is None
    # bonus is the *next* position (index 4) since all candidates accepted
    assert res.bonus_token == 99


def test_immediate_reject():
    v = Verifier()
    res = v.verify(_prop([10, 2, 3]), _scores([20, 2, 3, 99]))
    assert res.accepted_ids == []
    assert res.accepted_count == 0
    assert res.rejected_at == 0
    # bonus is the target's preferred token at position 0 (the reject point)
    assert res.bonus_token == 20


def test_partial_acceptance():
    v = Verifier()
    # draft says [1,2,3,4,5], target says [1,2,9,4,5,99]
    res = v.verify(_prop([1, 2, 3, 4, 5]), _scores([1, 2, 9, 4, 5, 99]))
    assert res.accepted_ids == [1, 2]
    assert res.accepted_count == 2
    assert res.rejected_at == 2
    # bonus = target's token at reject position
    assert res.bonus_token == 9


def test_empty_candidates():
    v = Verifier()
    res = v.verify(_prop([]), _scores([42]))
    assert res.accepted_count == 0
    assert res.bonus_token == 42


def test_rejection_mode_falls_back_to_strict():
    # rejection mode without full distributions should behave as strict.
    v = Verifier(mode=VerifyMode.REJECTION)
    res = v.verify(_prop([1, 2, 3]), _scores([1, 2, 3, 4]))
    assert res.accepted_ids == [1, 2, 3]
