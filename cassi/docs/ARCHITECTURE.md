# CASSI Architecture

This document describes the internal architecture of CASSI in depth. For a
quick overview, see the [main README](../README.md).

## High-level design

CASSI is organised around three principles:

1. **Backend-agnostic core.** The speculative-decoding loop, the adaptive-k
   controller, the verifier, the cross-request pipeline, and the rollback
   logic are all pure Python and depend only on the
   `BaseBackend` abstract interface. No PyTorch, no transformers, no
   tokenizer.
2. **Composable contributions.** Each of the three contributions (adaptive k,
   cross-request pipelining, smart rollback) is an independent class with a
   narrow interface. They can be enabled, disabled, or swapped individually
   without touching the engine.
3. **Reproducible defaults.** The default `ToyBackend` is deterministic given
   a seed. Tests, benchmarks, and notebooks all use a fixed seed so that
   results are reproducible across runs and machines.

## Module layout

```
cassi/
├── engine.py              # CassiEngine
├── utils.py               # Token, GenerationResult, Timer, EMA
├── backends/
│   ├── base.py            # BaseBackend (ABC) + DraftProposal + TargetScores
│   ├── toy.py             # ToyBackend — pure-Python, position-based
│   └── torch_backend.py   # TorchBackend skeleton (optional)
└── core/
    ├── draft_model.py     # DraftModel — wraps backend.draft_next
    ├── target_model.py    # TargetModel — wraps backend.target_score
    ├── verifier.py        # Verifier — parallel verification + bonus token
    ├── adaptive_k.py      # AdaptiveK — AIMD controller for k
    ├── cross_request.py   # CrossRequestPipeline — overlap across requests
    └── rollback.py        # SmartRollback — KV-cache salvage
```

## Engine loop

```
setup(prompt)
while tokens_emitted < max_new_tokens:
    k = adaptive_k.current_k()
    salvage = rollback_state.salvaged_ids[:k]
    draft_proposals = draft.propose(context, k - len(salvage))
    full_draft = salvage + draft_proposals
    scores = target.verify(context, full_draft)
    verdict = verifier.verify(full_draft, scores)
    emit(verdict.accepted_ids + verdict.bonus_token)
    adaptive_k.update(verdict.accepted_count, len(full_draft))
    rollback_state = rollback.apply(full_draft, verdict.accepted_count,
                                     verdict.rejected_at)
```

## Backend contract

A backend must implement:

```python
def setup(self, prompt: str) -> None
def reset(self) -> None
def draft_next(self, context_ids: list[int]) -> DraftProposal
def target_score(self, context_ids, candidate_ids) -> TargetScores
def encode(self, text: str) -> list[int]
def decode(self, ids: list[int]) -> str
```

The `target_score` method is the heart of speculative decoding. It must
return the target model's preferred token at each candidate position, **plus
one extra position** so the verifier can extract a bonus token when all
candidates are accepted.

## Verification semantics

Given a draft of length `k` and target scores of length `k+1`:

- If the verifier accepts tokens at positions `0..j-1` and rejects at
  position `j`, the bonus token is `scores.token_ids[j]` (the target's
  preferred token at the reject position).
- If all `k` candidates are accepted, the bonus token is
  `scores.token_ids[k]` (the target's preferred token one position beyond
  the last accepted).

This guarantees that every verification pass produces at least one emitted
token, so the engine never gets stuck.

## Adaptive-k controller

The controller is an exponential moving average over recent acceptance
rates, with an AIMD policy:

- `α ≥ 0.8`: multiplicative increase by `1.5` (clipped to `k_max`).
- `α ≤ 0.4`: multiplicative decrease by `0.5` (clipped to `k_min`).
- Otherwise: hold.

The thresholds are configurable. The analytic optimum from Leviathan et al.
is `k* ≈ α / (1 − α)`; the controller converges toward this value via
feedback rather than requiring the workload to be known in advance.

## Cross-request pipeline

The pipeline is a single-threaded *scheduler* that interleaves draft work
across requests while a target verification is in flight. The pipeline is
*not* a thread pool or async runtime — it deliberately avoids concurrency
so that the algorithm can be reasoned about and tested deterministically.

In a real deployment, the closure passed to `CrossRequestPipeline` would
itself invoke the engine; the pipeline's job is to keep draft steps for
queued requests ready by the time the target finishes verifying the current
one.

## Smart rollback

The rollback module preserves draft tokens *after* the first reject position
as a "hint" for the next draft round. The engine prepends these salvaged
tokens to the next draft and asks the draft model to fill only the
remaining `k - len(salvage)` positions.

The benefit is workload-dependent: on prompts where the draft fails at a
single rare entity or punctuation choice, CASSI avoids recomputing the
entire draft from scratch. The cost is a small amount of extra KV-cache
pressure on the target (it has to re-verify the salvaged tokens), but the
target pass is *already* paying that cost — the salvaged tokens are
verified in the same batch as the new ones.
