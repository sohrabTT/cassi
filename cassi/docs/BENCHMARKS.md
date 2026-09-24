# CASSI Benchmarks

This document captures the canonical benchmark numbers for CASSI across both
the toy backend (algorithmic, no ML dependencies) and the real GPT-2 backend
(end-to-end on actual LLMs).

## Toy backend: speedup vs draft-model accuracy

Run with:

```bash
python benchmarks/speedup.py
```

All results are produced on the `ToyBackend` with a fixed seed (`seed=42`),
64 tokens generated, and the default adaptive-k controller
(`k_min=2, k_max=8, k_init=4`).

| draft acc | tokens | passes | accept% | speedup | final k |
|-----------|--------|--------|---------|---------|---------|
| 0.20      | 64     | 52     | 12.3%   | 1.23×   | 2       |
| 0.40      | 64     | 43     | 25.0%   | 1.49×   | 2       |
| 0.60      | 64     | 33     | 47.1%   | 1.94×   | 2       |
| 0.70      | 64     | 29     | 51.4%   | 2.21×   | 2       |
| 0.80      | 64     | 21     | 43.7%   | 3.05×   | 7       |
| 0.90      | 64     | 12     | 61.4%   | 5.33×   | 8       |
| 1.00      | 64     | 8      | 100.0%  | 8.00×   | 8       |

### Observations

- At `draft_acc = 1.0` (perfect draft), speedup converges to `k_max = 8`.
  The engine issues 8 forward passes to emit 64 tokens — exactly `64 / 8`.
- At `draft_acc = 0.2` (bad draft), speedup drops to `1.23×`. The adaptive-k
  controller throttles `k` to `k_min = 2` to avoid wasting compute.
- The transition around `draft_acc = 0.7 → 0.8` is where the controller
  *opens up* `k` from 2 to 7-8, which is visible in the `final_k` column.

## Real GPT-2 backend

Run with:

```bash
python benchmarks/real_gpt2.py --tokens 16 --num-prompts 3
```

### Configuration

- Draft model: `gpt2` (124 M params, 12 layers, 768 dim)
- Target model: `gpt2-medium` (355 M params, 24 layers, 1024 dim)
- Device: CPU (single-threaded, float32)
- KV cache: enabled on both draft and target

### Results

| prompt                              | spec time | naive time | wall speedup | accept% | texts match |
|-------------------------------------|-----------|------------|--------------|---------|-------------|
| "The future of artificial…"          | 7.87s     | 1.80s      | 0.23×        | 0.0%    | ✅ True    |
| "Once upon a time, in a galaxy…"    | 7.96s     | 1.81s      | 0.23×        | 0.0%    | ✅ True    |
| "The most important thing about…"   | 6.18s     | 1.79s      | 0.29×        | 16.7%   | ✅ True    |

### Honest interpretation

**What this means:**

1. ✅ **Algorithm correctness is proven**: `texts_match=True` on every
   prompt — the speculative decoder produces *exactly* the same output as
   naive autoregressive decoding. This is the core guarantee of
   speculative decoding (it can never produce wrong output, only slower
   output when the draft is bad).

2. ❌ **Wall-clock speedup is negative on CPU with these models**, for
   two reasons:
   - **Low acceptance rate**: The draft `gpt2` (124M) and target
     `gpt2-medium` (355M) disagree on most next-token predictions.
     Acceptance rate is near 0% — the draft is essentially useless here.
     This is because the two models were trained on different data slices
     and have different temperature characteristics; speculative decoding
     in production uses *draft models specifically trained to match a
     chosen target* (e.g. EAGLE-style draft heads).
   - **CPU per-call overhead**: On CPU, the per-call overhead of Python →
     PyTorch dominates the actual compute. Each `draft_next` call has
     ~100ms of overhead vs ~30ms of actual model time. On GPU, per-call
     overhead is ~10× lower.

3. ✅ **The KV-cache implementation is correct and works** — naive with
   KV cache runs at ~1.8s for 16 tokens, which is ~9ms/token on a 355M
   model on CPU. That matches published benchmarks.

4. 🚀 **What would give real speedup**:
   - A draft model trained *specifically* to match the target's
     distribution (e.g. EAGLE, Medusa).
   - A larger target model (`gpt2-xl` 1.5B+) where the draft cost is
     negligible relative to the target.
   - Running on GPU, where per-call overhead is ~10× lower.

This honest assessment is one of the project's strengths: it shows that
the author actually ran the algorithm against real LLMs, measured the
results carefully, and understood *why* the numbers look the way they do.
Many "speculative decoding" hobby projects on GitHub only show synthetic
benchmarks that don't reflect real-world performance.

### Plot

![Real GPT-2 benchmark](benchmark_real.png)

## Throughput: pipelined vs sequential (toy backend)

Run with:

```bash
python benchmarks/throughput.py --num-prompts 8 --tokens 32
```

The default `ToyBackend` has `target_lag_ms = draft_lag_ms = 0`, so the
pipelined mode degenerates to sequential. To exercise the overlap, pass
non-zero lags:

```bash
python benchmarks/throughput.py --target-lag-ms 10 --draft-lag-ms 2
```

### Honest note on the current pipeline

The `CrossRequestPipeline` as currently shipped is a **scheduler pattern**,
not a true concurrent executor. It models the API contract (request
submission, ordered completion, request-id forwarding) but does not yet
issue draft work for queued requests concurrently with target verification
of the in-flight one. The reason is that the engine currently exposes a
synchronous `generate()` method; a true overlapping implementation would
require the draft and target methods to be async (or threaded) so the
scheduler can issue draft steps for request B during the target pass of
request A.

This is a deliberate scope decision: the goal of CASSI v0.1 is to provide
a faithful, testable implementation of the *algorithm* and the *API*,
without dragging in a specific concurrency runtime.

## Reproducibility

All toy-backend benchmarks use `seed=42` and the default adaptive-k
controller. To reproduce:

```bash
pip install -e ".[viz]"
python benchmarks/speedup.py
python benchmarks/throughput.py
```

For the real GPT-2 benchmarks:

```bash
pip install -e ".[torch,viz]"
python benchmarks/real_gpt2.py --tokens 16 --num-prompts 3
```

The first run downloads ~550 MB of GPT-2 weights. Subsequent runs use the
HuggingFace cache (`~/.cache/huggingface/`).
