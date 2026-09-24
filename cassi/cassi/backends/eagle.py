"""EAGLE-style draft head for high-acceptance speculative decoding.

This is a simpler, more honest implementation: we use the target model itself
for the draft predictions, by running it on the cached prefix and using its
own logits. The draft head is then *trained* to mimic the target's
distribution from the target's hidden states, but during speculative
decoding we always run the target model (with KV cache) for both draft and
verification, getting the same correctness guarantees as naive autoregressive
decoding.

Why this is correct
-------------------
The original EAGLE paper uses a *separate* draft head that autoregressively
generates k tokens from the target's last hidden state. This is faster than
running the target, but it requires careful hidden-state tracking during
rollback. Our implementation here uses a simpler design:

- The draft head predicts what the target's next-token argmax would be,
  given the target's current last hidden state.
- The verifier always runs the target to check.
- The draft head's job is to make the verifier accept as many tokens as
  possible; it's a *predictor of the target's argmax*, not a separate
  language model.

Architecture
------------
The draft head is a 2-layer MLP: hidden_dim -> hidden_dim -> vocab_size.

Training
--------
Cross-entropy on (hidden_state[t], target_argmax[t+1]) pairs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from cassi.backends.base import BaseBackend, DraftProposal, TargetScores
from cassi.utils import logger


class EagleDraftHead(nn.Module):
    """A 2-layer MLP that predicts next-token logits from a hidden state.

    Parameters
    ----------
    hidden_dim:
        Dimension of the input hidden state (e.g. 768 for GPT-2 small).
    vocab_size:
        Size of the output vocabulary (e.g. 50257 for GPT-2).
    intermediate_dim:
        Width of the hidden layer. Default ``hidden_dim``.
    dropout:
        Dropout probability during training. Default 0.0.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        intermediate_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        intermediate_dim = intermediate_dim or hidden_dim
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        x = self.fc1(hidden_state)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x)


class EagleDraftBackend(BaseBackend):
    """EAGLE-style draft head + target model.

    The draft head consumes the target model's last hidden state and
    predicts the next-token argmax. Because it sees the *same* hidden state
    the target uses, a well-trained head has very high acceptance rate
    (typically 70-90%+ on real workloads).

    Correctness
    ------------
    The draft head's prediction is *advisory* only. The target model's
    forward pass is the source of truth, so speculative decoding with this
    backend can never produce wrong output - only slower output when the
    draft is wrong.

    Parameters
    ----------
    target_model_name:
        HuggingFace model id for the target model. Default ``gpt2-medium``.
    device, dtype:
        Standard PyTorch device/dtype.
    draft_head_path:
        Path to a trained draft head checkpoint. If not provided, the head
        is randomly initialised (acceptance rate ~0% until trained).
    """

    name = "eagle"

    def __init__(
        self,
        target_model_name: str = "gpt2-medium",
        device: str = "cpu",
        dtype: str = "float32",
        draft_head_path: str | Path | None = None,
    ) -> None:
        self.target_model_name = target_model_name
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)

        logger.info(
            "EagleDraftBackend: loading target=%s device=%s",
            target_model_name, device,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._target = AutoModelForCausalLM.from_pretrained(
            target_model_name, torch_dtype=self.dtype,
        ).to(self.device).eval()

        hidden_dim = self._target.config.n_embd
        vocab_size = self._target.config.vocab_size
        self._draft_head = EagleDraftHead(
            hidden_dim=hidden_dim, vocab_size=vocab_size,
        ).to(self.device).to(self.dtype)

        if draft_head_path is not None:
            path = Path(draft_head_path)
            if path.exists():
                logger.info("Loading draft head from %s", path)
                state = torch.load(path, map_location=self.device)
                self._draft_head.load_state_dict(state)
                self._draft_head.eval()
            else:
                logger.warning("Draft head checkpoint %s not found; using random init.", path)
        else:
            logger.warning("No draft head checkpoint; using random init (acceptance will be ~0%%).")

        logger.info(
            "EagleDraftBackend ready: target params=%.1fM head params=%.1fM (%.2f%% of target)",
            sum(p.numel() for p in self._target.parameters()) / 1e6,
            sum(p.numel() for p in self._draft_head.parameters()) / 1e6,
            100 * sum(p.numel() for p in self._draft_head.parameters())
            / sum(p.numel() for p in self._target.parameters()),
        )

        # Per-request state
        self._prompt_ids: list[int] = []
        self._target_past: Any = None
        self._target_ctx_len: int = 0
        self._last_hidden: torch.Tensor | None = None

    # --- lifecycle --------------------------------------------------------
    def setup(self, prompt: str) -> None:
        self._prompt_ids = self.encode(prompt)
        ids = torch.tensor(
            [self._prompt_ids], dtype=torch.long, device=self.device
        )
        with torch.inference_mode():
            out = self._target(input_ids=ids, use_cache=True, output_hidden_states=True)
            self._target_past = out.past_key_values
            self._target_ctx_len = len(self._prompt_ids)
            self._last_hidden = out.hidden_states[-1][:, -1, :]  # (1, hidden_dim)

    def reset(self) -> None:
        self._prompt_ids = []
        self._target_past = None
        self._target_ctx_len = 0
        self._last_hidden = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # --- draft model ------------------------------------------------------
    @torch.inference_mode()
    def draft_next(self, context_ids: list[int]) -> DraftProposal:
        """Predict the next token.

        This implementation runs the target model on the cached prefix to
        get the hidden state at the last committed position, then applies
        the draft head. This is the *correct* EAGLE design: the draft head
        sees the hidden state for the actual context, not for any draft
        candidate that may not be accepted.
        """
        # Reconcile cache state.
        if len(context_ids) < self._target_ctx_len:
            self._target_past = self._truncate_past(
                self._target_past, len(context_ids)
            )
            self._target_ctx_len = len(context_ids)
            self._last_hidden = None
        elif len(context_ids) > self._target_ctx_len:
            new_ids = context_ids[self._target_ctx_len:]
            new_tensor = torch.tensor(
                [new_ids], dtype=torch.long, device=self.device
            )
            out = self._target(
                input_ids=new_tensor,
                past_key_values=self._target_past,
                use_cache=True,
                output_hidden_states=True,
            )
            self._target_past = out.past_key_values
            self._target_ctx_len = len(context_ids)
            self._last_hidden = out.hidden_states[-1][:, -1, :]

        # Re-derive hidden state if invalidated.
        if self._last_hidden is None and len(context_ids) > 0:
            self._target_past = self._truncate_past(
                self._target_past, len(context_ids) - 1
            )
            self._target_ctx_len = len(context_ids) - 1
            last_tensor = torch.tensor(
                [[context_ids[-1]]], dtype=torch.long, device=self.device
            )
            out = self._target(
                input_ids=last_tensor,
                past_key_values=self._target_past,
                use_cache=True,
                output_hidden_states=True,
            )
            self._target_past = out.past_key_values
            self._target_ctx_len = len(context_ids)
            self._last_hidden = out.hidden_states[-1][:, -1, :]

        if self._last_hidden is None:
            return DraftProposal(token_id=0, text="", logprob=-100.0)

        # Use draft head to predict. The head is trained to match the
        # target's argmax from the target's hidden state.
        logits = self._draft_head(self._last_hidden)  # (1, vocab)
        log_probs = F.log_softmax(logits, dim=-1)
        token_id = torch.argmax(logits, dim=-1)[0].item()
        score = log_probs[0, token_id].item()
        return DraftProposal(
            token_id=token_id,
            text=self._tokenizer.decode([token_id]),
            logprob=score,
        )

    # --- target model -----------------------------------------------------
    @torch.inference_mode()
    def target_score(
        self, context_ids: list[int], candidate_ids: list[int]
    ) -> TargetScores:
        """Run target on (last_context_token + candidates) in one forward pass.

        Returns the target's preferred token at each candidate position
        (length k), plus one extra bonus token (length k+1).

        IMPORTANT: After this call, the target's KV cache will contain
        ``context_ids + candidate_ids``. The engine's view of the context
        may be different (only the accepted candidates). The next
        ``draft_next()`` call will reconcile the cache by truncating back
        to ``len(context_ids + accepted)`` if needed.
        """
        k = len(candidate_ids)
        if k == 0:
            return TargetScores(token_ids=[], logprobs=[])

        # Reconcile cache with the engine's context.
        if len(context_ids) < self._target_ctx_len:
            self._target_past = self._truncate_past(
                self._target_past, len(context_ids)
            )
            self._target_ctx_len = len(context_ids)
            self._last_hidden = None
        elif len(context_ids) > self._target_ctx_len:
            new_ids = context_ids[self._target_ctx_len:]
            new_tensor = torch.tensor(
                [new_ids], dtype=torch.long, device=self.device
            )
            out = self._target(
                input_ids=new_tensor,
                past_key_values=self._target_past,
                use_cache=True,
                output_hidden_states=True,
            )
            self._target_past = out.past_key_values
            self._target_ctx_len = len(context_ids)
            self._last_hidden = out.hidden_states[-1][:, -1, :]

        # Truncate cache to len(context) - 1 so we can re-feed the last
        # context token together with all candidates in a single forward
        # pass. This gives us logits at positions 0..k.
        if len(context_ids) >= 1:
            self._target_past = self._truncate_past(
                self._target_past, len(context_ids) - 1
            )
            self._target_ctx_len = len(context_ids) - 1

            combined = [context_ids[-1]] + list(candidate_ids)
            combined_tensor = torch.tensor(
                [combined], dtype=torch.long, device=self.device
            )
            out = self._target(
                input_ids=combined_tensor,
                past_key_values=self._target_past,
                use_cache=True,
                output_hidden_states=True,
            )
            # Cache now contains context_ids + candidate_ids (full draft).
            # The next draft_next() call will truncate this to match what
            # the engine actually accepted.
            self._target_past = out.past_key_values
            self._target_ctx_len = len(context_ids) + k
            self._last_hidden = out.hidden_states[-1][:, -1, :]

            logits = out.logits[0]  # (k+1, vocab)
        else:
            cand_tensor = torch.tensor(
                [candidate_ids], dtype=torch.long, device=self.device
            )
            out = self._target(
                input_ids=cand_tensor,
                past_key_values=self._target_past,
                use_cache=True,
                output_hidden_states=True,
            )
            self._target_past = out.past_key_values
            self._target_ctx_len = k
            self._last_hidden = out.hidden_states[-1][:, -1, :]
            logits = out.logits[0]
            logits = torch.cat([logits, logits[-1:]], dim=0)

        log_probs = F.log_softmax(logits, dim=-1)
        target_token_ids: list[int] = []
        target_logprobs: list[float] = []
        for i in range(k + 1):
            tok = torch.argmax(logits[i]).item()
            target_token_ids.append(tok)
            target_logprobs.append(log_probs[i, tok].item())

        return TargetScores(
            token_ids=target_token_ids, logprobs=target_logprobs
        )

    # --- helpers ----------------------------------------------------------
    def _truncate_past(self, past: Any, new_len: int) -> Any:
        if past is None:
            return None
        if hasattr(past, "crop"):
            try:
                current_len = self._cache_seq_len(past)
                to_remove = current_len - new_len
                if to_remove > 0:
                    past.crop(-to_remove)
                return past
            except Exception:
                pass
        truncated = []
        for layer in past:
            k, v = layer
            truncated.append((k[..., :new_len, :], v[..., :new_len, :]))
        return tuple(truncated)

    @staticmethod
    def _cache_seq_len(past: Any) -> int:
        if hasattr(past, "get_seq_length"):
            return past.get_seq_length()
        try:
            return past[0][0].shape[2]
        except Exception:
            return 0

    # --- tokenizer shim ---------------------------------------------------
    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    # --- introspection ----------------------------------------------------
    def describe(self) -> dict[str, str]:
        return {
            "backend": self.name,
            "target": self.target_model_name,
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "kv_cache": "enabled",
            "draft_head": "eagle-mlp",
            "draft_head_params": f"{sum(p.numel() for p in self._draft_head.parameters())/1e6:.1f}M",
        }


__all__ = ["EagleDraftHead", "EagleDraftBackend"]
