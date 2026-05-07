"""Token-level log-probability utilities for causal language models.

Provides ``LogProbOutputs`` and helpers for computing next-token log p,
its mean, and its standard deviation given pre-computed logits and labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LogProbOutputs:
    # All tensors are shape [B, T-1] aligned to predicting labels[:, 1:]
    token_logp: torch.Tensor  # log p(x_t | x_<t)
    mu: torch.Tensor  # E[log p(z|prefix)]
    sigma: torch.Tensor  # std of log p(z|prefix)
    # optional raw
    logits: Optional[torch.Tensor] = None  # [B, T, V]


def next_token_logprobs_and_stats(logits: torch.Tensor, labels: torch.Tensor) -> LogProbOutputs:
    """Compute next-token log probs plus per-position mu/sigma over the vocab.

    Args:
      logits: [B, T, V], where logits[:, t] predicts token at position t+1.
      labels: [B, T] input_ids; we score labels[:, 1:] using logits[:, :-1].

    Returns:
      token_logp, mu, sigma: [B, T-1]
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected logits [B,T,V], got {tuple(logits.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"Expected labels [B,T], got {tuple(labels.shape)}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(f"logits shape {tuple(logits.shape)} incompatible with labels {tuple(labels.shape)}")

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)  # [B, T-1, V]
    # Gather ground-truth log-prob
    token_logp = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    probs = log_probs.exp()
    mu = (probs * log_probs).sum(dim=-1)  # [B, T-1]
    var = (probs * (log_probs - mu.unsqueeze(-1)) ** 2).sum(dim=-1)  # [B, T-1]
    sigma = torch.sqrt(var + 1e-12)

    return LogProbOutputs(token_logp=token_logp, mu=mu, sigma=sigma, logits=logits)


def argmax_next_token(logits: torch.Tensor) -> torch.Tensor:
    """Return argmax next-token predictions for each position.

    Args:
      logits: [B, T, V]
    Returns:
      argmax_ids: [B, T-1] where position t corresponds to argmax for token t+1 given prefix up to t.
    """
    shift_logits = logits[:, :-1, :]
    return shift_logits.argmax(dim=-1)
