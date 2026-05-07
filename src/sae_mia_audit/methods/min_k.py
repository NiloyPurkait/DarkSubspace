# ---------------------------------------------------------------------------
# Source repositories:
#   Min-K% Prob: Shi et al., "Detecting Pretraining Data from Large Language
#       Models" (ICLR 2024).  https://github.com/swj0419/detect-pretrain-code
#   Min-K%++: Zhang et al., "Min-K%++: Improved Baseline for Detecting
#       Pre-Training Data of LLMs" (ICLR 2025 Spotlight).
#       https://github.com/zjysteven/mink-plus-plus
# ---------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class MinKConfig:
    """Configuration for the Min-K%/Min-K%++ membership-detection baselines.

    ``k_frac`` is the fraction of lowest-scoring tokens to average; ``ignore_pad``
    excludes padding positions from the average when an attention mask is given.
    """

    k_frac: float = 0.2  # 20% of tokens (matches Shi et al. 2024 / MIMIR default)
    # Safety: ignore padding tokens in scoring (requires attention_mask)
    ignore_pad: bool = True


def _min_k_mean(token_scores: torch.Tensor, attention_mask: Optional[torch.Tensor], k_frac: float, ignore_pad: bool = True) -> torch.Tensor:
    """Compute mean over the lowest k% tokens per sequence.

    token_scores: [B, T-1]
    attention_mask: [B, T] or None (1=keep, 0=pad)
    returns: [B]
    """
    B, Tm1 = token_scores.shape
    device = token_scores.device

    if attention_mask is not None and ignore_pad:
        # attention_mask aligns to input_ids. token_scores aligns to predicting tokens 1..T-1.
        valid = attention_mask[:, 1:].to(device=device).bool()
    else:
        valid = torch.ones((B, Tm1), device=device, dtype=torch.bool)

    out = []
    for b in range(B):
        vals = token_scores[b][valid[b]]
        if vals.numel() == 0:
            out.append(torch.tensor(float("nan"), device=device))
            continue
        k = max(1, int(round(k_frac * vals.numel())))
        k = min(k, vals.numel())
        sel, _ = torch.topk(vals, k=k, largest=False)
        out.append(sel.mean())
    return torch.stack(out, dim=0)


@torch.no_grad()
def score_min_k(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cfg: MinKConfig,
) -> torch.Tensor:
    """Min-K% Prob score. Higher => more likely member.

    Token score is log p(x_t | x_<t) (log-prob of observed token).
    Sentence score = mean of the lowest k% token scores.
    """
    # Lazy import to keep the module import-light for unit tests.
    from sae_mia_audit.models.logprobs import next_token_logprobs_and_stats

    lp = next_token_logprobs_and_stats(logits, input_ids)
    token_scores = lp.token_logp  # [B, T-1]
    return _min_k_mean(token_scores, attention_mask, cfg.k_frac, ignore_pad=cfg.ignore_pad)


@torch.no_grad()
def score_min_kpp(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cfg: MinKConfig,
) -> torch.Tensor:
    """Min-K%++ score. Higher => more likely member.

    Token score: (log p(x_t|prefix) - mu(prefix)) / sigma(prefix),
    where mu and sigma are computed over the model's next-token distribution.
    Sentence score = mean of the lowest k% token scores.
    """
    # Lazy import to keep the module import-light for unit tests.
    from sae_mia_audit.models.logprobs import next_token_logprobs_and_stats

    lp = next_token_logprobs_and_stats(logits, input_ids)
    token_scores = (lp.token_logp - lp.mu) / lp.sigma
    return _min_k_mean(token_scores, attention_mask, cfg.k_frac, ignore_pad=cfg.ignore_pad)
