# ---------------------------------------------------------------------------
# Source: Raoof et al., "Infilling Score: A Pretraining Data Detection
#         Algorithm for Large Language Models" (ICLR 2025).
# Original repo: No public code repository found as of 2026-02-20.
#   Verified: first author (NRaoof) and co-author (giannisdaras) GitHub
#   profiles have no infilling-score repo.  Paper has no code link.
# Our implementation follows the paper's algorithm (Section 3).
# ---------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class InfillingConfig:
    m: int = 10  # future window
    batch_size: int = 16


def argmax_next_token(logits: torch.Tensor) -> torch.Tensor:
    # logits: [B, T, V]
    # return argmax tokens for positions 0..T-2 (predict next token)
    return torch.argmax(logits[:, :-1], dim=-1)


def _logp_mu_sigma_for_targets(logits: torch.Tensor, targets: torch.Tensor):
    """Return (logp(target), mu_logp, sigma_logp) for each row.

    logits: [B, V]
    targets: [B] (token ids)
    
    CRITICAL: mu and sigma are computed in log-prob space (not probability space)
    to avoid unit mismatch in z-score computation.
    """
    log_probs = F.log_softmax(logits, dim=-1)  # [B, V]
    logp = log_probs.gather(-1, targets[:, None]).squeeze(-1)  # [B]
    mu = log_probs.mean(dim=-1)  # [B] - mean of log-probs
    sigma = log_probs.std(dim=-1, unbiased=False).clamp_min(1e-6)  # [B] - std of log-probs
    return logp, mu, sigma


class InfillingScorer:
    def __init__(self, model, cfg: InfillingConfig, device: str):
        self.model = model
        self.cfg = cfg
        self.device = device

    @staticmethod
    def _effective_length(attention_mask: Optional[torch.Tensor], T: int) -> int:
        """Compute the number of non-padding tokens.

        If attention_mask is None, treat all tokens as valid.
        """
        if attention_mask is None:
            return int(T)
        m = attention_mask
        if m.dim() == 2:
            m = m[0]
        # treat any nonzero as valid
        return int(m.to(dtype=torch.bool).sum().item())

    def score_one(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> float:
        """Compute Infilling score for a single example.

        Correctness requirements:
          - attention_mask must be respected in both original and modified forwards
          - padding positions must never be scored
        """
        assert input_ids.shape[0] == 1
        T = int(input_ids.shape[1])

        # Respect attention mask (exclude padding from positions we modify/score)
        T_eff = self._effective_length(attention_mask, T)
        # Match legacy behavior for unpadded sequences: score i in [1, T-2].
        # With padding, use i in [1, T_eff-2] so we never touch/score padding.
        last_scored_i = min(T - 2, T_eff - 2)
        if last_scored_i < 1:
            return 0.0

        # Original forward (must pass attention_mask)
        out_orig = self.model.forward(input_ids=input_ids, attention_mask=attention_mask)
        logits_orig = out_orig.logits  # [1,T,V]

        # Precompute for original sequence
        argmax_ids = argmax_next_token(logits_orig)[0]  # [T-1]
        log_probs_orig = F.log_softmax(logits_orig[:, :-1], dim=-1)[0]  # [T-1,V]
        # CRITICAL: mu and sigma must be in log-prob space (not probability space)
        mu_orig = log_probs_orig.mean(dim=-1)  # [T-1] - mean of log-probs
        sigma_orig = log_probs_orig.std(dim=-1, unbiased=False).clamp_min(1e-6)  # [T-1] - std of log-probs

        scores = []
        positions = list(range(1, last_scored_i + 1))

        # Prepare attention mask for modified batches
        attn_1 = None
        if attention_mask is not None:
            attn_1 = attention_mask
            if attn_1.dim() == 1:
                attn_1 = attn_1.unsqueeze(0)

        for start in range(0, len(positions), self.cfg.batch_size):
            chunk_pos = positions[start : start + self.cfg.batch_size]
            # Create batch of modified sequences
            batch_mod = input_ids.repeat(len(chunk_pos), 1).clone()
            for bi, i in enumerate(chunk_pos):
                # i is always a non-padding token index by construction
                batch_mod[bi, i] = argmax_ids[i - 1]

            attn_mod = attn_1.repeat(len(chunk_pos), 1) if attn_1 is not None else None

            # Modified forward MUST use attention_mask to avoid attending to padding
            out_mod = self.model.forward(input_ids=batch_mod, attention_mask=attn_mod)
            logits_mod = out_mod.logits  # [Bch,T,V]

            for bi, i in enumerate(chunk_pos):
                pos_t = i - 1

                # s(x_i)
                tok_i = input_ids[0, i]
                logp_xi = log_probs_orig[pos_t, tok_i]
                mu_xi = mu_orig[pos_t]
                sigma_xi = sigma_orig[pos_t]
                s_xi = (logp_xi - mu_xi) / (sigma_xi + 1e-8)

                # s(x_i^*) in modified
                tok_star = batch_mod[bi, i]
                logp_star, mu_star, sigma_star = _logp_mu_sigma_for_targets(
                                                                    logits_mod[bi : bi + 1, pos_t, :],  # shape [1, V]
                                                                    tok_star.unsqueeze(0),              # shape [1]
                                                                )

                s_xi_star = (logp_star[0] - mu_star[0]) / (sigma_star[0] + 1e-8)

                # Future tokens: j from i+1 to min(i+m, last_real_token)
                j_max = min(T_eff - 1, i + self.cfg.m)
                s_future_mod = 0.0
                for j in range(i + 1, j_max + 1):
                    pos_j = j - 1
                    tok_j = input_ids[0, j]
                    logp_j, mu_j, sigma_j = _logp_mu_sigma_for_targets(
                                                                            logits_mod[bi : bi + 1, pos_j, :],  # shape [1, V]
                                                                            tok_j.unsqueeze(0),                # shape [1]
                                                                        )

                    s_j = (logp_j[0] - mu_j[0]) / (sigma_j[0] + 1e-8)
                    s_future_mod += float(s_j.detach().cpu().item())

                score_i = float((s_xi - s_xi_star).detach().cpu().item()) + s_future_mod
                scores.append(score_i)

        return float(sum(scores) / max(1, len(scores)))

    def score_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        B = input_ids.shape[0]
        scores = []
        for i in range(B):
            s = self.score_one(
                input_ids[i : i + 1],
                attention_mask[i : i + 1] if attention_mask is not None else None,
            )
            scores.append(s)
        return torch.tensor(scores, device="cpu", dtype=torch.float32)


def score_infilling_batched(model, batch: Dict[str, torch.Tensor], cfg: InfillingConfig, device: str) -> torch.Tensor:
    scorer = InfillingScorer(model=model, cfg=cfg, device=device)
    return scorer.score_batch(batch)
