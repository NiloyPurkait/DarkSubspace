"""
DC-PDD: Detecting Pretraining Data Using Distribution-Level Calibration.

Reference:
    Zhang et al., "Pretraining Data Detection for Large Language Models:
    A Divergence-based Calibration Method" (EMNLP 2024 Best Paper).
    https://github.com/zhang-wei-chao/DC-PDD

Core formula:
    alpha_t = p_model(x_t | x_<t) * log(1 / f_corpus(x_t))
    score   = -mean(clip(alpha_t, max=a))

Where:
    - p_model: next-token probability from the target LM
    - f_corpus: Laplace-smoothed unigram frequency from a reference corpus
    - a: LUP (Large Unusual Probability) cap threshold
    - SFO: Score only the First Occurrence of each token in a sequence

Sign convention: higher score => more likely member (lower divergence from training).

Implementation notes:
    - Zhang et al. use C4 as the reference corpus.  We use the benchmark's
      reference split instead— this is cleaner and avoids external corpus
      dependencies while following the same statistical principle.
    - Threshold `a` defaults to a data-adaptive percentile of the raw CE
      values across the reference split, avoiding manual tuning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F


# Deferred import to avoid slow transformers import at module load time
def _get_logprobs_fn():
    from sae_mia_audit.models.logprobs import next_token_logprobs_and_stats
    return next_token_logprobs_and_stats


@dataclass(frozen=True)
class DCPDDConfig:
    """Configuration for DC-PDD scoring.
    
    Attributes:
        a: LUP cap threshold.  If None, estimated as the 95th percentile
           of raw CE values on the reference split (data-adaptive).
        use_sfo: If True, score only the first occurrence of each token
                 per sequence (Zhang et al.'s SFO strategy).
        laplace_alpha: Laplace smoothing for unigram frequencies.
    """
    a: Optional[float] = None  # auto-calibrate if None
    use_sfo: bool = True
    laplace_alpha: float = 1.0
    # Percentile for auto-calibrating `a` when a=None
    a_percentile: float = 95.0


def compute_unigram_freq(
    token_ids: np.ndarray,
    vocab_size: int,
    laplace_alpha: float = 1.0,
) -> np.ndarray:
    """Compute Laplace-smoothed unigram frequency from token sequences.
    
    Args:
        token_ids: 1-D array of all token ids from reference corpus.
        vocab_size: Size of the tokenizer vocabulary.
        laplace_alpha: Smoothing parameter (1.0 = Laplace).
    
    Returns:
        freq: [vocab_size] array of smoothed token frequencies (sums to 1).
    """
    counts = np.bincount(token_ids, minlength=vocab_size).astype(np.float64)
    counts += laplace_alpha
    freq = counts / counts.sum()
    return freq.astype(np.float32)


def _score_dc_pdd_batch(
    token_logp: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    log_inv_freq: torch.Tensor,
    a: float,
    use_sfo: bool = True,
) -> torch.Tensor:
    """Score a batch of sequences using DC-PDD.
    
    Args:
        token_logp: [B, T-1] log p(x_t | x_<t).
        input_ids:  [B, T] token ids.
        attention_mask: [B, T] or None.
        log_inv_freq: [V] log(1/f(v)) for each vocab entry.
        a: LUP cap threshold.
        use_sfo: Use first-occurrence filtering.
    
    Returns:
        scores: [B] DC-PDD scores (higher => more likely member).
    """
    B, Tm1 = token_logp.shape
    device = token_logp.device

    # p(x_t | x_<t)
    token_prob = token_logp.exp()  # [B, T-1]

    # log(1 / f(x_t)) for each predicted token (positions 1..T-1)
    target_ids = input_ids[:, 1:]  # [B, T-1]
    log_inv_f = log_inv_freq[target_ids]  # [B, T-1]

    # Cross-entropy divergence: alpha_t = p_model(x_t) * log(1/f(x_t))
    ce = token_prob * log_inv_f  # [B, T-1]

    # LUP cap: clip to threshold a
    ce = ce.clamp(max=a)

    # Validity mask
    if attention_mask is not None:
        valid = attention_mask[:, 1:].to(device=device, dtype=torch.float32)
    else:
        valid = torch.ones((B, Tm1), device=device, dtype=torch.float32)

    # SFO: Score First Occurrence only
    if use_sfo:
        sfo_mask = torch.ones_like(valid)
        for b in range(B):
            seen = set()
            for t in range(Tm1):
                tid = target_ids[b, t].item()
                if tid in seen:
                    sfo_mask[b, t] = 0.0
                else:
                    seen.add(tid)
        valid = valid * sfo_mask

    # Score = -mean(ce) over valid tokens; higher => more likely member
    # (Members have lower divergence, so less negative score)
    n_valid = valid.sum(dim=1).clamp(min=1)
    scores = -(ce * valid).sum(dim=1) / n_valid

    return scores


@torch.no_grad()
def score_dc_pdd(
    model,
    texts: List[str],
    unigram_freq: np.ndarray,
    cfg: DCPDDConfig,
    seq_len: int = 256,
    batch_size: int = 4,
    ref_texts: Optional[List[str]] = None,
) -> np.ndarray:
    """Score texts using DC-PDD.
    
    Args:
        model: Model wrapper with .tokenizer and .forward().
        texts: Texts to score.
        unigram_freq: [V] Laplace-smoothed unigram frequencies.
        cfg: DC-PDD configuration.
        seq_len: Max sequence length.
        batch_size: Batch size for forward passes.
        ref_texts: If provided and cfg.a is None, used to auto-calibrate
                   the LUP threshold.  Otherwise ignored.
    
    Returns:
        scores: [N] array, higher => more likely member.
    """
    from sae_mia_audit.data.tokenizer import TokenizeConfig, tokenize_batch
    from tqdm import tqdm

    tok_cfg = TokenizeConfig(seq_len=seq_len, random_crop=False)
    device = next(model.model.parameters()).device

    # Precompute log(1/f) on device
    log_inv_freq = torch.from_numpy(
        np.log(1.0 / unigram_freq).astype(np.float32)
    ).to(device)

    # --- Auto-calibrate LUP threshold if needed ---
    a = cfg.a
    if a is None:
        # Estimate from reference texts
        if ref_texts is None or len(ref_texts) == 0:
            raise ValueError(
                "DC-PDD: cfg.a is None and no ref_texts provided for auto-calibration. "
                "Either set cfg.a explicitly or pass ref_texts."
            )
        all_ce = []
        for i in range(0, len(ref_texts), batch_size):
            chunk = ref_texts[i : i + batch_size]
            batch = tokenize_batch(model.tokenizer, chunk, tok_cfg)
            input_ids = batch["input_ids"].to(device)
            attn = batch.get("attention_mask", None)
            if attn is not None:
                attn = attn.to(device)
            out = model.forward(input_ids=input_ids, attention_mask=attn, output_hidden_states=False)
            lp = _get_logprobs_fn()(out.logits, input_ids)
            token_prob = lp.token_logp.exp()
            target_ids = input_ids[:, 1:]
            ce_vals = token_prob * log_inv_freq[target_ids]
            if attn is not None:
                mask = attn[:, 1:]
            else:
                mask = torch.ones_like(ce_vals)
            all_ce.append(ce_vals[mask.bool()].cpu().numpy())
        all_ce = np.concatenate(all_ce)
        a = float(np.percentile(all_ce, cfg.a_percentile))

    # --- Score all texts ---
    scores_list = []
    for i in tqdm(
        range(0, len(texts), batch_size),
        total=(len(texts) + batch_size - 1) // batch_size,
        desc="dc_pdd",
        dynamic_ncols=True,
    ):
        chunk = texts[i : i + batch_size]
        batch = tokenize_batch(model.tokenizer, chunk, tok_cfg)
        input_ids = batch["input_ids"].to(device)
        attn = batch.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(device)

        out = model.forward(input_ids=input_ids, attention_mask=attn, output_hidden_states=False)
        lp = _get_logprobs_fn()(out.logits, input_ids)

        s = _score_dc_pdd_batch(
            lp.token_logp, input_ids, attn, log_inv_freq, a, cfg.use_sfo
        )
        scores_list.append(s.cpu().numpy())

    return np.concatenate(scores_list)
