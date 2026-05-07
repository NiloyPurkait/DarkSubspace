"""Batch tokenisation utilities with optional deterministic random cropping.

Wraps a Hugging Face tokenizer to produce fixed-length [B, seq_len] tensors,
with right-padding and an optional isolated RNG for reproducible cropping
during evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import random
import torch

# Optional dependency: allow importing without transformers installed.
try:
    from transformers import PreTrainedTokenizerBase  # type: ignore
except Exception:  # pragma: no cover
    PreTrainedTokenizerBase = Any  # type: ignore


@dataclass(frozen=True)
class TokenizeConfig:
    seq_len: int = 256
    add_special_tokens: bool = True
    random_crop: bool = False  # if True, sample a random contiguous token window
    # Evaluation determinism: use a local RNG with explicit seed for random_crop
    crop_seed: Optional[int] = None  # if set, creates isolated RNG for random_crop


def _resolve_pad_id(tokenizer: PreTrainedTokenizerBase) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        return int(pad_id)

    # Fallback: many decoder-only LMs can be padded with EOS as long as attention_mask masks pads.
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        return int(eos_id)

    raise ValueError("Tokenizer has no pad_token_id or eos_token_id; cannot pad to fixed seq_len.")


def tokenize_batch(
    tokenizer: PreTrainedTokenizerBase,
    texts: List[str],
    cfg: TokenizeConfig,
) -> dict[str, torch.Tensor]:
    """
    Tokenize a list of documents into a fixed [B, seq_len] tensor.

    Behavior:
      - No HF truncation (we handle crop/trunc ourselves)
      - Right-pad to seq_len
      - If cfg.random_crop and sequence is longer than seq_len:
          choose a random contiguous token window
        else:
          take prefix window (start=0)
    
    If cfg.crop_seed is set, random_crop uses an isolated RNG seeded with
    crop_seed + batch_index for deterministic but varied cropping.
    """
    seq_len = int(cfg.seq_len)
    pad_id = _resolve_pad_id(tokenizer)
    
    # Create isolated RNG if crop_seed is specified (eval determinism)
    local_rng: Optional[random.Random] = None
    if cfg.crop_seed is not None:
        local_rng = random.Random(cfg.crop_seed)

    # Tokenize WITHOUT truncation/padding first (returns python lists)
    enc = tokenizer(
        texts,
        padding=False,
        truncation=False,
        return_tensors=None,
        add_special_tokens=cfg.add_special_tokens,
    )

    input_ids_list = enc["input_ids"]

    cropped_ids: List[List[int]] = []
    cropped_mask: List[List[int]] = []

    for idx, ids in enumerate(input_ids_list):
        ids = list(ids)
        n = len(ids)

        if n <= seq_len:
            pad_len = seq_len - n
            cropped_ids.append(ids + [pad_id] * pad_len)
            cropped_mask.append([1] * n + [0] * pad_len)
        else:
            if cfg.random_crop:
                # randint is inclusive at both ends; start in [0, n-seq_len]
                max_start = max(0, n - seq_len)
                # Use local RNG if available (eval determinism)
                if local_rng is not None:
                    start = local_rng.randint(0, max_start)
                else:
                    start = random.randint(0, max_start)

            else:
                start = 0
            span = ids[start : start + seq_len]
            cropped_ids.append(span)
            cropped_mask.append([1] * seq_len)

    return {
        "input_ids": torch.tensor(cropped_ids, dtype=torch.long),
        "attention_mask": torch.tensor(cropped_mask, dtype=torch.long),
    }
