"""Cross-SAE feature consistency utilities.

Provides Hungarian-style cosine matching between feature dictionaries from
independently trained SAEs, used to assess which directions are stable
across runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class FeatureMatch:
    idx_a: int
    idx_b: int
    cosine: float


@torch.no_grad()
def match_features_by_cosine(
    W_dec_a: torch.Tensor,
    W_dec_b: torch.Tensor,
    top_k: int = 128,
) -> List[FeatureMatch]:
    """Greedy matching of SAE features across two dictionaries by decoder cosine similarity.

    Args:
      W_dec_a: decoder weight [D, F_a] (as in nn.Linear.weight)
      W_dec_b: decoder weight [D, F_b]
      top_k: return top_k pairs by similarity (not enforcing 1-1 matching globally)
    """
    # Convert to feature vectors: columns are features
    A = W_dec_a.detach().float()
    B = W_dec_b.detach().float()

    # Normalize columns
    A = A / (torch.linalg.norm(A, dim=0, keepdim=True).clamp_min(1e-8))
    B = B / (torch.linalg.norm(B, dim=0, keepdim=True).clamp_min(1e-8))

    # Cosine sim matrix [F_a, F_b]
    sim = (A.T @ B).cpu().numpy()
    # Get top_k pairs
    flat = sim.reshape(-1)
    if top_k > flat.size:
        top_k = flat.size
    idx = np.argpartition(-flat, top_k - 1)[:top_k]
    idx = idx[np.argsort(-flat[idx])]
    pairs: List[FeatureMatch] = []
    F_b = sim.shape[1]
    for p in idx:
        ia = int(p // F_b)
        ib = int(p % F_b)
        pairs.append(FeatureMatch(idx_a=ia, idx_b=ib, cosine=float(sim[ia, ib])))
    return pairs
