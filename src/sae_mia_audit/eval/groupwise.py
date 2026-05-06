from __future__ import annotations

"""Groupwise evaluation utilities.

These helpers support *fairness slicing* of membership-auditing methods.

Typical use:
  - Choose a metadata key (e.g., 'source', 'domain', 'category')
  - Compute AUROC and TPR@FPR within each group
  - Report disparities (max-min, std) and highlight groups with elevated risk
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from sae_mia_audit.data.pdd import PDDExample
from sae_mia_audit.eval.metrics import MetricsResult, compute_metrics


@dataclass(frozen=True)
class GroupwiseResult:
    per_group: Dict[str, MetricsResult]
    # Simple disparity summaries (computed on AUROC and the two TPR@FPR points)
    auroc_range: float
    tpr1_range: float
    tpr01_range: float


def _as_str_group(g: Any) -> str:
    if g is None:
        return "<none>"
    if isinstance(g, (str, int, float, bool)):
        return str(g)
    return str(g)


def compute_groupwise_metrics(
    examples: Iterable[PDDExample],
    scores: np.ndarray,
    *,
    group_fn: Optional[Callable[[PDDExample], Hashable]] = None,
    meta_key: Optional[str] = None,
) -> GroupwiseResult:
    """Compute metrics per group.

    Provide either:
      - group_fn: function mapping example -> group id
      - meta_key: use example.meta[meta_key] as group id
    """
    ex_list = list(examples)
    scores = np.asarray(scores, dtype=float)
    if len(ex_list) != scores.shape[0]:
        raise ValueError("examples and scores must have same length")

    if group_fn is None:
        if meta_key is None:
            raise ValueError("Provide group_fn or meta_key")

        def group_fn(e: PDDExample):
            return e.meta.get(meta_key)

    groups: Dict[str, List[int]] = {}
    for i, e in enumerate(ex_list):
        g = _as_str_group(group_fn(e))
        groups.setdefault(g, []).append(i)

    per_group: Dict[str, MetricsResult] = {}
    aurocs, tpr1s, tpr01s = [], [], []

    for g, idx in groups.items():
        y = np.asarray([ex_list[i].label for i in idx], dtype=int)
        s = scores[np.asarray(idx, dtype=int)]
        # Need both classes for AUROC; if a group is single-class, skip
        if len(np.unique(y)) < 2:
            continue
        m = compute_metrics(y, s)
        per_group[g] = m
        aurocs.append(m.auroc)
        tpr1s.append(m.tpr_at_fpr_1pct)
        tpr01s.append(m.tpr_at_fpr_0_1pct)

    def _range(xs: List[float]) -> float:
        if not xs:
            return 0.0
        return float(max(xs) - min(xs))

    return GroupwiseResult(
        per_group=per_group,
        auroc_range=_range(aurocs),
        tpr1_range=_range(tpr1s),
        tpr01_range=_range(tpr01s),
    )
