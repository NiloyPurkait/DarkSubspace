from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .pdd import PDDExample


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 0
    calib_frac: float = 0.2
    stratify: bool = True

    # New (safe default): when provided, split by groups instead of individual
    # examples. This is required for datasets like MIMIR where member/nonmember
    # examples come in paired groups (meta['pair_idx']).
    group_key: Optional[str] = None

    # --- Count-based splitting (overrides calib_frac when set) ---
    # These are per-class counts. For example, n_calib_per_class=100 means
    # 100 members + 100 non-members go to calib, rest to test.
    n_calib_per_class: Optional[int] = None  # absolute count per class for calib
    n_test_per_class: Optional[int] = None   # absolute count per class for test


@dataclass
class FPRFeasibilityReport:
    """Report on whether target FPRs are feasible given sample sizes."""
    n_val_non: int
    n_test_non: int
    min_nonzero_val_fpr: float
    min_nonzero_test_fpr: float
    feasible_fprs: List[float]  # FPRs that are >= min_nonzero_val_fpr
    infeasible_fprs: List[float]  # FPRs that are < min_nonzero_val_fpr
    warnings: List[str]


def check_fpr_feasibility(
    n_val_non: int,
    n_test_non: int,
    target_fprs: Tuple[float, ...],
) -> FPRFeasibilityReport:
    """Check whether target FPRs are achievable given sample sizes.

    Args:
        n_val_non: Number of non-members in validation set (for threshold calibration).
        n_test_non: Number of non-members in test set (for FPR evaluation).
        target_fprs: Tuple of target FPR values to check.

    Returns:
        FPRFeasibilityReport with analysis and warnings.
    """
    min_val = 1.0 / max(1, n_val_non)
    min_test = 1.0 / max(1, n_test_non)
    
    feasible = []
    infeasible = []
    warnings = []
    
    for fpr in target_fprs:
        if fpr < min_val:
            infeasible.append(fpr)
            warnings.append(
                f"Target FPR={fpr:.1e} is below min achievable FPR={min_val:.4f} "
                f"(n_val_non={n_val_non}). Threshold will be max(val_non)."
            )
        else:
            feasible.append(fpr)
        
        if fpr < min_test:
            warnings.append(
                f"Target FPR={fpr:.1e} is below test resolution={min_test:.4f} "
                f"(n_test_non={n_test_non}). Test FPR will be 0 or jump to >={min_test:.4f}."
            )
    
    return FPRFeasibilityReport(
        n_val_non=n_val_non,
        n_test_non=n_test_non,
        min_nonzero_val_fpr=min_val,
        min_nonzero_test_fpr=min_test,
        feasible_fprs=feasible,
        infeasible_fprs=infeasible,
        warnings=warnings,
    )


def _stratified_split_by_count(
    labels: np.ndarray,
    n_calib_per_class: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (calib_idx, test_idx) with exactly n_calib_per_class per label in calib."""
    calib_idx: List[int] = []
    test_idx: List[int] = []

    for y in np.unique(labels):
        idx = np.flatnonzero(labels == y)
        rng.shuffle(idx)
        k = min(n_calib_per_class, len(idx) - 1)  # ensure at least 1 in test
        k = max(1, k)  # ensure at least 1 in calib
        calib_idx.extend(idx[:k].tolist())
        test_idx.extend(idx[k:].tolist())

    rng.shuffle(calib_idx)
    rng.shuffle(test_idx)
    return np.array(calib_idx, dtype=int), np.array(test_idx, dtype=int)


def _stratified_split_indices(
    labels: np.ndarray,
    calib_frac: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (calib_idx, test_idx) stratified by labels."""
    calib_idx: List[int] = []
    test_idx: List[int] = []

    for y in np.unique(labels):
        idx = np.flatnonzero(labels == y)
        rng.shuffle(idx)
        k = int(round(calib_frac * len(idx)))
        calib_idx.extend(idx[:k].tolist())
        test_idx.extend(idx[k:].tolist())

    rng.shuffle(calib_idx)
    rng.shuffle(test_idx)
    return np.array(calib_idx, dtype=int), np.array(test_idx, dtype=int)


def _grouped_split_indices(
    examples: List[PDDExample],
    group_key: str,
    calib_frac: float,
    rng: np.random.Generator,
    stratify: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (calib_idx, test_idx) where groups are kept intact.

    Notes
    -----
    - If an example is missing the group key, it is treated as its own group.
    - If `stratify=True`, we approximate label balance at the *group* level by
      assigning each group a label:
        * 0 or 1 if all examples in the group share that label
        * 2 if the group contains mixed labels
      and stratifying over that derived group label.

    This is designed to prevent pair leakage for MIMIR (group_key='pair_idx'),
    where each group is mixed (contains both labels). In that case, stratifying
    reduces to a uniform random split over groups.
    """
    groups: Dict[str, List[int]] = {}
    for i, ex in enumerate(examples):
        gid = ex.meta.get(group_key, None)
        if gid is None:
            gid = f"__ungrouped_{i}"  # unique
        groups.setdefault(str(gid), []).append(i)

    group_ids = list(groups.keys())

    if not stratify:
        rng.shuffle(group_ids)
        k = int(round(calib_frac * len(group_ids)))
        calib_g = group_ids[:k]
        test_g = group_ids[k:]
    else:
        # Derive a group label for stratification.
        g_labels: Dict[str, int] = {}
        for gid, idxs in groups.items():
            ys = [int(examples[i].label) for i in idxs]
            if all(y == ys[0] for y in ys):
                g_labels[gid] = int(ys[0])
            else:
                g_labels[gid] = 2  # mixed

        calib_g: List[str] = []
        test_g: List[str] = []
        for gl in sorted(set(g_labels.values())):
            ids = [gid for gid in group_ids if g_labels[gid] == gl]
            rng.shuffle(ids)
            k = int(round(calib_frac * len(ids)))
            calib_g.extend(ids[:k])
            test_g.extend(ids[k:])

        rng.shuffle(calib_g)
        rng.shuffle(test_g)

    calib_idx = [i for gid in calib_g for i in groups[gid]]
    test_idx = [i for gid in test_g for i in groups[gid]]

    return np.array(calib_idx, dtype=int), np.array(test_idx, dtype=int)


def train_calib_test_split(examples: List[PDDExample], cfg: SplitConfig) -> Tuple[List[PDDExample], List[PDDExample]]:
    rng = np.random.default_rng(cfg.seed)
    labels = np.array([ex.label for ex in examples], dtype=int)

    # Count-based splitting takes precedence over fraction-based
    if cfg.n_calib_per_class is not None:
        calib_idx, test_idx = _stratified_split_by_count(labels, cfg.n_calib_per_class, rng)
    elif cfg.group_key:
        calib_idx, test_idx = _grouped_split_indices(
            examples=examples,
            group_key=str(cfg.group_key),
            calib_frac=float(cfg.calib_frac),
            rng=rng,
            stratify=bool(cfg.stratify),
        )
    else:
        idx = np.arange(len(examples), dtype=int)
        if cfg.stratify:
            calib_idx, test_idx = _stratified_split_indices(labels, cfg.calib_frac, rng)
        else:
            rng.shuffle(idx)
            k = int(round(cfg.calib_frac * len(idx)))
            calib_idx, test_idx = idx[:k], idx[k:]

    calib = [examples[i] for i in calib_idx.tolist()]
    test = [examples[i] for i in test_idx.tolist()]
    return calib, test
