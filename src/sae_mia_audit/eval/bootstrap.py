"""Paired, stratified bootstrap utilities for delta confidence intervals.

Provides resampling-based CIs for AUROC, TPR-at-FPR, and group-difference
statistics, used to compare ablated vs. baseline scores while preserving
class balance under the null.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple, Optional

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .metrics import compute_metrics, MetricsResult


# =============================================================================
# Paired Stratified Bootstrap for Delta CIs
# =============================================================================

def _auroc_safe(y: np.ndarray, s: np.ndarray) -> float:
    """AUROC with safety check for single-class bootstrap samples."""
    if len(np.unique(y)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return 0.5


def _tpr_at_fpr(y: np.ndarray, s: np.ndarray, fpr_target: float = 0.01) -> float:
    """TPR at a given FPR threshold."""
    if len(np.unique(y)) < 2:
        return 0.0
    try:
        fpr, tpr, _ = roc_curve(y, s)
        idx = np.searchsorted(fpr, fpr_target, side='right') - 1
        idx = max(0, min(idx, len(tpr) - 1))
        return float(tpr[idx])
    except ValueError:
        return 0.0


def paired_stratified_bootstrap_delta_ci(
    labels: np.ndarray,
    scores_base: np.ndarray,
    scores_edited: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute paired, stratified bootstrap CI for delta = metric(edited) - metric(base).

    Implementation notes:
    - **Paired**: Resample indices once; apply to BOTH base and edited arrays
    - **Stratified**: Resample members and nonmembers separately, then concatenate
    - Returns CI for **delta** (edited - base), not absolute values
    
    Args:
        labels: Binary labels [N] (1=member, 0=nonmember).
        scores_base: Baseline scores [N].
        scores_edited: Edited/ablated scores [N].
        metric_fn: Function (labels, scores) -> scalar metric.
        n_boot: Number of bootstrap samples.
        seed: Random seed.
        alpha: Significance level (default 0.05 for 95% CI).
    
    Returns:
        Tuple of (delta_mean, (ci_low, ci_high)).
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    scores_base = np.asarray(scores_base)
    scores_edited = np.asarray(scores_edited)
    
    # Get member and nonmember indices
    mem_idx = np.where(labels == 1)[0]
    non_idx = np.where(labels == 0)[0]
    n_mem = len(mem_idx)
    n_non = len(non_idx)
    
    if n_mem == 0 or n_non == 0:
        # Cannot compute stratified CI with single class
        delta = metric_fn(labels, scores_edited) - metric_fn(labels, scores_base)
        return delta, (delta, delta)
    
    deltas = []
    for _ in range(n_boot):
        # Stratified resampling: resample members and nonmembers separately
        boot_mem = rng.choice(mem_idx, size=n_mem, replace=True)
        boot_non = rng.choice(non_idx, size=n_non, replace=True)
        boot_idx = np.concatenate([boot_mem, boot_non])
        
        # Paired: apply same indices to both base and edited
        y_boot = labels[boot_idx]
        s_base_boot = scores_base[boot_idx]
        s_edit_boot = scores_edited[boot_idx]
        
        m_base = metric_fn(y_boot, s_base_boot)
        m_edit = metric_fn(y_boot, s_edit_boot)
        deltas.append(m_edit - m_base)
    
    deltas = np.array(deltas)
    delta_mean = float(np.mean(deltas))
    ci_low = float(np.percentile(deltas, 100 * alpha / 2))
    ci_high = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    
    return delta_mean, (ci_low, ci_high)


def bootstrap_ci_of_group_diff(
    values: np.ndarray,
    labels: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute bootstrap CI for (mean_members - mean_nonmembers).

    Used to test whether an intervention affects members and nonmembers
    differently.
    
    Args:
        values: Per-example values (e.g., delta scores) [N].
        labels: Binary labels [N] (1=member, 0=nonmember).
        n_boot: Number of bootstrap samples.
        seed: Random seed.
        alpha: Significance level.
    
    Returns:
        Tuple of (diff_mean, (ci_low, ci_high)).
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    labels = np.asarray(labels)
    
    mem_vals = values[labels == 1]
    non_vals = values[labels == 0]
    
    if len(mem_vals) == 0 or len(non_vals) == 0:
        diff = 0.0
        return diff, (diff, diff)
    
    diffs = []
    for _ in range(n_boot):
        boot_mem = rng.choice(mem_vals, size=len(mem_vals), replace=True)
        boot_non = rng.choice(non_vals, size=len(non_vals), replace=True)
        diffs.append(boot_mem.mean() - boot_non.mean())
    
    diffs = np.array(diffs)
    diff_mean = float(np.mean(diffs))
    ci_low = float(np.percentile(diffs, 100 * alpha / 2))
    ci_high = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    
    return diff_mean, (ci_low, ci_high)


def compute_full_attack_metrics_with_bootstrap(
    labels: np.ndarray,
    scores_base: np.ndarray,
    scores_edited: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict:
    """
    Compute full attack metrics on entire test set with paired stratified bootstrap CIs.

    Centralized function that replaces scattered bootstrap logic.

    Returns:
        Dict with baseline, after_ablation, and delta metrics, each with CIs.
    """
    labels = np.asarray(labels)
    scores_base = np.asarray(scores_base)
    scores_edited = np.asarray(scores_edited)
    
    n_members = int((labels == 1).sum())
    n_nonmembers = int((labels == 0).sum())
    
    # Validate we have both classes
    if n_members == 0 or n_nonmembers == 0:
        return {
            "error": "Cannot compute AUROC: missing class",
            "n_members": n_members,
            "n_nonmembers": n_nonmembers,
            "baseline": None,
            "after_ablation": None,
            "delta": None,
        }
    
    # Point estimates
    base_auroc = _auroc_safe(labels, scores_base)
    edit_auroc = _auroc_safe(labels, scores_edited)
    base_tpr_1pct = _tpr_at_fpr(labels, scores_base, 0.01)
    edit_tpr_1pct = _tpr_at_fpr(labels, scores_edited, 0.01)
    base_tpr_01pct = _tpr_at_fpr(labels, scores_base, 0.001)
    edit_tpr_01pct = _tpr_at_fpr(labels, scores_edited, 0.001)
    
    # Paired stratified bootstrap for deltas
    _, auroc_delta_ci = paired_stratified_bootstrap_delta_ci(
        labels, scores_base, scores_edited, _auroc_safe, n_boot, seed
    )
    _, tpr1_delta_ci = paired_stratified_bootstrap_delta_ci(
        labels, scores_base, scores_edited, 
        lambda y, s: _tpr_at_fpr(y, s, 0.01), n_boot, seed
    )
    _, tpr01_delta_ci = paired_stratified_bootstrap_delta_ci(
        labels, scores_base, scores_edited,
        lambda y, s: _tpr_at_fpr(y, s, 0.001), n_boot, seed
    )
    
    # Standard bootstrap for individual metrics
    rng = np.random.default_rng(seed)
    base_aurocs, edit_aurocs = [], []
    base_tpr1s, edit_tpr1s = [], []
    
    mem_idx = np.where(labels == 1)[0]
    non_idx = np.where(labels == 0)[0]
    
    for _ in range(n_boot):
        boot_mem = rng.choice(mem_idx, size=len(mem_idx), replace=True)
        boot_non = rng.choice(non_idx, size=len(non_idx), replace=True)
        boot_idx = np.concatenate([boot_mem, boot_non])
        
        y_boot = labels[boot_idx]
        base_aurocs.append(_auroc_safe(y_boot, scores_base[boot_idx]))
        edit_aurocs.append(_auroc_safe(y_boot, scores_edited[boot_idx]))
        base_tpr1s.append(_tpr_at_fpr(y_boot, scores_base[boot_idx], 0.01))
        edit_tpr1s.append(_tpr_at_fpr(y_boot, scores_edited[boot_idx], 0.01))
    
    return {
        "n_samples": len(labels),
        "n_members": n_members,
        "n_nonmembers": n_nonmembers,
        "baseline": {
            "auroc": base_auroc,
            "auroc_ci_95": (float(np.percentile(base_aurocs, 2.5)), float(np.percentile(base_aurocs, 97.5))),
            "tpr_at_fpr_1pct": base_tpr_1pct,
            "tpr_at_fpr_1pct_ci_95": (float(np.percentile(base_tpr1s, 2.5)), float(np.percentile(base_tpr1s, 97.5))),
            "tpr_at_fpr_0_1pct": base_tpr_01pct,
        },
        "after_ablation": {
            "auroc": edit_auroc,
            "auroc_ci_95": (float(np.percentile(edit_aurocs, 2.5)), float(np.percentile(edit_aurocs, 97.5))),
            "tpr_at_fpr_1pct": edit_tpr_1pct,
            "tpr_at_fpr_1pct_ci_95": (float(np.percentile(edit_tpr1s, 2.5)), float(np.percentile(edit_tpr1s, 97.5))),
            "tpr_at_fpr_0_1pct": edit_tpr_01pct,
        },
        "delta": {
            "auroc": edit_auroc - base_auroc,
            "auroc_ci_95": auroc_delta_ci,
            "tpr_at_fpr_1pct": edit_tpr_1pct - base_tpr_1pct,
            "tpr_at_fpr_1pct_ci_95": tpr1_delta_ci,
            "tpr_at_fpr_0_1pct": edit_tpr_01pct - base_tpr_01pct,
            "tpr_at_fpr_0_1pct_ci_95": tpr01_delta_ci,
        },
    }


# =============================================================================
# Original bootstrap functions (kept for backward compatibility)
# =============================================================================

def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, Tuple[float, float]]:
    """Bootstrap percentile CI for key metrics."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)

    aurocs = []
    tpr1 = []
    tpr01 = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = compute_metrics(y_true[idx], y_score[idx])
        aurocs.append(m.auroc)
        tpr1.append(m.tpr_at_fpr_1pct)
        tpr01.append(m.tpr_at_fpr_0_1pct)

    def pct(x):
        lo = float(np.percentile(x, 2.5))
        hi = float(np.percentile(x, 97.5))
        return lo, hi

    return {
        "auroc": pct(aurocs),
        "tpr_at_fpr_1pct": pct(tpr1),
        "tpr_at_fpr_0_1pct": pct(tpr01),
    }


def bootstrap_ci_at_fixed_threshold(
    y_test: np.ndarray,
    test_scores: np.ndarray,
    threshold: float,
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, Tuple[float, float]]:
    """Bootstrap CI for TPR and FPR at a FIXED threshold.

    This is used for val-calibrated low-FPR evaluation where the threshold
    is set on the validation set and then applied to the test set.
    
    Args:
        y_test: Test set labels (1 = member, 0 = non-member).
        test_scores: Test set scores (higher = more likely member).
        threshold: Fixed threshold (calibrated on validation set).
        n_boot: Number of bootstrap samples.
        seed: Random seed.
    
    Returns:
        Dict with "tpr" and "fpr" keys, each mapping to (lo, hi) 95% CI.
    """
    rng = np.random.default_rng(seed)
    y_test = np.asarray(y_test, dtype=int)
    test_scores = np.asarray(test_scores, dtype=float)
    n = len(y_test)
    
    tprs = []
    fprs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y_test[idx]
        s_b = test_scores[idx]
        
        pred = s_b > threshold
        mem_mask = y_b == 1
        non_mask = y_b == 0
        
        tpr = float(pred[mem_mask].mean()) if mem_mask.sum() > 0 else 0.0
        fpr = float(pred[non_mask].mean()) if non_mask.sum() > 0 else 0.0
        
        tprs.append(tpr)
        fprs.append(fpr)
    
    def pct(x):
        lo = float(np.percentile(x, 2.5))
        hi = float(np.percentile(x, 97.5))
        return lo, hi
    
    return {
        "tpr": pct(tprs),
        "fpr": pct(fprs),
    }


def bootstrap_ci_low_fpr(
    y_test: np.ndarray,
    test_scores: np.ndarray,
    thresholds: Dict[float, float],
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[float, Dict[str, Tuple[float, float]]]:
    """Bootstrap CI for multiple val-calibrated thresholds.

    Args:
        y_test: Test set labels.
        test_scores: Test set scores.
        thresholds: Dict mapping target_fpr -> threshold (from val calibration).
        n_boot: Number of bootstrap samples.
        seed: Random seed.

    Returns:
        Dict mapping target_fpr -> {"tpr": (lo, hi), "fpr": (lo, hi)}.
    """
    results = {}
    for target_fpr, thr in thresholds.items():
        ci = bootstrap_ci_at_fixed_threshold(
            y_test, test_scores, thr, n_boot=n_boot, seed=seed
        )
        results[target_fpr] = ci
    return results
