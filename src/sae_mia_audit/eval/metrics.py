"""Evaluation metrics for membership-detection experiments.

Provides AUROC and TPR-at-fixed-FPR computations with configurable FPR
targets, plus convenience aliases used by the standard MIA literature.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


# Standard FPR targets for membership-detection evaluation
# 5%: Moderate regime (meaningful for small datasets)
# 1%: Low-FPR regime (standard in MIA literature)
# 0.1%: Very low-FPR regime (security-critical, requires large datasets)
DEFAULT_FPR_TARGETS = (0.05, 0.01, 0.001)


@dataclass(frozen=True)
class MetricsResult:
    """MIA evaluation metrics with configurable FPR targets.
    
    Attributes:
        auroc: Area Under ROC Curve (overall discrimination ability)
        tpr_at_fpr: Dict mapping FPR target -> TPR at that threshold
            e.g., {0.05: 0.35, 0.01: 0.12, 0.001: 0.02}
        
    For backward compatibility, also provides:
        tpr_at_fpr_1pct: TPR at FPR=1% (alias for tpr_at_fpr[0.01])
        tpr_at_fpr_0_1pct: TPR at FPR=0.1% (alias for tpr_at_fpr[0.001])
    """
    auroc: float
    tpr_at_fpr: Dict[float, float] = field(default_factory=dict)
    
    @property
    def tpr_at_fpr_5pct(self) -> float:
        """TPR at FPR=5% (backward-compatible alias)."""
        return self.tpr_at_fpr.get(0.05, float('nan'))
    
    @property
    def tpr_at_fpr_1pct(self) -> float:
        """TPR at FPR=1% (backward-compatible alias)."""
        return self.tpr_at_fpr.get(0.01, float('nan'))
    
    @property
    def tpr_at_fpr_0_1pct(self) -> float:
        """TPR at FPR=0.1% (backward-compatible alias)."""
        return self.tpr_at_fpr.get(0.001, float('nan'))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        d = {"auroc": self.auroc}
        for fpr, tpr in self.tpr_at_fpr.items():
            # Use consistent key naming: tpr_at_fpr_X where X is percentage
            pct_str = f"{fpr*100:.1f}".rstrip('0').rstrip('.')
            d[f"tpr_at_fpr_{pct_str}pct"] = tpr
        return d


def _tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    """Compute TPR at a specific FPR threshold using ROC curve.
    
    Returns the maximum TPR achievable at FPR <= target_fpr.
    
    Note: This does NOT interpolate between ROC points. For small datasets,
    achievable FPR values are discrete (1/n_neg, 2/n_neg, ...). If target_fpr
    falls between achievable points, we return the TPR at the largest FPR
    that doesn't exceed target_fpr.
    
    For a more conservative estimate, consider requiring FPR to be strictly
    achievable given sample size: min_fpr = 1/n_neg.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # Find largest tpr where fpr <= target_fpr (no interpolation)
    mask = fpr <= target_fpr
    if not np.any(mask):
        return 0.0
    return float(np.max(tpr[mask]))


def compute_metrics(
    y_true: np.ndarray, 
    y_score: np.ndarray,
    fpr_targets: Tuple[float, ...] = DEFAULT_FPR_TARGETS,
) -> MetricsResult:
    """Compute MIA evaluation metrics.
    
    Args:
        y_true: Ground truth labels (1=member, 0=non-member)
        y_score: Predicted scores (higher = more likely member)
        fpr_targets: FPR thresholds at which to compute TPR
        
    Returns:
        MetricsResult with AUROC and TPR at each FPR target
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    
    # Fail loudly on NaN/Inf — never silently substitute values, which
    # would produce dishonest metrics.  The caller should catch this and
    # record the method as failed.
    nan_mask = ~np.isfinite(y_score)
    if nan_mask.any():
        n_bad = int(nan_mask.sum())
        raise ValueError(
            f"compute_metrics: {n_bad}/{len(y_score)} scores are NaN/Inf. "
            f"This indicates a bug in the scoring method — refusing to "
            f"produce metrics from corrupted scores."
        )
    
    auroc = float(roc_auc_score(y_true, y_score))
    
    tpr_at_fpr = {}
    for fpr in fpr_targets:
        tpr_at_fpr[fpr] = _tpr_at_fpr(y_true, y_score, fpr)
    
    return MetricsResult(auroc=auroc, tpr_at_fpr=tpr_at_fpr)
