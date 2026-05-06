"""Calibration and thresholding utilities for membership inference evaluation.

This module provides functions for:
- Fixed-FPR threshold calibration using held-out non-members only
- Score normalization (z-score, quantile) fitted on non-members
- Per-feature and per-SAE calibration scopes

Key principle: All calibration is done on non-member data from a validation
split that is disjoint from both the training (reference) and test sets.
This prevents any form of test-set leakage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np
from scipy import stats


NormMode = Literal["none", "zscore", "quantile"]
NormScope = Literal["feature", "sae", "global"]


@dataclass
class ThresholdResult:
    """Result of threshold calibration at a target FPR."""

    target_fpr: float
    threshold: float
    # Achieved metrics on the calibration set (should match target_fpr closely)
    calib_fpr: float
    calib_tpr: float


def threshold_at_fpr(
    nonmember_scores: np.ndarray,
    fpr: float,
    method: str = "higher",
) -> float:
    """Compute threshold to achieve target FPR on non-member scores.

    Args:
        nonmember_scores: 1D array of scores from non-member examples.
        fpr: Target false positive rate (e.g., 0.01 for 1%).
        method: Quantile interpolation method. "higher" ensures FPR <= target.

    Returns:
        Threshold t such that P(score > t | non-member) <= fpr.
    """
    if nonmember_scores.size == 0:
        return float("inf")
    q = 1.0 - fpr
    return float(np.quantile(nonmember_scores, q, method=method))


def tpr_fpr_at_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> Tuple[float, float]:
    """Compute TPR and FPR at a given threshold.

    Convention: score > threshold => predicted member.

    Args:
        scores: 1D array of membership scores.
        labels: 1D array of true labels (1 = member, 0 = non-member).
        threshold: Decision threshold.

    Returns:
        (tpr, fpr) tuple.
    """
    predictions = scores > threshold
    members = labels == 1
    nonmembers = labels == 0

    tpr = float(predictions[members].mean()) if members.sum() > 0 else 0.0
    fpr = float(predictions[nonmembers].mean()) if nonmembers.sum() > 0 else 0.0

    return tpr, fpr


def calibrate_thresholds(
    val_nonmember_scores: np.ndarray,
    val_member_scores: np.ndarray,
    target_fprs: Tuple[float, ...] = (1e-2, 1e-3, 1e-4),
) -> Dict[float, ThresholdResult]:
    """Calibrate thresholds for multiple target FPRs using validation data.

    Args:
        val_nonmember_scores: Scores from validation non-members.
        val_member_scores: Scores from validation members.
        target_fprs: Target FPR levels.

    Returns:
        Dict mapping target_fpr -> ThresholdResult.
    """
    results = {}
    for fpr in target_fprs:
        thr = threshold_at_fpr(val_nonmember_scores, fpr)
        calib_tpr = float((val_member_scores > thr).mean()) if len(val_member_scores) > 0 else 0.0
        calib_fpr = float((val_nonmember_scores > thr).mean()) if len(val_nonmember_scores) > 0 else 0.0
        results[fpr] = ThresholdResult(
            target_fpr=fpr,
            threshold=thr,
            calib_fpr=calib_fpr,
            calib_tpr=calib_tpr,
        )
    return results


# ---------------------------------------------------------------------------
# Score Normalization
# ---------------------------------------------------------------------------


@dataclass
class ZScoreNormalizer:
    """Z-score normalizer fitted on reference non-members."""

    mu: np.ndarray  # [n_features] or scalar
    sigma: np.ndarray  # [n_features] or scalar
    eps: float = 1e-8

    @classmethod
    def fit(
        cls,
        ref_nonmember_scores: np.ndarray,
        scope: NormScope = "feature",
        eps: float = 1e-8,
    ) -> "ZScoreNormalizer":
        """Fit normalizer on reference non-member scores.

        Args:
            ref_nonmember_scores: Shape [n_samples, n_features] or [n_samples].
            scope: "feature" normalizes each feature independently,
                   "sae" or "global" normalizes all features together.
            eps: Small constant for numerical stability.

        Returns:
            Fitted normalizer.
        """
        if ref_nonmember_scores.ndim == 1:
            ref_nonmember_scores = ref_nonmember_scores[:, np.newaxis]

        if scope == "feature":
            mu = ref_nonmember_scores.mean(axis=0)
            sigma = ref_nonmember_scores.std(axis=0) + eps
        else:  # "sae" or "global"
            mu = np.array([ref_nonmember_scores.mean()])
            sigma = np.array([ref_nonmember_scores.std() + eps])

        return cls(mu=mu, sigma=sigma, eps=eps)

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Apply z-score normalization.

        Args:
            scores: Shape [n_samples, n_features] or [n_samples].

        Returns:
            Normalized scores with same shape.
        """
        squeeze = scores.ndim == 1
        if squeeze:
            scores = scores[:, np.newaxis]

        normalized = (scores - self.mu) / self.sigma

        if squeeze:
            normalized = normalized.ravel()

        return normalized


@dataclass
class QuantileNormalizer:
    """Quantile normalizer fitted on reference non-members.

    Maps scores to their empirical CDF values (uniform in [0, 1]).
    Optionally applies inverse normal transform (Gaussian quantile).
    """

    # Stored quantiles for interpolation [n_quantiles, n_features] or [n_quantiles]
    quantile_values: np.ndarray
    n_quantiles: int
    to_gaussian: bool = False

    @classmethod
    def fit(
        cls,
        ref_nonmember_scores: np.ndarray,
        scope: NormScope = "feature",
        n_quantiles: int = 1000,
        to_gaussian: bool = False,
    ) -> "QuantileNormalizer":
        """Fit quantile normalizer on reference non-member scores.

        Args:
            ref_nonmember_scores: Shape [n_samples, n_features] or [n_samples].
            scope: "feature" fits per-feature, "sae"/"global" fits globally.
            n_quantiles: Number of quantile points for interpolation.
            to_gaussian: If True, map to standard normal quantiles.

        Returns:
            Fitted normalizer.
        """
        if ref_nonmember_scores.ndim == 1:
            ref_nonmember_scores = ref_nonmember_scores[:, np.newaxis]

        quantile_probs = np.linspace(0, 1, n_quantiles)

        if scope == "feature":
            quantile_values = np.quantile(ref_nonmember_scores, quantile_probs, axis=0)
        else:  # "sae" or "global"
            flat = ref_nonmember_scores.ravel()
            quantile_values = np.quantile(flat, quantile_probs)[:, np.newaxis]

        return cls(
            quantile_values=quantile_values,
            n_quantiles=n_quantiles,
            to_gaussian=to_gaussian,
        )

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Apply quantile normalization.

        Args:
            scores: Shape [n_samples, n_features] or [n_samples].

        Returns:
            Normalized scores (uniform in [0,1] or Gaussian if to_gaussian).
        """
        squeeze = scores.ndim == 1
        if squeeze:
            scores = scores[:, np.newaxis]

        n_samples, n_features = scores.shape
        normalized = np.zeros_like(scores)

        for j in range(n_features):
            qv = self.quantile_values[:, min(j, self.quantile_values.shape[1] - 1)]
            # Map to quantile rank via searchsorted
            ranks = np.searchsorted(qv, scores[:, j], side="right")
            # Convert to [0, 1]
            u = ranks / self.n_quantiles
            u = np.clip(u, 1e-6, 1 - 1e-6)  # Avoid exact 0 or 1 for Gaussian

            if self.to_gaussian:
                normalized[:, j] = stats.norm.ppf(u)
            else:
                normalized[:, j] = u

        if squeeze:
            normalized = normalized.ravel()

        return normalized


def create_normalizer(
    ref_nonmember_scores: np.ndarray,
    mode: NormMode,
    scope: NormScope = "feature",
    eps: float = 1e-8,
    n_quantiles: int = 1000,
    to_gaussian: bool = False,
):
    """Factory to create a score normalizer.

    Args:
        ref_nonmember_scores: Reference non-member scores to fit on.
        mode: Normalization mode ("none", "zscore", "quantile").
        scope: "feature", "sae", or "global".
        eps: Epsilon for z-score stability.
        n_quantiles: Number of quantiles for quantile normalization.
        to_gaussian: For quantile norm, whether to map to Gaussian.

    Returns:
        Normalizer object with .transform() method, or None if mode is "none".
    """
    if mode == "none":
        return None

    if mode == "zscore":
        return ZScoreNormalizer.fit(ref_nonmember_scores, scope=scope, eps=eps)

    if mode == "quantile":
        return QuantileNormalizer.fit(
            ref_nonmember_scores,
            scope=scope,
            n_quantiles=n_quantiles,
            to_gaussian=to_gaussian,
        )

    raise ValueError(f"Unknown normalization mode: {mode}")


# ---------------------------------------------------------------------------
# Per-feature selection metrics
# ---------------------------------------------------------------------------


def feature_tpr_at_fpr(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
    fpr: float,
) -> np.ndarray:
    """Compute TPR@FPR for each feature.

    Args:
        member_scores: [n_members, n_features]
        nonmember_scores: [n_nonmembers, n_features]
        fpr: Target FPR.

    Returns:
        [n_features] array of TPR values.
    """
    if member_scores.ndim == 1:
        member_scores = member_scores[:, np.newaxis]
    if nonmember_scores.ndim == 1:
        nonmember_scores = nonmember_scores[:, np.newaxis]

    n_features = member_scores.shape[1]
    tprs = np.zeros(n_features)

    for j in range(n_features):
        thr = threshold_at_fpr(nonmember_scores[:, j], fpr)
        tprs[j] = float((member_scores[:, j] > thr).mean())

    return tprs


def feature_tail_separation(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
    qs: Tuple[float, ...] = (0.95, 0.96, 0.97, 0.98, 0.99),
) -> np.ndarray:
    """Compute mean tail separation for each feature.

    Tail separation = mean over quantiles q of (quantile(mem, q) - quantile(non, q)).

    Args:
        member_scores: [n_members, n_features]
        nonmember_scores: [n_nonmembers, n_features]
        qs: Quantile levels.

    Returns:
        [n_features] array of tail separation values.
    """
    if member_scores.ndim == 1:
        member_scores = member_scores[:, np.newaxis]
    if nonmember_scores.ndim == 1:
        nonmember_scores = nonmember_scores[:, np.newaxis]

    n_features = member_scores.shape[1]
    tail_seps = np.zeros(n_features)

    for j in range(n_features):
        diffs = []
        for q in qs:
            mem_q = np.quantile(member_scores[:, j], q)
            non_q = np.quantile(nonmember_scores[:, j], q)
            diffs.append(mem_q - non_q)
        tail_seps[j] = float(np.mean(diffs))

    return tail_seps


def feature_auc(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
) -> np.ndarray:
    """Compute AUC for each feature.

    Args:
        member_scores: [n_members, n_features]
        nonmember_scores: [n_nonmembers, n_features]

    Returns:
        [n_features] array of AUC values.
    """
    from sklearn.metrics import roc_auc_score

    if member_scores.ndim == 1:
        member_scores = member_scores[:, np.newaxis]
    if nonmember_scores.ndim == 1:
        nonmember_scores = nonmember_scores[:, np.newaxis]

    n_features = member_scores.shape[1]
    aucs = np.zeros(n_features)

    y = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])

    for j in range(n_features):
        scores = np.concatenate([member_scores[:, j], nonmember_scores[:, j]])
        if len(np.unique(y)) > 1 and len(np.unique(scores)) > 1:
            aucs[j] = roc_auc_score(y, scores)
        else:
            aucs[j] = 0.5

    return aucs


def select_features_by_metric(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
    metric: Literal["tpr@fpr", "tail_sep", "auc"],
    n_features: Optional[int] = None,
    fpr: float = 1e-3,
    qs: Tuple[float, ...] = (0.95, 0.96, 0.97, 0.98, 0.99),
) -> Tuple[np.ndarray, np.ndarray]:
    """Select top features by a given metric.

    Args:
        member_scores: [n_members, n_features]
        nonmember_scores: [n_nonmembers, n_features]
        metric: Selection metric.
        n_features: Number of features to select (None = all).
        fpr: Target FPR for tpr@fpr metric.
        qs: Quantiles for tail_sep metric.

    Returns:
        (selected_indices, metric_values) where selected_indices are sorted by metric.
    """
    if metric == "tpr@fpr":
        values = feature_tpr_at_fpr(member_scores, nonmember_scores, fpr)
    elif metric == "tail_sep":
        values = feature_tail_separation(member_scores, nonmember_scores, qs)
    elif metric == "auc":
        values = feature_auc(member_scores, nonmember_scores)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Sort by descending metric value
    order = np.argsort(-values)

    if n_features is not None:
        order = order[:n_features]

    return order, values[order]
