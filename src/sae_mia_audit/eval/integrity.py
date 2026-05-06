"""Split integrity and confound audit utilities (reviewer-proof).

This module provides:
1. Split disjointness checks (no data leakage)
2. Exact duplicate detection across splits
3. Length confound diagnostics
4. Determinism verification

These are critical for addressing reviewer attacks on experimental validity.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from sae_mia_audit.data.pdd import PDDExample


@dataclass
class IntegrityReport:
    """Comprehensive integrity report for reviewer-proofing."""
    
    # Split disjointness (MIMIR pair_idx, etc.)
    group_leakage_found: bool = False
    group_leakage_details: Dict[str, List] = field(default_factory=dict)
    
    # Exact text duplicates across splits
    duplicate_texts_found: bool = False
    duplicate_hashes: List[str] = field(default_factory=list)
    n_duplicates_per_split_pair: Dict[str, int] = field(default_factory=dict)
    
    # Length confound diagnostics
    length_auroc: float = 0.5
    length_auroc_significant: bool = False  # True if AUROC > 0.65 or < 0.35
    length_correlation_with_label: float = 0.0
    member_length_mean: float = 0.0
    member_length_std: float = 0.0
    nonmember_length_mean: float = 0.0
    nonmember_length_std: float = 0.0
    
    # Token count confounds (if tokenizer provided)
    token_auroc: float = 0.5
    token_auroc_significant: bool = False
    
    # Overall pass/fail
    @property
    def passed(self) -> bool:
        return (
            not self.group_leakage_found
            and not self.duplicate_texts_found
            and not self.length_auroc_significant
            and not self.token_auroc_significant
        )
    
    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "group_leakage": {
                "found": self.group_leakage_found,
                "details": self.group_leakage_details,
            },
            "duplicate_texts": {
                "found": self.duplicate_texts_found,
                "n_duplicates": len(self.duplicate_hashes),
                "per_split_pair": self.n_duplicates_per_split_pair,
            },
            "length_confound": {
                "auroc": self.length_auroc,
                "significant": self.length_auroc_significant,
                "correlation": self.length_correlation_with_label,
                "member_mean": self.member_length_mean,
                "member_std": self.member_length_std,
                "nonmember_mean": self.nonmember_length_mean,
                "nonmember_std": self.nonmember_length_std,
            },
            "token_confound": {
                "auroc": self.token_auroc,
                "significant": self.token_auroc_significant,
            },
        }


def _text_hash(text: str) -> str:
    """Compute SHA256 hash of text (normalized)."""
    normalized = text.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def check_group_disjointness(
    splits: Dict[str, List[PDDExample]],
    group_key: str = "pair_idx",
) -> Tuple[bool, Dict[str, List]]:
    """Check that no group_key values appear in multiple splits.
    
    Critical for MIMIR: if pair_idx crosses splits, there's data leakage.
    
    Args:
        splits: Dict mapping split name to list of examples
        group_key: Metadata key to check for disjointness
        
    Returns:
        (leakage_found, details) where details maps overlapping groups to split pairs
    """
    # Collect groups per split
    groups_per_split: Dict[str, Set] = {}
    for split_name, examples in splits.items():
        groups = set()
        for ex in examples:
            if group_key in ex.meta:
                groups.add(ex.meta[group_key])
        groups_per_split[split_name] = groups
    
    # Check pairwise overlaps
    leakage_details: Dict[str, List] = {}
    leakage_found = False
    split_names = list(groups_per_split.keys())
    
    for i, s1 in enumerate(split_names):
        for s2 in split_names[i+1:]:
            overlap = groups_per_split[s1] & groups_per_split[s2]
            if overlap:
                leakage_found = True
                key = f"{s1}<->{s2}"
                leakage_details[key] = list(overlap)[:10]  # Sample first 10
                leakage_details[f"{key}_count"] = len(overlap)
    
    return leakage_found, leakage_details


def check_text_duplicates(
    splits: Dict[str, List[PDDExample]],
) -> Tuple[bool, List[str], Dict[str, int]]:
    """Check for exact text duplicates across splits.
    
    Args:
        splits: Dict mapping split name to list of examples
        
    Returns:
        (duplicates_found, duplicate_hashes, per_split_pair_counts)
    """
    # Build hash -> (split, idx) mapping
    hash_to_locations: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    
    for split_name, examples in splits.items():
        for idx, ex in enumerate(examples):
            h = _text_hash(ex.text)
            hash_to_locations[h].append((split_name, idx))
    
    # Find hashes that appear in multiple splits
    cross_split_duplicates: List[str] = []
    per_split_pair: Dict[str, int] = defaultdict(int)
    
    for h, locations in hash_to_locations.items():
        splits_with_hash = set(loc[0] for loc in locations)
        if len(splits_with_hash) > 1:
            cross_split_duplicates.append(h)
            # Count per split pair
            split_list = sorted(splits_with_hash)
            for i, s1 in enumerate(split_list):
                for s2 in split_list[i+1:]:
                    per_split_pair[f"{s1}<->{s2}"] += 1
    
    return bool(cross_split_duplicates), cross_split_duplicates, dict(per_split_pair)


def check_length_confound(
    examples: List[PDDExample],
    threshold: float = 0.65,
) -> Tuple[float, bool, Dict[str, float]]:
    """Check if character length predicts membership.
    
    Args:
        examples: List of labeled examples
        threshold: AUROC above which confound is significant
        
    Returns:
        (auroc, is_significant, stats)
    """
    lengths = np.array([len(ex.text) for ex in examples])
    labels = np.array([ex.label for ex in examples])
    
    # Compute AUROC for length as predictor
    try:
        auroc = float(roc_auc_score(labels, lengths))
    except Exception:
        auroc = 0.5
    
    # Correlation
    if len(np.unique(labels)) > 1:
        corr = float(np.corrcoef(lengths, labels)[0, 1])
    else:
        corr = 0.0
    
    # Per-class stats
    mem_lengths = lengths[labels == 1]
    non_lengths = lengths[labels == 0]
    
    stats = {
        "member_mean": float(np.mean(mem_lengths)) if len(mem_lengths) > 0 else 0.0,
        "member_std": float(np.std(mem_lengths)) if len(mem_lengths) > 0 else 0.0,
        "nonmember_mean": float(np.mean(non_lengths)) if len(non_lengths) > 0 else 0.0,
        "nonmember_std": float(np.std(non_lengths)) if len(non_lengths) > 0 else 0.0,
        "correlation": corr,
    }
    
    # Significant if AUROC indicates predictive power
    is_significant = auroc > threshold or auroc < (1 - threshold)
    
    return auroc, is_significant, stats


def check_token_confound(
    examples: List[PDDExample],
    tokenizer,
    threshold: float = 0.65,
) -> Tuple[float, bool]:
    """Check if token count predicts membership.
    
    Args:
        examples: List of labeled examples
        tokenizer: HuggingFace tokenizer
        threshold: AUROC above which confound is significant
        
    Returns:
        (auroc, is_significant)
    """
    token_counts = []
    for ex in examples:
        toks = tokenizer.encode(ex.text, add_special_tokens=False)
        token_counts.append(len(toks))
    
    token_counts = np.array(token_counts)
    labels = np.array([ex.label for ex in examples])
    
    try:
        auroc = float(roc_auc_score(labels, token_counts))
    except Exception:
        auroc = 0.5
    
    is_significant = auroc > threshold or auroc < (1 - threshold)
    
    return auroc, is_significant


def run_integrity_audit(
    splits: Dict[str, List[PDDExample]],
    group_key: Optional[str] = "pair_idx",
    tokenizer=None,
    confound_threshold: float = 0.65,
) -> IntegrityReport:
    """Run comprehensive integrity audit on splits.
    
    Args:
        splits: Dict mapping split name ("ref", "val", "test") to examples
        group_key: Metadata key for group disjointness check (e.g., "pair_idx" for MIMIR)
        tokenizer: Optional tokenizer for token count confound check
        confound_threshold: AUROC threshold for significant confounds
        
    Returns:
        IntegrityReport with all audit results
    """
    report = IntegrityReport()
    
    # 1. Group disjointness (if group_key provided)
    if group_key:
        leakage, details = check_group_disjointness(splits, group_key)
        report.group_leakage_found = leakage
        report.group_leakage_details = details
    
    # 2. Text duplicates
    dups_found, dup_hashes, per_pair = check_text_duplicates(splits)
    report.duplicate_texts_found = dups_found
    report.duplicate_hashes = dup_hashes
    report.n_duplicates_per_split_pair = per_pair
    
    # 3. Length confound (on all examples combined)
    all_examples = []
    for exs in splits.values():
        all_examples.extend(exs)
    
    length_auroc, length_sig, length_stats = check_length_confound(
        all_examples, threshold=confound_threshold
    )
    report.length_auroc = length_auroc
    report.length_auroc_significant = length_sig
    report.length_correlation_with_label = length_stats["correlation"]
    report.member_length_mean = length_stats["member_mean"]
    report.member_length_std = length_stats["member_std"]
    report.nonmember_length_mean = length_stats["nonmember_mean"]
    report.nonmember_length_std = length_stats["nonmember_std"]
    
    # 4. Token confound (if tokenizer provided)
    if tokenizer is not None:
        token_auroc, token_sig = check_token_confound(
            all_examples, tokenizer, threshold=confound_threshold
        )
        report.token_auroc = token_auroc
        report.token_auroc_significant = token_sig
    
    return report


def compute_length_residualized_scores(
    scores: np.ndarray,
    lengths: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Residualize scores by regressing out length.
    
    Fits: score ~ length on non-members, then subtracts predicted component.
    This removes length-based signal while preserving genuine membership signal.
    
    Args:
        scores: Raw method scores [N]
        lengths: Text lengths [N]
        labels: Membership labels [N]
        
    Returns:
        (residualized_scores, diagnostics)
    """
    lengths = np.asarray(lengths).reshape(-1, 1)
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    
    # Fit on non-members only to avoid leakage
    non_mask = labels == 0
    X_non = lengths[non_mask]
    y_non = scores[non_mask]
    
    if len(y_non) < 10:
        # Not enough non-members for reliable regression
        return scores, {"skipped": True, "reason": "insufficient_nonmembers"}
    
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(X_non, y_non)
    
    # Predict and residualize
    predicted = reg.predict(lengths)
    residuals = scores - predicted
    
    # Diagnostics
    orig_auroc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.5
    resid_auroc = roc_auc_score(labels, residuals) if len(np.unique(labels)) > 1 else 0.5
    
    diagnostics = {
        "skipped": False,
        "coef": float(reg.coef_[0]),
        "intercept": float(reg.intercept_),
        "auroc_original": orig_auroc,
        "auroc_residualized": resid_auroc,
        "auroc_drop": orig_auroc - resid_auroc,
    }
    
    return residuals, diagnostics


def compute_length_stratified_auroc(
    scores: np.ndarray,
    lengths: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 5,
) -> Tuple[float, Dict[str, float]]:
    """Compute AUROC stratified by length bins.
    
    This addresses the "are you just detecting length?" critique by computing
    AUROC within length strata and averaging.
    
    Args:
        scores: Method scores [N]
        lengths: Text lengths [N]
        labels: Membership labels [N]
        n_bins: Number of length bins
        
    Returns:
        (stratified_auroc, per_bin_aurocs)
    """
    scores = np.asarray(scores)
    lengths = np.asarray(lengths)
    labels = np.asarray(labels)
    
    # Create length bins
    bin_edges = np.percentile(lengths, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1  # Include max
    
    bin_aurocs = {}
    valid_aurocs = []
    
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (lengths >= lo) & (lengths < hi)
        
        bin_scores = scores[mask]
        bin_labels = labels[mask]
        
        if len(bin_scores) < 10 or len(np.unique(bin_labels)) < 2:
            bin_aurocs[f"bin_{i}_({lo:.0f}-{hi:.0f})"] = float("nan")
            continue
        
        try:
            auroc = float(roc_auc_score(bin_labels, bin_scores))
            bin_aurocs[f"bin_{i}_({lo:.0f}-{hi:.0f})"] = auroc
            valid_aurocs.append(auroc)
        except Exception:
            bin_aurocs[f"bin_{i}_({lo:.0f}-{hi:.0f})"] = float("nan")
    
    # Average valid AUROCs
    stratified_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else 0.5
    bin_aurocs["stratified_mean"] = stratified_auroc
    
    return stratified_auroc, bin_aurocs


def verify_determinism(
    scores_run1: np.ndarray,
    scores_run2: np.ndarray,
    tolerance: float = 1e-8,
) -> Tuple[bool, Dict[str, float]]:
    """Verify that two runs produce identical scores.
    
    Args:
        scores_run1: Scores from first run
        scores_run2: Scores from second run
        tolerance: Maximum allowed difference
        
    Returns:
        (is_deterministic, diagnostics)
    """
    scores_run1 = np.asarray(scores_run1)
    scores_run2 = np.asarray(scores_run2)
    
    if scores_run1.shape != scores_run2.shape:
        return False, {"error": "shape_mismatch", "shape1": list(scores_run1.shape), "shape2": list(scores_run2.shape)}
    
    diff = np.abs(scores_run1 - scores_run2)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    n_nonzero = int((diff > tolerance).sum())
    
    is_deterministic = max_diff <= tolerance
    
    return is_deterministic, {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "n_nonzero": n_nonzero,
        "tolerance": tolerance,
        "is_deterministic": is_deterministic,
    }


def compute_confound_residualized_scores(
    scores: np.ndarray,
    confound_scores: np.ndarray,
    labels: np.ndarray,
    confound_name: str = "confound",
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Residualize method scores by regressing out an arbitrary confound.

    This generalises ``compute_length_residualized_scores``: it accepts
    *any* confound score vector (length, BoW P(member), TF-IDF P(member),
    etc.), fits a linear regression on non-members, and returns residuals.

    The key use-case is BoW-residualized AUROC: after regressing out the
    BoW classifier's predicted P(member), remaining AUROC reflects signal
    *beyond* distribution shift (Meeus et al., SaTML 2025; Das et al., 2024).

    Args:
        scores: Raw method scores [N].
        confound_scores: Confound predictor scores [N] (e.g. BoW P(member)).
        labels: Membership labels [N].
        confound_name: Label for diagnostics dict keys.

    Returns:
        (residualized_scores, diagnostics)
    """
    confound_scores = np.asarray(confound_scores).reshape(-1, 1)
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    non_mask = labels == 0
    X_non = confound_scores[non_mask]
    y_non = scores[non_mask]

    if len(y_non) < 10:
        return scores, {"skipped": True, "reason": "insufficient_nonmembers"}

    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(X_non, y_non)

    predicted = reg.predict(confound_scores)
    residuals = scores - predicted

    orig_auroc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.5
    resid_auroc = roc_auc_score(labels, residuals) if len(np.unique(labels)) > 1 else 0.5

    diagnostics = {
        "skipped": False,
        "confound": confound_name,
        "coef": float(reg.coef_[0]),
        "intercept": float(reg.intercept_),
        "auroc_original": orig_auroc,
        f"auroc_residualized_{confound_name}": resid_auroc,
        f"auroc_drop_{confound_name}": orig_auroc - resid_auroc,
    }

    return residuals, diagnostics
