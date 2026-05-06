"""Domain shift evaluation for membership inference attacks.

This module provides utilities for evaluating MIA methods under domain shift:
- Train on one benchmark's reference set, test on another benchmark
- Measure generalization vs benchmark-specific overfitting
- Report cross-benchmark transfer matrices

The goal is to assess whether a method's discriminative features are
general (transferable across domains) or benchmark-specific (potentially
artifacts of the calibration data).

For PoPets submission, domain shift analysis helps:
1. Demonstrate method robustness (if transfer works)
2. Justify benchmark-specific calibration (if transfer fails)
3. Identify potential confounds in benchmark construction
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass
class DomainShiftResult:
    """Result of domain shift evaluation for a single train/test pair."""
    
    train_domain: str  # Name of training domain/benchmark
    test_domain: str   # Name of test domain/benchmark
    
    # Performance metrics
    auroc: float
    tpr_at_1pct_fpr: float
    tpr_at_01pct_fpr: float
    
    # Sample counts
    n_train: int
    n_test: int
    n_test_members: int
    n_test_nonmembers: int
    
    # Baseline comparison (same-domain performance if available)
    same_domain_auroc: Optional[float] = None
    transfer_gap: Optional[float] = None  # same_domain - cross_domain


@dataclass
class DomainShiftMatrix:
    """Cross-domain transfer matrix for multiple benchmarks."""
    
    domains: List[str]  # List of domain names
    auroc_matrix: np.ndarray  # [n_domains, n_domains] - auroc[i,j] = train on i, test on j
    tpr_matrix: np.ndarray  # [n_domains, n_domains] - TPR@1%FPR for each pair
    
    # Summary statistics
    mean_diagonal: float  # Mean same-domain performance
    mean_off_diagonal: float  # Mean cross-domain performance
    transfer_ratio: float  # off_diagonal / diagonal (higher = better transfer)
    
    def get_transfer_gap(self, train_domain: str, test_domain: str) -> float:
        """Get AUROC gap between same-domain and cross-domain for a specific pair."""
        i = self.domains.index(train_domain)
        j = self.domains.index(test_domain)
        same_domain = self.auroc_matrix[j, j]  # Test domain's same-domain performance
        cross_domain = self.auroc_matrix[i, j]  # Cross-domain performance
        return same_domain - cross_domain


def _tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, fpr_target: float) -> float:
    """Compute TPR at a specific FPR threshold."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    mask = fpr <= fpr_target
    return float(np.max(tpr[mask])) if np.any(mask) else 0.0


def evaluate_domain_shift(
    train_data: Tuple[np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray],
    fit_and_score_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    train_domain: str,
    test_domain: str,
    same_domain_auroc: Optional[float] = None,
) -> DomainShiftResult:
    """Evaluate a method's performance under domain shift.
    
    Args:
        train_data: Tuple of (features, labels) for training domain.
        test_data: Tuple of (features, labels) for test domain.
        fit_and_score_fn: Function that takes (X_train, y_train, X_test) and returns scores.
        train_domain: Name of training domain.
        test_domain: Name of test domain.
        same_domain_auroc: Optional same-domain AUROC for comparison.
        
    Returns:
        DomainShiftResult with performance metrics.
    """
    X_train, y_train = train_data
    X_test, y_test = test_data
    
    # Get scores from the method
    scores = fit_and_score_fn(X_train, y_train, X_test)
    
    # Compute metrics
    auroc = float(roc_auc_score(y_test, scores)) if len(np.unique(y_test)) > 1 else 0.5
    tpr_1pct = _tpr_at_fpr(y_test, scores, 0.01)
    tpr_01pct = _tpr_at_fpr(y_test, scores, 0.001)
    
    transfer_gap = None
    if same_domain_auroc is not None:
        transfer_gap = same_domain_auroc - auroc
    
    return DomainShiftResult(
        train_domain=train_domain,
        test_domain=test_domain,
        auroc=auroc,
        tpr_at_1pct_fpr=tpr_1pct,
        tpr_at_01pct_fpr=tpr_01pct,
        n_train=len(y_train),
        n_test=len(y_test),
        n_test_members=int((y_test == 1).sum()),
        n_test_nonmembers=int((y_test == 0).sum()),
        same_domain_auroc=same_domain_auroc,
        transfer_gap=transfer_gap,
    )


def build_transfer_matrix(
    domain_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    fit_and_score_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> DomainShiftMatrix:
    """Build a full cross-domain transfer matrix.
    
    Args:
        domain_data: Dict mapping domain name -> (features, labels).
        fit_and_score_fn: Function that takes (X_train, y_train, X_test) and returns scores.
        
    Returns:
        DomainShiftMatrix with all pairwise results.
    """
    domains = sorted(domain_data.keys())
    n = len(domains)
    
    auroc_matrix = np.zeros((n, n))
    tpr_matrix = np.zeros((n, n))
    
    for i, train_domain in enumerate(domains):
        X_train, y_train = domain_data[train_domain]
        
        for j, test_domain in enumerate(domains):
            X_test, y_test = domain_data[test_domain]
            
            # Get scores
            scores = fit_and_score_fn(X_train, y_train, X_test)
            
            # Compute metrics
            auroc = float(roc_auc_score(y_test, scores)) if len(np.unique(y_test)) > 1 else 0.5
            tpr = _tpr_at_fpr(y_test, scores, 0.01)
            
            auroc_matrix[i, j] = auroc
            tpr_matrix[i, j] = tpr
    
    # Summary statistics
    diag = np.diag(auroc_matrix)
    off_diag_mask = ~np.eye(n, dtype=bool)
    off_diag = auroc_matrix[off_diag_mask] if n > 1 else np.array([])
    
    mean_diagonal = float(diag.mean())
    mean_off_diagonal = float(off_diag.mean()) if len(off_diag) > 0 else mean_diagonal
    transfer_ratio = mean_off_diagonal / mean_diagonal if mean_diagonal > 0 else 0.0
    
    return DomainShiftMatrix(
        domains=domains,
        auroc_matrix=auroc_matrix,
        tpr_matrix=tpr_matrix,
        mean_diagonal=mean_diagonal,
        mean_off_diagonal=mean_off_diagonal,
        transfer_ratio=transfer_ratio,
    )


def format_transfer_matrix(matrix: DomainShiftMatrix, metric: str = "auroc") -> str:
    """Format transfer matrix as a human-readable table.
    
    Args:
        matrix: DomainShiftMatrix to format.
        metric: "auroc" or "tpr" to select which matrix to display.
        
    Returns:
        Formatted string table.
    """
    data = matrix.auroc_matrix if metric == "auroc" else matrix.tpr_matrix
    domains = matrix.domains
    n = len(domains)
    
    # Determine column widths
    col_width = max(len(d) for d in domains) + 2
    col_width = max(col_width, 8)
    
    lines = []
    
    # Header
    header = " " * col_width + " | " + " | ".join(f"{d:>{col_width-2}}" for d in domains)
    lines.append(header)
    lines.append("-" * len(header))
    
    # Rows
    for i, train_domain in enumerate(domains):
        row_values = [f"{data[i, j]:.3f}" for j in range(n)]
        row = f"{train_domain:<{col_width-2}} | " + " | ".join(f"{v:>{col_width-2}}" for v in row_values)
        lines.append(row)
    
    # Summary
    lines.append("")
    lines.append(f"Mean diagonal (same-domain):   {matrix.mean_diagonal:.3f}")
    lines.append(f"Mean off-diagonal (transfer):  {matrix.mean_off_diagonal:.3f}")
    lines.append(f"Transfer ratio:                {matrix.transfer_ratio:.3f}")
    
    return "\n".join(lines)


def analyze_domain_shift(
    results: List[DomainShiftResult],
) -> Dict[str, Any]:
    """Analyze domain shift results for publication.
    
    Args:
        results: List of DomainShiftResult from pairwise evaluations.
        
    Returns:
        Analysis dict with:
          - mean_transfer_auroc: Mean cross-domain AUROC
          - mean_transfer_gap: Mean gap from same-domain
          - worst_transfer_pair: (train, test) with largest gap
          - best_transfer_pair: (train, test) with smallest gap
          - is_generalizable: Boolean indicating good transfer
    """
    if not results:
        return {}
    
    # Filter to cross-domain results only
    cross_domain = [r for r in results if r.train_domain != r.test_domain]
    
    if not cross_domain:
        return {"error": "No cross-domain results found"}
    
    aurocs = [r.auroc for r in cross_domain]
    gaps = [r.transfer_gap for r in cross_domain if r.transfer_gap is not None]
    
    # Find worst and best transfer pairs
    worst_idx = np.argmin(aurocs)
    best_idx = np.argmax(aurocs)
    
    analysis = {
        "n_pairs": len(cross_domain),
        "mean_transfer_auroc": float(np.mean(aurocs)),
        "std_transfer_auroc": float(np.std(aurocs)),
        "min_transfer_auroc": float(np.min(aurocs)),
        "max_transfer_auroc": float(np.max(aurocs)),
        "worst_transfer_pair": (cross_domain[worst_idx].train_domain, cross_domain[worst_idx].test_domain),
        "best_transfer_pair": (cross_domain[best_idx].train_domain, cross_domain[best_idx].test_domain),
    }
    
    if gaps:
        analysis["mean_transfer_gap"] = float(np.mean(gaps))
        analysis["max_transfer_gap"] = float(np.max(gaps))
        # Consider method generalizable if mean gap < 0.05 (5% AUROC)
        analysis["is_generalizable"] = analysis["mean_transfer_gap"] < 0.05
    
    return analysis
