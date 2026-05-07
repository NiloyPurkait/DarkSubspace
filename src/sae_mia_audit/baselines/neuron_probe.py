"""
Neuron-basis baseline probe + ablation.

This baseline addresses the question "could any basis do this?" by fitting a
linear probe on raw residual-stream activations and ablating the top neurons.

If SAE features outperform neuron ablation for equal k, it supports the claim
that SAEs provide more concentrated causal control over membership signals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


@dataclass
class NeuronProbeResult:
    """Result of neuron probe fitting."""
    coef: np.ndarray  # [D] coefficient vector
    intercept: float
    accuracy: float
    n_positive: int  # Number of neurons with positive contribution
    n_negative: int  # Number of neurons with negative contribution
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "intercept": self.intercept,
            "accuracy": self.accuracy,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
        }


@dataclass
class NeuronSelectionResult:
    """Result of neuron selection."""
    selected_dims: List[int]  # Indices of selected dimensions
    contributions: np.ndarray  # Contribution scores for selected dims
    mode: str  # "positive", "negative", or "topk"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_dims": self.selected_dims,
            "n_selected": len(self.selected_dims),
            "mode": self.mode,
            "mean_contribution": float(self.contributions.mean()) if len(self.contributions) > 0 else 0.0,
        }


class NeuronProbeBaseline:
    """
    Neuron-basis baseline for comparison with SAE interventions.
    
    Fits a logistic regression probe on aggregated residual stream activations
    and uses the learned weights for feature selection and ablation.
    """
    
    def __init__(
        self,
        n_features: int = 32,
        method: str = "logreg",
        C: float = 1.0,
        penalty: str = "l2",
        max_iter: int = 1000,
        seed: int = 0,
    ):
        """
        Args:
            n_features: Number of neurons to select (k for top-k selection).
            method: Selection method ("logreg" for logistic regression).
            C: Inverse regularisation strength.
            penalty: Regularization type ("l1", "l2", or "elasticnet").
            max_iter: Maximum iterations for fitting.
            seed: Random seed.
        """
        self.n_features = n_features
        self.method = method
        self.C = C
        self.penalty = penalty
        self.max_iter = max_iter
        self.seed = seed
        
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self._clf: Optional[LogisticRegression] = None
    
    def fit(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
    ) -> NeuronProbeResult:
        """
        Fit the neuron probe on calibration data.
        
        Args:
            activations: Aggregated residual stream activations [N, D].
            labels: Membership labels [N] (1=member, 0=nonmember).
        
        Returns:
            NeuronProbeResult with fit diagnostics.
        """
        # Standardize activations
        self._mean = activations.mean(axis=0)
        self._std = activations.std(axis=0) + 1e-8
        X = (activations - self._mean) / self._std
        
        # Fit logistic regression
        solver = "liblinear" if self.penalty == "l1" else "lbfgs"
        self._clf = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=solver,
            max_iter=self.max_iter,
            random_state=self.seed,
        )
        self._clf.fit(X, labels)
        
        self.coef_ = self._clf.coef_[0]
        self.intercept_ = float(self._clf.intercept_[0])
        
        # Diagnostics
        accuracy = float(self._clf.score(X, labels))
        n_positive = int((self.coef_ > 0).sum())
        n_negative = int((self.coef_ < 0).sum())
        
        return NeuronProbeResult(
            coef=self.coef_,
            intercept=self.intercept_,
            accuracy=accuracy,
            n_positive=n_positive,
            n_negative=n_negative,
        )
    
    def select_neurons(
        self,
        activations: np.ndarray,
        k: int,
        mode: str = "positive",
    ) -> NeuronSelectionResult:
        """
        Select top-k neurons by probe weight * activation contribution.
        
        Args:
            activations: Aggregated activations for a single example [D].
            k: Number of neurons to select.
            mode: "positive" (member-leaning), "negative" (nonmember-leaning), 
                  or "topk" (largest absolute).
        
        Returns:
            NeuronSelectionResult with selected dimension indices.
        """
        if self.coef_ is None:
            raise ValueError("Must call fit() before select_neurons()")
        
        # Normalize activations
        x = (activations - self._mean) / self._std
        
        # Contribution = weight * activation
        contrib = self.coef_ * x
        
        k = min(k, len(contrib))
        
        if mode == "positive":
            # Select dimensions with positive contribution (member-leaning)
            pos_mask = contrib > 0
            pos_idx = np.where(pos_mask)[0]
            if len(pos_idx) == 0:
                return NeuronSelectionResult([], np.array([]), mode)
            k = min(k, len(pos_idx))
            top_idx = pos_idx[np.argsort(-contrib[pos_idx])[:k]]
        elif mode == "negative":
            # Select dimensions with negative contribution (nonmember-leaning)
            neg_mask = contrib < 0
            neg_idx = np.where(neg_mask)[0]
            if len(neg_idx) == 0:
                return NeuronSelectionResult([], np.array([]), mode)
            k = min(k, len(neg_idx))
            top_idx = neg_idx[np.argsort(contrib[neg_idx])[:k]]
        elif mode == "topk":
            # Select dimensions with largest |contribution|
            top_idx = np.argsort(-np.abs(contrib))[:k]
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return NeuronSelectionResult(
            selected_dims=top_idx.tolist(),
            contributions=contrib[top_idx],
            mode=mode,
        )
    
    def get_global_selection(
        self,
        k: Optional[int] = None,
        mode: str = "positive",
    ) -> List[int]:
        """
        Select top-k neurons by probe weight magnitude only (no per-example activations).
        
        This provides a fixed set of neurons to ablate across all examples,
        analogous to noncircular feature selection.
        
        Args:
            k: Number of neurons to select. Defaults to self.n_features.
            mode: Selection mode ("positive", "negative", or "topk").
        
        Returns:
            List of selected neuron indices.
        """
        if self.coef_ is None:
            raise ValueError("Must call fit() before get_global_selection()")
        
        if k is None:
            k = self.n_features
        
        k = min(k, len(self.coef_))
        
        if mode == "positive":
            pos_mask = self.coef_ > 0
            pos_idx = np.where(pos_mask)[0]
            if len(pos_idx) == 0:
                return []
            k = min(k, len(pos_idx))
            top_idx = pos_idx[np.argsort(-self.coef_[pos_idx])[:k]]
        elif mode == "negative":
            neg_mask = self.coef_ < 0
            neg_idx = np.where(neg_mask)[0]
            if len(neg_idx) == 0:
                return []
            k = min(k, len(neg_idx))
            top_idx = neg_idx[np.argsort(self.coef_[neg_idx])[:k]]
        elif mode == "topk":
            top_idx = np.argsort(-np.abs(self.coef_))[:k]
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return top_idx.tolist()


def fit_neuron_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    C: float = 1.0,
    penalty: str = "l2",
    seed: int = 0,
) -> Tuple[NeuronProbeBaseline, NeuronProbeResult]:
    """
    Convenience function to fit a neuron probe.
    
    Args:
        activations: Aggregated residual stream activations [N, D].
        labels: Membership labels [N].
        C: Regularization strength.
        penalty: Regularization type.
        seed: Random seed.
    
    Returns:
        Tuple of (fitted baseline, fit result).
    """
    baseline = NeuronProbeBaseline(C=C, penalty=penalty, seed=seed)
    result = baseline.fit(activations, labels)
    return baseline, result


def select_neurons_by_probe(
    baseline: NeuronProbeBaseline,
    activations: np.ndarray,
    k: int,
    mode: str = "positive",
) -> List[int]:
    """
    Convenience function to select neurons.
    
    Args:
        baseline: Fitted NeuronProbeBaseline.
        activations: Aggregated activations [D].
        k: Number of neurons to select.
        mode: Selection mode.
    
    Returns:
        List of selected dimension indices.
    """
    result = baseline.select_neurons(activations, k, mode)
    return result.selected_dims


@torch.no_grad()
def ablate_neurons_in_hidden(
    h: torch.Tensor,
    dims: List[int],
    mode: str = "zero",
    replacement_value: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Ablate selected dimensions in hidden state tensor.
    
    Args:
        h: Hidden states [B, T, D].
        dims: Dimension indices to ablate.
        mode: "zero" (set to 0), "mean" (set to mean), or "replace" (use replacement_value).
        replacement_value: Optional tensor [D] for "replace" mode.
    
    Returns:
        Modified hidden states.
    """
    if not dims:
        return h
    
    h_abl = h.clone()
    dims_tensor = torch.tensor(dims, dtype=torch.long, device=h.device)
    
    if mode == "zero":
        h_abl[..., dims_tensor] = 0.0
    elif mode == "mean":
        # Set to mean across batch and sequence
        mean_val = h.mean(dim=(0, 1), keepdim=True)
        h_abl[..., dims_tensor] = mean_val[..., dims_tensor]
    elif mode == "replace":
        if replacement_value is None:
            raise ValueError("replacement_value required for 'replace' mode")
        h_abl[..., dims_tensor] = replacement_value[dims_tensor]
    else:
        raise ValueError(f"Unknown ablation mode: {mode}")
    
    return h_abl
