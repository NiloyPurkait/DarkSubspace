"""
Random rotation baseline for comparison with SAE-based interventions.

Phase 6.2: This stronger baseline tests whether SAE's learned basis is actually
important, or if any orthonormal basis would work equally well.

If SAE significantly outperforms random rotation, it supports the claim that
SAE features capture meaningful structure beyond arbitrary projections.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import torch
from scipy.stats import special_ortho_group


@dataclass
class RandomRotationResult:
    """Result of random rotation baseline."""
    selected_dims: List[int]  # Selected dimensions in rotated space
    n_selected: int
    mode: str
    seed: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_dims": self.selected_dims,
            "n_selected": self.n_selected,
            "mode": self.mode,
            "seed": self.seed,
        }


class RandomRotationBaseline:
    """
    Random rotation baseline for comparison with SAE interventions.
    
    Generates a random orthonormal rotation matrix, rotates activations,
    fits a probe in rotated space, and ablates in rotated coordinates
    before mapping back.
    """
    
    def __init__(self, d_hidden: int = None, d_sae: int = None, d_model: int = None, seed: int = 0):
        """
        Args:
            d_hidden: Dimension of the hidden/activation space.
            d_sae: Dimension of the SAE latent space (optional, for reference).
            d_model: Alias for d_hidden (for backward compatibility).
            seed: Random seed for reproducible rotation matrix.
        """
        # Support both old (d_model) and new (d_hidden) parameter names
        if d_hidden is not None:
            self.d_model = d_hidden
        elif d_model is not None:
            self.d_model = d_model
        else:
            raise ValueError("Must specify d_hidden or d_model")
        
        self.d_hidden = self.d_model
        self.d_sae = d_sae
        self.seed = seed
        
        # Generate random orthonormal matrix
        self.R = generate_rotation_matrix(self.d_model, seed)
        self.R_inv = self.R.T  # For orthonormal matrices, inverse = transpose
        
        # Probe coefficients (set after fitting)
        self.coef_: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
    
    def rotate(self, x: np.ndarray) -> np.ndarray:
        """Rotate activations to random basis. x: [..., D] -> [..., D]"""
        return x @ self.R
    
    def unrotate(self, y: np.ndarray) -> np.ndarray:
        """Map from rotated space back to original. y: [..., D] -> [..., D]"""
        return y @ self.R_inv
    
    def fit(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        C: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Fit a probe in rotated space.
        
        Args:
            activations: Original activations [N, D].
            labels: Membership labels [N].
            C: Regularization strength.
        
        Returns:
            Dict with fit diagnostics.
        """
        from sklearn.linear_model import LogisticRegression
        
        # Rotate to random basis
        y = self.rotate(activations)
        
        # Standardize
        self._mean = y.mean(axis=0)
        self._std = y.std(axis=0) + 1e-8
        y_std = (y - self._mean) / self._std
        
        # Fit probe
        clf = LogisticRegression(C=C, max_iter=1000, random_state=self.seed)
        clf.fit(y_std, labels)
        
        self.coef_ = clf.coef_[0]
        accuracy = float(clf.score(y_std, labels))
        
        return {
            "accuracy": accuracy,
            "n_positive": int((self.coef_ > 0).sum()),
            "n_negative": int((self.coef_ < 0).sum()),
            "seed": self.seed,
        }
    
    def select_features(
        self,
        activations: np.ndarray,
        k: int,
        mode: str = "positive",
    ) -> RandomRotationResult:
        """
        Select top-k features in rotated space.
        
        Args:
            activations: Original activations [D].
            k: Number of features to select.
            mode: "positive", "negative", or "topk".
        
        Returns:
            RandomRotationResult with selected dimension indices.
        """
        if self.coef_ is None:
            raise ValueError("Must call fit() before select_features()")
        
        # Rotate and normalize
        y = self.rotate(activations)
        y_std = (y - self._mean) / self._std
        
        # Contribution in rotated space
        contrib = self.coef_ * y_std
        
        k = min(k, len(contrib))
        
        if mode == "positive":
            pos_mask = contrib > 0
            pos_idx = np.where(pos_mask)[0]
            if len(pos_idx) == 0:
                return RandomRotationResult([], 0, mode, self.seed)
            k = min(k, len(pos_idx))
            top_idx = pos_idx[np.argsort(-contrib[pos_idx])[:k]]
        elif mode == "negative":
            neg_mask = contrib < 0
            neg_idx = np.where(neg_mask)[0]
            if len(neg_idx) == 0:
                return RandomRotationResult([], 0, mode, self.seed)
            k = min(k, len(neg_idx))
            top_idx = neg_idx[np.argsort(contrib[neg_idx])[:k]]
        elif mode == "topk":
            top_idx = np.argsort(-np.abs(contrib))[:k]
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return RandomRotationResult(
            selected_dims=top_idx.tolist(),
            n_selected=len(top_idx),
            mode=mode,
            seed=self.seed,
        )
    
    def get_global_selection(
        self,
        k: int,
        mode: str = "positive",
    ) -> RandomRotationResult:
        """
        Select top-k features by probe weight magnitude in rotated space.
        """
        if self.coef_ is None:
            raise ValueError("Must call fit() before get_global_selection()")
        
        k = min(k, len(self.coef_))
        
        if mode == "positive":
            pos_mask = self.coef_ > 0
            pos_idx = np.where(pos_mask)[0]
            if len(pos_idx) == 0:
                return RandomRotationResult([], 0, mode, self.seed)
            k = min(k, len(pos_idx))
            top_idx = pos_idx[np.argsort(-self.coef_[pos_idx])[:k]]
        elif mode == "negative":
            neg_mask = self.coef_ < 0
            neg_idx = np.where(neg_mask)[0]
            if len(neg_idx) == 0:
                return RandomRotationResult([], 0, mode, self.seed)
            k = min(k, len(neg_idx))
            top_idx = neg_idx[np.argsort(self.coef_[neg_idx])[:k]]
        elif mode == "topk":
            top_idx = np.argsort(-np.abs(self.coef_))[:k]
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return RandomRotationResult(
            selected_dims=top_idx.tolist(),
            n_selected=len(top_idx),
            mode=mode,
            seed=self.seed,
        )
    
    def compute_ablation_delta(
        self,
        activations: np.ndarray,
        selected_dims: List[int],
    ) -> np.ndarray:
        """
        Compute the delta to apply in original space for ablating selected rotated dims.
        
        Args:
            activations: Original activations [D].
            selected_dims: Dimensions to ablate in rotated space.
        
        Returns:
            Delta to subtract from original activations [D].
        """
        if not selected_dims:
            return np.zeros_like(activations)
        
        # Rotate to random basis
        y = self.rotate(activations)
        
        # Create ablation vector in rotated space (zero out selected dims)
        y_sel = np.zeros_like(y)
        y_sel[selected_dims] = y[selected_dims]
        
        # Map back to original space
        delta = self.unrotate(y_sel)
        
        return delta


def generate_rotation_matrix(d: int, seed: int = 0) -> np.ndarray:
    """
    Generate a random orthonormal rotation matrix.
    
    Uses scipy's special_ortho_group which generates matrices uniformly
    distributed on SO(d) (the special orthogonal group).
    
    Args:
        d: Dimension.
        seed: Random seed.
    
    Returns:
        Orthonormal matrix [d, d] with det = 1.
    """
    rng = np.random.default_rng(seed)
    # scipy needs old-style seed
    return special_ortho_group.rvs(d, random_state=rng.integers(0, 2**31))


def select_rotated_features(
    baseline: RandomRotationBaseline,
    activations: np.ndarray,
    k: int,
    mode: str = "positive",
) -> List[int]:
    """
    Convenience function to select features in rotated space.
    """
    result = baseline.select_features(activations, k, mode)
    return result.selected_dims


@torch.no_grad()
def ablate_rotated_features(
    h: torch.Tensor,
    baseline: RandomRotationBaseline,
    selected_dims: List[int],
) -> torch.Tensor:
    """
    Ablate selected features in rotated space.
    
    Args:
        h: Hidden states [B, T, D].
        baseline: Fitted RandomRotationBaseline.
        selected_dims: Dimensions to ablate in rotated space.
    
    Returns:
        Modified hidden states.
    """
    if not selected_dims:
        return h
    
    orig_shape = h.shape
    orig_dtype = h.dtype
    device = h.device
    
    # Flatten to [N, D]
    h_flat = h.reshape(-1, h.shape[-1]).float().cpu().numpy()
    
    # Compute delta for each position
    deltas = np.array([baseline.compute_ablation_delta(h_flat[i], selected_dims) 
                       for i in range(h_flat.shape[0])])
    
    # Apply ablation
    h_abl = h_flat - deltas
    
    # Reshape and convert back
    h_abl = torch.tensor(h_abl, dtype=orig_dtype, device=device).reshape(orig_shape)
    
    return h_abl
