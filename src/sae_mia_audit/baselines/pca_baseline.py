"""
PCA baseline for comparison with SAE-based interventions.

This baseline addresses the question "Do SAE features outperform PCA
directions (the top principal components of hidden-state variance)?"

If SAE features significantly outperform PCA, it supports the claim that
*sparsity* and *dictionary learning* capture membership-relevant structure
that simple variance-maximizing directions miss.

If PCA matches SAE, the interpretability benefit of SAE may still justify
their use, but the MIA performance advantage would not be attributable to
learned dictionary structure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class PCABaselineResult:
    """Result of PCA baseline feature selection."""
    selected_dims: List[int]
    n_selected: int
    mode: str
    explained_variance_ratio: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_dims": self.selected_dims,
            "n_selected": self.n_selected,
            "mode": self.mode,
            "explained_variance_ratio": self.explained_variance_ratio,
        }


class PCABaseline:
    """
    PCA baseline for comparison with SAE interventions.

    Projects activations onto principal components, fits a probe in
    PCA space, and supports ablation in PCA coordinates mapped back
    to the original basis.

    Unlike RandomRotationBaseline (which uses an arbitrary orthonormal
    basis), PCA uses the *data-dependent* directions of maximum variance.
    This is the strongest unsupervised linear baseline — if SAE does not
    beat PCA, the learned dictionary is not adding value over simple
    variance decomposition.
    """

    def __init__(self, d_hidden: int, n_components: Optional[int] = None, seed: int = 0):
        """
        Args:
            d_hidden: Dimension of the hidden/activation space.
            n_components: Number of PCA components to keep (None = all).
            seed: Random seed for reproducibility in probe fitting.
        """
        self.d_hidden = d_hidden
        self.n_components = n_components or d_hidden
        self.seed = seed

        # PCA transform (set after fit_pca)
        self.components_: Optional[np.ndarray] = None  # [n_components, D]
        self.mean_: Optional[np.ndarray] = None  # [D]
        self.explained_variance_ratio_: Optional[np.ndarray] = None

        # Probe coefficients (set after fit_probe)
        self.coef_: Optional[np.ndarray] = None
        self._probe_mean: Optional[np.ndarray] = None
        self._probe_std: Optional[np.ndarray] = None

    def fit_pca(self, activations: np.ndarray) -> Dict[str, Any]:
        """Fit PCA on activations (member + non-member).

        Args:
            activations: Hidden-state matrix [N, D].

        Returns:
            Dict with PCA diagnostics.
        """
        from sklearn.decomposition import PCA

        pca = PCA(n_components=self.n_components, random_state=self.seed)
        pca.fit(activations)

        self.components_ = pca.components_  # [K, D]
        self.mean_ = pca.mean_  # [D]
        self.explained_variance_ratio_ = pca.explained_variance_ratio_

        return {
            "n_components": int(self.components_.shape[0]),
            "total_explained_variance": float(self.explained_variance_ratio_.sum()),
            "top5_explained": self.explained_variance_ratio_[:5].tolist(),
        }

    def project(self, x: np.ndarray) -> np.ndarray:
        """Project activations to PCA space. x: [..., D] -> [..., K]"""
        if self.components_ is None:
            raise ValueError("Call fit_pca() first")
        return (x - self.mean_) @ self.components_.T

    def unproject(self, y: np.ndarray) -> np.ndarray:
        """Map from PCA space back to original. y: [..., K] -> [..., D]"""
        if self.components_ is None:
            raise ValueError("Call fit_pca() first")
        return y @ self.components_ + self.mean_

    def fit(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        C: float = 1.0,
    ) -> Dict[str, Any]:
        """Fit PCA (if not done) and probe in PCA space.

        Args:
            activations: Original activations [N, D].
            labels: Membership labels [N].
            C: Regularization strength for logistic regression.

        Returns:
            Dict with fit diagnostics.
        """
        from sklearn.linear_model import LogisticRegression

        # Fit PCA if not already done
        if self.components_ is None:
            self.fit_pca(activations)

        # Project to PCA space
        y = self.project(activations)

        # Standardize
        self._probe_mean = y.mean(axis=0)
        self._probe_std = y.std(axis=0) + 1e-8
        y_std = (y - self._probe_mean) / self._probe_std

        # Fit probe
        clf = LogisticRegression(C=C, max_iter=1000, random_state=self.seed)
        clf.fit(y_std, labels)

        self.coef_ = clf.coef_[0]
        accuracy = float(clf.score(y_std, labels))

        return {
            "accuracy": accuracy,
            "n_positive": int((self.coef_ > 0).sum()),
            "n_negative": int((self.coef_ < 0).sum()),
            "n_components": int(self.components_.shape[0]),
            "total_explained_variance": float(self.explained_variance_ratio_.sum()),
        }

    def get_global_selection(
        self,
        k: int,
        mode: str = "positive",
    ) -> PCABaselineResult:
        """Select top-k PCA components by probe weight magnitude."""
        if self.coef_ is None:
            raise ValueError("Must call fit() before get_global_selection()")

        k = min(k, len(self.coef_))

        if mode == "positive":
            pos_mask = self.coef_ > 0
            pos_idx = np.where(pos_mask)[0]
            if len(pos_idx) == 0:
                return PCABaselineResult([], 0, mode)
            k = min(k, len(pos_idx))
            top_idx = pos_idx[np.argsort(-self.coef_[pos_idx])[:k]]
        elif mode == "negative":
            neg_mask = self.coef_ < 0
            neg_idx = np.where(neg_mask)[0]
            if len(neg_idx) == 0:
                return PCABaselineResult([], 0, mode)
            k = min(k, len(neg_idx))
            top_idx = neg_idx[np.argsort(self.coef_[neg_idx])[:k]]
        elif mode == "topk":
            top_idx = np.argsort(-np.abs(self.coef_))[:k]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        evr = self.explained_variance_ratio_[top_idx].tolist() if self.explained_variance_ratio_ is not None else None

        return PCABaselineResult(
            selected_dims=top_idx.tolist(),
            n_selected=len(top_idx),
            mode=mode,
            explained_variance_ratio=evr,
        )

    def compute_ablation_delta(
        self,
        activations: np.ndarray,
        selected_dims: List[int],
    ) -> np.ndarray:
        """Compute delta to subtract from original space for ablating selected PCA dims.

        Args:
            activations: Original activations [D].
            selected_dims: PCA component indices to ablate.

        Returns:
            Delta to subtract from original activations [D].
        """
        if not selected_dims:
            return np.zeros_like(activations)

        # Project to PCA space
        y = self.project(activations)

        # Create ablation vector (zero-out selected dims)
        y_sel = np.zeros_like(y)
        y_sel[selected_dims] = y[selected_dims]

        # Map back to original space (without mean offset since we want a delta)
        delta = y_sel @ self.components_

        return delta


import torch


@torch.no_grad()
def ablate_pca_features(
    h: torch.Tensor,
    baseline: PCABaseline,
    selected_dims: List[int],
) -> torch.Tensor:
    """Ablate selected PCA components from hidden states.

    Args:
        h: Hidden states [B, T, D].
        baseline: Fitted PCABaseline.
        selected_dims: PCA component indices to ablate.

    Returns:
        Modified hidden states with selected PCA directions removed.
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
