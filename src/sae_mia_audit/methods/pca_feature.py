"""PCA-Feature-PDD: Supervised MIA using logistic regression over PCA-projected activations.

This is the PCA baseline counterpart to SAEFeaturePDD.  It answers the
reviewer question: *"Does the learned SAE dictionary provide detection
power beyond what principal-component projection offers?"*

Architecture (mirroring SAEFeaturePDD):
    1. Extract hidden states from specified transformer layers.
    2. Project each layer's hidden states through PCA (fitted on training data).
    3. Aggregate across token positions (mean, max, topk_mean).
    4. Concatenate PCA-projected features across layers.
    5. Train logistic regression (same solver/regularisation as SAEFeaturePDD).

If SAE-Feature significantly outperforms PCA-Feature, it supports the claim
that the *learned dictionary structure* (sparsity + overcomplete basis)
captures membership-relevant information that simple variance-maximising
directions miss.

**Threat model**: SUPERVISED — identical to SAEFeaturePDD.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

from sae_mia_audit.data.pdd import PDDExample
from sae_mia_audit.data.tokenizer import TokenizeConfig, tokenize_batch
from sae_mia_audit.models.wrapper import CausalLMWrapper
from sae_mia_audit.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class PCAFeatureConfig:
    """Configuration for PCA-Feature detection baseline.

    Parameters
    ----------
    layer_indices : list[int]
        Transformer layers from which to extract hidden states (0-indexed).
        Should match the layers used by SAE-Feature for a fair comparison.
    n_components : int or None
        Number of PCA components to retain per layer.
        - None or -1 → keep all components (= raw hidden states, full-rank probe)
        - Positive integer → reduce to this many components per layer.
        Typical values: 128, 256, 512, or match d_model for full rank.
    agg : str
        Token-level aggregation mode.  One of: 'mean', 'max', 'topk_mean'.
        Should match SAE-Feature's aggregation.
    agg_k : int
        k for topk_mean aggregation.
    seq_len : int
        Maximum sequence length for tokenization.
    batch_size : int
        Batch size for feature extraction.
    C : float
        Inverse regularisation strength for logistic regression.
    max_train : int or None
        Cap on training examples (None = use all).
    center : bool
        Whether to center PCA features before logistic regression.
    scale : bool
        Whether to scale PCA features to unit variance before logistic regression.
    pca_fit_on : str
        Which subset to fit PCA on: 'all' (members + non-members) or
        'nonmember' (non-members only, avoids leaking member structure).
    """
    layer_indices: List[int]
    n_components: Optional[int] = 256
    agg: str = "mean"
    agg_k: int = 64
    seq_len: int = 256
    batch_size: int = 4
    C: float = 1.0
    max_train: Optional[int] = None
    center: bool = True
    scale: bool = True
    pca_fit_on: str = "all"  # 'all' or 'nonmember'


class PCAFeaturePDD:
    """PCA-feature-space PDD score (logistic regression over PCA-projected codes).

    This is the PCA counterpart to SAEFeaturePDD.  Instead of encoding
    hidden states through a learned SAE dictionary, it projects through
    PCA components fitted on training data.

    The PCA transform replaces the SAE encode step; everything else
    (multi-layer extraction, token aggregation, logistic regression)
    is identical.
    """

    def __init__(self, model: CausalLMWrapper, cfg: PCAFeatureConfig, device: Optional[str] = None):
        self.model = model
        self.cfg = cfg

        model_device = next(self.model.model.parameters()).device
        self.device = torch.device(device) if device is not None else model_device
        self.layers = list(map(int, cfg.layer_indices))

        # PCA transforms per layer (fitted during fit())
        self._pcas: List[Optional[PCA]] = [None] * len(self.layers)

        # Per-layer feature statistics for centering/scaling
        self._feat_means: List[Optional[np.ndarray]] = [None] * len(self.layers)
        self._feat_stds: List[Optional[np.ndarray]] = [None] * len(self.layers)

        # Concatenated stats (fitted on all layers together)
        self._concat_mean: Optional[np.ndarray] = None
        self._concat_std: Optional[np.ndarray] = None

        self.clf: Optional[LogisticRegression] = None

    @torch.no_grad()
    def _extract_hidden_states(self, texts: Sequence[str]) -> List[np.ndarray]:
        """Extract per-layer aggregated hidden states, returning one array per layer.

        Returns list of arrays, each [N, D] where D = d_model.
        """
        tok_cfg = TokenizeConfig(seq_len=self.cfg.seq_len, random_crop=False)
        per_layer_chunks: List[List[np.ndarray]] = [[] for _ in self.layers]

        for i in tqdm(range(0, len(texts), self.cfg.batch_size),
                       desc="pca_feats", dynamic_ncols=True):
            chunk = list(texts[i : i + self.cfg.batch_size])
            batch = tokenize_batch(self.model.tokenizer, chunk, tok_cfg)
            input_ids = batch["input_ids"].to(self.model.model.device)
            attn = batch.get("attention_mask", None)
            if attn is not None:
                attn = attn.to(self.model.model.device)

            out = self.model.forward(
                input_ids=input_ids,
                attention_mask=attn,
                output_hidden_states=True,
            )
            hs = out.hidden_states

            for li, layer_idx in enumerate(self.layers):
                h = hs[layer_idx + 1]  # [B, T, D]
                B, T, D = h.shape

                # Build mask
                if attn is None:
                    m = torch.ones((B, T), device=h.device, dtype=torch.bool)
                else:
                    m = attn.to(device=h.device, dtype=torch.bool)

                z = h.float()  # [B, T, D]

                # Aggregate across tokens
                if self.cfg.agg == "mean":
                    mf = m.to(dtype=z.dtype).unsqueeze(-1)  # [B, T, 1]
                    denom = mf.sum(dim=1).clamp_min(1.0)
                    z_agg = (z * mf).sum(dim=1) / denom  # [B, D]

                elif self.cfg.agg == "max":
                    z_masked = z.masked_fill(~m.unsqueeze(-1), float("-inf"))
                    z_agg = z_masked.max(dim=1).values
                    z_agg = torch.where(torch.isfinite(z_agg), z_agg, torch.zeros_like(z_agg))

                elif self.cfg.agg == "topk_mean":
                    kk = min(self.cfg.agg_k, T)
                    z_masked = z.masked_fill(~m.unsqueeze(-1), float("-inf"))
                    vals = torch.topk(z_masked, k=kk, dim=1).values
                    vals = torch.where(torch.isfinite(vals), vals, torch.zeros_like(vals))
                    z_agg = vals.mean(dim=1)

                else:
                    raise ValueError(f"Unknown agg mode: {self.cfg.agg}")

                per_layer_chunks[li].append(z_agg.detach().cpu().numpy())

        return [np.concatenate(chunks, axis=0) for chunks in per_layer_chunks]

    def _project_pca(self, per_layer_hidden: List[np.ndarray], fit: bool = False,
                     labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Project hidden states through PCA and concatenate.

        Parameters
        ----------
        per_layer_hidden : list of [N, D] arrays
        fit : bool
            If True, fit PCA on this data.
        labels : array or None
            Needed if fit=True and pca_fit_on='nonmember'.

        Returns
        -------
        X : [N, total_features] concatenated PCA projections
        """
        projected = []

        for li, H in enumerate(per_layer_hidden):
            N, D = H.shape

            if fit:
                # Determine effective n_components
                n_comp = self.cfg.n_components
                if n_comp is None or n_comp < 0:
                    n_comp = D  # full rank
                n_comp = min(n_comp, D, N)  # PCA constraint

                pca = PCA(n_components=n_comp, random_state=0)

                # Fit on specified subset
                if self.cfg.pca_fit_on == "nonmember" and labels is not None:
                    H_fit = H[labels == 0]
                    if len(H_fit) < 2:
                        log.warning("Too few non-members for PCA fitting at layer %d, "
                                    "falling back to all data", self.layers[li])
                        H_fit = H
                else:
                    H_fit = H

                pca.fit(H_fit)
                self._pcas[li] = pca

                evr = pca.explained_variance_ratio_
                log.info("Layer %d PCA: %d components, %.1f%% variance explained",
                         self.layers[li], n_comp, 100.0 * evr.sum())

            pca = self._pcas[li]
            if pca is None:
                raise RuntimeError(f"PCA not fitted for layer index {li}")

            Z = pca.transform(H)  # [N, K]
            projected.append(Z)

        X = np.concatenate(projected, axis=1)  # [N, sum_K]
        return X

    def fit(self, examples: Sequence[PDDExample]) -> dict:
        """Fit PCA + logistic regression on training examples.

        Returns diagnostic dict with PCA/fit information.
        """
        ex = list(examples)
        if self.cfg.max_train is not None:
            ex = ex[: self.cfg.max_train]

        texts = [e.text for e in ex]
        y = np.asarray([e.label for e in ex], dtype=int)

        # 1. Extract hidden states
        per_layer = self._extract_hidden_states(texts)

        # 2. Fit PCA and project
        X = self._project_pca(per_layer, fit=True, labels=y)

        # 3. Center/scale concatenated features
        if self.cfg.center:
            self._concat_mean = X.mean(axis=0)
            X = X - self._concat_mean
        if self.cfg.scale:
            self._concat_std = X.std(axis=0) + 1e-8
            X = X / self._concat_std

        # 4. Fit logistic regression
        log.info("Fitting LogisticRegression on %d examples, %d features", X.shape[0], X.shape[1])
        clf = LogisticRegression(
            C=self.cfg.C,
            max_iter=2000,
            solver="lbfgs",
            n_jobs=1,
        )
        clf.fit(X, y)
        self.clf = clf

        # Diagnostics
        diag = {
            "n_train": len(ex),
            "n_features": X.shape[1],
            "train_accuracy": float(clf.score(X, y)),
            "layers": self.layers,
            "n_components_per_layer": [
                int(pca.n_components_) if pca is not None else 0
                for pca in self._pcas
            ],
            "explained_variance_per_layer": [
                float(pca.explained_variance_ratio_.sum()) if pca is not None else 0.0
                for pca in self._pcas
            ],
        }
        log.info("PCA-Feature fit: %s", diag)
        return diag

    def score(self, examples: Sequence[PDDExample]) -> np.ndarray:
        """Score examples using the fitted PCA-Feature classifier."""
        if self.clf is None:
            raise RuntimeError("Call fit() before score().")

        texts = [e.text for e in examples]
        per_layer = self._extract_hidden_states(texts)
        X = self._project_pca(per_layer, fit=False)

        # Apply same centering/scaling as training
        if self._concat_mean is not None:
            X = X - self._concat_mean
        if self._concat_std is not None:
            X = X / self._concat_std

        return self.clf.predict_proba(X)[:, 1]

    def top_features(self, top_k: int = 32) -> List[Tuple[str, float]]:
        """Return top positive-weight PCA components. Keys are 'layer{L}:pc{f}'."""
        if self.clf is None:
            raise RuntimeError("Call fit() before top_features().")

        w = self.clf.coef_.reshape(-1)
        idx = np.argsort(-np.abs(w))[:top_k]

        out: List[Tuple[str, float]] = []
        offset = 0
        for li, layer_idx in enumerate(self.layers):
            pca = self._pcas[li]
            K = pca.n_components_ if pca is not None else 0
            for j in idx:
                if offset <= j < offset + K:
                    fid = j - offset
                    out.append((f"layer{layer_idx}:pc{fid}", float(w[j])))
            offset += K

        return out
