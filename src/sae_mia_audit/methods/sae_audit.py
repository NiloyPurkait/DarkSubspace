from __future__ import annotations

"""SAE-Feature-PDD: Supervised MIA using logistic regression over SAE codes.

IMPORTANT — Supervised Threat Model
====================================
Unlike all other MIA baselines in this codebase (Loss, Zlib, Min-K%, Min-K%++,
ReCaLL, Con-ReCall, Neighbor, NA-PDD, SAE-NA-PDD), this method requires
**labeled** member and non-member examples to train a binary classifier.

This makes it a **supervised** attack, comparable to probing attacks in the
MIA literature.  When reporting results, always note the difference in
threat-model assumptions relative to unsupervised methods.
"""

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

from sae_mia_audit.data.pdd import PDDExample
from sae_mia_audit.data.tokenizer import TokenizeConfig, tokenize_batch
from sae_mia_audit.eval.calibration import (
    NormMode,
    NormScope,
    create_normalizer,
    select_features_by_metric,
)
from sae_mia_audit.methods.agg import AggMode, aggregate_batch
from sae_mia_audit.models.wrapper import CausalLMWrapper
from sae_mia_audit.sae.io import load_sae_checkpoint_any
from sae_mia_audit.utils.logging import get_logger
from sae_mia_audit.sae.adapters import SAEProtocol


FeatureSelectMetric = Literal["tpr@fpr", "tail_sep", "auc"]


@dataclass(frozen=True)
class SAEFeatureConfig:
    sae_paths: List[str]
    layer_indices: List[int]
    agg: AggMode = "mean"
    agg_k: Optional[int] = 8  # for topk_mean
    agg_trim_frac: Optional[float] = 0.1  # for trimmed_mean
    seq_len: int = 256
    batch_size: int = 4
    C: float = 1.0
    max_train: Optional[int] = None

    # Feature selection (default to tpr@fpr to avoid silent AUROC optimization)
    feature_select_metric: FeatureSelectMetric = "tpr@fpr"
    feature_select_fpr: float = 1e-3
    n_features_per_layer: Optional[int] = None  # None = use all

    # Score normalization (fit on non-members only)
    score_norm: NormMode = "none"
    score_norm_scope: NormScope = "feature"


def _ensure_sae_on_device_and_eval(sae: SAEProtocol, device: torch.device) -> SAEProtocol:
    """
    Robustly ensure SAE module weights are on `device` and in eval mode.
    Some loaders may still leave weights on CPU even if a device is passed.
    """
    try:
        p = next(sae.parameters())  # type: ignore[attr-defined]
        if p.device != device:
            sae = sae.to(device)  # type: ignore[union-attr]
    except Exception:
        # If SAEProtocol impl doesn't expose parameters(), rely on loader device.
        pass

    try:
        sae.eval()  # type: ignore[union-attr]
    except Exception:
        pass
    return sae


def _sae_param_dtype(sae: SAEProtocol) -> torch.dtype:
    try:
        return next(sae.parameters()).dtype  # type: ignore[attr-defined]
    except Exception:
        # Sensible default: fp32
        return torch.float32


def load_sae(path: str, device: torch.device) -> SAEProtocol:
    """Load an SAE checkpoint (repo-native or `sparse_autoencoder`) and enforce device."""
    sae = load_sae_checkpoint_any(path, device=str(device))
    sae = _ensure_sae_on_device_and_eval(sae, device)
    return sae


class SAEFeaturePDD:
    """SAE-feature-space PDD score (logistic regression over SAE codes).

    **Threat model**: SUPERVISED — requires labeled member/non-member training
    data to fit the logistic regression classifier.  This is a fundamentally
    different (and stronger) assumption than the unsupervised baselines (Loss,
    Zlib, Min-K%, ReCaLL, Con-ReCall, etc.) which only need unlabeled text and
    model access.  Results from this method should NOT be directly compared to
    unsupervised methods without clearly noting the threat-model difference.

    Supports:
    - Multiple aggregation modes (mean, max, topk_mean, trimmed_mean)
    - Score normalization fitted on non-members only
    - Feature selection by low-FPR metrics (tpr@fpr, tail_sep, auc)
    """

    def __init__(self, model: CausalLMWrapper, cfg: SAEFeatureConfig, device: Optional[str] = None):
        self.model = model
        self.cfg = cfg

        model_device = next(self.model.model.parameters()).device
        self.device = torch.device(device) if device is not None else model_device

        if len(cfg.sae_paths) != len(cfg.layer_indices):
            raise ValueError("sae_paths and layer_indices must match length")

        self.saes: List[SAEProtocol] = [load_sae(p, device=self.device) for p in cfg.sae_paths]
        self.layers = list(map(int, cfg.layer_indices))

        # Reviewer-safety: concatenating SAE bases for the same layer is almost
        # never meaningful. We allow it (for backward compatibility) but warn
        # loudly so experiments don't silently produce uninterpretable features.
        if len(set(self.layers)) != len(self.layers):
            get_logger(__name__).warning(
                "SAEFeaturePDD received duplicate layer indices. Features from multiple SAEs "
                "on the same layer will be concatenated, which is usually NOT meaningful. "
                "Prefer selecting one SAE per layer (e.g., via eval_pdd --sae-feature-selection first_per_layer)."
            )
        self.clf: Optional[LogisticRegression] = None
        self.normalizer = None
        self.selected_feature_indices: Optional[np.ndarray] = None

    @torch.no_grad()
    def _extract_features(self, texts: Sequence[str]) -> np.ndarray:
        """Extract SAE features with configurable aggregation.
        
        Uses fast masked torch operations (not slow Python loops).
        Respects attention_mask to exclude padding tokens from aggregation.
        """
        # B3: Explicit random_crop=False for deterministic evaluation
        tok_cfg = TokenizeConfig(seq_len=self.cfg.seq_len, random_crop=False)
        feats = []

        for i in tqdm(range(0, len(texts), self.cfg.batch_size), desc="sae_feats", dynamic_ncols=True):
            chunk = list(texts[i : i + self.cfg.batch_size])
            batch = tokenize_batch(self.model.tokenizer, chunk, tok_cfg)
            input_ids = batch["input_ids"].to(self.model.model.device)
            attn = batch.get("attention_mask", None)
            if attn is not None:
                attn = attn.to(self.model.model.device)

            out = self.model.forward(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
            hs = out.hidden_states
            per_layer = []

            for sae, layer_idx in zip(self.saes, self.layers):
                # hidden after layer layer_idx is hs[layer_idx+1]
                h = hs[layer_idx + 1]  # [B, T, D]
                B, T, D = h.shape

                # Ensure SAE is on correct device (robust)
                sae = _ensure_sae_on_device_and_eval(sae, self.device)
                sae_dtype = _sae_param_dtype(sae)

                # Move activations to SAE device + dtype BEFORE encode
                x = h.reshape(B * T, D).to(device=self.device, dtype=sae_dtype)

                z = sae.encode(x)  # [B*T, F]
                z = z.reshape(B, T, -1).float()  # [B, T, F]
                F_dim = z.shape[2]

                # Fast masked aggregation in torch (not slow Python loops)
                if attn is None:
                    m = torch.ones((B, T), device=z.device, dtype=torch.bool)
                else:
                    m = attn.to(device=z.device, dtype=torch.bool)

                if self.cfg.agg == "mean":
                    mf = m.to(dtype=z.dtype).unsqueeze(-1)  # [B, T, 1]
                    denom = mf.sum(dim=1).clamp_min(1.0)  # [B, 1]
                    z_agg = (z * mf).sum(dim=1) / denom  # [B, F]

                elif self.cfg.agg == "max":
                    z_masked = z.masked_fill(~m.unsqueeze(-1), float("-inf"))
                    z_agg = z_masked.max(dim=1).values
                    z_agg = torch.where(torch.isfinite(z_agg), z_agg, torch.zeros_like(z_agg))

                elif self.cfg.agg == "topk_mean":
                    kk = min(self.cfg.agg_k or 8, T)
                    z_masked = z.masked_fill(~m.unsqueeze(-1), float("-inf"))
                    # topk over time dimension
                    vals = torch.topk(z_masked, k=kk, dim=1).values  # [B, kk, F]
                    vals = torch.where(torch.isfinite(vals), vals, torch.zeros_like(vals))
                    z_agg = vals.mean(dim=1)  # [B, F]

                elif self.cfg.agg == "trimmed_mean":
                    # Fall back to numpy for trimmed_mean (less common)
                    from sae_mia_audit.methods.agg import aggregate
                    z_np = z.detach().cpu().numpy()
                    m_np = m.detach().cpu().numpy()
                    z_agg_np = np.zeros((B, F_dim), dtype=np.float32)
                    for b in range(B):
                        valid_len = int(m_np[b].sum())
                        if valid_len == 0:
                            continue
                        for f in range(F_dim):
                            z_agg_np[b, f] = aggregate(
                                z_np[b, :valid_len, f],
                                mode="trimmed_mean",
                                trim_frac=self.cfg.agg_trim_frac,
                            )
                    z_agg = torch.from_numpy(z_agg_np).to(z.device)

                else:
                    raise ValueError(f"Unknown agg mode: {self.cfg.agg}")

                per_layer.append(z_agg.detach().cpu().numpy())

            X = np.concatenate(per_layer, axis=1)
            feats.append(X)

        return np.concatenate(feats, axis=0)

    def fit(self, examples: Sequence[PDDExample]) -> None:
        """Fit the classifier with optional normalization and feature selection.
        
        Normalization is fitted on non-members only to avoid leaking member info.
        Feature selection uses the configured metric (default: tpr@fpr).
        """
        ex = list(examples)
        if self.cfg.max_train is not None:
            ex = ex[: self.cfg.max_train]
        X = self._extract_features([e.text for e in ex])
        y = np.asarray([e.label for e in ex], dtype=int)

        # Fit normalization on non-members only
        if self.cfg.score_norm != "none":
            X_nonmem = X[y == 0]
            self.normalizer = create_normalizer(
                X_nonmem,
                mode=self.cfg.score_norm,
                scope=self.cfg.score_norm_scope,
            )
            if self.normalizer is not None:
                X = self.normalizer.transform(X)

        # Feature selection by low-FPR metric (if configured)
        # Skip if n_features_per_layer is None or 0 (use all features)
        if self.cfg.n_features_per_layer is not None and self.cfg.n_features_per_layer > 0:
            n_total = self.cfg.n_features_per_layer * len(self.layers)
            X_mem = X[y == 1]
            X_nonmem = X[y == 0]
            self.selected_feature_indices, _ = select_features_by_metric(
                X_mem, X_nonmem,
                metric=self.cfg.feature_select_metric,
                n_features=n_total,
                fpr=self.cfg.feature_select_fpr,
            )
            X = X[:, self.selected_feature_indices]

        clf = LogisticRegression(
            C=self.cfg.C,
            max_iter=2000,
            solver="lbfgs",
            n_jobs=1,
        )
        clf.fit(X, y)
        self.clf = clf

    def score(self, examples: Sequence[PDDExample]) -> np.ndarray:
        """Score examples using the fitted classifier.
        
        Applies the same normalization and feature selection as during fit().
        """
        if self.clf is None:
            raise RuntimeError("Call fit() before score().")
        X = self._extract_features([e.text for e in examples])
        
        # Apply normalization (fitted on train non-members)
        if self.normalizer is not None:
            X = self.normalizer.transform(X)
        
        # Apply feature selection
        if self.selected_feature_indices is not None:
            X = X[:, self.selected_feature_indices]
        
        return self.clf.predict_proba(X)[:, 1]

    def top_features(self, top_k: int = 32) -> List[Tuple[str, float]]:
        """Return top positive-weight features (global) after training. Keys are 'layer{L}:feat{f}'."""
        if self.clf is None:
            raise RuntimeError("Call fit() before top_features().")

        w = self.clf.coef_.reshape(-1)
        idx = np.argsort(-w)[:top_k]

        out: List[Tuple[str, float]] = []
        offset = 0
        for sae, layer_idx in zip(self.saes, self.layers):
            F = int(getattr(sae, "d_sae"))
            for j in idx:
                if offset <= j < offset + F:
                    fid = j - offset
                    out.append((f"layer{layer_idx}:feat{fid}", float(w[j])))
            offset += F

        return out
