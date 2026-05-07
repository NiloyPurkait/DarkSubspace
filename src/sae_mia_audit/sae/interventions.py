"""Causal interventions on SAE features and neurons.

Provides feature-selection routines (top-k, mutual-information, correlation,
non-circular variants), ablation modes ("subtract"/"replace"), and validity
diagnostics used to ensure interventions actually act on the intended subspace.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence, List, Optional, Tuple, Dict, Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif

from .adapters import SAEProtocol


# ---------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------

AblationMode = Literal["subtract", "replace"]
FeatureSelectMode = Literal["topk", "random", "bottomk", "positive", "negative", "correlation", "mi"]


@dataclass
class FeatureSelectionDiagnostics:
    """Diagnostics for SAE feature selection.

    Captures how many features were selected, how active they actually were,
    and (for top-k) the balance of positive vs. negative contributions, so
    that downstream consumers can detect cancellation and selection-quality
    issues.
    """
    mode: str
    k: int
    n_selected: int
    # For topk mode: fraction with positive/negative contribution
    frac_positive: float = 0.0
    frac_negative: float = 0.0
    # Sum of positive and negative contributions (should be ~0 for topk if cancellation)
    sum_positive_contrib: float = 0.0
    sum_negative_contrib: float = 0.0
    net_contrib: float = 0.0
    # For non-circular modes
    selection_method: str = ""  # e.g., "w_based", "correlation", "mutual_info"
    # Active-only selection stats
    n_active: int = 0  # Number of truly active features (z > eps)
    n_masked: int = 0  # Number of features passing the mask
    active_eps: float = 0.0  # Epsilon used for active determination
    note: str = ""  # Additional diagnostic information
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "k": self.k,
            "n_selected": self.n_selected,
            "frac_positive": self.frac_positive,
            "frac_negative": self.frac_negative,
            "sum_positive_contrib": self.sum_positive_contrib,
            "sum_negative_contrib": self.sum_negative_contrib,
            "net_contrib": self.net_contrib,
            "selection_method": self.selection_method,
            "n_active": self.n_active,
            "n_masked": self.n_masked,
            "active_eps": self.active_eps,
            "note": self.note,
        }


@dataclass
class SanityCheckResult:
    """Result of a mechanistic sanity check."""
    name: str
    passed: bool
    metric: float
    threshold: float
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "metric": self.metric,
            "threshold": self.threshold,
            "details": self.details,
        }


@dataclass
class InterventionResult:
    """Result of a causal intervention experiment."""
    # Scores before/after intervention
    score_base: float
    score_edited: float
    delta_score: float
    # Loss before/after
    loss_base: Optional[float] = None
    loss_edited: Optional[float] = None
    delta_loss: Optional[float] = None
    # KL divergence between base and edited logits
    kl_divergence: Optional[float] = None
    # Which features were edited
    edited_feature_ids: List[int] = None
    edited_positions: Optional[List[int]] = None  # For token-level edits
    # Metadata
    edit_config: Dict[str, Any] = None

    def __post_init__(self):
        if self.edited_feature_ids is None:
            self.edited_feature_ids = []
        if self.edit_config is None:
            self.edit_config = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score_base": self.score_base,
            "score_edited": self.score_edited,
            "delta_score": self.delta_score,
            "loss_base": self.loss_base,
            "loss_edited": self.loss_edited,
            "delta_loss": self.delta_loss,
            "kl_divergence": self.kl_divergence,
            "edited_feature_ids": self.edited_feature_ids,
            "edited_positions": self.edited_positions,
            "edit_config": self.edit_config,
        }


# ---------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------

# Default minimum activation threshold for "active" features.
DEFAULT_ACTIVE_EPS = 1e-6


def select_sae_features(
    *,
    z_agg: torch.Tensor,
    w: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    mode: FeatureSelectMode = "topk",
    seed: int = 0,
    return_diagnostics: bool = False,
    active_eps: float = DEFAULT_ACTIVE_EPS,
    active_only: bool = True,
) -> List[int] | Tuple[List[int], FeatureSelectionDiagnostics]:
    """
    Select SAE feature indices for controlled causal ablations.

    By default, selects from the *truly active* feature set (mask & z > eps)
    to avoid selecting features with near-zero activations. This ensures
    random/bottomk baselines are meaningful controls.

    Args:
      z_agg: aggregated SAE activations [F] (nonnegative).
      w: SAE-NA-PDD weights [F].
      mask: boolean feature mask [F] (valid features).
      k: number of features to select.
      mode:
        - "topk": largest |z*w| among masked features (highest absolute contribution)
        - "bottomk": smallest |z*w| among masked features
        - "random": uniform random among masked features (seeded)
        - "positive": top-k features with POSITIVE z*w (member-leaning evidence)
        - "negative": top-k features with NEGATIVE z*w (non-member-leaning evidence)
      seed: RNG seed for reproducibility.
      return_diagnostics: If True, also return FeatureSelectionDiagnostics.
      active_eps: Minimum activation threshold for "active" features.
      active_only: If True (default), only select from features with z > active_eps.

    Returns:
      List of selected feature indices (and diagnostics if requested).
    """

    if z_agg.ndim != 1 or w.ndim != 1 or mask.ndim != 1:
        raise ValueError("z_agg, w, mask must all be 1D tensors [F]")
    if z_agg.shape != w.shape or z_agg.shape != mask.shape:
        raise ValueError(f"Shape mismatch: z_agg={z_agg.shape}, w={w.shape}, mask={mask.shape}")

    # Convert to numpy for selection logic
    z_np = z_agg.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()
    n_masked = int(mask_np.sum())
    
    # active = mask AND z_agg > eps (when active_only=True)
    if active_only:
        active_mask = mask_np & (z_np > active_eps)
    else:
        active_mask = mask_np
    n_truly_active = int(active_mask.sum())

    # Mask weights (so contrib is 0 outside the valid feature set)
    w_eff = w.clone()
    w_eff[~mask] = 0.0

    contrib = (z_agg * w_eff).detach().cpu().numpy()
    abs_contrib = np.abs(contrib)

    active = np.where(active_mask)[0]
    if len(active) == 0:
        if return_diagnostics:
            diag = FeatureSelectionDiagnostics(
                mode=mode, k=k, n_selected=0, selection_method="w_based",
                n_active=n_truly_active, n_masked=n_masked, active_eps=active_eps,
                note=f"No active features (eps={active_eps}). {n_masked} masked but {n_truly_active} with z>eps."
            )
            return [], diag
        return []

    k_orig = k
    k = min(int(k), len(active))
    if k <= 0:
        if return_diagnostics:
            diag = FeatureSelectionDiagnostics(
                mode=mode, k=k_orig, n_selected=0, selection_method="w_based",
                n_active=n_truly_active, n_masked=n_masked, active_eps=active_eps
            )
            return [], diag
        return []

    if mode == "topk":
        idx = active[np.argsort(-abs_contrib[active])[:k]]
    elif mode == "bottomk":
        # Bottom-k among ACTIVE features (not zeros)
        idx = active[np.argsort(abs_contrib[active])[:k]]
    elif mode == "random":
        # Random among ACTIVE features (not potentially-zero masked features)
        rng = np.random.default_rng(seed)
        idx = rng.choice(active, size=k, replace=False)
    elif mode == "positive":
        # Select features with positive contribution (member-leaning)
        pos_active = active[contrib[active] > 0]
        if len(pos_active) == 0:
            if return_diagnostics:
                diag = FeatureSelectionDiagnostics(
                    mode=mode, k=k_orig, n_selected=0, selection_method="w_based",
                    n_active=n_truly_active, n_masked=n_masked, active_eps=active_eps,
                    note="No features with positive contribution."
                )
                return [], diag
            return []
        k = min(k, len(pos_active))
        idx = pos_active[np.argsort(-contrib[pos_active])[:k]]
    elif mode == "negative":
        # Select features with negative contribution (non-member-leaning)
        neg_active = active[contrib[active] < 0]
        if len(neg_active) == 0:
            if return_diagnostics:
                diag = FeatureSelectionDiagnostics(
                    mode=mode, k=k_orig, n_selected=0, selection_method="w_based",
                    n_active=n_truly_active, n_masked=n_masked, active_eps=active_eps,
                    note="No features with negative contribution."
                )
                return [], diag
            return []
        k = min(k, len(neg_active))
        # Most negative first (largest negative magnitude)
        idx = neg_active[np.argsort(contrib[neg_active])[:k]]
    else:
        raise ValueError(f"Unknown feature selection mode: {mode}")

    result = idx.tolist()
    
    if return_diagnostics:
        # Compute diagnostics for the selected features
        sel_contrib = contrib[idx]
        pos_mask = sel_contrib > 0
        neg_mask = sel_contrib < 0
        
        diag = FeatureSelectionDiagnostics(
            mode=mode,
            k=k_orig,
            n_selected=len(result),
            frac_positive=float(pos_mask.sum() / len(result)) if len(result) > 0 else 0.0,
            frac_negative=float(neg_mask.sum() / len(result)) if len(result) > 0 else 0.0,
            sum_positive_contrib=float(sel_contrib[pos_mask].sum()) if pos_mask.any() else 0.0,
            sum_negative_contrib=float(sel_contrib[neg_mask].sum()) if neg_mask.any() else 0.0,
            net_contrib=float(sel_contrib.sum()),
            selection_method="w_based",
            n_active=n_truly_active,
            n_masked=n_masked,
            active_eps=active_eps,
            note=f"Selected from {n_truly_active} active features (z>{active_eps}) out of {n_masked} masked.",
        )
        return result, diag
    
    return result


def select_sae_features_noncircular(
    *,
    z_matrix: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    k: int,
    mode: Literal["correlation", "mi", "logreg"] = "correlation",
    seed: int = 0,
    return_both_directions: bool = False,
    active_eps: float = DEFAULT_ACTIVE_EPS,
) -> Tuple[List[int], FeatureSelectionDiagnostics] | Tuple[Tuple[List[int], List[int]], FeatureSelectionDiagnostics]:
    """
    Select SAE features using non-circular methods.

    These methods select features based on their relationship with membership
    labels rather than on the SAE-NA-PDD weights w, providing an independent
    check that the selected features are membership-predictive (and not merely
    re-deriving the weights they will then be evaluated against).

    Can return both member-leaning (positive) and nonmember-leaning (negative)
    feature sets for bidirectional ablation experiments.
    
    Args:
        z_matrix: SAE activations [N, F] for N examples.
        labels: Membership labels [N] (1=member, 0=non-member).
        mask: Boolean feature mask [F] (valid features).
        k: Number of features to select.
        mode:
            - "correlation": Pearson correlation with membership label
            - "mi": Mutual information with membership label (unsigned)
            - "logreg": L1-regularised logistic regression coefficients
        seed: RNG seed.
        return_both_directions: If True, return (selected_pos, selected_neg) tuple.
        active_eps: Minimum activation threshold for considering a feature active.
    
    Returns:
        If return_both_directions=False:
            Tuple of (selected feature indices, diagnostics).
        If return_both_directions=True:
            Tuple of ((selected_pos, selected_neg), diagnostics).
            selected_pos: member-leaning features
            selected_neg: nonmember-leaning features
    """
    N, F = z_matrix.shape
    if len(labels) != N:
        raise ValueError(f"Shape mismatch: z_matrix has {N} examples, labels has {len(labels)}")
    if len(mask) != F:
        raise ValueError(f"Shape mismatch: z_matrix has {F} features, mask has {len(mask)}")
    
    # Filter to truly active features (have z > eps on at least some examples)
    z_max = z_matrix.max(axis=0)
    active_mask = mask & (z_max > active_eps)
    
    active = np.where(active_mask)[0]
    n_masked = int(mask.sum())
    n_active = len(active)
    
    if n_active == 0:
        diag = FeatureSelectionDiagnostics(
            mode=mode, k=k, n_selected=0, selection_method=mode,
            n_active=0, n_masked=n_masked, active_eps=active_eps,
            note="No active features for noncircular selection."
        )
        if return_both_directions:
            return ([], []), diag
        return [], diag
    
    k = min(k, n_active)
    z_active = z_matrix[:, active]
    
    # Compute feature-label association scores
    if mode == "correlation":
        # Pearson correlation with membership label
        correlations = np.array([
            np.corrcoef(z_active[:, i], labels)[0, 1] if np.std(z_active[:, i]) > 1e-8 else 0.0
            for i in range(z_active.shape[1])
        ])
        # Handle NaN values
        correlations = np.nan_to_num(correlations, nan=0.0)
        scores = correlations
        
    elif mode == "mi":
        # Mutual information with membership label (unsigned)
        # Discretize activations for MI computation
        z_discrete = (z_active > np.median(z_active, axis=0)).astype(int)
        mi_scores = mutual_info_classif(z_discrete, labels, random_state=seed)
        # MI is unsigned; use mean difference to determine direction
        mean_member = z_active[labels == 1].mean(axis=0)
        mean_nonmember = z_active[labels == 0].mean(axis=0)
        direction = np.sign(mean_member - mean_nonmember)
        scores = mi_scores * direction  # Signed MI scores
        
    elif mode == "logreg":
        # L1-regularised logistic regression
        clf = LogisticRegression(
            penalty='l1', solver='liblinear', C=1.0, random_state=seed, max_iter=1000
        )
        # Standardize features
        z_std = (z_active - z_active.mean(axis=0)) / (z_active.std(axis=0) + 1e-8)
        clf.fit(z_std, labels)
        scores = clf.coef_[0]
    else:
        raise ValueError(f"Unknown non-circular selection mode: {mode}")
    
    if return_both_directions:
        # Select top-k member-leaning (positive scores) and top-k nonmember-leaning (negative scores)
        pos_scores = scores.copy()
        neg_scores = scores.copy()
        pos_scores[scores <= 0] = -np.inf
        neg_scores[scores >= 0] = np.inf
        
        top_pos_idx = np.argsort(-pos_scores)[:k]
        top_pos_idx = top_pos_idx[pos_scores[top_pos_idx] > -np.inf]
        selected_pos = active[top_pos_idx].tolist()
        
        top_neg_idx = np.argsort(neg_scores)[:k]
        top_neg_idx = top_neg_idx[neg_scores[top_neg_idx] < np.inf]
        selected_neg = active[top_neg_idx].tolist()
        
        diag = FeatureSelectionDiagnostics(
            mode=mode,
            k=k,
            n_selected=len(selected_pos) + len(selected_neg),
            selection_method=mode,
            n_active=n_active,
            n_masked=n_masked,
            active_eps=active_eps,
            note=f"Bidirectional: {len(selected_pos)} member-leaning, {len(selected_neg)} nonmember-leaning features.",
        )
        return (selected_pos, selected_neg), diag
    else:
        # Original behaviour: select member-leaning features only
        pos_scores = scores.copy()
        pos_scores[scores <= 0] = -np.inf
        top_idx = np.argsort(-pos_scores)[:k]
        top_idx = top_idx[pos_scores[top_idx] > -np.inf]
        selected = active[top_idx].tolist()
        
        diag = FeatureSelectionDiagnostics(
            mode=mode,
            k=k,
            n_selected=len(selected),
            selection_method=mode,
            n_active=n_active,
            n_masked=n_masked,
            active_eps=active_eps,
            note=f"Selected from {n_active} active features using {mode}.",
        )
        return selected, diag


def select_sae_features_token_level(
    *,
    z_tokens: torch.Tensor,
    w: torch.Tensor,
    mask: torch.Tensor,
    k_features: int,
    k_positions: int,
    attention_mask: Optional[torch.Tensor] = None,
    mode: FeatureSelectMode = "positive",
    seed: int = 0,
) -> List[Tuple[int, List[int]]]:
    """
    Select SAE features AND their top-activating token positions.

    This enables token-level interventions which are more precise and
    less disruptive than sequence-level interventions.

    Args:
      z_tokens: per-token SAE activations [T, F].
      w: SAE-NA-PDD weights [F].
      mask: boolean feature mask [F] (valid features).
      k_features: number of features to select.
      k_positions: number of token positions per feature.
      attention_mask: [T] mask for valid positions (1 = valid).
      mode: feature selection mode.
      seed: RNG seed.

    Returns:
      List of (feature_id, [position_indices]) tuples.
    """
    if z_tokens.ndim != 2:
        raise ValueError(f"z_tokens must be 2D [T, F], got shape {z_tokens.shape}")

    T, F = z_tokens.shape

    # Aggregate to get feature selection
    z_agg = z_tokens.mean(dim=0)  # [F]
    fids = select_sae_features(z_agg=z_agg, w=w, mask=mask, k=k_features, mode=mode, seed=seed)

    if not fids:
        return []

    # For each feature, find top-k positions by activation
    result = []
    for fid in fids:
        z_feat = z_tokens[:, fid]  # [T]

        if attention_mask is not None:
            # Mask out padding positions
            z_feat = z_feat.clone()
            z_feat[~attention_mask.bool()] = float("-inf")

        k_pos = min(k_positions, T)
        if attention_mask is not None:
            k_pos = min(k_pos, int(attention_mask.sum().item()))

        if k_pos <= 0:
            continue

        top_pos = torch.topk(z_feat, k=k_pos, largest=True).indices.tolist()
        # Filter out positions where activation is actually 0 or negative
        top_pos = [p for p in top_pos if z_tokens[p, fid] > 0]

        if top_pos:
            result.append((fid, top_pos))

    return result


# ---------------------------------------------------------------------
# Feature intervention
# ---------------------------------------------------------------------

@torch.no_grad()
def ablate_features_in_hidden(
    h: torch.Tensor,
    sae: SAEProtocol,
    feature_ids: Sequence[int],
    mode: AblationMode = "subtract",
) -> torch.Tensor:
    """
    Intervene on SAE features inside a hidden-state tensor.

    Two semantics:

    1) mode="subtract"  (recommended for mechanistic *feature removal*)
       Remove ONLY the decoded contribution of the selected features, leaving the
       rest of the hidden state unchanged:
         h' = h - (decode(z_sel) - decode(0))

       The decode(0) subtraction cancels decoder bias, making this intervention
       robust even when the decoder has an additive bias term.

    2) mode="replace"
       Project to the SAE reconstruction manifold but replace selected features
       with a baseline (mean activation):
         z_mod = encode(h); z_mod[fids] <- mean(z)[fids]; h' = decode(z_mod)

       This is more distribution-shifting than subtract and should be described
       as a "feature replacement / projection" intervention, not pure removal.

    Args:
      h: hidden states [B, T, D]
      sae: SAE module implementing encode/decode
      feature_ids: features to intervene on
      mode: {"subtract","replace"}

    Returns:
      h' with same shape and dtype as h.
    """

    if not feature_ids:
        return h

    orig_dtype = h.dtype
    device = h.device

    # Ensure SAE is on correct device
    if next(sae.parameters()).device != device:
        sae.to(device)
        sae.eval()

    B, T, D = h.shape
    x = h.reshape(B * T, D)

    sae_dtype = next(sae.parameters()).dtype
    x_fp = x.to(dtype=sae_dtype)

    # Encode
    z = sae.encode(x_fp)  # [B*T, F]

    fids = list(map(int, feature_ids))

    if mode == "subtract":
        # isolate selected features
        z_sel = torch.zeros_like(z)
        z_sel[:, fids] = z[:, fids]

        # bias-cancelled decoded contribution
        z0 = torch.zeros((1, z.shape[1]), device=z.device, dtype=z.dtype)
        base = sae.decode(z0)  # [1, D]
        delta = sae.decode(z_sel) - base  # [B*T, D]

        x_abl = x_fp - delta

    elif mode == "replace":
        z_mod = z.clone()
        z_mod[:, fids] = z.mean(dim=0, keepdim=True)[:, fids]
        x_abl = sae.decode(z_mod)

    else:
        raise ValueError(f"Unknown ablation mode: {mode}")

    # Restore dtype for the model
    h_abl = x_abl.reshape(B, T, D).to(dtype=orig_dtype)
    return h_abl


@torch.no_grad()
def ablate_features_token_level(
    h: torch.Tensor,
    sae: SAEProtocol,
    feature_positions: List[Tuple[int, List[int]]],
    mode: AblationMode = "subtract",
) -> torch.Tensor:
    """
    Intervene on SAE features at specific token positions only.

    This is a more surgical intervention than ablate_features_in_hidden,
    which ablates features across all positions.

    Args:
      h: hidden states [B, T, D] (B should be 1 for token-level)
      sae: SAE module implementing encode/decode
      feature_positions: List of (feature_id, [positions]) tuples
      mode: {"subtract","replace"}

    Returns:
      h' with same shape and dtype as h.
    """
    if not feature_positions:
        return h

    if h.shape[0] != 1:
        raise ValueError("Token-level ablation requires batch size 1")

    orig_dtype = h.dtype
    device = h.device

    # Ensure SAE is on correct device
    if next(sae.parameters()).device != device:
        sae.to(device)
        sae.eval()

    B, T, D = h.shape
    x = h.reshape(T, D)

    sae_dtype = next(sae.parameters()).dtype
    x_fp = x.to(dtype=sae_dtype)

    # Encode all positions
    z = sae.encode(x_fp)  # [T, F]

    if mode == "subtract":
        # For each (feature, positions), subtract the feature contribution at those positions
        z0 = torch.zeros((1, z.shape[1]), device=z.device, dtype=z.dtype)
        base = sae.decode(z0).squeeze(0)  # [D]

        x_abl = x_fp.clone()
        for fid, positions in feature_positions:
            for pos in positions:
                if pos < 0 or pos >= T:
                    continue
                # Isolate this feature at this position
                z_sel = torch.zeros_like(z[pos:pos+1])
                z_sel[0, fid] = z[pos, fid]
                delta = sae.decode(z_sel).squeeze(0) - base  # [D]
                x_abl[pos] = x_abl[pos] - delta

    elif mode == "replace":
        z_mod = z.clone()
        mean_z = z.mean(dim=0)  # [F]
        for fid, positions in feature_positions:
            for pos in positions:
                if pos < 0 or pos >= T:
                    continue
                z_mod[pos, fid] = mean_z[fid]
        x_abl = sae.decode(z_mod)

    else:
        raise ValueError(f"Unknown ablation mode: {mode}")

    h_abl = x_abl.reshape(B, T, D).to(dtype=orig_dtype)
    return h_abl


# ---------------------------------------------------------------------
# Mechanistic Sanity Checks (Group C mandatory diagnostics)
# ---------------------------------------------------------------------

@torch.no_grad()
def check_identity_patch(
    model,
    sae: SAEProtocol,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    layer_idx: int,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> SanityCheckResult:
    """
    Sanity check #1: Identity patch should yield identical logits.

    Patches with a no-op function and verifies logits match within tolerance.
    """
    from sae_mia_audit.models.wrapper import ActivationSite

    site = ActivationSite(layer_idx=layer_idx, tensor_name="residual_post_block")

    # Baseline forward
    out_base = model.forward(input_ids=input_ids, attention_mask=attention_mask)
    logits_base = out_base.logits.detach().float()

    # Identity patch: h -> h
    def identity_fn(h, attn_mask):
        return h

    out_patched = model.forward_with_patch(
        site=site,
        patch_fn=identity_fn,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits_patched = out_patched.logits.detach().float()

    # Check closeness
    max_diff = (logits_base - logits_patched).abs().max().item()
    passed = torch.allclose(logits_base, logits_patched, rtol=rtol, atol=atol)

    return SanityCheckResult(
        name="identity_patch",
        passed=passed,
        metric=max_diff,
        threshold=atol,
        details={
            "rtol": rtol,
            "atol": atol,
            "max_logit_diff": max_diff,
            "layer_idx": layer_idx,
        },
    )


@torch.no_grad()
def check_reconstruction_patch(
    model,
    sae: SAEProtocol,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    layer_idx: int,
    max_loss_increase: float = 0.5,
) -> SanityCheckResult:
    """
    Sanity check #2: decode(encode(h)) should minimally change loss.

    This verifies the SAE reconstruction quality in-context. A good SAE
    should not dramatically increase perplexity when we replace h with
    its reconstruction.
    """
    from sae_mia_audit.models.wrapper import ActivationSite

    device = input_ids.device
    site = ActivationSite(layer_idx=layer_idx, tensor_name="residual_post_block")

    # Ensure SAE is on correct device
    if next(sae.parameters()).device != device:
        sae.to(device)
        sae.eval()

    # Baseline loss
    out_base = model.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    loss_base = out_base.loss.item()

    # Reconstruction patch: h -> decode(encode(h))
    sae_dtype = next(sae.parameters()).dtype

    def recon_fn(h, attn_mask):
        B, T, D = h.shape
        x = h.reshape(B * T, D).to(dtype=sae_dtype)
        z = sae.encode(x)
        x_hat = sae.decode(z)
        return x_hat.reshape(B, T, D).to(dtype=h.dtype)

    out_patched = model.forward_with_patch(
        site=site,
        patch_fn=recon_fn,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    # Compute loss manually since forward_with_patch doesn't return loss
    logits_patched = out_patched.logits
    shift_logits = logits_patched[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_patched = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).item()

    delta_loss = loss_patched - loss_base
    passed = delta_loss <= max_loss_increase

    return SanityCheckResult(
        name="reconstruction_patch",
        passed=passed,
        metric=delta_loss,
        threshold=max_loss_increase,
        details={
            "loss_base": loss_base,
            "loss_patched": loss_patched,
            "delta_loss": delta_loss,
            "layer_idx": layer_idx,
        },
    )


@torch.no_grad()
def check_inactive_feature_edit(
    model,
    sae: SAEProtocol,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    layer_idx: int,
    n_inactive_features: int = 50,
    max_loss_change: float = 0.05,
    seed: int = 0,
) -> SanityCheckResult:
    """
    Sanity check #3: Editing inactive features should change almost nothing.

    For features that are not active (z_i = 0) on this input, zeroing them
    out should have negligible effect on the output.
    """
    from sae_mia_audit.models.wrapper import ActivationSite

    device = input_ids.device
    site = ActivationSite(layer_idx=layer_idx, tensor_name="residual_post_block")

    # Ensure SAE is on correct device
    if next(sae.parameters()).device != device:
        sae.to(device)
        sae.eval()

    # Baseline
    out_base = model.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    loss_base = out_base.loss.item()

    # Find inactive features
    h = model.extract(site, input_ids, attention_mask)
    B, T, D = h.shape
    sae_dtype = next(sae.parameters()).dtype
    x = h.reshape(B * T, D).to(dtype=sae_dtype)
    z = sae.encode(x)  # [B*T, F]
    z_max = z.max(dim=0).values  # [F]

    # Features that are zero everywhere
    inactive_mask = z_max <= 0
    inactive_ids = torch.where(inactive_mask)[0].cpu().numpy()

    if len(inactive_ids) == 0:
        # All features are active - this is unusual but possible
        return SanityCheckResult(
            name="inactive_feature_edit",
            passed=True,
            metric=0.0,
            threshold=max_loss_change,
            details={
                "loss_base": loss_base,
                "n_inactive_features": 0,
                "note": "All features active on this input",
            },
        )

    # Sample inactive features
    rng = np.random.default_rng(seed)
    n_sample = min(n_inactive_features, len(inactive_ids))
    sampled_inactive = rng.choice(inactive_ids, size=n_sample, replace=False).tolist()

    # Ablate inactive features (should do nothing)
    def ablate_fn(h, attn_mask):
        return ablate_features_in_hidden(h, sae, sampled_inactive, mode="subtract")

    out_patched = model.forward_with_patch(
        site=site,
        patch_fn=ablate_fn,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits_patched = out_patched.logits
    shift_logits = logits_patched[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_patched = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).item()

    delta_loss = abs(loss_patched - loss_base)
    passed = delta_loss <= max_loss_change

    return SanityCheckResult(
        name="inactive_feature_edit",
        passed=passed,
        metric=delta_loss,
        threshold=max_loss_change,
        details={
            "loss_base": loss_base,
            "loss_patched": loss_patched,
            "delta_loss": delta_loss,
            "n_inactive_features": len(inactive_ids),
            "n_sampled": n_sample,
            "layer_idx": layer_idx,
        },
    )


def run_all_sanity_checks(
    model,
    sae: SAEProtocol,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    layer_idx: int,
    seed: int = 0,
) -> List[SanityCheckResult]:
    """Run all three mandatory sanity checks and return results."""
    results = []

    results.append(check_identity_patch(
        model, sae, input_ids, attention_mask, layer_idx
    ))

    results.append(check_reconstruction_patch(
        model, sae, input_ids, attention_mask, layer_idx
    ))

    results.append(check_inactive_feature_edit(
        model, sae, input_ids, attention_mask, layer_idx, seed=seed
    ))

    return results


# ---------------------------------------------------------------------
# KL Divergence diagnostic
# ---------------------------------------------------------------------

@torch.no_grad()
def compute_kl_divergence(
    logits_base: torch.Tensor,
    logits_edited: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute KL(P_base || P_edited) averaged over valid positions.

    This measures "how disruptive was the patch" - useful for ensuring
    interventions don't catastrophically break the model.
    """
    import torch.nn.functional as F

    # Convert to log probabilities
    log_p_base = F.log_softmax(logits_base.float(), dim=-1)
    log_p_edited = F.log_softmax(logits_edited.float(), dim=-1)
    p_base = log_p_base.exp()

    # KL per position
    kl = (p_base * (log_p_base - log_p_edited)).sum(dim=-1)  # [B, T]

    if attention_mask is not None:
        mask = attention_mask.bool()
        kl = kl.masked_fill(~mask, 0.0)
        n_valid = mask.sum().clamp_min(1)
        return (kl.sum() / n_valid).item()
    else:
        return kl.mean().item()


# ---------------------------------------------------------------------
# Intervention validity metrics
# ---------------------------------------------------------------------

@dataclass
class InterventionValidityMetrics:
    """Metrics to validate that ablation is not just representation damage."""
    # Reconstruction error
    recon_error_base: float  # ||h - decode(encode(h))||
    recon_error_ablated: float  # ||h_abl - decode(encode(h_abl))||
    recon_error_delta: float  # How much worse is reconstruction after ablation
    # Norm changes
    hidden_norm_base: float
    hidden_norm_ablated: float
    hidden_norm_ratio: float  # ablated / base (should be close to 1)
    # KL divergence (if computed)
    kl_divergence: Optional[float] = None
    # Perplexity change (if computed externally)
    perplexity_base: Optional[float] = None
    perplexity_ablated: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "recon_error_base": self.recon_error_base,
            "recon_error_ablated": self.recon_error_ablated,
            "recon_error_delta": self.recon_error_delta,
            "hidden_norm_base": self.hidden_norm_base,
            "hidden_norm_ablated": self.hidden_norm_ablated,
            "hidden_norm_ratio": self.hidden_norm_ratio,
            "kl_divergence": self.kl_divergence,
            "perplexity_base": self.perplexity_base,
            "perplexity_ablated": self.perplexity_ablated,
        }
    
    def is_valid(self, max_norm_change: float = 0.5, max_recon_increase: float = 1.0) -> bool:
        """Check if intervention is within acceptable validity bounds."""
        norm_ok = abs(self.hidden_norm_ratio - 1.0) < max_norm_change
        recon_ok = self.recon_error_delta < max_recon_increase
        return norm_ok and recon_ok


@torch.no_grad()
def compute_intervention_validity(
    h_base: torch.Tensor,
    h_ablated: torch.Tensor,
    sae: SAEProtocol,
    logits_base: Optional[torch.Tensor] = None,
    logits_ablated: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> InterventionValidityMetrics:
    """
    Compute validity metrics for an ablation intervention.

    Implements the validity gate that ensures any score change reflects an
    intervention on the targeted feature subspace rather than a side effect
    of generic representation damage.
    
    Args:
        h_base: Original hidden states [B, T, D]
        h_ablated: Ablated hidden states [B, T, D]
        sae: SAE model for reconstruction error
        logits_base: Original logits (optional, for KL)
        logits_ablated: Ablated logits (optional, for KL)
        attention_mask: Attention mask [B, T]
    
    Returns:
        InterventionValidityMetrics
    """
    device = h_base.device
    sae_dtype = next(sae.parameters()).dtype
    
    # Flatten for SAE
    B, T, D = h_base.shape
    x_base = h_base.reshape(B * T, D).to(dtype=sae_dtype)
    x_ablated = h_ablated.reshape(B * T, D).to(dtype=sae_dtype)
    
    # Reconstruction errors
    z_base = sae.encode(x_base)
    x_recon_base = sae.decode(z_base)
    recon_err_base = (x_base - x_recon_base).pow(2).mean().item()
    
    z_ablated = sae.encode(x_ablated)
    x_recon_ablated = sae.decode(z_ablated)
    recon_err_ablated = (x_ablated - x_recon_ablated).pow(2).mean().item()
    
    # Norm changes
    norm_base = x_base.norm(dim=-1).mean().item()
    norm_ablated = x_ablated.norm(dim=-1).mean().item()
    norm_ratio = norm_ablated / max(norm_base, 1e-8)
    
    # KL divergence if logits provided
    kl = None
    if logits_base is not None and logits_ablated is not None:
        kl = compute_kl_divergence(logits_base, logits_ablated, attention_mask)
    
    return InterventionValidityMetrics(
        recon_error_base=recon_err_base,
        recon_error_ablated=recon_err_ablated,
        recon_error_delta=recon_err_ablated - recon_err_base,
        hidden_norm_base=norm_base,
        hidden_norm_ablated=norm_ablated,
        hidden_norm_ratio=norm_ratio,
        kl_divergence=kl,
    )