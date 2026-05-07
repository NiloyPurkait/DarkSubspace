#!/usr/bin/env python3
"""
behavioral_channels.py.

Estimates the knowledge subspace S_K and recall subspace S_R from mean-pooled
residual stream activations via covariance-difference contrastive PCA, and
emits per-layer cosine, principal angles, and cross-validated probe AUROC for
membership and recall to ``runs/dark_subspace/behavioral_channels/<condition>/orthogonality.json``.

Used in Methods §3.2 (channel decomposition), Results §4.1 (geometry), and
Appendix `app:bcd_details`.
Reproduce:
    env/bin/python3 scripts/dark_subspace/behavioral_channels.py \\
        --model-path runs/controlled_ft/run_20260306_055225/ft_epoch5/model \\
        --member-texts data/memcirc_ctrl_ft/member.jsonl \\
        --nonmember-texts data/memcirc_ctrl_ft/nonmember.jsonl \\
        --extractability-json runs/dark_subspace/extractability_control_v2/p69_epoch5_layer16/extractability_control.json \\
        --sae-path runs/sae/memcirc_p69_epoch5_layer16_8x_l1_1e4_member/sae_final.pt \\
        --feature-classification runs/dark_subspace/discovery/p69_epoch5_layer16_8x_l1_1e4_member_story/feature_classification.json \\
        --sae-layer 16 --layers 0 4 8 12 16 20 24 28 31 \\
        --output-dir runs/dark_subspace/behavioral_channels/p69_epoch5

Method outline.
  1. Forward-pass member and nonmember texts, collect residual-stream activations.
  2. Contrastive mean d_K = mean(H_mem) - mean(H_nonmem) (knowledge direction).
  3. For high-recall vs low-recall members, d_R (recall direction).
  4. Contrastive PCA on covariance difference yields S_K, S_R subspaces.
  5. Principal angle analysis between S_K and S_R.
  6. SAE feature alignment, project CF decoder vectors onto S_K, S_R.
  7. Feature Sufficiency Criterion FSC_K, FSC_R.

Outputs.
    directions.npz          d_K, d_R per layer, S_K, S_R basis vectors.
    orthogonality.json      principal angles, cosine similarities, per layer.
    sae_alignment.json      FSC_K, FSC_R, per-feature projections onto S_K/S_R.
    linear_probes.json      probe accuracy for membership, extractability.
    config.json             experiment config.
    summary.json            headline numbers (angles, FSC, probe accuracy).
"""

import _bootstrap  # noqa: F401

import argparse
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.linalg import subspace_angles
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from tqdm.auto import tqdm

from sae_mia_audit.models.wrapper import load_model_and_tokenizer, CausalLMWrapper
from sae_mia_audit.sae.io import load_sae_checkpoint_any
from sae_mia_audit.sae.adapters import SAEProtocol
from sae_mia_audit.data.tokenizer import TokenizeConfig, tokenize_batch
from sae_mia_audit.utils.hf import HFModelSpec
from sae_mia_audit.utils.seed import SeedConfig, set_global_seed
from sae_mia_audit.utils.logging import setup_logging, get_logger

log = get_logger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────────

def _load_texts(path: str, max_n: Optional[int] = None) -> List[str]:
    """Load texts from JSONL file."""
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            texts.append(json.loads(line)["text"])
            if max_n is not None and len(texts) >= max_n:
                break
    return texts


def _batched(items, n):
    for i in range(0, len(items), n):
        yield items[i : i + n]


@torch.no_grad()
def collect_residual_activations(
    model: CausalLMWrapper,
    texts: List[str],
    layers: List[int],
    seq_len: int,
    batch_size: int,
    device: str,
) -> Dict[int, np.ndarray]:
    """Collect per-text mean residual-stream activations at specified layers.

    Returns:
        Dict mapping layer_idx → np.ndarray of shape (n_texts, d_model)
    """
    tok_cfg = TokenizeConfig(seq_len=seq_len, random_crop=False)
    # Initialize accumulators
    layer_acts = {l: [] for l in layers}

    for chunk in tqdm(
        list(_batched(texts, batch_size)),
        desc="Collecting activations",
        dynamic_ncols=True,
    ):
        batch = tokenize_batch(model.tokenizer, chunk, tok_cfg)
        input_ids = batch["input_ids"].to(device)
        attn = batch.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(device)

        out = model.forward(
            input_ids=input_ids,
            attention_mask=attn,
            output_hidden_states=True,
        )
        # hidden_states: tuple of (n_layers+1,) tensors of shape (B, T, d_model)
        for l in layers:
            h = out.hidden_states[l]  # (B, T, d_model)
            if attn is not None:
                mask = attn.unsqueeze(-1).float()
                h_mean = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                h_mean = h.mean(dim=1)
            layer_acts[l].append(h_mean.cpu().float().numpy())

    return {l: np.concatenate(acts, axis=0) for l, acts in layer_acts.items()}


def contrastive_pca(
    H_pos: np.ndarray,
    H_neg: np.ndarray,
    n_components: int = 10,
    center: bool = True,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contrastive PCA: find directions that maximally separate pos from neg.

    Uses the covariance-difference method (Abid et al. 2018): eigendecompose
    C_pos - alpha * C_neg to find directions of high variance in the positive
    class and low variance in the negative class.

    Args:
        H_pos: (n_pos, d) activations for positive class
        H_neg: (n_neg, d) activations for negative class
        n_components: number of principal components to return
        center: whether to center the data (always True for correct cPCA)
        alpha: contrastive strength; larger alpha penalizes negative variance more

    Returns:
        (components, explained_variance_ratio, mean_diff)
        - components: (n_components, d) top eigenvectors of C_pos - alpha * C_neg
        - explained_variance_ratio: (n_components,) |eigenvalue| / sum(|eigenvalues|)
        - mean_diff: (d,) mean difference vector (contrastive mean)
    """
    mu_pos = H_pos.mean(axis=0)
    mu_neg = H_neg.mean(axis=0)
    mean_diff = mu_pos - mu_neg

    # Covariance matrices
    H_pos_c = H_pos - mu_pos
    H_neg_c = H_neg - mu_neg
    C_pos = (H_pos_c.T @ H_pos_c) / max(len(H_pos) - 1, 1)
    C_neg = (H_neg_c.T @ H_neg_c) / max(len(H_neg) - 1, 1)

    # Contrastive covariance
    C_contrast = C_pos - alpha * C_neg

    # Eigendecomposition (want largest eigenvalues)
    eigenvalues, eigenvectors = np.linalg.eigh(C_contrast)
    # eigh returns ascending order, we want descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Top components
    components = eigenvectors[:, :n_components].T  # (n_components, d)
    total_var = np.sum(np.abs(eigenvalues))
    explained_ratio = np.abs(eigenvalues[:n_components]) / max(total_var, 1e-12)

    return components, explained_ratio, mean_diff


def contrastive_pca_paired(
    H_pos: np.ndarray,
    H_neg: np.ndarray,
    n_components: int = 10,
    center: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LEGACY paired-difference cPCA (kept for backward compatibility).

    Note: this legacy implementation computes PCA on
    ``D = H_pos[:n] - H_neg[:n]`` where the pairing is arbitrary (the data
    are not truly paired). The resulting PCA captures pooled within-class
    variance noise rather than discriminative directions, so prefer
    :func:`contrastive_pca` (covariance-difference method) for new work.

    Args:
        H_pos: (n_pos, d) activations for positive class
        H_neg: (n_neg, d) activations for negative class
        n_components: number of principal components to return
        center: whether to center the data

    Returns:
        (components, explained_variance_ratio, mean_diff)
    """
    # Contrastive mean
    mu_pos = H_pos.mean(axis=0)
    mu_neg = H_neg.mean(axis=0)
    mean_diff = mu_pos - mu_neg

    # Center both around global mean, then compute difference distribution
    global_mean = np.concatenate([H_pos, H_neg], axis=0).mean(axis=0)

    if center:
        H_pos_c = H_pos - global_mean
        H_neg_c = H_neg - global_mean
    else:
        H_pos_c = H_pos
        H_neg_c = H_neg

    # Difference matrix: for paired analysis, sample min(n_pos, n_neg) pairs
    n = min(len(H_pos_c), len(H_neg_c))
    D = H_pos_c[:n] - H_neg_c[:n]  # (n, d) difference vectors

    # PCA on difference distribution
    D_centered = D - D.mean(axis=0)
    U, S, Vt = np.linalg.svd(D_centered, full_matrices=False)
    total_var = (S ** 2).sum()
    explained_ratio = (S[:n_components] ** 2) / max(total_var, 1e-12)

    components = Vt[:n_components]  # (n_components, d)
    return components, explained_ratio, mean_diff


def principal_angles_between_subspaces(
    A: np.ndarray, B: np.ndarray,
) -> np.ndarray:
    """Compute principal angles between two subspaces.

    Args:
        A: (k, d) orthonormal basis for subspace A (rows are basis vectors)
        B: (m, d) orthonormal basis for subspace B

    Returns:
        Array of min(k, m) principal angles in radians.
    """
    # scipy.linalg.subspace_angles expects column-major (d, k) matrices
    angles = subspace_angles(A.T, B.T)
    return angles


def compute_fsc(
    subspace_basis: np.ndarray,
    feature_directions: np.ndarray,
) -> float:
    """Feature Sufficiency Criterion: how much of the subspace is spanned by features.

    FSC = ||P_F(S)||_F^2 / ||S||_F^2

    where P_F is projection onto the feature subspace, S is the target subspace.

    Args:
        subspace_basis: (k, d) orthonormal basis for target subspace
        feature_directions: (n_features, d) feature decoder directions (need not be orthonormal)

    Returns:
        FSC value in [0, 1]. 1.0 = features fully span the subspace.
    """
    if len(feature_directions) == 0:
        return 0.0

    # Orthonormalize feature directions via QR
    Q, _ = np.linalg.qr(feature_directions.T)  # (d, min(n_feat, d))
    # Q columns are orthonormal basis for feature subspace

    # Project each subspace basis vector onto feature subspace
    # P_F(s_i) = Q @ Q^T @ s_i
    projections = subspace_basis @ Q  # (k, min(n_feat, d))
    proj_norms_sq = (projections ** 2).sum()

    # ||S||_F^2 = k (since subspace_basis is orthonormal)
    subspace_norm_sq = subspace_basis.shape[0]

    return float(proj_norms_sq / max(subspace_norm_sq, 1e-12))


def train_linear_probe(
    X: np.ndarray, y: np.ndarray, n_folds: int = 5,
) -> Dict[str, float]:
    """Train a logistic regression probe and return cross-validated accuracy."""
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    scores = cross_val_score(clf, X, y, cv=n_folds, scoring="roc_auc")
    return {
        "auroc_mean": float(scores.mean()),
        "auroc_std": float(scores.std()),
        "auroc_per_fold": [float(s) for s in scores],
    }


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Behavioral Channel Decomposition: discover S_K and S_R subspaces"
    )
    parser.add_argument("--model-path", required=True, help="Path to FT model checkpoint")
    parser.add_argument("--member-texts", required=True, help="JSONL of member texts")
    parser.add_argument("--nonmember-texts", required=True, help="JSONL of nonmember texts")
    parser.add_argument("--extractability-json", default=None,
                        help="JSON with per-text extractability scores (for S_R). "
                             "If not provided, uses loss-based proxy for recall signal.")
    parser.add_argument("--sae-path", default=None, help="SAE checkpoint for alignment analysis")
    parser.add_argument("--feature-classification", default=None,
                        help="Feature classification JSON (for CF feature list)")
    parser.add_argument("--sae-layer", type=int, default=None,
                        help="Layer where SAE was trained (for SAE alignment analysis)")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layers to analyse. If not set, auto-selects evenly spaced layers.")
    parser.add_argument("--n-components", type=int, default=10,
                        help="Number of PCA components for subspace estimation")
    parser.add_argument("--cpca-method", choices=["covariance", "paired"], default="covariance",
                        help="Contrastive PCA method: 'covariance' (Abid et al. 2018, default) "
                             "or 'paired' (legacy paired-difference, kept for backward compat)")
    parser.add_argument("--cpca-alpha", type=float, default=1.0,
                        help="Contrastive strength for covariance-difference cPCA (default: 1.0)")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-texts", type=int, default=None,
                        help="Max texts per class (member/nonmember). Default: use all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--revision", default=None,
                        help="Model revision (e.g. 'step143000' for Pythia checkpoints)")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    setup_logging(logging.INFO)
    set_global_seed(SeedConfig(seed=args.seed))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

    # ── Load model ──────────────────────────────────────────────────
    log.info(f"Loading model from {args.model_path}")
    spec = HFModelSpec(name_or_path=args.model_path, torch_dtype="bfloat16",
                       revision=args.revision)
    wrapper = load_model_and_tokenizer(spec)
    n_layers = wrapper.model.config.num_hidden_layers
    d_model = wrapper.model.config.hidden_size
    log.info(f"Model loaded: {n_layers} layers, d_model={d_model}")

    # Determine layers to analyse
    if args.layers is not None:
        layers = sorted(args.layers)
    else:
        # Auto-select ~10 evenly spaced layers including first, middle, last
        step = max(1, n_layers // 9)
        layers = sorted(set([0] + list(range(0, n_layers, step)) + [n_layers - 1]))
    if args.sae_layer is not None and args.sae_layer not in layers:
        layers.append(args.sae_layer)
        layers = sorted(layers)
    log.info(f"Analyzing layers: {layers}")

    # ── Load texts ──────────────────────────────────────────────────
    log.info("Loading texts...")
    member_texts = _load_texts(args.member_texts, args.max_texts)
    nonmember_texts = _load_texts(args.nonmember_texts, args.max_texts)
    log.info(f"Loaded {len(member_texts)} member, {len(nonmember_texts)} nonmember texts")

    # ── Determine extractability labels for recall subspace ─────────
    # We need to split members into "high-recall" and "low-recall" groups.
    # Option A: Use per-text extractability from a prior control experiment.
    # Option B: Use loss-based proxy (lower loss = more memorised → likely more extractable).
    extract_labels = None
    if args.extractability_json is not None:
        log.info(f"Loading extractability data from {args.extractability_json}")
        with open(args.extractability_json) as f:
            ext_data = json.load(f)
        # Try to get per-text scores from the NO_HOOK condition
        # If per-text data isn't available, fall back to loss-based proxy
        if "per_text_scores" in ext_data:
            scores = ext_data["per_text_scores"]
            # Median split
            median = np.median(scores)
            extract_labels = np.array([1 if s > median else 0 for s in scores])
            log.info(f"Using per-text extractability scores (median={median:.4f})")

    if extract_labels is None:
        # Loss-based proxy: compute per-text loss to identify memorised vs non-memorised members
        log.info("Computing loss-based proxy for extractability...")
        tok_cfg = TokenizeConfig(seq_len=args.seq_len, random_crop=False)
        losses = []
        with torch.no_grad():
            for chunk in _batched(member_texts, args.batch_size):
                batch = tokenize_batch(wrapper.tokenizer, chunk, tok_cfg)
                ids = batch["input_ids"].to(args.device)
                attn = batch.get("attention_mask", None)
                if attn is not None:
                    attn = attn.to(args.device)
                out = wrapper.forward(input_ids=ids, attention_mask=attn)
                # Per-example loss via manual computation
                logits = out.logits[:, :-1, :].contiguous()
                labels = ids[:, 1:].contiguous()
                for b in range(logits.shape[0]):
                    log_probs = torch.nn.functional.log_softmax(logits[b], dim=-1)
                    token_losses = -log_probs[range(labels.shape[1]), labels[b]]
                    if attn is not None:
                        mask = attn[b, 1:].float()
                        mean_loss = (token_losses * mask).sum() / mask.sum().clamp(min=1)
                    else:
                        mean_loss = token_losses.mean()
                    losses.append(mean_loss.item())

        losses = np.array(losses)
        # Lower loss = more memorised = likely higher recall
        median_loss = np.median(losses)
        extract_labels = np.array([1 if l < median_loss else 0 for l in losses])
        log.info(f"Loss-based proxy: median_loss={median_loss:.4f}, "
                 f"n_high_recall={extract_labels.sum()}, n_low_recall={len(extract_labels) - extract_labels.sum()}")

    # ── Collect activations ─────────────────────────────────────────
    log.info("Collecting member activations...")
    H_mem = collect_residual_activations(
        wrapper, member_texts, layers, args.seq_len, args.batch_size, args.device
    )
    log.info("Collecting nonmember activations...")
    H_nonmem = collect_residual_activations(
        wrapper, nonmember_texts, layers, args.seq_len, args.batch_size, args.device
    )

    # ── Per-layer analysis ──────────────────────────────────────────
    results_per_layer = {}
    directions_data = {}

    # Select cPCA implementation
    if args.cpca_method == "covariance":
        cpca_fn = lambda hp, hn, nc: contrastive_pca(hp, hn, n_components=nc, alpha=args.cpca_alpha)
        log.info(f"Using covariance-difference cPCA (Abid et al. 2018), alpha={args.cpca_alpha}")
    else:
        cpca_fn = lambda hp, hn, nc: contrastive_pca_paired(hp, hn, n_components=nc)
        log.info("Using LEGACY paired-difference cPCA (not recommended)")

    for l in layers:
        log.info(f"\n{'='*60}")
        log.info(f"Layer {l}")
        log.info(f"{'='*60}")

        h_m = H_mem[l]     # (n_mem, d_model)
        h_nm = H_nonmem[l]  # (n_nonmem, d_model)

        # 1. Knowledge subspace (S_K): member vs nonmember
        log.info("Computing knowledge subspace S_K (member vs nonmember)...")
        S_K_basis, S_K_var, d_K = cpca_fn(h_m, h_nm, args.n_components)

        # 2. Recall subspace (S_R): high-recall vs low-recall members
        h_high = h_m[extract_labels == 1]
        h_low = h_m[extract_labels == 0]
        log.info(f"Computing recall subspace S_R ({h_high.shape[0]} high-recall vs {h_low.shape[0]} low-recall)...")
        S_R_basis, S_R_var, d_R = cpca_fn(h_high, h_low, args.n_components)

        # 3. Orthogonality analysis
        angles = principal_angles_between_subspaces(S_K_basis, S_R_basis)
        angles_deg = np.degrees(angles)

        # Cosine similarity between mean directions
        cos_means = float(np.dot(d_K, d_R) / (np.linalg.norm(d_K) * np.linalg.norm(d_R) + 1e-12))

        # 4. Linear probes
        log.info("Training membership probe (S_K quality check)...")
        X_probe = np.concatenate([h_m, h_nm], axis=0)
        y_probe = np.array([1] * len(h_m) + [0] * len(h_nm))
        membership_probe = train_linear_probe(X_probe, y_probe)

        log.info("Training recall probe (S_R quality check)...")
        recall_probe = train_linear_probe(h_m, extract_labels)

        # 5. Store results
        layer_result = {
            "layer": l,
            "d_K_norm": float(np.linalg.norm(d_K)),
            "d_R_norm": float(np.linalg.norm(d_R)),
            "cosine_d_K_d_R": cos_means,
            "principal_angles_deg": angles_deg.tolist(),
            "mean_principal_angle_deg": float(angles_deg.mean()),
            "min_principal_angle_deg": float(angles_deg.min()),
            "max_principal_angle_deg": float(angles_deg.max()),
            "S_K_variance_explained": S_K_var.tolist(),
            "S_R_variance_explained": S_R_var.tolist(),
            "S_K_cumulative_var": float(S_K_var.sum()),
            "S_R_cumulative_var": float(S_R_var.sum()),
            "membership_probe": membership_probe,
            "recall_probe": recall_probe,
        }
        results_per_layer[str(l)] = layer_result

        # Store direction data for .npz
        directions_data[f"d_K_layer{l}"] = d_K.astype(np.float32)
        directions_data[f"d_R_layer{l}"] = d_R.astype(np.float32)
        directions_data[f"S_K_basis_layer{l}"] = S_K_basis.astype(np.float32)
        directions_data[f"S_R_basis_layer{l}"] = S_R_basis.astype(np.float32)

        log.info(f"Layer {l}: cos(d_K, d_R)={cos_means:.4f}, "
                 f"angles={angles_deg.min():.1f}deg to {angles_deg.max():.1f}deg, "
                 f"membership_auroc={membership_probe['auroc_mean']:.3f}, "
                 f"recall_auroc={recall_probe['auroc_mean']:.3f}")

    # ── SAE Alignment Analysis (only at SAE layer) ──────────────────
    sae_alignment = None
    if args.sae_path is not None and args.sae_layer is not None:
        log.info(f"\nSAE alignment analysis at layer {args.sae_layer}...")
        sae = load_sae_checkpoint_any(args.sae_path, device=args.device)
        sae.eval()

        # Get decoder directions. decoder_weight is [d_model, d_sae], transpose to [d_sae, d_model]
        W_dec = sae.decoder_weight.detach().cpu().float().numpy().T  # (d_sae, d_model)
        d_sae = W_dec.shape[0]

        # Load CF feature list
        cf_indices = []
        if args.feature_classification is not None:
            with open(args.feature_classification) as f:
                feat_data = json.load(f)
            if isinstance(feat_data, list):
                cf_indices = [f["feature_idx"] for f in feat_data
                              if f.get("category") == "content_familiar"]
            else:
                cf_indices = feat_data.get("content_familiar", [])

        log.info(f"SAE: d_sae={d_sae}, n_CF={len(cf_indices)}")

        sl = str(args.sae_layer)
        S_K_basis = directions_data[f"S_K_basis_layer{args.sae_layer}"]
        S_R_basis = directions_data[f"S_R_basis_layer{args.sae_layer}"]
        d_K = directions_data[f"d_K_layer{args.sae_layer}"]
        d_R = directions_data[f"d_R_layer{args.sae_layer}"]

        # FSC using all SAE features
        fsc_K_all = compute_fsc(S_K_basis, W_dec)
        fsc_R_all = compute_fsc(S_R_basis, W_dec)

        # FSC using only CF features
        if len(cf_indices) > 0:
            W_cf = W_dec[cf_indices]
            fsc_K_cf = compute_fsc(S_K_basis, W_cf)
            fsc_R_cf = compute_fsc(S_R_basis, W_cf)
        else:
            fsc_K_cf = 0.0
            fsc_R_cf = 0.0

        # Per-feature dot products with d_K and d_R
        d_K_unit = d_K / (np.linalg.norm(d_K) + 1e-12)
        d_R_unit = d_R / (np.linalg.norm(d_R) + 1e-12)

        # Normalize decoder vectors
        W_norms = np.linalg.norm(W_dec, axis=1, keepdims=True)
        W_unit = W_dec / (W_norms + 1e-12)

        proj_K = W_unit @ d_K_unit  # (d_sae,) cosine with knowledge direction
        proj_R = W_unit @ d_R_unit  # (d_sae,) cosine with recall direction

        # CF features vs all features: alignment comparison
        cf_proj_K = proj_K[cf_indices] if len(cf_indices) > 0 else np.array([])
        cf_proj_R = proj_R[cf_indices] if len(cf_indices) > 0 else np.array([])

        sae_alignment = {
            "sae_layer": args.sae_layer,
            "d_sae": int(d_sae),
            "n_cf_features": len(cf_indices),
            "fsc_K_all_features": float(fsc_K_all),
            "fsc_R_all_features": float(fsc_R_all),
            "fsc_K_cf_features": float(fsc_K_cf),
            "fsc_R_cf_features": float(fsc_R_cf),
            "mean_cos_all_dK": float(np.abs(proj_K).mean()),
            "mean_cos_all_dR": float(np.abs(proj_R).mean()),
            "mean_cos_cf_dK": float(np.abs(cf_proj_K).mean()) if len(cf_proj_K) > 0 else 0.0,
            "mean_cos_cf_dR": float(np.abs(cf_proj_R).mean()) if len(cf_proj_R) > 0 else 0.0,
            "top10_K_features": [int(i) for i in np.argsort(np.abs(proj_K))[-10:][::-1]],
            "top10_R_features": [int(i) for i in np.argsort(np.abs(proj_R))[-10:][::-1]],
            "top10_K_cosines": [float(proj_K[i]) for i in np.argsort(np.abs(proj_K))[-10:][::-1]],
            "top10_R_cosines": [float(proj_R[i]) for i in np.argsort(np.abs(proj_R))[-10:][::-1]],
        }

        log.info(f"FSC_K (all features): {fsc_K_all:.4f}")
        log.info(f"FSC_R (all features): {fsc_R_all:.4f}")
        log.info(f"FSC_K (CF only): {fsc_K_cf:.4f}")
        log.info(f"FSC_R (CF only): {fsc_R_cf:.4f}")
        log.info(f"Mean |cos(feat, d_K)|. all: {np.abs(proj_K).mean():.4f}, CF: {np.abs(cf_proj_K).mean():.4f}" if len(cf_proj_K) > 0 else "")
        log.info(f"Mean |cos(feat, d_R)|. all: {np.abs(proj_R).mean():.4f}, CF: {np.abs(cf_proj_R).mean():.4f}" if len(cf_proj_R) > 0 else "")

        # Save per-feature projections
        np.savez_compressed(
            out_dir / "feature_projections.npz",
            proj_K=proj_K.astype(np.float32),
            proj_R=proj_R.astype(np.float32),
            cf_indices=np.array(cf_indices, dtype=np.int32),
        )

        del sae
        torch.cuda.empty_cache()

    # ── Save all results ────────────────────────────────────────────
    # Directions
    np.savez_compressed(out_dir / "directions.npz", **directions_data)
    log.info(f"Saved directions to {out_dir / 'directions.npz'}")

    # Per-layer orthogonality
    ortho_results = {
        "layers_analyzed": layers,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_components": args.n_components,
        "n_member": len(member_texts),
        "n_nonmember": len(nonmember_texts),
        "per_layer": results_per_layer,
    }
    (out_dir / "orthogonality.json").write_text(
        json.dumps(ortho_results, indent=2, default=str)
    )

    # SAE alignment
    if sae_alignment is not None:
        (out_dir / "sae_alignment.json").write_text(
            json.dumps(sae_alignment, indent=2, default=str)
        )

    # Summary (headline numbers)
    # The headline block binds to args.sae_layer, the pre-specified per-model
    # analysis layer documented in the paper's tab:model_details (selected once,
    # before any SAE training, by maximising cross-validated AUROC of a logistic
    # membership probe on raw mean-pooled residual-stream activations under a
    # fixed five-fold split; see methods.tex paragraph "For each model, we first
    # choose the SAE analysis layer."). The best_membership_layer/_auroc fields
    # below are diagnostic only -- they report which layer in this run's --layers
    # sweep peaked, as a sanity check, and are not the values cited by the paper.
    best_layer = max(results_per_layer.values(),
                     key=lambda x: x["membership_probe"]["auroc_mean"])
    sae_layer_key = str(args.sae_layer) if args.sae_layer is not None else None
    sae_layer_result = results_per_layer.get(sae_layer_key, {})

    summary = {
        "model_path": args.model_path,
        "n_layers": n_layers,
        "d_model": d_model,
        "layers_analyzed": layers,
        "best_membership_layer": best_layer["layer"],
        "best_membership_auroc": best_layer["membership_probe"]["auroc_mean"],
        "headline": {
            "sae_layer": args.sae_layer,
            "cosine_d_K_d_R": sae_layer_result.get("cosine_d_K_d_R"),
            "mean_principal_angle": sae_layer_result.get("mean_principal_angle_deg"),
            "min_principal_angle": sae_layer_result.get("min_principal_angle_deg"),
            "membership_auroc": sae_layer_result.get("membership_probe", {}).get("auroc_mean"),
            "recall_auroc": sae_layer_result.get("recall_probe", {}).get("auroc_mean"),
        },
    }
    if sae_alignment is not None:
        summary["headline"]["fsc_K_all"] = sae_alignment["fsc_K_all_features"]
        summary["headline"]["fsc_R_all"] = sae_alignment["fsc_R_all_features"]
        summary["headline"]["fsc_K_cf"] = sae_alignment["fsc_K_cf_features"]
        summary["headline"]["fsc_R_cf"] = sae_alignment["fsc_R_cf_features"]

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # ── Print summary ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Behavioural channel decomposition summary")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Layers analysed: {layers}")
    print(f"Data: {len(member_texts)} member, {len(nonmember_texts)} nonmember texts")
    print()

    print(f"{'Layer':>5} {'cos(dK,dR)':>10} {'MinAngle':>10} {'MeanAngle':>10} "
          f"{'MemAUROC':>10} {'RecAUROC':>10}")
    print("-" * 60)
    for l in layers:
        r = results_per_layer[str(l)]
        marker = " *" if l == args.sae_layer else ""
        print(f"{l:>5} {r['cosine_d_K_d_R']:>10.4f} {r['min_principal_angle_deg']:>9.1f}° "
              f"{r['mean_principal_angle_deg']:>9.1f}° "
              f"{r['membership_probe']['auroc_mean']:>10.3f} "
              f"{r['recall_probe']['auroc_mean']:>10.3f}{marker}")

    if sae_alignment is not None:
        print()
        print("SAE Alignment (Feature Sufficiency Criterion):")
        print(f"  FSC_K (all features): {sae_alignment['fsc_K_all_features']:.4f}")
        print(f"  FSC_R (all features): {sae_alignment['fsc_R_all_features']:.4f}")
        print(f"  FSC_K (CF features):  {sae_alignment['fsc_K_cf_features']:.4f}")
        print(f"  FSC_R (CF features):  {sae_alignment['fsc_R_cf_features']:.4f}")

    print()
    print("KEY:")
    print("  cos(dK,dR) near 0 → knowledge and recall directions are orthogonal")
    print("  Angles near 90° → subspaces are separable (double dissociation possible)")
    print("  FSC_R << FSC_K → SAE features align with knowledge, not recall (sparsity-alignment bias)")
    print("  * = SAE layer")
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
