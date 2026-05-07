#!/usr/bin/env python3
"""feature_ablation_dark_subspace.py.

Top-k SAE classifier feature ablation in the encoded code, decoded back to
activations, measuring membership AUROC and verbatim extraction at each k.

Used in Section "Results" (R:84-87) and the feature-ablation control
appendix of the paper.
Reproduce: env/bin/python3 scripts/dark_subspace/feature_ablation_dark_subspace.py --model-path <ft_model> --bcd-dir <bcd_dir> --sae-path <sae> --member-texts <member.jsonl> --nonmember-texts <nonmember.jsonl> --layer <L> --output-dir <out> --model-id <id>

Protocol
--------
1. Collect mean-pooled activations h at layer L for all member/nonmember texts.
2. Encode through SAE -> sparse feature vector z per text.
3. For each k in {0, 1, 5, 10, 20, 50, 100, ALL_ACTIVE}:
   a. Identify top-k features most correlated with membership (point-biserial).
   b. Zero those features -> z_ablated.
   c. Decode z_ablated through SAE decoder -> h'_ablated.
   d. Compute score_K AUROC on h'_ablated (project onto d_K).
   e. Compute score_K AUROC on residual = h - h'_ablated.
4. Baseline. Standard SAE reconstruction (k=0, no ablation).
"""

import _bootstrap  # noqa: F401

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

try:
    from sae_mia_audit.models.wrapper import load_model_and_tokenizer
    from sae_mia_audit.utils.hf import HFModelSpec
    from sae_mia_audit.utils.seed import SeedConfig, set_global_seed
    from sae_mia_audit.utils.logging import setup_logging, get_logger
    from sae_mia_audit.sae.io import load_sae_checkpoint_any
    _HAS_PROJECT_INFRA = True
except ImportError as e:
    _HAS_PROJECT_INFRA = False
    _IMPORT_ERROR = str(e)

try:
    from sklearn.metrics import roc_auc_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    from scipy.stats import pointbiserialr
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

if _HAS_PROJECT_INFRA:
    log = get_logger(__name__)
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_texts(path: str, max_n: Optional[int] = None) -> List[str]:
    """Load texts from JSONL file (field: 'text')."""
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


def _sanitize_for_json(obj):
    """Replace non-finite floats with None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if not np.isfinite(obj):
            return None
        return obj
    return obj


def bidirectional_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC, taking the max of score and -score directions."""
    a = roc_auc_score(labels, scores)
    b = roc_auc_score(labels, -scores)
    return max(a, b)


# ---------------------------------------------------------------------------
# Activation collection (same pattern as sae_dark_subspace.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_activations(
    model, tokenizer, texts: List[str], layer: int,
    seq_len: int, batch_size: int, device: str,
) -> np.ndarray:
    """Collect mean-pooled hidden states at a specific layer."""
    all_acts = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Collecting activations"):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=seq_len, padding=True,
        ).to(device)
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[layer]  # (B, T, D)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, D)
        all_acts.append(pooled.cpu().float().numpy())
    return np.concatenate(all_acts, axis=0)


# ---------------------------------------------------------------------------
# Feature membership correlation
# ---------------------------------------------------------------------------

def compute_feature_membership_correlations(
    latent: np.ndarray, labels: np.ndarray,
) -> np.ndarray:
    """Compute point-biserial correlation between each SAE feature and membership.

    Parameters
    ----------
    latent : (N, d_sae) sparse feature activations
    labels : (N,) binary membership labels

    Returns
    -------
    correlations : (d_sae,) absolute point-biserial correlation per feature
    """
    n_features = latent.shape[1]
    correlations = np.zeros(n_features, dtype=np.float64)

    # Only compute for features that are ever active (non-zero for at least 2 texts)
    active_mask = np.sum(latent > 0, axis=0) >= 2
    n_active = int(active_mask.sum())
    log.info(f"Computing correlations for {n_active}/{n_features} ever-active features")

    if _HAS_SCIPY:
        for j in np.where(active_mask)[0]:
            r, _ = pointbiserialr(labels, latent[:, j])
            correlations[j] = abs(r) if np.isfinite(r) else 0.0
    else:
        # Manual fallback: Pearson on binary x continuous
        label_mean = labels.mean()
        label_std = labels.std()
        if label_std < 1e-12:
            log.warning("All labels identical; correlations are all zero")
            return correlations
        for j in np.where(active_mask)[0]:
            feat_col = latent[:, j]
            feat_std = feat_col.std()
            if feat_std < 1e-12:
                continue
            r = np.corrcoef(labels, feat_col)[0, 1]
            correlations[j] = abs(r) if np.isfinite(r) else 0.0

    return correlations


# ---------------------------------------------------------------------------
# Core feature ablation experiment
# ---------------------------------------------------------------------------

def run_feature_ablation(
    activations: np.ndarray,
    labels: np.ndarray,
    d_K: np.ndarray,
    global_mean: Optional[np.ndarray],
    sae,
    device: str,
    k_values: List[int],
) -> Dict:
    """Run the feature ablation experiment.

    For each k, ablate the top-k membership-correlated SAE features,
    decode back to activation space, and measure score_K.
    """
    n_texts = len(activations)
    d_K_norm = d_K / (np.linalg.norm(d_K) + 1e-12)

    # --- Center activations for score_K ---
    if global_mean is not None:
        centered = activations - global_mean[np.newaxis, :]
    else:
        centered = activations - activations.mean(axis=0, keepdims=True)

    # --- Original score_K ---
    scores_original = centered @ d_K_norm
    auroc_original = bidirectional_auroc(labels, scores_original)
    log.info(f"score_K on ORIGINAL activations: AUROC = {auroc_original:.4f}")

    # --- SAE encode all texts ---
    h_tensor = torch.tensor(activations, dtype=torch.float32, device=device)
    batch_sz = 256
    all_z = []
    all_recon = []
    for i in range(0, len(h_tensor), batch_sz):
        batch = h_tensor[i : i + batch_sz]
        z = sae.encode(batch)
        h_hat = sae.decode(z)
        all_z.append(z.detach().cpu().float().numpy())
        all_recon.append(h_hat.detach().cpu().float().numpy())

    latent = np.concatenate(all_z, axis=0)          # (N, d_sae)
    standard_recon = np.concatenate(all_recon, axis=0)  # (N, D)

    # --- Standard reconstruction score_K (k=0 baseline) ---
    if global_mean is not None:
        recon_centered = standard_recon - global_mean[np.newaxis, :]
    else:
        recon_centered = standard_recon - standard_recon.mean(axis=0, keepdims=True)

    scores_standard_recon = recon_centered @ d_K_norm
    auroc_standard_recon = bidirectional_auroc(labels, scores_standard_recon)
    log.info(f"score_K on STANDARD SAE recon (k=0): AUROC = {auroc_standard_recon:.4f}")

    # --- Feature sparsity stats ---
    mean_active = float(np.mean(np.sum(latent > 0, axis=1)))
    total_features = latent.shape[1]
    log.info(f"SAE features: mean_active={mean_active:.1f}/{total_features}")

    # --- Compute feature-membership correlations ---
    log.info("Computing feature-membership correlations (point-biserial)...")
    correlations = compute_feature_membership_correlations(latent, labels)

    # Rank features by |correlation| descending
    ranked_indices = np.argsort(-correlations)
    n_ever_active = int(np.sum(correlations > 0))
    log.info(
        f"Top correlations: "
        f"#1={correlations[ranked_indices[0]]:.4f} (feat {ranked_indices[0]}), "
        f"#2={correlations[ranked_indices[1]]:.4f} (feat {ranked_indices[1]}), "
        f"#3={correlations[ranked_indices[2]]:.4f} (feat {ranked_indices[2]})"
    )

    # --- Resolve k_values: replace sentinel -1 with ALL_ACTIVE ---
    resolved_k_values = []
    for k in k_values:
        if k == -1:
            resolved_k_values.append(n_ever_active)
        else:
            resolved_k_values.append(min(k, n_ever_active))
    log.info(f"k values (resolved): {resolved_k_values}")

    # --- Run ablation for each k ---
    ablation_results = []
    per_text_k100_scores = None

    for k in resolved_k_values:
        log.info(f"\n--- Ablating top-{k} features ---")

        if k == 0:
            # This is the standard reconstruction (no ablation)
            ablation_results.append({
                "k": 0,
                "ablated_features": [],
                "score_K_ablated_recon": float(auroc_standard_recon),
                "score_K_residual": float(bidirectional_auroc(
                    labels,
                    (centered - recon_centered) @ d_K_norm
                )),
                "feature_correlations": [],
            })
            continue

        # Features to ablate: top-k by correlation
        features_to_ablate = ranked_indices[:k].tolist()
        feat_corrs = correlations[features_to_ablate].tolist()

        # Zero those features in z
        z_ablated = latent.copy()
        z_ablated[:, features_to_ablate] = 0.0

        # Decode z_ablated through SAE
        z_tensor = torch.tensor(z_ablated, dtype=torch.float32, device=device)
        all_h_ablated = []
        for i in range(0, len(z_tensor), batch_sz):
            batch = z_tensor[i : i + batch_sz]
            h_hat = sae.decode(batch)
            all_h_ablated.append(h_hat.detach().cpu().float().numpy())
        h_ablated = np.concatenate(all_h_ablated, axis=0)

        # score_K on ablated reconstruction
        if global_mean is not None:
            ablated_centered = h_ablated - global_mean[np.newaxis, :]
        else:
            ablated_centered = h_ablated - h_ablated.mean(axis=0, keepdims=True)

        scores_ablated = ablated_centered @ d_K_norm
        auroc_ablated = bidirectional_auroc(labels, scores_ablated)

        # score_K on residual = h - h'_ablated
        residual = activations - h_ablated
        if global_mean is not None:
            residual_centered = centered - ablated_centered
        else:
            residual_centered = residual - residual.mean(axis=0, keepdims=True)

        scores_residual = residual_centered @ d_K_norm
        auroc_residual = bidirectional_auroc(labels, scores_residual)

        log.info(
            f"  k={k}: ablated_recon AUROC={auroc_ablated:.4f}, "
            f"residual AUROC={auroc_residual:.4f}"
        )

        ablation_results.append({
            "k": int(k),
            "ablated_features": [int(f) for f in features_to_ablate],
            "score_K_ablated_recon": float(auroc_ablated),
            "score_K_residual": float(auroc_residual),
            "feature_correlations": [float(c) for c in feat_corrs],
        })

        # Save per-text scores for k=100 (for bootstrap CIs)
        if k == 100:
            per_text_k100_scores = scores_ablated.tolist()

    # --- Also save per-text scores for key conditions ---
    per_text_scores = {
        "labels": labels.tolist(),
        "score_K_original": scores_original.tolist(),
        "score_K_standard_recon": scores_standard_recon.tolist(),
    }
    if per_text_k100_scores is not None:
        per_text_scores["score_K_ablated_k100"] = per_text_k100_scores

    return {
        "original_score_K": float(auroc_original),
        "standard_recon_score_K": float(auroc_standard_recon),
        "ablation_results": ablation_results,
        "per_text_scores": per_text_scores,
        "sae_stats": {
            "mean_active_features": float(mean_active),
            "total_features": int(total_features),
            "n_ever_active": int(n_ever_active),
            "sparsity": float(mean_active / total_features),
        },
        "top_20_correlated_features": [
            {
                "feature_index": int(ranked_indices[i]),
                "correlation": float(correlations[ranked_indices[i]]),
            }
            for i in range(min(20, len(ranked_indices)))
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Feature Ablation Dark Subspace: Does removing "
        "membership-correlated SAE features eliminate score_K?"
    )
    parser.add_argument("--model-path", required=True, help="Path to fine-tuned model")
    parser.add_argument("--bcd-dir", required=True, help="Path to channel-decomposition directions directory")
    parser.add_argument("--sae-path", required=True, help="Path to SAE checkpoint (sae_final.pt)")
    parser.add_argument("--member-texts", required=True)
    parser.add_argument("--nonmember-texts", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-texts", type=int, default=0,
                        help="Max texts per class (0 = all)")
    parser.add_argument("--k-values", nargs="+", type=int,
                        default=[1, 5, 10, 20, 50, 100, -1],
                        help="Number of features to ablate. -1 = all active.")
    args = parser.parse_args()

    if not _HAS_PROJECT_INFRA:
        raise RuntimeError(f"Project infrastructure required: {_IMPORT_ERROR}")
    if not _HAS_SKLEARN:
        raise RuntimeError("sklearn required: pip install scikit-learn")

    setup_logging(logging.INFO)
    set_global_seed(SeedConfig(seed=args.seed))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    config["script"] = "feature_ablation_dark_subspace.py"
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

    # --- Load channel-decomposition directions ---
    bcd_data = np.load(Path(args.bcd_dir) / "directions.npz", allow_pickle=True)
    dk_key = f"d_K_layer{args.layer}"
    if dk_key not in bcd_data:
        dk_key = "d_K"
    d_K = bcd_data[dk_key]
    global_mean = bcd_data["global_mean"] if "global_mean" in bcd_data else None
    log.info(f"Loaded d_K ({dk_key}): shape={d_K.shape}")

    # --- Load SAE ---
    log.info(f"Loading SAE from {args.sae_path}")
    sae = load_sae_checkpoint_any(args.sae_path, device=args.device)
    log.info(f"SAE loaded: d_model={sae.d_model}, d_sae={sae.d_sae}")

    # --- Load model ---
    log.info(f"Loading model from {args.model_path}")
    spec = HFModelSpec(name_or_path=args.model_path, torch_dtype="bfloat16")
    wrapper = load_model_and_tokenizer(spec)
    model = wrapper.model.to(args.device).eval()
    tokenizer = wrapper.tokenizer

    # --- Load texts ---
    max_n = args.max_texts if args.max_texts > 0 else None
    member_texts = _load_texts(args.member_texts, max_n)
    nonmember_texts = _load_texts(args.nonmember_texts, max_n)
    log.info(f"Loaded {len(member_texts)} member, {len(nonmember_texts)} nonmember texts")

    all_texts = member_texts + nonmember_texts
    labels = np.array([1] * len(member_texts) + [0] * len(nonmember_texts))

    # --- Collect activations ---
    activations = collect_activations(
        model, tokenizer, all_texts, args.layer,
        args.seq_len, args.batch_size, args.device,
    )
    log.info(f"Activations: shape={activations.shape}")

    # Free model GPU memory (SAE is small)
    del model
    torch.cuda.empty_cache()

    # --- Run ablation experiment ---
    results = run_feature_ablation(
        activations, labels, d_K, global_mean, sae, args.device,
        args.k_values,
    )

    # --- Save results ---
    output = {
        "model": args.model_id,
        "experiment": "feature_ablation",
        "sae_type": "mixed_data",
        "sae_path": args.sae_path,
        "layer": args.layer,
        "n_member": len(member_texts),
        "n_nonmember": len(nonmember_texts),
        **results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(_sanitize_for_json(output), f, indent=2)
    log.info(f"Results saved to {results_path}")

    # --- Print summary ---
    print("\n" + "=" * 76)
    print("Feature ablation dark subspace experiment")
    print("=" * 76)
    print(f"Model: {args.model_id}  |  Layer: {args.layer}  |  SAE: {args.sae_path}")
    print()
    print(f"  Original score_K AUROC:          {results['original_score_K']:.4f}")
    print(f"  Standard SAE recon AUROC (k=0):  {results['standard_recon_score_K']:.4f}")
    print()
    print(f"  {'k':>6}  {'Ablated Recon AUROC':>20}  {'Residual AUROC':>16}  {'Top corr':>10}")
    print("  " + "-" * 58)
    for entry in results["ablation_results"]:
        top_corr = entry["feature_correlations"][0] if entry["feature_correlations"] else 0.0
        print(
            f"  {entry['k']:>6}  {entry['score_K_ablated_recon']:>20.4f}  "
            f"{entry['score_K_residual']:>16.4f}  {top_corr:>10.4f}"
        )

    print()
    print(f"  SAE: {results['sae_stats']['mean_active_features']:.0f} "
          f"mean active / {results['sae_stats']['total_features']} total "
          f"({results['sae_stats']['n_ever_active']} ever-active)")
    print()

    # Interpret key finding
    abl_100 = [r for r in results["ablation_results"] if r["k"] == 100]
    abl_all = results["ablation_results"][-1]  # last entry is ALL_ACTIVE

    if abl_100:
        k100 = abl_100[0]
        drop_100 = results["original_score_K"] - k100["score_K_ablated_recon"]
        print(f"  Key result (k=100): ablating 100 most correlated features")
        print(f"    score_K drop: {drop_100:+.4f}")
        if k100["score_K_ablated_recon"] > 0.65:
            print("    --> score_K PERSISTS after ablation. Dark subspace confirmed.")
        elif k100["score_K_ablated_recon"] < 0.55:
            print("    --> score_K ELIMINATED by ablation. Signal IS in SAE features.")
        else:
            print("    --> Partial effect. Ambiguous.")

    drop_all = results["original_score_K"] - abl_all["score_K_ablated_recon"]
    print(f"\n  ALL active features ablated (k={abl_all['k']}): "
          f"recon AUROC={abl_all['score_K_ablated_recon']:.4f} "
          f"(drop={drop_all:+.4f})")
    if abl_all["score_K_ablated_recon"] > 0.60:
        print("    --> Even with ALL correlated features removed, score_K persists!")
        print("    --> SAE features fundamentally cannot capture the memorisation signal.")
    print()


if __name__ == "__main__":
    main()
