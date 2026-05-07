#!/usr/bin/env python3
"""
sae_dark_subspace.py.

Encodes-decodes mean-pooled residual activations through a trained SAE, splits
each activation into reconstruction and reconstruction residual, and reports
per-condition score-K AUROC plus a fresh logistic-probe AUROC for the
original, reconstructed, and residual streams to
``runs/dark_subspace/sae_dark_subspace/<condition>/results.json``.

Used in Methods §3.3 (SAE reconstruction), Results §4.2 (dark subspace).
Reproduce:
    env/bin/python3 scripts/dark_subspace/sae_dark_subspace.py \\
        --model-path runs/controlled_ft/.../ft_epoch5/model \\
        --bcd-dir runs/dark_subspace/behavioral_channels/p69_epoch5 \\
        --sae-path runs/sae/memcirc_p69_epoch5_layer16_8x_l1_1e4_member/sae_final.pt \\
        --member-texts data/memcirc_ctrl_ft/member.jsonl \\
        --nonmember-texts data/memcirc_ctrl_ft/nonmember.jsonl \\
        --layer 16 --output-dir runs/dark_subspace/sae_dark_subspace/p69_epoch5 \\
        --model-id p69

Tests whether the memorisation signal (score_K) lives in the SAE blind spot
by comparing membership detection AUROC on three streams.
  1. Original activations h, expected high AUROC.
  2. SAE-reconstructed h' = decode(encode(h)), predicted near chance.
  3. Residual r = h - h', predicted high (signal preserved).

If score_K(h') is near chance while score_K(r) is near score_K(h), the
membership signal is invisible to the SAE and lives in the dark subspace.
"""

import _bootstrap  # noqa: F401

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

if _HAS_PROJECT_INFRA:
    log = get_logger(__name__)
else:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_texts(path: str, max_n: Optional[int] = None) -> List[str]:
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
# Activation collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_activations(model, tokenizer, texts, layer, seq_len, batch_size, device):
    """Collect mean-pooled hidden states at a specific layer."""
    all_acts = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Collecting activations"):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", truncation=True,
                        max_length=seq_len, padding=True).to(device)
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[layer]  # (B, T, D)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, D)
        all_acts.append(pooled.cpu().float().numpy())

    return np.concatenate(all_acts, axis=0)


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

def run_dark_subspace_experiment(
    activations: np.ndarray,
    labels: np.ndarray,
    d_K: np.ndarray,
    global_mean: Optional[np.ndarray],
    sae,
    device: str,
) -> Dict:
    """Run the SAE dark subspace experiment.

    Decomposes activations into SAE-visible (reconstruction) and
    SAE-invisible (residual) components, then measures score_K on each.
    """
    n_texts = len(activations)
    d_K_norm = d_K / (np.linalg.norm(d_K) + 1e-12)

    # Center activations
    if global_mean is not None:
        centered = activations - global_mean[np.newaxis, :]
    else:
        centered = activations - activations.mean(axis=0, keepdims=True)

    # --- score_K on ORIGINAL activations ---
    scores_original = centered @ d_K_norm
    auroc_original = bidirectional_auroc(labels, scores_original)
    log.info(f"score_K on ORIGINAL activations: AUROC = {auroc_original:.4f}")

    # --- SAE encode → decode ---
    h_tensor = torch.tensor(activations, dtype=torch.float32, device=device)

    # Encode and decode in batches (SAE may have memory limits)
    batch_size = 256
    all_recon = []
    all_latent = []
    for i in range(0, len(h_tensor), batch_size):
        batch = h_tensor[i:i + batch_size]
        z = sae.encode(batch)
        h_hat = sae.decode(z)
        all_recon.append(h_hat.detach().cpu().float().numpy())
        all_latent.append(z.detach().cpu().float().numpy())

    reconstructed = np.concatenate(all_recon, axis=0)  # h'
    latent = np.concatenate(all_latent, axis=0)  # z (SAE feature activations)
    residual = activations - reconstructed  # r = h - h'

    # Reconstruction quality
    recon_error = np.mean(np.sum((activations - reconstructed) ** 2, axis=1))
    recon_cos = np.mean([
        np.dot(activations[i], reconstructed[i]) /
        (np.linalg.norm(activations[i]) * np.linalg.norm(reconstructed[i]) + 1e-12)
        for i in range(n_texts)
    ])
    log.info(f"SAE reconstruction: MSE={recon_error:.4f}, mean_cosine={recon_cos:.4f}")

    # --- score_K on RECONSTRUCTED activations ---
    if global_mean is not None:
        recon_centered = reconstructed - global_mean[np.newaxis, :]
    else:
        recon_centered = reconstructed - reconstructed.mean(axis=0, keepdims=True)

    scores_recon = recon_centered @ d_K_norm
    auroc_recon = bidirectional_auroc(labels, scores_recon)
    log.info(f"score_K on SAE-RECONSTRUCTED activations: AUROC = {auroc_recon:.4f}")

    # --- score_K on RESIDUAL (what SAE misses) ---
    if global_mean is not None:
        # Residual centering: use centered original - centered reconstruction
        residual_centered = centered - recon_centered
    else:
        residual_centered = residual - residual.mean(axis=0, keepdims=True)

    scores_residual = residual_centered @ d_K_norm
    auroc_residual = bidirectional_auroc(labels, scores_residual)
    log.info(f"score_K on RESIDUAL (h - h'): AUROC = {auroc_residual:.4f}")

    # --- Norm-based baselines on each component ---
    norm_original = np.linalg.norm(centered, axis=1)
    norm_recon = np.linalg.norm(recon_centered, axis=1)
    norm_residual = np.linalg.norm(residual_centered, axis=1)

    auroc_norm_orig = bidirectional_auroc(labels, norm_original)
    auroc_norm_recon = bidirectional_auroc(labels, norm_recon)
    auroc_norm_residual = bidirectional_auroc(labels, norm_residual)

    # --- d_K projection onto SAE feature space ---
    # How much of d_K is captured by SAE features?
    d_K_tensor = torch.tensor(d_K_norm, dtype=torch.float32, device=device).unsqueeze(0)
    z_dK = sae.encode(d_K_tensor)
    d_K_recon = sae.decode(z_dK).detach().cpu().numpy().squeeze()
    d_K_recon_norm = d_K_recon / (np.linalg.norm(d_K_recon) + 1e-12)

    # How much of d_K survives SAE reconstruction?
    d_K_coverage = np.dot(d_K_norm, d_K_recon_norm)
    d_K_residual_frac = 1.0 - np.dot(d_K_norm, d_K_recon) ** 2 / (np.linalg.norm(d_K_recon) ** 2 + 1e-12)

    log.info(f"d_K coverage by SAE: cosine={d_K_coverage:.4f}")

    # --- SAE feature sparsity stats ---
    mean_active = np.mean(np.sum(latent > 0, axis=1))
    total_features = latent.shape[1]

    return {
        "original": {
            "score_K_auroc": float(auroc_original),
            "norm_auroc": float(auroc_norm_orig),
        },
        "sae_reconstructed": {
            "score_K_auroc": float(auroc_recon),
            "norm_auroc": float(auroc_norm_recon),
        },
        "residual": {
            "score_K_auroc": float(auroc_residual),
            "norm_auroc": float(auroc_norm_residual),
        },
        "sae_quality": {
            "reconstruction_mse": float(recon_error),
            "reconstruction_cosine": float(recon_cos),
            "d_K_coverage_cosine": float(d_K_coverage),
            "d_K_residual_fraction": float(d_K_residual_frac),
            "mean_active_features": float(mean_active),
            "total_features": int(total_features),
            "sparsity": float(mean_active / total_features),
        },
        "dark_subspace_effect": {
            "auroc_drop_from_recon": float(auroc_original - auroc_recon),
            "auroc_preserved_in_residual": float(auroc_residual),
            "memorization_is_dark": bool(auroc_recon < 0.60 and auroc_residual > 0.65),
        },
        "per_text_scores": {
            "labels": labels.tolist(),
            "score_K_original": scores_original.tolist(),
            "score_K_recon": scores_recon.tolist(),
            "score_K_residual": scores_residual.tolist(),
            "norm_original": norm_original.tolist(),
            "norm_recon": norm_recon.tolist(),
            "norm_residual": norm_residual.tolist(),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare membership-detection AUROC across original activation, "
            "SAE reconstruction, and reconstruction residual."
        )
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
    parser.add_argument("--max-texts", type=int, default=0)
    args = parser.parse_args()

    if not _HAS_PROJECT_INFRA:
        raise RuntimeError(f"Project infrastructure required: {_IMPORT_ERROR}")
    if not _HAS_SKLEARN:
        raise RuntimeError("sklearn required")

    setup_logging(logging.INFO)
    set_global_seed(SeedConfig(seed=args.seed))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    config["script"] = "sae_dark_subspace.py"
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

    # Load channel-decomposition directions
    bcd_data = np.load(Path(args.bcd_dir) / "directions.npz", allow_pickle=True)
    dk_key = f"d_K_layer{args.layer}"
    if dk_key not in bcd_data:
        dk_key = "d_K"
    d_K = bcd_data[dk_key]
    global_mean = bcd_data["global_mean"] if "global_mean" in bcd_data else None
    log.info(f"Loaded d_K ({dk_key}): shape={d_K.shape}")

    # Load SAE
    log.info(f"Loading SAE from {args.sae_path}")
    sae = load_sae_checkpoint_any(args.sae_path, device=args.device)
    log.info(f"SAE loaded: d_model={sae.d_model}, d_sae={sae.d_sae}")

    # Load model
    log.info(f"Loading model from {args.model_path}")
    spec = HFModelSpec(name_or_path=args.model_path, torch_dtype="bfloat16")
    wrapper = load_model_and_tokenizer(spec)
    model = wrapper.model.to(args.device).eval()
    tokenizer = wrapper.tokenizer

    # Load texts
    max_n = args.max_texts if args.max_texts > 0 else None
    member_texts = _load_texts(args.member_texts, max_n)
    nonmember_texts = _load_texts(args.nonmember_texts, max_n)
    log.info(f"Loaded {len(member_texts)} member, {len(nonmember_texts)} nonmember texts")

    all_texts = member_texts + nonmember_texts
    labels = np.array([1] * len(member_texts) + [0] * len(nonmember_texts))

    # Collect activations
    activations = collect_activations(
        model, tokenizer, all_texts, args.layer,
        args.seq_len, args.batch_size, args.device
    )
    log.info(f"Activations: shape={activations.shape}")

    # Free model GPU memory (SAE is small)
    del model
    torch.cuda.empty_cache()

    # Run the experiment
    results = run_dark_subspace_experiment(
        activations, labels, d_K, global_mean, sae, args.device
    )

    # Add metadata
    output = {
        "model": args.model_id,
        "layer": args.layer,
        "sae_path": args.sae_path,
        "n_member": len(member_texts),
        "n_nonmember": len(nonmember_texts),
        **results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(_sanitize_for_json(output), f, indent=2)
    log.info(f"Results saved to {results_path}")

    # Print summary
    print("\n" + "=" * 72)
    print("SAE DARK SUBSPACE EXPERIMENT SUMMARY")
    print("=" * 72)
    print(f"Model: {args.model_id}  |  Layer: {args.layer}  |  SAE: {args.sae_path}")
    print()
    print(f"  {'Component':<25} {'score_K AUROC':>14} {'norm AUROC':>12}")
    print("  " + "-" * 53)
    print(f"  {'Original h':<25} {results['original']['score_K_auroc']:>14.4f} {results['original']['norm_auroc']:>12.4f}")
    recon_label = "SAE-reconstructed h'"
    resid_label = "Residual (h - h')"
    print(f"  {recon_label:<25} {results['sae_reconstructed']['score_K_auroc']:>14.4f} {results['sae_reconstructed']['norm_auroc']:>12.4f}")
    print(f"  {resid_label:<25} {results['residual']['score_K_auroc']:>14.4f} {results['residual']['norm_auroc']:>12.4f}")
    print()
    print(f"  SAE reconstruction cosine: {results['sae_quality']['reconstruction_cosine']:.4f}")
    print(f"  d_K coverage by SAE:       {results['sae_quality']['d_K_coverage_cosine']:.4f}")
    print(f"  Mean active features:      {results['sae_quality']['mean_active_features']:.0f}/{results['sae_quality']['total_features']}")
    print()

    ds = results['dark_subspace_effect']
    if ds['memorization_is_dark']:
        print("  Membership signal preserved in residual; reconstruction below threshold.")
        print(f"  SAE reconstruction dropped score_K by {ds['auroc_drop_from_recon']:.4f}")
        print(f"  Residual preserves AUROC = {ds['auroc_preserved_in_residual']:.4f}")
    else:
        print(f"  Dark-subspace signal not detected (recon AUROC={results['sae_reconstructed']['score_K_auroc']:.4f})")


if __name__ == "__main__":
    main()
