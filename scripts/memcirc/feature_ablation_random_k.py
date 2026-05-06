#!/usr/bin/env python3
"""feature_ablation_random_k.py.

Random-k feature ablation null distribution. Calibrates the targeted top-k
ablation against random matched-size feature subsets.

Used in Appendix (A:709, "top-k feature ablation protocol") of the paper.
Reproduce: env/bin/python3 scripts/memcirc/feature_ablation_random_k.py --model-path <ft_model> --bcd-dir <bcd_dir> --sae-path <sae> --member-texts <member.jsonl> --nonmember-texts <nonmember.jsonl> --layer <L> --output-dir <out> --model-id <id> --seeds 0 1 2 3 4 --k-values 1 5 10 20 50 100

Companion to `feature_ablation_dark_subspace.py`. Instead of ranking SAE
features by their point-biserial correlation with membership and ablating
the top-k, this script ablates k randomly selected ever-active SAE
features (per seed). The resulting (recon AUROC, residual AUROC) curves
serve as a null distribution against which the true (correlation-ranked)
ablation can be compared.

If ablating k random features produces a drop comparable to ablating the
top-k correlated features, then the "no single feature concentrates
membership" finding is trivial (any feature removal degrades recon). If
the true-ranked ablation drops more than random, the ranking is picking up
membership-relevant structure.

The script reuses the activation-collection and SAE encode/decode logic
from `feature_ablation_dark_subspace.py`. Only the feature-selection rule
changes (random sample vs correlation-ranked top-k).
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

from sae_mia_audit.models.wrapper import load_model_and_tokenizer
from sae_mia_audit.utils.hf import HFModelSpec
from sae_mia_audit.utils.seed import SeedConfig, set_global_seed
from sae_mia_audit.utils.logging import setup_logging, get_logger
from sae_mia_audit.sae.io import load_sae_checkpoint_any

from sklearn.metrics import roc_auc_score  # noqa: F401 (used transitively)

# Reuse helpers from the canonical feature-ablation script. `_bootstrap`
# has already added scripts/memcirc/ to sys.path, so this is a flat import.
from feature_ablation_dark_subspace import (  # type: ignore
    _load_texts,
    _sanitize_for_json,
    bidirectional_auroc,
    collect_activations,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core random-k ablation
# ---------------------------------------------------------------------------

def run_random_k_ablation(
    activations: np.ndarray,
    labels: np.ndarray,
    d_K: np.ndarray,
    global_mean: Optional[np.ndarray],
    sae,
    device: str,
    k_values: List[int],
    seeds: List[int],
) -> Dict:
    """For each seed and each k, ablate k random ever-active SAE features.

    Returns per-(seed, k) recon/residual AUROCs plus aggregate mean/std
    across seeds.
    """
    d_K_norm = d_K / (np.linalg.norm(d_K) + 1e-12)

    if global_mean is not None:
        centered = activations - global_mean[np.newaxis, :]
    else:
        centered = activations - activations.mean(axis=0, keepdims=True)

    scores_original = centered @ d_K_norm
    auroc_original = bidirectional_auroc(labels, scores_original)
    log.info(f"score_K on ORIGINAL activations: AUROC = {auroc_original:.4f}")

    # --- Encode / standard decode ---
    h_tensor = torch.tensor(activations, dtype=torch.float32, device=device)
    batch_sz = 256
    all_z = []
    all_recon = []
    with torch.no_grad():
        for i in range(0, len(h_tensor), batch_sz):
            batch = h_tensor[i : i + batch_sz]
            z = sae.encode(batch)
            h_hat = sae.decode(z)
            all_z.append(z.detach().cpu().float().numpy())
            all_recon.append(h_hat.detach().cpu().float().numpy())
    latent = np.concatenate(all_z, axis=0)
    standard_recon = np.concatenate(all_recon, axis=0)

    if global_mean is not None:
        recon_centered = standard_recon - global_mean[np.newaxis, :]
    else:
        recon_centered = standard_recon - standard_recon.mean(axis=0, keepdims=True)
    scores_standard_recon = recon_centered @ d_K_norm
    auroc_standard_recon = bidirectional_auroc(labels, scores_standard_recon)
    log.info(
        f"score_K on STANDARD SAE recon (k=0): AUROC = {auroc_standard_recon:.4f}"
    )

    # Ever-active feature pool
    ever_active_mask = np.sum(latent > 0, axis=0) >= 2
    ever_active_indices = np.where(ever_active_mask)[0]
    n_ever_active = len(ever_active_indices)
    mean_active = float(np.mean(np.sum(latent > 0, axis=1)))
    log.info(
        f"SAE active pool: {n_ever_active}/{latent.shape[1]} ever-active features; "
        f"mean_active={mean_active:.1f}"
    )

    # --- Sweep ---
    per_seed_results: Dict[int, List[Dict]] = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        log.info(f"\n=== Seed {seed} ===")
        seed_entries = []
        for k in k_values:
            k_eff = min(k, n_ever_active)
            features_to_ablate = rng.choice(
                ever_active_indices, size=k_eff, replace=False
            ).tolist()

            z_ablated = latent.copy()
            z_ablated[:, features_to_ablate] = 0.0

            z_tensor = torch.tensor(z_ablated, dtype=torch.float32, device=device)
            all_h_ablated = []
            with torch.no_grad():
                for i in range(0, len(z_tensor), batch_sz):
                    batch = z_tensor[i : i + batch_sz]
                    h_hat = sae.decode(batch)
                    all_h_ablated.append(h_hat.detach().cpu().float().numpy())
            h_ablated = np.concatenate(all_h_ablated, axis=0)

            if global_mean is not None:
                ablated_centered = h_ablated - global_mean[np.newaxis, :]
            else:
                ablated_centered = h_ablated - h_ablated.mean(axis=0, keepdims=True)

            scores_ablated = ablated_centered @ d_K_norm
            auroc_ablated = bidirectional_auroc(labels, scores_ablated)

            residual_centered = centered - ablated_centered
            scores_residual = residual_centered @ d_K_norm
            auroc_residual = bidirectional_auroc(labels, scores_residual)

            log.info(
                f"  seed={seed} k={k_eff}: "
                f"recon AUROC={auroc_ablated:.4f}, "
                f"residual AUROC={auroc_residual:.4f}"
            )
            seed_entries.append({
                "k": int(k_eff),
                "ablated_features": [int(f) for f in features_to_ablate],
                "score_K_ablated_recon": float(auroc_ablated),
                "score_K_residual": float(auroc_residual),
            })
        per_seed_results[seed] = seed_entries

    # --- Aggregate per k across seeds ---
    aggregate: List[Dict] = []
    for idx, k in enumerate(k_values):
        recon_vals = np.array([
            per_seed_results[s][idx]["score_K_ablated_recon"] for s in seeds
        ])
        resid_vals = np.array([
            per_seed_results[s][idx]["score_K_residual"] for s in seeds
        ])
        aggregate.append({
            "k": int(min(k, n_ever_active)),
            "n_seeds": len(seeds),
            "mean_score_K_ablated_recon": float(recon_vals.mean()),
            "std_score_K_ablated_recon": float(recon_vals.std(ddof=1))
            if len(seeds) > 1 else 0.0,
            "mean_score_K_residual": float(resid_vals.mean()),
            "std_score_K_residual": float(resid_vals.std(ddof=1))
            if len(seeds) > 1 else 0.0,
            "per_seed_recon": recon_vals.tolist(),
            "per_seed_residual": resid_vals.tolist(),
        })

    return {
        "original_score_K": float(auroc_original),
        "standard_recon_score_K": float(auroc_standard_recon),
        "seeds": list(seeds),
        "k_values": [int(min(k, n_ever_active)) for k in k_values],
        "per_seed_results": {str(s): per_seed_results[s] for s in seeds},
        "aggregate": aggregate,
        "sae_stats": {
            "mean_active_features": float(mean_active),
            "total_features": int(latent.shape[1]),
            "n_ever_active": int(n_ever_active),
            "sparsity": float(mean_active / latent.shape[1]),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Random-k Feature Ablation Null Baseline"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--bcd-dir", required=True)
    parser.add_argument("--sae-path", required=True)
    parser.add_argument("--member-texts", required=True)
    parser.add_argument("--nonmember-texts", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-texts", type=int, default=0)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
        help="Random seeds for feature subset selection",
    )
    parser.add_argument(
        "--k-values", nargs="+", type=int,
        default=[1, 5, 10, 20, 50, 100],
        help="k feature counts to sweep (no -1 sentinel here)",
    )
    args = parser.parse_args()

    setup_logging(logging.INFO)
    # Activation-collection seed fixed to match the canonical experiment.
    set_global_seed(SeedConfig(seed=42))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    config["script"] = "feature_ablation_random_k.py"
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

    bcd_data = np.load(Path(args.bcd_dir) / "directions.npz", allow_pickle=True)
    dk_key = f"d_K_layer{args.layer}"
    if dk_key not in bcd_data:
        dk_key = "d_K"
    d_K = bcd_data[dk_key]
    global_mean = bcd_data["global_mean"] if "global_mean" in bcd_data else None
    log.info(f"Loaded d_K ({dk_key}): shape={d_K.shape}")

    log.info(f"Loading SAE from {args.sae_path}")
    sae = load_sae_checkpoint_any(args.sae_path, device=args.device)
    log.info(f"SAE loaded: d_model={sae.d_model}, d_sae={sae.d_sae}")

    log.info(f"Loading model from {args.model_path}")
    spec = HFModelSpec(name_or_path=args.model_path, torch_dtype="bfloat16")
    wrapper = load_model_and_tokenizer(spec)
    model = wrapper.model.to(args.device).eval()
    tokenizer = wrapper.tokenizer

    max_n = args.max_texts if args.max_texts > 0 else None
    member_texts = _load_texts(args.member_texts, max_n)
    nonmember_texts = _load_texts(args.nonmember_texts, max_n)
    all_texts = member_texts + nonmember_texts
    labels = np.array([1] * len(member_texts) + [0] * len(nonmember_texts))
    log.info(
        f"Loaded {len(member_texts)} member, {len(nonmember_texts)} nonmember texts"
    )

    activations = collect_activations(
        model, tokenizer, all_texts, args.layer,
        args.seq_len, args.batch_size, args.device,
    )
    log.info(f"Activations: shape={activations.shape}")

    del model
    torch.cuda.empty_cache()

    results = run_random_k_ablation(
        activations, labels, d_K, global_mean, sae, args.device,
        args.k_values, args.seeds,
    )

    output = {
        "model": args.model_id,
        "experiment": "feature_ablation_random_k",
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

    # --- Summary ---
    print("\n" + "=" * 76)
    print("RANDOM-K FEATURE ABLATION NULL BASELINE")
    print("=" * 76)
    print(f"Model: {args.model_id}  |  Layer: {args.layer}")
    print(f"Seeds: {args.seeds}  |  k values: {args.k_values}")
    print()
    print(f"  Original score_K AUROC:          {results['original_score_K']:.4f}")
    print(f"  Standard SAE recon AUROC (k=0):  {results['standard_recon_score_K']:.4f}")
    print()
    print(f"  {'k':>6}  {'mean recon':>12}  {'std recon':>10}  "
          f"{'mean resid':>12}  {'std resid':>10}")
    print("  " + "-" * 58)
    for entry in results["aggregate"]:
        print(
            f"  {entry['k']:>6}  "
            f"{entry['mean_score_K_ablated_recon']:>12.4f}  "
            f"{entry['std_score_K_ablated_recon']:>10.4f}  "
            f"{entry['mean_score_K_residual']:>12.4f}  "
            f"{entry['std_score_K_residual']:>10.4f}"
        )
    print()


if __name__ == "__main__":
    main()
