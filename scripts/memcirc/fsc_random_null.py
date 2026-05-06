#!/usr/bin/env python3
"""fsc_random_null.py.

Computes the feature sufficiency criterion FSC(S_K, F) for selected
classifier features versus a random-subset null distribution and produces
the FSC random-null table.

Used in Section "Results" (R:77-88) and Appendix `app:fsc` (A:489-525) of the paper.
Reproduce: env/bin/python3 scripts/memcirc/fsc_random_null.py [--n-random 1000] [--json]

For each model with BCD results, samples N random subsets of size |CF|
from the full SAE dictionary, computes FSC for each, and reports:
  - mean, std, 95th/99th percentile of random FSC distribution
  - observed FSC and its percentile rank
  - FSC for random subsets of size 50 and 100

This addresses the reviewer concern that low FSC (around 8 to 16 percent)
is trivially expected when selecting few features from a 16K+ dictionary.
"""

import _bootstrap  # noqa: F401

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def compute_fsc_fast(subspace_basis: np.ndarray, feature_directions: np.ndarray) -> float:
    """Feature Sufficiency Criterion (fast version).

    FSC = ||P_F(S)||_F^2 / ||S||_F^2

    For small feature subsets, we compute P_F via the Gram matrix inverse
    instead of full QR decomposition. For large subsets, uses SVD-truncated approach.

    subspace_basis: (k, d) orthonormal
    feature_directions: (n_features, d)
    """
    n_feat = len(feature_directions)
    if n_feat == 0:
        return 0.0

    k = subspace_basis.shape[0]

    # Method: S @ F^T gives (k, n_feat), then compute projection via pseudoinverse
    # P_F = F^T (F F^T)^{-1} F when F is (n_feat, d)
    # ||P_F(s_i)||^2 = s_i^T F^T (F F^T)^{-1} F s_i
    # Let A = S @ F^T  (k, n_feat)
    # Let G = F @ F^T   (n_feat, n_feat)
    # FSC = tr(A @ G^{-1} @ A^T) / k

    F = feature_directions  # (n_feat, d)
    S = subspace_basis      # (k, d)

    A = S @ F.T  # (k, n_feat)
    G = F @ F.T  # (n_feat, n_feat)

    # Regularize for numerical stability
    G += 1e-8 * np.eye(n_feat)

    try:
        # Solve G^{-1} @ A^T via Cholesky
        L = np.linalg.cholesky(G)
        # G^{-1} A^T = L^{-T} L^{-1} A^T
        Y = np.linalg.solve(L, A.T)  # (n_feat, k)
        proj_sq = (Y ** 2).sum()
    except np.linalg.LinAlgError:
        # Fall back to pseudoinverse
        G_inv = np.linalg.pinv(G)
        proj_sq = np.trace(A @ G_inv @ A.T)

    return float(np.clip(proj_sq / k, 0.0, 1.0))


def precompute_projections(subspace_basis, W_dec):
    """Precompute S @ W_dec^T once for all random subsets.

    Returns: (k, d_sae) matrix of all pairwise dot products.
    """
    return subspace_basis @ W_dec.T  # (k, d_sae)


def compute_fsc_from_precomputed(SWT, idx, W_dec):
    """Fast FSC using precomputed S @ W^T and Gram matrix on subset.

    SWT: (k, d_sae) precomputed
    idx: subset indices
    W_dec: (d_sae, d_model) full decoder
    """
    k = SWT.shape[0]
    n = len(idx)

    A = SWT[:, idx]  # (k, n)

    # Gram matrix of selected features
    F_sub = W_dec[idx]  # (n, d_model)
    G = F_sub @ F_sub.T  # (n, n)
    G += 1e-8 * np.eye(n)

    try:
        L = np.linalg.cholesky(G)
        Y = np.linalg.solve(L, A.T)  # (n, k)
        proj_sq = (Y ** 2).sum()
    except np.linalg.LinAlgError:
        G_inv = np.linalg.pinv(G)
        proj_sq = np.trace(A @ G_inv @ A.T)

    return float(np.clip(proj_sq / k, 0.0, 1.0))


def load_decoder_weights(sae_path: str) -> np.ndarray:
    """Load SAE decoder weight matrix as numpy array.

    Returns: (d_sae, d_model) array where each row is a feature direction.
    """
    import torch

    ckpt = torch.load(sae_path, map_location="cpu", weights_only=False)

    # Repo-native checkpoint
    if "sae_cfg" in ckpt:
        cfg = ckpt["sae_cfg"]
        sd = ckpt["state_dict"]

        tied = cfg.get("tied_weights", False)
        if tied:
            enc_w = sd["encoder.weight"]  # (d_sae, d_model)
            W_dec = enc_w  # already (d_sae, d_model)
        else:
            # nn.Linear(d_sae, d_model) stores weight as (d_model, d_sae)
            W_dec = sd["decoder.weight"].T  # (d_sae, d_model)

        return W_dec.numpy().astype(np.float32)

    # SAIF checkpoint
    if "learned_dict" in ckpt:
        return ckpt["learned_dict"].numpy().astype(np.float32)

    raise ValueError(f"Unknown checkpoint format: {sae_path}")


def run_model(model_name, bcd_dir, sae_path, sae_layer, n_random, rng, extra_sizes=(50, 100)):
    """Run FSC null baseline for one model."""

    # Load S_K basis
    directions = np.load(os.path.join(bcd_dir, "directions.npz"))
    S_K_key = f"S_K_basis_layer{sae_layer}"
    if S_K_key not in directions:
        return None
    S_K_basis = directions[S_K_key]  # (k, d_model)

    # Load sae_alignment for observed FSC and CF count
    with open(os.path.join(bcd_dir, "sae_alignment.json")) as f:
        alignment = json.load(f)

    n_cf = alignment["n_cf_features"]
    d_sae = alignment["d_sae"]
    observed_fsc_K = alignment["fsc_K_cf_features"]

    # Load decoder weights
    print(f"  Loading SAE ({sae_path})...", file=sys.stderr)
    W_dec = load_decoder_weights(sae_path)  # (d_sae, d_model)
    assert W_dec.shape[0] == d_sae, f"d_sae mismatch: {W_dec.shape[0]} vs {d_sae}"
    print(f"  W_dec shape: {W_dec.shape}", file=sys.stderr)

    # Precompute S_K @ W_dec^T
    SWT = precompute_projections(S_K_basis, W_dec)  # (k, d_sae)
    print(f"  Precomputed SWT: {SWT.shape}", file=sys.stderr)

    results = {
        "model": model_name,
        "d_sae": d_sae,
        "n_cf": n_cf,
        "S_K_rank": int(S_K_basis.shape[0]),
        "d_model": int(S_K_basis.shape[1]),
        "observed_fsc_K_cf": observed_fsc_K,
    }

    # Random null at CF size
    print(f"  Computing {n_random} random FSC at size {n_cf}...", file=sys.stderr)
    random_fsc_values = []
    for i in range(n_random):
        idx = rng.choice(d_sae, size=n_cf, replace=False)
        fsc = compute_fsc_from_precomputed(SWT, idx, W_dec)
        random_fsc_values.append(fsc)

    random_fsc = np.array(random_fsc_values)
    percentile_rank = float(np.mean(random_fsc <= observed_fsc_K) * 100)

    results["null_at_cf_size"] = {
        "subset_size": n_cf,
        "n_random": n_random,
        "mean": float(random_fsc.mean()),
        "std": float(random_fsc.std()),
        "p50": float(np.percentile(random_fsc, 50)),
        "p95": float(np.percentile(random_fsc, 95)),
        "p99": float(np.percentile(random_fsc, 99)),
        "min": float(random_fsc.min()),
        "max": float(random_fsc.max()),
        "observed": observed_fsc_K,
        "observed_percentile_rank": percentile_rank,
    }

    # Random null at extra sizes (50, 100)
    results["null_at_extra_sizes"] = {}
    for sz in extra_sizes:
        if sz > d_sae:
            continue
        print(f"  Computing {n_random} random FSC at size {sz}...", file=sys.stderr)
        extra_fsc = []
        for _ in range(n_random):
            idx = rng.choice(d_sae, size=sz, replace=False)
            fsc = compute_fsc_from_precomputed(SWT, idx, W_dec)
            extra_fsc.append(fsc)
        extra_fsc = np.array(extra_fsc)
        results["null_at_extra_sizes"][str(sz)] = {
            "subset_size": sz,
            "n_random": n_random,
            "mean": float(extra_fsc.mean()),
            "std": float(extra_fsc.std()),
            "p95": float(np.percentile(extra_fsc, 95)),
            "p99": float(np.percentile(extra_fsc, 99)),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="FSC Random-Feature Null Baseline")
    parser.add_argument("--n-random", type=int, default=1000, help="Number of random subsets")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Model configs: (name, bcd_dir, sae_layer)
    bcd_base = ROOT / "runs" / "memcirc" / "behavioral_channels"
    models = [
        ("p1b", "p1b_epoch5", 8),
        ("p69", "p69_epoch5", 16),
        ("p12b", "p12b_epoch5", 24),
        ("neo", "neo_epoch5", 16),
        ("opt67", "opt67_epoch5", 24),
        ("falcon", "falcon7b_epoch5_v2", 16),
        ("mistral", "mistral_epoch5_v2", 16),
        ("llama3", "llama3_epoch5_v2", 16),
        ("qwen2", "qwen2_epoch5", 16),
        ("gemma2_2b", "gemma2_2b_epoch5", 16),
    ]

    all_results = []
    for short_name, bcd_subdir, sae_layer in models:
        bcd_dir = str(bcd_base / bcd_subdir)

        # Read SAE path from config.json
        cfg_path = os.path.join(bcd_dir, "config.json")
        if not os.path.exists(cfg_path):
            print(f"SKIP {short_name}: no config.json", file=sys.stderr)
            continue

        with open(cfg_path) as f:
            cfg = json.load(f)
        sae_path = cfg.get("sae_path", "")

        # Make path absolute if relative
        if not os.path.isabs(sae_path):
            sae_path = str(ROOT / sae_path)

        if not os.path.exists(sae_path):
            print(f"SKIP {short_name}: SAE not found at {sae_path}", file=sys.stderr)
            continue

        print(f"Processing {short_name}...", file=sys.stderr)
        result = run_model(short_name, bcd_dir, sae_path, sae_layer, args.n_random, rng)
        if result:
            all_results.append(result)

    if args.json:
        print(json.dumps(all_results, indent=2))
    else:
        print("\n" + "=" * 100)
        print("FSC RANDOM-FEATURE NULL BASELINE")
        print("=" * 100)

        for r in all_results:
            null = r["null_at_cf_size"]
            print(f"\n--- {r['model']} ---")
            print(f"  d_sae={r['d_sae']}, n_cf={r['n_cf']}, S_K rank={r['S_K_rank']}, d_model={r['d_model']}")
            print(f"  Observed FSC_K (CF):  {r['observed_fsc_K_cf']:.4f}")
            print(f"  Random null (n={null['n_random']}, size={null['subset_size']}):")
            print(f"    mean={null['mean']:.4f}, std={null['std']:.4f}")
            print(f"    p50={null['p50']:.4f}, p95={null['p95']:.4f}, p99={null['p99']:.4f}")
            print(f"    min={null['min']:.4f}, max={null['max']:.4f}")
            print(f"  Observed percentile rank: {null['observed_percentile_rank']:.1f}%")

            for sz_str, extra in r.get("null_at_extra_sizes", {}).items():
                print(f"  Random null (size={extra['subset_size']}):")
                print(f"    mean={extra['mean']:.4f}, std={extra['std']:.4f}, p95={extra['p95']:.4f}, p99={extra['p99']:.4f}")

        # Summary table
        print("\n" + "=" * 100)
        print("SUMMARY TABLE")
        print(f"{'Model':<10} {'n_cf':>6} {'d_sae':>8} {'Obs FSC':>10} {'Null mean':>10} {'Null p95':>10} {'Null p99':>10} {'Obs %ile':>10}")
        print("-" * 80)
        for r in all_results:
            null = r["null_at_cf_size"]
            print(f"{r['model']:<10} {r['n_cf']:>6} {r['d_sae']:>8} {r['observed_fsc_K_cf']:>10.4f} {null['mean']:>10.4f} {null['p95']:>10.4f} {null['p99']:>10.4f} {null['observed_percentile_rank']:>9.1f}%")


if __name__ == "__main__":
    main()
