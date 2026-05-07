#!/usr/bin/env python3
"""label_shuffled_null.py.

Permutation null on the cosine alignment between the leading knowledge
direction $d_K$ and the leading recall direction $d_R$.

Tests whether the observed $\\cos(d_K, d_R)$ is plausible under a null in
which the member/non-member labels are randomly shuffled before fitting
$d_K$. A small two-sided p-value is evidence that the observed alignment is
not an artefact of label assignment.

Used in Appendix `app:label_shuffled_null` of the paper.

Reproduce:
    env/bin/python3 scripts/dark_subspace/label_shuffled_null.py \\
        --activations runs/dark_subspace/canonical_activations/<model_tag>/activations.npz \\
        --bcd-dir runs/dark_subspace/behavioral_channels/<model_tag> \\
        --output-dir runs/dark_subspace/label_shuffled_null/<model_tag> \\
        --layer 16 --n-permutations 1000 --seed 42

Inputs.
  ``--activations``: ``.npz`` with arrays ``H_member`` and ``H_nonmember``
      (mean-pooled residual-stream activations at the analysis layer for the
      member and non-member documents respectively). Produced by
      ``scripts/dark_subspace/extract_canonical_activations.py``.
  ``--bcd-dir``: directory containing ``directions.npz`` from the matching
      ``behavioral_channels.py`` run. The observed ``d_R`` is loaded from
      ``d_R_layer<L>``.

Outputs.
  ``results.json`` with the observed cosine, the null distribution
  (mean, std, percentiles, two-sided p-value), and run metadata.
"""

import _bootstrap  # noqa: F401

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from behavioral_channels import contrastive_pca


log = logging.getLogger("label_shuffled_null")


def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Permutation null on cos(d_K, d_R): re-fits d_K under shuffled "
            "member/non-member labels and compares the resulting cosine "
            "distribution to the observed value."
        )
    )
    ap.add_argument("--activations", type=Path, required=True,
                    help="NPZ with H_member and H_nonmember mean-pooled "
                         "activations at the analysis layer.")
    ap.add_argument("--bcd-dir", type=Path, required=True,
                    help="behavioral_channels.py output directory; "
                         "directions.npz is read for d_R.")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Directory to write results.json.")
    ap.add_argument("--layer", type=int, required=True,
                    help="Analysis layer index used for d_R lookup.")
    ap.add_argument("--n-permutations", type=int, default=1000,
                    help="Number of label-shuffled permutations (default 1000).")
    ap.add_argument("--cpca-alpha", type=float, default=1.0,
                    help="Contrastive PCA alpha matching the BCD run.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for the permutation draws.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    act = np.load(args.activations)
    if "H_member" not in act or "H_nonmember" not in act:
        raise KeyError(
            f"{args.activations} must contain 'H_member' and 'H_nonmember' arrays"
        )
    H_mem = act["H_member"].astype(np.float64)
    H_non = act["H_nonmember"].astype(np.float64)
    log.info(f"H_member shape={H_mem.shape}, H_nonmember shape={H_non.shape}")

    directions_path = args.bcd_dir / "directions.npz"
    if not directions_path.exists():
        raise FileNotFoundError(f"Missing {directions_path}")
    directions = np.load(directions_path)
    d_R_key = f"d_R_layer{args.layer}"
    if d_R_key not in directions.files:
        raise KeyError(
            f"{directions_path} has no key {d_R_key}; available keys: "
            f"{sorted(directions.files)}"
        )
    d_R = directions[d_R_key].astype(np.float64)
    log.info(f"loaded d_R for layer {args.layer} from {directions_path}")

    _, _, observed_d_K = contrastive_pca(H_mem, H_non, n_components=1, alpha=args.cpca_alpha)
    observed_cos = _cosine(observed_d_K, d_R)
    log.info(f"observed cos(d_K, d_R) = {observed_cos:.6f}")

    H_all = np.concatenate([H_mem, H_non], axis=0)
    n_mem = len(H_mem)
    n_total = len(H_all)

    rng = np.random.default_rng(args.seed)
    null_cosines = np.empty(args.n_permutations, dtype=np.float64)
    for i in range(args.n_permutations):
        perm_idx = rng.permutation(n_total)
        mem_idx = perm_idx[:n_mem]
        non_idx = perm_idx[n_mem:]
        _, _, d_K_perm = contrastive_pca(
            H_all[mem_idx], H_all[non_idx], n_components=1, alpha=args.cpca_alpha,
        )
        null_cosines[i] = _cosine(d_K_perm, d_R)
        if (i + 1) % 100 == 0:
            log.info(f"  permutation {i+1}/{args.n_permutations}")

    null_mean = float(null_cosines.mean())
    null_std = float(null_cosines.std(ddof=1))
    p_two_sided = float((np.abs(null_cosines) >= abs(observed_cos)).sum() + 1) / (args.n_permutations + 1)

    results = {
        "activations": str(args.activations),
        "bcd_dir": str(args.bcd_dir),
        "layer": args.layer,
        "n_permutations": args.n_permutations,
        "cpca_alpha": args.cpca_alpha,
        "seed": args.seed,
        "n_member": int(n_mem),
        "n_nonmember": int(n_total - n_mem),
        "observed_cosine_d_K_d_R": observed_cos,
        "null": {
            "mean": null_mean,
            "std": null_std,
            "p_two_sided": p_two_sided,
            "percentile_2p5": float(np.percentile(null_cosines, 2.5)),
            "percentile_97p5": float(np.percentile(null_cosines, 97.5)),
            "values": null_cosines.tolist(),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info(f"wrote {out_path}")
    print(
        f"observed cos(d_K, d_R) = {observed_cos:.4f} | "
        f"null mean={null_mean:+.4f} std={null_std:.4f} | "
        f"two-sided p = {p_two_sided:.4f}"
    )


if __name__ == "__main__":
    main()
