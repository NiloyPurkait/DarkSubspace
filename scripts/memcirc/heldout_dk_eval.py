#!/usr/bin/env python3
"""heldout_dk_eval.py.

Held-out partition-fit falsification probe for d_K. Refits d_K on a 70
percent split, evaluates score-K AUROC drop on the disjoint 30 percent
across 10 splits with n_boot=10000.

Used in the held-out d_K protocol appendix.
Reproduce:
    env/bin/python3 scripts/memcirc/heldout_dk_eval.py \\
        --output paper5/results/heldout_dk_2026-05-02.json \\
        --train-frac 0.7 --n-bootstrap 10000 --n-splits 10 --device cpu

Goal. Falsification probe on the BCD direction d_K.
  - Re-fit d_K on a 70 percent partition of the canonical N=2000 cache (1400 examples).
  - Evaluate residual, SAE-reconstructed, and original score_K AUROC plus drop on
    the disjoint 30 percent held-out partition (600 examples).
  - Compare held-out drop vs in-partition drop (from on-disk per_text_scores) using
    paired bootstrap CIs (n_boot=10000).
  - Pass criterion. Held-out drop > in-partition drop minus 2 sigma (not purely
    partition-fit).

This is a falsification probe. A NULL outcome (held-out below in-partition
by more than 2 sigma) is publishable and is disclosed honestly.

Inputs (canonical caches).
  P69 layer 16 (Pythia-6.9B FT epoch 5).
    runs/memcirc/activations_canonical/p69_epoch5_layer16_member.npy   (1000, 4096)
    runs/memcirc/activations_canonical/p69_epoch5_layer16_nonmember.npy (1000, 4096)
  P12B layer 18 (Pythia-12B FT epoch 5).
    runs/memcirc/activations_canonical/p12b_epoch5_layer18_member.npy   (1000, 5120)
    runs/memcirc/activations_canonical/p12b_epoch5_layer18_nonmember.npy (1000, 5120)

SAE checkpoints (highest reconstruction cosine per anchor).
  P69 mixed-data canonical (N=6 anchor seed 43, drop 0.213, recon_cos 0.979).
  P12B fresh-init canonical (seed 49 at recon_cos 0.992, drop 0.160).

Output JSON. paper5/results/heldout_dk_2026-05-02.json.

Decision rule (per anchor).
  PASS if heldout_drop_mean > in_partition_drop - 2 * in_partition_drop_sigma.
  FAIL otherwise.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata


def _fast_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney rank-AUROC; ~9x faster than sklearn for bootstrap loops.

    Equivalent to sklearn.metrics.roc_auc_score for binary 0/1 labels.
    """
    pos_mask = labels.astype(bool)
    n_pos = int(pos_mask.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = rankdata(scores)
    return float((ranks[pos_mask].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))


# Anchors: each entry maps to (canonical activation cache, SAE for that anchor,
# in-partition drop estimate read from on-disk per_text_scores at the anchor seed).
ANCHORS: List[Dict] = [
    {
        "label": "P69 N=6 anchor (Pythia-6.9B layer 16, seed 43)",
        "model": "p69",
        "layer": 16,
        "used_seed": 43,
        "member_acts": "runs/memcirc/activations_canonical/p69_epoch5_layer16_member.npy",
        "nonmember_acts": "runs/memcirc/activations_canonical/p69_epoch5_layer16_nonmember.npy",
        "sae_path": "runs/sae/train_sae__runs_controlled_ft_run_20260306_055225_ft_epoch5_model__layer16__mult4__l10.0005__20260413_184801/sae_final.pt",
        "anchor_results_json": "runs/memcirc/sae_dark_subspace/p69_mixed_sae_seed43/results.json",
        # Fallback in-partition stats if the per_text_scores are unavailable.
        # These numbers are read from disk before use.
        "in_partition_drop_anchor": 0.2129,
        "in_partition_drop_sigma": 0.0034,  # std across N=6 P69 seeds 42..47
    },
    {
        "label": "P12B fresh-init anchor (Pythia-12B layer 18, seed 49)",
        "model": "p12b",
        "layer": 18,
        "used_seed": 49,
        "member_acts": "runs/memcirc/activations_canonical/p12b_epoch5_layer18_member.npy",
        "nonmember_acts": "runs/memcirc/activations_canonical/p12b_epoch5_layer18_nonmember.npy",
        "sae_path": "runs/sae_array/p12b_freshinit/task2_seed49/train_sae__runs_controlled_ft_run_20260308_001316_ft_epoch5_model__layer18__mult4__l10.0005__20260501_222705/sae_final.pt",
        "anchor_results_json": "runs/memcirc/sae_dark_subspace/p12b_mixed_sae_seed49/results.json",
        "in_partition_drop_anchor": 0.1602,
        "in_partition_drop_sigma": 0.015,  # std across the three-init P12B cohort (seeds 47..49)
    },
]


def _verify_inputs(anchor: Dict, repo_root: Path = REPO_ROOT) -> Tuple[bool, str]:
    missing = []
    for k in ("member_acts", "nonmember_acts", "sae_path"):
        path = repo_root / anchor[k]
        if not path.exists():
            missing.append((k, str(path)))
    if missing:
        return False, "missing inputs: " + ", ".join(f"{k}={p}" for k, p in missing)
    return True, "all inputs present"


def _load_acts_npy(p: Path) -> np.ndarray:
    return np.load(p).astype(np.float32)


def _bidirectional_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    a = _fast_auroc(labels, scores)
    b = _fast_auroc(labels, -scores)
    return float(max(a, b))


def _signed_auroc(labels: np.ndarray, scores: np.ndarray, sign: int) -> float:
    """Compute AUROC with a fixed sign (+1 = members above, -1 = below).

    Used to keep the sign consistent across train/test partitions, matching the
    canonical sign-fix at training time so flips do not artificially inflate
    held-out AUROC.
    """
    s = scores if sign >= 0 else -scores
    return float(_fast_auroc(labels, s))


def _fit_dk_mean_diff(
    member_acts: np.ndarray, nonmember_acts: np.ndarray
) -> Tuple[np.ndarray, int]:
    """Fit d_K as sign-fixed mean-difference (members > nonmembers).

    Returns (d_K_unit, sign_fixed=+1).
    """
    d = member_acts.mean(axis=0) - nonmember_acts.mean(axis=0)
    nrm = float(np.linalg.norm(d))
    if nrm < 1e-12:
        raise ValueError("d_K has near-zero norm")
    return d / nrm, +1


def _sae_reconstruct(
    activations: np.ndarray, sae_path: Path, device: str = "cpu", batch: int = 256,
) -> np.ndarray:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "memcirc"))
    import _bootstrap  # noqa: F401
    from sae_mia_audit.sae.io import load_sae_checkpoint_any
    sae = load_sae_checkpoint_any(str(sae_path), device=device)
    sae.eval()
    h = torch.tensor(activations, dtype=torch.float32, device=device)
    out_chunks = []
    with torch.no_grad():
        for i in range(0, len(h), batch):
            z = sae.encode(h[i:i + batch])
            h_hat = sae.decode(z)
            out_chunks.append(h_hat.cpu().float().numpy())
    return np.concatenate(out_chunks, axis=0)


def _bootstrap_ci_paired(
    labels: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_boot: int = 10000,
    rng: Optional[np.random.Generator] = None,
    sign_a: int = +1,
    sign_b: int = +1,
) -> Tuple[float, float, float, float, float, float]:
    """Paired bootstrap on (AUROC_a - AUROC_b).

    Resample example indices with replacement; on each resample compute both
    AUROCs and their difference. Returns:
      (auroc_a_mean, auroc_b_mean, drop_mean, drop_ci_low, drop_ci_high, drop_std)
    where drop = AUROC_a - AUROC_b is the in-partition vs reconstruction drop
    (positive = score_K_orig > score_K_recon, ie. dark subspace effect).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n = len(labels)
    drops_l = []
    auroc_a_l = []
    auroc_b_l = []
    s_a_signed = score_a if sign_a >= 0 else -score_a
    s_b_signed = score_b if sign_b >= 0 else -score_b

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        l = labels[idx]
        if l.sum() == 0 or l.sum() == n:
            # degenerate fold (all-one-class) -> skip
            continue
        a = _fast_auroc(l, s_a_signed[idx])
        b = _fast_auroc(l, s_b_signed[idx])
        auroc_a_l.append(a)
        auroc_b_l.append(b)
        drops_l.append(a - b)

    drops_v = np.asarray(drops_l)
    auroc_a_v = np.asarray(auroc_a_l)
    auroc_b_v = np.asarray(auroc_b_l)

    return (
        float(auroc_a_v.mean()),
        float(auroc_b_v.mean()),
        float(drops_v.mean()),
        float(np.quantile(drops_v, 0.025)),
        float(np.quantile(drops_v, 0.975)),
        float(drops_v.std()),
    )


def _bootstrap_ci_auroc(
    labels: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 10000,
    rng: Optional[np.random.Generator] = None,
    sign: int = +1,
) -> Tuple[float, float, float]:
    """Bootstrap CI on a single AUROC (signed)."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(labels)
    s_signed = scores if sign >= 0 else -scores
    aurocs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        l = labels[idx]
        if l.sum() == 0 or l.sum() == n:
            continue
        aurocs.append(_fast_auroc(l, s_signed[idx]))
    if not aurocs:
        return 0.5, 0.5, 0.5
    a = np.array(aurocs)
    return float(a.mean()), float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))


def evaluate_split(
    member_acts: np.ndarray,
    nonmember_acts: np.ndarray,
    sae_path: Path,
    split_seed: int,
    train_frac: float = 0.7,
    n_boot: int = 10000,
    device: str = "cpu",
) -> Dict:
    """One held-out split: re-fit d_K on train_frac partition, eval on rest."""
    rng = np.random.default_rng(split_seed)

    n_mem, n_non = len(member_acts), len(nonmember_acts)
    n_mem_train = int(round(train_frac * n_mem))
    n_non_train = int(round(train_frac * n_non))

    mem_perm = rng.permutation(n_mem)
    non_perm = rng.permutation(n_non)
    mem_a, mem_b = mem_perm[:n_mem_train], mem_perm[n_mem_train:]
    non_a, non_b = non_perm[:n_non_train], non_perm[n_non_train:]

    # Fit d_K on partition A (train).
    d_K, sign = _fit_dk_mean_diff(member_acts[mem_a], nonmember_acts[non_a])

    # Held-out partition B (test).
    acts_b = np.concatenate([member_acts[mem_b], nonmember_acts[non_b]], axis=0)
    labels_b = np.concatenate([
        np.ones(len(mem_b), dtype=np.int64),
        np.zeros(len(non_b), dtype=np.int64),
    ])

    # Center on B.
    centered_b = acts_b - acts_b.mean(axis=0, keepdims=True)
    scores_orig = centered_b @ d_K

    # SAE reconstruct B.
    h_hat_b = _sae_reconstruct(acts_b, sae_path, device=device)
    recon_centered_b = h_hat_b - h_hat_b.mean(axis=0, keepdims=True)
    scores_recon = recon_centered_b @ d_K

    residual_centered_b = centered_b - recon_centered_b
    scores_resid = residual_centered_b @ d_K

    # Reconstruction quality.
    recon_cos = float(np.mean([
        np.dot(acts_b[i], h_hat_b[i])
        / (np.linalg.norm(acts_b[i]) * np.linalg.norm(h_hat_b[i]) + 1e-12)
        for i in range(len(acts_b))
    ]))

    # Held-out drop (orig - recon) with paired bootstrap CI.
    auroc_orig, auroc_recon, drop_mean, drop_lo, drop_hi, drop_std = _bootstrap_ci_paired(
        labels_b, scores_orig, scores_recon,
        n_boot=n_boot, rng=np.random.default_rng(split_seed + 1000),
        sign_a=sign, sign_b=sign,
    )
    auroc_resid_mean, auroc_resid_lo, auroc_resid_hi = _bootstrap_ci_auroc(
        labels_b, scores_resid,
        n_boot=n_boot, rng=np.random.default_rng(split_seed + 2000),
        sign=sign,
    )

    return dict(
        split_seed=int(split_seed),
        n_mem_train=int(len(mem_a)), n_non_train=int(len(non_a)),
        n_mem_test=int(len(mem_b)), n_non_test=int(len(non_b)),
        sign_fixed=int(sign),
        recon_cos=float(recon_cos),
        # Held-out point AUROCs (mean from bootstrap)
        auroc_original_b=float(auroc_orig),
        auroc_recon_b=float(auroc_recon),
        auroc_residual_b=float(auroc_resid_mean),
        auroc_residual_b_ci_low=float(auroc_resid_lo),
        auroc_residual_b_ci_high=float(auroc_resid_hi),
        # Held-out drop with paired bootstrap CI
        heldout_drop=float(drop_mean),
        heldout_drop_ci_low=float(drop_lo),
        heldout_drop_ci_high=float(drop_hi),
        heldout_drop_std=float(drop_std),
    )


def _read_anchor_in_partition(anchor: Dict, repo_root: Path,
                              n_boot: int = 10000) -> Dict:
    """Read in-partition drop from on-disk per_text_scores at the anchor seed."""
    p = repo_root / anchor["anchor_results_json"]
    if not p.exists():
        return {
            "in_partition_drop_mean": float(anchor["in_partition_drop_anchor"]),
            "in_partition_drop_ci_low": float(anchor["in_partition_drop_anchor"] - 2 * anchor["in_partition_drop_sigma"]),
            "in_partition_drop_ci_high": float(anchor["in_partition_drop_anchor"] + 2 * anchor["in_partition_drop_sigma"]),
            "in_partition_drop_std": float(anchor["in_partition_drop_sigma"]),
            "in_partition_auroc_orig": None,
            "in_partition_auroc_recon": None,
            "in_partition_auroc_residual": None,
            "anchor_recon_cos": None,
            "anchor_results_json": str(anchor["anchor_results_json"]),
            "source": "fallback (anchor results.json not found)",
        }
    res = json.load(open(p))
    pts = res.get("per_text_scores", {})
    labels = np.array(pts["labels"], dtype=np.int64)
    scores_orig = np.array(pts["score_K_original"], dtype=np.float64)
    scores_recon = np.array(pts["score_K_recon"], dtype=np.float64)
    scores_resid = np.array(pts["score_K_residual"], dtype=np.float64)

    # Determine sign from on-disk reported AUROCs (canonical sign-fix is
    # already baked in to the bidirectional_auroc reporting; we recover the
    # consistent sign by matching whichever direction yields the on-disk AUROC
    # numbers to within rounding).
    reported_orig = res["original"]["score_K_auroc"]
    pos_a = _fast_auroc(labels, scores_orig)
    sign = +1 if abs(pos_a - reported_orig) < abs((1 - pos_a) - reported_orig) else -1

    # Paired bootstrap on the in-partition drop.
    o_mean, r_mean, drop_mean, drop_lo, drop_hi, drop_std = _bootstrap_ci_paired(
        labels, scores_orig, scores_recon,
        n_boot=n_boot, rng=np.random.default_rng(7),
        sign_a=sign, sign_b=sign,
    )
    res_mean, res_lo, res_hi = _bootstrap_ci_auroc(
        labels, scores_resid, n_boot=n_boot,
        rng=np.random.default_rng(11), sign=sign,
    )
    return {
        "in_partition_drop_mean": float(drop_mean),
        "in_partition_drop_ci_low": float(drop_lo),
        "in_partition_drop_ci_high": float(drop_hi),
        "in_partition_drop_std": float(drop_std),
        "in_partition_auroc_orig": float(o_mean),
        "in_partition_auroc_recon": float(r_mean),
        "in_partition_auroc_residual": float(res_mean),
        "in_partition_auroc_residual_ci_low": float(res_lo),
        "in_partition_auroc_residual_ci_high": float(res_hi),
        "anchor_recon_cos": float(res.get("sae_quality", {}).get("reconstruction_cosine", float("nan"))),
        "anchor_results_json": str(anchor["anchor_results_json"]),
        "anchor_sign_fixed": int(sign),
        "source": "on-disk per_text_scores",
    }


def run_anchor(anchor: Dict, n_splits: int, train_frac: float, n_boot: int,
               repo_root: Path, device: str = "cpu") -> Dict:
    label = anchor["label"]
    print(f"\n=== anchor: {label}")
    mem_path = repo_root / anchor["member_acts"]
    non_path = repo_root / anchor["nonmember_acts"]
    sae_path = repo_root / anchor["sae_path"]

    member_acts = _load_acts_npy(mem_path)
    nonmember_acts = _load_acts_npy(non_path)
    print(f"  member={member_acts.shape} nonmember={nonmember_acts.shape}")

    in_part = _read_anchor_in_partition(anchor, repo_root, n_boot=n_boot)
    print(f"  in-partition drop: {in_part['in_partition_drop_mean']:.4f} "
          f"95% CI [{in_part['in_partition_drop_ci_low']:.4f}, "
          f"{in_part['in_partition_drop_ci_high']:.4f}] "
          f"(source: {in_part['source']})")

    splits_out = []
    for s in range(n_splits):
        out = evaluate_split(
            member_acts, nonmember_acts, sae_path,
            split_seed=s, train_frac=train_frac,
            n_boot=n_boot, device=device,
        )
        splits_out.append(out)
        print(f"  split {s}: orig={out['auroc_original_b']:.4f} "
              f"recon={out['auroc_recon_b']:.4f} "
              f"resid={out['auroc_residual_b']:.4f} "
              f"drop={out['heldout_drop']:.4f} "
              f"recon_cos={out['recon_cos']:.4f}")

    drops = np.array([s["heldout_drop"] for s in splits_out])
    resid = np.array([s["auroc_residual_b"] for s in splits_out])
    orig = np.array([s["auroc_original_b"] for s in splits_out])
    recon = np.array([s["auroc_recon_b"] for s in splits_out])
    cos = np.array([s["recon_cos"] for s in splits_out])

    heldout_drop_mean = float(drops.mean())
    heldout_drop_std = float(drops.std())
    heldout_drop_ci_low = float(np.quantile(drops, 0.025))
    heldout_drop_ci_high = float(np.quantile(drops, 0.975))

    in_part_mean = in_part["in_partition_drop_mean"]
    in_part_sigma = in_part["in_partition_drop_std"] if in_part["in_partition_drop_std"] > 0 else anchor["in_partition_drop_sigma"]
    in_part_2sigma_threshold = in_part_mean - 2.0 * in_part_sigma
    drop_difference = heldout_drop_mean - in_part_mean
    partition_fit_check_passed = bool(heldout_drop_mean > in_part_2sigma_threshold)

    summary = dict(
        anchor=anchor["label"],
        model=anchor["model"],
        layer=anchor["layer"],
        used_seed=anchor["used_seed"],
        member_acts_path=anchor["member_acts"],
        nonmember_acts_path=anchor["nonmember_acts"],
        sae_path=anchor["sae_path"],
        anchor_recon_cos=in_part["anchor_recon_cos"],
        n_splits=n_splits,
        train_frac=float(train_frac),
        n_bootstrap=int(n_boot),
        n_mem_total=int(len(member_acts)),
        n_non_total=int(len(nonmember_acts)),
        # In-partition bench:
        in_partition_drop_mean=float(in_part_mean),
        in_partition_drop_ci_low=float(in_part["in_partition_drop_ci_low"]),
        in_partition_drop_ci_high=float(in_part["in_partition_drop_ci_high"]),
        in_partition_drop_std=float(in_part_sigma),
        in_partition_auroc_orig=in_part.get("in_partition_auroc_orig"),
        in_partition_auroc_recon=in_part.get("in_partition_auroc_recon"),
        in_partition_auroc_residual=in_part.get("in_partition_auroc_residual"),
        # Held-out bench (mean over n_splits):
        heldout_drop_mean=heldout_drop_mean,
        heldout_drop_ci_low=heldout_drop_ci_low,
        heldout_drop_ci_high=heldout_drop_ci_high,
        heldout_drop_std=heldout_drop_std,
        heldout_auroc_original_mean=float(orig.mean()),
        heldout_auroc_original_std=float(orig.std()),
        heldout_auroc_recon_mean=float(recon.mean()),
        heldout_auroc_recon_std=float(recon.std()),
        heldout_auroc_residual_mean=float(resid.mean()),
        heldout_auroc_residual_std=float(resid.std()),
        heldout_recon_cos_mean=float(cos.mean()),
        heldout_recon_cos_std=float(cos.std()),
        # Comparison:
        drop_difference=float(drop_difference),  # heldout - in_partition
        in_part_minus_2sigma_threshold=float(in_part_2sigma_threshold),
        partition_fit_check_passed=partition_fit_check_passed,
        per_split=splits_out,
    )

    # Pre-registered decision rule.
    delta = abs(heldout_drop_mean - in_part_mean)
    if delta <= in_part_sigma and resid.mean() >= 0.70:
        decision = "ACCEPT"
    elif delta <= 2 * in_part_sigma:
        decision = "MARGINAL"
    else:
        decision = "NULL"
    summary["pre_reg_decision"] = decision
    return summary


def main():
    ap = argparse.ArgumentParser(description="Held-out d_K direction estimation")
    ap.add_argument("--n-splits", type=int, default=10,
                    help="Number of random 70/30 splits (default 10).")
    ap.add_argument("--train-frac", type=float, default=0.7,
                    help="Train (d_K-fit) partition fraction (default 0.7).")
    ap.add_argument("--n-bootstrap", type=int, default=10000,
                    help="Bootstrap resamples for paired-bootstrap CIs.")
    ap.add_argument("--device", default="cpu", help="cpu or cuda.")
    ap.add_argument("--output",
                    default="paper5/results/heldout_dk_2026-05-02.json")
    ap.add_argument("--anchors", default="all", help="Comma-sep subset of "
                    "anchor models (e.g. 'p69' or 'p69,p12b'); default 'all'.")
    args = ap.parse_args()

    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Pre-flight: identify which anchors are feasible.
    print("=" * 78)
    print("Held-out d_K direction estimation (canonical N=2000, 70/30)")
    print(f"  n_splits={args.n_splits} train_frac={args.train_frac} "
          f"n_boot={args.n_bootstrap} device={args.device}")
    print()
    print("Pre-flight check on anchor inputs:")

    anchor_subset = ANCHORS if args.anchors == "all" else [
        a for a in ANCHORS if a["model"] in [s.strip() for s in args.anchors.split(",")]
    ]

    blockers = []
    feasible = []
    for anchor in anchor_subset:
        ok, msg = _verify_inputs(anchor)
        print(f"  {anchor['label']}: {'OK' if ok else 'BLOCKED'}  ({msg})")
        if ok:
            feasible.append(anchor)
        else:
            blockers.append({"anchor": anchor["label"], "reason": msg})

    if not feasible:
        out = {
            "experiment": "held-out d_K (BLOCKED -- no feasible anchors)",
            "status": "BLOCKED",
            "blockers": blockers,
            "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print()
        print("STATUS: BLOCKED. Wrote blocker manifest to:")
        print(f"  {out_path}")
        sys.exit(2)

    anchor_results = []
    for anchor in feasible:
        anchor_results.append(run_anchor(
            anchor,
            n_splits=args.n_splits,
            train_frac=args.train_frac,
            n_boot=args.n_bootstrap,
            repo_root=REPO_ROOT,
            device=args.device,
        ))

    output = {
        "experiment": "held-out d_K direction estimation",
        "split_design": {
            "train_frac": float(args.train_frac),
            "n_splits": int(args.n_splits),
            "n_bootstrap": int(args.n_bootstrap),
            "split_seeds": list(range(args.n_splits)),
            "n_total_canonical": 2000,
            "n_train_per_split": int(round(args.train_frac * 2000)),
            "n_test_per_split": 2000 - int(round(args.train_frac * 2000)),
        },
        "anchors": anchor_results,
        "blockers_skipped": blockers,
        "elapsed_sec": float(time.time() - t0),
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "script": "scripts/memcirc/heldout_dk_eval.py",
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Summary print.
    print()
    print("=" * 78)
    print("RESULT SUMMARY")
    print("=" * 78)
    for r in anchor_results:
        print(f"  {r['anchor']}")
        print(f"    in-partition drop: {r['in_partition_drop_mean']:.4f} "
              f"95% CI [{r['in_partition_drop_ci_low']:.4f}, "
              f"{r['in_partition_drop_ci_high']:.4f}] "
              f"sigma={r['in_partition_drop_std']:.4f}")
        print(f"    held-out drop:     {r['heldout_drop_mean']:.4f} "
              f"95% CI [{r['heldout_drop_ci_low']:.4f}, "
              f"{r['heldout_drop_ci_high']:.4f}] "
              f"sigma={r['heldout_drop_std']:.4f}")
        print(f"    held-out residual AUROC: {r['heldout_auroc_residual_mean']:.4f} +/- {r['heldout_auroc_residual_std']:.4f}")
        print(f"    drop difference (held-out - in-partition): {r['drop_difference']:.4f}")
        print(f"    in_part - 2 sigma = {r['in_part_minus_2sigma_threshold']:.4f}")
        print(f"    PASS criterion (heldout > in_part - 2 sigma): "
              f"{'PASS' if r['partition_fit_check_passed'] else 'FAIL'}")
        print(f"    pre-registered decision: {r['pre_reg_decision']}")
        print()
    print(f"  wrote {out_path}")
    print(f"  elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
