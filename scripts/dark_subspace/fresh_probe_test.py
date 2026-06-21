#!/usr/bin/env python3
"""fresh_probe_test.py.

Trains fresh L2-regularised logistic regression probes on full activation
vectors (original, reconstruction, residual) under the privacy-aware SAE
conditions and emits the shuffle permutation null calibration.

Used in Section "Results" (R:124-127) and Appendix ``app:fresh_probe_v2``
(A:807-825) of the paper.
Reproduce: .venv/bin/python scripts/dark_subspace/fresh_probe_test.py --model-path <ft_model> --sae-path <sae> --bcd-dir <bcd_dir> --member-texts <member.jsonl> --nonmember-texts <nonmember.jsonl> --layer <L> --output-dir <out> --model-id <id>

Breaks the circularity where d_K is used for both SAE optimisation
(`finetune_sae_dk.py` minimises `mean((residual @ d_K)^2)`) and evaluation
(`score_K` = AUROC of activation projections onto d_K). Instead of
score_K, we train a logistic regression probe on the FULL reconstructed
activation vectors with k-fold stratified cross-validation.

If the privacy-aware SAE truly removes membership information (not just
the d_K projection), the probe AUROC on reconstructed activations should
drop to chance. If it only removes the d_K component, the probe may
still detect membership through other features.

Outputs:
  - probe_auroc on original, reconstructed, and residual activations
  - score_K auroc on original, reconstructed, and residual (for comparison)
  - per-fold probe AUROCs (to assess variance)
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
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

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
    """Load texts from a JSONL file."""
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
# Activation collection (same as sae_dark_subspace.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_activations(
    model, tokenizer, texts, layer, seq_len, batch_size, device
) -> np.ndarray:
    """Collect mean-pooled hidden states at a specific layer."""
    all_acts = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Collecting activations"):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=seq_len,
            padding=True,
        ).to(device)
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[layer]  # (B, T, D)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, D)
        all_acts.append(pooled.cpu().float().numpy())

    return np.concatenate(all_acts, axis=0)


# ---------------------------------------------------------------------------
# SAE decomposition
# ---------------------------------------------------------------------------

@torch.no_grad()
def sae_decompose(
    activations: np.ndarray, sae, device: str, batch_size: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """Run SAE encode-decode, return (reconstructed, residual)."""
    h_tensor = torch.tensor(activations, dtype=torch.float32, device=device)
    all_recon = []

    for i in range(0, len(h_tensor), batch_size):
        batch = h_tensor[i : i + batch_size]
        z = sae.encode(batch)
        h_hat = sae.decode(z)
        all_recon.append(h_hat.detach().cpu().float().numpy())

    reconstructed = np.concatenate(all_recon, axis=0)
    residual = activations - reconstructed
    return reconstructed, residual


# ---------------------------------------------------------------------------
# Logistic regression probe with stratified k-fold CV
# ---------------------------------------------------------------------------

def probe_auroc_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> Dict:
    """Train a logistic regression probe with stratified k-fold CV.

    Returns dict with mean AUROC, std, and per-fold AUROCs.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_aurocs = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize features (fit on train, transform both)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train logistic regression
        clf = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            C=1.0,
            random_state=seed,
        )
        clf.fit(X_train_scaled, y_train)

        # Predict probabilities and compute AUROC
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        fold_auc = roc_auc_score(y_test, y_prob)
        fold_aurocs.append(fold_auc)

        log.info(f"  Fold {fold_idx + 1}/{n_folds}: probe AUROC = {fold_auc:.4f}")

    mean_auc = float(np.mean(fold_aurocs))
    std_auc = float(np.std(fold_aurocs))
    return {
        "mean_auroc": mean_auc,
        "std_auroc": std_auc,
        "per_fold_aurocs": [float(a) for a in fold_aurocs],
    }


def _atomic_write_json(path: "Path", payload: dict) -> None:
    """Write JSON atomically: write to <path>.tmp then os.replace to <path>."""
    import os
    from pathlib import Path as _P
    path = _P(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def permutation_null_aurocs(
    X: np.ndarray,
    y: np.ndarray,
    n_permutations: int,
    n_folds: int = 5,
    seed: int = 42,
    incremental_write_path: Optional[str] = None,
    incremental_write_every: int = 0,
) -> Dict:
    """Run permutation null: shuffle labels independently, run probe each time.

    For each permutation:
      1. Shuffle y with a new random seed (independent of fold splits)
      2. Run probe_auroc_cv on (X, y_shuffled) to get mean AUROC
    Returns dict with all permutation AUROCs, mean, and std.

    If `incremental_write_path` is set and `incremental_write_every > 0`, a
    partial JSON snapshot is written atomically (tmp + os.replace) every N
    permutations so progress is not lost on preemption / timeout.
    """
    rng = np.random.RandomState(seed)
    perm_aurocs = []

    def _snapshot(final: bool = False) -> Dict:
        return {
            "n_permutations_target": n_permutations,
            "n_permutations_completed": len(perm_aurocs),
            "final": bool(final),
            "aurocs": [float(a) for a in perm_aurocs],
            "mean": float(np.mean(perm_aurocs)) if perm_aurocs else None,
            "std": float(np.std(perm_aurocs)) if perm_aurocs else None,
        }

    for i in range(n_permutations):
        y_shuffled = rng.permutation(y)
        # Use a different CV seed per permutation to avoid fold-label correlation
        perm_seed = seed + i + 1
        result = probe_auroc_cv(X, y_shuffled, n_folds=n_folds, seed=perm_seed)
        perm_aurocs.append(result["mean_auroc"])
        if (i + 1) % 10 == 0 or i == 0:
            log.info(
                f"  Permutation {i + 1}/{n_permutations}: "
                f"null AUROC = {result['mean_auroc']:.4f}"
            )
        if (
            incremental_write_path
            and incremental_write_every > 0
            and ((i + 1) % incremental_write_every == 0)
        ):
            try:
                _atomic_write_json(incremental_write_path, _snapshot(final=False))
                log.info(
                    f"  Incremental snapshot written: {incremental_write_path} "
                    f"({i + 1}/{n_permutations})"
                )
            except Exception as e:
                log.warning(f"  Incremental snapshot write failed: {e}")

    final = {
        "n_permutations": n_permutations,
        "aurocs": [float(a) for a in perm_aurocs],
        "mean": float(np.mean(perm_aurocs)),
        "std": float(np.std(perm_aurocs)),
    }
    if incremental_write_path and incremental_write_every > 0:
        try:
            _atomic_write_json(incremental_write_path, {**_snapshot(final=True), **final})
        except Exception as e:
            log.warning(f"  Final incremental snapshot write failed: {e}")
    return final


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """L2-normalize each row of X."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return X / norms


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

def run_fresh_probe_experiment(
    activations: np.ndarray,
    labels: np.ndarray,
    d_K: np.ndarray,
    global_mean: Optional[np.ndarray],
    sae,
    device: str,
    n_folds: int = 5,
    seed: int = 42,
    n_permutations: int = 0,
    do_l2_normalize: bool = False,
    incremental_write_dir: Optional[str] = None,
    incremental_write_every: int = 0,
) -> Dict:
    """Run the fresh probe experiment.

    For each of {original, reconstructed, residual} activations:
      1. Compute score_K AUROC (the circular metric, for comparison)
      2. Train a logistic regression probe with k-fold CV (the independent metric)
    """
    d_K_norm = d_K / (np.linalg.norm(d_K) + 1e-12)

    # Center activations for score_K
    if global_mean is not None:
        centered = activations - global_mean[np.newaxis, :]
    else:
        centered = activations - activations.mean(axis=0, keepdims=True)

    # --- score_K on ORIGINAL ---
    scores_original = centered @ d_K_norm
    auroc_score_K_original = bidirectional_auroc(labels, scores_original)
    log.info(f"score_K on ORIGINAL: AUROC = {auroc_score_K_original:.4f}")

    # --- Probe on ORIGINAL ---
    log.info("Training probe on ORIGINAL activations...")
    probe_original = probe_auroc_cv(activations, labels, n_folds=n_folds, seed=seed)
    log.info(
        f"Probe on ORIGINAL: mean AUROC = {probe_original['mean_auroc']:.4f} "
        f"(+/- {probe_original['std_auroc']:.4f})"
    )

    # --- SAE decomposition ---
    log.info("Running SAE encode-decode...")
    reconstructed, residual = sae_decompose(activations, sae, device)

    # Reconstruction quality
    recon_mse = float(np.mean(np.sum((activations - reconstructed) ** 2, axis=1)))
    recon_cos = float(np.mean([
        np.dot(activations[i], reconstructed[i])
        / (np.linalg.norm(activations[i]) * np.linalg.norm(reconstructed[i]) + 1e-12)
        for i in range(len(activations))
    ]))
    log.info(f"SAE reconstruction: MSE={recon_mse:.4f}, mean_cosine={recon_cos:.4f}")

    # --- score_K on RECONSTRUCTED ---
    if global_mean is not None:
        recon_centered = reconstructed - global_mean[np.newaxis, :]
    else:
        recon_centered = reconstructed - reconstructed.mean(axis=0, keepdims=True)
    scores_recon = recon_centered @ d_K_norm
    auroc_score_K_recon = bidirectional_auroc(labels, scores_recon)
    log.info(f"score_K on RECONSTRUCTED: AUROC = {auroc_score_K_recon:.4f}")

    # --- Probe on RECONSTRUCTED ---
    log.info("Training probe on RECONSTRUCTED activations...")
    probe_recon = probe_auroc_cv(reconstructed, labels, n_folds=n_folds, seed=seed)
    log.info(
        f"Probe on RECONSTRUCTED: mean AUROC = {probe_recon['mean_auroc']:.4f} "
        f"(+/- {probe_recon['std_auroc']:.4f})"
    )

    # --- score_K on RESIDUAL ---
    if global_mean is not None:
        residual_centered = centered - recon_centered
    else:
        residual_centered = residual - residual.mean(axis=0, keepdims=True)
    scores_residual = residual_centered @ d_K_norm
    auroc_score_K_residual = bidirectional_auroc(labels, scores_residual)
    log.info(f"score_K on RESIDUAL: AUROC = {auroc_score_K_residual:.4f}")

    # --- Probe on RESIDUAL ---
    log.info("Training probe on RESIDUAL activations...")
    probe_residual = probe_auroc_cv(residual, labels, n_folds=n_folds, seed=seed)
    log.info(
        f"Probe on RESIDUAL: mean AUROC = {probe_residual['mean_auroc']:.4f} "
        f"(+/- {probe_residual['std_auroc']:.4f})"
    )

    result = {
        "original": {
            "score_K_auroc": float(auroc_score_K_original),
            "probe_auroc_mean": probe_original["mean_auroc"],
            "probe_auroc_std": probe_original["std_auroc"],
            "probe_auroc_per_fold": probe_original["per_fold_aurocs"],
        },
        "sae_reconstructed": {
            "score_K_auroc": float(auroc_score_K_recon),
            "probe_auroc_mean": probe_recon["mean_auroc"],
            "probe_auroc_std": probe_recon["std_auroc"],
            "probe_auroc_per_fold": probe_recon["per_fold_aurocs"],
        },
        "residual": {
            "score_K_auroc": float(auroc_score_K_residual),
            "probe_auroc_mean": probe_residual["mean_auroc"],
            "probe_auroc_std": probe_residual["std_auroc"],
            "probe_auroc_per_fold": probe_residual["per_fold_aurocs"],
        },
        "sae_quality": {
            "reconstruction_mse": recon_mse,
            "reconstruction_cosine": recon_cos,
        },
        "circularity_check": {
            "score_K_recon_minus_probe_recon": float(
                auroc_score_K_recon - probe_recon["mean_auroc"]
            ),
            "description": (
                "If score_K shows improvement but probe does not, "
                "the improvement is circular (only along d_K). "
                "If both drop, the SAE genuinely removes membership info."
            ),
        },
    }

    # --- Permutation null ---
    if n_permutations > 0:
        log.info(f"Running permutation null ({n_permutations} permutations)...")
        perm_null = {}
        for component, X in [
            ("original", activations),
            ("sae_reconstructed", reconstructed),
            ("residual", residual),
        ]:
            log.info(f"  Permutation null on {component}...")
            inc_path = None
            if incremental_write_dir and incremental_write_every > 0:
                inc_path = f"{incremental_write_dir}/permnull_{component}.partial.json"
            perm_null[component] = permutation_null_aurocs(
                X, labels, n_permutations, n_folds=n_folds, seed=seed,
                incremental_write_path=inc_path,
                incremental_write_every=incremental_write_every,
            )
            log.info(
                f"  {component} null: mean={perm_null[component]['mean']:.4f} "
                f"std={perm_null[component]['std']:.4f}"
            )
        result["permutation_null"] = perm_null

    # --- L2-normalized probe ---
    if do_l2_normalize:
        log.info("Running L2-normalized probes...")
        acts_l2 = l2_normalize(activations)
        recon_l2 = l2_normalize(reconstructed)
        resid_l2 = l2_normalize(residual)

        l2_results = {}
        for component, X, label in [
            ("original", acts_l2, "Original (L2-norm)"),
            ("sae_reconstructed", recon_l2, "Reconstructed (L2-norm)"),
            ("residual", resid_l2, "Residual (L2-norm)"),
        ]:
            log.info(f"  Probe on {label}...")
            probe_res = probe_auroc_cv(X, labels, n_folds=n_folds, seed=seed)
            l2_results[component] = {
                "probe_auroc_mean": probe_res["mean_auroc"],
                "probe_auroc_std": probe_res["std_auroc"],
                "probe_auroc_per_fold": probe_res["per_fold_aurocs"],
            }
            log.info(
                f"  {label}: mean AUROC = {probe_res['mean_auroc']:.4f} "
                f"(+/- {probe_res['std_auroc']:.4f})"
            )

        # Also run permutation null on L2-normalized if requested
        if n_permutations > 0:
            log.info(f"Running permutation null on L2-normalized ({n_permutations} permutations)...")
            l2_perm_null = {}
            for component, X in [
                ("original", acts_l2),
                ("sae_reconstructed", recon_l2),
                ("residual", resid_l2),
            ]:
                log.info(f"  Permutation null on L2-norm {component}...")
                inc_path = None
                if incremental_write_dir and incremental_write_every > 0:
                    inc_path = f"{incremental_write_dir}/permnull_l2_{component}.partial.json"
                l2_perm_null[component] = permutation_null_aurocs(
                    X, labels, n_permutations, n_folds=n_folds, seed=seed,
                    incremental_write_path=inc_path,
                    incremental_write_every=incremental_write_every,
                )
            l2_results["permutation_null"] = l2_perm_null

        result["l2_normalized"] = l2_results

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fresh Probe Test: Independent membership classifier "
            "on SAE-reconstructed activations"
        )
    )
    parser.add_argument(
        "--model-path", required=True, help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--sae-path", required=True, help="Path to SAE checkpoint (sae_final.pt)"
    )
    parser.add_argument(
        "--bcd-dir",
        required=True,
        help="Path to channel-decomposition directions directory (for score_K comparison)",
    )
    parser.add_argument("--member-texts", required=True)
    parser.add_argument("--nonmember-texts", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-texts", type=int, default=0)
    parser.add_argument(
        "--n-permutations", type=int, default=0,
        help="Number of label-permutation null iterations (0 = skip)",
    )
    parser.add_argument(
        "--l2-normalize", action="store_true",
        help="Also run probes on L2-normalized activation vectors",
    )
    parser.add_argument(
        "--incremental-write-every", type=int, default=0,
        help=(
            "Write a partial JSON snapshot of the permutation null every N "
            "permutations (atomic tmp+replace). 0 disables. Snapshots are "
            "written to <output-dir>/permnull_<component>.partial.json."
        ),
    )
    args = parser.parse_args()

    if not _HAS_PROJECT_INFRA:
        raise RuntimeError(f"Project infrastructure required: {_IMPORT_ERROR}")
    if not _HAS_SKLEARN:
        raise RuntimeError(
            "sklearn required: pip install scikit-learn"
        )

    setup_logging(logging.INFO)
    set_global_seed(SeedConfig(seed=args.seed))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    config["script"] = "fresh_probe_test.py"
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

    # Load channel-decomposition directions (for score_K comparison only)
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
    log.info(
        f"Loaded {len(member_texts)} member, {len(nonmember_texts)} nonmember texts"
    )

    all_texts = member_texts + nonmember_texts
    labels = np.array([1] * len(member_texts) + [0] * len(nonmember_texts))

    # Collect activations
    activations = collect_activations(
        model, tokenizer, all_texts, args.layer,
        args.seq_len, args.batch_size, args.device,
    )
    log.info(f"Activations: shape={activations.shape}")

    # Free model GPU memory (SAE is small)
    del model
    torch.cuda.empty_cache()

    # Run the experiment
    results = run_fresh_probe_experiment(
        activations, labels, d_K, global_mean, sae, args.device,
        n_folds=args.n_folds, seed=args.seed,
        n_permutations=args.n_permutations,
        do_l2_normalize=args.l2_normalize,
        incremental_write_dir=str(out_dir),
        incremental_write_every=args.incremental_write_every,
    )

    # Add metadata
    output = {
        "model": args.model_id,
        "layer": args.layer,
        "sae_path": args.sae_path,
        "n_member": len(member_texts),
        "n_nonmember": len(nonmember_texts),
        "n_folds": args.n_folds,
        **results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(_sanitize_for_json(output), f, indent=2)
    log.info(f"Results saved to {results_path}")

    # Print summary
    print()
    print("=" * 76)
    print("Fresh probe test summary")
    print("=" * 76)
    print(f"Model: {args.model_id}  |  Layer: {args.layer}  |  SAE: {args.sae_path}")
    print(f"Folds: {args.n_folds}  |  N: {len(labels)} ({sum(labels)} mem, {sum(1 - labels)} nonmem)")
    print()
    print(f"  {'Component':<25} {'score_K AUROC':>14} {'Probe AUROC':>14} {'Probe Std':>12}")
    print("  " + "-" * 67)

    for component, label in [
        ("original", "Original h"),
        ("sae_reconstructed", "SAE-reconstructed h'"),
        ("residual", "Residual (h - h')"),
    ]:
        r = results[component]
        print(
            f"  {label:<25} {r['score_K_auroc']:>14.4f} "
            f"{r['probe_auroc_mean']:>14.4f} {r['probe_auroc_std']:>12.4f}"
        )

    print()
    print(f"  SAE reconstruction cosine: {results['sae_quality']['reconstruction_cosine']:.4f}")
    print()

    cc = results["circularity_check"]
    delta = cc["score_K_recon_minus_probe_recon"]
    if abs(delta) > 0.05:
        print(f"  ** CIRCULARITY DETECTED: score_K and probe disagree by {delta:+.4f} **")
        print(f"     score_K may be overly optimistic due to d_K optimization.")
    else:
        print(f"  score_K and probe agree (delta = {delta:+.4f}). No circularity detected.")

    # Permutation null summary
    if "permutation_null" in results:
        print()
        print("  Permutation Null (overfitting floor):")
        pn = results["permutation_null"]
        for component in ["original", "sae_reconstructed", "residual"]:
            if component in pn:
                print(
                    f"    {component:<25} null mean = {pn[component]['mean']:.4f} "
                    f"(+/- {pn[component]['std']:.4f})"
                )

    # L2-normalized summary
    if "l2_normalized" in results:
        print()
        print("  L2-Normalized Probes:")
        l2 = results["l2_normalized"]
        for component, label in [
            ("original", "Original (L2)"),
            ("sae_reconstructed", "Reconstructed (L2)"),
            ("residual", "Residual (L2)"),
        ]:
            if component in l2:
                r = l2[component]
                print(
                    f"    {label:<25} probe AUROC = {r['probe_auroc_mean']:.4f} "
                    f"(+/- {r['probe_auroc_std']:.4f})"
                )
        if "permutation_null" in l2:
            print("  L2-Normalized Permutation Null:")
            for component in ["original", "sae_reconstructed", "residual"]:
                if component in l2["permutation_null"]:
                    pn = l2["permutation_null"][component]
                    print(
                        f"    {component:<25} null mean = {pn['mean']:.4f} "
                        f"(+/- {pn['std']:.4f})"
                    )

    print()
    print("  Interpretation:")
    print("    If probe AUROC on reconstructed drops to ~0.50: SAE removes membership info")
    print("    If probe AUROC stays high but score_K drops: only d_K component removed (circular)")
    print("    If both stay high: SAE preserves membership info in reconstruction")
    if "permutation_null" in results:
        print("    Permutation null shows the overfitting floor: real signal = probe AUROC - null mean")
    print()


if __name__ == "__main__":
    main()
