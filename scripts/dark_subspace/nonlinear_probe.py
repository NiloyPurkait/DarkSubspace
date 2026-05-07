#!/usr/bin/env python3
"""nonlinear_probe.py.

MLP versus logistic-regression membership probe at the analysis layer.
Tests whether the residual-stream membership signal that the linear probe
recovers is also recoverable by a non-linear classifier, and whether either
class of probe substantially out-performs the other.

Used in Appendix `app:nonlinear` (`tab:nonlinear`) of the paper.

Reproduce:
    env/bin/python3 scripts/dark_subspace/nonlinear_probe.py \\
        --activations runs/dark_subspace/canonical_activations/<model_tag>/activations.npz \\
        --output-dir runs/dark_subspace/nonlinear_probe/<model_tag> \\
        --layer 16 --n-folds 5 --seed 42

Inputs.
  ``--activations``: ``.npz`` containing ``H_member`` and ``H_nonmember``,
      each of shape ``(n, d)`` for the analysis layer (mean-pooled
      residual-stream activations). Produced by
      ``scripts/dark_subspace/extract_canonical_activations.py``.

Outputs.
  ``results.json`` with stratified-K-fold AUROC for both probes, fold-matched
  pairwise comparisons, bootstrap CIs, and run metadata.
"""

import _bootstrap  # noqa: F401

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


log = logging.getLogger("nonlinear_probe")


def _tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_thresh: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    eligible = fpr <= fpr_thresh
    if not eligible.any():
        return 0.0
    return float(tpr[eligible].max())


def _bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
):
    n = len(y_true)
    aurocs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        tries = 0
        while (yt.min() == yt.max()) and tries < 10:
            idx = rng.integers(0, n, size=n)
            yt = y_true[idx]
            ys = y_score[idx]
            tries += 1
        aurocs[b] = roc_auc_score(yt, ys) if yt.min() != yt.max() else 0.5
    return {
        "auroc_mean": float(aurocs.mean()),
        "auroc_ci95_lo": float(np.percentile(aurocs, 2.5)),
        "auroc_ci95_hi": float(np.percentile(aurocs, 97.5)),
    }


def _evaluate_probe(name, model_factory, X, y, n_folds, n_boot, seed):
    log.info(f"=== Probe: {name} ===")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_scores = np.full(len(y), np.nan, dtype=np.float64)
    fold_aurocs = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr = scaler.transform(X_tr)
        X_te = scaler.transform(X_te)

        clf = model_factory(seed + fold_idx)
        clf.fit(X_tr, y_tr)
        if hasattr(clf, "decision_function"):
            scores = clf.decision_function(X_te)
        else:
            scores = clf.predict_proba(X_te)[:, 1]
        oof_scores[test_idx] = scores
        fold_auroc = roc_auc_score(y_te, scores)
        fold_aurocs.append(float(fold_auroc))
        log.info(
            f"  fold {fold_idx}: n_train={len(train_idx)} n_test={len(test_idx)} "
            f"AUROC={fold_auroc:.4f} ({time.time()-t0:.1f}s)"
        )

    assert not np.isnan(oof_scores).any()
    pooled_auroc = float(roc_auc_score(y, oof_scores))
    pooled_tpr1 = float(_tpr_at_fpr(y, oof_scores, 0.01))
    pooled_tpr5 = float(_tpr_at_fpr(y, oof_scores, 0.05))

    rng = np.random.default_rng(seed + 1000)
    boot = _bootstrap_ci(y, oof_scores, n_boot=n_boot, rng=rng)

    return {
        "probe": name,
        "n_folds": n_folds,
        "fold_aurocs": fold_aurocs,
        "fold_auroc_mean": float(np.mean(fold_aurocs)),
        "fold_auroc_std": float(np.std(fold_aurocs)),
        "pooled_auroc": pooled_auroc,
        "pooled_tpr_at_1pct_fpr": pooled_tpr1,
        "pooled_tpr_at_5pct_fpr": pooled_tpr5,
        "bootstrap": {"n_boot": n_boot, **boot},
        "oof_scores": oof_scores.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "MLP-vs-linear membership probe comparison on mean-pooled "
            "residual-stream activations at the analysis layer."
        )
    )
    ap.add_argument("--activations", type=Path, required=True,
                    help="NPZ with H_member and H_nonmember at the analysis layer.")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Directory to write results.json.")
    ap.add_argument("--layer", type=int, required=True,
                    help="Analysis layer index (recorded in metadata).")
    ap.add_argument("--n-folds", type=int, default=5,
                    help="Stratified K-fold split (default 5).")
    ap.add_argument("--n-boot", type=int, default=5000,
                    help="Bootstrap replicates for AUROC CI.")
    ap.add_argument("--mlp-hidden", type=int, nargs="+", default=[256],
                    help="MLP hidden-layer sizes (default 256).")
    ap.add_argument("--mlp-alpha", type=float, default=1e-4,
                    help="MLP L2 weight decay (default 1e-4).")
    ap.add_argument("--mlp-max-iter", type=int, default=200,
                    help="MLP training iterations (default 200).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for the fold split and the bootstrap.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    act = np.load(args.activations)
    if "H_member" not in act or "H_nonmember" not in act:
        raise KeyError(
            f"{args.activations} must contain 'H_member' and 'H_nonmember'"
        )
    H_mem = act["H_member"].astype(np.float32)
    H_non = act["H_nonmember"].astype(np.float32)
    log.info(f"H_member shape={H_mem.shape}, H_nonmember shape={H_non.shape}")

    X = np.concatenate([H_mem, H_non], axis=0)
    y = np.concatenate([np.ones(len(H_mem)), np.zeros(len(H_non))]).astype(int)

    linear = _evaluate_probe(
        "linear_logreg",
        lambda s: LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=s),
        X, y, args.n_folds, args.n_boot, args.seed,
    )
    mlp = _evaluate_probe(
        "mlp",
        lambda s: MLPClassifier(
            hidden_layer_sizes=tuple(args.mlp_hidden),
            alpha=args.mlp_alpha,
            max_iter=args.mlp_max_iter,
            early_stopping=True,
            random_state=s,
        ),
        X, y, args.n_folds, args.n_boot, args.seed,
    )

    delta = mlp["pooled_auroc"] - linear["pooled_auroc"]
    paired = [m - l for m, l in zip(mlp["fold_aurocs"], linear["fold_aurocs"])]

    results = {
        "activations": str(args.activations),
        "layer": args.layer,
        "n_folds": args.n_folds,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "mlp_hidden_layer_sizes": list(args.mlp_hidden),
        "mlp_alpha": args.mlp_alpha,
        "n_member": int(len(H_mem)),
        "n_nonmember": int(len(H_non)),
        "probes": {
            "linear_logreg": linear,
            "mlp": mlp,
        },
        "comparison": {
            "delta_mlp_minus_linear_pooled": delta,
            "delta_per_fold": paired,
            "delta_per_fold_mean": float(np.mean(paired)),
            "delta_per_fold_std": float(np.std(paired)),
        },
    }

    print(
        f"linear AUROC = {linear['pooled_auroc']:.4f} "
        f"CI95=[{linear['bootstrap']['auroc_ci95_lo']:.4f}, "
        f"{linear['bootstrap']['auroc_ci95_hi']:.4f}]"
    )
    print(
        f"mlp    AUROC = {mlp['pooled_auroc']:.4f} "
        f"CI95=[{mlp['bootstrap']['auroc_ci95_lo']:.4f}, "
        f"{mlp['bootstrap']['auroc_ci95_hi']:.4f}]"
    )
    print(f"delta (mlp - linear) = {delta:+.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
