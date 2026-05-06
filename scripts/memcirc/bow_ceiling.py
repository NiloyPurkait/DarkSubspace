#!/usr/bin/env python3
"""bow_ceiling.py.

Bag-of-words token-level vocabulary control. Logistic regression on
CountVectorizer features over the same 1000+1000 split, with five-fold
stratified cross-validation and AUROC pooled across out-of-fold predictions.

Used in Section Results and Appendix app:bow_baseline of the paper.
Reproduce: env/bin/python3 scripts/memcirc/bow_ceiling.py \
    --member-texts data/memcirc_ctrl_ft/member.jsonl \
    --nonmember-texts data/memcirc_ctrl_ft/nonmember.jsonl \
    --output-dir runs/memcirc/bow_ceiling/memcirc_ctrl_ft \
    --n-folds 5 --n-boot 5000 --seed 42
"""

import _bootstrap  # noqa: F401

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold


log = logging.getLogger("bow_ceiling")


def _load_jsonl_texts(path: Path) -> list[str]:
    texts = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # mirror sae_dark_subspace text-field convention
            t = obj.get("text", obj.get("content", obj.get("document", "")))
            if not isinstance(t, str):
                t = str(t)
            texts.append(t)
    return texts


def _tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_thresh: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # First index where fpr <= threshold (largest tpr among low-fpr points).
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
    tpr1 = np.empty(n_boot, dtype=np.float64)
    tpr5 = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        # Require both classes; else resample
        tries = 0
        while (yt.min() == yt.max()) and tries < 10:
            idx = rng.integers(0, n, size=n)
            yt = y_true[idx]
            ys = y_score[idx]
            tries += 1
        aurocs[b] = roc_auc_score(yt, ys) if yt.min() != yt.max() else 0.5
        tpr1[b] = _tpr_at_fpr(yt, ys, 0.01)
        tpr5[b] = _tpr_at_fpr(yt, ys, 0.05)
    return {
        "auroc_mean": float(aurocs.mean()),
        "auroc_ci95_lo": float(np.percentile(aurocs, 2.5)),
        "auroc_ci95_hi": float(np.percentile(aurocs, 97.5)),
        "tpr_at_1pct_mean": float(tpr1.mean()),
        "tpr_at_1pct_ci95_lo": float(np.percentile(tpr1, 2.5)),
        "tpr_at_1pct_ci95_hi": float(np.percentile(tpr1, 97.5)),
        "tpr_at_5pct_mean": float(tpr5.mean()),
        "tpr_at_5pct_ci95_lo": float(np.percentile(tpr5, 2.5)),
        "tpr_at_5pct_ci95_hi": float(np.percentile(tpr5, 97.5)),
    }


def _run_variant(
    variant_name: str,
    vectorizer,
    texts: list[str],
    y: np.ndarray,
    n_folds: int,
    n_boot: int,
    base_seed: int,
) -> dict:
    log.info(f"=== Variant: {variant_name} ===")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=base_seed)
    oof_scores = np.full(len(y), np.nan, dtype=np.float64)
    fold_aurocs = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(texts, y)):
        t0 = time.time()
        # Important: fit vectorizer ONLY on training fold (avoid leakage)
        # Use a fresh clone per fold so stateful learners don't carry over.
        vec = vectorizer.__class__(**vectorizer.get_params())
        X_train = vec.fit_transform([texts[i] for i in train_idx])
        X_test = vec.transform([texts[i] for i in test_idx])
        y_train = y[train_idx]
        y_test = y[test_idx]

        clf = LogisticRegression(
            C=1.0,
            solver="liblinear",
            random_state=base_seed + fold_idx,
            max_iter=1000,
        )
        clf.fit(X_train, y_train)
        scores = clf.decision_function(X_test)
        oof_scores[test_idx] = scores
        fold_auroc = roc_auc_score(y_test, scores)
        fold_aurocs.append(float(fold_auroc))
        log.info(
            f"  fold {fold_idx}: n_train={len(train_idx)} n_test={len(test_idx)} "
            f"n_features={X_train.shape[1]} AUROC={fold_auroc:.4f} "
            f"({time.time()-t0:.1f}s)"
        )

    # Pooled OOF metrics
    assert not np.isnan(oof_scores).any(), "Every sample must have OOF prediction."
    pooled_auroc = float(roc_auc_score(y, oof_scores))
    pooled_tpr1 = float(_tpr_at_fpr(y, oof_scores, 0.01))
    pooled_tpr5 = float(_tpr_at_fpr(y, oof_scores, 0.05))

    rng = np.random.default_rng(base_seed + 1000)
    boot = _bootstrap_ci(y, oof_scores, n_boot=n_boot, rng=rng)

    # Confound classification
    if pooled_auroc >= 0.90:
        level = "SEVERE"
    elif pooled_auroc >= 0.70:
        level = "MODERATE"
    elif pooled_auroc >= 0.55:
        level = "MILD"
    else:
        level = "NEGLIGIBLE"

    return {
        "variant": variant_name,
        "n_folds": n_folds,
        "fold_aurocs": fold_aurocs,
        "fold_auroc_mean": float(np.mean(fold_aurocs)),
        "fold_auroc_std": float(np.std(fold_aurocs)),
        "pooled_auroc": pooled_auroc,
        "pooled_tpr_at_1pct_fpr": pooled_tpr1,
        "pooled_tpr_at_5pct_fpr": pooled_tpr5,
        "bootstrap": {"n_boot": n_boot, **boot},
        "confound_level": level,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member-texts", type=Path, required=True)
    ap.add_argument("--nonmember-texts", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--max-features",
        type=int,
        default=50000,
        help="vocab cap for vectorizers (keep top-k by frequency).",
    )
    ap.add_argument(
        "--ngram-max",
        type=int,
        default=2,
        help="max n-gram length; 2 => unigrams+bigrams (default).",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    mem_texts = _load_jsonl_texts(args.member_texts)
    non_texts = _load_jsonl_texts(args.nonmember_texts)
    log.info(
        f"loaded {len(mem_texts)} member texts from {args.member_texts.name}, "
        f"{len(non_texts)} nonmember from {args.nonmember_texts.name}"
    )

    # Members = label 1 (positive), nonmembers = label 0
    texts = mem_texts + non_texts
    y = np.concatenate([np.ones(len(mem_texts)), np.zeros(len(non_texts))]).astype(int)

    # Balanced by construction if mem_n == non_n; verify.
    log.info(f"total N={len(texts)}, n_pos={int(y.sum())}, n_neg={int((1-y).sum())}")

    # Two BoW variants
    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, args.ngram_max),
        max_features=args.max_features,
        min_df=2,
    )
    count = CountVectorizer(
        lowercase=True,
        ngram_range=(1, args.ngram_max),
        max_features=args.max_features,
        min_df=2,
    )

    results = {
        "member_texts": str(args.member_texts),
        "nonmember_texts": str(args.nonmember_texts),
        "n_member": len(mem_texts),
        "n_nonmember": len(non_texts),
        "n_folds": args.n_folds,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "vectorizer_max_features": args.max_features,
        "vectorizer_ngram_range": [1, args.ngram_max],
        "variants": {},
    }

    results["variants"]["tfidf_lr"] = _run_variant(
        "tfidf_lr",
        tfidf,
        texts,
        y,
        n_folds=args.n_folds,
        n_boot=args.n_boot,
        base_seed=args.seed,
    )
    results["variants"]["count_lr"] = _run_variant(
        "count_lr",
        count,
        texts,
        y,
        n_folds=args.n_folds,
        n_boot=args.n_boot,
        base_seed=args.seed,
    )

    # Compact human-facing summary
    for vname, v in results["variants"].items():
        print(
            f"[{vname}] pooled AUROC={v['pooled_auroc']:.4f} "
            f"TPR@1%FPR={v['pooled_tpr_at_1pct_fpr']:.4f} "
            f"TPR@5%FPR={v['pooled_tpr_at_5pct_fpr']:.4f} "
            f"| CI95 AUROC=[{v['bootstrap']['auroc_ci95_lo']:.4f}, "
            f"{v['bootstrap']['auroc_ci95_hi']:.4f}] "
            f"| {v['confound_level']}"
        )

    # Overall headline confound level = max of variants
    severity_rank = {"NEGLIGIBLE": 0, "MILD": 1, "MODERATE": 2, "SEVERE": 3}
    headline = max(
        (v["confound_level"] for v in results["variants"].values()),
        key=lambda lvl: severity_rank[lvl],
    )
    results["headline_confound_level"] = headline
    results["severe_confound_flag"] = headline == "SEVERE"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info(f"wrote {out_path}")
    print(f"\nheadline_confound_level={headline} severe_flag={results['severe_confound_flag']}")


if __name__ == "__main__":
    main()
