#!/usr/bin/env python3
"""length_baseline.py.

Length-feature membership classifier on the controlled member/non-member
split. Tests whether character-count, word-count, or model-tokeniser-token
count separates members from non-members on its own.

Used in Appendix `app:length_baseline` of the paper. The paper reports
per-model AUROCs near 0.50 across the controlled Pythia setting, supporting
the claim that the residual-over-reconstruction ordering is not driven by a
length artefact.

Reproduce:
    .venv/bin/python scripts/dark_subspace/length_baseline.py \
        --member-texts data/memcirc_ctrl_ft/member.jsonl \
        --nonmember-texts data/memcirc_ctrl_ft/nonmember.jsonl \
        --output-dir runs/dark_subspace/length_baseline/memcirc_ctrl_ft \
        --n-folds 5 --n-boot 5000 --seed 42

Outputs a `results.json` matching the schema of the other baselines under
`results/dark_subspace/generated/`. The default features are character
length and word count; pass `--tokenizer <hf-name-or-path>` to additionally
score model-tokeniser token counts.
"""

import _bootstrap  # noqa: F401

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from sae_mia_audit.methods.baselines import (
    BlindConfig,
    score_length,
    score_word_count,
    score_token_count,
)


log = logging.getLogger("length_baseline")


def _load_jsonl_texts(path: Path) -> list[str]:
    texts = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get("text", obj.get("content", obj.get("document", "")))
            if not isinstance(t, str):
                t = str(t)
            texts.append(t)
    return texts


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


def _run_feature(
    feature_name: str,
    feature_values: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    n_boot: int,
    base_seed: int,
) -> dict:
    log.info(f"=== Feature: {feature_name} ===")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=base_seed)
    oof_scores = np.full(len(y), np.nan, dtype=np.float64)
    fold_aurocs = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(feature_values.reshape(-1, 1), y)):
        t0 = time.time()
        X_train = feature_values[train_idx].reshape(-1, 1)
        X_test = feature_values[test_idx].reshape(-1, 1)
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
            f"AUROC={fold_auroc:.4f} ({time.time()-t0:.2f}s)"
        )

    assert not np.isnan(oof_scores).any(), "Every sample must have OOF prediction."
    pooled_auroc = float(roc_auc_score(y, oof_scores))
    pooled_tpr1 = float(_tpr_at_fpr(y, oof_scores, 0.01))
    pooled_tpr5 = float(_tpr_at_fpr(y, oof_scores, 0.05))

    rng = np.random.default_rng(base_seed + 1000)
    boot = _bootstrap_ci(y, oof_scores, n_boot=n_boot, rng=rng)

    if pooled_auroc >= 0.70:
        level = "MODERATE"
    elif pooled_auroc >= 0.55:
        level = "MILD"
    else:
        level = "NEGLIGIBLE"

    return {
        "feature": feature_name,
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
    ap = argparse.ArgumentParser(
        description=(
            "Length-feature membership baseline on the controlled split: "
            "tests whether character/word/token count alone separates members "
            "from non-members."
        )
    )
    ap.add_argument("--member-texts", type=Path, required=True,
                    help="JSONL of member documents (text in field 'text').")
    ap.add_argument("--nonmember-texts", type=Path, required=True,
                    help="JSONL of non-member documents (same field convention).")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Directory to write results.json.")
    ap.add_argument("--n-folds", type=int, default=5,
                    help="Stratified K-fold split (default 5).")
    ap.add_argument("--n-boot", type=int, default=5000,
                    help="Bootstrap replicates for AUROC CI (default 5000).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for the fold split and the bootstrap.")
    ap.add_argument("--tokenizer", type=str, default=None,
                    help="Optional HF tokenizer name or path. If supplied, "
                         "also reports a token-count baseline using this "
                         "tokenizer.")
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

    texts = mem_texts + non_texts
    y = np.concatenate([np.ones(len(mem_texts)), np.zeros(len(non_texts))]).astype(int)
    log.info(f"total N={len(texts)}, n_pos={int(y.sum())}, n_neg={int((1-y).sum())}")

    cfg = BlindConfig()

    results = {
        "member_texts": str(args.member_texts),
        "nonmember_texts": str(args.nonmember_texts),
        "n_member": len(mem_texts),
        "n_nonmember": len(non_texts),
        "n_folds": args.n_folds,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "features": {},
    }

    char_counts = score_length(texts, cfg)
    results["features"]["char_count"] = _run_feature(
        "char_count", char_counts, y, args.n_folds, args.n_boot, args.seed,
    )

    word_counts = score_word_count(texts, cfg)
    results["features"]["word_count"] = _run_feature(
        "word_count", word_counts, y, args.n_folds, args.n_boot, args.seed,
    )

    if args.tokenizer is not None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        token_counts = score_token_count(texts, tok, cfg)
        results["features"]["token_count"] = _run_feature(
            "token_count", token_counts, y, args.n_folds, args.n_boot, args.seed,
        )

    severity_rank = {"NEGLIGIBLE": 0, "MILD": 1, "MODERATE": 2}
    headline = max(
        (f["confound_level"] for f in results["features"].values()),
        key=lambda lvl: severity_rank[lvl],
    )
    results["headline_confound_level"] = headline

    for fname, f in results["features"].items():
        print(
            f"[{fname}] pooled AUROC={f['pooled_auroc']:.4f} "
            f"CI95=[{f['bootstrap']['auroc_ci95_lo']:.4f}, "
            f"{f['bootstrap']['auroc_ci95_hi']:.4f}] | {f['confound_level']}"
        )
    print(f"\nheadline_confound_level={headline}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
