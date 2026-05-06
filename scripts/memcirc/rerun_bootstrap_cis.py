#!/usr/bin/env python3
"""rerun_bootstrap_cis.py.

Re-runs paired bootstrap CIs at n_boot=10000 on the paraphrase, standard
probes, BoW, and dark-subspace JSONs and writes a *.pre_n10k_backup audit
trail.

Used in the n_boot=10000 standardisation paragraph and Methods disclosure
of the paper. Reproduce:
    env/bin/python3 scripts/memcirc/rerun_bootstrap_cis.py --target all

CPU only. No GPU, no model re-loading, no SLURM jobs.

Targets.
    standard_probes  reruns AUROC, TPR@1%FPR, TPR@5%FPR bootstrap CIs for
                     all 7 methods of baseline_attacks_suite at n_boot=10000
                     (was 1000) for the four orthogonal-complement gate
                     models (p69, p12b, neo, qwen2). Updates each
                     runs/memcirc/baseline_attacks/<model>/results.json in
                     place. Includes paired bootstraps.
    paraphrase       reruns the delta_paraphrased_minus_original AUROC and
                     TPR1 paired bootstrap CIs at n_boot=10000 (was 1000)
                     for three paper-cited models (p69, qwen2, p12b). Reads
                     cached per_text_scores.json (scores_orig_vs_nonmember
                     and scores_para_vs_nonmember). Updates
                     runs/memcirc/paraphrase_sensitivity/<m>/results.json
                     in place.
    bow              reruns the BoW-ceiling bootstrap (per variant) at
                     n_boot=10000 (was 5000). Requires re-fitting the
                     TF-IDF / Count + LogReg pipeline to recover pooled OOF
                     scores, which are not cached. CPU only, takes about
                     30s. Updates
                     runs/memcirc/bow_ceiling/memcirc_ctrl_ft/results.json
                     in place.
    all              all three above, in order.

Backup policy.
Before overwriting any results.json, the script copies it to
results.json.pre_n10k_backup if no such backup yet exists.
"""

import _bootstrap  # noqa: F401

import argparse
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold


log = logging.getLogger("rerun_bootstrap_cis")

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = REPO_ROOT / "runs" / "memcirc" / "baseline_attacks"
PARAPHRASE_ROOT = REPO_ROOT / "runs" / "memcirc" / "paraphrase_sensitivity"
BOW_ROOT = REPO_ROOT / "runs" / "memcirc" / "bow_ceiling" / "memcirc_ctrl_ft"

GATE_MODELS = ("p69", "p12b", "neo", "qwen2")
PARA_MODELS = ("p69", "qwen2", "p12b")

N_BOOT_NEW = 10000
SEED_NEW = 12345


# --- shared metric helpers ---

def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = fpr <= target_fpr
    if not np.any(mask):
        return 0.0
    return float(np.max(tpr[mask]))


def _safe_auroc(y, x) -> float:
    if y.min() == y.max():
        return 0.5
    return float(roc_auc_score(y, x))


def auroc_bi(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, int]:
    finite = np.isfinite(y_score)
    if not finite.all():
        med = float(np.median(y_score[finite])) if finite.any() else 0.0
        y_score = np.where(finite, y_score, med)
    a_pos = float(roc_auc_score(y_true, y_score))
    a_neg = float(roc_auc_score(y_true, -y_score))
    if a_neg > a_pos:
        return a_neg, -1
    return a_pos, +1


def bootstrap_ci(
    y_true: np.ndarray, y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int, seed: int,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            vals.append(metric_fn(y_true[idx], y_score[idx]))
        except Exception:
            continue
    if not vals:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(vals)
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def paired_bootstrap_ci(
    y_true: np.ndarray, s_a: np.ndarray, s_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int, seed: int,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            va = metric_fn(y_true[idx], s_a[idx])
            vb = metric_fn(y_true[idx], s_b[idx])
            diffs.append(va - vb)
        except Exception:
            continue
    if not diffs:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(diffs)
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def _backup_once(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".pre_n10k_backup")
    if not bak.exists():
        shutil.copy2(path, bak)
        log.info(f"  backup -> {bak.name}")


# --- target: standard_probes ---

METHODS = ("loss", "zlib_ratio", "minkprob_20", "minkprob_10", "minkpp_20",
           "original_d_K", "residual_d_K")


def rerun_standard_probes(model: str) -> dict:
    src = BASELINE_ROOT / model / "per_text_scores.json"
    res_path = BASELINE_ROOT / model / "results.json"
    log.info(f"[{model}] standard_probes: reading {src}")
    pts = json.loads(src.read_text())

    labels = np.array(pts["labels"], dtype=int)
    results = json.loads(res_path.read_text())
    methods = results.setdefault("methods", {})
    paired = results.setdefault("paired_bootstrap", {})
    deltas = []

    # Precompute per-method oriented arrays (stable across paired comps below)
    oriented = {}
    for name in METHODS:
        s = np.array(pts[name], dtype=np.float64)
        finite = np.isfinite(s)
        if not finite.all():
            med = float(np.median(s[finite])) if finite.any() else 0.0
            s = np.where(finite, s, med)
        _, sign = auroc_bi(labels, s)
        oriented[name] = (sign * s, sign)

    for name in METHODS:
        s_oriented, sign = oriented[name]
        old = methods.get(name, {}).copy()

        auroc_mean, auroc_lo, auroc_hi = bootstrap_ci(
            labels, s_oriented, _safe_auroc,
            n_boot=N_BOOT_NEW, seed=SEED_NEW,
        )
        tpr1_mean, tpr1_lo, tpr1_hi = bootstrap_ci(
            labels, s_oriented, lambda y, x: tpr_at_fpr(y, x, 0.01),
            n_boot=N_BOOT_NEW, seed=SEED_NEW,
        )
        tpr5_mean, tpr5_lo, tpr5_hi = bootstrap_ci(
            labels, s_oriented, lambda y, x: tpr_at_fpr(y, x, 0.05),
            n_boot=N_BOOT_NEW, seed=SEED_NEW,
        )

        methods.setdefault(name, {})
        methods[name].update({
            "auroc_boot_mean": auroc_mean,
            "auroc_ci95_lo": auroc_lo,
            "auroc_ci95_hi": auroc_hi,
            "tpr1_boot_mean": tpr1_mean,
            "tpr1_ci95_lo": tpr1_lo,
            "tpr1_ci95_hi": tpr1_hi,
            "tpr5_boot_mean": tpr5_mean,
            "tpr5_ci95_lo": tpr5_lo,
            "tpr5_ci95_hi": tpr5_hi,
        })

        deltas.append({
            "method": name,
            "auroc_ci95_lo_old": old.get("auroc_ci95_lo"),
            "auroc_ci95_hi_old": old.get("auroc_ci95_hi"),
            "auroc_ci95_lo_new": auroc_lo,
            "auroc_ci95_hi_new": auroc_hi,
        })

    # Paired bootstraps (residual_d_K vs each)
    ref_s, _ = oriented["residual_d_K"]
    for name in ("loss", "zlib_ratio", "minkprob_20", "minkprob_10", "minkpp_20", "original_d_K"):
        cmp_s, _ = oriented[name]
        d_mean, d_lo, d_hi = paired_bootstrap_ci(
            labels, ref_s, cmp_s, lambda y, x: tpr_at_fpr(y, x, 0.01),
            n_boot=N_BOOT_NEW, seed=SEED_NEW,
        )
        paired[f"tpr1_residual_minus_{name}"] = {"mean": d_mean, "ci95_lo": d_lo, "ci95_hi": d_hi}
        d_mean, d_lo, d_hi = paired_bootstrap_ci(
            labels, ref_s, cmp_s, _safe_auroc,
            n_boot=N_BOOT_NEW, seed=SEED_NEW,
        )
        paired[f"auroc_residual_minus_{name}"] = {"mean": d_mean, "ci95_lo": d_lo, "ci95_hi": d_hi}

    # Update provenance
    results.setdefault("bootstrap_history", []).append({
        "n_boot": N_BOOT_NEW,
        "seed": SEED_NEW,
        "rerun_script": "scripts/memcirc/rerun_bootstrap_cis.py",
        "rerun_target": "standard_probes",
        "rerun_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    })
    results["n_boot_canonical"] = N_BOOT_NEW
    results["bootstrap_seed_canonical"] = SEED_NEW

    _backup_once(res_path)
    res_path.write_text(json.dumps(results, indent=2))
    log.info(f"[{model}] standard_probes: wrote {res_path}")
    return {"model": model, "deltas": deltas}


# --- target: paraphrase ---

def rerun_paraphrase(model: str) -> dict:
    src = PARAPHRASE_ROOT / model / "per_text_scores.json"
    res_path = PARAPHRASE_ROOT / model / "results.json"
    log.info(f"[{model}] paraphrase: reading {src}")
    pts = json.loads(src.read_text())
    results = json.loads(res_path.read_text())

    y_orig = np.array(pts["labels_orig"], dtype=int)
    s_orig = np.array(pts["scores_orig_vs_nonmember"], dtype=np.float64)
    s_para = np.array(pts["scores_para_vs_nonmember"], dtype=np.float64)

    sign_o = int(results["original"]["orientation_sign"])

    old = results.get("delta_paraphrased_minus_original", {}).copy()

    d_auroc, d_lo, d_hi = paired_bootstrap_ci(
        y_orig, s_orig, s_para, _safe_auroc,
        n_boot=N_BOOT_NEW, seed=SEED_NEW,
    )
    d_tpr1, d_tpr1_lo, d_tpr1_hi = paired_bootstrap_ci(
        y_orig, sign_o * s_orig, sign_o * s_para,
        lambda y, x: tpr_at_fpr(y, x, 0.01),
        n_boot=N_BOOT_NEW, seed=SEED_NEW,
    )

    results["delta_paraphrased_minus_original"] = {
        "auroc_mean": d_auroc, "auroc_ci95_lo": d_lo, "auroc_ci95_hi": d_hi,
        "tpr1_mean": d_tpr1, "tpr1_ci95_lo": d_tpr1_lo, "tpr1_ci95_hi": d_tpr1_hi,
    }
    results.setdefault("bootstrap_history", []).append({
        "n_boot": N_BOOT_NEW,
        "seed": SEED_NEW,
        "rerun_script": "scripts/memcirc/rerun_bootstrap_cis.py",
        "rerun_target": "paraphrase",
        "rerun_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    })
    results["n_boot_canonical"] = N_BOOT_NEW
    results["bootstrap_seed_canonical"] = SEED_NEW

    _backup_once(res_path)
    res_path.write_text(json.dumps(results, indent=2))
    log.info(
        f"[{model}] paraphrase: delta auroc {old.get('auroc_mean'):.6f} "
        f"[{old.get('auroc_ci95_lo'):.6f}, {old.get('auroc_ci95_hi'):.6f}] "
        f"-> {d_auroc:.6f} [{d_lo:.6f}, {d_hi:.6f}]"
    )
    return {
        "model": model,
        "delta_auroc_lo_old": old.get("auroc_ci95_lo"),
        "delta_auroc_hi_old": old.get("auroc_ci95_hi"),
        "delta_auroc_lo_new": d_lo,
        "delta_auroc_hi_new": d_hi,
        "delta_auroc_mean_old": old.get("auroc_mean"),
        "delta_auroc_mean_new": d_auroc,
        "delta_tpr1_mean_old": old.get("tpr1_mean"),
        "delta_tpr1_lo_old": old.get("tpr1_ci95_lo"),
        "delta_tpr1_hi_old": old.get("tpr1_ci95_hi"),
        "delta_tpr1_mean_new": d_tpr1,
        "delta_tpr1_lo_new": d_tpr1_lo,
        "delta_tpr1_hi_new": d_tpr1_hi,
    }


# --- target: bow ---

def _bow_load_texts(path: Path) -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get("text", obj.get("content", obj.get("document", "")))
            if not isinstance(t, str):
                t = str(t)
            out.append(t)
    return out


def _bow_oof_scores(
    vectorizer_template,
    texts: List[str],
    y: np.ndarray,
    n_folds: int,
    base_seed: int,
) -> Tuple[np.ndarray, List[float]]:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=base_seed)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    fold_aurocs = []
    for fold_idx, (tr, te) in enumerate(skf.split(texts, y)):
        vec = vectorizer_template.__class__(**vectorizer_template.get_params())
        X_tr = vec.fit_transform([texts[i] for i in tr])
        X_te = vec.transform([texts[i] for i in te])
        clf = LogisticRegression(C=1.0, solver="liblinear",
                                 random_state=base_seed + fold_idx, max_iter=1000)
        clf.fit(X_tr, y[tr])
        s = clf.decision_function(X_te)
        oof[te] = s
        fold_aurocs.append(float(roc_auc_score(y[te], s)))
    assert not np.isnan(oof).any()
    return oof, fold_aurocs


def _bow_boot_block(y: np.ndarray, oof: np.ndarray, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    aurocs = np.empty(n_boot, dtype=np.float64)
    tpr1 = np.empty(n_boot, dtype=np.float64)
    tpr5 = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y[idx]; ys = oof[idx]
        tries = 0
        while (yt.min() == yt.max()) and tries < 10:
            idx = rng.integers(0, n, size=n)
            yt = y[idx]; ys = oof[idx]
            tries += 1
        aurocs[b] = roc_auc_score(yt, ys) if yt.min() != yt.max() else 0.5
        tpr1[b] = tpr_at_fpr(yt, ys, 0.01)
        tpr5[b] = tpr_at_fpr(yt, ys, 0.05)
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


def rerun_bow() -> dict:
    res_path = BOW_ROOT / "results.json"
    results = json.loads(res_path.read_text())
    member_path = REPO_ROOT / results["member_texts"]
    nonmember_path = REPO_ROOT / results["nonmember_texts"]
    n_folds = int(results["n_folds"])
    base_seed = int(results["seed"])
    max_features = int(results.get("vectorizer_max_features", 50000))
    ngram_max = int(results.get("vectorizer_ngram_range", [1, 2])[1])

    log.info(f"[bow] loading {member_path}, {nonmember_path}")
    mem = _bow_load_texts(member_path)
    non = _bow_load_texts(nonmember_path)
    texts = mem + non
    y = np.concatenate([np.ones(len(mem)), np.zeros(len(non))]).astype(int)

    deltas = []
    for vname, vec in [
        ("tfidf_lr", TfidfVectorizer(lowercase=True, ngram_range=(1, ngram_max),
                                     max_features=max_features, min_df=2)),
        ("count_lr", CountVectorizer(lowercase=True, ngram_range=(1, ngram_max),
                                     max_features=max_features, min_df=2)),
    ]:
        log.info(f"[bow] {vname}: refit OOF")
        oof, fold_aurocs = _bow_oof_scores(vec, texts, y, n_folds, base_seed)

        # Bootstrap with the standardized n_boot and seed (12345). The original
        # in-place value used base_seed+1000.
        log.info(f"[bow] {vname}: bootstrap n={N_BOOT_NEW} seed={SEED_NEW}")
        boot = _bow_boot_block(y, oof, n_boot=N_BOOT_NEW, seed=SEED_NEW)

        var = results["variants"][vname]
        old_boot = var.get("bootstrap", {}).copy()

        var["bootstrap"] = {"n_boot": N_BOOT_NEW, **boot}
        # also re-record the deterministic point auroc/tpr (these are
        # deterministic given seed; should be identical to prior run)
        var["pooled_auroc"] = float(roc_auc_score(y, oof))
        var["pooled_tpr_at_1pct_fpr"] = float(tpr_at_fpr(y, oof, 0.01))
        var["pooled_tpr_at_5pct_fpr"] = float(tpr_at_fpr(y, oof, 0.05))
        var["fold_aurocs"] = fold_aurocs
        var["fold_auroc_mean"] = float(np.mean(fold_aurocs))
        var["fold_auroc_std"] = float(np.std(fold_aurocs))

        deltas.append({
            "variant": vname,
            "auroc_lo_old": old_boot.get("auroc_ci95_lo"),
            "auroc_hi_old": old_boot.get("auroc_ci95_hi"),
            "auroc_lo_new": boot["auroc_ci95_lo"],
            "auroc_hi_new": boot["auroc_ci95_hi"],
        })

    results["n_boot"] = N_BOOT_NEW
    results.setdefault("bootstrap_history", []).append({
        "n_boot": N_BOOT_NEW,
        "seed": SEED_NEW,
        "rerun_script": "scripts/memcirc/rerun_bootstrap_cis.py",
        "rerun_target": "bow",
        "rerun_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "note": "bootstrap rerun used seed=12345; OOF score generation seed=42 (unchanged)",
    })
    results["n_boot_canonical"] = N_BOOT_NEW
    results["bootstrap_seed_canonical"] = SEED_NEW

    _backup_once(res_path)
    res_path.write_text(json.dumps(results, indent=2))
    log.info(f"[bow] wrote {res_path}")
    return {"deltas": deltas}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=("standard_probes", "paraphrase", "bow", "all"),
                    required=True)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    summary = {"standard_probes": [], "paraphrase": [], "bow": None}

    if args.target in ("standard_probes", "all"):
        for m in GATE_MODELS:
            summary["standard_probes"].append(rerun_standard_probes(m))
    if args.target in ("paraphrase", "all"):
        for m in PARA_MODELS:
            summary["paraphrase"].append(rerun_paraphrase(m))
    if args.target in ("bow", "all"):
        summary["bow"] = rerun_bow()

    print("\n--- summary ---")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
