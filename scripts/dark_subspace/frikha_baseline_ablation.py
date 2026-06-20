#!/usr/bin/env python3
"""frikha_baseline_ablation.py.

Frikha-style feature-selection baselines on a trained SAE.

Direct empirical test of the paper's Section 4.3 inferential claim that
"methods that select SAE features by training-set correlation target the
recall channel rather than the membership signal." PrivacyScalpel (Frikha
et al. 2025) defines three feature-selection criteria for steering SAE
features. We implement all three on the existing P69 mixed-data SAE and
ablate the selected features at depths {1, 5, 50, 200}.

Three criteria (verbatim from arxiv.org/abs/2503.11232 Section 3.3):

  1. top_k_magnitude:
     Aggregate absolute SAE activations across member sequences, rank
     features by mean magnitude, take the top-k.

  2. mean_diff:
     Compute v = mean(z | member) - mean(z | non-member) in SAE latent
     space, take the top-k features by |v_j|.

  3. steering_probe:
     Train a logistic regression on z -> membership label, take the top-k
     features by |coef_j|.

For each (criterion, depth) cell:
  - score_K AUROC after ablation (membership detection on h'_ablated, the
    decoded reconstruction with the selected features zeroed)
  - score_K AUROC on the residual (h - h'_ablated)
  - "extraction proxy": score_K AUROC on the SAE-reconstruction stream
    (h'_ablated). This is the Frikha-style attack target — the recall
    channel that lives within the SAE's reconstruction. A drop here means
    the criterion successfully neutralised the recall channel.

Decision criterion:
  All three Frikha criteria should produce
    (a) extraction (recon AUROC) drop >= 0.05 AND
    (b) residual probe AUROC preserved within 0.02 of original.
  Confirms the paper's Section 4.3 inferential claim empirically.

Reproduce:
  .venv/bin/python scripts/dark_subspace/frikha_baseline_ablation.py \\
      --model-path runs/controlled_ft/run_20260306_055225/ft_epoch5/model \\
      --bcd-dir runs/dark_subspace/behavioral_channels/p69_epoch5 \\
      --sae-path runs/sae/<seed42_postfix_sae>/sae_final.pt \\
      --member-texts data/memcirc_ctrl_ft/member.jsonl \\
      --nonmember-texts data/memcirc_ctrl_ft/nonmember.jsonl \\
      --layer 16 \\
      --output-dir results/dark_subspace/generated/frikha_features \\
      --model-id p69_mixed_seed42

Cost: ~10-30 min on a single A40/L40S/H100. Analysis only (no training).
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
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

if _HAS_PROJECT_INFRA:
    log = get_logger(__name__)
else:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (parallel to scripts/dark_subspace/sae_dark_subspace.py)
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
    elif isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def bidirectional_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    a = roc_auc_score(labels, scores)
    b = roc_auc_score(labels, -scores)
    return max(a, b)


@torch.no_grad()
def collect_activations(model, tokenizer, texts, layer, seq_len, batch_size, device):
    all_acts = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Collecting activations"):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", truncation=True,
                        max_length=seq_len, padding=True).to(device)
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[layer]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        all_acts.append(pooled.cpu().float().numpy())
    return np.concatenate(all_acts, axis=0)


# ---------------------------------------------------------------------------
# The three Frikha selection criteria
# ---------------------------------------------------------------------------

def select_top_k_magnitude(latent: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Criterion 1: rank features by mean absolute activation on members."""
    member_mask = labels == 1
    member_z = latent[member_mask]
    # Frikha: aggregate activations across member sequences. Mean |z| per feature.
    scores = np.abs(member_z).mean(axis=0)
    return np.argsort(-scores)


def select_mean_diff(latent: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Criterion 2: v = mean(z|y=1) - mean(z|y=0); rank by |v_j|."""
    mu_pos = latent[labels == 1].mean(axis=0)
    mu_neg = latent[labels == 0].mean(axis=0)
    v = mu_pos - mu_neg
    return np.argsort(-np.abs(v))


def select_steering_probe(latent: np.ndarray, labels: np.ndarray, seed: int = 0) -> np.ndarray:
    """Criterion 3: logistic probe on z, rank by |coef_j|."""
    # Standardise per-feature for stable probe (probe magnitudes interpreted as importance)
    std = latent.std(axis=0)
    std_safe = np.where(std > 1e-8, std, 1.0)
    z_std = latent / std_safe[np.newaxis, :]
    # Use small L2 regularisation; primary purpose is to extract a stable
    # ranking, not a calibrated probability. liblinear handles d_sae >> N.
    clf = LogisticRegression(
        penalty="l2", C=1.0, solver="liblinear",
        max_iter=2000, random_state=int(seed),
    )
    clf.fit(z_std, labels)
    coefs = clf.coef_.ravel()
    return np.argsort(-np.abs(coefs))


CRITERIA = {
    "top_k_magnitude": select_top_k_magnitude,
    "mean_diff": select_mean_diff,
    "steering_probe": select_steering_probe,
}


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

def encode_decode(activations: np.ndarray, sae, device: str, batch_sz: int = 256
                  ) -> Tuple[np.ndarray, np.ndarray]:
    h_tensor = torch.tensor(activations, dtype=torch.float32, device=device)
    all_z = []
    all_recon = []
    for i in range(0, len(h_tensor), batch_sz):
        batch = h_tensor[i:i + batch_sz]
        z = sae.encode(batch)
        h_hat = sae.decode(z)
        all_z.append(z.detach().cpu().float().numpy())
        all_recon.append(h_hat.detach().cpu().float().numpy())
    return np.concatenate(all_z, axis=0), np.concatenate(all_recon, axis=0)


def decode_ablated(latent: np.ndarray, ablated_idx: np.ndarray, sae, device: str,
                   batch_sz: int = 256) -> np.ndarray:
    z = latent.copy()
    z[:, ablated_idx] = 0.0
    z_t = torch.tensor(z, dtype=torch.float32, device=device)
    out = []
    for i in range(0, len(z_t), batch_sz):
        out.append(sae.decode(z_t[i:i + batch_sz]).detach().cpu().float().numpy())
    return np.concatenate(out, axis=0)


def score_K_auroc(stream_centered: np.ndarray, d_K_norm: np.ndarray,
                  labels: np.ndarray) -> float:
    s = stream_centered @ d_K_norm
    return bidirectional_auroc(labels, s)


def run_frikha_experiment(
    activations: np.ndarray,
    labels: np.ndarray,
    d_K: np.ndarray,
    global_mean: Optional[np.ndarray],
    sae,
    device: str,
    depths: List[int],
    seed: int,
) -> Dict:
    n = len(activations)
    d_K_norm = d_K / (np.linalg.norm(d_K) + 1e-12)

    # Center activations using the same convention as sae_dark_subspace.py
    if global_mean is not None:
        centered = activations - global_mean[np.newaxis, :]
    else:
        centered = activations - activations.mean(axis=0, keepdims=True)

    # Original score_K AUROC
    auroc_original = float(bidirectional_auroc(labels, centered @ d_K_norm))

    # Encode all texts once
    latent, recon = encode_decode(activations, sae, device)
    if global_mean is not None:
        recon_centered = recon - global_mean[np.newaxis, :]
    else:
        recon_centered = recon - recon.mean(axis=0, keepdims=True)
    residual_centered = centered - recon_centered

    # Baseline (no ablation): recon-stream score_K and residual score_K
    auroc_recon_baseline = float(bidirectional_auroc(labels, recon_centered @ d_K_norm))
    auroc_residual_baseline = float(bidirectional_auroc(labels, residual_centered @ d_K_norm))

    # Logistic probe on the FULL SAE latent (the residual probe AUROC is the
    # decision criterion's "residual probe" — it asks: how well can we still
    # predict membership from the SAE features after ablation?). For
    # baseline, compute on full latent before any zeroing.
    # We instead use a held-out split: 5-fold CV mean for stability.
    from sklearn.model_selection import StratifiedKFold

    def _probe_auroc_on_latent(z_arr: np.ndarray) -> float:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        aucs = []
        std = z_arr.std(axis=0)
        std_safe = np.where(std > 1e-8, std, 1.0)
        z_std = z_arr / std_safe[np.newaxis, :]
        for tr, te in skf.split(z_std, labels):
            clf = LogisticRegression(
                penalty="l2", C=1.0, solver="liblinear",
                max_iter=2000, random_state=seed,
            )
            clf.fit(z_std[tr], labels[tr])
            p = clf.predict_proba(z_std[te])[:, 1]
            aucs.append(roc_auc_score(labels[te], p))
        return float(np.mean(aucs))

    log.info("Computing baseline 5-fold latent probe AUROC ...")
    baseline_latent_probe_auroc = _probe_auroc_on_latent(latent)
    log.info(f"baseline_latent_probe_auroc = {baseline_latent_probe_auroc:.4f}")

    # SAE feature stats
    mean_active = float(np.mean(np.sum(latent > 0, axis=1)))
    total_features = int(latent.shape[1])

    results: Dict = {
        "baseline_no_ablation": {
            "score_K_auroc_original": auroc_original,
            "score_K_auroc_recon": auroc_recon_baseline,
            "score_K_auroc_residual": auroc_residual_baseline,
            "latent_probe_auroc_5fold": baseline_latent_probe_auroc,
        },
        "sae_stats": {
            "mean_active_features": mean_active,
            "total_features": total_features,
            "sparsity": mean_active / total_features if total_features else 0.0,
        },
        "criteria": {},
    }

    for crit_name, crit_fn in CRITERIA.items():
        log.info(f"\n=== Criterion: {crit_name} ===")
        if crit_name == "steering_probe":
            ranked = crit_fn(latent, labels, seed=seed)
        else:
            ranked = crit_fn(latent, labels)

        crit_results: Dict = {}
        for k in depths:
            ablated_idx = ranked[:k]
            h_ablated = decode_ablated(latent, ablated_idx, sae, device)
            if global_mean is not None:
                h_abl_centered = h_ablated - global_mean[np.newaxis, :]
            else:
                h_abl_centered = h_ablated - h_ablated.mean(axis=0, keepdims=True)
            res_centered = centered - h_abl_centered

            auroc_recon_post = float(bidirectional_auroc(labels, h_abl_centered @ d_K_norm))
            auroc_residual_post = float(bidirectional_auroc(labels, res_centered @ d_K_norm))

            # Residual probe AUROC: 5-fold CV on the latent z with the
            # selected features zeroed (the "what's left in the SAE after
            # ablation" probe). This is the decision criterion's
            # "residual probe AUROC".
            z_post = latent.copy()
            z_post[:, ablated_idx] = 0.0
            residual_probe_auroc = _probe_auroc_on_latent(z_post)

            crit_results[f"top_{k}"] = {
                "k": int(k),
                "ablated_feature_indices": [int(x) for x in ablated_idx],
                "score_K_auroc_recon_post_ablation": auroc_recon_post,
                "score_K_auroc_residual_post_ablation": auroc_residual_post,
                "residual_probe_auroc_5fold": residual_probe_auroc,
                # Decision-criterion deltas
                "extraction_drop_vs_recon_baseline": float(
                    auroc_recon_baseline - auroc_recon_post
                ),
                "residual_probe_delta_vs_baseline": float(
                    residual_probe_auroc - baseline_latent_probe_auroc
                ),
            }
            log.info(
                f"  k={k}: recon_AUROC={auroc_recon_post:.4f} "
                f"(drop {auroc_recon_baseline - auroc_recon_post:+.4f}), "
                f"residual_AUROC={auroc_residual_post:.4f}, "
                f"residual_probe={residual_probe_auroc:.4f} "
                f"(delta {residual_probe_auroc - baseline_latent_probe_auroc:+.4f})"
            )
        results["criteria"][crit_name] = crit_results

    # Decision-criterion verdict: feature selection reduces verbatim extraction
    # but leaves the residual membership signal intact.
    verdict: Dict = {}
    for crit_name, crit_results in results["criteria"].items():
        # Use deepest depth (top_200) for the verdict; report all depths above.
        deepest = crit_results[f"top_{depths[-1]}"]
        ext_drop = deepest["extraction_drop_vs_recon_baseline"]
        res_probe_delta = deepest["residual_probe_delta_vs_baseline"]
        # PLAN: extraction drop >= 0.05 AND |residual probe delta| <= 0.02
        passes_extraction = ext_drop >= 0.05
        preserves_residual = abs(res_probe_delta) <= 0.02
        verdict[crit_name] = {
            "depth_used_for_verdict": int(depths[-1]),
            "extraction_drop": float(ext_drop),
            "extraction_drop_threshold": 0.05,
            "passes_extraction_drop": bool(passes_extraction),
            "residual_probe_delta": float(res_probe_delta),
            "residual_probe_threshold_abs": 0.02,
            "preserves_residual_probe": bool(preserves_residual),
            "passes_full_criterion": bool(passes_extraction and preserves_residual),
        }
    results["decision_criterion_verdict"] = verdict

    return results


def main():
    p = argparse.ArgumentParser(description="Frikha-style feature-selection ablation baselines.")
    p.add_argument("--model-path", required=True)
    p.add_argument("--bcd-dir", required=True)
    p.add_argument("--sae-path", required=True)
    p.add_argument("--member-texts", required=True)
    p.add_argument("--nonmember-texts", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-texts", type=int, default=0)
    p.add_argument("--depths", nargs="+", type=int, default=[1, 5, 50, 200])
    args = p.parse_args()

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
    config["script"] = "frikha_baseline_ablation.py"
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
    log.info(f"Loaded {len(member_texts)} member, {len(nonmember_texts)} nonmember texts")

    all_texts = member_texts + nonmember_texts
    labels = np.array([1] * len(member_texts) + [0] * len(nonmember_texts))

    activations = collect_activations(
        model, tokenizer, all_texts, args.layer,
        args.seq_len, args.batch_size, args.device,
    )
    log.info(f"Activations: shape={activations.shape}")

    del model
    torch.cuda.empty_cache()

    results = run_frikha_experiment(
        activations, labels, d_K, global_mean, sae, args.device,
        depths=list(args.depths), seed=int(args.seed),
    )

    output = {
        "experiment": "frikha_feature_selection",
        "experiment_name": "frikha_baseline_p69",
        "purpose": (
            "Direct empirical test of the paper Section 4.3 "
            "claim that SAE feature-selection-by-correlation methods target "
            "the recall channel rather than the membership signal."
        ),
        "model": args.model_id,
        "layer": args.layer,
        "sae_path": args.sae_path,
        "ft_model_path": args.model_path,
        "n_member": len(member_texts),
        "n_nonmember": len(nonmember_texts),
        "depths": list(args.depths),
        **results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

    (out_dir / "results.json").write_text(
        json.dumps(_sanitize_for_json(output), indent=2)
    )
    log.info(f"Results saved to {out_dir / 'results.json'}")

    # Print summary
    print("\n" + "=" * 76)
    print("Frikha-style baseline ablation summary")
    print("=" * 76)
    base = results["baseline_no_ablation"]
    print(f"  Original score_K AUROC:           {base['score_K_auroc_original']:.4f}")
    print(f"  Recon score_K AUROC (no ablate):  {base['score_K_auroc_recon']:.4f}")
    print(f"  Residual score_K AUROC:           {base['score_K_auroc_residual']:.4f}")
    print(f"  Latent probe AUROC (5-fold):      {base['latent_probe_auroc_5fold']:.4f}")
    print()
    for crit_name, crit_results in results["criteria"].items():
        print(f"  Criterion: {crit_name}")
        for k_key, cell in crit_results.items():
            print(
                f"    {k_key:>8} | recon={cell['score_K_auroc_recon_post_ablation']:.4f} "
                f"(drop {cell['extraction_drop_vs_recon_baseline']:+.4f}) | "
                f"residual={cell['score_K_auroc_residual_post_ablation']:.4f} | "
                f"probe={cell['residual_probe_auroc_5fold']:.4f} "
                f"(delta {cell['residual_probe_delta_vs_baseline']:+.4f})"
            )
    print()
    print("  Decision-criterion verdict (deepest depth):")
    for crit_name, v in results["decision_criterion_verdict"].items():
        print(
            f"    {crit_name}: extraction_drop {v['extraction_drop']:+.4f} "
            f"(>= 0.05: {v['passes_extraction_drop']}); "
            f"residual_probe_delta {v['residual_probe_delta']:+.4f} "
            f"(|.| <= 0.02: {v['preserves_residual_probe']}); "
            f"PASSES = {v['passes_full_criterion']}"
        )


if __name__ == "__main__":
    main()
