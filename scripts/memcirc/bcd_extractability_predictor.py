#!/usr/bin/env python3
"""bcd_extractability_predictor.py.

Computes the recall-extractability predictor (loss-vs-ROUGE-L) per model
and writes `runs/memcirc/bcd_extractability/<model>_epoch5/extractability_predictor.json`.

Used in Appendix `app:bcd_details` (A:49-53), with reviewer concern C3 of the paper.
Reproduce: env/bin/python3 scripts/memcirc/bcd_extractability_predictor.py --model-path <ft_model> --bcd-dir <bcd_dir> --member-texts <member.jsonl> --layer <L> --output-dir <out>

For each member text this script computes:
  - Per-text loss (cross-entropy under the fine-tuned model)
  - Per-text ROUGE-L (greedy generation from a prefix)
  - BCD projection scores. score_K, score_R, score_SK, score_SR

Then measures:
  - Spearman correlations of each score vs ROUGE-L
  - Partial Spearman correlation of score_SR vs ROUGE-L controlling for loss

This answers whether the BCD "recall subspace" captures extractability
information beyond what per-text loss already provides.
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
from scipy.stats import spearmanr, rankdata
from numpy.polynomial.polynomial import polyfit, polyval
from tqdm.auto import tqdm

from sae_mia_audit.models.wrapper import load_model_and_tokenizer, CausalLMWrapper
from sae_mia_audit.data.tokenizer import TokenizeConfig, tokenize_batch
from sae_mia_audit.utils.seed import SeedConfig, set_global_seed
from sae_mia_audit.utils.logging import setup_logging, get_logger
from sae_mia_audit.utils.hf import HFModelSpec

# Inter-script imports (via _bootstrap sys.path)
from validate_recall_proxy import compute_per_text_loss, compute_per_text_rouge_l

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_texts(path: str, max_n: Optional[int] = None) -> List[str]:
    """Load texts from JSONL file (one JSON object per line, field='text')."""
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            texts.append(json.loads(line)["text"])
            if max_n is not None and max_n > 0 and len(texts) >= max_n:
                break
    return texts


def _batched(items, n):
    """Yield successive n-sized chunks from items."""
    for i in range(0, len(items), n):
        yield items[i : i + n]


# ---------------------------------------------------------------------------
# Activation extraction (layer-L mean-pooled hidden states)
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_activations(
    wrapper: CausalLMWrapper,
    texts: List[str],
    layer: int,
    seq_len: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Collect per-text mean-pooled residual activations at a given layer.

    Uses output_hidden_states=True and masks padding via attention_mask.

    Args:
        wrapper: CausalLMWrapper instance.
        texts: List of text strings.
        layer: Layer index for activation extraction.
        seq_len: Maximum sequence length for tokenization.
        batch_size: Processing batch size.
        device: Device string.

    Returns:
        activations: (n_texts, d_model) mean-pooled hidden states at `layer`.
    """
    tok_cfg = TokenizeConfig(seq_len=seq_len, random_crop=False)
    all_acts = []

    for chunk in tqdm(
        list(_batched(texts, batch_size)),
        desc=f"Collecting layer-{layer} activations",
        dynamic_ncols=True,
    ):
        batch = tokenize_batch(wrapper.tokenizer, chunk, tok_cfg)
        input_ids = batch["input_ids"].to(device)
        attn = batch.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(device)

        out = wrapper.forward(
            input_ids=input_ids,
            attention_mask=attn,
            output_hidden_states=True,
        )

        # Mean-pooled activations at the target layer
        h = out.hidden_states[layer]  # (B, T, d_model)
        if attn is not None:
            mask = attn.unsqueeze(-1).float()
            h_mean = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            h_mean = h.mean(dim=1)
        all_acts.append(h_mean.cpu().float().numpy())

    activations = np.concatenate(all_acts, axis=0)
    return activations


# ---------------------------------------------------------------------------
# BCD projection scores
# ---------------------------------------------------------------------------

def compute_bcd_scores(
    activations: np.ndarray,
    d_K: np.ndarray,
    d_R: np.ndarray,
    S_K_basis: Optional[np.ndarray],
    S_R_basis: Optional[np.ndarray],
    global_mean: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute BCD-based projection scores from activations and directions.

    Args:
        activations: (n_texts, d_model) mean-pooled hidden states.
        d_K: Knowledge direction vector, shape (d_model,).
        d_R: Recall direction vector, shape (d_model,).
        S_K_basis: Knowledge subspace basis, shape (d_model, k) or None.
        S_R_basis: Recall subspace basis, shape (d_model, k) or None.
        global_mean: Global mean activation for centering, shape (d_model,).

    Returns:
        Dict of score arrays, each shape (n_texts,):
            score_K:  |dot(centered_act, d_K_hat)|
            score_R:  |dot(centered_act, d_R_hat)|
            score_SK: ||proj_{S_K}(centered_act)||
            score_SR: ||proj_{S_R}(centered_act)||
    """
    centered = activations - global_mean[np.newaxis, :]

    # Normalize direction vectors
    d_K_norm = d_K / (np.linalg.norm(d_K) + 1e-12)
    d_R_norm = d_R / (np.linalg.norm(d_R) + 1e-12)

    # Projection magnitudes onto 1-D directions
    score_K = np.abs(centered @ d_K_norm)   # (n_texts,)
    score_R = np.abs(centered @ d_R_norm)   # (n_texts,)

    scores: Dict[str, np.ndarray] = {
        "score_K": score_K,
        "score_R": score_R,
    }

    # Subspace projection norms
    if S_K_basis is not None:
        proj_K = centered @ S_K_basis        # (n_texts, k)
        scores["score_SK"] = np.sqrt(np.sum(proj_K ** 2, axis=1))  # L2 norm
    if S_R_basis is not None:
        proj_R = centered @ S_R_basis        # (n_texts, k)
        scores["score_SR"] = np.sqrt(np.sum(proj_R ** 2, axis=1))  # L2 norm

    return scores


# ---------------------------------------------------------------------------
# Partial Spearman correlation
# ---------------------------------------------------------------------------

def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """Spearman correlation between x and y, controlling for z.

    Procedure:
      1. Rank all three variables.
      2. Regress rank(x) and rank(y) on rank(z) via linear fit.
      3. Compute Spearman rho on the residuals.

    Args:
        x: (n,) first variable.
        y: (n,) second variable.
        z: (n,) control variable.

    Returns:
        (rho, p) -- Spearman correlation and two-sided p-value.
    """
    rx = rankdata(x)
    ry = rankdata(y)
    rz = rankdata(z)

    # Residualize ranks via linear regression on rz
    coef_x = polyfit(rz, rx, 1)
    coef_y = polyfit(rz, ry, 1)
    res_x = rx - polyval(rz, coef_x)
    res_y = ry - polyval(rz, coef_y)

    return spearmanr(res_x, res_y)


# ---------------------------------------------------------------------------
# JSON sanitization
# ---------------------------------------------------------------------------

def _sanitize_for_json(obj):
    """Recursively replace NaN/Inf float values with None for JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if not np.isfinite(obj):
            return None
        return obj
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "BCD Extractability Predictor: test whether BCD projection "
            "magnitudes predict per-document extractability (ROUGE-L) "
            "better than loss alone."
        )
    )
    parser.add_argument(
        "--model-path", required=True,
        help="Path to fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--bcd-dir", required=True,
        help="Path to BCD output directory (contains directions.npz)",
    )
    parser.add_argument(
        "--member-texts", required=True,
        help="Path to member JSONL file",
    )
    parser.add_argument(
        "--layer", type=int, default=16,
        help="Layer index for activation extraction (default: 16)",
    )
    parser.add_argument(
        "--n-texts", type=int, default=0,
        help="Number of texts to use; 0 = all (default: 0)",
    )
    parser.add_argument(
        "--prefix-len", type=int, default=50,
        help="Prefix length (tokens) for greedy generation (default: 50)",
    )
    parser.add_argument(
        "--seq-len", type=int, default=256,
        help="Maximum sequence length for tokenization (default: 256)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for loss / activation extraction (default: 8)",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Compute device (default: cuda)",
    )
    parser.add_argument(
        "--revision", default=None,
        help="Model revision (e.g. 'step143000' for Pythia checkpoints)",
    )
    args = parser.parse_args()

    # ── Setup ────────────────────────────────────────────────────────────
    setup_logging(logging.INFO)
    set_global_seed(SeedConfig(seed=args.seed))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    config["script"] = "bcd_extractability_predictor.py"
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    log.info(f"Config saved to {out_dir / 'config.json'}")

    # ── Load BCD directions ─────────────────────────────────────────────
    bcd_dir = Path(args.bcd_dir)
    directions_path = bcd_dir / "directions.npz"
    if not directions_path.exists():
        raise FileNotFoundError(
            f"BCD directions file not found: {directions_path}\n"
            f"Expected in --bcd-dir={bcd_dir}"
        )

    log.info(f"Loading BCD directions from {directions_path}")
    bcd_data = np.load(directions_path, allow_pickle=True)

    # Keys are layer-indexed: d_K_layer{L}, d_R_layer{L}, etc.
    layer = args.layer
    dk_key = f"d_K_layer{layer}"
    dr_key = f"d_R_layer{layer}"
    if dk_key not in bcd_data:
        dk_key, dr_key = "d_K", "d_R"
    d_K = bcd_data[dk_key]
    d_R = bcd_data[dr_key]
    log.info(
        f"Loaded {dk_key} (shape={d_K.shape}, norm={np.linalg.norm(d_K):.4f}), "
        f"{dr_key} (shape={d_R.shape}, norm={np.linalg.norm(d_R):.4f})"
    )

    sk_key = f"S_K_basis_layer{layer}"
    sr_key = f"S_R_basis_layer{layer}"
    S_K_basis = bcd_data[sk_key] if sk_key in bcd_data else bcd_data.get("S_K_basis", None)
    S_R_basis = bcd_data[sr_key] if sr_key in bcd_data else bcd_data.get("S_R_basis", None)
    if S_K_basis is not None:
        log.info(f"Loaded S_K_basis: shape={S_K_basis.shape}")
    else:
        log.info("S_K_basis not found in directions.npz; score_SK will be unavailable")
    # Transpose if needed: directions.npz stores (k, d_model), we need (d_model, k)
    if S_K_basis is not None and S_K_basis.shape[0] < S_K_basis.shape[1]:
        S_K_basis = S_K_basis.T
    if S_R_basis is not None and S_R_basis.shape[0] < S_R_basis.shape[1]:
        S_R_basis = S_R_basis.T
    if S_R_basis is not None:
        log.info(f"Loaded S_R_basis: shape={S_R_basis.shape}")
    else:
        log.info("S_R_basis not found in directions.npz; score_SR will be unavailable")

    # Global mean for centering
    if "global_mean" in bcd_data:
        global_mean = bcd_data["global_mean"]
        log.info(f"Using global_mean from directions.npz (norm={np.linalg.norm(global_mean):.4f})")
    else:
        global_mean = None
        log.info("No global_mean in directions.npz; will compute from collected activations")

    # ── Load model ──────────────────────────────────────────────────────
    log.info(f"Loading model from {args.model_path}")
    spec = HFModelSpec(
        name_or_path=args.model_path,
        torch_dtype="float16",
        revision=args.revision,
    )
    wrapper = load_model_and_tokenizer(spec)
    wrapper.model.eval()
    d_model = wrapper.model.config.hidden_size
    log.info(f"Model loaded: d_model={d_model}, n_layers={wrapper.info.n_layers}")

    # Validate layer index
    if args.layer >= wrapper.info.n_layers:
        log.warning(
            f"Requested layer {args.layer} >= n_layers={wrapper.info.n_layers}. "
            f"Using layer {wrapper.info.n_layers // 2} instead."
        )
        layer = wrapper.info.n_layers // 2
    else:
        layer = args.layer

    # ── Load texts ──────────────────────────────────────────────────────
    log.info(f"Loading member texts from {args.member_texts}")
    all_texts = _load_texts(args.member_texts)
    log.info(f"Loaded {len(all_texts)} texts")

    if args.n_texts > 0 and args.n_texts < len(all_texts):
        rng = np.random.RandomState(args.seed)
        indices = rng.choice(len(all_texts), size=args.n_texts, replace=False)
        texts = [all_texts[i] for i in sorted(indices)]
        log.info(f"Subsampled to {len(texts)} texts (seed={args.seed})")
    else:
        texts = all_texts

    n_texts = len(texts)
    tok_cfg = TokenizeConfig(seq_len=args.seq_len, random_crop=False)

    # ── Step 1: Collect activations ─────────────────────────────────────
    log.info(f"Collecting layer-{layer} activations for {n_texts} texts...")
    t0 = time.time()
    activations = collect_activations(
        wrapper, texts, layer, args.seq_len, args.batch_size, args.device
    )
    act_time = time.time() - t0
    log.info(
        f"Activations collected: shape={activations.shape}, time={act_time:.1f}s"
    )

    # ── Step 2: Compute per-text loss ───────────────────────────────────
    log.info("Computing per-text loss...")
    t0 = time.time()
    losses = compute_per_text_loss(
        wrapper, texts, tok_cfg, args.device, args.batch_size
    )
    loss_time = time.time() - t0
    log.info(
        f"Loss: mean={losses.mean():.4f}, std={losses.std():.4f}, time={loss_time:.1f}s"
    )

    # ── Step 3: Compute per-text ROUGE-L ────────────────────────────────
    log.info("Computing per-text ROUGE-L (greedy generation)...")
    t0 = time.time()
    # Use smaller batch size for generation (memory-intensive)
    gen_batch_size = max(1, args.batch_size // 2)
    rouge_l = compute_per_text_rouge_l(
        wrapper, texts, tok_cfg, args.prefix_len, args.device,
        batch_size=gen_batch_size,
    )
    gen_time = time.time() - t0
    log.info(
        f"ROUGE-L: mean={np.nanmean(rouge_l):.4f}, "
        f"std={np.nanstd(rouge_l):.4f}, time={gen_time:.1f}s"
    )

    # Free GPU memory -- only numpy from here
    del wrapper
    torch.cuda.empty_cache()
    log.info("Model freed from GPU memory")

    # ── Step 4: Compute BCD scores ──────────────────────────────────────
    if global_mean is None:
        global_mean = activations.mean(axis=0)
        log.info(
            f"Computed global_mean from activations "
            f"(norm={np.linalg.norm(global_mean):.4f})"
        )

    # Validate dimension compatibility
    if d_K.shape[0] != activations.shape[1]:
        raise ValueError(
            f"Dimension mismatch: d_K has dim={d_K.shape[0]} but activations "
            f"have dim={activations.shape[1]}. Check --layer matches BCD layer."
        )

    log.info("Computing BCD projection scores...")
    bcd_scores = compute_bcd_scores(
        activations, d_K, d_R, S_K_basis, S_R_basis, global_mean
    )
    for name, scores in bcd_scores.items():
        log.info(
            f"  {name}: mean={scores.mean():.4f}, std={scores.std():.4f}, "
            f"range=[{scores.min():.4f}, {scores.max():.4f}]"
        )

    # ── Filter out NaN entries (texts too short for generation) ─────────
    valid = ~np.isnan(rouge_l)
    n_valid = int(valid.sum())
    log.info(f"Valid texts for correlation: {n_valid}/{n_texts}")

    if n_valid < 10:
        raise ValueError(
            f"Only {n_valid} valid texts (non-NaN ROUGE-L). Need at least 10 "
            f"for meaningful correlation. Increase --n-texts or --seq-len."
        )

    losses_v = losses[valid]
    rouge_v = rouge_l[valid]
    scores_v = {k: v[valid] for k, v in bcd_scores.items()}

    # ── Step 5: Spearman correlations ───────────────────────────────────
    log.info("Computing Spearman correlations...")
    correlations = {}

    # loss vs ROUGE-L (expect negative: lower loss -> higher ROUGE-L)
    rho, p = spearmanr(losses_v, rouge_v)
    correlations["loss_vs_rouge"] = {"spearman_rho": float(rho), "p": float(p)}
    log.info(f"  loss vs ROUGE-L:     rho={rho:.4f}, p={p:.2e}")

    for score_name in ["score_K", "score_R", "score_SK", "score_SR"]:
        if score_name not in scores_v:
            continue
        rho, p = spearmanr(scores_v[score_name], rouge_v)
        correlations[f"{score_name}_vs_rouge"] = {
            "spearman_rho": float(rho), "p": float(p)
        }
        log.info(f"  {score_name} vs ROUGE-L: rho={rho:.4f}, p={p:.2e}")

    # ── Step 6: Partial correlations ────────────────────────────────────
    log.info("Computing partial correlations (controlling for loss)...")
    partial_correlations = {}

    for score_name in ["score_K", "score_R", "score_SK", "score_SR"]:
        if score_name not in scores_v:
            continue
        rho_partial, p_partial = partial_spearman(
            scores_v[score_name], rouge_v, losses_v
        )
        key = f"{score_name}_vs_rouge_controlling_loss"
        partial_correlations[key] = {
            "rho": float(rho_partial), "p": float(p_partial)
        }
        log.info(
            f"  {score_name} vs ROUGE-L | loss: "
            f"rho={rho_partial:.4f}, p={p_partial:.2e}"
        )

    # ── Build output JSON ───────────────────────────────────────────────
    log.info("Building output JSON...")

    per_text = {
        "losses": [float(x) for x in losses],
        "rouge_l": [float(x) for x in rouge_l],
        "score_K": bcd_scores["score_K"].tolist(),
        "score_R": bcd_scores["score_R"].tolist(),
    }
    if "score_SK" in bcd_scores:
        per_text["score_SK"] = bcd_scores["score_SK"].tolist()
    if "score_SR" in bcd_scores:
        per_text["score_SR"] = bcd_scores["score_SR"].tolist()

    output = {
        "model": args.model_path,
        "bcd_dir": args.bcd_dir,
        "layer": layer,
        "n_texts": n_texts,
        "n_texts_valid": n_valid,
        "d_model": int(activations.shape[1]),
        "prefix_len": args.prefix_len,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "correlations": _sanitize_for_json(correlations),
        "partial_correlations": _sanitize_for_json(partial_correlations),
        "summary_stats": {
            "loss_mean": float(losses_v.mean()),
            "loss_std": float(losses_v.std()),
            "rouge_l_mean": float(rouge_v.mean()),
            "rouge_l_std": float(rouge_v.std()),
        },
        "per_text": _sanitize_for_json(per_text),
        "timing": {
            "activation_s": float(act_time),
            "loss_s": float(loss_time),
            "generation_s": float(gen_time),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

    output_path = out_dir / "extractability_predictor.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Results saved to {output_path}")

    # ── Print summary ───────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("BCD EXTRACTABILITY PREDICTOR SUMMARY")
    print("=" * 72)
    print(f"Model:      {args.model_path}")
    print(f"BCD dir:    {args.bcd_dir}")
    print(f"Layer:      {layer}   d_model: {activations.shape[1]}")
    print(f"Texts:      {n_valid} valid / {n_texts} total")
    print(f"Prefix len: {args.prefix_len} tokens")
    print()

    print("Spearman correlations with ROUGE-L:")
    print(f"  {'Predictor':<30} {'rho':>8} {'p-value':>12}")
    print("  " + "-" * 52)
    for key, vals in correlations.items():
        rho_val = vals["spearman_rho"]
        p_val = vals["p"]
        rho_s = f"{rho_val:.4f}" if rho_val is not None else "N/A"
        p_s = f"{p_val:.2e}" if p_val is not None else "N/A"
        label = key.replace("_vs_rouge", "")
        print(f"  {label:<30} {rho_s:>8} {p_s:>12}")

    print()
    print("Partial correlations (controlling for loss):")
    print(f"  {'Predictor':<30} {'rho':>8} {'p-value':>12}")
    print("  " + "-" * 52)
    for key, vals in partial_correlations.items():
        rho_val = vals["rho"]
        p_val = vals["p"]
        rho_s = f"{rho_val:.4f}" if rho_val is not None else "N/A"
        p_s = f"{p_val:.2e}" if p_val is not None else "N/A"
        label = key.replace("_vs_rouge_controlling_loss", " | loss")
        print(f"  {label:<30} {rho_s:>8} {p_s:>12}")

    # Highlight the key finding
    print()
    sr_key = "score_SR_vs_rouge_controlling_loss"
    if sr_key in partial_correlations:
        sr = partial_correlations[sr_key]
        rho_sr = sr["rho"]
        p_sr = sr["p"]
        if p_sr < 0.01 and abs(rho_sr) > 0.1:
            print(
                f"KEY FINDING: score_SR predicts ROUGE-L BEYOND loss "
                f"(partial rho={rho_sr:.4f}, p={p_sr:.2e})"
            )
        elif p_sr < 0.05:
            print(
                f"MARGINAL: score_SR shows weak partial correlation with ROUGE-L "
                f"(partial rho={rho_sr:.4f}, p={p_sr:.2e})"
            )
        else:
            print(
                f"NULL: score_SR does NOT predict ROUGE-L beyond loss "
                f"(partial rho={rho_sr:.4f}, p={p_sr:.2e})"
            )
    else:
        print("score_SR unavailable (S_R_basis not in directions.npz)")

    print()
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
