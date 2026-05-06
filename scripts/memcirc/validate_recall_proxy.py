#!/usr/bin/env python3
"""Validate the recall proxy used in BCD (behavioral_channels.py).

The BCD recall proxy labels member texts as "high recall" if their
per-text loss is in the bottom 50% (lower loss = more memorized).
This script validates that proxy by computing the Spearman rank
correlation between:
  - Per-text member loss (from the fine-tuned model)
  - Per-text ROUGE-L extraction score (greedy generation from prefix)

If the correlation is strongly negative (lower loss -> higher ROUGE-L),
the loss-based proxy is a reasonable stand-in for extraction-based recall.

Usage:
  env/bin/python3 scripts/memcirc/validate_recall_proxy.py \
    --model-path runs/controlled_ft/run_20260306_055225/ft_epoch5/model \
    --member-texts data/memcirc_ctrl_ft/member.jsonl \
    --output-dir runs/memcirc/recall_proxy_validation/p69_epoch5 \
    [--n-texts 200] [--prefix-len 50] [--seq-len 256] [--seed 42]

Outputs:
  recall_proxy_validation.json  -- per-text losses, ROUGE-L scores,
                                   Spearman rho, p-value, classification
                                   agreement statistics
"""

import _bootstrap  # noqa: F401

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from scipy import stats

from sae_mia_audit.models.wrapper import load_model_and_tokenizer, CausalLMWrapper
from sae_mia_audit.data.tokenizer import TokenizeConfig, tokenize_batch
from sae_mia_audit.utils.seed import SeedConfig, set_global_seed
from sae_mia_audit.utils.logging import setup_logging, get_logger
from sae_mia_audit.utils.hf import HFModelSpec

log = get_logger(__name__)


def _batched(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _lcs_length(a: list, b: list) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def compute_per_text_loss(
    wrapper: CausalLMWrapper,
    texts: List[str],
    tok_cfg: TokenizeConfig,
    device: str,
    batch_size: int = 8,
) -> np.ndarray:
    """Compute per-text cross-entropy loss."""
    losses = []
    with torch.no_grad():
        for chunk in _batched(texts, batch_size):
            batch = tokenize_batch(wrapper.tokenizer, chunk, tok_cfg)
            ids = batch["input_ids"].to(device)
            attn = batch.get("attention_mask", None)
            if attn is not None:
                attn = attn.to(device)
            out = wrapper.forward(input_ids=ids, attention_mask=attn)
            logits = out.logits[:, :-1, :].contiguous()
            labels = ids[:, 1:].contiguous()
            for b in range(logits.shape[0]):
                log_probs = torch.nn.functional.log_softmax(logits[b], dim=-1)
                token_losses = -log_probs[range(labels.shape[1]), labels[b]]
                if attn is not None:
                    mask = attn[b, 1:].float()
                    mean_loss = (token_losses * mask).sum() / mask.sum().clamp(min=1)
                else:
                    mean_loss = token_losses.mean()
                losses.append(mean_loss.item())
    return np.array(losses)


def compute_per_text_rouge_l(
    wrapper: CausalLMWrapper,
    texts: List[str],
    tok_cfg: TokenizeConfig,
    prefix_len: int,
    device: str,
    batch_size: int = 4,
) -> np.ndarray:
    """Compute per-text ROUGE-L F1 from greedy generation after prefix."""
    rouge_scores = []
    n_skipped = 0
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch = tokenize_batch(wrapper.tokenizer, batch_texts, tok_cfg)
            ids = batch["input_ids"].to(device)

            if ids.shape[1] <= prefix_len + 10:
                n_skipped += ids.shape[0]
                for _ in range(ids.shape[0]):
                    rouge_scores.append(float('nan'))
                continue

            prefix = ids[:, :prefix_len]
            target = ids[:, prefix_len:]
            gen_len = target.shape[1]

            # Autoregressive generation
            generated = prefix.clone()
            for _ in range(gen_len):
                out = wrapper.forward(input_ids=generated)
                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

            gen_suffix = generated[:, prefix_len:]
            min_len = min(gen_suffix.shape[1], target.shape[1])

            for b in range(ids.shape[0]):
                pred = gen_suffix[b, :min_len].cpu().numpy()
                gold = target[b, :min_len].cpu().numpy()
                lcs_len = _lcs_length(pred.tolist(), gold.tolist())
                prec = lcs_len / max(len(pred), 1)
                recall = lcs_len / max(len(gold), 1)
                f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
                rouge_scores.append(f1)

            if (i // batch_size) % 10 == 0:
                log.info(f"  Generation progress: {i + len(batch_texts)}/{len(texts)} texts")

    if n_skipped > 0:
        log.warning(f"Skipped {n_skipped} texts too short for prefix_len={prefix_len}")
    return np.array(rouge_scores)


def main():
    parser = argparse.ArgumentParser(
        description="Validate loss-based recall proxy against ROUGE-L extraction"
    )
    parser.add_argument("--model-path", required=True,
                        help="Path to fine-tuned model checkpoint")
    parser.add_argument("--member-texts", required=True,
                        help="JSONL of member texts")
    parser.add_argument("--n-texts", type=int, default=0,
                        help="Number of texts to evaluate (0 = all)")
    parser.add_argument("--prefix-len", type=int, default=50,
                        help="Prefix length for extraction (default: 50, matching DD)")
    parser.add_argument("--seq-len", type=int, default=256,
                        help="Sequence length for tokenization")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for loss computation")
    parser.add_argument("--batch-size-gen", type=int, default=4,
                        help="Batch size for generation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()
    setup_logging()
    set_global_seed(SeedConfig(seed=args.seed))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Load model ---------------------------------------------------------
    log.info(f"Loading model from {args.model_path}")
    spec = HFModelSpec(name_or_path=args.model_path, torch_dtype="float16")
    wrapper = load_model_and_tokenizer(spec)
    wrapper.model.eval()

    # -- Load texts ---------------------------------------------------------
    log.info(f"Loading member texts from {args.member_texts}")
    with open(args.member_texts) as f:
        all_texts = [json.loads(line)["text"] for line in f]
    log.info(f"Loaded {len(all_texts)} texts")

    if args.n_texts > 0 and args.n_texts < len(all_texts):
        rng = np.random.RandomState(args.seed)
        indices = rng.choice(len(all_texts), size=args.n_texts, replace=False)
        texts = [all_texts[i] for i in indices]
        log.info(f"Subsampled to {len(texts)} texts (seed={args.seed})")
    else:
        texts = all_texts
        indices = list(range(len(texts)))

    tok_cfg = TokenizeConfig(seq_len=args.seq_len, random_crop=False)

    # -- Compute per-text loss ----------------------------------------------
    log.info("Computing per-text loss...")
    t0 = time.time()
    losses = compute_per_text_loss(wrapper, texts, tok_cfg, args.device, args.batch_size)
    loss_time = time.time() - t0
    log.info(f"Loss computation: {loss_time:.1f}s, mean={losses.mean():.4f}, std={losses.std():.4f}")

    # -- Compute per-text ROUGE-L -------------------------------------------
    log.info("Computing per-text ROUGE-L (greedy generation)...")
    t0 = time.time()
    rouge_l = compute_per_text_rouge_l(
        wrapper, texts, tok_cfg, args.prefix_len, args.device, args.batch_size_gen
    )
    gen_time = time.time() - t0
    log.info(f"Generation: {gen_time:.1f}s, mean ROUGE-L={np.nanmean(rouge_l):.4f}")

    # -- Filter out NaN entries (texts too short) ---------------------------
    valid = ~np.isnan(rouge_l)
    losses_valid = losses[valid]
    rouge_valid = rouge_l[valid]
    n_valid = int(valid.sum())
    log.info(f"Valid texts for correlation: {n_valid}/{len(texts)}")

    # -- Spearman rank correlation ------------------------------------------
    rho, pvalue = stats.spearmanr(losses_valid, rouge_valid)
    log.info(f"Spearman rho(loss, ROUGE-L) = {rho:.4f}, p = {pvalue:.2e}")

    # Also Pearson for reference
    pearson_r, pearson_p = stats.pearsonr(losses_valid, rouge_valid)
    log.info(f"Pearson r(loss, ROUGE-L) = {pearson_r:.4f}, p = {pearson_p:.2e}")

    # -- Classification agreement -------------------------------------------
    # BCD uses median split on loss: below median = high recall (label 1)
    median_loss = np.median(losses_valid)
    loss_proxy_label = (losses_valid < median_loss).astype(int)

    # Ground truth: median split on ROUGE-L: above median = high recall
    median_rouge = np.median(rouge_valid)
    rouge_label = (rouge_valid > median_rouge).astype(int)

    agreement = (loss_proxy_label == rouge_label).mean()
    log.info(f"Median-split classification agreement: {agreement:.4f}")

    # Contingency analysis
    tp = int(((loss_proxy_label == 1) & (rouge_label == 1)).sum())
    fp = int(((loss_proxy_label == 1) & (rouge_label == 0)).sum())
    fn = int(((loss_proxy_label == 0) & (rouge_label == 1)).sum())
    tn = int(((loss_proxy_label == 0) & (rouge_label == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    log.info(f"Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")
    log.info(f"Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}")

    # Quartile analysis: does extraction drop monotonically with loss quartiles?
    quartiles = np.percentile(losses_valid, [25, 50, 75])
    q_labels = np.digitize(losses_valid, quartiles)  # 0=lowest loss, 3=highest
    quartile_stats = []
    for q in range(4):
        mask = q_labels == q
        if mask.sum() > 0:
            q_stat = {
                "quartile": q,
                "n": int(mask.sum()),
                "loss_mean": float(losses_valid[mask].mean()),
                "loss_std": float(losses_valid[mask].std()),
                "rouge_l_mean": float(rouge_valid[mask].mean()),
                "rouge_l_std": float(rouge_valid[mask].std()),
            }
            quartile_stats.append(q_stat)
            log.info(f"Q{q} (n={q_stat['n']}): loss={q_stat['loss_mean']:.4f}, "
                     f"ROUGE-L={q_stat['rouge_l_mean']:.4f}")

    # -- Save results -------------------------------------------------------
    output = {
        "model_path": args.model_path,
        "member_texts": args.member_texts,
        "n_texts_total": len(texts),
        "n_texts_valid": n_valid,
        "prefix_len": args.prefix_len,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "correlation": {
            "spearman_rho": float(rho),
            "spearman_pvalue": float(pvalue),
            "pearson_r": float(pearson_r),
            "pearson_pvalue": float(pearson_p),
        },
        "classification_agreement": {
            "median_loss_threshold": float(median_loss),
            "median_rouge_threshold": float(median_rouge),
            "agreement_rate": float(agreement),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": float(precision),
            "recall_metric": float(recall),
            "f1": float(f1),
        },
        "quartile_analysis": quartile_stats,
        "summary_stats": {
            "loss_mean": float(losses_valid.mean()),
            "loss_std": float(losses_valid.std()),
            "loss_min": float(losses_valid.min()),
            "loss_max": float(losses_valid.max()),
            "rouge_l_mean": float(rouge_valid.mean()),
            "rouge_l_std": float(rouge_valid.std()),
            "rouge_l_min": float(rouge_valid.min()),
            "rouge_l_max": float(rouge_valid.max()),
        },
        "per_text": {
            "indices": [int(i) for i in indices[valid] if valid.any()] if isinstance(indices, np.ndarray) else [indices[j] for j in range(len(indices)) if valid[j]],
            "losses": [float(x) for x in losses_valid],
            "rouge_l": [float(x) for x in rouge_valid],
            "loss_proxy_label": [int(x) for x in loss_proxy_label],
            "rouge_label": [int(x) for x in rouge_label],
        },
        "timing": {
            "loss_computation_s": float(loss_time),
            "generation_s": float(gen_time),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_path = out_dir / "recall_proxy_validation.json"
    out_path.write_text(json.dumps(output, indent=2))
    log.info(f"\nResults saved to {out_path}")

    # -- Print summary ------------------------------------------------------
    log.info(f"\n{'='*60}")
    log.info("RECALL PROXY VALIDATION SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"Model: {args.model_path}")
    log.info(f"Texts: {n_valid} valid / {len(texts)} total")
    log.info(f"Spearman rho(loss, ROUGE-L) = {rho:.4f} (p = {pvalue:.2e})")
    log.info(f"Pearson r(loss, ROUGE-L)    = {pearson_r:.4f} (p = {pearson_p:.2e})")
    log.info(f"Median-split agreement      = {agreement:.1%}")
    log.info(f"Proxy precision/recall/F1   = {precision:.4f}/{recall:.4f}/{f1:.4f}")
    if rho < -0.5 and pvalue < 0.01:
        log.info("VERDICT: Strong negative correlation -- loss proxy is well-validated")
    elif rho < -0.3 and pvalue < 0.05:
        log.info("VERDICT: Moderate negative correlation -- loss proxy is reasonable")
    elif rho < 0 and pvalue < 0.05:
        log.info("VERDICT: Weak negative correlation -- loss proxy has limited validity")
    else:
        log.info("VERDICT: No significant negative correlation -- loss proxy NOT validated")


if __name__ == "__main__":
    main()
