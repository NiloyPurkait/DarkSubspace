#!/usr/bin/env python
"""eval_sae_features.py.

Generic per-feature SAE health evaluator. Computes dead-feature counts, rare-feature
counts, lifetime activation-frequency distribution, L0 at multiple thresholds, and
always-on feature detection over a large heldout activation stream.

Used in the dictionary-health appendix and as a sanity check for every trained SAE
in the paper.

Reproduce::

    env/bin/python3 scripts/shared/eval_sae_features.py \\
        --sae-checkpoint runs/sae/.../l1_3.00e-02/sae_final.pt \\
        --model EleutherAI/pythia-1b \\
        --layer 2 \\
        --eval-tokens 100_000_000 \\
        --output runs/sae/.../feature_health.json

Statistics computed
-------------------
1. Dead features. Features that never activate over the entire eval stream.
2. Rare features. Features that activate less than a threshold fraction of tokens.
3. Lifetime activation frequency distribution (histogram).
4. Per-feature activation statistics (mean, std, max).
5. L0 at multiple thresholds (z>0, z>1e-3, z>1e-2).
6. Top-1% always-on features detection (a different collapse signal from dead).
7. Effective number of features used.
"""
from __future__ import annotations

from repo_bootstrap import ensure_src_on_path
ensure_src_on_path()

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import numpy as np
from tqdm.auto import tqdm

from sae_mia_audit.data.sae_corpus import SAECorpusSpec, load_sae_corpus
from sae_mia_audit.data.tokenizer import TokenizeConfig, tokenize_batch
from sae_mia_audit.models.wrapper import load_model_and_tokenizer
from sae_mia_audit.sae.sae import SAEConfig, SparseAutoencoder
from sae_mia_audit.utils.hf import HFModelSpec
from sae_mia_audit.utils.logging import setup_logging, get_logger


def _get_local_rank() -> int:
    for k in ("LOCAL_RANK", "SLURM_LOCALID"):
        if k in os.environ:
            try:
                return int(os.environ[k])
            except Exception:
                pass
    return 0


def _resolve_device_str(device: str) -> str:
    d = str(device).strip().lower()
    if d in ("", "auto"):
        d = "cuda" if torch.cuda.is_available() else "cpu"
    if d.startswith("cuda:"):
        return d
    if d == "cuda":
        if torch.cuda.is_available():
            lr = _get_local_rank()
            n = torch.cuda.device_count()
            if n > 0 and 0 <= lr < n:
                return f"cuda:{lr}"
            return "cuda:0"
        return "cpu"
    return d


def _resolve_model_dtype(dtype: Optional[str], resolved_device: str) -> str:
    if dtype is None or str(dtype).strip().lower() == "auto":
        if resolved_device.startswith("cuda") and torch.cuda.is_available():
            try:
                if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                    return "bfloat16"
            except Exception:
                pass
            return "float16"
        return "float32"
    return str(dtype)


def _extract_layer_activations(
    model,
    layer_idx: int,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    capture: str = "hook",
) -> torch.Tensor:
    """Return [B, T, D] layer activations."""
    if capture == "hook":
        return model.capture_layer_output(layer_idx=layer_idx, input_ids=input_ids, attention_mask=attention_mask)
    out = model.forward(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hs = out.hidden_states
    return hs[layer_idx + 1]


def compute_feature_health(
    model,
    sae: SparseAutoencoder,
    layer_idx: int,
    eval_tokens: int,
    seq_len: int = 256,
    batch_size: int = 4,
    corpus_name: str = "allenai/c4",
    corpus_subset: str = "en",
    seed: int = 42,
    device: str = "cuda",
    activation_capture: str = "hook",
    rare_threshold: float = 1e-4,
    l0_thresholds: List[float] = None,  # Multiple L0 thresholds
    top_always_on_pct: float = 0.01,  # Top X% always-on detection
) -> Dict[str, Any]:
    """Compute comprehensive feature health statistics.
    
    Args:
        model: Loaded model wrapper
        sae: SAE checkpoint
        layer_idx: Which layer's activations to evaluate
        eval_tokens: Total tokens to evaluate (100M for publication-quality)
        seq_len: Sequence length for tokenization
        batch_size: Batch size for forward passes
        corpus_name: HF dataset name
        corpus_subset: Dataset subset
        seed: Random seed
        device: Device for computation
        activation_capture: "hook" or "hidden_states"
        rare_threshold: Fraction below which features are considered "rare"
        l0_thresholds: List of thresholds for L0 computation (default: [0, 1e-3, 1e-2])
        top_always_on_pct: Fraction for "always-on" feature detection (default: 1%)
        
    Returns:
        Dictionary with feature health statistics
    """
    if l0_thresholds is None:
        l0_thresholds = [0.0, 1e-3, 1e-2]
    
    d_sae = sae.d_sae
    
    # Accumulators
    feature_activation_count = torch.zeros(d_sae, dtype=torch.long, device="cpu")
    feature_activation_sum = torch.zeros(d_sae, dtype=torch.float64, device="cpu")
    feature_activation_sq_sum = torch.zeros(d_sae, dtype=torch.float64, device="cpu")
    feature_activation_max = torch.zeros(d_sae, dtype=torch.float32, device="cpu")
    
    # L0 at multiple thresholds, count activations above each threshold
    l0_threshold_counts = {thresh: torch.zeros(d_sae, dtype=torch.long, device="cpu") 
                          for thresh in l0_thresholds}
    l0_per_token_sums = {thresh: 0.0 for thresh in l0_thresholds}  # For mean L0 per token
    
    total_tokens_processed = 0
    
    # Corpus setup
    corpus_spec = SAECorpusSpec(
        name=corpus_name,
        subset=corpus_subset,
        split="train",
        streaming=True,
        seed=seed,
        shuffle=True,
        shuffle_buffer_size=10_000,
        min_chars=50,
        drop_empty=True,
    )
    texts = iter(load_sae_corpus(corpus_spec))
    # Use crop_seed for deterministic eval cropping
    tok_cfg = TokenizeConfig(seq_len=seq_len, random_crop=True, crop_seed=seed)
    
    # Move SAE to device
    sae = sae.to(device).eval()
    model_dtype = next(model.model.parameters()).dtype
    sae = sae.to(model_dtype)
    
    buffer: List[str] = []
    pbar = tqdm(total=eval_tokens, desc="Evaluating feature health", unit="tok")
    
    for text in texts:
        if total_tokens_processed >= eval_tokens:
            break
            
        buffer.append(text)
        if len(buffer) < batch_size:
            continue
        
        batch = tokenize_batch(model.tokenizer, buffer, tok_cfg)
        input_ids = batch["input_ids"].to(model.model.device)
        attn = batch.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(model.model.device)
        
        with torch.no_grad():
            h = _extract_layer_activations(
                model=model,
                layer_idx=layer_idx,
                input_ids=input_ids,
                attention_mask=attn,
                capture=activation_capture,
            )
            
            B, T, D = h.shape
            x = h.reshape(B * T, D)
            
            # Filter out padding tokens
            if attn is not None:
                mask = attn.reshape(B * T).to(dtype=torch.bool)
                x = x[mask]
            
            # Encode through SAE
            z = sae.encode(x)  # [N, d_sae]
            
            # Update statistics
            n_tokens = z.shape[0]
            
            # Count non-zero activations per feature
            nonzero = (z > 0).to(torch.long)  # [N, d_sae]
            feature_activation_count += nonzero.sum(dim=0).cpu()
            
            # L0 at multiple thresholds
            for thresh in l0_thresholds:
                above_thresh = (z > thresh).to(torch.long)  # [N, d_sae]
                l0_threshold_counts[thresh] += above_thresh.sum(dim=0).cpu()
                # Track mean L0 per token at this threshold
                l0_per_token_sums[thresh] += above_thresh.sum().item()
            
            # Sum of activations per feature (for mean)
            feature_activation_sum += z.sum(dim=0).to(torch.float64).cpu()
            
            # Sum of squared activations (for std)
            feature_activation_sq_sum += (z ** 2).sum(dim=0).to(torch.float64).cpu()
            
            # Max activation per feature
            z_max = z.max(dim=0).values.cpu()
            feature_activation_max = torch.maximum(feature_activation_max, z_max)
            
            total_tokens_processed += n_tokens
            pbar.update(n_tokens)
        
        buffer = []
    
    pbar.close()
    
    # Compute final statistics
    n = max(1, total_tokens_processed)
    
    # Dead features: never activated
    dead_mask = (feature_activation_count == 0)
    n_dead = int(dead_mask.sum().item())
    dead_frac = n_dead / d_sae
    
    # Rare features: activated less than threshold fraction of tokens
    activation_freq = feature_activation_count.float() / n
    rare_mask = (activation_freq < rare_threshold) & ~dead_mask
    n_rare = int(rare_mask.sum().item())
    rare_frac = n_rare / d_sae
    
    # Alive features (activated at least once)
    alive_mask = ~dead_mask
    n_alive = int(alive_mask.sum().item())
    
    # Mean activation per feature (when active)
    # Avoid division by zero for dead features
    safe_counts = feature_activation_count.float().clamp_min(1)
    feature_mean_when_active = (feature_activation_sum / safe_counts).float()
    feature_mean_when_active[dead_mask] = 0.0
    
    # Std of activations when active
    # Var = E[X^2] - E[X]^2
    feature_mean_sq = (feature_activation_sq_sum / safe_counts).float()
    feature_var = feature_mean_sq - (feature_mean_when_active ** 2)
    feature_std_when_active = torch.sqrt(feature_var.clamp_min(0))
    feature_std_when_active[dead_mask] = 0.0
    
    # Histogram of activation frequencies (log-scale bins)
    freq_np = activation_freq.numpy()
    # Bins: 0, (0, 1e-6], (1e-6, 1e-5], ..., (1e-1, 1]
    log_bins = [0] + [10 ** i for i in range(-6, 1)]
    hist_counts, hist_edges = np.histogram(freq_np[freq_np > 0], bins=log_bins)
    
    # Feature health distribution
    healthy_mask = activation_freq >= 1e-3  # Activated on at least 0.1% of tokens
    n_healthy = int(healthy_mask.sum().item())
    
    # L0 per token at multiple thresholds
    l0_per_token_by_threshold = {}
    for thresh in l0_thresholds:
        mean_l0 = l0_per_token_sums[thresh] / max(1, total_tokens_processed)
        l0_per_token_by_threshold[f"L0_thresh_{thresh:.0e}"] = float(mean_l0)
    
    # Effective number of features used (firing rate > threshold)
    effective_features_by_threshold = {}
    for thresh in [1e-6, 1e-5, 1e-4, 1e-3]:
        n_effective = int((activation_freq >= thresh).sum().item())
        effective_features_by_threshold[f"effective_features_freq_{thresh:.0e}"] = n_effective
    
    # Top-1% always-on features (a different collapse signal than dead).
    # These are features that activate on a very high fraction of tokens.
    top_k_always_on = max(1, int(d_sae * top_always_on_pct))
    always_on_threshold = 0.5  # Features activating on >50% of tokens
    high_freq_mask = activation_freq > always_on_threshold
    n_always_on = int(high_freq_mask.sum().item())
    
    # Get the top-k most frequently firing features
    top_freq_values, top_freq_indices = torch.topk(activation_freq, min(top_k_always_on, d_sae))
    top_always_on_features = [
        {"feature_id": int(idx), "activation_freq": float(freq)}
        for idx, freq in zip(top_freq_indices.tolist(), top_freq_values.tolist())
    ]
    
    results = {
        "eval_tokens": int(total_tokens_processed),
        "d_sae": int(d_sae),
        "rare_threshold": float(rare_threshold),
        
        # Dead feature analysis
        "n_dead_features": n_dead,
        "dead_feature_frac": float(dead_frac),
        "dead_feature_ids": dead_mask.nonzero(as_tuple=True)[0].tolist()[:1000],  # First 1000 for space
        
        # Rare feature analysis
        "n_rare_features": n_rare,
        "rare_feature_frac": float(rare_frac),
        
        # Alive/healthy features
        "n_alive_features": n_alive,
        "alive_feature_frac": float(n_alive / d_sae),
        "n_healthy_features": n_healthy,
        "healthy_feature_frac": float(n_healthy / d_sae),
        
        # Activation frequency distribution
        "activation_freq_histogram": {
            "bin_edges": [float(e) for e in hist_edges.tolist()],
            "counts": [int(c) for c in hist_counts.tolist()],
        },
        
        # Per-feature statistics (summary)
        "activation_freq_stats": {
            "mean": float(activation_freq.mean().item()),
            "std": float(activation_freq.std().item()),
            "median": float(activation_freq.median().item()),
            "max": float(activation_freq.max().item()),
            "min_nonzero": float(activation_freq[activation_freq > 0].min().item()) if (activation_freq > 0).any() else 0.0,
        },
        
        "mean_activation_when_active_stats": {
            "mean": float(feature_mean_when_active[alive_mask].mean().item()) if n_alive > 0 else 0.0,
            "std": float(feature_mean_when_active[alive_mask].std().item()) if n_alive > 0 else 0.0,
            "max": float(feature_mean_when_active.max().item()),
        },
        
        "max_activation_stats": {
            "mean": float(feature_activation_max.mean().item()),
            "std": float(feature_activation_max.std().item()),
            "max": float(feature_activation_max.max().item()),
        },
        
        # L0 per token at multiple thresholds
        "l0_per_token_by_threshold": l0_per_token_by_threshold,

        # Effective number of features used
        "effective_features_by_threshold": effective_features_by_threshold,

        # Always-on features detection (a different collapse signal than dead)
        "always_on_analysis": {
            "threshold": float(always_on_threshold),
            "n_always_on_features": n_always_on,
            "always_on_frac": float(n_always_on / d_sae),
            "top_always_on_features": top_always_on_features,
        },
        
        # Verdict
        "verdict": {
            "dictionary_collapse": bool(dead_frac > 0.5),
            "high_dead_features": bool(dead_frac > 0.3),
            "acceptable": bool(dead_frac <= 0.3),
        },
    }
    
    return results


def main() -> int:
    setup_logging()
    log = get_logger("eval_sae_features")
    
    ap = argparse.ArgumentParser(description="Evaluate SAE feature health over large activation stream")
    ap.add_argument("--sae-checkpoint", type=str, required=True, help="Path to SAE checkpoint (sae_final.pt)")
    ap.add_argument("--model", type=str, required=True, help="HF model name")
    ap.add_argument("--layer", type=int, required=True, help="Layer index")
    ap.add_argument("--eval-tokens", type=int, default=100_000_000, help="Number of tokens to evaluate (100M for publication-quality statistics)")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="auto")
    ap.add_argument("--corpus", type=str, default="allenai/c4")
    ap.add_argument("--corpus-subset", type=str, default="en")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--activation-capture", type=str, choices=["hook", "hidden_states"], default="hook")
    ap.add_argument("--rare-threshold", type=float, default=1e-4, help="Frequency below which features are 'rare'")
    ap.add_argument("--l0-thresholds", type=float, nargs="+", default=[0.0, 1e-3, 1e-2],
                   help="Thresholds for L0 computation (multiple thresholds)")
    ap.add_argument("--top-always-on-pct", type=float, default=0.01,
                   help="Fraction for always-on feature detection (top 1%%)")
    ap.add_argument("--output", type=str, default=None, help="Output JSON path (default: alongside checkpoint)")
    ap.add_argument("--trust-remote-code", action="store_true")
    
    args = ap.parse_args()
    
    resolved_device = _resolve_device_str(args.device)
    torch_dtype = _resolve_model_dtype(args.dtype, resolved_device)
    
    # Load model
    log.info(f"Loading model: {args.model}")
    spec = HFModelSpec(
        name_or_path=args.model,
        torch_dtype=torch_dtype,
        device=resolved_device,
        trust_remote_code=args.trust_remote_code,
    )
    model = load_model_and_tokenizer(spec)
    model.model.to(resolved_device)
    model.model.eval()
    
    # Load SAE
    log.info(f"Loading SAE checkpoint: {args.sae_checkpoint}")
    ckpt = torch.load(args.sae_checkpoint, map_location=resolved_device)
    sae_cfg = SAEConfig(**ckpt["sae_cfg"])
    sae = SparseAutoencoder(sae_cfg)
    sae.load_state_dict(ckpt["state_dict"])
    
    log.info(f"SAE config: d_model={sae_cfg.d_model}, d_sae={sae_cfg.d_sae}, l1_coeff={sae_cfg.l1_coeff}")
    
    # Compute feature health
    results = compute_feature_health(
        model=model,
        sae=sae,
        layer_idx=args.layer,
        eval_tokens=args.eval_tokens,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        corpus_name=args.corpus,
        corpus_subset=args.corpus_subset,
        seed=args.seed,
        device=resolved_device,
        activation_capture=args.activation_capture,
        rare_threshold=args.rare_threshold,
        l0_thresholds=args.l0_thresholds,
        top_always_on_pct=args.top_always_on_pct,
    )
    
    # Add metadata
    results["metadata"] = {
        "sae_checkpoint": str(args.sae_checkpoint),
        "model": args.model,
        "layer": args.layer,
        "sae_l1_coeff": sae_cfg.l1_coeff,
        "sae_l1_form": getattr(sae_cfg, "l1_form", "mean"),
        "sae_tied_weights": getattr(sae_cfg, "tied_weights", False),
    }
    
    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        ckpt_path = Path(args.sae_checkpoint)
        output_path = ckpt_path.parent / "feature_health.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    
    # Summary
    log.info("=" * 60)
    log.info("FEATURE HEALTH SUMMARY")
    log.info("=" * 60)
    log.info(f"Evaluated {results['eval_tokens']:,} tokens")
    log.info(f"Dictionary size: {results['d_sae']:,}")
    log.info(f"Dead features: {results['n_dead_features']:,} ({results['dead_feature_frac']:.1%})")
    log.info(f"Rare features: {results['n_rare_features']:,} ({results['rare_feature_frac']:.1%})")
    log.info(f"Healthy features: {results['n_healthy_features']:,} ({results['healthy_feature_frac']:.1%})")
    
    # L0 at multiple thresholds
    log.info("")
    log.info("L0 per token at multiple thresholds:")
    for thresh_key, l0_val in results["l0_per_token_by_threshold"].items():
        log.info(f"  {thresh_key}: {l0_val:.2f}")
    
    # Always-on features
    always_on = results["always_on_analysis"]
    log.info("")
    log.info(f"Always-on features (>{always_on['threshold']:.0%} activation rate): "
            f"{always_on['n_always_on_features']} ({always_on['always_on_frac']:.2%})")
    if always_on["n_always_on_features"] > 0:
        log.info(f"  Top always-on: {always_on['top_always_on_features'][:5]}")
    
    verdict = results["verdict"]
    if verdict["dictionary_collapse"]:
        log.warning("DICTIONARY COLLAPSE DETECTED (>50% dead features)")
    elif verdict["high_dead_features"]:
        log.warning("HIGH DEAD FEATURES (>30% dead)")
    else:
        log.info("Feature health acceptable")
    
    log.info(f"Results saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
