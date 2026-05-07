#!/usr/bin/env python
"""train_sae.py.

Canonical sparse autoencoder trainer. Supports mixed-data, member-only, and
disjoint-corpus training regimes for the dark-subspace experiments. Invoked by
all SLURM wrappers under ``scripts/dark_subspace/shell/`` and produces the
``sae_final.pt`` checkpoints consumed by ``sae_dark_subspace.py``.

Used in Section 2.2 (SAE training protocol) of the paper.

Reproduce::

    env/bin/python3 scripts/shared/train_sae.py \\
        --model EleutherAI/pythia-6.9b --layers 16 \\
        --d-model-mult 4 --l1-coeff 5e-4 \\
        --train-tokens 200000000 --seed 42 \\
        --corpus path/to/corpus.jsonl --runs-dir runs/sae
"""

from __future__ import annotations

# This makes `python scripts/shared/train_sae.py ...` work reliably in
# environments where `PYTHONPATH`/working-directory is not preserved (e.g.
# multi-node `accelerate`). It is intentionally lightweight and does not
# replace best practice (`pip install -e .`).
from repo_bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
from collections import deque
from typing import Deque, Dict, Iterable, Iterator, List, Optional

import torch
from tqdm.auto import tqdm

from sae_mia_audit.data.sae_corpus import SAECorpusSpec, load_sae_corpus
from sae_mia_audit.data.tokenizer import TokenizeConfig, tokenize_batch
from sae_mia_audit.models.wrapper import load_model_and_tokenizer
from sae_mia_audit.sae.sae import SAEConfig, SparseAutoencoder
from sae_mia_audit.sae.trainer import SAETrainConfig, SAETrainer, MultiSAETrainer
from sae_mia_audit.sae.topk import FeatureTopKCollector
from sae_mia_audit.sae.interpret import heuristic_label_from_contexts
from sae_mia_audit.utils.hf import HFModelSpec
from sae_mia_audit.utils.logging import setup_logging, get_logger
from sae_mia_audit.utils.run_dir import make_run_dir, snapshot_reproducibility
from sae_mia_audit.utils.seed import SeedConfig, set_global_seed


# ---------------------------------------------------------------------
# Device / dtype helpers
# ---------------------------------------------------------------------

def _get_local_rank() -> int:
    for k in ("LOCAL_RANK", "SLURM_LOCALID"):
        if k in os.environ:
            try:
                return int(os.environ[k])
            except Exception:
                pass
    return 0


def _resolve_device_str(device: str) -> str:
    """Resolve device string with launcher-friendly CUDA local-rank handling."""
    d = str(device).strip().lower()

    if d in ("", "auto"):
        d = "cuda" if torch.cuda.is_available() else "cpu"

    # Explicit cuda:N is respected as-is
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


def _normalize_device_map(device_map: Optional[str]) -> Optional[str]:
    if device_map is None:
        return None
    dm = str(device_map).strip()
    if dm == "" or dm.lower() in ("none", "null"):
        return None
    return dm


def _resolve_model_dtype(dtype: Optional[str], resolved_device: str) -> str:
    """Pick a sensible model forward dtype if dtype is 'auto'."""
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


def _parse_activation_dtype(dtype: str, model_param_dtype: torch.dtype) -> torch.dtype:
    """Activation dtype for SAE training batches."""
    d = str(dtype).strip().lower()
    if d in ("auto", ""):
        # Default: fp32 for stability unless model itself is fp32.
        return torch.float32 if model_param_dtype != torch.float32 else model_param_dtype
    if d in ("fp32", "float32"):
        return torch.float32
    if d in ("fp16", "float16", "half"):
        return torch.float16
    if d in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unknown --activation-dtype: {dtype}")


def _assert_single_process() -> None:
    # SAE training here is intentionally single-process for reproducibility.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise SystemExit(
            f"scripts/shared/train_sae.py expects a single process, but WORLD_SIZE={world_size}. "
            "Re-run with: accelerate launch --num_processes 1 scripts/shared/train_sae.py ..."
        )


def _assert_single_device_model(model, device_map: Optional[str]) -> None:
    # If the user requested device_map sharding, we can't safely assume a single-device
    # tensor path for activation extraction in this script.
    if device_map is not None:
        devs = {p.device for p in model.model.parameters()}
        if len(devs) > 1:
            raise RuntimeError(
                "This training script assumes a single-device model for activation extraction. "
                f"Detected parameters on multiple devices: {sorted(map(str, devs))}. "
                "Re-run with --device-map none (default) or adapt the activation capture to handle sharded models."
            )


def _get_model_param_dtype(model) -> torch.dtype:
    try:
        return next(model.model.parameters()).dtype
    except StopIteration:  # pragma: no cover
        return torch.float32


# ---------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------

def _extract_layer_activations(
    model,
    layer_idx: int,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    capture: str,
) -> torch.Tensor:
    """Return [B, T, D] layer activations.

    capture:
      - "hidden_states": use HF output_hidden_states=True (legacy)
      - "hook": use forward hook to capture only the target layer output (faster)
    """
    if capture == "hook":
        return model.capture_layer_output(layer_idx=layer_idx, input_ids=input_ids, attention_mask=attention_mask)
    out = model.forward(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hs = out.hidden_states
    # hidden after layer layer_idx is hs[layer_idx+1]
    return hs[layer_idx + 1]


# ---------------------------------------------------------------------
# Corpus hygiene: clean / shuffle / dedup
# ---------------------------------------------------------------------

def _normalize_text_for_hash(t: str) -> str:
    # Light normalization for dedup; deterministic.
    # (We do NOT lowercase; casing can matter for tokenization.)
    return " ".join(t.strip().split())


def _iter_clean_texts(
    texts: Iterable[str],
    *,
    min_chars: int = 1,
    max_chars: Optional[int] = None,
) -> Iterator[str]:
    for t in texts:
        if t is None:
            continue
        s = str(t).strip()
        if len(s) < int(min_chars):
            continue
        if max_chars is not None and len(s) > int(max_chars):
            s = s[: int(max_chars)]
        yield s


def _iter_shuffle_buffer(
    texts: Iterable[str],
    *,
    buffer_size: int,
    seed: int,
) -> Iterator[str]:
    """Streaming shuffle via a fixed-size buffer (deterministic given seed)."""
    b = int(buffer_size)
    if b <= 0:
        yield from texts
        return

    rng = random.Random(int(seed))
    buf: List[str] = []

    it = iter(texts)
    for _ in range(b):
        try:
            buf.append(next(it))
        except StopIteration:
            break

    if not buf:
        return

    # Reservoir-style: for each incoming item, swap with a random buffer element.
    for x in it:
        j = rng.randrange(len(buf))
        yield buf[j]
        buf[j] = x

    rng.shuffle(buf)
    yield from buf


def _iter_dedup_texts(
    texts: Iterable[str],
    *,
    enabled: bool,
    max_items: int,
    normalize: bool = True,
) -> Iterator[str]:
    """Drop duplicate documents using a bounded LRU set of hashes."""
    if not enabled:
        yield from texts
        return

    cap = max(1, int(max_items))
    seen: set[bytes] = set()
    order: Deque[bytes] = deque()

    for t in texts:
        s = _normalize_text_for_hash(t) if normalize else t
        h = hashlib.blake2b(s.encode("utf-8"), digest_size=16).digest()
        if h in seen:
            continue

        seen.add(h)
        order.append(h)
        if len(order) > cap:
            old = order.popleft()
            try:
                seen.remove(old)
            except KeyError:
                pass

        yield t


# ---------------------------------------------------------------------
# Activation batch iterator
# ---------------------------------------------------------------------

def iter_activation_batches(
    model,
    texts: Iterator[str],
    *,
    layer_idx: int,
    seq_len: int,
    forward_batch_size: int,
    tokens_per_step: int,
    device: str,
    train_tokens: int,
    seed: int,
    activation_capture: str = "hidden_states",
    activation_sampling: str = "repeat",
    token_filter: str = "all",
    activation_dtype: torch.dtype = torch.float32,
    pad_stats_callback: Optional[callable] = None,  # Callback to report pad stats
) -> Iterator[torch.Tensor]:
    """Yield [tokens_per_step, d_model] activation batches for SAE training.

    Backward compatibility:
      - activation_capture defaults to "hidden_states" (old behaviour).
      - activation_sampling defaults to "repeat" (old behaviour when batch_tokens < tokens_per_step).

    Recommended for paper runs:
      --activation-capture hook --activation-sampling accumulate --token-filter nonpad
      --corpus-shuffle-buffer 10000 --corpus-dedup
      
    Padding hygiene:
      - Tracks and reports padding token statistics via pad_stats_callback
    """
    # CPU generator keeps determinism stable across devices.
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    
    ## random_crop=True ensures coverage beyond prefix tokens,
    # matching prior SAE training practice
    tok_cfg = TokenizeConfig(seq_len=seq_len, random_crop=True)
    
    # Padding statistics tracking
    total_tokens_seen = 0
    total_pad_tokens = 0

    processed = 0

    buffer: List[str] = []

    # Only used for activation_sampling="accumulate"
    act_buf: Optional[torch.Tensor] = None  # [N, D] on `device`

    for text in texts:
        buffer.append(text)
        if len(buffer) < int(forward_batch_size):
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
        x_all = h.reshape(B * T, D)
        
        # Track padding statistics
        batch_total_tokens = B * T
        total_tokens_seen += batch_total_tokens

        if token_filter == "nonpad" and attn is not None:
            m = attn.reshape(B * T).to(dtype=torch.bool)
            n_nonpad = int(m.sum().item())
            n_pad = batch_total_tokens - n_nonpad
            total_pad_tokens += n_pad
            x_all = x_all[m]
        else:
            # No filtering, but still count pads for statistics
            if attn is not None:
                n_pad = int((attn == 0).sum().item())
                total_pad_tokens += n_pad

        # Cast activations for SAE training stability
        x_all = x_all.detach().to(device=device, dtype=activation_dtype)

        if activation_sampling == "repeat":
            n = x_all.shape[0]
            if n == 0:
                buffer = []
                continue

            if tokens_per_step <= n:
                idx = torch.randperm(n, generator=rng)[:tokens_per_step]
                xb = x_all[idx.to(x_all.device)]
            else:
                reps = int(math.ceil(tokens_per_step / n))
                xb = x_all.repeat((reps, 1))[:tokens_per_step]

            yield xb
            processed += int(tokens_per_step)
            buffer = []
            if processed >= int(train_tokens):
                # Report final padding statistics
                if pad_stats_callback is not None and total_tokens_seen > 0:
                    pad_stats_callback({
                        "total_tokens_seen": total_tokens_seen,
                        "total_pad_tokens": total_pad_tokens,
                        "pad_frac": total_pad_tokens / max(1, total_tokens_seen),
                        "token_filter": token_filter,
                    })
                break
            continue

        # activation_sampling == "accumulate"
        if act_buf is None:
            act_buf = x_all
        else:
            act_buf = torch.cat([act_buf, x_all], dim=0)

        while act_buf is not None and act_buf.shape[0] >= int(tokens_per_step):
            n = int(act_buf.shape[0])
            perm = torch.randperm(n, generator=rng)  # CPU
            perm = perm.to(act_buf.device)          # ensure GPU-safe indexing
            act_buf = act_buf[perm]
            xb = act_buf[:tokens_per_step]
            act_buf = act_buf[tokens_per_step:]

            yield xb
            processed += int(tokens_per_step)
            if processed >= int(train_tokens):
                # Report final padding statistics
                if pad_stats_callback is not None and total_tokens_seen > 0:
                    pad_stats_callback({
                        "total_tokens_seen": total_tokens_seen,
                        "total_pad_tokens": total_pad_tokens,
                        "pad_frac": total_pad_tokens / max(1, total_tokens_seen),
                        "token_filter": token_filter,
                    })
                return

        buffer = []
    
    # Report padding statistics at generator exhaustion
    if pad_stats_callback is not None and total_tokens_seen > 0:
        pad_stats_callback({
            "total_tokens_seen": total_tokens_seen,
            "total_pad_tokens": total_pad_tokens,
            "pad_frac": total_pad_tokens / max(1, total_tokens_seen),
            "token_filter": token_filter,
        })


# ---------------------------------------------------------------------
# Final evaluation pass (eval_summary.json)
# ---------------------------------------------------------------------

@torch.no_grad()
def compute_final_eval_summary(
    model,
    sae: SparseAutoencoder,
    layer_idx: int,
    eval_tokens: int,
    seq_len: int,
    batch_size: int,
    device: str,
    activation_capture: str = "hook",
    activation_dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    """Compute final evaluation metrics on a fresh batch of activations.
    
    This ensures "final" metrics are truly final (computed over a fresh,
    held-out activation slice) rather than just the last logged batch
    during training.
    """
    from sae_mia_audit.data.sae_corpus import SAECorpusSpec, load_sae_corpus
    
    # Use float32 for numerical stability (same as training)
    eval_dtype = torch.float32
    sae = sae.to(device=device, dtype=eval_dtype).eval()
    
    # Use a different seed to ensure disjoint corpus slice
    corpus_spec = SAECorpusSpec(
        name="allenai/c4",
        subset="en",
        split="validation",  # Use validation split for final eval
        streaming=True,
        seed=99999,  # Different seed from training
        shuffle=True,
        shuffle_buffer_size=5_000,
        min_chars=50,
        drop_empty=True,
    )
    texts = iter(load_sae_corpus(corpus_spec))
    # Use crop_seed for deterministic eval cropping
    tok_cfg = TokenizeConfig(seq_len=seq_len, random_crop=True, crop_seed=99999)
    
    # Accumulators
    total_mse = 0.0
    total_var = 0.0
    total_l0 = 0.0
    total_l1_mean = 0.0
    total_l1_sum = 0.0
    total_tokens = 0
    # OR-based dead feature tracking: True if feature ever activated
    feature_ever_active = torch.zeros(sae.d_sae, dtype=torch.bool, device="cpu")
    
    buffer: List[str] = []
    tokens_processed = 0
    
    for text in texts:
        if tokens_processed >= eval_tokens:
            break
            
        buffer.append(text)
        if len(buffer) < batch_size:
            continue
        
        batch = tokenize_batch(model.tokenizer, buffer, tok_cfg)
        input_ids = batch["input_ids"].to(model.model.device)
        attn = batch.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(model.model.device)
        
        h = _extract_layer_activations(
            model=model,
            layer_idx=layer_idx,
            input_ids=input_ids,
            attention_mask=attn,
            capture=activation_capture,
        )
        
        B, T, D = h.shape
        x = h.reshape(B * T, D)
        
        # Filter padding
        if attn is not None:
            mask = attn.reshape(B * T).to(dtype=torch.bool)
            x = x[mask]
        
        # Cast to float32 for numerical stability (must match SAE dtype)
        x = x.to(dtype=eval_dtype)
        n = x.shape[0]
        if n == 0:
            buffer = []
            continue
        
        # Forward through SAE
        x_hat, z, _ = sae(x)
        
        # Metrics
        mse = ((x_hat - x) ** 2).mean().item()
        x_centered = x - x.mean(dim=0, keepdim=True)
        var = (x_centered ** 2).mean().item()
        
        l0 = (z > 0).float().sum(dim=-1).mean().item()
        l1_mean = z.abs().mean().item()
        l1_sum = z.abs().sum(dim=-1).mean().item()
        
        # OR-based dead feature tracking: accumulate which features ever fire
        feature_ever_active |= (z > 0).any(dim=0).cpu()
        
        total_mse += mse * n
        total_var += var * n
        total_l0 += l0 * n
        total_l1_mean += l1_mean * n
        total_l1_sum += l1_sum * n
        total_tokens += n
        tokens_processed += n
        
        buffer = []
    
    # Compute final averages
    n = max(1, total_tokens)

    return {
        "eval_tokens": total_tokens,
        "eval_source": "c4_validation",
        "metrics": {
            "recon_mse": total_mse / n,
            "fvu": (total_mse / n) / max(1e-12, total_var / n),
            "l0_mean": total_l0 / n,
            "l1_mean": total_l1_mean / n,
            "l1_sum": total_l1_sum / n,
            # OR-based: features that NEVER activated are dead
            "dead_feature_frac": float((~feature_ever_active).float().mean().item()),
        },
    }


# ---------------------------------------------------------------------
# Evaluation utilities (padding-safe + token-weighted)
# ---------------------------------------------------------------------

@torch.no_grad()
def eval_ppl(model, eval_texts: List[str], seq_len: int, batch_size: int) -> float:
    """Token-weighted perplexity on eval_texts, ignoring padding tokens."""
    model.model.eval()
    # Use crop_seed for deterministic eval cropping
    tok_cfg = TokenizeConfig(seq_len=seq_len, random_crop=True, crop_seed=42)


    total_nll = 0.0
    total_tokens = 0

    for i in range(0, len(eval_texts), batch_size):
        chunk = eval_texts[i : i + batch_size]
        batch = tokenize_batch(model.tokenizer, chunk, tok_cfg)
        input_ids = batch["input_ids"].to(model.model.device)
        attn = batch.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(model.model.device)

        labels = input_ids.clone()
        if attn is not None:
            labels = labels.masked_fill(attn == 0, -100)

        out = model.model(input_ids=input_ids, attention_mask=attn, labels=labels)

        # HF CausalLM loss is averaged over non-ignored shift positions.
        # Token-weight to get correct global mean.
        n_tok = int((labels[:, 1:] != -100).sum().detach().cpu().item())
        total_nll += float(out.loss.detach().cpu().item()) * max(1, n_tok)
        total_tokens += n_tok

    mean_loss = total_nll / max(1, total_tokens)
    return float(math.exp(mean_loss))


@torch.no_grad()
def eval_ppl_with_reconstruction(
    model,
    sae: SparseAutoencoder,
    layer_idx: int,
    eval_texts: List[str],
    seq_len: int,
    batch_size: int,
) -> float:
    # Keep SAE dtype aligned with model dtype for clean intervention.
    model_dtype = _get_model_param_dtype(model)
    sae = sae.to(model.model.device).to(model_dtype)
    sae.eval()

    def repl(h):
        B, T, D = h.shape
        x = h.reshape(B * T, D)
        x_hat, _, _ = sae(x)
        return x_hat.reshape(B, T, D)

    handle = model.register_residual_hook(layer_idx, repl)
    try:
        ppl = eval_ppl(model, eval_texts, seq_len=seq_len, batch_size=batch_size)
    finally:
        handle.remove()
    return ppl


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    setup_logging()
    log = get_logger("train_sae")

    _assert_single_process()

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="HF model name, e.g. EleutherAI/pythia-1b")
    ap.add_argument("--revision", type=str, default=None)
    ap.add_argument("--layers", type=int, nargs="+", required=True, help="0-indexed transformer layers")

    ap.add_argument("--d-model-mult", type=int, default=4, help="Dictionary size multiplier (d_sae = mult*d_model)")
    ap.add_argument("--l1-coeff", type=float, default=1e-3, help="Single L1 coefficient (use --l1-coeffs for a sweep)")
    ap.add_argument("--l1-coeffs", type=float, nargs="*", default=None, help="Optional sweep of L1 coefficients to train in parallel")
    # Optional elastic-net L2 penalty on sparse codes (Mahdizadehaghdam 2018).
    # Loss gains l2_coeff * (z ** 2).sum(dim=-1).mean(). Default 0.0 = pure L1 (backward compat).
    ap.add_argument(
        "--l2-coeff",
        type=float,
        default=0.0,
        help=(
            "L2 penalty on sparse codes (elastic-net form, Mahdizadehaghdam 2018). "
            "Default 0.0 disables. For elastic-net hyperparameter sweep try {1e-4, 3e-4, 1e-3}."
        ),
    )
    
    # L1 form (standard sparsity objective)
    ap.add_argument(
        "--l1-form",
        type=str,
        choices=["mean", "sum"],
        default="sum",
        help="L1 sparsity penalty form. 'sum' = z.abs().sum(dim=-1).mean() (standard); 'mean' = z.abs().mean() (legacy).",
    )
    
    # Tied weights option
    ap.add_argument(
        "--tied-weights",
        action="store_true",
        help="Use tied weights (decoder = encoder.T). Standard in some SAE papers.",
    )
    
    # Dead feature resampling
    ap.add_argument(
        "--resample-dead-features",
        action="store_true",
        help="Enable periodic resampling of dead features using reconstruction residual.",
    )
    ap.add_argument(
        "--resample-every",
        type=int,
        default=1000,
        help="Steps between dead feature resampling (only if --resample-dead-features).",
    )
    ap.add_argument(
        "--resample-dead-threshold",
        type=float,
        default=1e-6,
        help="Features with firing rate below this are considered dead.",
    )
    
    # Load balancing auxiliary loss
    ap.add_argument(
        "--aux-coeff",
        type=float,
        default=0.0,
        help="Auxiliary loss coefficient for load balancing (0 disables). Try 0.01-0.1.",
    )
    ap.add_argument(
        "--aux-target-firing-rate",
        type=float,
        default=0.01,
        help="Target feature firing rate for load balancing (default 0.01 = 1%%).",
    )

    ap.add_argument("--train-tokens", type=int, default=2_000_000)
    ap.add_argument("--tokens-per-step", type=int, default=8192)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4, help="Number of sequences per model forward pass")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-steps", type=int, default=None, help="Override steps; else derived from train_tokens/tokens_per_step")

    # Device / dtype
    ap.add_argument("--device", type=str, default="auto", help="Device for SAE training activations (auto|cpu|cuda|cuda:N)")
    ap.add_argument("--device-map", type=str, default=None, help="HuggingFace device_map (None or 'auto'). Use with care.")
    ap.add_argument("--dtype", type=str, default="auto", help="Model forward dtype (auto|float32|float16|bfloat16)")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--deterministic", action="store_true", help="Enable deterministic algorithms (best-effort).")

    # Activation extraction knobs (do not change defaults silently).
    ap.add_argument(
        "--activation-capture",
        type=str,
        choices=["hidden_states", "hook"],
        default="hidden_states",
        help="How to capture layer activations. 'hidden_states' matches old behaviour; 'hook' is faster.",
    )
    ap.add_argument(
        "--activation-sampling",
        type=str,
        choices=["repeat", "accumulate"],
        default="repeat",
        help="How to satisfy tokens_per_step when batch_tokens < tokens_per_step. 'repeat' matches old behaviour.",
    )
    ap.add_argument(
        "--token-filter",
        type=str,
        choices=["all", "nonpad"],
        default="all",
        help="Whether to include padded tokens in SAE training activations. 'all' matches old behaviour.",
    )
    ap.add_argument(
        "--activation-dtype",
        type=str,
        default="float32",
        help="Dtype for SAE training activation batches (float32|float16|bfloat16|auto).",
    )

    ap.add_argument("--seed", type=int, default=0)

    # Corpus
    ap.add_argument("--corpus", type=str, default="allenai/c4")
    ap.add_argument("--corpus-subset", type=str, default="en")
    ap.add_argument("--corpus-split", type=str, default="train")
    ap.add_argument("--corpus-limit-examples", type=int, default=None)
    ap.add_argument(
                    "--corpus-drop-empty",
                    action="store_true",
                    help="Drop empty docs before tokenization (recommended).",
                )

    # Corpus hygiene flags (defaults remain legacy unless --mode paper)
    ap.add_argument("--corpus-min-chars", type=int, default=1)
    ap.add_argument("--corpus-shuffle-buffer", type=int, default=0, help="Streaming shuffle buffer size (0 disables).")
    ap.add_argument("--corpus-dedup", action="store_true", help="Drop duplicate documents using a bounded hash set.")
    ap.add_argument("--corpus-dedup-max", type=int, default=200_000, help="Max hashes to keep for dedup (LRU).")
    ap.add_argument("--corpus-text-field", type=str, default=None, help="Text field name in dataset (auto-inferred if None).")

    # Optional eval / interpretation
    ap.add_argument("--eval-ppl", action="store_true", help="Compute perplexity impact by swapping in reconstructions")
    ap.add_argument("--ppl-eval-examples", type=int, default=128)
    ap.add_argument("--interpret", action="store_true", help="Collect top activating contexts for a sample of features")
    ap.add_argument("--interpret-features", type=int, default=256)
    ap.add_argument("--interpret-topk", type=int, default=8)
    ap.add_argument("--interpret-examples", type=int, default=256)
    
    # Final evaluation pass
    ap.add_argument("--final-eval", action="store_true", 
                   help="Run final evaluation pass on heldout data (writes eval_summary.json)")
    ap.add_argument("--final-eval-tokens", type=int, default=50_000,
                   help="Tokens for the held-out final evaluation pass")

    ap.add_argument("--runs-dir", type=str, default="runs/sae")
    
    # Resume from checkpoint
    ap.add_argument("--resume", action="store_true",
                    help="Resume training from latest checkpoint in --resume-dir")
    ap.add_argument("--resume-dir", type=str, default=None,
                    help="Directory containing checkpoints to resume from (default: auto-detect latest)")

    # Convenience preset; default keeps legacy defaults.
    ap.add_argument("--mode", type=str, choices=["none", "quick", "paper"], default="none")

    args = ap.parse_args()

    # Mode presets (gated; do not silently change legacy behaviour).
    if args.mode == "paper":
        args.activation_sampling = "accumulate"
        args.activation_capture = "hook"
        args.token_filter = "nonpad"
        # Corpus hygiene defaults for paper runs
        if args.corpus_shuffle_buffer == 0:
            args.corpus_shuffle_buffer = 10_000
        if not args.corpus_dedup:
            args.corpus_dedup = True
        if args.corpus_min_chars <= 1:
            args.corpus_min_chars = 50
        if not args.corpus_drop_empty:
            args.corpus_drop_empty = True
        # Enable final evaluation for paper mode
        if not args.final_eval:
            args.final_eval = True
    elif args.mode == "quick":
        pass

    set_global_seed(SeedConfig(seed=args.seed, deterministic=bool(args.deterministic)))

    resolved_device = _resolve_device_str(args.device)
    device_map = _normalize_device_map(args.device_map)
    torch_dtype = _resolve_model_dtype(args.dtype, resolved_device)

    # Build HFModelSpec in a version-tolerant way (repo may add fields over time).
    spec_kwargs = dict(
        name_or_path=args.model,
        revision=args.revision,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
        device=resolved_device,
    )
    spec_fields = getattr(HFModelSpec, "__dataclass_fields__", {})  # type: ignore[attr-defined]
    spec_kwargs = {k: v for k, v in spec_kwargs.items() if (not spec_fields) or (k in spec_fields)}
    spec = HFModelSpec(**spec_kwargs)  # type: ignore[arg-type]

    model = load_model_and_tokenizer(spec)
    if device_map is None:
        model.model.to(resolved_device)
    _assert_single_device_model(model, device_map=device_map)

    # Ensure deterministic forward behaviour (no dropout).
    try:
        model.model.eval()
    except Exception:
        pass

    model_param_dtype = _get_model_param_dtype(model)
    act_dtype = _parse_activation_dtype(args.activation_dtype, model_param_dtype)

    log.info(
        json.dumps(
            {
                "event": "env",
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "model": args.model,
                "revision": args.revision,
                "resolved_device": resolved_device,
                "device_map": device_map,
                "model_param_dtype": str(model_param_dtype),
                "activation_dtype": str(act_dtype),
            },
            indent=2,
        )
    )
    log.info(f"Loaded model: {model.info}")

    # Evaluation texts (small, deterministic)
    eval_texts: List[str] = []
    if args.eval_ppl or args.interpret:
        try:
            from datasets import load_dataset  # type: ignore
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            eval_texts = [x["text"] for x in ds if x.get("text", "").strip()]
        except Exception:
            eval_texts = ["Hello world."] * max(int(args.ppl_eval_examples), int(args.interpret_examples))

    for layer_idx in args.layers:
        l1_list = args.l1_coeffs if args.l1_coeffs else [args.l1_coeff]
        l1_tag = "sweep" if len(l1_list) > 1 else f"l1{l1_list[0]:g}"
        
        # Include architecture variants in run name for ablation tracking
        arch_tag = ""
        if args.tied_weights:
            arch_tag += "_tied"
        if args.l1_form == "mean":
            arch_tag += "_l1mean"
        
        run_name = f"train_sae__{args.model.replace('/', '_')}__layer{layer_idx}__mult{args.d_model_mult}__{l1_tag}{arch_tag}"
        
        # Handle resume: use existing run_dir if resuming
        resume_step = 0
        if args.resume:
            if args.resume_dir:
                run_dir = Path(args.resume_dir)
                if not run_dir.exists():
                    log.error(f"Resume directory does not exist: {run_dir}")
                    return 1
                log.info(f"Resuming from specified directory: {run_dir}")
            else:
                # Auto-detect: find latest matching run directory
                runs_base = Path(args.runs_dir)
                if runs_base.exists():
                    matching_runs = sorted(
                        [d for d in runs_base.iterdir() if d.is_dir() and run_name in d.name],
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    if matching_runs:
                        run_dir = matching_runs[0]
                        log.info(f"Resuming from latest matching run: {run_dir}")
                    else:
                        log.warning(f"No matching run found for {run_name}, starting fresh")
                        run_dir = make_run_dir(args.runs_dir, run_name)
                else:
                    log.warning(f"Runs directory does not exist, starting fresh: {runs_base}")
                    run_dir = make_run_dir(args.runs_dir, run_name)
        else:
            run_dir = make_run_dir(args.runs_dir, run_name)

        cfg_dict = (
            vars(args)
            | {
                "layer_idx": int(layer_idx),
                "resolved_device": resolved_device,
                "resolved_dtype": torch_dtype,
                "resolved_device_map": device_map,
                "model_param_dtype": str(model_param_dtype),
                "activation_dtype": str(act_dtype),
            }
        )
        snapshot_reproducibility(run_dir, cfg_dict)

        # SAE config - now includes l1_form and tied_weights
        d_model = int(model.info.d_model)
        d_sae = int(args.d_model_mult) * int(d_model)

        saes: Dict[str, SparseAutoencoder] = {}
        for l1 in l1_list:
            sae_cfg = SAEConfig(
                d_model=d_model,
                d_sae=d_sae,
                l1_coeff=float(l1),
                l1_form=args.l1_form,  # Standard "sum" vs legacy "mean"
                l2_coeff=float(args.l2_coeff),  # elastic-net L2 on codes
                tied_weights=args.tied_weights,  # Tied weights ablation
                aux_coeff=float(args.aux_coeff),  # Load balancing
                aux_target_firing_rate=float(args.aux_target_firing_rate),
                use_bias=True,
                normalize_decoder=True,
            )
            key = f"l1_{float(l1):.2e}".replace("+", "")
            saes[key] = SparseAutoencoder(sae_cfg)

        # Trainer config
        steps = args.max_steps
        if steps is None:
            steps = int(math.ceil(int(args.train_tokens) / int(args.tokens_per_step)))

        # Build SAETrainConfig in a version-tolerant way (repo may add fields).
        train_kwargs = dict(
            lr=float(args.lr),
            max_steps=int(steps),
            device=resolved_device,
            log_every=50,
            save_every=1000,
            # Dead feature resampling
            resample_dead_features=bool(args.resample_dead_features),
            resample_every=int(args.resample_every),
            resample_dead_threshold=float(args.resample_dead_threshold),
        )
        train_fields = getattr(SAETrainConfig, "__dataclass_fields__", {})  # type: ignore[attr-defined]
        train_kwargs = {k: v for k, v in train_kwargs.items() if (not train_fields) or (k in train_fields)}
        train_cfg = SAETrainConfig(**train_kwargs)  # type: ignore[arg-type]

        if len(saes) == 1:
            trainer: SAETrainer | MultiSAETrainer = SAETrainer(sae=next(iter(saes.values())), cfg=train_cfg, out_dir=run_dir)
        else:
            trainer = MultiSAETrainer(saes=saes, cfg=train_cfg, out_dir=run_dir)
        
        # Handle resume: load checkpoint if available
        if args.resume:
            try:
                if isinstance(trainer, MultiSAETrainer):
                    # For MultiSAETrainer, find checkpoint in first SAE subdir
                    first_key = next(iter(saes.keys()))
                    ckpt_path = trainer.find_latest_checkpoint(run_dir / first_key)
                else:
                    ckpt_path = trainer.find_latest_checkpoint(run_dir)
                
                if ckpt_path is not None:
                    resume_step = trainer.load(ckpt_path if isinstance(trainer, SAETrainer) else ckpt_path.stem)
                    log.info(f"Loaded checkpoint from step {resume_step}, will skip {resume_step} batches")
                else:
                    log.warning("No checkpoint found, starting fresh")
                    resume_step = 0
            except Exception as e:
                log.error(f"Failed to load checkpoint: {e}")
                log.warning("Starting fresh due to checkpoint load failure")
                resume_step = 0

        # Corpus iterator
        corpus_spec = SAECorpusSpec(
            name=args.corpus,
            subset=args.corpus_subset,
            split=args.corpus_split,
            streaming=True,
            limit_examples=args.corpus_limit_examples,

            seed=args.seed,

            # Drive these from CLI:
            shuffle=(int(args.corpus_shuffle_buffer) > 0),
            shuffle_buffer_size=int(args.corpus_shuffle_buffer),

            dedupe=bool(args.corpus_dedup),
            dedupe_window=int(args.corpus_dedup_max),

            min_chars=int(args.corpus_min_chars),
            drop_empty=bool(args.corpus_drop_empty),
            text_field=args.corpus_text_field,
        )

        texts: Iterable[str] = load_sae_corpus(corpus_spec)

        # Hygiene wrappers (deterministic)
        #texts = _iter_clean_texts(texts, min_chars=args.corpus_min_chars, max_chars=args.corpus_max_chars)
        #texts = _iter_dedup_texts(texts, enabled=bool(args.corpus_dedup), max_items=int(args.corpus_dedup_max), normalize=True)

        # `load_sae_corpus(...)` already performs a buffered shuffle.
        #texts = _iter_shuffle_buffer(texts, buffer_size=int(args.corpus_shuffle_buffer), seed=int(args.seed))
        
        # Pad statistics tracking
        pad_stats_holder: Dict[str, Any] = {}
        def pad_stats_callback(stats: dict) -> None:
            pad_stats_holder.update(stats)

        activations = iter_activation_batches(
            model=model,
            texts=iter(texts),
            layer_idx=int(layer_idx),
            seq_len=int(args.seq_len),
            forward_batch_size=int(args.batch_size),
            tokens_per_step=int(args.tokens_per_step),
            device=resolved_device,
            train_tokens=int(args.train_tokens),
            seed=int(args.seed),
            activation_capture=str(args.activation_capture),
            activation_sampling=str(args.activation_sampling),
            token_filter=str(args.token_filter),
            activation_dtype=act_dtype,
            pad_stats_callback=pad_stats_callback,
        )

        log.info(
            json.dumps(
                {
                    "event": "train_start",
                    "run_dir": str(run_dir),
                    "layer_idx": int(layer_idx),
                    "d_model": d_model,
                    "d_sae": d_sae,
                    "n_saes": len(saes),
                    "l1_coeffs": [float(x) for x in l1_list],
                    "l1_form": args.l1_form,  # Document L1 form
                    "tied_weights": args.tied_weights,  # Document weight tying
                    "aux_coeff": args.aux_coeff,  # Load balancing
                    "aux_target_firing_rate": args.aux_target_firing_rate,
                    "resample_dead_features": args.resample_dead_features,
                    "resample_every": args.resample_every,
                    "resample_dead_threshold": args.resample_dead_threshold,
                    "train_cfg": train_cfg.__dict__,
                    "activation_capture": args.activation_capture,
                    "activation_sampling": args.activation_sampling,
                    "token_filter": args.token_filter,
                    "train_tokens_target": int(args.train_tokens),
                    "tokens_per_step": int(args.tokens_per_step),
                    "steps": int(steps),
                    "expected_tokens": int(steps) * int(args.tokens_per_step),
                    "corpus": {
                        "name": args.corpus,
                        "subset": args.corpus_subset,
                        "split": args.corpus_split,
                        "limit_examples": args.corpus_limit_examples,
                        "seed": int(args.seed),
                        "shuffle": bool(int(args.corpus_shuffle_buffer) > 0),
                        "shuffle_buffer_size": int(args.corpus_shuffle_buffer),
                        "dedupe": bool(args.corpus_dedup),
                        "dedupe_window": int(args.corpus_dedup_max),
                        "min_chars": int(args.corpus_min_chars),
                        "drop_empty": bool(args.corpus_drop_empty),
                    },

                },
                indent=2,
            )
        )

        trainer.train(activations, skip_steps=resume_step)
        
        # Log and save padding statistics
        if pad_stats_holder:
            log.info(
                json.dumps(
                    {
                        "event": "pad_statistics",
                        "total_tokens_seen": pad_stats_holder.get("total_tokens_seen", 0),
                        "total_pad_tokens": pad_stats_holder.get("total_pad_tokens", 0),
                        "pad_frac": pad_stats_holder.get("pad_frac", 0.0),
                        "token_filter": pad_stats_holder.get("token_filter", "unknown"),
                        "pad_tokens_filtered": args.token_filter == "nonpad",
                    },
                    indent=2,
                )
            )
            # Save pad stats to file
            (run_dir / "pad_statistics.json").write_text(
                json.dumps(pad_stats_holder, indent=2), encoding="utf-8"
            )

        # Optional: perplexity impact
        if args.eval_ppl and eval_texts:
            ppl_base = eval_ppl(
                model,
                eval_texts[: int(args.ppl_eval_examples)],
                seq_len=int(args.seq_len),
                batch_size=max(1, int(args.batch_size)),
            )
            ppl_report: Dict[str, object] = {"ppl_base": ppl_base, "by_sae": {}}

            if len(saes) == 1:
                sae_final = torch.load(run_dir / "sae_final.pt", map_location=model.model.device)
                sae_eval = SparseAutoencoder(SAEConfig(**sae_final["sae_cfg"]))
                sae_eval.load_state_dict(sae_final["state_dict"])
                ppl_recon = eval_ppl_with_reconstruction(
                    model,
                    sae_eval,
                    layer_idx=int(layer_idx),
                    eval_texts=eval_texts[: int(args.ppl_eval_examples)],
                    seq_len=int(args.seq_len),
                    batch_size=max(1, int(args.batch_size)),
                )
                ppl_report["by_sae"] = {"single": {"ppl_recon": ppl_recon, "delta": ppl_recon - ppl_base}}
            else:
                by: Dict[str, dict] = {}
                for key in saes.keys():
                    ckpt_path = run_dir / key / "sae_final.pt"
                    if not ckpt_path.exists():
                        continue
                    sae_final = torch.load(ckpt_path, map_location=model.model.device)
                    sae_eval = SparseAutoencoder(SAEConfig(**sae_final["sae_cfg"]))
                    sae_eval.load_state_dict(sae_final["state_dict"])
                    ppl_recon = eval_ppl_with_reconstruction(
                        model,
                        sae_eval,
                        layer_idx=int(layer_idx),
                        eval_texts=eval_texts[: int(args.ppl_eval_examples)],
                        seq_len=int(args.seq_len),
                        batch_size=max(1, int(args.batch_size)),
                    )
                    by[key] = {"ppl_recon": ppl_recon, "delta": ppl_recon - ppl_base}
                ppl_report["by_sae"] = by

            (run_dir / "ppl.json").write_text(json.dumps(ppl_report, indent=2), encoding="utf-8")
            log.info(f"Perplexity base={ppl_base:.3f} (see ppl.json for recon deltas)")

        # Optional: interpretability summary via top activating contexts
        if args.interpret and eval_texts:
            g = torch.Generator(device="cpu").manual_seed(int(args.seed) + 1234)
            feat_ids = torch.randperm(d_sae, generator=g)[: int(args.interpret_features)].tolist()
            collector = FeatureTopKCollector(feature_ids=feat_ids, k=int(args.interpret_topk), tokenizer=model.tokenizer, window=20)

            tok_cfg = TokenizeConfig(seq_len=args.seq_len, random_crop=False)

            # Interpret the *primary* SAE for this run.
            if len(saes) == 1:
                sae = next(iter(saes.values())).to(model.model.device).to(model_param_dtype).eval()
            else:
                first_key = sorted(saes.keys())[0]
                ckpt = torch.load(run_dir / first_key / "sae_final.pt", map_location=model.model.device)
                sae = SparseAutoencoder(SAEConfig(**ckpt["sae_cfg"]))
                sae.load_state_dict(ckpt["state_dict"])
                sae = sae.to(model.model.device).to(model_param_dtype).eval()

            for t in tqdm(eval_texts[: int(args.interpret_examples)], desc="interpret", dynamic_ncols=True):
                batch = tokenize_batch(model.tokenizer, [t], tok_cfg)
                input_ids = batch["input_ids"].to(model.model.device)
                attn = batch.get("attention_mask", None)
                if attn is not None:
                    attn = attn.to(model.model.device)

                with torch.no_grad():
                    h = _extract_layer_activations(
                        model=model,
                        layer_idx=int(layer_idx),
                        input_ids=input_ids,
                        attention_mask=attn,
                        capture=str(args.activation_capture),
                    )
                B, T, D = h.shape
                x = h.reshape(B * T, D)

                z = sae.encode(x).reshape(B, T, -1).detach().cpu()

                # Mask pad tokens so they cannot win top-k.
                if attn is not None:
                    m = attn.detach().cpu().to(dtype=torch.bool).unsqueeze(-1)
                    z = z * m

                collector.update(z=z, input_ids=input_ids.detach().cpu(), meta={"source": "wikitext2"})

            topk_json = collector.to_jsonable()
            (run_dir / "feature_topk.json").write_text(json.dumps(topk_json, indent=2), encoding="utf-8")

            labels: Dict[str, dict] = {}
            for fid in feat_ids:
                contexts = [it["text_window"] for it in topk_json.get(str(fid), [])]
                lab = heuristic_label_from_contexts(feature_id=int(fid), contexts=contexts)
                labels[str(fid)] = {"label": lab.label, "top_ngrams": lab.top_ngrams}
            (run_dir / "feature_labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
        
        # Final evaluation pass on heldout data
        if args.final_eval:
            log.info("Running final evaluation pass (writes eval_summary.json)...")
            eval_summary: Dict[str, Any] = {
                "event": "eval_summary",
                "eval_tokens": int(args.final_eval_tokens),
                "by_sae": {},
            }
            
            if len(saes) == 1:
                sae_final = torch.load(run_dir / "sae_final.pt", map_location=model.model.device)
                sae_eval = SparseAutoencoder(SAEConfig(**sae_final["sae_cfg"]))
                sae_eval.load_state_dict(sae_final["state_dict"])
                
                summary = compute_final_eval_summary(
                    model=model,
                    sae=sae_eval,
                    layer_idx=int(layer_idx),
                    eval_tokens=int(args.final_eval_tokens),
                    seq_len=int(args.seq_len),
                    batch_size=max(1, int(args.batch_size)),
                    device=resolved_device,
                    activation_capture=str(args.activation_capture),
                    activation_dtype=act_dtype,
                )
                eval_summary["by_sae"]["single"] = summary
            else:
                for key in saes.keys():
                    ckpt_path = run_dir / key / "sae_final.pt"
                    if not ckpt_path.exists():
                        continue
                    sae_final = torch.load(ckpt_path, map_location=model.model.device)
                    sae_eval = SparseAutoencoder(SAEConfig(**sae_final["sae_cfg"]))
                    sae_eval.load_state_dict(sae_final["state_dict"])
                    
                    summary = compute_final_eval_summary(
                        model=model,
                        sae=sae_eval,
                        layer_idx=int(layer_idx),
                        eval_tokens=int(args.final_eval_tokens),
                        seq_len=int(args.seq_len),
                        batch_size=max(1, int(args.batch_size)),
                        device=resolved_device,
                        activation_capture=str(args.activation_capture),
                        activation_dtype=act_dtype,
                    )
                    eval_summary["by_sae"][key] = summary
            
            (run_dir / "eval_summary.json").write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
            log.info(f"Final evaluation saved to eval_summary.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
