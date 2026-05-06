#!/usr/bin/env python3
"""norm_baseline.py.

Computes the activation-norm AUROC at the best layer per model and produces
the GPT-family vs LLaMA-family norm-direction split.

Used in Section Results and Table tab:norm_direction of the paper.
Reproduce: env/bin/python3 scripts/dark_subspace/norm_baseline.py \
    --model-path <ft_model_dir> \
    --member-texts data/memcirc_ctrl_ft/member.jsonl \
    --nonmember-texts data/memcirc_ctrl_ft/nonmember.jsonl \
    --layers 8 16 24 \
    --output-dir runs/dark_subspace/norm_baseline/<run_tag>
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
    _HAS_PROJECT_INFRA = True
except ImportError as e:
    _HAS_PROJECT_INFRA = False
    _IMPORT_ERROR = str(e)

try:
    from sklearn.metrics import roc_auc_score
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


def _sanitize_for_json(obj):
    """Recursively replace non-finite floats with None for JSON serialization."""
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
# Norm collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_norms(
    model,
    tokenizer,
    texts: List[str],
    layers: List[int],
    seq_len: int,
    batch_size: int,
    device: str,
) -> Dict[int, np.ndarray]:
    """Collect per-text L2 norms of mean-pooled hidden states at each layer.

    Performs a forward pass with output_hidden_states=True, extracts hidden
    states at each requested layer, mean-pools with the attention mask, then
    computes the L2 norm of the pooled vector.

    Args:
        model: HuggingFace causal LM (or CausalLMWrapper).
        tokenizer: Corresponding tokenizer.
        texts: List of text strings.
        layers: Layer indices to extract (0 = embedding layer).
        seq_len: Maximum sequence length for tokenization.
        batch_size: Processing batch size.
        device: Device string (e.g. "cuda", "cpu").

    Returns:
        Dict mapping layer index -> np.ndarray of shape (n_texts,) containing
        the L2 norm of the mean-pooled hidden state for each text.
    """
    # Detect if model is a CausalLMWrapper or a raw HF model
    has_wrapper = hasattr(model, "forward") and hasattr(model, "tokenizer")

    layer_norms: Dict[int, List[float]] = {l: [] for l in layers}

    for chunk in tqdm(
        list(_batched(texts, batch_size)),
        desc="Collecting activation norms",
        dynamic_ncols=True,
    ):
        # Tokenize
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=seq_len,
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(device)

        # Forward pass
        if has_wrapper:
            out = model.forward(
                input_ids=input_ids,
                attention_mask=attn,
                output_hidden_states=True,
            )
        else:
            out = model(
                input_ids=input_ids,
                attention_mask=attn,
                output_hidden_states=True,
            )

        # Extract norms at each requested layer
        for layer_idx in layers:
            h = out.hidden_states[layer_idx]  # (B, T, d_model)
            if attn is not None:
                mask = attn.unsqueeze(-1).float()  # (B, T, 1)
                pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = h.mean(dim=1)  # (B, d_model)
            norms = pooled.norm(dim=-1)  # (B,)
            layer_norms[layer_idx].extend(norms.cpu().float().tolist())

    return {l: np.array(v) for l, v in layer_norms.items()}


# ---------------------------------------------------------------------------
# AUROC computation
# ---------------------------------------------------------------------------

def compute_norm_aurocs(
    layer_norms: Dict[int, np.ndarray],
    labels: np.ndarray,
) -> Dict[str, dict]:
    """Compute bidirectional norm AUROC for each layer.

    For each layer, tries both score orientations (norms and -norms) and
    records the higher AUROC along with descriptive statistics.

    Args:
        layer_norms: Dict mapping layer index -> (n_texts,) array of L2 norms.
        labels: (n_texts,) binary array (1=member, 0=nonmember).

    Returns:
        Dict mapping str(layer_index) -> {
            auroc, direction, mean_member_norm, mean_nonmember_norm,
            std_member_norm, std_nonmember_norm
        }
    """
    if not _HAS_SKLEARN:
        raise RuntimeError("sklearn is required for AUROC computation")

    labels = np.asarray(labels, dtype=int)
    member_mask = labels == 1
    nonmember_mask = labels == 0

    results: Dict[str, dict] = {}
    for layer_idx, norms in layer_norms.items():
        norms = np.asarray(norms, dtype=float)

        auroc_pos = float(roc_auc_score(labels, norms))
        auroc_neg = float(roc_auc_score(labels, -norms))

        if auroc_neg > auroc_pos:
            auroc = auroc_neg
            direction = "negative"
        else:
            auroc = auroc_pos
            direction = "positive"

        results[str(layer_idx)] = {
            "auroc": auroc,
            "direction": direction,
            "mean_member_norm": float(norms[member_mask].mean()) if member_mask.any() else None,
            "mean_nonmember_norm": float(norms[nonmember_mask].mean()) if nonmember_mask.any() else None,
            "std_member_norm": float(norms[member_mask].std()) if member_mask.any() else None,
            "std_nonmember_norm": float(norms[nonmember_mask].std()) if nonmember_mask.any() else None,
        }

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compute AUROC(||h_l||, membership_label) per layer. "
            "Confound check: is BCD just capturing activation norm differences?"
        )
    )
    p.add_argument("--model-path", required=True, help="Path to FT model checkpoint")
    p.add_argument("--member-texts", required=True, help="Path to member.jsonl")
    p.add_argument("--nonmember-texts", required=True, help="Path to nonmember.jsonl")
    p.add_argument(
        "--layers", required=True, nargs="+", type=int,
        help="Space-separated layer indices to evaluate",
    )
    p.add_argument("--output-dir", required=True, help="Output directory for results")
    p.add_argument("--seq-len", type=int, default=256, help="Max sequence length (default: 256)")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--device", default="cuda", help="Device: cuda or cpu (default: cuda)")
    p.add_argument(
        "--max-texts", type=int, default=0,
        help="Max texts per split (0 = all, default: 0)",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Infrastructure checks
    if not _HAS_PROJECT_INFRA:
        log.warning(
            "Project infrastructure not available (%s). "
            "Proceeding with reduced functionality.",
            _IMPORT_ERROR,
        )
    if not _HAS_SKLEARN:
        raise SystemExit("ERROR: sklearn is required. Install scikit-learn.")

    # Logging + seed
    if _HAS_PROJECT_INFRA:
        setup_logging()
        set_global_seed(SeedConfig(seed=args.seed))
    else:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    t0 = time.time()
    config = {
        "script": "scripts/dark_subspace/norm_baseline.py",
        "model_path": args.model_path,
        "member_texts": args.member_texts,
        "nonmember_texts": args.nonmember_texts,
        "layers": args.layers,
        "output_dir": args.output_dir,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device,
        "max_texts": args.max_texts,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    log.info("Config saved to %s/config.json", out_dir)

    # Resolve device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available; falling back to CPU.")
        device = "cpu"

    # Load model
    log.info("Loading model from %s", args.model_path)
    if _HAS_PROJECT_INFRA:
        spec = HFModelSpec(name_or_path=args.model_path)
        wrapper = load_model_and_tokenizer(spec)
        model = wrapper
        tokenizer = wrapper.tokenizer
        wrapper.model.to(device)
        wrapper.model.eval()
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path)
        model.to(device)
        model.eval()

    # Load texts
    max_n = args.max_texts if args.max_texts > 0 else None
    log.info("Loading member texts from %s (max_n=%s)", args.member_texts, max_n)
    member_texts = _load_texts(args.member_texts, max_n=max_n)
    log.info("Loading nonmember texts from %s (max_n=%s)", args.nonmember_texts, max_n)
    nonmember_texts = _load_texts(args.nonmember_texts, max_n=max_n)

    n_mem = len(member_texts)
    n_nonmem = len(nonmember_texts)
    log.info("Loaded %d member texts, %d nonmember texts", n_mem, n_nonmem)

    all_texts = member_texts + nonmember_texts
    labels = np.array([1] * n_mem + [0] * n_nonmem, dtype=int)

    # Collect norms
    log.info("Collecting activation norms for layers %s", args.layers)
    layer_norms = collect_norms(
        model=model,
        tokenizer=tokenizer,
        texts=all_texts,
        layers=args.layers,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
    )

    # Compute AUROCs
    log.info("Computing per-layer norm AUROCs")
    per_layer = compute_norm_aurocs(layer_norms, labels)

    # Identify best layer
    best_layer = max(per_layer, key=lambda k: per_layer[k]["auroc"])
    best_auroc = per_layer[best_layer]["auroc"]
    log.info(
        "Best layer: %s (AUROC=%.4f, direction=%s)",
        best_layer, best_auroc, per_layer[best_layer]["direction"],
    )

    # Assemble results
    results = {
        "per_layer": per_layer,
        "best_layer": int(best_layer),
        "best_auroc": best_auroc,
        "n_member": n_mem,
        "n_nonmember": n_nonmem,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    results = _sanitize_for_json(results)

    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
