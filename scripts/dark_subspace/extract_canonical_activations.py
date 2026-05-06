#!/usr/bin/env python3
"""
extract_canonical_activations.py.

Extracts the canonical mean-pooled residual stream activations at the analysis
layer for each evaluation document, and serialises them so downstream
ablation, low-FPR, and baseline-attack scripts read identical inputs.

Used in infrastructure (held-out partition probe and downstream ablation
scripts depend on these activations).
Reproduce:
    env/bin/python3 scripts/dark_subspace/extract_canonical_activations.py \\
        --model-path runs/controlled_ft/run_20260306_055225/ft_epoch5/model \\
        --member-texts data/memcirc_ctrl_ft/member.jsonl \\
        --nonmember-texts data/memcirc_ctrl_ft/nonmember.jsonl \\
        --layer 16 --seq-len 256 --batch-size 8 --seed 42 \\
        --out-prefix runs/dark_subspace/activations_canonical/p69_epoch5_layer16

Loads the same texts and runs the same forward pass and pooling as
``sae_dark_subspace.py`` (canonical N=1000 members and 1000 nonmembers at
seq_len=256, mean pool with attention mask). Saves to disk so the held-out
d_K probe can re-fit on a 70 percent partition and evaluate on a disjoint 30
percent split.

Outputs (relative to --out-prefix).
    {prefix}_member.npy    shape (N_member, d_model), float32.
    {prefix}_nonmember.npy shape (N_nonmember, d_model), float32.
    {prefix}_meta.json     metadata (model, layer, seed, sampling, sha).
"""

import _bootstrap  # noqa: F401

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from sae_mia_audit.models.wrapper import load_model_and_tokenizer
from sae_mia_audit.utils.hf import HFModelSpec
from sae_mia_audit.utils.seed import SeedConfig, set_global_seed
from sae_mia_audit.utils.logging import setup_logging, get_logger


log = get_logger(__name__)


def _load_texts(path: str, max_n: Optional[int] = None) -> List[str]:
    """Load texts from a jsonl file (matches sae_dark_subspace.py:_load_texts)."""
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


@torch.no_grad()
def collect_activations(model, tokenizer, texts, layer, seq_len, batch_size, device):
    """Mean-pool hidden states at a layer (canonical to sae_dark_subspace.py)."""
    all_acts = []
    for i in tqdm(range(0, len(texts), batch_size), desc="extract"):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=seq_len, padding=True,
        ).to(device)
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[layer]  # (B, T, D)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        all_acts.append(pooled.cpu().float().numpy())
    return np.concatenate(all_acts, axis=0)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(
        description="Canonical activation extractor for the held-out partition probe."
    )
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--member-texts", required=True)
    ap.add_argument("--nonmember-texts", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-texts", type=int, default=0,
                    help="Cap each member/nonmember count (0 = use all jsonl rows)")
    ap.add_argument("--torch-dtype", default="bfloat16",
                    help="Model dtype for forward pass (bfloat16 default).")
    ap.add_argument("--out-prefix", required=True,
                    help="Output basename prefix; writes _member.npy/_nonmember.npy/_meta.json")
    args = ap.parse_args()

    setup_logging(logging.INFO)
    set_global_seed(SeedConfig(seed=args.seed))

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading model from {args.model_path} (dtype={args.torch_dtype})")
    spec = HFModelSpec(name_or_path=args.model_path, torch_dtype=args.torch_dtype)
    wrapper = load_model_and_tokenizer(spec)
    model = wrapper.model.to(args.device).eval()
    tokenizer = wrapper.tokenizer

    max_n = args.max_texts if args.max_texts > 0 else None
    member_texts = _load_texts(args.member_texts, max_n)
    nonmember_texts = _load_texts(args.nonmember_texts, max_n)
    log.info(f"Loaded {len(member_texts)} member, {len(nonmember_texts)} nonmember texts")

    t0 = time.time()
    member_acts = collect_activations(
        model, tokenizer, member_texts, args.layer,
        args.seq_len, args.batch_size, args.device,
    )
    nonmember_acts = collect_activations(
        model, tokenizer, nonmember_texts, args.layer,
        args.seq_len, args.batch_size, args.device,
    )
    elapsed = time.time() - t0

    member_path = str(out_prefix) + "_member.npy"
    nonmember_path = str(out_prefix) + "_nonmember.npy"
    np.save(member_path, member_acts.astype(np.float32))
    np.save(nonmember_path, nonmember_acts.astype(np.float32))

    meta = {
        "model_path": args.model_path,
        "layer": args.layer,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "torch_dtype": args.torch_dtype,
        "n_member": int(member_acts.shape[0]),
        "n_nonmember": int(nonmember_acts.shape[0]),
        "d_model": int(member_acts.shape[1]),
        "member_texts_path": args.member_texts,
        "nonmember_texts_path": args.nonmember_texts,
        "member_acts_sha256": _sha256(member_path),
        "nonmember_acts_sha256": _sha256(nonmember_path),
        "elapsed_sec": float(elapsed),
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "script": "scripts/dark_subspace/extract_canonical_activations.py",
    }
    meta_path = str(out_prefix) + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Wrote {member_path} shape={member_acts.shape}")
    log.info(f"Wrote {nonmember_path} shape={nonmember_acts.shape}")
    log.info(f"Wrote {meta_path}")
    log.info(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
