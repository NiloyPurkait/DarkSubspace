#!/usr/bin/env python3
"""random_direction_baseline.py.

Fixed-orientation random-direction baseline. Samples 100 random unit
directions per model and reports the membership AUROC distribution for the
any-direction null. If random directions sit at chance and the residual
probe sits well above chance, the geometric signal is not generic.

Used in the additional controls appendix.
Reproduce:
    env/bin/python3 scripts/memcirc/random_direction_baseline.py \\
        --model-path runs/controlled_ft/<run>/ft_epoch5/model \\
        --member-texts data/memcirc_ctrl_ft/member.jsonl \\
        --nonmember-texts data/memcirc_ctrl_ft/nonmember.jsonl \\
        --layer 16 \\
        --output-path runs/memcirc/random_direction_baseline/p69.json \\
        --n-directions 100 --seed 42
"""

import _bootstrap  # noqa: F401

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def load_jsonl(p):
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def extract_activations(model, tokenizer, texts, layer, device, seq_len=256, batch_size=4):
    """Extract mean-pooled activations at the given layer for each text."""
    model.eval()
    activations = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=seq_len)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            out = model(input_ids, attention_mask=attn, output_hidden_states=True)
            h = out.hidden_states[layer]  # [B, T, d]
            # Mean-pool over non-pad tokens
            mask = attn.unsqueeze(-1).float()
            pooled = (h * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
            activations.append(pooled.cpu().float().numpy())
    return np.concatenate(activations, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--member-texts", required=True)
    p.add_argument("--nonmember-texts", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--n-directions", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading model from {args.model_path}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float32).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading member/nonmember texts")
    members = [r.get("text", r.get("content", "")) for r in load_jsonl(args.member_texts)]
    nonmembers = [r.get("text", r.get("content", "")) for r in load_jsonl(args.nonmember_texts)]
    print(f"  n_members = {len(members)}")
    print(f"  n_nonmembers = {len(nonmembers)}")

    print(f"Extracting activations at layer {args.layer}")
    h_mem = extract_activations(model, tokenizer, members, args.layer, args.device, args.seq_len, args.batch_size)
    h_non = extract_activations(model, tokenizer, nonmembers, args.layer, args.device, args.seq_len, args.batch_size)
    print(f"  h_mem shape: {h_mem.shape}")
    print(f"  h_non shape: {h_non.shape}")

    d_model = h_mem.shape[1]

    # Free model memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Sampling {args.n_directions} random unit directions in R^{d_model}")
    rng = np.random.default_rng(args.seed)
    aurocs = []
    aurocs_bidir = []
    for i in range(args.n_directions):
        v = rng.normal(0.0, 1.0, size=d_model).astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-9)
        scores_mem = h_mem @ v
        scores_non = h_non @ v
        labels = np.concatenate([np.ones(len(scores_mem)), np.zeros(len(scores_non))])
        scores = np.concatenate([scores_mem, scores_non])
        a = roc_auc_score(labels, scores)
        aurocs.append(a)
        aurocs_bidir.append(max(a, 1 - a))

    aurocs = np.array(aurocs)
    aurocs_bidir = np.array(aurocs_bidir)

    result = {
        "model_path": args.model_path,
        "layer": args.layer,
        "d_model": d_model,
        "n_member": len(members),
        "n_nonmember": len(nonmembers),
        "n_directions": args.n_directions,
        "seed": args.seed,
        "fixed_orientation": {
            "mean": float(aurocs.mean()),
            "std": float(aurocs.std()),
            "min": float(aurocs.min()),
            "max": float(aurocs.max()),
            "p05": float(np.percentile(aurocs, 5)),
            "p50": float(np.percentile(aurocs, 50)),
            "p95": float(np.percentile(aurocs, 95)),
            "p99": float(np.percentile(aurocs, 99)),
        },
        "bidirectional": {
            "mean": float(aurocs_bidir.mean()),
            "std": float(aurocs_bidir.std()),
            "min": float(aurocs_bidir.min()),
            "max": float(aurocs_bidir.max()),
            "p50": float(np.percentile(aurocs_bidir, 50)),
            "p95": float(np.percentile(aurocs_bidir, 95)),
            "p99": float(np.percentile(aurocs_bidir, 99)),
        },
        "all_aurocs": aurocs.tolist(),
    }

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out}")
    print(f"  Fixed-orientation AUROC mean = {aurocs.mean():.4f} (std {aurocs.std():.4f})")
    print(f"  Fixed-orientation 95th pct   = {np.percentile(aurocs, 95):.4f}")
    print(f"  Fixed-orientation max        = {aurocs.max():.4f}")
    print(f"  Bidirectional 95th pct       = {np.percentile(aurocs_bidir, 95):.4f}")


if __name__ == "__main__":
    main()
