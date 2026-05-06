#!/usr/bin/env python3
"""build_disjoint_owt_corpus.py.

Builds the 2000-document SHA-256 disjoint OpenWebText partition and writes
data/memcirc_ctrl_disjoint/disjointness_proof.json with intersection_count=0.

Used in the disjoint-corpus appendix.
Reproduce:
    env/bin/python3 scripts/memcirc/build_disjoint_owt_corpus.py \
        --member data/memcirc_ctrl_ft/member.jsonl \
        --nonmember data/memcirc_ctrl_ft/nonmember.jsonl \
        --output data/memcirc_ctrl_disjoint/mixed_disjoint.jsonl \
        --proof data/memcirc_ctrl_disjoint/disjointness_proof.json \
        --n-docs 2000 \
        --seed 100

Specification.
- Same length distribution (mean 889 chars, std 84) and min-chars filter
  (corpus_min_chars = 50) as the original eval pool.
- 2000 documents, one JSON object per line ({"text": "..."}).
- Disjointness verification (set arithmetic, SHA-256 hash) saved to
  data/memcirc_ctrl_disjoint/disjointness_proof.json.

Strategy.
- Stream OpenWebText via HuggingFace datasets (the same loader used by
  scripts/shared/train_sae.py corpus path).
- Hash every doc and skip any hash present in the eval pool.
- Use a different rng seed than the eval pool's subsample_seed = 43 to
  reduce expected collisions to near zero. Default is seed 100.
- Sample more than 2000 candidates, filter by length window, then trim.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def load_jsonl_texts(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            t = obj.get("text", "")
            if t:
                yield t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--member", required=True)
    p.add_argument("--nonmember", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--proof", required=True)
    p.add_argument("--n-docs", type=int, default=2000)
    p.add_argument("--min-chars", type=int, default=50)
    p.add_argument("--target-min-len", type=int, default=620,
                   help="Eval pool min length 620 chars (split_metadata.json).")
    p.add_argument("--target-max-len", type=int, default=1024,
                   help="Eval pool max length 1024 chars.")
    p.add_argument("--candidate-multiplier", type=int, default=10,
                   help="Stream candidate-multiplier * n-docs candidates before filtering.")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--owt-name", default="openwebtext")
    p.add_argument("--owt-split", default="train")
    args = p.parse_args()

    rng = random.Random(args.seed)

    eval_hashes = set()
    for t in load_jsonl_texts(Path(args.member)):
        eval_hashes.add(text_sha(t))
    n_member = len(eval_hashes)
    for t in load_jsonl_texts(Path(args.nonmember)):
        eval_hashes.add(text_sha(t))
    n_union = len(eval_hashes)
    print(f"[disjoint-corpus] eval pool union hash count: {n_union} (member only: {n_member})")

    # Stream OWT.
    from datasets import load_dataset
    print(f"[disjoint-corpus] loading {args.owt_name} streaming, split={args.owt_split}, seed={args.seed}")
    ds = load_dataset(args.owt_name, split=args.owt_split, streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=10_000)

    selected = []
    seen_hashes = set()
    n_collisions = 0
    n_skipped_len = 0
    target_candidates = args.candidate_multiplier * args.n_docs
    n_streamed = 0
    for ex in ds:
        n_streamed += 1
        text = ex.get("text", "")
        if not text:
            continue
        if len(text) < args.min_chars:
            continue
        # Length window matched to eval pool stats.
        if len(text) < args.target_min_len or len(text) > args.target_max_len:
            n_skipped_len += 1
            if len(selected) >= args.n_docs and n_streamed > target_candidates:
                break
            continue
        h = text_sha(text)
        if h in eval_hashes:
            n_collisions += 1
            continue
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        selected.append({"text": text, "_sha256": h})
        if len(selected) >= args.n_docs:
            break

    print(f"[disjoint-corpus] streamed {n_streamed} docs, "
          f"{len(selected)} kept, {n_collisions} collisions vs eval pool, "
          f"{n_skipped_len} length-window rejects.")

    if len(selected) < args.n_docs:
        raise RuntimeError(
            f"Only collected {len(selected)} disjoint docs, expected {args.n_docs}. "
            f"Increase --candidate-multiplier or relax length window."
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proof_path = Path(args.proof)
    proof_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in selected:
            f.write(json.dumps({"text": rec["text"]}) + "\n")

    selected_hashes = [rec["_sha256"] for rec in selected]
    intersection = set(selected_hashes) & eval_hashes
    proof = {
        "schema_version": 1,
        "n_disjoint_docs": len(selected),
        "n_member_hashes": n_member,
        "n_eval_union_hashes": n_union,
        "intersection_count": len(intersection),
        "intersection_examples": list(intersection)[:10],
        "candidate_streamed": n_streamed,
        "collisions_skipped": n_collisions,
        "length_window_skipped": n_skipped_len,
        "seed": args.seed,
        "owt_name": args.owt_name,
        "owt_split": args.owt_split,
        "min_chars": args.min_chars,
        "target_min_len": args.target_min_len,
        "target_max_len": args.target_max_len,
        "output_path": str(out_path),
    }
    with proof_path.open("w", encoding="utf-8") as f:
        json.dump(proof, f, indent=2)

    print(f"[disjoint-corpus] wrote {len(selected)} docs to {out_path}")
    print(f"[disjoint-corpus] disjointness proof at {proof_path}: intersection={len(intersection)}")
    if len(intersection) > 0:
        raise RuntimeError(
            f"Disjointness violated: {len(intersection)} hashes overlap with eval pool. "
            f"This must be zero. Aborting."
        )


if __name__ == "__main__":
    main()
