#!/usr/bin/env python
"""find_latest_sae.py.

Finds the most recent ``sae_final.pt`` checkpoint matching a (model, layer)
hyperparameter glob, with optional sweep-key selection by perplexity.

Used by SAE-loading scripts (``sae_dark_subspace.py``, ``subspace_ablation_eval.py``)
to resolve a checkpoint path without hard-coding run directories.

Reproduce::

    .venv/bin/python scripts/shared/find_latest_sae.py \\
        --runs-dir runs/sae --model EleutherAI/pythia-6.9b --layer 16
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Optional


def _latest_run_by_name(candidates: list[str]) -> str:
    """Return the lexicographically latest run name (run dirs include timestamps)."""
    return sorted(candidates)[-1]


def _choose_best_sweep_key_by_ppl(run_dir: Path) -> Optional[str]:
    """Pick the sweep key whose checkpoint minimises absolute perplexity delta."""
    ppl_path = run_dir / "ppl.json"
    if not ppl_path.exists():
        return None
    try:
        data = json.loads(ppl_path.read_text(encoding="utf-8"))
        by = data.get("by_sae", {})
        best_key = None
        best_score = None
        for k, v in by.items():
            delta = v.get("delta", None)
            if delta is None:
                continue
            score = abs(float(delta))
            if best_score is None or score < best_score:
                best_score = score
                best_key = k
        return best_key
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=str, default="runs/sae")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument(
        "--sae-key",
        type=str,
        default=None,
        help="Optional subdir name for sweep runs (e.g., 'l1_1.00e-03'). If not set, selection depends on --sweep-select.",
    )
    ap.add_argument(
        "--sweep-select",
        type=str,
        choices=["first", "best_ppl"],
        default="first",
        help="When --sae-key is not provided and the latest run is a sweep: choose first checkpoint (legacy) or best by ppl.json.",
    )
    args = ap.parse_args()

    pat = f"{args.runs_dir}/train_sae__{args.model.replace('/', '_')}__layer{args.layer}__*"
    candidates = sorted(glob.glob(pat))
    if not candidates:
        raise SystemExit(f"No SAE runs found matching: {pat}")
    latest = _latest_run_by_name(candidates)
    latest_p = Path(latest)

    # Single-SAE run
    ckpt = latest_p / "sae_final.pt"
    if ckpt.exists():
        print(str(ckpt))
        return 0

    # Sweep run: checkpoints live in subdirectories
    if args.sae_key is not None:
        ckpt = latest_p / args.sae_key / "sae_final.pt"
        if not ckpt.exists():
            raise SystemExit(f"Missing checkpoint for --sae-key={args.sae_key}: {ckpt}")
        print(str(ckpt))
        return 0

    # Legacy behaviour: pick the first found sweep checkpoint
    if args.sweep_select == "first":
        candidates2 = sorted(glob.glob(str(latest_p / "*" / "sae_final.pt")))
        if not candidates2:
            raise SystemExit(f"No sweep checkpoints found under: {latest_p}")
        print(candidates2[0])
        return 0

    # best_ppl: choose the sweep key that minimally changes perplexity (|delta|)
    best_key = _choose_best_sweep_key_by_ppl(latest_p)
    if best_key is None:
        raise SystemExit(
            f"--sweep-select best_ppl requires {latest_p/'ppl.json'} with per-SAE deltas. "
            "Run scripts/shared/train_sae.py with --eval-ppl to generate it, or use --sweep-select first."
        )

    ckpt = latest_p / best_key / "sae_final.pt"
    if not ckpt.exists():
        raise SystemExit(f"Missing checkpoint for selected key={best_key}: {ckpt}")
    print(str(ckpt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
