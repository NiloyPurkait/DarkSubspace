#!/usr/bin/env python3
"""make_random_sae.py.

Constructs a random-init SAE of matching shape on Pythia-6.9B layer 16 and
reports the dark-subspace evaluation against it. Rules out reconstruction
shape only operators as the source of the effect.

Used in the additional controls appendix.
Reproduce:
    .venv/bin/python scripts/dark_subspace/make_random_sae.py \\
        --reference-sae runs/sae/<trained_sae>/sae_final.pt \\
        --output-path runs/sae/random_init_p69_layer16/sae_final.pt \\
        --seed 42
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn.init as init


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference-sae", required=True, help="Path to a trained SAE checkpoint to copy shape from")
    p.add_argument("--output-path", required=True, help="Where to write the random-init SAE checkpoint")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=["gaussian", "kaiming"], default="kaiming",
                   help="gaussian: N(0, 1/sqrt(d_model)). kaiming: torch.nn.init.kaiming_uniform_.")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    print(f"Loading reference SAE from {args.reference_sae}")
    ckpt = torch.load(args.reference_sae, map_location="cpu", weights_only=False)
    sae_cfg = ckpt["sae_cfg"]
    state = ckpt["state_dict"]

    d_model = sae_cfg["d_model"]
    d_sae = sae_cfg["d_sae"]
    print(f"  d_model = {d_model}")
    print(f"  d_sae   = {d_sae}")

    new_state = {}

    # encoder.weight: [d_sae, d_model]
    enc_w = torch.empty(d_sae, d_model)
    if args.mode == "kaiming":
        init.kaiming_uniform_(enc_w, a=math.sqrt(5))
    else:
        init.normal_(enc_w, mean=0.0, std=1.0 / math.sqrt(d_model))
    new_state["encoder.weight"] = enc_w

    # encoder.bias: [d_sae]
    new_state["encoder.bias"] = torch.zeros(d_sae)

    # decoder.weight: [d_model, d_sae]
    dec_w = torch.empty(d_model, d_sae)
    if args.mode == "kaiming":
        init.kaiming_uniform_(dec_w, a=math.sqrt(5))
    else:
        init.normal_(dec_w, mean=0.0, std=1.0 / math.sqrt(d_sae))
    # Normalize decoder columns to unit norm (matches trained SAE convention)
    if sae_cfg.get("normalize_decoder", True):
        dec_w = dec_w / (dec_w.norm(dim=0, keepdim=True) + 1e-8)
    new_state["decoder.weight"] = dec_w

    # decoder.bias: [d_model]
    new_state["decoder.bias"] = torch.zeros(d_model)

    # Build new checkpoint
    new_ckpt = {
        "step": 0,
        "sae_cfg": sae_cfg,  # reuse config (l1, etc. don't matter for inference)
        "train_cfg": ckpt.get("train_cfg", {}),
        "state_dict": new_state,
        "opt_state": {},
        "total_resampled": 0,
        "_random_init": True,
        "_random_init_mode": args.mode,
        "_random_init_seed": args.seed,
        "_reference_sae": args.reference_sae,
    }

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_ckpt, out)
    print(f"Wrote random-init SAE to {out}")
    print(f"  init mode: {args.mode}")
    print(f"  shapes:")
    for k, v in new_state.items():
        print(f"    {k}: {tuple(v.shape)}")


if __name__ == "__main__":
    main()
