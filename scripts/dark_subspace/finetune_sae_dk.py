#!/usr/bin/env python3
"""finetune_sae_dk.py.

Fine-tunes a pre-trained mixed-data SAE with the additional lambda_d_K
projection penalty term (Equation 1) so the reconstruction explicitly
captures d_K on a held-out audit partition.

Used in Section "Methods" (M:114-127, `sec:methods:sae_recon`) and Section
"Results" (R:122-127), with reviewer concern C12 of the paper.
Reproduce: env/bin/python3 scripts/dark_subspace/finetune_sae_dk.py --sae-path <sae> --model-path <ft_model> --bcd-dir <bcd_dir> --layer <L> --dk-coeff 0.01 --steps 5000 --corpus <mixed.jsonl> --output-dir <out>

Loads a pre-trained SAE and continues training with an additional loss
term that penalises reconstruction error along the memorisation direction
d_K. This is more stable than training from scratch because features are
already learned.
"""
import _bootstrap  # noqa: F401

import argparse
import json
import math
import logging
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from sae_mia_audit.models.wrapper import load_model_and_tokenizer
from sae_mia_audit.utils.hf import HFModelSpec
from sae_mia_audit.utils.seed import SeedConfig, set_global_seed
from sae_mia_audit.utils.logging import setup_logging, get_logger
from sae_mia_audit.sae.io import load_sae_checkpoint_any
from sae_mia_audit.data.sae_corpus import SAECorpusSpec, load_sae_corpus
from sae_mia_audit.data.tokenizer import TokenizeConfig, tokenize_batch

log = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SAE with d_K penalty")
    parser.add_argument("--sae-path", required=True, help="Pre-trained SAE checkpoint")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--bcd-dir", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--dk-coeff", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4, help="Lower LR for fine-tuning")
    parser.add_argument("--tokens-per-step", type=int, default=4096)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    setup_logging(logging.INFO)
    set_global_seed(SeedConfig(seed=args.seed))
    device = args.device
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    config["script"] = "finetune_sae_dk.py"
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

    # Load d_K
    bcd_data = np.load(Path(args.bcd_dir) / "directions.npz", allow_pickle=True)
    dk_key = f"d_K_layer{args.layer}"
    if dk_key not in bcd_data:
        dk_key = "d_K"
    d_K = bcd_data[dk_key]
    d_K_norm = d_K / (np.linalg.norm(d_K) + 1e-12)
    d_K_tensor = torch.tensor(d_K_norm, dtype=torch.float32, device=device)
    log.info(f"Loaded d_K: shape={d_K.shape}")

    # Load pre-trained SAE
    log.info(f"Loading pre-trained SAE from {args.sae_path}")
    sae = load_sae_checkpoint_any(args.sae_path, device=device)
    # Get the underlying module for training
    if hasattr(sae, 'saif'):
        sae_module = sae.saif
    elif hasattr(sae, 'encoder'):
        sae_module = sae
    else:
        sae_module = sae
    log.info(f"SAE loaded: d_model={sae.d_model}, d_sae={sae.d_sae}")

    # Measure BEFORE metrics
    log.info("Computing d_K penalty BEFORE fine-tuning...")

    # Load model
    spec = HFModelSpec(name_or_path=args.model_path, torch_dtype="bfloat16")
    wrapper = load_model_and_tokenizer(spec)
    model = wrapper.model.to(device).eval()
    tokenizer = wrapper.tokenizer

    # Load texts
    texts = list(load_sae_corpus(SAECorpusSpec(
        name=args.corpus, split="train", streaming=False,
        text_field="text", drop_empty=True, min_chars=50,
    )))
    log.info(f"Loaded {len(texts)} texts")

    tok_cfg = TokenizeConfig(seq_len=args.seq_len, random_crop=True)

    # Optimizer with a lower LR for fine-tuning
    trainable_params = [p for p in sae_module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    metrics_path = out_dir / "metrics.jsonl"
    t0 = time.time()
    step = 0
    text_idx = 0

    pbar = tqdm(total=args.steps, desc="Fine-tune SAE + d_K")

    while step < args.steps:
        # Collect a batch of activations
        batch_texts = []
        while len(batch_texts) < args.batch_size:
            batch_texts.append(texts[text_idx % len(texts)])
            text_idx += 1

        batch = tokenize_batch(tokenizer, batch_texts, tok_cfg)
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask,
                       output_hidden_states=True, use_cache=False)
            h = out.hidden_states[args.layer]  # (B, T, D)

        # Flatten to (N, D) for SAE, only non-padding tokens
        tokens_list = []
        for b in range(h.shape[0]):
            n_tok = int(attn_mask[b].sum().item())
            tokens_list.append(h[b, :n_tok])
        x = torch.cat(tokens_list, dim=0).float()  # (N, D)

        if x.shape[0] < 10:
            continue

        # Truncate to tokens_per_step
        x = x[:args.tokens_per_step]

        # Standard SAE forward + loss
        z = sae.encode(x)
        x_hat = sae.decode(z)
        recon_loss = torch.nn.functional.mse_loss(x_hat, x)
        l1_loss = z.abs().sum(dim=-1).mean()

        # d_K penalty: penalize reconstruction error along d_K
        residual = x - x_hat
        dk_error = (residual @ d_K_tensor) ** 2  # (N,)
        dk_penalty = dk_error.mean()

        # Total loss
        total_loss = recon_loss + 0.0001 * l1_loss + args.dk_coeff * dk_penalty

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        # Logging
        if step % args.log_every == 0:
            with torch.no_grad():
                l0 = (z > 0).float().sum(dim=-1).mean().item()
                fvu_num = recon_loss.item()
                x_var = torch.mean((x - x.mean(dim=0)) ** 2).item()
                fvu = fvu_num / max(x_var, 1e-12)

                metrics = {
                    "step": step,
                    "total_loss": float(total_loss.item()),
                    "recon_loss": float(recon_loss.item()),
                    "dk_penalty": float(dk_penalty.item()),
                    "l1_loss": float(l1_loss.item()),
                    "fvu": float(fvu),
                    "l0": float(l0),
                }
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(metrics) + "\n")

                pbar.set_postfix(
                    loss=f"{total_loss.item():.4f}",
                    dk=f"{dk_penalty.item():.4f}",
                    fvu=f"{fvu:.4f}",
                    l0=f"{l0:.0f}",
                )

        step += 1
        pbar.update(1)

    pbar.close()

    # Save fine-tuned SAE
    final_path = out_dir / "sae_final.pt"
    if hasattr(sae_module, 'cfg'):
        torch.save({"sae_cfg": sae_module.cfg.__dict__, "state_dict": sae_module.state_dict()}, final_path)
    else:
        torch.save(sae_module.state_dict(), final_path)
    log.info(f"Fine-tuned SAE saved: {final_path}")

    summary = {
        "steps": step,
        "dk_coeff": args.dk_coeff,
        "final_dk_penalty": float(dk_penalty.item()),
        "final_fvu": float(fvu),
        "final_l0": float(l0),
        "elapsed_sec": time.time() - t0,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"Summary: dk_penalty={dk_penalty.item():.6f}, fvu={fvu:.4f}, l0={l0:.0f}")


if __name__ == "__main__":
    main()
