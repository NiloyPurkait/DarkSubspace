#!/usr/bin/env bash
#SBATCH --job-name=random_init_sae
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/random_init_sae_%j.out
#SBATCH --error=logs/random_init_sae_%j.err
#
# sbatch_random_init_sae.sh.
#
# Generates a random-init SAE checkpoint with full audit trail (config.json,
# train_summary.json, manifest.json) and runs the dark-subspace eval against
# it. This is the random-init operator control for the dark-subspace claim.
#
# Used in Appendix random-init SAE control of the paper.
# Reproduce: sbatch scripts/dark_subspace/shell/sbatch_random_init_sae.sh
#
# Architecture matches the canonical Pythia-6.9B six-seed anchor recipe.
# d_model_mult=4, d_sae=16384, layer 16. The random-init checkpoint shares the
# architecture of the paired trained SAE (referenced via REF_SAE).
#
# SAE init is CPU-light. Eval is dark_subspace at ~30 minutes.
# Wall budget. ~1 GPU-h.

set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p logs runs/sae/random_init_p69_layer16_v2

PY=.venv/bin/python
P69_MODEL="runs/controlled_ft/run_20260306_055225/ft_epoch5/model"
P69_BCD="runs/dark_subspace/behavioral_channels/p69_epoch5"
REF_SAE="runs/sae/train_sae__runs_controlled_ft_run_20260306_055225_ft_epoch5_model__layer16__mult4__l10.0005__20260413_184801/sae_final.pt"
RAND_SAE_DIR="runs/sae/random_init_p69_layer16_v2"
RAND_SAE="${RAND_SAE_DIR}/sae_final.pt"
MEMBER="data/memcirc_ctrl_ft/member.jsonl"
NONMEMBER="data/memcirc_ctrl_ft/nonmember.jsonl"
LAYER=16
SEED=42

echo "=== Step 1. Generate random-init SAE checkpoint with audit trail ==="
${PY} scripts/dark_subspace/make_random_sae.py \
  --reference-sae "$REF_SAE" \
  --output-path "$RAND_SAE" \
  --mode kaiming --seed $SEED

# Materialise config.json + train_summary.json stubs alongside sae_final.pt
# so that downstream loaders see a full audit trail.
${PY} - <<PYEOF
import json, torch, hashlib
from pathlib import Path
ckpt = torch.load("$RAND_SAE", map_location="cpu", weights_only=False)
cfg = dict(ckpt.get("sae_cfg", {}))
cfg["_random_init"] = True
cfg["_random_init_mode"] = ckpt.get("_random_init_mode")
cfg["_random_init_seed"] = ckpt.get("_random_init_seed")
cfg["_reference_sae"] = ckpt.get("_reference_sae")
Path("$RAND_SAE_DIR/config.json").write_text(json.dumps(cfg, indent=2))
summary = {
    "n_steps": 0,
    "is_random_init": True,
    "mode": cfg.get("_random_init_mode"),
    "seed": cfg.get("_random_init_seed"),
    "reference_sae": cfg.get("_reference_sae"),
    "d_model": cfg.get("d_model"),
    "d_sae": cfg.get("d_sae"),
    "layer_idx": cfg.get("layer_idx"),
}
Path("$RAND_SAE_DIR/train_summary.json").write_text(json.dumps(summary, indent=2))
sd_keys = list(ckpt["state_dict"].keys())
manifest = {
    "schema_version": "random_init_v2",
    "sae_final_pt_sha256": hashlib.sha256(open("$RAND_SAE","rb").read()).hexdigest(),
    "state_dict_keys": sd_keys,
}
Path("$RAND_SAE_DIR/manifest.json").write_text(json.dumps(manifest, indent=2))
print("[random_init] wrote config.json, train_summary.json, manifest.json")
PYEOF

echo "=== Step 2. Dark-subspace eval against random-init SAE ==="
${PY} scripts/dark_subspace/sae_dark_subspace.py \
  --model-path "$P69_MODEL" \
  --bcd-dir "$P69_BCD" \
  --sae-path "$RAND_SAE" \
  --member-texts "$MEMBER" \
  --nonmember-texts "$NONMEMBER" \
  --layer $LAYER \
  --output-dir runs/dark_subspace/sae_dark_subspace/p69_random_init_v2 \
  --model-id p69_random_init_v2

echo "=== DONE: $(date) ==="
echo "SAE: $RAND_SAE"
echo "Eval: runs/dark_subspace/sae_dark_subspace/p69_random_init_v2/results.json"
