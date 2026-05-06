#!/bin/bash
#SBATCH --job-name=std_probes_p69
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=logs/std_probes_p69_%j.out
#SBATCH --error=logs/std_probes_p69_%j.err
#
# sbatch_standard_probes_p69.sh.
#
# Replicates published MIA probes (loss, Min-K%, zlib) under SAE patching
# decomposition. Three patching conditions ({h_orig, h_recon, h_residual})
# applied as forward hooks at layer 16 of Pythia-6.9B fine-tuned for 5 epochs,
# crossed with three probes.
#
# Used in Section MIA-probe replication of the paper.
# Reproduce: sbatch scripts/dark_subspace/shell/sbatch_standard_probes_p69.sh
#
# Output. AUROC per (condition, probe). Drop is AUROC(orig) - AUROC(recon).
# Residual recovery is AUROC(residual) - AUROC(recon). Hypothesis. Drop > 0.05
# and residual > 0.55 across at least 2 of 3 probes.
#
# Saturation note. Loss probes can saturate near AUROC 1.0 at epoch 5; the
# AUROC drop on h_recon is still meaningful because it tests whether the
# SAE-reconstructed activation preserves the saturating signal. An epoch-3
# follow-up is supported by the canonical experimental design.
#
# Inference-only at 6.9B fits fp16 on a 48 GB GPU.
# Wall budget. ~3-4 GPU-h.

set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p logs

PY=env/bin/python3
P69_MODEL="runs/controlled_ft/run_20260306_055225/ft_epoch5/model"
SAE_PATH="runs/sae/train_sae__runs_controlled_ft_run_20260306_055225_ft_epoch5_model__layer16__mult4__l10.0005__20260413_184801/sae_final.pt"
MEMBER="data/memcirc_ctrl_ft/member.jsonl"
NONMEMBER="data/memcirc_ctrl_ft/nonmember.jsonl"
LAYER=16

OUT_DIR="runs/dark_subspace/standard_mia_probes/p69_dark_subspace_replication"
mkdir -p "$OUT_DIR"

if [ ! -f "$SAE_PATH" ]; then
  echo "ERROR: SAE checkpoint not found: $SAE_PATH"
  exit 1
fi

echo "=== Standard probe decomposition on P69 epoch 5 ==="
echo "Model: $P69_MODEL"
echo "SAE:   $SAE_PATH"
echo "Out:   $OUT_DIR"

${PY} scripts/dark_subspace/standard_mia_probe_decomposition.py \
  --model-path "$P69_MODEL" \
  --sae-path "$SAE_PATH" \
  --layer $LAYER \
  --member-texts "$MEMBER" \
  --nonmember-texts "$NONMEMBER" \
  --output-dir "$OUT_DIR" \
  --model-tag p69_epoch5 \
  --seed 42

echo "=== DONE: $(date) ==="
echo "Results: $OUT_DIR/results.json"
