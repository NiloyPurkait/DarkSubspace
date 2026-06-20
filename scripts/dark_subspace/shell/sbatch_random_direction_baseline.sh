#!/usr/bin/env bash
#SBATCH --job-name=rand_dir_baseline
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/rand_dir_baseline_%j.out
#SBATCH --error=logs/rand_dir_baseline_%j.err
#SBATCH --nodes=1
#
# sbatch_random_direction_baseline.sh.
#
# For each gate-passing model, samples 100 random unit directions in the
# layer-activation space and computes member/nonmember AUROC along each.
# Compares the resulting null distribution to the actual residual probe AUROC.
#
# Used in Appendix random-direction null distribution of the paper.
# Reproduce: sbatch scripts/dark_subspace/shell/sbatch_random_direction_baseline.sh

set -eu
cd "$(dirname "$0")/../../.."
mkdir -p logs runs/dark_subspace/random_direction_baseline

PY=".venv/bin/python"
SCRIPT="scripts/dark_subspace/random_direction_baseline.py"
MEM="data/memcirc_ctrl_ft/member.jsonl"
NON="data/memcirc_ctrl_ft/nonmember.jsonl"

echo "=== P69 layer 16 ==="
${PY} ${SCRIPT} \
  --model-path runs/controlled_ft/run_20260306_055225/ft_epoch5/model \
  --member-texts ${MEM} --nonmember-texts ${NON} \
  --layer 16 \
  --output-path runs/dark_subspace/random_direction_baseline/p69_layer16.json \
  --n-directions 100 --seed 42

echo ""
echo "=== Neo layer 16 ==="
${PY} ${SCRIPT} \
  --model-path runs/controlled_ft/run_20260221_115025/ft_epoch5/model \
  --member-texts ${MEM} --nonmember-texts ${NON} \
  --layer 16 \
  --output-path runs/dark_subspace/random_direction_baseline/neo_layer16.json \
  --n-directions 100 --seed 42

echo ""
echo "=== Qwen2 7B layer 16 ==="
${PY} ${SCRIPT} \
  --model-path runs/controlled_ft/run_20260313_192753/ft_epoch5/model \
  --member-texts ${MEM} --nonmember-texts ${NON} \
  --layer 16 \
  --output-path runs/dark_subspace/random_direction_baseline/qwen2_layer16.json \
  --n-directions 100 --seed 42

echo ""
echo "=== P12B layer 18 ==="
${PY} ${SCRIPT} \
  --model-path runs/controlled_ft/run_20260308_001316/ft_epoch5/model \
  --member-texts ${MEM} --nonmember-texts ${NON} \
  --layer 18 \
  --output-path runs/dark_subspace/random_direction_baseline/p12b_layer18.json \
  --n-directions 100 --seed 42

echo ""
echo "=== Random-direction baseline done: $(date) ==="
echo "Outputs: runs/dark_subspace/random_direction_baseline/{p69,neo,qwen2,p12b}_layer*.json"
