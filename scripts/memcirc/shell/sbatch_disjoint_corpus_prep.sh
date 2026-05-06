#!/bin/bash
#SBATCH --job-name=disjoint_corpus_prep
#SBATCH --partition=GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/disjoint_corpus_prep_%j.out
#SBATCH --error=logs/disjoint_corpus_prep_%j.err
#
# sbatch_disjoint_corpus_prep.sh.
#
# Builds a 2000-document OpenWebText jsonl that is provably disjoint (SHA-256
# token-set) from the member.jsonl and nonmember.jsonl partitions used by the
# fine-tuning recipe.
#
# Used in Appendix corpus-disjoint robustness check of the paper.
# Reproduce: sbatch scripts/memcirc/shell/sbatch_disjoint_corpus_prep.sh
#
# No GPU requested (--gres omitted). Cluster only exposes GPU partitions, so we
# attach to GPU with a CPU-only reservation. This is a 2-minute task on CPU.

set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p logs data/memcirc_ctrl_disjoint

PY=env/bin/python3

${PY} scripts/memcirc/build_disjoint_owt_corpus.py \
    --member data/memcirc_ctrl_ft/member.jsonl \
    --nonmember data/memcirc_ctrl_ft/nonmember.jsonl \
    --output data/memcirc_ctrl_disjoint/mixed_disjoint.jsonl \
    --proof data/memcirc_ctrl_disjoint/disjointness_proof.json \
    --n-docs 2000 \
    --seed 100

echo "[disjoint_corpus_prep] DONE. $(wc -l < data/memcirc_ctrl_disjoint/mixed_disjoint.jsonl) lines."
