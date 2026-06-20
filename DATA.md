# Data and Artifact Provisioning

This document is the single entry point for an external researcher who wants to
obtain or regenerate the artifacts behind "The Dark Subspace of Fine-Tuning
Memorisation". It consolidates the per-experiment "regenerate from script X"
notes that are otherwise scattered across the Claim-Source Map in `README.md`.
Nothing here contradicts the README. Where a fact also appears there, this
document mirrors it.

The repository ships a curated JSON tree under `results/dark_subspace/` and a
CPU-only verifier. The verifier needs no models, no GPUs, and no network. Full
reproduction needs GPU compute, the controlled fine-tuning corpus, the
fine-tuned checkpoints, and the SAE checkpoints, none of which are bundled.

## Models

The base models below are public HuggingFace checkpoints. Each is fine-tuned
once under the controlled recipe described in the next section, and the analysis
SAE is trained at the fixed per-model layer. The SAE layer is the analysis layer
fixed in advance in Appendix Table `tab:model_details` (selected before any SAE
training by maximising cross-validated membership AUROC on raw residual-stream
activations). The parameter counts and SAE layers below are taken from that
table.

| Model | HuggingFace base identifier | Params | SAE analysis layer |
| --- | --- | --- | --- |
| Pythia-1B | `EleutherAI/pythia-1b` | 1.0B | 8 |
| Pythia-6.9B | `EleutherAI/pythia-6.9b` | 6.9B | 16 |
| Pythia-12B | `EleutherAI/pythia-12b` | 12B | 18 |
| GPT-Neo-2.7B | `EleutherAI/gpt-neo-2.7B` | 2.7B | 16 |
| OPT-6.7B | `facebook/opt-6.7b` | 6.7B | 24 |
| Falcon-7B | `tiiuae/falcon-7b` | 7.0B | 16 |
| Mistral-7B | `mistralai/Mistral-7B-v0.1` | 7.2B | 16 |
| Llama-3-8B | `meta-llama/Meta-Llama-3-8B` | 8.0B | 16 |
| Qwen2-7B | `Qwen/Qwen2-7B` | 7.6B | 16 |

Notes.

- Llama-3-8B and Mistral-7B are gated on HuggingFace and need an `HF_TOKEN`. The
  Gemma-2-2B scaling row (bundled separately, see the README Scope Notes) is also
  gated.
- The base models above are the inputs to fine-tuning. The fine-tuned checkpoints
  used for every paper-cited number are produced by the controlled fine-tuning
  recipe (see the next section). They are not a public download. Fine-tuned
  weights are stored locally and loaded through the HuggingFace model-loading
  interface, with all inference run in half precision on A40, L40S, or H100 GPUs.
- The per-model SAE training wrappers under
  `scripts/dark_subspace/shell/multiseed/` record the exact layer, dictionary
  multiplier, L1 coefficient, and token budget for each model.

## Controlled corpus

All experiments use a controlled OpenWebText-derived split into a member set and
a non-member set. The member documents are the fine-tuning corpus. The
non-member documents are held out from fine-tuning and used as the negative class
for membership measurement. The fine-tuning corpus and the member and non-member
text files live under `data/memcirc_ctrl_ft/` (`mixed.jsonl`, `member.jsonl`,
`nonmember.jsonl`) in the run tree. The `memcirc` prefix is an earlier project
label retained only in path strings for provenance, as documented in the README
naming notes.

For the corpus-disjoint dictionary control (Appendix `app:corpus_disjoint`), a
separate OWT partition that is disjoint from the evaluation pool is built by
`scripts/dark_subspace/build_disjoint_owt_corpus.py`, and the SAE is retrained on
that partition. That control regenerates to
`runs/dark_subspace/sae_dark_subspace/p69_disjoint_owt_seed${SEED}/results.json`
on rerun and is not bundled.

## SAE checkpoints

The headline result is the Pythia-6.9B mixed-data SAE cohort. The five SAEs share
identical hyperparameters and differ only in random initialisation.

- Layer 16.
- Dictionary multiplier 4 (so the dictionary is four times the hidden width).
- L1 coefficient 5e-4.
- Roughly 200M training tokens.
- Dead-feature resampling on (every 500 steps, threshold 1e-6, aux coefficient 0.1).
- Seeds 42 through 46. The cohort is harmonised to the same five seed labels as
  the Pythia-1B multi-seed cohort. The seed-42 SAE was retrained once during the
  validity-gate sweep and carries the on-disk label `42_postfix`, which is
  canonically labelled `42` in the table and the harmonised JSON.

The training command is `scripts/shared/train_sae.py`. The headline invocation
(matching the per-seed wrapper
`scripts/dark_subspace/shell/sbatch_multiseed_mixed_sae_p69.sh`) is

```bash
.venv/bin/python scripts/shared/train_sae.py \
  --model <fine-tuned Pythia-6.9B checkpoint> \
  --layers 16 \
  --d-model-mult 4 \
  --l1-coeff 0.0005 \
  --train-tokens 200000000 \
  --tokens-per-step 4096 \
  --seq-len 256 \
  --batch-size 4 \
  --lr 3e-4 \
  --seed <42..46> \
  --corpus data/memcirc_ctrl_ft/mixed.jsonl \
  --corpus-text-field text \
  --aux-coeff 0.1 \
  --resample-dead-features --resample-every 500 --resample-dead-threshold 1e-6 \
  --mode paper --final-eval \
  --runs-dir runs/sae
```

The SLURM wrapper requests one GPU, 48G of memory, and a 36-hour wall-clock
ceiling on the `GPU` partition. Each seed runs SAE training followed by the
dark-subspace evaluation (`scripts/dark_subspace/sae_dark_subspace.py`), writing
per-seed results to
`runs/dark_subspace/sae_dark_subspace/p69_mixed_sae_seed${SEED}/results.json`.
The five SAE checkpoints plus their per-text score arrays regenerate to the
`runs/` tree on rerun and are not bundled. As the README records, SAE
checkpoints, fine-tuned weights, and the controlled corpus total tens of GB
across the bundled experiments, while the bundled JSONs total a few MB.

The harmonised cohort summary is bundled at
`results/dark_subspace/paper_claims/p69_n5_harmonized_2026-05-06.json`, and `make
headline` reads the N=5 cohort means from it.

## What is bundled versus regenerated

The authoritative mapping from each paper passage to its source script and JSON
is the Claim-Source Map in `README.md`. The short version is as follows.

- Bundled. The JSON tree under `results/dark_subspace/`. This includes the
  `paper_claims/` records consumed by the verifier and the curated mirror under
  `generated/`. Review does not require the full `runs/` directory.
- Regenerated. The `runs/` tree. SAE checkpoints, fine-tuned model weights, the
  controlled corpus, and per-text score arrays all regenerate from the scripts
  and SLURM wrappers, and are excluded by `.gitignore`. Rows in the Claim-Source
  Map marked "not bundled" regenerate on rerun.

## Reproducibility note

The paper-cited values were produced under `SeedConfig.deterministic=False` (the
default in `src/sae_mia_audit/utils/seed.py`), which enables CUDNN heuristics
(`cudnn.benchmark=True`) needed for tractable wall-clock on the 12B pipeline.
CUDNN then selects different convolution kernels per run, so bit-reproducibility
of GPU outputs is not guaranteed across reruns even at fixed Python, NumPy, and
torch RNG seeds. The bundled JSONs record `seed` and `bootstrap_seed` so the
random-number streams can be matched at those levels. Reviewers who require
bit-reproducible CUDA outputs can pass `deterministic=True` to `SeedConfig`,
though the paper-cited values were not produced that way. Small numerical
differences, typically below the 0.002 verifier tolerance, are expected from
CUDNN-kernel non-determinism.

## Edit Log

| Date | Agent | Change |
| --- | --- | --- |
| 2026-06-20 | Writer | Created DATA.md provisioning doc for the ICML camera-ready public repo (model roster with HF IDs and SAE layers, controlled corpus, headline SAE training command, bundled-versus-regenerated split, CUDA non-determinism note). |
