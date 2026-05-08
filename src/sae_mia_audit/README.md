# `sae_mia_audit`

Support package for the Dark Subspace artefact. This package is intentionally small and exposes only the helpers the paper scripts under `scripts/dark_subspace/` actually import. It is not a general-purpose membership-inference library.

| Subpackage | Purpose | Used by |
| --- | --- | --- |
| `data/` | SAE training-corpus loader (`load_sae_corpus`) and tokenisation helpers (`TokenizeConfig`, `tokenize_batch`). | `scripts/dark_subspace/finetune_sae_dk.py`, `behavioral_channels.py`, `paraphrase_sensitivity.py`, `bcd_extractability_predictor.py`, `baseline_attacks_suite.py` |
| `models/` | Hugging Face causal-LM wrapper (`load_model_and_tokenizer`, `CausalLMWrapper`) and log-probability helpers used by the loss-attack baseline. | All paper scripts that read residual-stream activations |
| `sae/` | Sparse-autoencoder module (`SparseAutoencoder`, `SAEConfig`), trainer (`SAETrainer`, `MultiSAETrainer`), checkpoint I/O (`load_sae_checkpoint_any`), and the SAE-protocol adapter for SAIF-format checkpoints. | `scripts/dark_subspace/sae_dark_subspace.py`, `subspace_ablation_eval.py`, `feature_ablation_dark_subspace.py`, `feature_ablation_random_k.py`, `finetune_sae_dk.py`, `fresh_probe_test.py`, `paraphrase_sensitivity.py`, `standard_mia_probe_decomposition.py` |
| `methods/` | A single `baselines` module providing the loss-attack and zlib-loss-ratio scoring functions. | `scripts/dark_subspace/length_baseline.py` and `baseline_attacks_suite.py` |
| `utils/` | Hugging Face spec parsing (`HFModelSpec`), structured logging (`setup_logging`, `get_logger`), and global seed configuration (`SeedConfig`, `set_global_seed`). | All paper scripts |

Earlier iterations of this package included exploratory code for non-paper baselines (PCA, neuron-probe, random-rotation), bootstrap and group-wise eval helpers, alternative MIA methods (Min-K%, infilling, DC-PDD, NA-PDD, SAE-NA-PDD, probe-based, sae_audit), SAE intervention helpers, consistency-matching utilities, and a feature top-k collector. None of those modules were imported by any paper script and they have been removed from the public artefact. The reproducibility caveat in the root `README.md` documents what is bundled and what regenerates on rerun.

## Install

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

The CPU-only claim verifier (`scripts/dark_subspace/verify_claims.py`) does not require installing this package. It reads only the JSON files under `results/dark_subspace/`.

## Console script

`pip install -e .` installs a `sae-mia-audit` console script (`cli.py`) that is a thin pass-through to `scripts/shared/train_sae.py`. Most users invoke the paper scripts under `scripts/dark_subspace/` directly with `--help` rather than going through the CLI.
