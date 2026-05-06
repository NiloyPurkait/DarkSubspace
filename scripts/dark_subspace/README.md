# Dark Subspace Scripts

This directory contains the paper-specific code for the Dark Subspace artifact. The scripts are intentionally grouped by experiment role instead of combined into one large driver, because most reviewer checks target one claim or one control at a time.

## Reviewer Entry Points

| Script | Purpose |
| --- | --- |
| `verify_claims.py` | CPU-only check that shipped JSONs match paper-cited values. |
| `figure_data_loader.py` | Smoke-loads JSON sources used by figure scripts. |
| `plot_figures.py`, `plot_advanced_figures.py` | Regenerate paper figures from JSON records. |

## Core Experiment Scripts

| Script | Purpose |
| --- | --- |
| `behavioral_channels.py` | Fits BCD directions and reports layer-wise geometry/probe AUROC. |
| `sae_dark_subspace.py` | Computes original, SAE-reconstructed, and SAE-residual membership scores. |
| `subspace_ablation_eval.py` | Runs K-PC residual ablations and controls. |
| `standard_mia_probe_decomposition.py` | Checks standard MIA probes against the same decomposition. |

## Controls and Diagnostics

| Script | Purpose |
| --- | --- |
| `bow_ceiling.py` | Bag-of-words surface-form baseline. |
| `norm_baseline.py` | Activation-norm baseline. |
| `paraphrase_sensitivity.py` | Word-order paraphrase orientation diagnostic. |
| `feature_ablation_dark_subspace.py`, `feature_ablation_random_k.py` | Feature ablation controls. |
| `random_direction_baseline.py`, `make_random_sae.py` | Random-direction and random-SAE controls. |
| `heldout_dk_eval.py`, `validate_recall_proxy.py` | Held-out and recall-label validation checks. |

## Aggregation

| Script | Purpose |
| --- | --- |
| `aggregate_multiseed.py` | Aggregates multi-seed JSON outputs. |
| `sae_noise_floor_aggregate.py` | Aggregates the Pythia-6.9B mixed-SAE noise-floor cohort. |
| `p69_n5_harmonize.py` | Produces the canonical harmonized N=5 Pythia-6.9B cohort record. |
| `per_row_bootstrap_kocl2.py`, `per_row_bootstrap_kocl2_residual_minus_recon.py` | Per-row bootstrap calculations for cohort comparisons. |

## SLURM Wrappers

`shell/` contains cluster launchers for the expensive jobs. These wrappers are not required for the CPU verifier. They document the actual command lines, resources, seeds, and output directories for GPU reproduction.

Figure plotting scripts write to `outputs/figures/` by default. Set `FIGDIR` when writing directly into a separate manuscript checkout.

Historical dataset and checkpoint labels still use `memcirc` in some paths, for example `data/memcirc_ctrl_ft/` and `runs/sae/memcirc_*`. Those labels are provenance identifiers from the experiment campaign. New reviewer-facing code paths use `scripts/dark_subspace/` and `results/dark_subspace/`.
