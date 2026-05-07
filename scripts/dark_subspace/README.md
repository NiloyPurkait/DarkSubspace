# Dark Subspace Scripts

This directory contains the paper-specific code for the Dark Subspace artifact. The scripts are intentionally grouped by experiment role instead of combined into one large driver, because most reviewer checks target one claim or one control at a time.

For the term-to-paper mapping (BCD, K-OC-2, errPC, OC, memcirc), see the **Glossary** subsection of the root `README.md`. For paper-passage-to-script-and-JSON mapping, see the **Claim-Source Map** subsection of the root `README.md`.

## Reviewer subset

These are the scripts a reviewer is most likely to invoke. The verifier reads only shipped JSONs and is the recommended first check.

| Script | Purpose |
| --- | --- |
| `verify_claims.py` | CPU-only check that shipped JSONs match paper-cited values. |
| `figure_data_loader.py` | Smoke-loads the JSON sources used by figure scripts and reports any missing files. |
| `plot_figures.py`, `plot_advanced_figures.py`, `plot_privacy_aware_comparison.py`, `plot_score_distributions.py` | Regenerate paper figures from shipped JSON records. |
| `behavioral_channels.py` | Fits BCD directions and reports layer-wise geometry/probe AUROC. |
| `sae_dark_subspace.py` | Computes original, SAE-reconstructed, and SAE-residual membership scores. |
| `subspace_ablation_eval.py` | Runs K-PC residual ablations and controls (used in `tab:kpc_kten_cells`). |
| `bow_ceiling.py` | Bag-of-words surface-form baseline. |
| `paraphrase_sensitivity.py` | Word-order paraphrase orientation diagnostic. |

## Full reproduction toolkit

All 37 scripts under `scripts/dark_subspace/`, grouped by role:

### Reviewer entry points

| Script | Purpose |
| --- | --- |
| `verify_claims.py` | CPU-only verifier (asserted-check summary). |
| `figure_data_loader.py` | Smoke-loads JSON sources for figures; defines `MODEL_REGISTRY` (preferred `_v2` re-runs noted in source comment). |

### Plotting

| Script | Purpose |
| --- | --- |
| `plot_figures.py` | Standard figures over the shipped JSON tree. |
| `plot_advanced_figures.py` | Cross-model and aggregate figures. |
| `plot_privacy_aware_comparison.py` | Figure `fig:privacy_aware`. |
| `plot_score_distributions.py` | Figure `fig:score_distributions` and the full appendix variant. |

### Core experiment scripts

| Script | Purpose |
| --- | --- |
| `behavioral_channels.py` | BCD: fits $\dK$, $\dR$, $\SK$, $\SR$, principal angles, and per-layer probe AUROC. |
| `sae_dark_subspace.py` | Computes original / SAE-reconstructed / SAE-residual membership scores at the SAE layer. |
| `subspace_ablation_eval.py` | K-PC (errPC) residual ablation with random-rotation, matched-Gaussian, and column-mask controls. |
| `standard_mia_probe_decomposition.py` | Standard MIA probes (loss attack, MIN-K%, zlib) under the same decomposition. |
| `bcd_extractability_predictor.py` | Per-text loss / ROUGE-L / score_K / score_R / score_SK / score_SR predictor. |
| `extract_canonical_activations.py` | Materialises mean-pooled residual-stream activations on the canonical evaluation pool. |

### Controls and diagnostics

| Script | Purpose |
| --- | --- |
| `bow_ceiling.py` | Bag-of-words surface-form baseline (`app:bow_baseline`). |
| `norm_baseline.py` | Activation-norm baseline (`tab:norm_direction`, `tab:l2_normalized`). |
| `l2_normalized_auroc.py` | L2-normalised residual membership AUROC (`tab:l2_normalized`). |
| `paraphrase_sensitivity.py` | Word-order paraphrase orientation diagnostic (`app:tpr_paraphrase`). |
| `tpr_at_low_fpr.py` | TPR at 0.1% FPR for the residual $\dK$ channel (`tab:tpr_at_0p1pct_fpr`). |
| `feature_ablation_dark_subspace.py` | Top-$k$ classifier-feature ablation. |
| `feature_ablation_random_k.py` | Random-feature ablation control matched to the classifier-feature subset. |
| `random_direction_baseline.py` | 100 random unit-direction membership AUROC per model. |
| `make_random_sae.py` | Random-init SAE generator for the random-init SAE control. |
| `heldout_dk_eval.py` | Held-out partition-fit reductions for $\dK$ (`tab:heldout_dk_per_split`). |
| `validate_recall_proxy.py` | Validates the loss-based recall proxy against ROUGE-L. |
| `recall_label_validation.py` | Recall-label sanity check used in `app:bcd_details`. |
| `fresh_probe_test.py` | Fresh-probe permutation null for `tab:fresh_probe_v2`. |
| `fsc_random_null.py` | Random-subset null for the feature sufficiency criterion (`tab:fsc_values`). |
| `baseline_attacks_suite.py` | Standard pre-training-MIA attack suite for output-level scope checks. |

### SAE training and fine-tuning

| Script | Purpose |
| --- | --- |
| `finetune_sae_dk.py` | Privacy-aware SAE fine-tune with $\dK$ reconstruction penalty (Eq. 1). |
| `build_disjoint_owt_corpus.py` | Builds the OWT partition disjoint from the canonical evaluation pool (`app:corpus_disjoint`). |

(The canonical SAE trainer itself is `scripts/shared/train_sae.py`.)

### Aggregation

| Script | Purpose |
| --- | --- |
| `aggregate_multiseed.py` | Aggregates multi-seed JSON outputs across cohorts. |
| `sae_noise_floor_aggregate.py` | Aggregates the Pythia-6.9B mixed-SAE noise-floor cohort. |
| `p69_n5_harmonize.py` | Produces the canonical harmonized N=5 Pythia-6.9B cohort record. |
| `p12b_multiseed_query.py` | Reads `runs/dark_subspace/sae_dark_subspace/p12b_mixed_sae_seed{47..51}/results.json` and reports the multi-seed table. |
| `per_row_bootstrap_kocl2.py` | Per-row paired bootstrap for the K-OC-2 cohort (`app:koc2_bootstrap`). |
| `per_row_bootstrap_kocl2_residual_minus_recon.py` | Same bootstrap with the residual-minus-reconstruction margin. |
| `rerun_bootstrap_cis.py` | Reruns bootstrap CIs in place on shipped JSONs (used to refresh paraphrase and BoW intervals; recorded in `bootstrap_history` fields). |

### Path bootstrap

| Script | Purpose |
| --- | --- |
| `_bootstrap.py` | Local sys.path shim; imported as the first line of each paper script. |
| `configs/` | JSON config files (e.g. `subspace_ablation_roster.json`) consumed by the experiment scripts. |

## SLURM wrappers

`shell/` contains cluster launchers for the expensive jobs. These wrappers are not required for the CPU verifier. They document the actual command lines, resources, seeds, and output directories for GPU reproduction. Per-architecture multi-seed launchers live under `shell/multiseed/`. Notable wrappers:

- `shell/sbatch_p12b_multiseed_array.sh` — Pythia-12B SAE seeds 47-51 (seeds 50, 51 are not shipped in this artefact; see Provenance Notes in the root README).
- `shell/sbatch_p69_seed42_postfix.sh` — Pythia-6.9B seed-42 retrain that closes the six-seed mixed-data cohort under the post-corpus-cycling-fix pipeline.
- `shell/sbatch_pre_ft_baseline.sh` — Pre-FT negative control on the un-fine-tuned base Pythia-6.9B.
- `shell/sbatch_p69_disjoint_owt_sae.sh` — Corpus-disjoint dictionary control on Pythia-6.9B.
- `shell/sbatch_random_direction_baseline.sh`, `shell/sbatch_random_init_sae.sh` — Random-direction and random-init SAE controls.

Figure plotting scripts write to `outputs/figures/` by default. Set `FIGDIR` when writing directly into a separate manuscript checkout.

Historical dataset and checkpoint labels still use `memcirc` in some paths, for example `data/memcirc_ctrl_ft/` and `runs/sae/memcirc_*`. Those labels are provenance identifiers from the experiment campaign. New reviewer-facing code paths use `scripts/dark_subspace/` and `results/dark_subspace/`.
