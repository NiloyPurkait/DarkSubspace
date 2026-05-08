# Dark Subspace Scripts

This directory contains the paper-specific code for this artefact. Scripts are grouped by experiment role rather than combined into one large driver, so that individual claims and controls can be verified independently.

A small number of engineering labels (`errPC`, `kocl2`, `memcirc`) survive in path strings and one filename. Their paper-passage equivalents are tabulated under **Naming notes** in the root `README.md`. The full paper-passage-to-script-and-JSON mapping is the **Claim-Source Map**, also in the root `README.md`.

## Recommended starting points

The verifier is the recommended first check. It reads only the JSON files under `results/dark_subspace/`.

| Script | Purpose |
| --- | --- |
| `verify_claims.py` | CPU-only check that the JSON records match paper-cited values. |
| `figure_data_loader.py` | Validates that all JSON sources used by the figure scripts exist. |
| `plot_figures.py`, `plot_advanced_figures.py`, `plot_privacy_aware_comparison.py`, `plot_score_distributions.py` | Regenerate paper figures from the JSON records. |
| `behavioral_channels.py` | Fits the channel-decomposition directions and reports per-layer geometry and probe AUROC. |
| `sae_dark_subspace.py` | Computes original, SAE-reconstructed, and SAE-residual membership scores. |
| `subspace_ablation_eval.py` | K-PC residual ablation and controls (used in `tab:kpc_kten_cells`). |
| `bow_ceiling.py` | Bag-of-words surface-form baseline. |
| `paraphrase_sensitivity.py` | Word-order paraphrase orientation diagnostic. |

## Full script index

All 37 scripts under `scripts/dark_subspace/`, grouped by role.

### Entry points

| Script | Purpose |
| --- | --- |
| `verify_claims.py` | CPU-only verifier (asserted-check summary). |
| `figure_data_loader.py` | Validates that all JSON sources for the figure scripts exist, and defines `MODEL_REGISTRY` (preferred `_v2` re-runs noted in source comment). |

### Plotting

| Script | Purpose |
| --- | --- |
| `plot_figures.py` | Standard figures over the JSON tree under `results/dark_subspace/`. |
| `plot_advanced_figures.py` | Cross-model and aggregate figures. |
| `plot_privacy_aware_comparison.py` | Figure `fig:privacy_aware`. |
| `plot_score_distributions.py` | Figure `fig:score_distributions` and the full appendix variant. |

### Core experiment scripts

| Script | Purpose |
| --- | --- |
| `behavioral_channels.py` | Channel decomposition. Fits $d_K$, $d_R$, $S_K$, $S_R$, principal angles, and per-layer probe AUROC. |
| `sae_dark_subspace.py` | Computes original, SAE-reconstructed, and SAE-residual membership scores at the SAE layer. |
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
| `length_baseline.py` | Length-feature membership classifier on the controlled split (`app:length_baseline`). |
| `nonlinear_probe.py` | MLP-vs-linear probe comparison at the analysis layer (`tab:nonlinear`). |
| `label_shuffled_null.py` | Permutation null on $\cos(d_K, d_R)$ under shuffled labels (`app:label_shuffled_null`). |
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
| `dd_table_render.py` | Renders the extraction-detection separation tables (`tab:dd_full`, `tab:dd_extraction`, `tab:epoch_dd`) from cached per-cell records. |

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
| `p69_n5_harmonize.py` | Produces the canonical harmonised N=5 Pythia-6.9B cohort record. |
| `p12b_multiseed_query.py` | Reads `runs/dark_subspace/sae_dark_subspace/p12b_mixed_sae_seed{47..51}/results.json` and reports the multi-seed table. |
| `per_row_bootstrap_kocl2.py` | Per-row paired bootstrap for the directional sign-test cohort (Appendix `app:koc2_bootstrap`). The script filename uses `kocl2` as an engineering label. |
| `per_row_bootstrap_kocl2_residual_minus_recon.py` | Same bootstrap with the residual-minus-reconstruction margin. |
| `rerun_bootstrap_cis.py` | Reruns bootstrap confidence intervals in place on the JSON records (used to refresh paraphrase and BoW intervals, recorded in `bootstrap_history` fields). |

### Path bootstrap

| Script | Purpose |
| --- | --- |
| `_bootstrap.py` | Local sys.path shim, imported as the first line of each paper script. |
| `configs/` | JSON config files (e.g. `subspace_ablation_roster.json`) consumed by the experiment scripts. |

## SLURM wrappers

`shell/` contains the cluster launchers for the GPU jobs. The wrappers are not required for the CPU verifier. They document the command lines, resources, seeds, and output directories used for GPU reproduction. Per-architecture multi-seed launchers live under `shell/multiseed/`. Selected entries follow.

- `shell/sbatch_p12b_multiseed_array.sh`. Pythia-12B SAE multi-seed array.
- `shell/sbatch_p69_seed42_postfix.sh`. Pythia-6.9B seed-42 mixed-data SAE training run.
- `shell/sbatch_pre_ft_baseline.sh`. Pre-fine-tuning control on the un-fine-tuned base Pythia-6.9B.
- `shell/sbatch_p69_disjoint_owt_sae.sh`. Corpus-disjoint dictionary control on Pythia-6.9B.
- `shell/sbatch_random_direction_baseline.sh`, `shell/sbatch_random_init_sae.sh`. Random-direction and random-init SAE controls.

Figure plotting scripts write to `outputs/figures/` by default. Set `FIGDIR` to redirect output to a separate manuscript directory.

Historical dataset and checkpoint labels still use `memcirc` in some paths, for example `data/memcirc_ctrl_ft/` and `runs/sae/memcirc_*`. Those labels are retained as earlier run provenance identifiers. The current code paths use `scripts/dark_subspace/` and `results/dark_subspace/`.
