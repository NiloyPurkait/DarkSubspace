# The Dark Subspace of Fine-Tuning Memorisation

Code and results artefact for an anonymous double-blind submission to the ICML 2026 Mechanistic Interpretability Workshop.

This repository contains the paper-specific experiment code, a curated set of JSON results, and a CPU-only verifier that checks the paper-cited numerical claims against the included JSON records. The manuscript is distributed separately and is not in this repository.

## Quickstart

The verifier uses only the Python standard library and reads the JSON files under `results/dark_subspace/`.

```bash
python3 scripts/dark_subspace/verify_claims.py
```

Expected output (final summary line):

```text
Asserted check summary: N/N pass, 0 fail
All asserted checks pass within tolerance.
```

For full experiment scripts, install the package dependencies first, then run any script with `--help` to inspect its CLI:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .

# After install, the experiment scripts accept --help for CLI documentation.
.venv/bin/python scripts/dark_subspace/behavioral_channels.py --help
.venv/bin/python scripts/dark_subspace/sae_dark_subspace.py --help
.venv/bin/python scripts/dark_subspace/subspace_ablation_eval.py --help
```

## Repository Layout

| Path | Contents |
| --- | --- |
| `results/dark_subspace/` | JSON records consumed by the verifier and referenced from the claim map. |
| `scripts/dark_subspace/` | Paper-specific experiment, aggregation, plotting, and verification scripts. |
| `scripts/dark_subspace/shell/` | SLURM wrappers for GPU jobs and multi-seed arrays. |
| `scripts/shared/` | Shared SAE training, plotting style, path bootstrap, and utility code. |
| `src/sae_mia_audit/` | Reusable Python package used by the paper scripts. |

Large inputs, generated checkpoints, and manuscript source are not included in the repository. Full reproduction expects the controlled fine-tuning corpora, model checkpoints, and SAE checkpoints at the paths documented in the relevant scripts.

## Verification

`scripts/dark_subspace/verify_claims.py` is the recommended first check. It validates the main paper numbers using only the JSON records under `results/dark_subspace/`:

- Channel-decomposition geometry and layer-sweep values.
- Main reconstruction-and-residual AUROC rows.
- Pythia-6.9B N=5 mixed-data SAE cohort (harmonised across the multi-seed runs).
- Pythia-12B three-seed mixed-data SAE replication.
- Norm-baseline and scaling-table values.
- Standard MIA probe results.

The verifier does not load models, require GPUs, submit SLURM jobs, access the network, or write files.

## Naming notes

The Claim-Source Map below uses paper terminology throughout. A small number of engineering labels survive in JSON path strings and a few script filenames; the table below maps each one back to the paper passage so reviewers can navigate directly.

| Label in path or filename | Paper passage |
| --- | --- |
| `behavioral_channels.py`, paths under `behavioral_channels/` | Channel decomposition, paper §3.2 ("Separating the knowledge channel from the recall channel") and Appendix §`app:bcd_details`. The paper-cited values bind to the per-model analysis layer fixed in advance in Appendix Table `tab:model_details` (selected before any SAE training by maximising cross-validated logistic membership AUROC on raw residual-stream activations; see Methods §3.1). The `best_membership_layer` / `best_membership_auroc` fields in `summary.json` are run-time diagnostics over the `--layers` sweep, not the paper-cited headline. |
| `errPC` in paths under `causal_ablation/p12b_errPC_K{5,10}/` | K-PC residual ablation, Appendix Table `tab:kpc_kten_cells`. The label names the engineering implementation (top-K right-singular vectors of the SAE reconstruction residual); the paper prose calls it "K-PC". |
| `per_row_bootstrap_kocl2.py` filename | Per-row paired bootstrap for the directional sign-test cohort, Appendix §`app:koc2_bootstrap`. The script filename uses `kocl2` as an engineering label; the paper appendix label is `koc2`. |
| `memcirc` in paths under `data/memcirc_ctrl_ft/`, `runs/sae/memcirc_*`, `bow_ceiling/memcirc_ctrl_ft/` | Earlier project label retained only in path strings for provenance. Current code paths use `dark_subspace/`. |

## Claim-Source Map

The table below maps each paper passage to the script and JSON source that reproduce the corresponding result. `results/dark_subspace/generated/` mirrors the relevant JSON files from the run tree, so review does not require the full `runs/` directory.

Rows are grouped into **main claims** (the headline argument: channel decomposition, the residual-above-reconstruction ordering and its replications, the alternative-explanation tests, the feature-intervention dissociation, and the double-dissociation finding) and **supporting claims and controls** (robustness, confound, and scope tests).

### Main claims

| Paper passage | What to verify | Source script | JSON source |
| --- | --- | --- | --- |
| Methods §3.2 ("Separating the knowledge channel from the recall channel"); Results §4.1; Appendix Table `tab:bcd_main` | Channel-decomposition geometry, recall-direction separation, stability across SAE seeds | `scripts/dark_subspace/behavioral_channels.py`, `scripts/dark_subspace/sae_noise_floor_aggregate.py` | `results/dark_subspace/generated/behavioral_channels/*/orthogonality.json`, `results/dark_subspace/generated/sae_noise_floor/p69_aggregate.json` |
| Results §4.2 ("SAE reconstruction fails to preserve membership signal recoverable from the residual"); Table `tab:dark_subspace` | Pythia-6.9B N=5 mixed-data SAE reconstruction reduction | `scripts/dark_subspace/p69_n5_harmonize.py` | `results/dark_subspace/paper_claims/p69_n5_harmonized_2026-05-06.json` |
| Appendix `app:p12b_replication`; Table `tab:dark_subspace` | Pythia-12B mixed-data SAE multi-seed replication. The paper reports a five-seed cohort; this artefact bundles the first three seeds (47, 48, 49). The remaining two seeds (50, 51) reproduce by rerunning the array script with the same range. | `scripts/dark_subspace/p12b_multiseed_query.py`, `scripts/dark_subspace/shell/sbatch_p12b_multiseed_array.sh` (covers seeds 47 to 51 via `--array=0-4`) | `results/dark_subspace/generated/sae_dark_subspace/p12b_mixed_sae_seed{47,48,49}/results.json` (bundled). Seeds 50 and 51 regenerate to the same path pattern on rerun. |
| Appendix `app:koc2_bootstrap` ("Paired Bootstrap for the Directional Sign-Test Settings") | Per-row residual-versus-original cohort bootstrap and one-sided binomial sign test on seven cohort rows (five inverting plus two anchors) plus two Qwen2 mult=4 secondary seeds reported separately | `scripts/dark_subspace/per_row_bootstrap_kocl2.py` | `results/dark_subspace/paper_claims/cohort_bootstrap.json` (`cohort_rows` array of seven entries; `qwen2_mult4_secondary_seeds` array for the two extra rows) |
| Results §4.3 ("Four alternative explanations fail to account for the residual signal"); Appendix `tab:fsc_values` | FSC against $S_K$ for classifier features and full dictionary, with random-subset null and feature-ablation controls | `scripts/dark_subspace/fsc_random_null.py`, `scripts/dark_subspace/feature_ablation_dark_subspace.py`, `scripts/dark_subspace/feature_ablation_random_k.py`, `scripts/dark_subspace/behavioral_channels.py` | `results/dark_subspace/generated/sae_dark_subspace/p69_feature_ablation/results.json`, `results/dark_subspace/generated/bcd_extractability/*/extractability_predictor.json`. The `behavioral_channels.py` `sae_alignment.json` output is regenerated on rerun and is not included in the repository. |
| Methods §3.5 / Results §4.4 (`tab:dd_full`, `tab:dd_extraction`): subspace erasure with downstream membership detection and verbatim extraction | Membership AUROC, mean member loss, exact-match rate, and extraction ROUGE-L under $S_K$/$S_R$ erasure (Methods Eq. 3) | `scripts/dark_subspace/dd_table_render.py` renders the table from per-cell records produced by the GPU pipeline (model load, basis fit, forward-pass erasure hook, decoded continuations, ROUGE-L scoring); see the script docstring for the per-cell record schema. | Regenerates to `results/dark_subspace/generated/double_dissociation/<model_tag>/results.json` on rerun; per-cell records are not bundled in this artefact. |
| Methods §3.5 ("Interventions separate extraction from detection"); Results §4.5; Appendix Table `tab:kpc_kten_cells` | K-PC residual ablation at K=10 and K=5 (with random-rotation, matched-Gaussian, and column-mask controls) | `scripts/dark_subspace/subspace_ablation_eval.py` | `results/dark_subspace/generated/causal_ablation/p12b_errPC_K10/results.json`, `results/dark_subspace/generated/causal_ablation_K5/p12b_errPC_K5/results.json` (paths use `errPC` as the engineering label for K-PC; see "Naming notes" above) |
| Results §4.4 ("Feature edits do not close the residual membership gap"); Figure `fig:privacy_aware`; Appendix Table `tab:fresh_probe_v2` | Privacy-aware SAE comparison ($d_K$-penalised SAE versus standard SAE) | `scripts/dark_subspace/finetune_sae_dk.py`, `scripts/dark_subspace/fresh_probe_test.py` | `results/dark_subspace/generated/sae_dark_subspace/p69_ft_dk{0.1,1.0}/results.json` |

### Supporting claims and controls

| Paper passage | What to verify | Source script | JSON source |
| --- | --- | --- | --- |
| Results §4.5; Appendix Table `tab:norm_direction` | Norm-direction baseline | `scripts/dark_subspace/norm_baseline.py` | `results/dark_subspace/generated/norm_baseline/*/results.json` |
| Appendix `tab:heldout_dk_per_split`, `app:heldout_dk_protocol` ("Held-Out Estimation Preserves Ordering but Reduces Magnitude") | Held-out partition-fit reductions for $d_K$ on Pythia-6.9B and Pythia-12B | `scripts/dark_subspace/heldout_dk_eval.py` | `results/dark_subspace/paper_claims/heldout_dk.json` |
| Appendix `tab:l2_normalized` ("L2-normalised residual membership AUROC") | Norm-versus-direction split of residual signal | `scripts/dark_subspace/l2_normalized_auroc.py` plus `scripts/dark_subspace/norm_baseline.py` | The activation-norm half of the table is reproducible from the bundled `norm_baseline/*/results.json`. The L2-normalised residual AUROC half regenerates to `results/dark_subspace/generated/l2_normalized/<model_tag>/results.json` on rerun against the matching SAE-residual scores; not bundled in this artefact. |
| Appendix `tab:epoch_dd` ("Double Dissociation Across Fine-Tuning Epochs") | Pythia-1B double dissociation across fine-tuning epochs | `scripts/dark_subspace/dd_table_render.py --table epoch_dd` | Regenerates to `results/dark_subspace/generated/double_dissociation_epochs/<model_tag>/results.json` on rerun; per-epoch records are not bundled in this artefact. |
| Appendix `tab:dynamics`, `app:training_dynamics` ("Recall channel emerges before knowledge channel during fine-tuning") | Pythia-1B epoch and pretraining-checkpoint dynamics | `scripts/dark_subspace/behavioral_channels.py` (per-checkpoint SLURM in `scripts/dark_subspace/shell/multiseed/sbatch_p1b.sh`) | `results/dark_subspace/generated/behavioral_channels/{p1b_epoch1,p1b_epoch3,p1b_epoch5}/orthogonality.json` |
| Appendix `tab:scaling`, `app:scaling` ("Scaling Curve") | Pythia-70M to Pythia-12B $\mathrm{score}_K$ scaling sweep | `scripts/dark_subspace/behavioral_channels.py` (per-model SLURM wrappers under `scripts/dark_subspace/shell/multiseed/`) | `results/dark_subspace/generated/behavioral_channels/{p70m_epoch5,p160m_epoch5,p410m_epoch5,p1b_epoch5,p2.8b_epoch5,p69_epoch5,p12b_epoch5}/orthogonality.json` |
| Results §4.1; Appendix `app:per_layer` | Pre-fine-tuning baseline and fine-tuned layer sweep | `scripts/dark_subspace/behavioral_channels.py` (driver: `scripts/dark_subspace/shell/sbatch_pre_ft_baseline.sh`) | `results/dark_subspace/generated/behavioral_channels/{p69_BASE_pre_ft,p69_epoch5_layer_sweep}/orthogonality.json` |
| Methods §3.7 ("Validity gate for quantitative claims"); Appendix bootstrap and control tables | Bootstrap replicate count (n_boot=10000) | `scripts/dark_subspace/subspace_ablation_eval.py`, `scripts/dark_subspace/per_row_bootstrap_kocl2.py`, `scripts/dark_subspace/rerun_bootstrap_cis.py` | Script arguments and bootstrap JSON metadata |
| Results §4.6 ("Confound and operating-point controls do not reverse the finding"); Appendix `app:bow_baseline` | Bag-of-words ceiling | `scripts/dark_subspace/bow_ceiling.py` | `results/dark_subspace/generated/bow_ceiling/memcirc_ctrl_ft/results.json` |
| Results §4.6; Appendix `app:tpr_paraphrase` | Word-order paraphrase orientation flip; paraphrase TPR at 1% and 5% FPR | `scripts/dark_subspace/paraphrase_sensitivity.py` | `results/dark_subspace/generated/paraphrase_sensitivity/{p69,qwen2,p12b}/results.json` |
| Appendix `tab:tpr_at_0p1pct_fpr`, `app:tpr_paraphrase` | TPR at 0.1% FPR for residual $d_K$ across four models | `scripts/dark_subspace/tpr_at_low_fpr.py` (consumes per-text score arrays from the matching `sae_dark_subspace.py` outputs) | Regenerates to `results/dark_subspace/generated/tpr_low_fpr/<model_tag>/results.json` on rerun; not bundled in this artefact. |
| Results §4.6; Appendix `app:standard_probes` | Standard published MIA probes (loss attack, MIN-K%, zlib) under reconstruction/residual decomposition | `scripts/dark_subspace/standard_mia_probe_decomposition.py` | `results/dark_subspace/generated/standard_mia_probes/p69_dark_subspace_replication/results.json` |
| Appendix `tab:nonlinear` ("Non-Linear Probing Comparison") | MLP-vs-linear probe AUROC at the analysis layer | `scripts/dark_subspace/nonlinear_probe.py` (consumes mean-pooled activations from `extract_canonical_activations.py`) | Regenerates to `results/dark_subspace/generated/nonlinear_probe/<model_tag>/results.json` on rerun; not bundled in this artefact. |
| Appendix `app:length_baseline` ("Length-Feature Baseline") | Length-feature membership classifier on the controlled split | `scripts/dark_subspace/length_baseline.py` (consumes the member and non-member text JSONLs; CPU-only) | Regenerates to `results/dark_subspace/generated/length_baseline/<model_tag>/results.json` on rerun; not bundled in this artefact. |
| Appendix `app:label_shuffled_null` ("Label-Shuffled Permutation Test") | Permutation null on $\cos(d_K, d_R)$ under shuffled member/non-member labels | `scripts/dark_subspace/label_shuffled_null.py` (consumes mean-pooled activations and the saved $d_R$ from the channel-decomposition output) | Regenerates to `results/dark_subspace/generated/label_shuffled_null/<model_tag>/results.json` on rerun; not bundled in this artefact. |
| Appendix `app:corpus_disjoint` ("Corpus-Disjoint Dictionary Control") | Pythia-6.9B mixed-data SAE retrained on an OWT partition disjoint from the evaluation pool | `scripts/dark_subspace/build_disjoint_owt_corpus.py` (corpus prep) plus `scripts/dark_subspace/sae_dark_subspace.py` (driver: `scripts/dark_subspace/shell/sbatch_p69_disjoint_owt_sae.sh`) | Regenerates to `runs/dark_subspace/sae_dark_subspace/p69_disjoint_owt_seed${SEED}/results.json` on rerun; not bundled in this artefact. |
| Appendix `app:additional_controls` — random-direction baseline | 100 random unit-direction membership AUROC per model | `scripts/dark_subspace/random_direction_baseline.py` (driver: `scripts/dark_subspace/shell/sbatch_random_direction_baseline.sh`) | Regenerates to `results/dark_subspace/generated/random_direction/<model_tag>/results.json` on rerun; not bundled in this artefact. |
| Appendix `app:additional_controls` — random-init SAE | Replacing the trained SAE with a randomly initialised SAE on Pythia-6.9B layer 16 | `scripts/dark_subspace/make_random_sae.py` plus `scripts/dark_subspace/sae_dark_subspace.py` (driver: `scripts/dark_subspace/shell/sbatch_random_init_sae.sh`) | Regenerates to `results/dark_subspace/generated/sae_dark_subspace/p69_random_init_sae/results.json` on rerun; not bundled in this artefact. |
| Appendix `app:additional_controls` — pre-fine-tuning control | Channel decomposition on the un-fine-tuned base Pythia-6.9B | `scripts/dark_subspace/behavioral_channels.py` (driver: `scripts/dark_subspace/shell/sbatch_pre_ft_baseline.sh`) | `results/dark_subspace/generated/behavioral_channels/p69_BASE_pre_ft/orthogonality.json` |

### Configuration-audit tables

Two paper tables summarise hyperparameter and SAE-quality settings rather than experimental results. The underlying training runs are produced by `scripts/shared/train_sae.py` via the SLURM wrappers under `scripts/dark_subspace/shell/`; the tables themselves are hand-assembled audits of those settings.

| Paper passage | Source |
| --- | --- |
| `tab:per_model_hps` (per-model SAE hyperparameter audit) | Compiled from the per-model `scripts/dark_subspace/shell/multiseed/sbatch_*.sh` wrappers, which record the model checkpoint, layer, dictionary multiplier, L1 coefficient, and training-token budget actually used. |
| `tab:sae_gate_audit` (Qwen2 mixed-data SAE audit, Appendix `app:qwen2_pilot`) | Compiled from the Qwen2 SAE training runs documented in `scripts/dark_subspace/shell/multiseed/sbatch_qwen2_mult8.sh`. |

### Figures

Figures are regenerated from the JSON tree under `results/dark_subspace/` by the plotting scripts under `scripts/dark_subspace/`; the rendered figure files themselves are not in the repository.

| Figure(s) | Producing script |
| --- | --- |
| `fig:score_distributions` | `scripts/dark_subspace/plot_score_distributions.py` |
| `fig:privacy_aware` | `scripts/dark_subspace/plot_privacy_aware_comparison.py` |
| `fig:layer_trajectories`, `fig:arch_and_scaling`, `fig:dark_subspace_heatmap`, `fig:fsc`, `fig:epoch`, `fig:norm_direction`, `fig:layer_heatmap`, `fig:sae_quality_scatter` | `scripts/dark_subspace/plot_figures.py`, `scripts/dark_subspace/plot_advanced_figures.py` |

## Full Reproduction

The GPU pipeline is split by experiment class rather than collapsed into one monolithic driver. This makes individual controls and reruns auditable without coupling unrelated jobs.

Useful entry points (after `pip install -e .`):

```bash
# Recompute the channel decomposition for a configured model.
python3 scripts/dark_subspace/behavioral_channels.py --help

# Run the SAE reconstruction/residual decomposition.
python3 scripts/dark_subspace/sae_dark_subspace.py --help

# Run the K-PC causal ablation.
python3 scripts/dark_subspace/subspace_ablation_eval.py --help

# Regenerate figure data from JSONs.
python3 scripts/dark_subspace/figure_data_loader.py
```

Figure plotting scripts write to `outputs/figures/` by default; set `FIGDIR=/path/to/figures` to redirect output to a separate manuscript directory. SLURM wrappers in `scripts/dark_subspace/shell/` document the cluster commands used for the main multi-seed and control jobs. The wrappers write outputs under `runs/dark_subspace/` and SAE checkpoints under `runs/sae/`.

Some historical corpus and checkpoint labels still contain `memcirc` in paths such as `data/memcirc_ctrl_ft/` or `runs/sae/memcirc_*`. Those names are retained because they are embedded in earlier run provenance records. The current code and result layout use `dark_subspace`.

## Scope Notes

- The verifier checks numerical consistency against cached JSON records; it is not a substitute for rerunning model training.
- Full reproduction requires controlled corpora, fine-tuned checkpoints, SAE checkpoints, and GPU resources.
- Mistral-7B and Llama-3-8B SAE rows are retained as documented exclusions where reconstruction-quality gates failed.
- Gemma-2-2B is reported in `tab:dark_subspace` (single seed, `app:gemma_stretch`); the JSON records and verifier do not cover the Gemma row directly. Gemma is referenced through the qualitative cross-architecture set; a Gemma-specific SAE result JSON is not included in `results/dark_subspace/generated/sae_dark_subspace/`.
- The controlled fine-tuning corpus is OpenWebText-derived; cross-corpus generalisation is outside the scope of this artefact.

## Citation

```bibtex
@inproceedings{anonymous2026darksubspace,
  title  = {The Dark Subspace of Fine-Tuning Memorisation},
  author = {Anonymous},
  booktitle = {ICML 2026 Workshop on Mechanistic Interpretability},
  year   = {2026},
  note   = {Anonymous double-blind submission}
}
```

## License

MIT. See `LICENSE`.
