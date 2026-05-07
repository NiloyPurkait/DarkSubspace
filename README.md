# The Dark Subspace of Fine-Tuning Memorisation

Anonymous reviewer artifact for an ICML 2026 Mechanistic Interpretability Workshop submission.

This repository contains the paper-specific experiment code, a curated set of generated JSON results, and a CPU-only verifier that checks the paper-cited numerical claims against the shipped JSONs. The manuscript is distributed separately for review and is not included in this code/results artifact.

## Quickstart

The verifier uses only the Python standard library and reads the JSON files under `results/dark_subspace/`.

```bash
python3 scripts/dark_subspace/verify_claims.py
```

Expected result (the asserted-check count grows as additional checks are added; the precise count is whatever `verify_claims.py` itself reports, and all checks should pass):

```text
ASSERTED CHECK SUMMARY: N/N PASS, 0 FAIL
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
| `results/dark_subspace/` | Shipped JSON records used by the verifier and the claim map. |
| `scripts/dark_subspace/` | Paper-specific experiment, aggregation, plotting, and verification scripts. |
| `scripts/dark_subspace/shell/` | SLURM wrappers for GPU jobs and multi-seed arrays. |
| `scripts/shared/` | Shared SAE training, plotting style, path bootstrap, and utility code. |
| `src/sae_mia_audit/` | Reusable Python package used by the paper scripts. |

Large inputs, generated checkpoints, and manuscript source are not included in the repository. Full reproduction expects the controlled fine-tuning corpora, model checkpoints, and SAE checkpoints at the paths documented in the relevant scripts.

## Reviewer Verification

`scripts/dark_subspace/verify_claims.py` is the intended first check. It validates the main paper numbers using only shipped JSONs:

- BCD geometry and layer-sweep values.
- Main reconstruction/residual AUROC rows.
- Pythia-6.9B harmonized N=5 mixed-SAE cohort.
- Pythia-12B three-init mixed-SAE replication.
- Norm-baseline and scaling-table source values.
- Standard MIA probe result availability.

The verifier does not load models, require GPUs, submit SLURM jobs, access the network, or write files.

## Glossary

The repository uses several internal labels that map to terminology in the paper. The mapping is:

| Repo-internal term | Paper terminology |
| --- | --- |
| `BCD` / `behavioral_channels` | Channel decomposition (\S3.2 of the paper, "Separating the knowledge channel from the recall channel"); produces $\SK$/$\SR$ subspaces and $\dK$/$\dR$ directions. |
| `K-OC-2` | The directional sign-test cohort used in Appendix `app:koc2_bootstrap` ("Paired Bootstrap for the Directional Sign-Test Settings"). The shipped `cohort_bootstrap.json` records seven cohort rows: five inverting rows that enter the binomial sign test plus two anchor rows (Pythia-6.9B N=6 anchor and Pythia-12B seed 47), with two additional Qwen2 mult=4 secondary fine-tune seeds reported separately. |
| `errPC` | K-PC residual ablation (`tab:kpc_kten_cells`); strips the top-K right-singular vectors of the reconstruction residual. |
| `OC` | Orthogonal-complement (used in scripts that project onto the complement of a subspace). |
| `memcirc` | Historical campaign label retained only in path strings such as `data/memcirc_ctrl_ft/` and `runs/sae/memcirc_*` for provenance traceability. Reviewer-facing code paths use `dark_subspace`. |

## Claim-Source Map

The table below connects each paper passage to the script and shipped JSON source that reproduce the corresponding result. `results/dark_subspace/generated/` mirrors the relevant JSON leaves from the original run tree, so review does not require the full ignored `runs/` directory.

| Paper passage | What to verify | Source script | Shipped JSON source |
| --- | --- | --- | --- |
| Methods: "Separating the knowledge channel from the recall channel"; Results: "Membership and recall directions are weakly aligned"; Appendix Table `tab:bcd_main` | BCD geometry, recall direction separation, multi-seed BCD stability | `scripts/dark_subspace/behavioral_channels.py`, `scripts/dark_subspace/sae_noise_floor_aggregate.py` | `results/dark_subspace/generated/behavioral_channels/*/orthogonality.json`, `results/dark_subspace/generated/sae_noise_floor/p69_aggregate.json` |
| Results: "SAE reconstruction fails to preserve membership signal recoverable from the residual"; Table `tab:dark_subspace` | Pythia-6.9B N=5 mixed-SAE drop | `scripts/dark_subspace/p69_n5_harmonize.py` | `results/dark_subspace/paper_claims/p69_n5_harmonized_2026-05-06.json` |
| Appendix: "Pythia-12B Replication Detail"; Table `tab:dark_subspace` | Pythia-12B mixed-SAE multi-init replication | `scripts/dark_subspace/p12b_multiseed_query.py`, `scripts/dark_subspace/shell/sbatch_p12b_multiseed_array.sh` | `results/dark_subspace/generated/sae_dark_subspace/p12b_mixed_sae_seed{47,48,49}/results.json` |
| Appendix: "Paired Bootstrap for the Directional Sign-Test Settings" (`app:koc2_bootstrap`) | Per-row residual-versus-original cohort bootstrap | `scripts/dark_subspace/per_row_bootstrap_kocl2.py` | `results/dark_subspace/paper_claims/cohort_bootstrap.json` |
| Results: "Four alternative explanations fail to account for the residual signal"; Appendix: "Feature Sufficiency Criterion Values" (`tab:fsc_values`) | Feature sufficiency and feature-ablation controls | `scripts/dark_subspace/fsc_random_null.py`, `scripts/dark_subspace/feature_ablation_dark_subspace.py`, `scripts/dark_subspace/feature_ablation_random_k.py` | `results/dark_subspace/generated/sae_dark_subspace/p69_feature_ablation/results.json` |
| Results: "Feature edits do not close the residual membership gap"; Figure `fig:privacy_aware`; Appendix Table `tab:fresh_probe_v2` | Privacy-aware SAE comparison | `scripts/dark_subspace/finetune_sae_dk.py`, `scripts/dark_subspace/fresh_probe_test.py` | `results/dark_subspace/generated/sae_dark_subspace/p69_ft_dk{0.1,1.0}/results.json` |
| Methods: "Interventions separate extraction from detection"; Results: "Residual signal is partition-sensitive and partially concentrated"; Appendix Table `tab:kpc_kten_cells` | K-PC (errPC) residual ablation at K=10 and K=5 (with random-rotation, matched-Gaussian, and column-mask controls) | `scripts/dark_subspace/subspace_ablation_eval.py` | `results/dark_subspace/generated/causal_ablation/p12b_errPC_K10/results.json`, `results/dark_subspace/generated/causal_ablation_K5/p12b_errPC_K5/results.json` |
| Results: "Residual signal is partition-sensitive and partially concentrated"; Appendix Table `tab:norm_direction` | Norm-direction baseline | `scripts/dark_subspace/norm_baseline.py` | `results/dark_subspace/generated/norm_baseline/*/results.json` |
| Results: "Confound and operating-point controls do not reverse the finding"; Appendix `app:bow_baseline` | Bag-of-words ceiling | `scripts/dark_subspace/bow_ceiling.py` | `results/dark_subspace/generated/bow_ceiling/memcirc_ctrl_ft/results.json` |
| Results: "Membership and recall directions are weakly aligned"; Appendix `app:per_layer` | Pre-FT baseline and FT layer sweep | `scripts/dark_subspace/behavioral_channels.py` (driver: `scripts/dark_subspace/shell/sbatch_pre_ft_baseline.sh`) | `results/dark_subspace/generated/behavioral_channels/{p69_BASE_pre_ft,p69_epoch5_layer_sweep}/orthogonality.json` |
| Results: "Confound and operating-point controls do not reverse the finding"; Appendix `app:tpr_paraphrase` | Word-order paraphrase orientation flip; paraphrase TPR at 1% and 5% FPR | `scripts/dark_subspace/paraphrase_sensitivity.py` | `results/dark_subspace/generated/paraphrase_sensitivity/{p69,qwen2,p12b}/results.json` |
| Appendix `app:tpr_paraphrase` (`tab:tpr_at_0p1pct_fpr`) | TPR at 0.1% FPR for residual $\dK$ across four models | `scripts/dark_subspace/tpr_at_low_fpr.py` | Not bundled in `results/dark_subspace/generated/`; values reported as ledger from the broader campaign. |
| Results: "Confound and operating-point controls do not reverse the finding"; Appendix `app:standard_probes` | Standard published MIA probes (loss attack, MIN-K%, zlib) under reconstruction/residual decomposition | `scripts/dark_subspace/standard_mia_probe_decomposition.py` | `results/dark_subspace/generated/standard_mia_probes/p69_dark_subspace_replication/results.json` |
| Methods: "Validity gate for quantitative claims" (`app:per_model_hps_detail`); Appendix bootstrap/control tables | Bootstrap-count disclosure (n_boot=10000) | `scripts/dark_subspace/subspace_ablation_eval.py`, `scripts/dark_subspace/per_row_bootstrap_kocl2.py`, `scripts/dark_subspace/rerun_bootstrap_cis.py` | Script arguments and shipped bootstrap JSON metadata |
| Appendix: "Held-Out Estimation Preserves Ordering but Reduces Magnitude" (`tab:heldout_dk_per_split`, `app:heldout_dk_protocol`) | Held-out partition-fit reductions for $\dK$ on Pythia-6.9B and Pythia-12B | `scripts/dark_subspace/heldout_dk_eval.py` | `results/dark_subspace/paper_claims/heldout_dk.json` |
| Appendix: "Scaling Curve" (`tab:scaling`, `app:scaling`) | Pythia-70M to Pythia-12B $\mathrm{score}_K$ scaling sweep | `scripts/dark_subspace/behavioral_channels.py` (per-model SLURM wrappers under `scripts/dark_subspace/shell/multiseed/`) | `results/dark_subspace/generated/behavioral_channels/{p70m_epoch5,p160m_epoch5,p410m_epoch5,p1b_epoch5,p2.8b_epoch5,p69_epoch5,p12b_epoch5}/orthogonality.json` |
| Appendix: "Recall channel emerges before knowledge channel during fine-tuning" (`tab:dynamics`, `app:training_dynamics`) | Pythia-1B epoch and pretraining-checkpoint dynamics | `scripts/dark_subspace/behavioral_channels.py` (per-checkpoint SLURM in `scripts/dark_subspace/shell/multiseed/sbatch_p1b.sh`) | `results/dark_subspace/generated/behavioral_channels/{p1b_epoch1,p1b_epoch3,p1b_epoch5}/orthogonality.json` |
| Appendix: "Feature Sufficiency Criterion Values" (`tab:fsc_values`) | FSC against $\SK$ for classifier features and full dictionary, with random subset null | `scripts/dark_subspace/fsc_random_null.py`, `scripts/dark_subspace/behavioral_channels.py` | `results/dark_subspace/generated/bcd_extractability/*/extractability_predictor.json` (FSC source values); the `behavioral_channels.py` `sae_alignment.json` output is generated locally on rerun and is not bundled. |
| Appendix: "L2-normalised residual membership AUROC" (`tab:l2_normalized`) | Norm-versus-direction split of residual signal | `scripts/dark_subspace/l2_normalized_auroc.py` plus `scripts/dark_subspace/norm_baseline.py` | Not bundled in `results/dark_subspace/generated/`; the `norm_baseline/*/results.json` files contain only the per-layer activation-norm AUROC. The L2-normalised residual values are reported as a ledger from the broader campaign. |
| Appendix: "Corpus-Disjoint Dictionary Control" (`app:corpus_disjoint`) | Pythia-6.9B mixed-data SAE retrained on an OWT partition disjoint from the evaluation pool | `scripts/dark_subspace/build_disjoint_owt_corpus.py` (corpus prep) plus `scripts/dark_subspace/sae_dark_subspace.py` (driver: `scripts/dark_subspace/shell/sbatch_p69_disjoint_owt_sae.sh`, output dir `runs/dark_subspace/sae_dark_subspace/p69_disjoint_owt_seed${SEED}/`) | Not bundled in `results/dark_subspace/generated/`; values reported as ledger from the broader campaign. |
| Appendix: "Additional Controls" (`app:additional_controls`): random-direction baseline | 100 random unit-direction membership AUROC per model | `scripts/dark_subspace/random_direction_baseline.py` (driver: `scripts/dark_subspace/shell/sbatch_random_direction_baseline.sh`) | Not bundled in `results/dark_subspace/generated/`; values reported as ledger from the broader campaign. |
| Appendix: "Additional Controls" (`app:additional_controls`): random-init SAE | Replacing the trained SAE with a randomly initialised SAE on Pythia-6.9B layer 16 | `scripts/dark_subspace/make_random_sae.py` plus `scripts/dark_subspace/sae_dark_subspace.py` (driver: `scripts/dark_subspace/shell/sbatch_random_init_sae.sh`) | Not bundled in `results/dark_subspace/generated/`; values reported as ledger from the broader campaign. |
| Appendix: "Additional Controls" (`app:additional_controls`): pre-FT control | Channel-decomposition probe on the un-fine-tuned base Pythia-6.9B | `scripts/dark_subspace/behavioral_channels.py` (driver: `scripts/dark_subspace/shell/sbatch_pre_ft_baseline.sh`) | `results/dark_subspace/generated/behavioral_channels/p69_BASE_pre_ft/orthogonality.json` |

### Tables and figures with no shipped producing script

The following paper tables document protocols whose producing scripts are not part of this artefact. Reviewers should treat the values as ledger/reported numbers reproduced from the broader campaign rather than as artefacts regenerable from this code drop. The audit was performed by reading every script under `scripts/dark_subspace/`.

| Paper passage | Status |
| --- | --- |
| `tab:dd_full`, `tab:dd_extraction` (Methods Eq. 3 subspace erasure on full activations, with downstream loss/exact-match/ROUGE-L generation) | Not reproducible from shipped scripts. The shipped `scripts/dark_subspace/subspace_ablation_eval.py` performs only the K-PC residual ablation; it does not perform the erasure-then-generation pipeline that produces mean member loss, exact-match rate, and extraction ROUGE-L under $\SK$/$\SR$ erasure. The values are reported as a ledger from the broader campaign. |
| `tab:epoch_dd` (Pythia-1B double dissociation across epochs) | Not reproducible from shipped scripts; same generation pipeline as above. |
| `tab:nonlinear` (MLP probe vs linear probe) | Not reproducible from shipped scripts. The MLP-probe runner is not in this artefact. Values reported as a ledger from the broader campaign. |
| `tab:per_model_hps` (per-model SAE hyperparameter audit) | Manual ledger of the SAE training configurations recorded in the `runs/sae/*` directory naming convention; not produced by a single script. The `scripts/dark_subspace/shell/multiseed/` wrappers document the per-model settings actually used. |
| `tab:sae_gate_audit` (Qwen2 mixed-data audit ledger) | Manual ledger of the SAE-quality outcomes documented in Appendix `app:qwen2_pilot`. The underlying SAE runs are produced by `scripts/shared/train_sae.py` via the SLURM wrappers, but the table itself is a hand-assembled summary. |
| `app:length_baseline` (length-feature classifier on the controlled split) | Not reproducible from shipped scripts; the length scorer used (see `src/sae_mia_audit/methods/baselines.py::score_length`, `score_word_count`) is in the package, but no top-level driver shipped for this table. |
| `app:label_shuffled_null` (label-shuffled cosine permutation null) | Not reproducible from shipped scripts; values reported as a ledger from the broader campaign. |
| Figure pointers (`fig:privacy_aware`, `fig:score_distributions`, `fig:layer_trajectories`, `fig:arch_and_scaling`, `fig:dark_subspace_heatmap`, `fig:fsc`, `fig:epoch`, `fig:norm_direction`, `fig:layer_heatmap`, `fig:sae_quality_scatter`) | Reproduced from shipped JSONs by `scripts/dark_subspace/plot_figures.py`, `plot_advanced_figures.py`, `plot_privacy_aware_comparison.py`, `plot_score_distributions.py`. Figures themselves are not bundled. |

## Full Reproduction

The GPU pipeline is split by experiment class rather than collapsed into one monolithic driver. This makes individual controls and reruns auditable without coupling unrelated jobs.

Useful entry points (after `pip install -e .`):

```bash
# Recompute BCD geometry for a configured model.
python3 scripts/dark_subspace/behavioral_channels.py --help

# Run the SAE reconstruction/residual decomposition.
python3 scripts/dark_subspace/sae_dark_subspace.py --help

# Run the K-PC causal ablation.
python3 scripts/dark_subspace/subspace_ablation_eval.py --help

# Regenerate figure data from JSONs.
python3 scripts/dark_subspace/figure_data_loader.py
```

Figure plotting scripts write generated figures under `outputs/figures/` by default; set `FIGDIR=/path/to/figures` to target a manuscript checkout. SLURM wrappers in `scripts/dark_subspace/shell/` document the cluster commands used for the main multi-seed and control jobs. The wrappers create generated outputs under `runs/dark_subspace/` and SAE checkpoints under `runs/sae/`.

Some historical corpus and checkpoint labels still contain `memcirc` in paths such as `data/memcirc_ctrl_ft/` or `runs/sae/memcirc_*`. Those names are retained because they are embedded in provenance records from the experiment campaign. The current code and result layout use `dark_subspace`.

## Scope Notes

- The shipped verifier checks numerical consistency against cached JSONs; it is not a substitute for rerunning model training.
- Full reproduction requires controlled corpora, fine-tuned checkpoints, SAE checkpoints, and GPU resources.
- Mistral-7B and Llama-3-8B SAE rows are retained as documented exclusions where reconstruction-quality gates failed.
- Gemma-2-2B is reported in `tab:dark_subspace` (single seed, `app:gemma_stretch`); the shipped JSONs and verifier do not cover the Gemma row directly. Gemma is referenced through the qualitative cross-architecture set; a Gemma-specific SAE result JSON is not bundled in `results/dark_subspace/generated/sae_dark_subspace/`.
- The controlled fine-tuning corpus is OpenWebText-derived; cross-corpus generalization is outside the artifact scope.

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
