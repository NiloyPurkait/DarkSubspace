# The Dark Subspace of Fine-Tuning Memorisation

Anonymous reviewer artifact for an ICML 2026 Mechanistic Interpretability Workshop submission.

This repository contains the paper-specific experiment code, a curated set of generated JSON results, and a CPU-only verifier that checks the paper-cited numerical claims against the shipped JSONs. The manuscript is distributed separately for review and is not included in this code/results artifact.

## Quickstart

The verifier uses only the Python standard library and reads the JSON files under `results/dark_subspace/`.

```bash
python3 scripts/dark_subspace/verify_claims.py
```

Expected result:

```text
ASSERTED CHECK SUMMARY: 35/35 PASS, 0 FAIL
All asserted checks pass within tolerance.
```

For full experiment scripts, install the package dependencies first:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

## Repository Layout

| Path | Contents |
| --- | --- |
| `results/dark_subspace/` | Shipped JSON records used by the verifier and reviewer-facing claim map. |
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

## Claim-Source Map

The table below connects each reviewer-facing paper passage to the script and shipped JSON source that reproduce the corresponding result. `results/dark_subspace/generated/` mirrors the relevant JSON leaves from the original run tree, so review does not require the full ignored `runs/` directory.

| Paper passage | What to verify | Source script | Shipped JSON source |
| --- | --- | --- | --- |
| Methods: "Separating membership knowledge from recall behavior"; Results: "Membership and recall directions are weakly aligned"; Appendix Table `tab:bcd_main` | BCD geometry, recall direction separation, multi-seed BCD stability | `scripts/dark_subspace/behavioral_channels.py`, `scripts/dark_subspace/sae_noise_floor_aggregate.py` | `results/dark_subspace/generated/behavioral_channels/*/orthogonality.json`, `results/dark_subspace/generated/sae_noise_floor/p69_aggregate.json` |
| Results: "SAE reconstruction fails to preserve membership signal recoverable from the residual"; Table `tab:dark_subspace` | Pythia-6.9B N=5 mixed-SAE drop | `scripts/dark_subspace/p69_n5_harmonize.py` | `results/dark_subspace/paper_claims/p69_n5_harmonized_2026-05-06.json` |
| Appendix: "Pythia-12B Replication Detail"; Table `tab:dark_subspace` | Pythia-12B mixed-SAE three-init replication | `scripts/dark_subspace/shell/sbatch_p12b_multiseed_array.sh` | `results/dark_subspace/generated/sae_dark_subspace/p12b_mixed_sae_seed{47,48,49}/results.json` |
| Appendix: "Per-Row Bootstrap on the K-OC-2 Cohort" | Per-row residual-versus-original cohort bootstrap | `scripts/dark_subspace/per_row_bootstrap_kocl2.py` | `results/dark_subspace/paper_claims/cohort_bootstrap.json` |
| Results: "Decoder span, sparse codes, residual norm, and top features do not explain the residual signal"; Appendix: "Feature Sufficiency Criterion Values" | Feature sufficiency and feature-ablation controls | `scripts/dark_subspace/fsc_random_null.py`, `scripts/dark_subspace/feature_ablation_dark_subspace.py` | `results/dark_subspace/generated/sae_dark_subspace/p69_feature_ablation/results.json` |
| Results: "Feature edits do not close the residual membership gap"; Figure `fig:privacy_aware`; Appendix Table `tab:fresh_probe_v2` | Privacy-aware SAE comparison | `scripts/dark_subspace/finetune_sae_dk.py`, `scripts/dark_subspace/fresh_probe_test.py` | `results/dark_subspace/generated/sae_dark_subspace/p69_ft_dk{0.1,1.0}/results.json` |
| Methods: "Interventions separate extraction from detection"; Results: "Geometry and concentration of the residual signal"; Appendix Table `tab:kpc_kten_cells` | K-PC causal ablation at K=10 and K=5 | `scripts/dark_subspace/subspace_ablation_eval.py` | `results/dark_subspace/generated/causal_ablation/p12b_errPC_K10/results.json`, `results/dark_subspace/generated/causal_ablation_K5/p12b_errPC_K5/results.json` |
| Results: "Geometry and concentration of the residual signal"; Appendix Table `tab:norm_direction` | Norm-direction baseline | `scripts/dark_subspace/norm_baseline.py` | `results/dark_subspace/generated/norm_baseline/*/results.json` |
| Results: "Robustness to confound and operating point controls"; Appendix: "Bag-of-Words Vocabulary Baseline" | Bag-of-words ceiling | `scripts/dark_subspace/bow_ceiling.py` | `results/dark_subspace/generated/bow_ceiling/memcirc_ctrl_ft/results.json` |
| Results: "Membership and recall directions are weakly aligned"; Appendix: "Per-Layer Decomposition Tables" | Pre-FT baseline and FT layer sweep | `scripts/dark_subspace/behavioral_channels.py` | `results/dark_subspace/generated/behavioral_channels/{p69_BASE_pre_ft,p69_epoch5_layer_sweep}/orthogonality.json` |
| Results: "Robustness to confound and operating point controls"; Appendix: "TPR at Low FPR and Paraphrase Diagnostic" | Word-order paraphrase orientation flip | `scripts/dark_subspace/paraphrase_sensitivity.py` | `results/dark_subspace/generated/paraphrase_sensitivity/{p69,qwen2,p12b}/results.json` |
| Results: "Robustness to confound and operating point controls"; Appendix: "Standard Probe Replication and Output-Layer Scope" | Standard published MIA probes | `scripts/dark_subspace/standard_mia_probe_decomposition.py` | `results/dark_subspace/generated/standard_mia_probes/p69_dark_subspace_replication/results.json` |
| Methods: "Validity gate for quantitative claims"; Appendix bootstrap/control tables | Bootstrap-count disclosure | `scripts/dark_subspace/subspace_ablation_eval.py`, `scripts/dark_subspace/per_row_bootstrap_kocl2.py` | Script arguments and shipped bootstrap JSON metadata |

## Full Reproduction

The GPU pipeline is split by experiment class rather than collapsed into one monolithic driver. This makes individual controls and reruns auditable without coupling unrelated jobs.

Useful entry points:

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

Some historical corpus and checkpoint labels still contain `memcirc` in paths such as `data/memcirc_ctrl_ft/` or `runs/sae/memcirc_*`. Those names are retained because they are embedded in provenance records from the experiment campaign. The reviewer-facing code and result layout use `dark_subspace`.

## Scope Notes

- The shipped verifier checks numerical consistency against cached JSONs; it is not a substitute for rerunning model training.
- Full reproduction requires controlled corpora, fine-tuned checkpoints, SAE checkpoints, and GPU resources.
- Mistral-7B and Llama-3-8B SAE rows are retained as documented exclusions where reconstruction-quality gates failed.
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
