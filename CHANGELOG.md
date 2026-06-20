# Changelog

All notable changes to this artifact are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2026-06-20 - Camera-ready (ICML MI Workshop 2026, Spotlight)

The submission-to-camera-ready delta. The double-blind submission artefact was
de-anonymised, two reviewer-requested scope tests were added, and the release was
packaged for public use.

### Changed

- De-anonymised the authors, venue, citation, and license metadata for the
  camera-ready release.
- Corrected the Claim-Source Map mapping for the per-row paired-bootstrap table
  so that it points to the Residual-minus-Reconstruction margins.
- Standardised the interpreter path in script docstrings to `.venv/bin/python`.

### Added

- TopK SAE scope test (Appendix `app:topk_scope`, reviewer w72z) via
  `scripts/dark_subspace/aggregate_topk_scope.py` with its SLURM wrappers
  (`scripts/dark_subspace/shell/sbatch_topk_p69_scope_array.sh` and
  `sbatch_topk_p69_scope_eval_array.sh`), summarised in
  `results/dark_subspace/generated/topk_scope/cluster_summary.json`.
- Frikha and PrivacyScalpel feature-selection audit (Appendix
  `app:frikha_features`, reviewer w72z) via
  `scripts/dark_subspace/frikha_baseline_ablation.py` and
  `scripts/dark_subspace/frikha_n5_aggregate.py` with their SLURM wrappers,
  summarised in
  `results/dark_subspace/generated/frikha_features/cluster_summary.json`.
- TopK SAE training support in `scripts/shared/train_sae.py` and
  `src/sae_mia_audit/sae/sae.py`, plus the new modules
  `src/sae_mia_audit/sae/topk.py` and `src/sae_mia_audit/sae/interpret.py` that
  the release `train_sae.py` already imported.
- Hero figure generator and static camera-ready figures under `assets/figures/`.
- `DATA.md` provisioning doc, `Makefile`, `requirements-lock.txt`, and a package
  install and import CI smoke job.

## [0.1.0] - 2026-05-08 - Initial double-blind submission artefact
