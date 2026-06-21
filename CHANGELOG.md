# Changelog

All notable changes to this artifact are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] - 2026-06-21 - Supplementary heatmap reconciled to Table 2

### Fixed

- Corrected the Pythia-6.9B (mixed) residual cell of the supplementary heatmap
  (`assets/figures/dark_subspace_heatmap.{pdf,png}`) from 0.779 to 0.781 to match
  the camera-ready paper and Table 2. The cell now sources the canonical N=5
  harmonised cohort (`results/dark_subspace/paper_claims/p69_n5_harmonized_2026-05-06.json`,
  Pattern A drop seed 47, residual mean 0.780748) instead of the N=6 noise-floor
  aggregate (residual mean 0.778729). This is one of four Pythia rows reconciled
  to the N=5 cohort values below.
- Repointed `scripts/dark_subspace/figure_data_loader.py` so the mixed-row routing
  reads the N=5 harmonised cohort rather than the N=6 noise-floor aggregate.
- Repointed `scripts/dark_subspace/p69_n5_harmonize.py` base/aggregate paths to the
  release `results/dark_subspace/generated/...` layout (the raw `runs/...` tree is
  not shipped) with an in-repo fallback to the committed harmonised record, and made
  it write to a non-committed verification file by default so it never overwrites the
  committed JSON. It reproduces the residual mean 0.780748 (non-null).
- Corrected the Pythia-1B row of the supplementary heatmap from the single-seed
  values (0.660/0.515/0.677) to the camera-ready Table-2 N=5 mixed-SAE cohort
  (0.663/0.593/0.670). Shipped the previously-absent five per-seed records
  `results/dark_subspace/generated/sae_dark_subspace/p1b_mixed_sae_seed{42..46}/results.json`
  (depth-matched layer 14; orig/recon/resid/cos means 0.662941/0.592533/0.670307/0.868798)
  and added a `load_p1b_mixed_n5` path to `scripts/dark_subspace/figure_data_loader.py`
  (mirrors `load_p12b_mixed_n5`), routing the Pythia-1B row to it. This supersedes the
  single-seed `p1b_epoch5` member-only source.
- Regenerated the full supplementary heatmap from the real
  `scripts/dark_subspace/plot_advanced_figures.py::fig1_dark_subspace_heatmap()` so
  every row is sourced from `figure_data_loader.py`. The Pythia-6.9B (member-only),
  Pythia-6.9B (mixed), Pythia-1B, and Pythia-12B rows now show their N=5 cohort values
  (0.803/0.633/0.833, 0.803/0.594/0.781, 0.663/0.593/0.670, 0.764/0.612/0.723),
  matching Table 2. A pypdf 27-cell audit confirms all 9 rows now match Table 2 and the
  monorepo figure `manuscript/figures/dark_subspace_heatmap.pdf` cell-for-cell
  (paper == artifact).

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
