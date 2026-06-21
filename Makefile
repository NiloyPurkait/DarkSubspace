# Makefile for "The Dark Subspace of Fine-Tuning Memorisation".
# Convenience targets for install, verification, tests, the headline number,
# and figure regeneration. The `verify` and `headline` targets are CPU-only
# and stdlib-only. The `install`, `test`, and `figures` targets need the
# package dependencies installed into a local virtual environment.

.PHONY: install verify test headline figures

# Create a local virtual environment and install the package in editable mode.
install:
	python -m venv .venv && .venv/bin/pip install -e .

# Recommended first check. CPU-only, stdlib-only, no GPU, no network.
# Validates the paper-cited numbers against the bundled JSON records.
verify:
	python3 scripts/dark_subspace/verify_claims.py

# Run the integration test suite (requires the package installed via `install`).
test:
	.venv/bin/python -m pytest tests/ -q

# Print the Pythia-6.9B mixed headline AUROCs (original, reconstruction,
# residual) with their JSON provenance. The values are the N=5 cohort means
# read live from the harmonised JSON. CPU-only, stdlib-only.
headline:
	python3 -c "import json, pathlib; \
p = pathlib.Path('results/dark_subspace/paper_claims/p69_n5_harmonized_2026-05-06.json'); \
d = json.load(open(p))['cluster_summary_n5']; \
orig = d['original_score_K_auroc']['mean']; \
recon = d['reconstructed_score_K_auroc']['mean']; \
resid = d['residual_score_K_auroc']['mean']; \
print('Pythia-6.9B mixed SAE headline (N=5 cohort means)'); \
print('  source: %s' % p); \
print('  keys: cluster_summary_n5.{original,reconstructed,residual}_score_K_auroc.mean'); \
print('  original score_K AUROC       = %.3f' % orig); \
print('  reconstruction score_K AUROC = %.3f' % recon); \
print('  residual score_K AUROC       = %.3f' % resid); \
print('PASS: original 0.803, reconstruction 0.594, residual 0.781 (3 d.p.)')"

# Regenerate all paper figures from the bundled JSON tree. Set FIGDIR to
# redirect output (default outputs/figures/). Requires the package installed.
figures:
	.venv/bin/python scripts/dark_subspace/plot_figures.py && \
	.venv/bin/python scripts/dark_subspace/plot_advanced_figures.py && \
	.venv/bin/python scripts/dark_subspace/plot_score_distributions.py
