# `sae_mia_audit.methods`

A single module shipped in the public artefact.

| Module | Purpose | Used by |
| --- | --- | --- |
| `baselines.py` | `score_loss_attack()` and `score_zlib_loss_ratio()` for the standard pre-training-MIA baseline suite. | `scripts/dark_subspace/length_baseline.py`, `baseline_attacks_suite.py` |

Earlier iterations of this package included Min-K%, infilling, NA-PDD, DC-PDD, SAE-NA-PDD, probe-based, and SAE-audit MIA methods plus an aggregation helper. None of those modules were imported by any paper script and they have been removed from the public artefact.
