# `sae_mia_audit`

Reusable Python package used by the Dark Subspace artefact scripts.

Starting points live in the repository root `README.md` and in `scripts/dark_subspace/`. This package contains shared implementation components used by the experiment scripts:

| Subpackage | Purpose |
| --- | --- |
| `data/` | Dataset loading, splits, tokenisation, and SAE training corpora. |
| `models/` | Hugging Face causal-LM wrappers and activation capture. |
| `sae/` | Sparse autoencoder modules, checkpoint I/O, interventions, and metrics. |
| `methods/` | MIA baselines, probes, PDD variants, and SAE-based scoring methods. |
| `eval/` | AUROC, calibration, bootstrap intervals, domain-shift checks, and groupwise metrics. |
| `utils/` | Logging, run directories, Hugging Face helpers, and seeding. |

Install from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

The CPU-only claim verifier does not require installing this package.
