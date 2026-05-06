# The Dark Subspace of Fine-Tuning Memorisation

Cross-architecture residual concentration with Pythia-6.9B and 12B causal reference points.

[Anonymous double-blind submission to the ICML 2026 Mechanistic Interpretability Workshop. Reviewer-facing artifact.]

This repository accompanies the manuscript. It contains the paper sources, the code that produced the dark-subspace decomposition, the controls and diagnostics, and a CPU-only verifier that confirms paper-cited values match the on-disk JSON artifacts.

The full reviewer guide is in [`paper5/ARTIFACT_README.md`](paper5/ARTIFACT_README.md). It maps each numbered paper claim to its source script, source result JSON, and a one-line CPU reproduction command.

## Quickstart for reviewers

```bash
python3 -m venv env
env/bin/python3 -m pip install --upgrade pip
env/bin/python3 -m pip install -e .

# CPU-only verifier. Reads the JSONs cited by the paper and asserts they match
# paper-cited values. Exits with code 1 on any mismatch. Runs in seconds.
env/bin/python3 scripts/memcirc/verify_paper5.py
```

## Repository layout

| Path | Contents |
|------|----------|
| `paper5/` | Manuscript sources (LaTeX), pre-registration, ICML 2026 style files, reviewer guide. |
| `scripts/memcirc/` | Paper-specific code (BCD, SAE training driver wrappers, dark-subspace evaluation, controls, plotting, verifier). |
| `scripts/shared/` | Shared infrastructure (canonical SAE trainer, repository bootstrap, figure style, SAE feature evaluator). |
| `configs/` | Per-experiment configuration JSONs consumed by the SHIP-set scripts. |
| `data/` | Not shipped (prerequisite). The verifier path does not need data. The full reproduction path expects `data/memcirc_ctrl_ft/` and `data/memcirc_ctrl_disjoint/`. |
| `runs/` | Not shipped (generated). Per-experiment SAE checkpoints and result JSONs accumulate under `runs/memcirc/` and `runs/sae/`. |

## Where the reviewer guide lives

The reviewer entry point is [`paper5/ARTIFACT_README.md`](paper5/ARTIFACT_README.md). It contains the paper-to-code map for every numbered claim, the per-experiment runtime envelope, the CPU-only verification path, and the caveats section.

## Reproduce a single experiment

The CPU-only verifier is the intended reviewer path. For a minimal slice of the GPU pipeline, the per-row paired bootstrap on the cohort residual-vs-original margin reads pre-computed scores and writes the per-row JSON.

```bash
env/bin/python3 scripts/memcirc/per_row_bootstrap_kocl2.py
```

The full pipeline (controlled fine-tuning, BCD, SAE training, dark-subspace evaluation, controls) is large and was developed across several months on GPU clusters. Reviewers should not expect to reproduce it in a workshop review window.

## Pre-registration

Pre-registration items PR-1 through PR-8 are in [`paper5/PREREGISTRATION.md`](paper5/PREREGISTRATION.md). The submission tag will be GPG-signed at the lock commit before final upload.

## License

MIT. See [`LICENSE`](LICENSE).

## Anonymous submission notes

This is the reviewer-facing artifact for an anonymous double-blind submission. The manuscript is in `paper5/main.tex` and its included sources. Large generated artifacts (`data/`, `runs/`, `env/`, model checkpoints) are not tracked. The CPU-only verifier needs only the cached `results.json` files released alongside the artifact. There is no maintainer for the review window.
