# Contributing

This repository is the published artifact for "The Dark Subspace of Fine-Tuning
Memorisation" (ICML 2026 Workshop on Mechanistic Interpretability, Spotlight). It
is a frozen research artifact, not an actively developed project, so we are not
soliciting feature pull requests. For questions, corrections, or reproduction
issues, please open an issue on the GitHub issue tracker.

The first sanity check is the CPU-only verifier, which needs no dependencies and
runs in a few seconds.

```bash
make verify   # or: python3 scripts/dark_subspace/verify_claims.py
```

For development of the experiment code, install the dev extras and run the
configured checks. Ruff and mypy settings live in `pyproject.toml`.

```bash
pip install -e ".[dev]"
pre-commit run --all-files
pytest tests/
```
