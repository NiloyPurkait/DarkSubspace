# Tests

Three CPU-only integration tests that exercise contracts on the public artefact.

| Test | What it checks |
| --- | --- |
| `test_verifier_exits_zero` | `scripts/dark_subspace/verify_claims.py` runs to completion and reports `Asserted check summary: N/N pass, 0 fail`. |
| `test_paper_claim_jsons_parse` | Every `.json` under `results/dark_subspace/` parses as valid JSON. |
| `test_no_internal_paths_leak` | No shipped JSON contains a cluster-style home directory (`/home/u<digits>`), the legacy `runs/memcirc/` token, or the unresolved `<runs>/sae-mia-audit` placeholder. |

These tests do not exercise the GPU pipeline scripts. End-to-end re-execution requires GPU resources and is out of scope for the CI workflow at `.github/workflows/verify.yml`.

Run locally.

```bash
python3 -m pip install pytest
python3 -m pytest tests/ -q
```

CI runs the same commands on every push to `main` and on pull requests via `.github/workflows/verify.yml`.
