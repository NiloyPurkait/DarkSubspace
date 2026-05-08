# `sae_mia_audit.utils`

Three small infrastructure modules used by the paper scripts.

| Module | Purpose | Used by |
| --- | --- | --- |
| `hf.py` | `HFModelSpec` dataclass for HuggingFace model identifier parsing. | All paper scripts that load a model |
| `logging.py` | `setup_logging()` and `get_logger()` for `rich`-formatted structured logging. | All paper scripts |
| `seed.py` | `SeedConfig` and `set_global_seed()` for Python/NumPy/torch RNG seeding. See the module docstring for the determinism policy and the `README.md` "Reproducibility caveat" for CUDA non-determinism. | All paper scripts |

Earlier iterations of this package included a run-directory helper (`run_dir.py`) that was not imported by any paper script and has been removed from the public artefact.
