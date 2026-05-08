# `sae_mia_audit.utils`

Small infrastructure modules used by the paper scripts.

| Module | Purpose | Used by |
| --- | --- | --- |
| `hf.py` | `HFModelSpec` dataclass for HuggingFace model identifier parsing. | All paper scripts that load a model |
| `logging.py` | `setup_logging()` and `get_logger()` for `rich`-formatted structured logging. | All paper scripts |
| `seed.py` | `SeedConfig` and `set_global_seed()` for Python/NumPy/torch RNG seeding. See the module docstring for the determinism policy and the `README.md` "Reproducibility caveat" for CUDA non-determinism. | All paper scripts |
| `run_dir.py` | Timestamped output directories and reproducibility-snapshot helpers (`make_run_dir`, `snapshot_reproducibility`, `dataclass_to_dict`). | `scripts/shared/train_sae.py` |
