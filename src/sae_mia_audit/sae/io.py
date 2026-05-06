from __future__ import annotations

"""I/O utilities for sparse autoencoders.

This repo primarily uses a minimal SAE implementation (:class:`~sae_mia_audit.sae.sae.SparseAutoencoder`)
and a simple checkpoint format:

Checkpoint format (torch.save dict):
  - 'sae_cfg': SAEConfig as a dict
  - 'state_dict': SAE weights
  - (optional) 'train_cfg', 'step', 'opt_state'

For research workflows, you may also want to train SAEs using external libraries
(e.g. `ai-safety-foundation/sparse_autoencoder`). Those libraries typically save a
serialized model/state object rather than this repo's dict format.

To support both workflows, this module provides:
  - :func:`load_sae_checkpoint` (legacy: repo-native dict checkpoints only)
  - :func:`load_sae_checkpoint_any` (recommended: supports both formats)

The returned object is guaranteed to implement the `encode`/`decode` interface
used throughout this repo (via adapters when needed).
"""

from pathlib import Path
from typing import Any, Dict, Union

import torch

from .adapters import SAIFSparseAutoencoderAdapter, SAEProtocol
from .sae import SAEConfig, SparseAutoencoder


def load_sae_checkpoint(path: Union[str, Path], device: str = "cpu") -> SparseAutoencoder:
    """Load a trained SAE from a *repo-native* checkpoint (legacy).

    Use :func:`load_sae_checkpoint_any` if you might load SAEs trained with
    `sparse_autoencoder` (ai-safety-foundation).

    Args:
        path: Path to a checkpoint produced by this repo (e.g., `sae_final.pt`).
        device: torch device string for map_location.

    Returns:
        A `SparseAutoencoder` in eval() mode.
    """
    p = Path(path)
    ckpt: Dict[str, Any] = torch.load(p, map_location=device)
    cfg_dict = ckpt.get("sae_cfg", None)
    if cfg_dict is None:
        raise ValueError(f"Checkpoint missing 'sae_cfg' (not a repo-native SAE): {p}")
    cfg = SAEConfig(**cfg_dict)
    sae = SparseAutoencoder(cfg)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return sae


def load_sae_checkpoint_any(path: Union[str, Path], device: str = "cpu") -> SAEProtocol:
    """Load an SAE checkpoint from either this repo or `sparse_autoencoder` (SAIF).

    Supported formats:
      1) Repo-native dict checkpoints (see module docstring)
      2) ai-safety-foundation/sparse_autoencoder `SparseAutoencoder.save(...)` artifacts

    Returns:
        An object with `encode(x)->z` and `decode(z)->x_hat` methods.
    """
    p = Path(path)

    # Fast path: try repo-native dict checkpoint
    try:
        obj = torch.load(p, map_location=device)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"Failed to load SAE checkpoint: {p}") from e

    if isinstance(obj, dict) and "sae_cfg" in obj and "state_dict" in obj:
        cfg = SAEConfig(**obj["sae_cfg"])
        sae = SparseAutoencoder(cfg)
        sae.load_state_dict(obj["state_dict"])
        sae.to(device)
        sae.eval()
        return sae

    # Otherwise try loading as a sparse_autoencoder artifact.
    try:
        from sparse_autoencoder.autoencoder.model import SparseAutoencoder as SAIFSAE  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ValueError(
            f"Unknown SAE checkpoint format for {p}. " 
            "If this is a `sparse_autoencoder` artifact, install it via `pip install sparse_autoencoder`."
        ) from e

    # SAIF loader always loads to CPU, then we move to device.
    saif_model = SAIFSAE.load(str(p))
    saif_model.eval()
    try:
        saif_model.to(device)
    except Exception:
        # Some environments pass non-standard devices; safest is to rely on caller.
        pass
    return SAIFSparseAutoencoderAdapter(saif_model).to(device).eval()


def load_sae_cfg(path: Union[str, Path]) -> SAEConfig:
    """Read only the SAEConfig from a repo-native checkpoint."""
    p = Path(path)
    ckpt: Dict[str, Any] = torch.load(p, map_location="cpu")
    cfg_dict = ckpt.get("sae_cfg", None)
    if cfg_dict is None:
        raise ValueError(f"Checkpoint missing 'sae_cfg': {p}")
    return SAEConfig(**cfg_dict)
