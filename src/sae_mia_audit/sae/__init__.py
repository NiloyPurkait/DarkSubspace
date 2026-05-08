"""Sparse-autoencoder implementation, training, and I/O used by the paper.

Re-exports only the names the paper scripts and the ``sae-mia-audit train-sae``
console-script entry point depend on. Helpers for interventions, consistency
matching, top-k feature collection, and post-hoc interpretation that lived in
this package have been removed because no paper script imports them.
"""
from .sae import SparseAutoencoder, SAEConfig
from .trainer import SAETrainConfig, SAETrainer, MultiSAETrainer
from .io import load_sae_checkpoint, load_sae_cfg, load_sae_checkpoint_any
from .adapters import SAEProtocol, SAIFSparseAutoencoderAdapter, SAEInfo

__all__ = [
    "SAEConfig",
    "SparseAutoencoder",
    "SAETrainConfig",
    "SAETrainer",
    "MultiSAETrainer",
    "load_sae_checkpoint",
    "load_sae_cfg",
    "load_sae_checkpoint_any",
    "SAEProtocol",
    "SAIFSparseAutoencoderAdapter",
    "SAEInfo",
]
