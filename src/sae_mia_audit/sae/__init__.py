"""Sparse autoencoder implementation, training, I/O, and intervention helpers."""
from .sae import SparseAutoencoder, SAEConfig
from .trainer import SAETrainConfig, SAETrainer, MultiSAETrainer
from .interventions import (
    ablate_features_in_hidden,
    ablate_features_token_level,
    select_sae_features,
    select_sae_features_token_level,
    select_sae_features_noncircular,
    run_all_sanity_checks,
    check_identity_patch,
    check_reconstruction_patch,
    check_inactive_feature_edit,
    compute_kl_divergence,
    compute_intervention_validity,
    SanityCheckResult,
    InterventionResult,
    InterventionValidityMetrics,
    FeatureSelectionDiagnostics,
    AblationMode,
    FeatureSelectMode,
)
from .consistency import match_features_by_cosine
from .io import load_sae_checkpoint, load_sae_cfg, load_sae_checkpoint_any
from .adapters import SAEProtocol, SAIFSparseAutoencoderAdapter, SAEInfo

__all__ = [
    # Core SAE
    "SAEConfig",
    "SparseAutoencoder",
    "SAETrainConfig",
    "SAETrainer",
    "MultiSAETrainer",
    # Interventions (Group C)
    "ablate_features_in_hidden",
    "ablate_features_token_level",
    "select_sae_features",
    "select_sae_features_token_level",
    "select_sae_features_noncircular",
    "run_all_sanity_checks",
    "check_identity_patch",
    "check_reconstruction_patch",
    "check_inactive_feature_edit",
    "compute_kl_divergence",
    "compute_intervention_validity",
    "SanityCheckResult",
    "InterventionResult",
    "InterventionValidityMetrics",
    "FeatureSelectionDiagnostics",
    "AblationMode",
    "FeatureSelectMode",
    # Consistency
    "match_features_by_cosine",
    # I/O
    "load_sae_checkpoint",
    "load_sae_cfg",
    "load_sae_checkpoint_any",
    # Adapters
    "SAEProtocol",
    "SAIFSparseAutoencoderAdapter",
    "SAEInfo",
]
