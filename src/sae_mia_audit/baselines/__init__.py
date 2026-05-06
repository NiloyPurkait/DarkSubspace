"""
Baseline methods for comparison with SAE-based interventions.

Phase 6: These baselines answer the reviewer question "why SAEs specifically?"
by comparing SAE-based feature selection to simpler alternatives.
"""

from .neuron_probe import (
    NeuronProbeBaseline,
    fit_neuron_probe,
    select_neurons_by_probe,
    ablate_neurons_in_hidden,
)
from .pca_baseline import (
    PCABaseline,
    ablate_pca_features,
)
from .random_rotation import (
    RandomRotationBaseline,
    generate_rotation_matrix,
    select_rotated_features,
    ablate_rotated_features,
)

__all__ = [
    "NeuronProbeBaseline",
    "fit_neuron_probe",
    "select_neurons_by_probe",
    "ablate_neurons_in_hidden",
    "PCABaseline",
    "ablate_pca_features",
    "RandomRotationBaseline",
    "generate_rotation_matrix",
    "select_rotated_features",
    "ablate_rotated_features",
]
