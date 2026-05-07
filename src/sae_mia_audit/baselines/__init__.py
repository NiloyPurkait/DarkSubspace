"""
Baseline methods for comparison with SAE-based interventions.

These baselines isolate the contribution of the SAE basis by comparing
SAE feature selection against simpler alternatives (raw neurons, PCA,
random rotations of the residual stream).
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
