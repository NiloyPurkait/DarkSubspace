from __future__ import annotations

"""Adapters and lightweight protocols for working with multiple SAE implementations.

This repo has its own minimal SAE implementation (see :mod:`sae_mia_audit.sae.sae`),
but we also support loading SAEs produced by the external
`ai-safety-foundation/sparse_autoencoder` library.

Why this exists:
- For publishable audits, you often want to leverage community-maintained SAE tooling
  (unit-norm decoders, tied biases, dead-feature mitigation, etc.).
- Downstream (PDD) methods in this repo only require a small interface: `encode` and `decode`.

The adapters in this module provide that interface without forcing downstream code
to depend on any particular SAE backend.

Safety/ethics:
This code is intended for authorized auditing of open-weight models and datasets.
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class SAEProtocol(Protocol):
    """Minimal interface expected by PDD methods and interventions."""

    def encode(self, x: torch.Tensor) -> torch.Tensor: ...
    def decode(self, z: torch.Tensor) -> torch.Tensor: ...

    def parameters(self): ...  # Required for device/dtype inference

    @property
    def d_model(self) -> int: ...

    @property
    def d_sae(self) -> int: ...


@dataclass(frozen=True)
class SAEInfo:
    backend: str  # e.g. 'internal', 'saif'
    d_model: int
    d_sae: int
    layer: int | None = None
    l1_coeff: float | None = None


class SAIFSparseAutoencoderAdapter:
    """Adapter for `sparse_autoencoder.autoencoder.model.SparseAutoencoder`.

    The upstream library exposes a forward pass returning `(learned_activations, decoded_activations)`.
    This adapter provides `encode` / `decode` methods consistent with this repo's expectations.

    Notes:
      - The upstream model applies a tied bias before encoding and after decoding.
      - The decoder is typically constrained to unit-norm weights (via a post-step hook).
    """

    def __init__(self, saif_model: Any):
        self.saif = saif_model
        # Best-effort extract dims
        try:
            self._d_model = int(saif_model.config.n_input_features)
            self._d_sae = int(saif_model.config.n_learned_features)
        except Exception:
            # fallback: infer from state dict
            sd = saif_model.state_dict()
            ew = sd.get("encoder.weight", None)
            if ew is None:
                raise ValueError("Could not infer d_model/d_sae from SAIF model")
            self._d_sae, self._d_model = int(ew.shape[0]), int(ew.shape[1])

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def d_sae(self) -> int:
        return self._d_sae

    def to(self, device: str | torch.device):
        self.saif.to(device)
        return self

    def eval(self):
        self.saif.eval()
        return self

    def parameters(self):
        """Return an iterator over SAE parameters (required for device/dtype inference)."""
        return self.saif.parameters()

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x2 = self.saif.pre_encoder_bias(x)
        z = self.saif.encoder(x2)
        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.saif.decoder(z)
        x = self.saif.post_decoder_bias(x)
        return x

    def state_dict(self):
        return self.saif.state_dict()
