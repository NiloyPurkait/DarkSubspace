from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Type alias for L1 form options
L1Form = Literal["mean", "sum"]


@dataclass(frozen=True)
class SAEConfig:
    d_model: int
    d_sae: int

    # Regularization
    l1_coeff: float = 1e-3

    # L1 form: "mean" = z.abs().mean() [legacy], "sum" = z.abs().sum(dim=-1).mean() [standard]
    # "sum" computes L1 norm per token (sum over features) then averages over batch.
    # "mean" computes mean activation magnitude over all (batch, feature) pairs.
    # Most SAE papers use "sum" form; "mean" can lead to many small activations.
    l1_form: L1Form = "sum"

    # S6.13 (2026-04-21): optional L2 penalty on sparse codes (elastic-net form).
    # Loss += l2_coeff * (z ** 2).sum(dim=-1).mean()
    # Default 0.0 = backward compatible (pure L1). Matches Mahdizadehaghdam 2018 eq (2)
    # and Shen-Liu-Wang 2014 elastic-net sparse-coding formulation. Intended use:
    # discourage the all-mass-on-few-features collapse pathology seen on Mistral.
    l2_coeff: float = 0.0

    # Bias handling
    # Back-compat: use_bias controls both unless per-module overrides are provided.
    use_bias: bool = True
    use_encoder_bias: Optional[bool] = None
    use_decoder_bias: Optional[bool] = None

    # Weight tying: if True, decoder.weight = encoder.weight.T (transposed)
    # Standard in some SAE papers (Cunningham et al., 2023); reduces parameters.
    tied_weights: bool = False

    # Training-time constraints
    # Constrain decoder feature vectors (columns) to unit norm to prevent L1 gaming via scaling.
    normalize_decoder: bool = True

    # Best-practice: encode around decoder bias (SAE-lens style).
    # z = ReLU(W_enc (x - b_dec) + b_enc), x_hat = W_dec z + b_dec
    center_input: bool = True
    
    # =========================================================================
    # D5: Load balancing regularization (optional, for high-overcompleteness)
    # =========================================================================
    # Penalizes deviation from uniform feature usage to prevent collapse.
    # Computed as: aux_coeff * (firing_rate - target_rate)^2.mean()
    # Set aux_coeff > 0 to enable (e.g., 0.01-0.1).
    aux_coeff: float = 0.0  # 0 = disabled; try 0.01-0.1 for large dicts
    aux_target_firing_rate: float = 0.01  # Target: each feature fires on ~1% of tokens


class SparseAutoencoder(nn.Module):
    """ReLU sparse autoencoder for residual-stream activations.

    Paper-aligned details:
      - decoder column normalization (feature vectors)
      - optional encoder centering by decoder bias (b_dec)
      - configurable encoder/decoder bias usage
      - optional tied weights (decoder = encoder.T)
      - configurable L1 form (mean vs sum)
    """

    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg

        enc_bias = cfg.use_encoder_bias if cfg.use_encoder_bias is not None else cfg.use_bias
        dec_bias = cfg.use_decoder_bias if cfg.use_decoder_bias is not None else cfg.use_bias

        self.encoder = nn.Linear(cfg.d_model, cfg.d_sae, bias=bool(enc_bias))
        
        if cfg.tied_weights:
            # Tied weights: decoder shares encoder weights (transposed)
            # We still need a separate bias if requested
            self._decoder_bias = nn.Parameter(torch.zeros(cfg.d_model)) if dec_bias else None
            self._tied_weights = True
        else:
            self.decoder = nn.Linear(cfg.d_sae, cfg.d_model, bias=bool(dec_bias))
            self._tied_weights = False

        # Common init: small weights; zero biases.
        nn.init.normal_(self.encoder.weight, std=0.02)
        if self.encoder.bias is not None:
            nn.init.zeros_(self.encoder.bias)
        
        if not cfg.tied_weights:
            nn.init.normal_(self.decoder.weight, std=0.02)
            if self.decoder.bias is not None:
                nn.init.zeros_(self.decoder.bias)

        if cfg.normalize_decoder:
            self._renorm_decoder_()

    @property
    def d_model(self) -> int:
        return int(self.cfg.d_model)

    @property
    def d_sae(self) -> int:
        return int(self.cfg.d_sae)

    @property
    def decoder_weight(self) -> torch.Tensor:
        """Return decoder weight matrix [d_model, d_sae].
        
        For tied weights, this is encoder.weight.T.
        """
        if self._tied_weights:
            # encoder.weight: [d_sae, d_model], so transpose to get [d_model, d_sae]
            return self.encoder.weight.t()
        return self.decoder.weight

    @property
    def decoder_bias(self) -> Optional[torch.Tensor]:
        """Return decoder bias if present."""
        if self._tied_weights:
            return self._decoder_bias
        return self.decoder.bias if hasattr(self.decoder, 'bias') else None

    def _maybe_center_input(self, x: torch.Tensor) -> torch.Tensor:
        # Center encoder input around decoder bias (b_dec) if available.
        b_dec = self.decoder_bias
        if self.cfg.center_input and (b_dec is not None):
            # decoder bias shape: [d_model], broadcasts over batch
            return x - b_dec
        return x

    @torch.no_grad()
    def _renorm_decoder_(self) -> None:
        """Project decoder columns (feature vectors) to unit norm.

        For untied weights: nn.Linear(d_sae -> d_model) stores weight with shape [d_model, d_sae].
        Each SAE feature corresponds to one column (size d_model).
        
        For tied weights: encoder.weight has shape [d_sae, d_model], so we normalize rows.
        """
        if self._tied_weights:
            # Normalize encoder rows (which become decoder columns when transposed)
            w = self.encoder.weight  # [d_sae, d_model]
            row_norms = torch.linalg.vector_norm(w, dim=1, keepdim=True).clamp_min(1e-8)
            w.div_(row_norms)
        else:
            w = self.decoder.weight  # [d_model, d_sae]
            col_norms = torch.linalg.vector_norm(w, dim=0, keepdim=True).clamp_min(1e-8)
            w.div_(col_norms)

    def maybe_renorm_decoder(self) -> None:
        if self.cfg.normalize_decoder:
            self._renorm_decoder_()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self._maybe_center_input(x)
        z_pre = self.encoder(x_in)
        return F.relu(z_pre)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent z to reconstruction."""
        # For tied weights, manually compute W_dec @ z + b_dec
        # decoder_weight is [d_model, d_sae], so we use F.linear which does z @ W.T
        if self._tied_weights:
            # z: [..., d_sae], decoder_weight: [d_model, d_sae]
            # F.linear(z, W) computes z @ W.T, so we get [..., d_model]
            out = F.linear(z, self.decoder_weight)
            if self._decoder_bias is not None:
                out = out + self._decoder_bias
            return out
        return self.decoder(z)

    def _compute_l1(self, z: torch.Tensor) -> torch.Tensor:
        """Compute L1 sparsity term based on configured form.
        
        "mean": z.abs().mean() - mean over all (batch, feature) pairs
        "sum":  z.abs().sum(dim=-1).mean() - L1 norm per token, then mean over batch
        
        The "sum" form is standard in SAE literature and encourages truly sparse codes.
        The "mean" form can allow many small activations while keeping the penalty low.
        """
        if self.cfg.l1_form == "sum":
            # L1 norm per token (sum over features), then average over batch
            return z.abs().sum(dim=-1).mean()
        else:  # "mean" (legacy)
            return z.abs().mean()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (x_hat, z, l1_term)."""
        z = self.encode(x)
        x_hat = self.decode(z)
        l1 = self._compute_l1(z)
        return x_hat, z, l1

    def loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        x_hat, z, l1 = self.forward(x)

        recon = F.mse_loss(x_hat, x)
        loss = recon + (float(self.cfg.l1_coeff) * l1)

        # S6.13 (2026-04-21): optional L2 penalty on sparse codes for elastic-net.
        # Computed as sum over features per token, then mean over batch
        # (matches "sum" form of L1 for dimensional consistency).
        l2_term = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if float(self.cfg.l2_coeff) > 0.0:
            l2_term = (z ** 2).sum(dim=-1).mean()
            loss = loss + (float(self.cfg.l2_coeff) * l2_term)

        # D5: Load balancing auxiliary loss (if enabled)
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.cfg.aux_coeff > 0:
            # Compute per-feature firing rate on this batch
            # z: [batch, d_sae], firing if z > 0
            firing_rate = (z > 0).float().mean(dim=0)  # [d_sae]
            # Penalize deviation from target firing rate
            aux_loss = ((firing_rate - self.cfg.aux_target_firing_rate) ** 2).mean()
            loss = loss + (self.cfg.aux_coeff * aux_loss)

        # Metrics (no gradient needed)
        with torch.no_grad():
            # FVU uses batch variance; stable, scale-free, common SAE metric
            x_centered = x - x.mean(dim=0, keepdim=True)
            ss_tot = torch.mean(x_centered ** 2).clamp_min(1e-12)
            fvu = recon.detach() / ss_tot

            l0 = (z > 0).float().sum(dim=-1).mean()
            active_frac = l0 / max(1, z.shape[-1])
            
            # Also compute both L1 forms for comparison logging
            l1_mean = z.abs().mean()
            l1_sum = z.abs().sum(dim=-1).mean()

        metrics = {
            "loss": loss.detach(),
            "recon_mse": recon.detach(),
            "fvu": fvu.detach(),
            "l1_term": l1.detach(),  # The actual L1 term used in loss
            "l1_mean": l1_mean.detach(),  # Legacy metric (mean form)
            "l1_sum": l1_sum.detach(),  # Standard metric (sum form)
            "l0_mean": l0.detach(),
            "active_frac": active_frac.detach(),
            "aux_loss": aux_loss.detach(),  # D5: Load balancing loss
            "l2_term": l2_term.detach(),  # S6.13: elastic-net L2 on codes (sum form)
            "l2_coeff": float(self.cfg.l2_coeff),  # S6.13: echo for logs
        }
        return loss, metrics
