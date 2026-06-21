"""Minimal sparse autoencoder module used throughout sae_mia_audit.

Provides :class:`SparseAutoencoder` and :class:`SAEConfig` along with the
forward, encode, decode, and loss computations used by the trainer.
"""
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

    # Optional L2 penalty on sparse codes (elastic-net form).
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
    # Load balancing regularisation (optional, for high-overcompleteness)
    # =========================================================================
    # Penalizes deviation from uniform feature usage to prevent collapse.
    # Computed as: aux_coeff * (firing_rate - target_rate)^2.mean()
    # Set aux_coeff > 0 to enable (e.g., 0.01-0.1).
    aux_coeff: float = 0.0  # 0 = disabled; try 0.01-0.1 for large dicts
    aux_target_firing_rate: float = 0.01  # Target: each feature fires on ~1% of tokens

    # =========================================================================
    # TopK activation (Gao et al. 2024 k-Sparse Autoencoders)
    # =========================================================================
    # When set to a positive integer K, the encoder applies a TopK activation:
    # only the K largest pre-ReLU activations per token are kept (others zeroed).
    # This replaces L1 sparsity with a hard sparsity constraint. When topk is
    # set, l1_coeff is effectively unused (loss = recon only). Default None
    # preserves legacy ReLU+L1 behaviour.
    topk: Optional[int] = None

    # =========================================================================
    # JumpReLU activation (Rajamanoharan et al. 2024 / Gemma Scope recipe)
    # =========================================================================
    # When jumprelu=True the encoder applies, per feature i, a learned-threshold
    # gate on the ReLU pre-activation:
    #     a   = ReLU(W_enc (x - b_dec) + b_enc)        # standard ReLU code
    #     z_i = a_i * H(a_i - theta_i)                 # JumpReLU: gate by step
    # where theta_i >= 0 is a per-feature learned threshold and H is the
    # Heaviside step. H is non-differentiable in theta, so we use the Gemma Scope
    # straight-through estimator (STE): a centered-rectangle pseudo-derivative of
    # bandwidth eps for the gradient of H w.r.t. its argument (see _JumpReLUSTE
    # and _StepSTE below). The threshold is stored as a log parameter
    # (log_theta) so theta = exp(log_theta) is always strictly positive; this
    # also matches the Gemma Scope practice of optimising in log space.
    #
    # Sparsity: JumpReLU replaces the L1 penalty with an L0 penalty,
    #     L0 = sum_i H(a_i - theta_i)   (expected # active features per token),
    # whose gradient w.r.t. theta flows through the SAME rectangle STE. We REUSE
    # the existing `l1_coeff` field as the L0 coefficient (minimal change, mirrors
    # how topk reused the existing plumbing); no new coefficient field is added.
    # `l1_form` is ignored under JumpReLU (the L0 term is always a per-token sum
    # averaged over the batch). Default False preserves legacy ReLU+L1 behaviour.
    # jumprelu and topk are mutually exclusive (topk takes precedence if both set;
    # the trainer should not set both).
    jumprelu: bool = False
    # Initial per-feature threshold in raw activation units (Gemma Scope ~1e-3).
    jumprelu_theta_init: float = 1e-3
    # STE rectangle bandwidth eps (Gemma Scope kernel width; ~1e-3). Controls how
    # far below/above the threshold a sample contributes gradient to theta.
    jumprelu_bandwidth: float = 1e-3


# =============================================================================
# JumpReLU straight-through estimators (Rajamanoharan et al. 2024, Appendix B,
# "Gemma Scope" recipe). Both functions are exact in the forward pass (true
# Heaviside step) and use a centered-rectangle pseudo-derivative of width `eps`
# in the backward pass so that the per-feature thresholds learn.
#
# For a gate g = H(a - theta):
#   - The forward value is the true step function (0/1).
#   - d g / d theta  is approximated by  -(1/eps) * K((a - theta)/eps),
#     where K is the centered rectangle  K(u) = 1 if |u| <= 1/2 else 0.
#   - d g / d a      is approximated by  +(1/eps) * K((a - theta)/eps)
#     (equal magnitude, opposite sign), which lets reconstruction gradient also
#     nudge the threshold consistently. Only samples within the band
#     |a - theta| <= eps/2 contribute gradient.
# =============================================================================


class _StepSTE(torch.autograd.Function):
    """Heaviside step H(a - theta) with rectangle STE for the L0 penalty.

    Forward returns 1.0 where a > theta else 0.0 (per element). Backward routes
    the incoming gradient through the rectangle kernel to both `a` and `theta`.
    Used by the L0 sparsity term (count of active features).
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, theta: torch.Tensor, eps: float):
        ctx.save_for_backward(a, theta)
        ctx.eps = float(eps)
        return (a > theta).to(a.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        a, theta = ctx.saved_tensors
        eps = ctx.eps
        # Centered rectangle kernel: 1 inside the band |a - theta| <= eps/2.
        in_band = ((a - theta).abs() <= (0.5 * eps)).to(a.dtype)
        kernel = in_band / eps
        grad_a = grad_out * kernel  # dH/da  ~ +(1/eps) K
        grad_theta = grad_out * (-kernel)  # dH/dtheta ~ -(1/eps) K
        return grad_a, grad_theta, None


class _JumpReLUSTE(torch.autograd.Function):
    """JumpReLU gate value z = a * H(a - theta) with rectangle STE on theta.

    Forward returns the true gated activation (a where a > theta, else 0).
    Backward:
      d z / d a     = H(a - theta)               [the gate itself; std path]
                      + a * (1/eps) K((a-theta)/eps)   [STE width term]
      d z / d theta = a * (-(1/eps) K((a-theta)/eps))  [Gemma Scope pseudo-deriv]
    Used for the reconstruction path so threshold gradient flows from the loss.
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, theta: torch.Tensor, eps: float):
        gate = (a > theta).to(a.dtype)
        ctx.save_for_backward(a, theta, gate)
        ctx.eps = float(eps)
        return a * gate

    @staticmethod
    def backward(ctx, grad_out):
        a, theta, gate = ctx.saved_tensors
        eps = ctx.eps
        in_band = ((a - theta).abs() <= (0.5 * eps)).to(a.dtype)
        kernel = in_band / eps
        # d z / d a: the gate (where active) plus the STE width contribution.
        grad_a = grad_out * (gate + a * kernel)
        # d z / d theta: Gemma Scope pseudo-derivative (negative band kernel * a).
        grad_theta = grad_out * (-(a * kernel))
        return grad_a, grad_theta, None


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

        # JumpReLU per-feature learned threshold, stored in log space so that
        # theta = exp(log_theta) > 0 always. Init so exp(log_theta) == theta_init.
        # Always registered (cheap, d_sae params) so save/load roundtrips even
        # when jumprelu is toggled; only USED when cfg.jumprelu is True.
        theta_init = max(float(cfg.jumprelu_theta_init), 1e-12)
        self.jumprelu_log_theta = nn.Parameter(
            torch.full((cfg.d_sae,), float(torch.log(torch.tensor(theta_init))))
        )

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

    @property
    def jumprelu_theta(self) -> torch.Tensor:
        """Per-feature positive threshold theta = exp(log_theta), shape [d_sae]."""
        return torch.exp(self.jumprelu_log_theta)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self._maybe_center_input(x)
        z_pre = self.encoder(x_in)
        # TopK activation (Gao et al. 2024): keep only K largest pre-ReLU
        # activations per token, then ReLU (clamps any negatives at 0).
        if self.cfg.topk is not None and int(self.cfg.topk) > 0:
            k = int(self.cfg.topk)
            k = min(k, z_pre.shape[-1])
            topk_vals, topk_idx = torch.topk(z_pre, k=k, dim=-1)
            mask = torch.zeros_like(z_pre)
            mask.scatter_(-1, topk_idx, 1.0)
            z = F.relu(z_pre * mask)
            return z
        # JumpReLU (Rajamanoharan et al. 2024): gate the ReLU code by a learned
        # per-feature threshold. z_i = a_i * H(a_i - theta_i), a = ReLU(z_pre).
        # Threshold gradient flows via the rectangle STE (_JumpReLUSTE).
        if self.cfg.jumprelu:
            a = F.relu(z_pre)
            theta = self.jumprelu_theta  # [d_sae], broadcasts over batch
            return _JumpReLUSTE.apply(a, theta, float(self.cfg.jumprelu_bandwidth))
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

    def _compute_jumprelu_l0(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable L0 sparsity term for JumpReLU.

        L0 = mean over batch of sum_i H(a_i - theta_i), where a = ReLU(encoder
        pre-activation). Gradient w.r.t. theta flows through the rectangle STE
        (_StepSTE), so increasing l1_coeff pushes thresholds up and drives the
        active-feature count down. Returns a scalar (expected active features
        per token).
        """
        x_in = self._maybe_center_input(x)
        a = F.relu(self.encoder(x_in))
        theta = self.jumprelu_theta
        gate = _StepSTE.apply(a, theta, float(self.cfg.jumprelu_bandwidth))
        return gate.sum(dim=-1).mean()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (x_hat, z, l1_term).

        For JumpReLU the third element is the differentiable L0 sparsity term
        (expected active features per token) rather than an L1 magnitude; it is
        still scaled by l1_coeff in `loss` (the reused coefficient field).
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        if self.cfg.jumprelu:
            sparsity = self._compute_jumprelu_l0(x)
        else:
            sparsity = self._compute_l1(z)
        return x_hat, z, sparsity

    def loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        x_hat, z, l1 = self.forward(x)

        recon = F.mse_loss(x_hat, x)
        loss = recon + (float(self.cfg.l1_coeff) * l1)

        # Optional L2 penalty on sparse codes for elastic-net.
        # Computed as sum over features per token, then mean over batch
        # (matches "sum" form of L1 for dimensional consistency).
        l2_term = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if float(self.cfg.l2_coeff) > 0.0:
            l2_term = (z ** 2).sum(dim=-1).mean()
            loss = loss + (float(self.cfg.l2_coeff) * l2_term)

        # Load balancing auxiliary loss (if enabled)
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
            "aux_loss": aux_loss.detach(),  # Load balancing loss
            "l2_term": l2_term.detach(),  # elastic-net L2 on codes (sum form)
            "l2_coeff": float(self.cfg.l2_coeff),  # echo for logs
        }
        # JumpReLU diagnostics: surface mean threshold and the L0 sparsity term
        # so the training logs show thresholds learning and L0 falling.
        if self.cfg.jumprelu:
            with torch.no_grad():
                metrics["jumprelu_theta_mean"] = self.jumprelu_theta.mean().detach()
                metrics["jumprelu_l0"] = l1.detach()  # the L0 sparsity term used
        return loss, metrics
