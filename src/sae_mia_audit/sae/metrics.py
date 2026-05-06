from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SAEMetrics:
    recon_mse: float
    fvu: float
    l0_mean: float
    l1_mean: float  # Legacy: z.abs().mean()
    l1_sum: float   # Standard: z.abs().sum(dim=-1).mean()
    active_frac: float


@torch.no_grad()
def compute_sae_metrics(x: torch.Tensor, x_hat: torch.Tensor, z: torch.Tensor) -> SAEMetrics:
    recon_mse = torch.mean((x_hat - x) ** 2).item()
    x_centered = x - x.mean(dim=0, keepdim=True)
    ss_tot = torch.mean(x_centered ** 2).item()
    fvu = float(recon_mse / max(ss_tot, 1e-12))
    l0_mean = (z > 0).float().sum(dim=-1).mean().item()
    l1_mean = z.abs().mean().item()
    l1_sum = z.abs().sum(dim=-1).mean().item()
    active_frac = float(l0_mean / max(1.0, z.shape[-1]))
    return SAEMetrics(
        recon_mse=recon_mse, 
        fvu=fvu, 
        l0_mean=l0_mean, 
        l1_mean=l1_mean, 
        l1_sum=l1_sum,
        active_frac=active_frac
    )
