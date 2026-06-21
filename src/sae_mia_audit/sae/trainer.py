"""Trainers for sparse autoencoders.

Provides single-SAE and multi-SAE training loops with optional weight tying,
load-balancing auxiliary loss, and dead-feature resampling.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from .sae import SparseAutoencoder
from sae_mia_audit.utils.logging import get_logger


def _is_dist_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _get_rank() -> int:
    if _is_dist_initialized():
        return int(torch.distributed.get_rank())
    return int(os.environ.get("RANK", "0"))


def _get_world_size() -> int:
    if _is_dist_initialized():
        return int(torch.distributed.get_world_size())
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_main_process() -> bool:
    return _get_rank() == 0


def _get_local_rank() -> int:
    # accelerate sets LOCAL_RANK; SLURM uses SLURM_LOCALID
    for k in ("LOCAL_RANK", "SLURM_LOCALID"):
        if k in os.environ:
            try:
                return int(os.environ[k])
            except ValueError:
                pass
    return 0


def _resolve_device(device: str) -> torch.device:
    """Resolve a user device string under multi-process launchers.

    If device="cuda", we pick cuda:{local_rank} when possible. This prevents
    accidental cuda:0 use on every rank in multi-process runs.
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        if device == "cuda":
            lr = _get_local_rank()
            n = torch.cuda.device_count()
            if n > 0 and lr < n:
                return torch.device("cuda", lr)
            return torch.device("cuda", 0)
        return torch.device(device)
    return torch.device(device)


def _parse_dtype(name: str) -> torch.dtype:
    n = name.lower().strip()
    if n in ("float32", "fp32", "torch.float32"):
        return torch.float32
    if n in ("float16", "fp16", "torch.float16"):
        return torch.float16
    if n in ("bfloat16", "bf16", "torch.bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype string: {name!r}")


def _jsonl_append(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class SAETrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.0
    max_steps: int = 10_000
    grad_clip: float = 1.0
    log_every: int = 50
    save_every: int = 1000
    device: str = "cuda"

    # IMPORTANT: For SAEs, training in bf16 frequently makes the L1 term
    # numerically disappear (loss ~= recon), especially for small l1_coeff.
    # Default to float32 to ensure sparsity pressure is real.
    param_dtype: str = "float32"

    # If launched with accelerate/torchrun (WORLD_SIZE>1) but the trainer is not
    # explicitly sharding data across ranks, it's safer to train only on rank 0.
    # Set False only if your activation stream is rank-sharded and you understand
    # the implications.
    only_main_process: bool = True

    # Quality / reproducibility artifacts (safe defaults).
    write_metrics_jsonl: bool = True
    write_summary_json: bool = True
    
    # =========================================================================
    # Dead feature resampling (strongest fix for dictionary collapse)
    # =========================================================================
    # When enabled, periodically reinitializes dead features using the
    # reconstruction residual distribution. This prevents "dead forever" features.
    resample_dead_features: bool = False  # Set True to enable
    resample_every: int = 1000  # Steps between resampling checks
    resample_dead_threshold: float = 1e-6  # Firing rate below which feature is "dead"
    resample_window_steps: int = 100  # Steps over which to track firing rates


class SAETrainer:
    def __init__(self, sae: SparseAutoencoder, cfg: SAETrainConfig, out_dir: Path):
        self.sae = sae
        self.cfg = cfg
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log = get_logger(__name__)

        self.opt = AdamW(self.sae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.step = 0

        self._device = _resolve_device(cfg.device)
        self._dtype = _parse_dtype(cfg.param_dtype)
        self._rank = _get_rank()
        self._world = _get_world_size()
        self._main = _is_main_process()

        self._metrics_path = self.out_dir / "train_metrics.jsonl"
        self._summary_path = self.out_dir / "train_summary.json"
        self._last_logged_metrics: Optional[dict] = None
        
        # Dead feature tracking for resampling
        self._feature_firing_counts: Optional[torch.Tensor] = None  # [d_sae]
        self._feature_tracking_tokens: int = 0
        self._total_resampled: int = 0

        if self._main and self.cfg.write_metrics_jsonl:
            # Create/clear file for determinism.
            self._metrics_path.write_text("", encoding="utf-8")

    def load(self, path: Path) -> int:
        """Load a checkpoint and restore state.
        
        Returns: The step number from the checkpoint.
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        state = torch.load(path, map_location="cpu", weights_only=False)
        
        # Restore SAE weights
        self.sae.load_state_dict(state["state_dict"])
        
        # Restore optimizer state
        if "opt_state" in state:
            self.opt.load_state_dict(state["opt_state"])
        
        # Restore step counter
        self.step = state.get("step", 0)
        
        # Restore resampling stats
        self._total_resampled = state.get("total_resampled", 0)
        
        self._log_info(f"Loaded checkpoint from step {self.step}: {path}")
        return self.step

    @classmethod
    def find_latest_checkpoint(cls, out_dir: Path) -> Optional[Path]:
        """Find the most recent checkpoint in out_dir.
        
        Returns: Path to latest checkpoint, or None if not found.
        """
        if not out_dir.exists():
            return None
        
        checkpoints = list(out_dir.glob("sae_step*.pt"))
        if not checkpoints:
            return None
        
        # Sort by step number
        def get_step(p: Path) -> int:
            try:
                # sae_step1000.pt -> 1000
                name = p.stem  # sae_step1000
                return int(name.replace("sae_step", ""))
            except ValueError:
                return 0
        
        checkpoints.sort(key=get_step, reverse=True)
        return checkpoints[0]

    def _log_info(self, msg: str) -> None:
        if self._main:
            self.log.info(msg)

    def save(self, name: str) -> None:
        if self.cfg.only_main_process and not self._main:
            return
        path = self.out_dir / name
        tmp = path.with_suffix(path.suffix + ".tmp")

        state = {
            "step": self.step,
            "sae_cfg": self.sae.cfg.__dict__,
            "train_cfg": self.cfg.__dict__,
            "state_dict": self.sae.state_dict(),
            "opt_state": self.opt.state_dict(),
            "total_resampled": self._total_resampled,  # Track resampling stats
        }
        torch.save(state, tmp)
        tmp.replace(path)
        self._log_info(f"Saved checkpoint: {path}")
    
    # =========================================================================
    # Dead feature resampling
    # =========================================================================
    @torch.no_grad()
    def _update_feature_firing_stats(self, z: torch.Tensor) -> None:
        """Track feature firing counts for resampling decisions."""
        if not self.cfg.resample_dead_features:
            return
        
        # Initialize if needed
        if self._feature_firing_counts is None:
            self._feature_firing_counts = torch.zeros(
                self.sae.d_sae, device="cpu", dtype=torch.long
            )
        
        # Count firings: z > 0 means feature activated
        firings = (z > 0).sum(dim=0).cpu()  # [d_sae]
        self._feature_firing_counts += firings
        self._feature_tracking_tokens += z.shape[0]
    
    @torch.no_grad()
    def _maybe_resample_dead_features(self, x: torch.Tensor) -> int:
        """Resample dead features using reconstruction residual.
        
        This is standard practice for preventing dictionary collapse in
        overcomplete SAEs. Dead features are reinitialized from the
        reconstruction residual distribution.
        
        Returns: Number of features resampled.
        """
        if not self.cfg.resample_dead_features:
            return 0
        if self._feature_firing_counts is None:
            return 0
        if self._feature_tracking_tokens == 0:
            return 0
        
        # Compute firing rates over current window
        firing_rates = self._feature_firing_counts.float() / max(1, self._feature_tracking_tokens)
        dead_mask = firing_rates < self.cfg.resample_dead_threshold  # [d_sae]
        n_dead = int(dead_mask.sum().item())
        
        if n_dead == 0:
            # BUG-FIX: Always reset tracking for next window, even when no
            # features are dead. Previously the reset only ran when n_dead > 0,
            # causing counts to accumulate across all windows and masking
            # features that die later in training.
            self._feature_firing_counts.zero_()
            self._feature_tracking_tokens = 0
            return 0
        
        dead_indices = dead_mask.nonzero(as_tuple=True)[0].tolist()
        
        # Compute reconstruction residual
        x_hat, _, _ = self.sae(x)
        residual = x - x_hat  # [batch, d_model]
        
        # Sample residual vectors for reinitialization
        batch_size = residual.shape[0]
        if batch_size < n_dead:
            # Repeat residual to have enough samples
            reps = (n_dead + batch_size - 1) // batch_size
            residual = residual.repeat(reps, 1)
        
        # Shuffle and select
        perm = torch.randperm(residual.shape[0], device=residual.device)
        selected_residuals = residual[perm[:n_dead]]  # [n_dead, d_model]
        
        # Normalize to unit vectors
        norms = torch.linalg.vector_norm(selected_residuals, dim=1, keepdim=True).clamp_min(1e-8)
        selected_residuals = selected_residuals / norms
        
        # Reinitialize decoder columns and encoder rows for dead features
        if self.sae._tied_weights:
            # For tied weights, encoder rows become decoder columns
            # encoder.weight: [d_sae, d_model]
            for i, feat_idx in enumerate(dead_indices):
                self.sae.encoder.weight.data[feat_idx] = selected_residuals[i]
        else:
            # encoder.weight: [d_sae, d_model], decoder.weight: [d_model, d_sae]
            for i, feat_idx in enumerate(dead_indices):
                # Reset decoder column (feature direction)
                self.sae.decoder.weight.data[:, feat_idx] = selected_residuals[i]
                # Reset encoder row to transpose (maintains tied-like initialization)
                self.sae.encoder.weight.data[feat_idx] = selected_residuals[i]
        
        # Reset encoder bias for dead features
        if self.sae.encoder.bias is not None:
            for feat_idx in dead_indices:
                self.sae.encoder.bias.data[feat_idx] = 0.0
        
        # Reset optimizer state for resampled features
        # This prevents momentum from stale gradients affecting the new feature
        for param_name, param in self.sae.named_parameters():
            if param_name in ("encoder.weight", "encoder.bias", "decoder.weight"):
                state = self.opt.state.get(param, {})
                if "exp_avg" in state:
                    if param.dim() == 2:
                        if "encoder" in param_name:
                            for feat_idx in dead_indices:
                                state["exp_avg"][feat_idx] = 0.0
                                state["exp_avg_sq"][feat_idx] = 0.0
                        else:  # decoder
                            for feat_idx in dead_indices:
                                state["exp_avg"][:, feat_idx] = 0.0
                                state["exp_avg_sq"][:, feat_idx] = 0.0
                    elif param.dim() == 1:  # bias
                        for feat_idx in dead_indices:
                            state["exp_avg"][feat_idx] = 0.0
                            state["exp_avg_sq"][feat_idx] = 0.0
        
        # Re-normalize decoder after resampling
        self.sae.maybe_renorm_decoder()
        
        # Reset tracking for next window
        self._feature_firing_counts.zero_()
        self._feature_tracking_tokens = 0
        self._total_resampled += n_dead
        
        self._log_info(f"Resampled {n_dead} dead features (total: {self._total_resampled})")
        
        return n_dead

    def _compute_quality_metrics(self, x: torch.Tensor) -> dict:
        """Compute lightweight SAE quality metrics on a batch of activations.

        This runs under `torch.no_grad()` and should not affect training.
        """
        out: dict = {}
        try:
            with torch.no_grad():
                # SparseAutoencoder forward is expected to return (x_hat, z, aux?)
                x_hat, z, _aux = self.sae(x)  # type: ignore[misc]
                # Reconstruction quality
                mse = torch.mean((x_hat - x) ** 2)
                var = torch.mean((x - x.mean(dim=0)) ** 2)
                eps = torch.tensor(1e-8, device=x.device, dtype=x.dtype)
                fvu = mse / (var + eps)

                out["recon_mse_batch"] = float(mse.detach().cpu().item())
                out["fvu_batch"] = float(fvu.detach().cpu().item())

                # Sparsity
                nz = (z != 0).float()
                out["z_frac_nonzero"] = float(nz.mean().detach().cpu().item())
                out["z_l0_per_token"] = float(nz.sum(dim=1).mean().detach().cpu().item())

                # Feature activity (batch-local)
                feat_active = nz.sum(dim=0)
                out["dead_feature_frac_batch"] = float((feat_active == 0).float().mean().detach().cpu().item())
                out["z_mean_abs"] = float(z.abs().mean().detach().cpu().item())
        except Exception:
            # If SAE forward signature differs, skip without failing training.
            pass
        return out

    def _maybe_write_metrics(self, step: int, mean_metrics: dict, quality_metrics: dict) -> None:
        if not (self._main and self.cfg.write_metrics_jsonl):
            return
        rec = {
            "event": "train_metrics",
            "step": int(step),
            "rank": int(self._rank),
            "world_size": int(self._world),
            "device": str(self._device),
            "param_dtype": self.cfg.param_dtype,
            "metrics": mean_metrics,
            "quality": quality_metrics,
        }
        _jsonl_append(self._metrics_path, rec)
        self._last_logged_metrics = rec

    def _maybe_write_summary(self) -> None:
        if not (self._main and self.cfg.write_summary_json):
            return
        summary = {
            "event": "train_summary",
            "final_step": int(self.step),
            "device": str(self._device),
            "param_dtype": self.cfg.param_dtype,
            "last_logged": self._last_logged_metrics,
        }
        self._summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def train(self, activations: Iterable[torch.Tensor], skip_steps: int = 0) -> None:
        """Train the SAE on activation batches.
        
        Args:
            activations: Iterable of activation tensors.
            skip_steps: Number of activation batches to skip (for resuming).
        """
        # If launched with multiple ranks but not intentionally sharded, avoid
        # duplicate training/logging/checkpoint races.
        if self.cfg.only_main_process and not self._main:
            return

        self.sae.to(device=self._device, dtype=self._dtype)
        self.sae.train()
        
        # Handle resume: skip activations that were already trained on
        if skip_steps > 0:
            self._log_info(f"Resuming from step {self.step}: skipping {skip_steps} batches...")
            skip_pbar = tqdm(
                total=skip_steps,
                desc="Skipping (resume)",
                dynamic_ncols=True,
                disable=not self._main,
            )
            skipped = 0
            activation_iter = iter(activations)
            for _ in range(skip_steps):
                try:
                    next(activation_iter)
                    skipped += 1
                    skip_pbar.update(1)
                except StopIteration:
                    self._log_info(f"WARNING: Ran out of data while skipping! Only skipped {skipped}/{skip_steps}")
                    break
            skip_pbar.close()
            self._log_info(f"Skipped {skipped} batches, resuming training from step {self.step}")
            activations = activation_iter  # Continue from here

        pbar = tqdm(
            total=self.cfg.max_steps,
            initial=self.step,  # Start progress bar at current step for resume
            desc=f"train_sae (rank {self._rank}/{self._world})",
            dynamic_ncols=True,
            disable=not self._main,
        )

        metrics_accum: Dict[str, float] = {}
        n_accum = 0

        # One-time environment log
        self._log_info(
            json.dumps(
                {
                    "event": "sae_train_start",
                    "rank": self._rank,
                    "world_size": self._world,
                    "device": str(self._device),
                    "param_dtype": self.cfg.param_dtype,
                    "max_steps": self.cfg.max_steps,
                }
            )
        )

        for x in activations:
            if self.step >= self.cfg.max_steps:
                break

            x = x.to(device=self._device, dtype=self._dtype, non_blocking=True)
            
            # Track feature firing for dead feature detection
            with torch.no_grad():
                _, z_track, _ = self.sae(x)
                self._update_feature_firing_stats(z_track)

            loss, metrics = self.sae.loss(x)

            # Ensure loss is tracked even if SAE.loss doesn't include it.
            metrics = dict(metrics)
            metrics["loss"] = float(loss.detach().item())

            # Diagnostics: show whether L1 is actually contributing
            if "recon_mse" in metrics:
                try:
                    metrics["loss_minus_recon"] = float(metrics["loss"] - float(metrics["recon_mse"]))
                except Exception:
                    pass
            # Use correct L1 term based on l1_form setting
            if hasattr(self.sae, "cfg") and hasattr(self.sae.cfg, "l1_coeff"):
                try:
                    l1_form = getattr(self.sae.cfg, "l1_form", "mean")
                    l1_term = metrics.get("l1_sum" if l1_form == "sum" else "l1_mean", 0.0)
                    metrics["l1_pen_est"] = float(self.sae.cfg.l1_coeff * float(l1_term))
                    metrics["l1_coeff"] = float(self.sae.cfg.l1_coeff)
                    # Note: l1_form is a string, don't add to numeric metrics (available via sae.cfg)
                except Exception:
                    pass

            loss.backward()

            if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.sae.parameters(), self.cfg.grad_clip)

            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            self.sae.maybe_renorm_decoder()

            self.step += 1
            
            # Periodic dead feature resampling
            if (self.cfg.resample_dead_features and 
                self.step % self.cfg.resample_every == 0 and
                self.step >= self.cfg.resample_window_steps):
                n_resampled = self._maybe_resample_dead_features(x)
                metrics["n_resampled"] = float(n_resampled)

            for k, v in metrics.items():
                metrics_accum[k] = metrics_accum.get(k, 0.0) + float(v)
            n_accum += 1

            if self.step % self.cfg.log_every == 0:
                mean_metrics = {k: v / max(1, n_accum) for k, v in metrics_accum.items()}
                # Compute additional quality metrics on the current batch.
                quality = self._compute_quality_metrics(x)
                # Add resampling stats to quality metrics
                if self.cfg.resample_dead_features:
                    quality["total_resampled"] = self._total_resampled
                self._log_info(json.dumps({"step": self.step, **mean_metrics}))
                self._maybe_write_metrics(step=self.step, mean_metrics=mean_metrics, quality_metrics=quality)
                metrics_accum = {}
                n_accum = 0

            if self.step % self.cfg.save_every == 0:
                self.save(f"sae_step{self.step}.pt")

            pbar.update(1)

        pbar.close()
        self.save("sae_final.pt")
        self._maybe_write_summary()


class MultiSAETrainer:
    """Train multiple SAEs on the same activation stream.

    Key correctness points:
      - Train SAEs in float32 by default so the L1 term doesn't vanish in bf16.
      - When combining per-SAE losses for a single backward pass, use SUM, not MEAN.
        Each SAE has disjoint parameters, so summing yields correct per-SAE grads.
        Using mean would scale each SAE's gradients by 1/num_saes (slower training).
    """

    def __init__(self, saes: Dict[str, SparseAutoencoder], cfg: SAETrainConfig, out_dir: Path):
        if not saes:
            raise ValueError("MultiSAETrainer requires at least one SAE")

        self.saes = saes
        self.cfg = cfg
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log = get_logger(__name__)

        self.opts: Dict[str, AdamW] = {
            name: AdamW(sae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay) for name, sae in saes.items()
        }
        self.step = 0

        self._device = _resolve_device(cfg.device)
        self._dtype = _parse_dtype(cfg.param_dtype)
        self._rank = _get_rank()
        self._world = _get_world_size()
        self._main = _is_main_process()

        self._metrics_path = self.out_dir / "train_metrics.jsonl"
        self._summary_path = self.out_dir / "train_summary.json"
        self._last_logged_metrics: Optional[dict] = None
        
        # Dead feature tracking per SAE
        self._feature_firing_counts: Dict[str, torch.Tensor] = {}
        self._feature_tracking_tokens: int = 0
        self._total_resampled: Dict[str, int] = {k: 0 for k in saes.keys()}

        if self._main and self.cfg.write_metrics_jsonl:
            self._metrics_path.write_text("", encoding="utf-8")

    def _log_info(self, msg: str) -> None:
        if self._main:
            self.log.info(msg)

    def _subdir(self, name: str) -> Path:
        d = self.out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, name: str) -> None:
        if self.cfg.only_main_process and not self._main:
            return

        for key, sae in self.saes.items():
            path = self._subdir(key) / name
            tmp = path.with_suffix(path.suffix + ".tmp")
            state = {
                "step": self.step,
                "sae_cfg": sae.cfg.__dict__,
                "train_cfg": self.cfg.__dict__,
                "state_dict": sae.state_dict(),
                "opt_state": self.opts[key].state_dict(),
                "total_resampled": self._total_resampled.get(key, 0),
            }
            torch.save(state, tmp)
            tmp.replace(path)

        self._log_info(f"Saved multi-SAE checkpoint(s): {name}")
    
    # =========================================================================
    # Dead feature resampling for MultiSAETrainer
    # =========================================================================
    @torch.no_grad()
    def _update_feature_firing_stats(self, x: torch.Tensor) -> None:
        """Track feature firing counts for all SAEs."""
        if not self.cfg.resample_dead_features:
            return
        
        for key, sae in self.saes.items():
            _, z, _ = sae(x)
            
            if key not in self._feature_firing_counts:
                self._feature_firing_counts[key] = torch.zeros(
                    sae.d_sae, device="cpu", dtype=torch.long
                )
            
            firings = (z > 0).sum(dim=0).cpu()
            self._feature_firing_counts[key] += firings
        
        self._feature_tracking_tokens += x.shape[0]
    
    @torch.no_grad()
    def _maybe_resample_dead_features(self, x: torch.Tensor) -> Dict[str, int]:
        """Resample dead features for all SAEs. Returns counts per SAE."""
        if not self.cfg.resample_dead_features:
            return {}
        if self._feature_tracking_tokens == 0:
            return {}
        
        results = {}
        
        for key, sae in self.saes.items():
            counts = self._feature_firing_counts.get(key)
            if counts is None:
                continue
            
            firing_rates = counts.float() / max(1, self._feature_tracking_tokens)
            dead_mask = firing_rates < self.cfg.resample_dead_threshold
            n_dead = int(dead_mask.sum().item())
            
            if n_dead == 0:
                results[key] = 0
                # BUG-FIX: Reset counts for this SAE even when n_dead == 0,
                # so the next window starts fresh. Previously only reset when
                # resampling occurred, causing counts to accumulate and masking
                # features that die later in training.
                self._feature_firing_counts[key].zero_()
                continue
            
            dead_indices = dead_mask.nonzero(as_tuple=True)[0].tolist()
            
            # Compute reconstruction residual
            x_hat, _, _ = sae(x)
            residual = x - x_hat
            
            batch_size = residual.shape[0]
            if batch_size < n_dead:
                reps = (n_dead + batch_size - 1) // batch_size
                residual = residual.repeat(reps, 1)
            
            perm = torch.randperm(residual.shape[0], device=residual.device)
            selected_residuals = residual[perm[:n_dead]]
            norms = torch.linalg.vector_norm(selected_residuals, dim=1, keepdim=True).clamp_min(1e-8)
            selected_residuals = selected_residuals / norms
            
            # Reinitialize
            if sae._tied_weights:
                for i, feat_idx in enumerate(dead_indices):
                    sae.encoder.weight.data[feat_idx] = selected_residuals[i]
            else:
                for i, feat_idx in enumerate(dead_indices):
                    sae.decoder.weight.data[:, feat_idx] = selected_residuals[i]
                    sae.encoder.weight.data[feat_idx] = selected_residuals[i]
            
            if sae.encoder.bias is not None:
                for feat_idx in dead_indices:
                    sae.encoder.bias.data[feat_idx] = 0.0
            
            # Reset optimizer state
            opt = self.opts[key]
            for param_name, param in sae.named_parameters():
                if param_name in ("encoder.weight", "encoder.bias", "decoder.weight"):
                    state = opt.state.get(param, {})
                    if "exp_avg" in state:
                        if param.dim() == 2:
                            if "encoder" in param_name:
                                for feat_idx in dead_indices:
                                    state["exp_avg"][feat_idx] = 0.0
                                    state["exp_avg_sq"][feat_idx] = 0.0
                            else:
                                for feat_idx in dead_indices:
                                    state["exp_avg"][:, feat_idx] = 0.0
                                    state["exp_avg_sq"][:, feat_idx] = 0.0
                        elif param.dim() == 1:
                            for feat_idx in dead_indices:
                                state["exp_avg"][feat_idx] = 0.0
                                state["exp_avg_sq"][feat_idx] = 0.0
            
            sae.maybe_renorm_decoder()
            self._total_resampled[key] = self._total_resampled.get(key, 0) + n_dead
            results[key] = n_dead
            
            # Reset counts for this SAE
            self._feature_firing_counts[key].zero_()
        
        self._feature_tracking_tokens = 0
        
        total = sum(results.values())
        if total > 0:
            self._log_info(f"Resampled {results} dead features")
        
        return results

    def _compute_quality_metrics(self, sae: SparseAutoencoder, x: torch.Tensor) -> dict:
        out: dict = {}
        try:
            with torch.no_grad():
                x_hat, z, _aux = sae(x)  # type: ignore[misc]
                mse = torch.mean((x_hat - x) ** 2)
                var = torch.mean((x - x.mean(dim=0)) ** 2)
                eps = torch.tensor(1e-8, device=x.device, dtype=x.dtype)
                fvu = mse / (var + eps)

                out["recon_mse_batch"] = float(mse.detach().cpu().item())
                out["fvu_batch"] = float(fvu.detach().cpu().item())

                nz = (z != 0).float()
                out["z_frac_nonzero"] = float(nz.mean().detach().cpu().item())
                out["z_l0_per_token"] = float(nz.sum(dim=1).mean().detach().cpu().item())
                feat_active = nz.sum(dim=0)
                out["dead_feature_frac_batch"] = float((feat_active == 0).float().mean().detach().cpu().item())
                out["z_mean_abs"] = float(z.abs().mean().detach().cpu().item())
        except Exception:
            pass
        return out

    def _maybe_write_metrics(self, step: int, mean_metrics: dict, quality_by_sae: dict) -> None:
        if not (self._main and self.cfg.write_metrics_jsonl):
            return
        rec = {
            "event": "train_metrics",
            "step": int(step),
            "rank": int(self._rank),
            "world_size": int(self._world),
            "device": str(self._device),
            "param_dtype": self.cfg.param_dtype,
            "metrics": mean_metrics,
            "quality_by_sae": quality_by_sae,
        }
        _jsonl_append(self._metrics_path, rec)
        self._last_logged_metrics = rec

    def _maybe_write_summary(self) -> None:
        if not (self._main and self.cfg.write_summary_json):
            return
        summary = {
            "event": "train_summary",
            "final_step": int(self.step),
            "device": str(self._device),
            "param_dtype": self.cfg.param_dtype,
            "num_saes": int(len(self.saes)),
            "last_logged": self._last_logged_metrics,
        }
        self._summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def load(self, checkpoint_name: str = "sae_step*.pt") -> int:
        """Load checkpoints for all SAEs and restore state.
        
        Args:
            checkpoint_name: Checkpoint filename pattern to load.
            
        Returns: The step number from checkpoints.
        """
        loaded_step = None
        for key, sae in self.saes.items():
            subdir = self._subdir(key)
            ckpt_path = self.find_latest_checkpoint(subdir)
            if ckpt_path is None:
                raise FileNotFoundError(f"No checkpoint found for SAE {key} in {subdir}")
            
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sae.load_state_dict(state["state_dict"])
            
            if "opt_state" in state:
                self.opts[key].load_state_dict(state["opt_state"])
            
            step = state.get("step", 0)
            if loaded_step is None:
                loaded_step = step
            elif loaded_step != step:
                self._log_info(f"WARNING: Step mismatch across SAEs: {loaded_step} vs {step}")
            
            self._total_resampled[key] = state.get("total_resampled", 0)
            self._log_info(f"Loaded {key} from step {step}: {ckpt_path}")
        
        self.step = loaded_step or 0
        return self.step

    @classmethod
    def find_latest_checkpoint(cls, out_dir: Path) -> Optional[Path]:
        """Find the most recent checkpoint in out_dir.
        
        Returns: Path to latest checkpoint, or None if not found.
        """
        if not out_dir.exists():
            return None
        
        checkpoints = list(out_dir.glob("sae_step*.pt"))
        if not checkpoints:
            return None
        
        def get_step(p: Path) -> int:
            try:
                name = p.stem
                return int(name.replace("sae_step", ""))
            except ValueError:
                return 0
        
        checkpoints.sort(key=get_step, reverse=True)
        return checkpoints[0]

    def train(self, activations: Iterable[torch.Tensor], skip_steps: int = 0) -> None:
        """Train all SAEs on activation batches.
        
        Args:
            activations: Iterable of activation tensors.
            skip_steps: Number of activation batches to skip (for resuming).
        """
        if self.cfg.only_main_process and not self._main:
            return

        for sae in self.saes.values():
            sae.to(device=self._device, dtype=self._dtype)
            sae.train()
        
        # Handle resume: skip activations that were already trained on
        if skip_steps > 0:
            self._log_info(f"Resuming from step {self.step}: skipping {skip_steps} batches...")
            skip_pbar = tqdm(
                total=skip_steps,
                desc="Skipping (resume)",
                dynamic_ncols=True,
                disable=not self._main,
            )
            skipped = 0
            activation_iter = iter(activations)
            for _ in range(skip_steps):
                try:
                    next(activation_iter)
                    skipped += 1
                    skip_pbar.update(1)
                except StopIteration:
                    self._log_info(f"WARNING: Ran out of data while skipping! Only skipped {skipped}/{skip_steps}")
                    break
            skip_pbar.close()
            self._log_info(f"Skipped {skipped} batches, resuming training from step {self.step}")
            activations = activation_iter

        pbar = tqdm(
            total=self.cfg.max_steps,
            initial=self.step,  # Start progress bar at current step for resume
            desc=f"train_sae_sweep (rank {self._rank}/{self._world})",
            dynamic_ncols=True,
            disable=not self._main,
        )

        metrics_accum: Dict[str, float] = {}
        n_accum = 0

        self._log_info(
            json.dumps(
                {
                    "event": "multi_sae_train_start",
                    "rank": self._rank,
                    "world_size": self._world,
                    "device": str(self._device),
                    "param_dtype": self.cfg.param_dtype,
                    "num_saes": len(self.saes),
                    "max_steps": self.cfg.max_steps,
                }
            )
        )

        for x in activations:
            if self.step >= self.cfg.max_steps:
                break

            x = x.to(device=self._device, dtype=self._dtype, non_blocking=True)

            # Update feature firing stats before loss computation
            self._update_feature_firing_stats(x)

            # Zero grads
            for opt in self.opts.values():
                opt.zero_grad(set_to_none=True)

            # Compute each SAE loss; backprop once on the SUM for correct scaling.
            losses = []
            per_metrics: Dict[str, Dict[str, float]] = {}

            for key, sae in self.saes.items():
                loss, metrics = sae.loss(x)
                losses.append(loss)

                md = dict(metrics)
                md["loss"] = float(loss.detach().item())
                if "recon_mse" in md:
                    try:
                        md["loss_minus_recon"] = float(md["loss"] - float(md["recon_mse"]))
                    except Exception:
                        pass
                # Use correct L1 term based on l1_form setting
                if hasattr(sae, "cfg") and hasattr(sae.cfg, "l1_coeff"):
                    try:
                        l1_form = getattr(sae.cfg, "l1_form", "mean")
                        l1_term = md.get("l1_sum" if l1_form == "sum" else "l1_mean", 0.0)
                        md["l1_pen_est"] = float(sae.cfg.l1_coeff * float(l1_term))
                        md["l1_coeff"] = float(sae.cfg.l1_coeff)
                        # Note: l1_form is a string, don't add to numeric metrics (available via sae.cfg)
                    except Exception:
                        pass

                per_metrics[key] = {k: float(v) for k, v in md.items()}

            # SUM (not mean) so each SAE sees the intended gradient magnitude.
            loss_total = torch.stack(losses).sum()
            loss_total.backward()

            # Optimizer steps
            for key, sae in self.saes.items():
                if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(sae.parameters(), self.cfg.grad_clip)
                self.opts[key].step()
                sae.maybe_renorm_decoder()

            # Periodically resample dead features
            if (
                self.cfg.resample_dead_features
                and self.step > 0
                and self.step % self.cfg.resample_every == 0
            ):
                self._maybe_resample_dead_features(x)

            self.step += 1

            # Accumulate metrics (prefix with SAE key)
            for key, md in per_metrics.items():
                for k, v in md.items():
                    metrics_accum[f"{key}/{k}"] = metrics_accum.get(f"{key}/{k}", 0.0) + float(v)
            n_accum += 1

            if self.step % self.cfg.log_every == 0:
                mean_metrics = {k: v / max(1, n_accum) for k, v in metrics_accum.items()}
                # Quality metrics per SAE on this batch (best-effort).
                quality_by_sae = {k: self._compute_quality_metrics(sae, x) for k, sae in self.saes.items()}
                self._log_info(json.dumps({"step": self.step, **mean_metrics}))
                self._maybe_write_metrics(step=self.step, mean_metrics=mean_metrics, quality_by_sae=quality_by_sae)
                metrics_accum = {}
                n_accum = 0

            if self.step % self.cfg.save_every == 0:
                self.save(f"sae_step{self.step}.pt")

            pbar.update(1)

        pbar.close()
        self.save("sae_final.pt")
        self._maybe_write_summary()
