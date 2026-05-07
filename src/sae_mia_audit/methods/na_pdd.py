"""NA-PDD: neuron-activation-based pretraining-data detection.

Implementation of the method introduced in Tang, Zhu, & Yang (EMNLP 2025),
"Identifying Pre-training Data in LLMs: A Neuron Activation-Based Detection
Framework". Original repository:
https://github.com/tanghongyi0406/CCNewsPDD .
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers.activations import ACT2FN

from sae_mia_audit.data.tokenizer import TokenizeConfig, tokenize_batch
from sae_mia_audit.models.wrapper import CausalLMWrapper


@dataclass(frozen=True)
class NAPDDConfig:
    """Configuration for the NA-PDD neuron-activation membership-detection method.

    ``tau`` is the activation threshold; ``alpha`` weights member-vs-nonmember
    log-odds; ``top_k_layers`` selects how many layers to aggregate; ``agg``
    is the cross-token reduction (``"max"`` or ``"mean"``).
    """

    tau: float = 0.0
    alpha: float = 2.0
    top_k_layers: int = 5
    # batching
    batch_size: int = 4
    seq_len: int = 256
    # aggregation for neuron value across tokens
    agg: str = "max"  # 'max' or 'mean'


def _get_act_fn(model: torch.nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    cfg = model.config
    act_name = getattr(cfg, "hidden_act", None) or getattr(cfg, "activation_function", None) or "gelu"
    if isinstance(act_name, str):
        if act_name in ACT2FN:
            return ACT2FN[act_name]
        if act_name.lower() in ACT2FN:
            return ACT2FN[act_name.lower()]
    return ACT2FN["gelu"]


def _mlp_pre_act_module(layer: torch.nn.Module):
    # GPTNeoX (Pythia)
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "dense_h_to_4h"):
        return layer.mlp.dense_h_to_4h
    # GPT-Neo (EleutherAI/gpt-neo-*) — uses c_fc / c_proj naming
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "c_fc"):
        return layer.mlp.c_fc
    # OPT
    if hasattr(layer, "fc1"):
        return layer.fc1
    # LLaMA
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate_proj"):
        return layer.mlp.gate_proj
    raise ValueError(f"Unsupported layer type for NA-PDD neurons: {type(layer)}")


class _ActivationRecorder:
    """Record per-layer boolean activations (token-aggregated) on GPU.

    Stores: layer_idx -> active_bool [B, F] on the same device as model activations.
    Handles modules that output either [B,T,F], [B*T,F], or [B,F] by reshaping when possible.
    
    Aggregation respects attention_mask to avoid padding bias.
    """

    def __init__(self, act_fn: Callable[[torch.Tensor], torch.Tensor], tau: float, agg: str):
        self.act_fn = act_fn
        self.tau = float(tau)
        self.agg = agg
        self.active_by_layer: Dict[int, torch.Tensor] = {}  # layer -> [B, F] bool
        # Store attention mask for masked aggregation (prevents padding bias)
        self._current_attn_mask: Optional[torch.Tensor] = None  # [B, T] bool

    def set_attention_mask(self, attn_mask: Optional[torch.Tensor]):
        """Set attention mask for current batch (call before forward)."""
        self._current_attn_mask = attn_mask

    def clear(self):
        self.active_by_layer = {}
        self._current_attn_mask = None

    def hook_for_layer(self, layer_idx: int):
        def _hook(_module, inputs, output):
            x = output
            if not torch.is_tensor(x):
                return

            # Attempt to canonicalize to [B, T, F]
            if x.ndim == 3:
                x3 = x
            elif x.ndim == 2:
                # Could be [B*T, F] or [B, F]. Use input shape if available.
                inp = inputs[0] if inputs and torch.is_tensor(inputs[0]) else None
                if inp is not None and inp.ndim == 3:
                    B, T = int(inp.shape[0]), int(inp.shape[1])
                    if x.shape[0] == B * T:
                        x3 = x.view(B, T, -1)
                    elif x.shape[0] == B:
                        x3 = x.unsqueeze(1)  # [B,1,F]
                    else:
                        # Fallback: treat as [B,1,F] with B=x.shape[0]
                        x3 = x.unsqueeze(1)
                else:
                    # No reliable T; treat as [B,1,F]
                    x3 = x.unsqueeze(1)
            elif x.ndim == 1:
                # Treat as [1,1,F]
                x3 = x.view(1, 1, -1)
            else:
                return

            x3 = self.act_fn(x3)  # [B, T, F]

            # B: Apply attention mask to avoid padding bias
            # If mask is provided, mask out padding positions before aggregation
            mask = self._current_attn_mask  # [B, T] or None
            if mask is not None and x3.shape[1] == mask.shape[1]:
                # Expand mask to [B, T, 1] for broadcasting
                mask_expanded = mask.unsqueeze(-1).to(x3.dtype)  # [B, T, 1]
                if self.agg == "max":
                    # Set padding positions to -inf before max
                    x3_masked = x3.masked_fill(mask_expanded == 0, float('-inf'))
                    x_agg = x3_masked.max(dim=1).values  # [B, F]
                    # Handle all-padding case (shouldn't happen with valid data)
                    x_agg = torch.where(torch.isinf(x_agg), torch.zeros_like(x_agg), x_agg)
                elif self.agg == "mean":
                    # Masked mean: sum(x * mask) / sum(mask)
                    x3_masked = x3 * mask_expanded  # [B, T, F]
                    sum_x = x3_masked.sum(dim=1)  # [B, F]
                    count = mask_expanded.sum(dim=1).clamp(min=1)  # [B, 1]
                    x_agg = sum_x / count  # [B, F]
                else:
                    raise ValueError(f"Unknown agg={self.agg}")
            else:
                # No mask or shape mismatch: use original aggregation
                if self.agg == "max":
                    x_agg = x3.max(dim=1).values  # [B, F]
                elif self.agg == "mean":
                    x_agg = x3.mean(dim=1)  # [B, F]
                else:
                    raise ValueError(f"Unknown agg={self.agg}")

            self.active_by_layer[layer_idx] = (x_agg > self.tau).detach()

        return _hook


class NAPDD:
    """Neuron Activation-based PDD (NA-PDD), GPU-native and hook-robust."""

    def __init__(self, model: CausalLMWrapper, cfg: NAPDDConfig):
        self.model = model
        self.cfg = cfg
        self.device = next(self.model.model.parameters()).device

        self.act_fn = _get_act_fn(self.model.model)
        self.layers = self.model.get_transformer_layers()

        self.recorder = _ActivationRecorder(self.act_fn, tau=cfg.tau, agg=cfg.agg)
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        for li, layer in enumerate(self.layers):
            mod = _mlp_pre_act_module(layer)
            self.handles.append(mod.register_forward_hook(self.recorder.hook_for_layer(li)))

        # Learned state after fit()
        self.f_train: Optional[List[torch.Tensor]] = None
        self.f_non: Optional[List[torch.Tensor]] = None
        self.Nmem: Optional[List[torch.Tensor]] = None  # list of bool tensors per layer [F] on self.device
        self.Nnon: Optional[List[torch.Tensor]] = None
        self.discriminative_layers: Optional[List[int]] = None
        self._denom_mem: Optional[List[float]] = None
        self._denom_non: Optional[List[float]] = None

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    @torch.no_grad()
    def _batch_iter(self, texts: Sequence[str]):
        bs = int(self.cfg.batch_size)
        for i in range(0, len(texts), bs):
            yield texts[i : i + bs]

    @staticmethod
    def _infer_F(active: torch.Tensor) -> int:
        # active expected [B,F]
        if active.ndim == 2:
            return int(active.shape[1])
        if active.ndim == 1:
            return int(active.shape[0])
        return int(active.shape[-1])

    @torch.no_grad()
    def _collect_freqs(self, texts: Sequence[str]) -> Tuple[List[torch.Tensor], int]:
        """Return per-layer activation counts (float32 on GPU) and number of examples."""
        # Explicit random_crop=False for deterministic evaluation
        tok_cfg = TokenizeConfig(seq_len=self.cfg.seq_len, random_crop=False)
        counts: List[torch.Tensor] = []
        counts_init = False
        n = 0

        total = (len(texts) + self.cfg.batch_size - 1) // self.cfg.batch_size
        for chunk in tqdm(self._batch_iter(texts), total=total, desc="na_pdd_freqs", dynamic_ncols=True):
            batch = tokenize_batch(self.model.tokenizer, list(chunk), tok_cfg)
            input_ids = batch["input_ids"].to(self.device)
            attn = batch.get("attention_mask", None)
            if attn is not None:
                attn = attn.to(self.device)

            self.recorder.clear()
            # B: Set attention mask BEFORE forward so hooks can use it for masked aggregation
            self.recorder.set_attention_mask(attn)
            _ = self.model.forward(input_ids=input_ids, attention_mask=attn, output_hidden_states=False)
            B = int(input_ids.shape[0])

            if not counts_init:
                counts = []
                for li in range(len(self.layers)):
                    active = self.recorder.active_by_layer.get(li)
                    if active is None:
                        # If this triggers, the hook isn't firing. With the robust hook above,
                        # the only remaining cause is that _mlp_pre_act_module picked a module
                        # that is not called in this architecture/version.
                        raise RuntimeError(f"Missing activations for layer {li} (hook did not fire).")
                    F = self._infer_F(active)
                    counts.append(torch.zeros(F, dtype=torch.float32, device=self.device))
                counts_init = True

            for li in range(len(self.layers)):
                active = self.recorder.active_by_layer.get(li)
                if active is None:
                    raise RuntimeError(f"Missing activations for layer {li} (hook did not fire).")
                # active: [B,F] bool
                counts[li] += active.to(dtype=torch.float32).sum(dim=0)

            n += B

        return counts, n

    @torch.no_grad()
    def fit(self, member_texts: Sequence[str], nonmember_texts: Sequence[str]) -> None:
        train_counts, n_train = self._collect_freqs(member_texts)
        non_counts, n_non = self._collect_freqs(nonmember_texts)

        f_train = [c / float(max(1, n_train)) for c in train_counts]
        f_non = [c / float(max(1, n_non)) for c in non_counts]
        self.f_train = f_train
        self.f_non = f_non

        alpha = float(self.cfg.alpha)
        self.Nmem = [(ft > alpha * fn).to(dtype=torch.bool, device=self.device) for ft, fn in zip(f_train, f_non)]
        self.Nnon = [(fn > alpha * ft).to(dtype=torch.bool, device=self.device) for ft, fn in zip(f_train, f_non)]

        self._denom_mem = [float(max(1, int(m.sum().item()))) for m in self.Nmem]
        self._denom_non = [float(max(1, int(m.sum().item()))) for m in self.Nnon]

        S = [int(self.Nmem[i].sum().item() - self.Nnon[i].sum().item()) for i in range(len(self.layers))]
        K = min(int(self.cfg.top_k_layers), len(S))
        self.discriminative_layers = sorted(range(len(S)), key=lambda i: S[i], reverse=True)[:K]

    @torch.no_grad()
    def score_texts(self, texts: Sequence[str]) -> np.ndarray:
        if self.Nmem is None or self.Nnon is None or self.discriminative_layers is None:
            raise RuntimeError("Call fit() before scoring.")
        if self._denom_mem is None or self._denom_non is None:
            raise RuntimeError("Internal error: denominators not initialized. Call fit() again.")

        # Explicit random_crop=False for deterministic evaluation
        tok_cfg = TokenizeConfig(seq_len=self.cfg.seq_len, random_crop=False)
        scores: List[float] = []

        total = (len(texts) + self.cfg.batch_size - 1) // self.cfg.batch_size
        for chunk in tqdm(self._batch_iter(texts), total=total, desc="na_pdd_score", dynamic_ncols=True):
            batch = tokenize_batch(self.model.tokenizer, list(chunk), tok_cfg)
            input_ids = batch["input_ids"].to(self.device)
            attn = batch.get("attention_mask", None)
            if attn is not None:
                attn = attn.to(self.device)

            self.recorder.clear()
            # B: Set attention mask BEFORE forward so hooks can use it for masked aggregation
            self.recorder.set_attention_mask(attn)
            _ = self.model.forward(input_ids=input_ids, attention_mask=attn, output_hidden_states=False)

            B = int(input_ids.shape[0])
            for b in range(B):
                smem_vals: List[float] = []
                snon_vals: List[float] = []

                for li in self.discriminative_layers:
                    active_layer = self.recorder.active_by_layer.get(li)
                    if active_layer is None:
                        raise RuntimeError(f"Missing activations for layer {li} during scoring (hook did not fire).")

                    # active_layer: [B,F]
                    active = active_layer[b]
                    Nmem = self.Nmem[li]
                    Nnon = self.Nnon[li]
                    denom_mem = self._denom_mem[li]
                    denom_non = self._denom_non[li]

                    # All tensors on GPU -> no device mismatch
                    smem = float((active & Nmem).sum().item()) / denom_mem
                    snon = float((active & Nnon).sum().item()) / denom_non
                    smem_vals.append(smem)
                    snon_vals.append(snon)

                sbar_mem = float(np.mean(smem_vals)) if smem_vals else 0.0
                sbar_non = float(np.mean(snon_vals)) if snon_vals else 1e-6
                scores.append(sbar_mem / max(1e-6, sbar_non))

        return np.asarray(scores, dtype=np.float64)
