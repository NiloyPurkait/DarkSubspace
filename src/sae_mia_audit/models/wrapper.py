"""Causal language model wrapper and activation-site descriptors.

Provides ``CausalLMWrapper``, ``ModelInfo``, and ``ActivationSite``, plus the
``load_model_and_tokenizer`` entry point used by training and evaluation
scripts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import torch

# Optional dependency: allow importing the repo without transformers installed.
try:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase  # type: ignore
except Exception:  # pragma: no cover
    PreTrainedModel = Any  # type: ignore
    PreTrainedTokenizerBase = Any  # type: ignore

from sae_mia_audit.utils.hf import HFModelSpec, load_causal_lm, load_tokenizer


# -----------------------------------------------------------------------------
# Activation Site Specification (Group C: mechanistic interventions)
# -----------------------------------------------------------------------------

# Supported tensor names at a transformer block boundary
TensorName = Literal["residual_post_block", "residual_pre_mlp", "residual_post_attn"]


@dataclass(frozen=True)
class ActivationSite:
    """Specifies the exact location of an activation tensor in a transformer.
    
    This is the "spine" of Group C: attribution and intervention must use the
    SAME site that the SAE was trained on.
    
    Attributes:
        layer_idx: 0-indexed transformer layer. For `residual_post_block`, this
            corresponds to hidden_states[layer_idx + 1] when using HF output.
        tensor_name: Which tensor within the layer:
            - "residual_post_block": output of the full transformer block (default for SAE training)
            - "residual_pre_mlp": residual stream before MLP (after attention + residual)
            - "residual_post_attn": output of attention sublayer (before MLP)
        d_model: Expected hidden dimension (for validation).
    
    Note:
        Most SAEs in this repo are trained on "residual_post_block", which is the
        output of transformer layer L (i.e., hidden_states[L+1] in HF convention).
    """
    layer_idx: int
    tensor_name: TensorName = "residual_post_block"
    d_model: Optional[int] = None

    def __post_init__(self):
        if self.layer_idx < 0:
            raise ValueError(f"layer_idx must be >= 0, got {self.layer_idx}")

    def matches(self, other: "ActivationSite") -> bool:
        """Check if two sites refer to the same activation location."""
        return self.layer_idx == other.layer_idx and self.tensor_name == other.tensor_name


@dataclass(frozen=True)
class ModelInfo:
    """Lightweight description of a loaded causal language model.

    Captures the HF identifier (``name_or_path``), inferred layer count and
    hidden size, and the model architecture family (e.g., ``gpt_neox``,
    ``opt``).
    """

    name_or_path: str
    n_layers: int
    d_model: int
    model_type: str  # e.g. 'gpt_neox', 'opt', ...


def _infer_model_info(model: PreTrainedModel, name_or_path: str) -> ModelInfo:
    cfg = model.config
    model_type = getattr(cfg, "model_type", type(cfg).__name__)
    # Try common attributes
    n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None) or getattr(cfg, "num_layers", None)
    if n_layers is None:
        raise ValueError(f"Could not infer n_layers from config fields for {name_or_path}: {cfg}")

    d_model = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    if d_model is None:
        raise ValueError(f"Could not infer d_model from config fields for {name_or_path}: {cfg}")

    return ModelInfo(name_or_path=name_or_path, n_layers=int(n_layers), d_model=int(d_model), model_type=str(model_type))


class CausalLMWrapper:
    """Small convenience wrapper around a HuggingFace causal LM.

    This wrapper intentionally provides:
      - `register_residual_hook(...)` for interventions (used by mechanistic scripts)
      - `capture_layer_output(...)` for efficient activation extraction without
        `output_hidden_states=True` (used by SAE training scripts).
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, info: ModelInfo):
        self.model = model
        self.tokenizer = tokenizer
        self.info = info

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ):
        """Run a no-grad forward pass with KV cache disabled.

        Returns the raw HF model output. Pass ``output_hidden_states=True``
        to get all layer activations.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            use_cache=False,
        )

    def get_transformer_layers(self):
        """Return a list-like of transformer block modules for hook registration."""
        # GPTNeoX (Pythia): model.gpt_neox.layers
        if hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "layers"):
            return self.model.gpt_neox.layers
        # GPT-Neo: model.transformer.h
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        # OPT: model.model.decoder.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "decoder") and hasattr(self.model.model.decoder, "layers"):
            return self.model.model.decoder.layers
        # LLaMA-like: model.model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        raise ValueError(f"Unsupported model structure for hooks: {type(self.model)}")

    def register_residual_hook(self, layer_idx: int, fn: Callable[[torch.Tensor], torch.Tensor]):
        """Register a forward hook to modify the *output hidden state* of a transformer layer.

        The hook replaces the first element of the module output if it is a tuple.
        """
        layers = self.get_transformer_layers()
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"layer_idx out of range: {layer_idx} / {len(layers)}")

        module = layers[layer_idx]

        def _hook(_module, _inputs, output):
            if isinstance(output, tuple):
                h = output[0]
                h2 = fn(h)
                return (h2,) + output[1:]
            return fn(output)

        handle = module.register_forward_hook(_hook)
        return handle

    @torch.no_grad()
    def capture_layer_output(
        self,
        layer_idx: int,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Capture the [B, T, D] hidden state *after* a transformer block.

        This avoids `output_hidden_states=True`, which materializes all layers.

        Returns:
            h: torch.Tensor of shape [B, T, D] on the model device/dtype.
        """
        layers = self.get_transformer_layers()
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"layer_idx out of range: {layer_idx} / {len(layers)}")

        holder: dict[str, torch.Tensor] = {}

        def _cap(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            # Detach to ensure we don't keep graph references.
            holder["h"] = h.detach()
            return output

        handle = layers[layer_idx].register_forward_hook(_cap)
        try:
            _ = self.forward(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        finally:
            handle.remove()

        if "h" not in holder:
            raise RuntimeError("Activation capture hook did not fire; unsupported model block output?")
        return holder["h"]

    def resolve_layer_indices(self, which: str):
        """Convenience for 'early', 'mid', 'late'."""
        n = self.info.n_layers
        if which == "early":
            return max(0, n // 8)
        if which == "mid":
            return n // 2
        if which == "late":
            return max(0, n - 2)
        raise ValueError(f"Unknown which={which}")

    # -------------------------------------------------------------------------
    # Group C: Activation extraction and patching at exact sites
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def extract(
        self,
        site: ActivationSite,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract activation tensor at the specified site.

        This is the "read" side of the Group C spine. The returned tensor has
        shape [B, T, D] and can be passed to an SAE's encode() method.

        Args:
            site: Activation site specification.
            input_ids: [B, T] token ids.
            attention_mask: [B, T] attention mask (1 = real token, 0 = pad).

        Returns:
            h: [B, T, D] activation tensor (detached, on model device/dtype).
        """
        if site.tensor_name == "residual_post_block":
            # This is the default SAE training site: output of transformer layer
            return self.capture_layer_output(site.layer_idx, input_ids, attention_mask)

        # For other tensor names, we need more specific hooks
        # These are less common but included for completeness
        layers = self.get_transformer_layers()
        if site.layer_idx < 0 or site.layer_idx >= len(layers):
            raise ValueError(f"layer_idx out of range: {site.layer_idx} / {len(layers)}")

        holder: Dict[str, torch.Tensor] = {}
        block = layers[site.layer_idx]

        if site.tensor_name == "residual_pre_mlp":
            # Hook into the MLP's input
            mlp = self._get_mlp_module(block)
            if mlp is None:
                raise ValueError(f"Could not find MLP module in layer {site.layer_idx}")

            def _cap_pre_mlp(_module, inputs, _output):
                # MLP input is typically (hidden_states,) or (hidden_states, ...) 
                h = inputs[0] if isinstance(inputs, tuple) else inputs
                holder["h"] = h.detach()

            handle = mlp.register_forward_hook(_cap_pre_mlp)
        elif site.tensor_name == "residual_post_attn":
            # Hook into attention output (before MLP)
            attn = self._get_attn_module(block)
            if attn is None:
                raise ValueError(f"Could not find attention module in layer {site.layer_idx}")

            def _cap_post_attn(_module, _inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                holder["h"] = h.detach()

            handle = attn.register_forward_hook(_cap_post_attn)
        else:
            raise ValueError(f"Unsupported tensor_name: {site.tensor_name}")

        try:
            _ = self.forward(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        finally:
            handle.remove()

        if "h" not in holder:
            raise RuntimeError(f"Activation capture hook did not fire for site {site}")

        # Validate dimensions if specified
        h = holder["h"]
        if site.d_model is not None and h.shape[-1] != site.d_model:
            raise ValueError(f"Extracted activation has d_model={h.shape[-1]}, expected {site.d_model}")

        return h

    @torch.no_grad()
    def forward_with_patch(
        self,
        site: ActivationSite,
        patch_fn: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ):
        """Run forward pass with activation patching at the specified site.

        This is the "write" side of the Group C spine. The patch_fn receives
        the activation tensor and attention mask, and returns the modified
        activation tensor.

        Args:
            site: Activation site specification (must match SAE training site).
            patch_fn: Callable (h, attention_mask) -> h_edited.
                h has shape [B, T, D], attention_mask has shape [B, T] or None.
            input_ids: [B, T] token ids.
            attention_mask: [B, T] attention mask (1 = real token, 0 = pad).
            output_hidden_states: Whether to return all hidden states.

        Returns:
            Model outputs (with patched activations flowing through).
        """
        self.model.eval()

        if site.tensor_name == "residual_post_block":
            # Patch the output of the transformer block
            def _patch_block_output(h: torch.Tensor) -> torch.Tensor:
                return patch_fn(h, attention_mask)

            handle = self.register_residual_hook(site.layer_idx, _patch_block_output)
            try:
                return self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=output_hidden_states,
                )
            finally:
                handle.remove()

        # For other tensor names, construct appropriate hooks
        layers = self.get_transformer_layers()
        block = layers[site.layer_idx]

        if site.tensor_name == "residual_pre_mlp":
            mlp = self._get_mlp_module(block)
            if mlp is None:
                raise ValueError(f"Could not find MLP module in layer {site.layer_idx}")

            def _patch_pre_mlp(_module, inputs):
                h = inputs[0] if isinstance(inputs, tuple) else inputs
                h_edited = patch_fn(h, attention_mask)
                if isinstance(inputs, tuple):
                    return (h_edited,) + inputs[1:]
                return h_edited

            # Use pre-hook to modify input to MLP
            handle = mlp.register_forward_pre_hook(_patch_pre_mlp)

        elif site.tensor_name == "residual_post_attn":
            attn = self._get_attn_module(block)
            if attn is None:
                raise ValueError(f"Could not find attention module in layer {site.layer_idx}")

            def _patch_post_attn(_module, _inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                h_edited = patch_fn(h, attention_mask)
                if isinstance(output, tuple):
                    return (h_edited,) + output[1:]
                return h_edited

            handle = attn.register_forward_hook(_patch_post_attn)
        else:
            raise ValueError(f"Unsupported tensor_name: {site.tensor_name}")

        try:
            return self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
            )
        finally:
            handle.remove()

    def _get_mlp_module(self, block):
        """Get the MLP module from a transformer block."""
        # GPTNeoX: block.mlp
        if hasattr(block, "mlp"):
            return block.mlp
        # OPT: block.fc1 (first part of MLP)
        if hasattr(block, "fc1"):
            return block.fc1
        return None

    def _get_attn_module(self, block):
        """Get the attention module from a transformer block."""
        # GPTNeoX: block.attention
        if hasattr(block, "attention"):
            return block.attention
        # OPT: block.self_attn
        if hasattr(block, "self_attn"):
            return block.self_attn
        return None


def load_model_and_tokenizer(spec: HFModelSpec) -> CausalLMWrapper:
    """Load a HuggingFace causal LM and tokenizer and return them wrapped.

    Returns a :class:`CausalLMWrapper` bundling the loaded model, tokenizer,
    and a :class:`ModelInfo` summary. Raises :class:`ImportError` if the
    optional ``transformers`` dependency is unavailable.
    """
    # Fail with an informative message if transformers is absent.
    if PreTrainedTokenizerBase is Any:  # pragma: no cover
        raise ImportError("transformers is required to load models/tokenizers. Install with: pip install transformers")

    tok = load_tokenizer(spec.name_or_path, revision=getattr(spec, "revision", None), trust_remote_code=getattr(spec, "trust_remote_code", False))
    model = load_causal_lm(spec)
    info = _infer_model_info(model, spec.name_or_path)
    return CausalLMWrapper(model=model, tokenizer=tok, info=info)
