# `sae_mia_audit.models`

HuggingFace causal-LM wrappers used by the paper scripts.

| Module | Purpose | Used by |
| --- | --- | --- |
| `wrapper.py` | `load_model_and_tokenizer()` and `CausalLMWrapper` (model loading, residual-stream activation capture, layer hook registration). | All paper scripts that read residual-stream activations |
| `logprobs.py` | `next_token_logprobs_and_stats()` (next-token log-probabilities for the loss-attack baseline). | `sae_mia_audit.methods.baselines` |

The activation-capture path uses HuggingFace's standard layer-hook interface against the residual stream at the per-model SAE analysis layer fixed in the manuscript Appendix Table `tab:model_details`.
