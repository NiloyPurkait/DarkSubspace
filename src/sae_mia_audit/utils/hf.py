"""HuggingFace model/tokenizer loading helpers with retries and dtype/device resolution."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import torch

# Optional dependency: allow importing the repo in environments without
# transformers installed (e.g., unit tests that don't touch HF code).
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

logger = logging.getLogger(__name__)


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
    """Resolve device strings with an 'auto' default.

    Supported:
      - auto: cuda if available else cpu
      - cuda: uses cuda:{LOCAL_RANK} when available (multi-proc launchers)
      - cuda:N / cpu

    Notes
    -----
    If CUDA is unavailable but a CUDA device is requested, we fall back to CPU.
    """
    d = str(device).lower().strip()
    if d == "auto":
        d = "cuda" if torch.cuda.is_available() else "cpu"

    if d.startswith("cuda"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        if d == "cuda":
            lr = _get_local_rank()
            n = torch.cuda.device_count()
            if n > 0 and 0 <= lr < n:
                return torch.device("cuda", lr)
            return torch.device("cuda", 0)
        return torch.device(d)

    return torch.device("cpu")


def _resolve_dtype(dtype_str: str, device: torch.device) -> torch.dtype:
    """Resolve a dtype string to a torch.dtype.

    Supports:
      - auto: bf16 on CUDA when supported else fp16; fp32 on CPU
      - bfloat16/bf16, float16/fp16, float32/fp32

    For backward compatibility, unknown strings fall back to bf16.
    """
    s = str(dtype_str).lower().strip()

    if s == "auto":
        if device.type == "cuda":
            # Prefer bf16 if supported; otherwise fp16.
            try:
                is_bf16 = bool(torch.cuda.is_bf16_supported())
            except Exception:
                # Older torch versions may not have is_bf16_supported.
                is_bf16 = True
            return torch.bfloat16 if is_bf16 else torch.float16
        return torch.float32

    if s in ("bfloat16", "bf16", "torch.bfloat16"):
        return torch.bfloat16
    if s in ("float16", "fp16", "torch.float16"):
        return torch.float16
    if s in ("float32", "fp32", "torch.float32"):
        return torch.float32

    return torch.bfloat16


@dataclass(frozen=True)
class HFModelSpec:
    name_or_path: str
    revision: Optional[str] = None
    trust_remote_code: bool = False

    # Back-compat: historically defaulted to bf16.
    # We also accept "auto".
    torch_dtype: str = "bfloat16"  # 'auto' | 'bfloat16' | 'float16' | 'float32'

    # If device_map is not None (e.g., 'auto'), HF may shard weights across devices.
    # In that case we must not call .to(device).
    device_map: Optional[str] = None  # e.g. 'auto' or None

    # New: explicit device placement when device_map is None.
    # Default auto: cuda if available else cpu.
    device: str = "auto"


def load_tokenizer(name_or_path: str, revision: Optional[str] = None, trust_remote_code: bool = False):
    if AutoTokenizer is None:  # pragma: no cover
        raise ImportError("transformers is required to load_tokenizer(). Install with: pip install transformers")

    # Retry logic for HuggingFace Hub downloads to handle transient network issues
    max_retries = int(os.environ.get("HF_HUB_MAX_RETRIES", "3"))
    retry_delay = float(os.environ.get("HF_HUB_RETRY_DELAY", "10.0"))
    
    last_exception = None
    for attempt in range(max_retries):
        try:
            tok = AutoTokenizer.from_pretrained(name_or_path, revision=revision, trust_remote_code=trust_remote_code)
            if tok.pad_token_id is None:
                # For causal LMs, pad_token_id is often undefined. Use eos as pad.
                tok.pad_token = tok.eos_token
            return tok
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            # Retry on timeout, connection, or network errors
            is_transient = any(kw in error_str for kw in ["timeout", "connection", "network", "temporarily"])
            
            if attempt < max_retries - 1 and is_transient:
                delay = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Tokenizer download attempt {attempt + 1}/{max_retries} failed: {e}")
                logger.warning(f"Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                break
    
    logger.error(f"Failed to load tokenizer after {max_retries} attempts")
    raise last_exception


def load_causal_lm(spec: HFModelSpec):
    if AutoModelForCausalLM is None:  # pragma: no cover
        raise ImportError("transformers is required to load_causal_lm(). Install with: pip install transformers")

    device = _resolve_device(getattr(spec, "device", "auto"))
    dtype = _resolve_dtype(getattr(spec, "torch_dtype", "bfloat16"), device=device)

    # Retry logic for HuggingFace Hub downloads to handle transient network issues
    max_retries = int(os.environ.get("HF_HUB_MAX_RETRIES", "3"))
    retry_delay = float(os.environ.get("HF_HUB_RETRY_DELAY", "10.0"))
    
    last_exception = None
    for attempt in range(max_retries):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                spec.name_or_path,
                revision=getattr(spec, "revision", None),
                trust_remote_code=getattr(spec, "trust_remote_code", False),
                torch_dtype=dtype,
                device_map=getattr(spec, "device_map", None),
            )

            # If not using device_map sharding, ensure we actually move parameters.
            if getattr(spec, "device_map", None) is None:
                model.to(device)

            model.eval()
            return model
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            # Retry on timeout, connection, or network errors
            is_transient = any(kw in error_str for kw in ["timeout", "connection", "network", "temporarily"])
            
            if attempt < max_retries - 1 and is_transient:
                delay = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Model download attempt {attempt + 1}/{max_retries} failed: {e}")
                logger.warning(f"Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                break
    
    logger.error(f"Failed to load model after {max_retries} attempts")
    raise last_exception
