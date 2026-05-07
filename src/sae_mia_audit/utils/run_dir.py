"""Run directory and reproducibility-snapshot helpers.

Provides timestamped output directories that are safe under ``accelerate``
and ``torchrun``, and writes config, git-commit, environment, and ``pip
freeze`` snapshots alongside each run.
"""
from __future__ import annotations

import json
import os
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.distributed as dist


# -------------------------
# Distributed helpers
# -------------------------

def _dist_available() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    if _dist_available():
        return int(dist.get_rank())
    return int(os.environ.get("RANK", "0"))


def _get_world_size() -> int:
    if _dist_available():
        return int(dist.get_world_size())
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_main_process() -> bool:
    return _get_rank() == 0


# -------------------------
# Run directory creation
# -------------------------

def make_run_dir(root: str, name: str) -> Path:
    """Create a timestamped run directory safe under accelerate/torchrun/SLURM.

    Rank-0 creates the directory.
    Other ranks wait until it exists.
    """
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root) / f"{name}__{ts}"

    if _is_main_process():
        run_dir.mkdir(parents=True, exist_ok=True)

    # Synchronize so all ranks see the directory
    if _dist_available():
        dist.barrier()
    else:
        # Fallback for accelerate edge cases where dist isn't initialized yet
        for _ in range(50):
            if run_dir.exists():
                break
            time.sleep(0.1)

    return run_dir


# -------------------------
# Reproducibility snapshot
# -------------------------

def _try_git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _torch_env_snapshot() -> dict[str, Any]:
    snap: dict[str, Any] = {
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": (torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
    }

    if torch.cuda.is_available():
        try:
            snap["cuda_device_count"] = int(torch.cuda.device_count())
            devices = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices.append(
                    {
                        "index": int(i),
                        "name": getattr(props, "name", None),
                        "total_memory_bytes": int(getattr(props, "total_memory", 0)),
                        "major": int(getattr(props, "major", -1)),
                        "minor": int(getattr(props, "minor", -1)),
                    }
                )
            snap["cuda_devices"] = devices
        except Exception:
            pass

    return snap


def snapshot_reproducibility(run_dir: Path, config: Mapping[str, Any]) -> None:
    """Write config, git commit (if available), and environment info.

    This function is safe under multi-process launchers:
      - only rank-0 writes files
      - other ranks return immediately
    """
    if not _is_main_process():
        return

    # Config (stable filename used by existing scripts)
    (run_dir / "config.json").write_text(
        json.dumps(dict(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Git commit (best effort)
    commit = _try_git_commit()
    if commit is not None:
        (run_dir / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")

    # pip freeze (best effort)
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.DEVNULL,
        )
        (run_dir / "requirements.freeze.txt").write_bytes(out)
    except Exception:
        pass

    # Environment snapshot (stable filename used by existing scripts)
    env = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "HF_HOME": os.environ.get("HF_HOME"),
        "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
        "RANK": os.environ.get("RANK"),
        "LOCAL_RANK": os.environ.get("LOCAL_RANK"),
        "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
    }
    (run_dir / "env.json").write_text(
        json.dumps(env, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # New: run_metadata.json (camera-ready artifact)
    meta = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "argv": list(sys.argv),
        "command": " ".join(shlex.quote(x) for x in sys.argv),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "distributed": {
            "rank": _get_rank(),
            "world_size": _get_world_size(),
        },
        "torch": _torch_env_snapshot(),
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )


# -------------------------
# Utilities
# -------------------------

def dataclass_to_dict(dc: Any) -> dict[str, Any]:
    return asdict(dc)
