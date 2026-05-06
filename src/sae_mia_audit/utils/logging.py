from __future__ import annotations

import logging
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(level: int = logging.INFO, rich_tracebacks: bool = True) -> None:
    """Configure logging once (safe to call multiple times)."""
    handlers = [RichHandler(console=Console(stderr=True), rich_tracebacks=rich_tracebacks)]
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name if name else "sae_mia_audit")
