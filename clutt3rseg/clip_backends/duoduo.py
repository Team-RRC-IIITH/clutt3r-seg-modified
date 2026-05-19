"""Adapter for using DuoduoCLIP as an external dependency.

Clutt3R-Seg intentionally does not vendor DuoduoCLIP source code. Install or
clone the upstream project separately, then set DUODUOCLIP_ROOT to that checkout
or pass --duoduo-root to the entry-point scripts.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Optional, Union


DEFAULT_DUODUO_CHECKPOINT = "Four_1to6F_bs1600_LT6.ckpt"


class DuoduoCLIPNotAvailableError(ImportError):
    """Raised when the external DuoduoCLIP checkout cannot be imported."""


def _resolve_duoduo_root(duoduo_root: Optional[Union[str, Path]]) -> Optional[Path]:
    root_value = duoduo_root or os.environ.get("DUODUOCLIP_ROOT")
    if not root_value:
        return None

    root = Path(root_value).expanduser().resolve()
    wrapper_path = root / "src" / "model" / "wrapper.py"
    if not wrapper_path.exists():
        raise DuoduoCLIPNotAvailableError(
            f"DUODUOCLIP_ROOT must point to an upstream DuoduoCLIP checkout. "
            f"Expected {wrapper_path}."
        )
    return root


def _import_duoduo_wrapper(duoduo_root: Optional[Union[str, Path]] = None):
    root = _resolve_duoduo_root(duoduo_root)
    if root is not None and str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        wrapper = importlib.import_module("src.model.wrapper")
    except Exception as exc:
        raise DuoduoCLIPNotAvailableError(
            "DuoduoCLIP is required for CLIP feature extraction but was not "
            "found. Clone https://github.com/3dlg-hcvc/DuoduoCLIP separately "
            "and set DUODUOCLIP_ROOT=/path/to/DuoduoCLIP, or pass "
            "--duoduo-root /path/to/DuoduoCLIP."
        ) from exc

    if root is not None:
        imported_from = Path(getattr(wrapper, "__file__", "")).resolve()
        if not imported_from.is_relative_to(root):
            raise DuoduoCLIPNotAvailableError(
                f"Imported src.model.wrapper from {imported_from}, but "
                f"DUODUOCLIP_ROOT points to {root}. Start a fresh Python "
                "process or remove the conflicting src package from PYTHONPATH."
            )

    if not hasattr(wrapper, "get_model"):
        raise DuoduoCLIPNotAvailableError(
            "The imported DuoduoCLIP wrapper does not expose get_model()."
        )
    return wrapper


def load_duoduo_clip(
    checkpoint: str = DEFAULT_DUODUO_CHECKPOINT,
    device: str = "cuda",
    duoduo_root: Optional[Union[str, Path]] = None,
):
    """Load a DuoduoCLIP model from the external DuoduoCLIP package."""
    wrapper = _import_duoduo_wrapper(duoduo_root)
    return wrapper.get_model(checkpoint, device=device)

