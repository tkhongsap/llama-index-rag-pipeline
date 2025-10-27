"""Compatibility shim for the sales promotion package."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent.parent / "src-sale-promotion"
_COMPAT_PATH = _PACKAGE_DIR / "_compat_src_sale_promotion.py"

if not _COMPAT_PATH.exists():  # pragma: no cover - defensive guard
    raise ImportError(
        "The legacy 'src-sale-promotion' package directory is missing. "
        "Cannot initialise compatibility layer for 'src_sale_promotion'."
    )

_spec = importlib.util.spec_from_file_location(
    "src_sale_promotion.__compat__", _COMPAT_PATH
)
if _spec is None or _spec.loader is None:  # pragma: no cover - importlib safety
    raise ImportError("Unable to load compatibility helpers for 'src_sale_promotion'.")

_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
_module.install(sys.modules[__name__])
