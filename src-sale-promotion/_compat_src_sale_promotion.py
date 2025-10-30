"""Compatibility helpers for exposing the legacy package."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable


def install(target: ModuleType) -> None:
    """Configure ``target`` to proxy the legacy ``src-sale-promotion`` package."""
    package_dir = Path(__file__).resolve().parent
    legacy_init = package_dir / "__init__.py"

    spec = importlib.machinery.ModuleSpec(target.__name__, loader=None, is_package=True)
    spec.submodule_search_locations = [str(package_dir)]

    target.__file__ = str(legacy_init)
    target.__path__ = [str(package_dir)]
    target.__package__ = target.__name__
    target.__spec__ = spec

    _attach_public_attributes(target, legacy_init)


def _attach_public_attributes(module: ModuleType, legacy_init: Path) -> None:
    """Load metadata from the legacy ``__init__`` if present."""
    if not legacy_init.exists():
        return

    spec = importlib.util.spec_from_file_location(
        f"{module.__name__}.__legacy_init__", legacy_init
    )
    if spec is None or spec.loader is None:  # pragma: no cover - importlib fallback
        return

    legacy_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = legacy_module
    spec.loader.exec_module(legacy_module)

    if hasattr(legacy_module, "__all__"):
        public_names: Iterable[str] = legacy_module.__all__  # type: ignore[assignment]
    else:
        public_names = (
            name
            for name in dir(legacy_module)
            if not name.startswith("_") or name in {"__all__", "__version__", "__author__"}
        )

    for name in public_names:
        setattr(module, name, getattr(legacy_module, name))

    if hasattr(legacy_module, "__all__"):
        module.__all__ = list(legacy_module.__all__)  # type: ignore[attr-defined]
