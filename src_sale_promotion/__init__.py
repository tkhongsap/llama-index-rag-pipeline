"""
Compatibility shim for importing the `src-sale-promotion` package with the
`src_sale_promotion` module name.

Python identifiers cannot contain hyphens, so this package exposes the existing
codebase (stored under `src-sale-promotion/`) through a hyphen-free module path.
"""

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LEGACY_DIR = _PROJECT_ROOT / "src-sale-promotion"

if not _LEGACY_DIR.exists():
    raise ImportError(
        f"Expected legacy package directory at {_LEGACY_DIR}, but it was not found."
    )

# Make Python treat `src-sale-promotion` as the search path for this package's
# submodules (e.g., `src_sale_promotion.data_processing`).
__path__ = [str(_LEGACY_DIR)]

# Execute the legacy package's __init__ so metadata like __version__ is preserved.
_legacy_init = _LEGACY_DIR / "__init__.py"
if _legacy_init.exists():
    exec(_legacy_init.read_text(encoding="utf-8"), globals(), globals())
