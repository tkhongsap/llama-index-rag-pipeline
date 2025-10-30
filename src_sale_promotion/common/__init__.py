"""
Common path constants for the sales promotion pipeline (underscore package).

This bridges code that imports `src_sale_promotion.common` to the actual
locations used in this repository under `src-sale-promotion/` and top-level
`data/` directories.
"""

from pathlib import Path


# Repository root
REPO_ROOT = Path(__file__).resolve().parents[2]

# Top-level data directory
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Input/processed live under the hyphenated package directory
PROJECT_DIR = REPO_ROOT / "src-sale-promotion"

# Input CSVs (default to the packaged data directory)
INPUT_DOCS_DIR = PROJECT_DIR / "data"
INPUT_DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Processed documents (JSONL + markdown)
# Align with existing outputs at src-sale-promotion/data/processed
PROCESSED_DOCS_DIR = PROJECT_DIR / "data" / "processed"
PROCESSED_DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Embeddings root (top-level to match tests and retrieval config expectations)
# Embedding runs will create timestamped subdirectories here:
#   data/embedding/embeddings_sales_YYYYMMDD_HHMMSS
EMBEDDING_DIR = DATA_DIR / "embedding"
EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)


__all__ = [
    "REPO_ROOT",
    "DATA_DIR",
    "PROJECT_DIR",
    "INPUT_DOCS_DIR",
    "PROCESSED_DOCS_DIR",
    "EMBEDDING_DIR",
]
