"""
Common utilities and shared constants for sales promotion pipeline.
"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"

# Input CSVs live with the package data directory
INPUT_DOCS_DIR = PROJECT_ROOT / "src-sale-promotion" / "data"

# Embeddings root (top-level, without extra subdirectory) to match tests/retrieval
EMBEDDING_DIR = DATA_DIR / "embedding"

# Processed documents are stored under the package's data/processed
PROCESSED_DOCS_DIR = PROJECT_ROOT / "src-sale-promotion" / "data" / "processed"

# Ensure directories exist
INPUT_DOCS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DOCS_DIR.mkdir(parents=True, exist_ok=True)
