"""
Common utilities and shared constants for sales promotion pipeline.
"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DOCS_DIR = DATA_DIR / "input_docs" / "sale-promotion"
EMBEDDING_DIR = DATA_DIR / "embedding" / "sale-promotion"
PROCESSED_DOCS_DIR = DATA_DIR / "processed" / "sale-promotion"

# Ensure directories exist
INPUT_DOCS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DOCS_DIR.mkdir(parents=True, exist_ok=True)
