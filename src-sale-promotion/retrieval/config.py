"""
Configuration for Sales Promotion Retrieval System

Centralized configuration for all retrieval strategies, models, caching, and performance settings.
"""

import os
from pathlib import Path
from typing import List, Optional, Union
from dotenv import load_dotenv

from src_sale_promotion.common import EMBEDDING_DIR as ROOT_EMBEDDING_ROOT

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.resolve()
LEGACY_EMBEDDING_ROOT = (BASE_DIR / "../data/embedding").resolve()

# Candidate roots searched for embeddings when no override is provided.
EMBEDDING_ROOT_CANDIDATES: List[Path] = [
    ROOT_EMBEDDING_ROOT,
    LEGACY_EMBEDDING_ROOT,
]

# Model Configuration
# Phase 5: Changed from gpt-5 to gpt-4o-mini for 10x speed + 73% cost reduction
# Rollback: Set LLM_MODEL=gpt-5 in .env
MODEL_CONFIG = {
    "llm_model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
    "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
    "embedding_dims": 3072,
    "llm_temperature": 0.0,
    "enable_streaming": os.getenv("ENABLE_STREAMING", "true").lower() == "true",
}

# Retrieval Configuration
RETRIEVAL_CONFIG = {
    "similarity_top_k": int(os.getenv("SIMILARITY_TOP_K", "5")),
    "similarity_cutoff": float(os.getenv("SIMILARITY_CUTOFF", "0.7")),
    "enable_reranking": False,  # Can add later
}

# Caching Configuration
CACHE_CONFIG = {
    "enable_caching": os.getenv("ENABLE_CACHING", "true").lower() == "true",
    "cache_max_size": int(os.getenv("CACHE_MAX_SIZE", "100")),
    "cache_ttl": int(os.getenv("CACHE_TTL", "3600")),  # 1 hour
}

# Metadata Index Configuration
METADATA_CONFIG = {
    "enable_fast_metadata_index": True,
    "indexed_fields": [
        # Customer fields
        "customer_type", "customer_tier", "is_wholesale", "is_retail",
        # Product fields
        "product_category", "product_brand", "product_size",
        # Promotion fields
        "discount_percentage", "buy_quantity", "promotion_type",
        # Date fields
        "is_active", "expires_soon", "duration_days",
        "start_date_gregorian", "end_date_gregorian"
    ],
    "enable_auto_filter_detection": True,
}

# Strategy Configuration
STRATEGY_CONFIG = {
    "vector": {
        "enabled": True,
        "top_k": 5,
        "similarity_cutoff": 0.7
    },
    "summary": {
        "enabled": True,
        "summary_top_k": 3,
        "detail_top_k": 2
    },
    "recursive": {
        "enabled": True,
        "top_level_top_k": 3,
        "detail_top_k": 2,
        "max_depth": 2
    },
    "metadata": {
        "enabled": True,
        "auto_filter_detection": True,
        "top_k": 5
    },
    "hybrid": {
        "enabled": True,
        "alpha": 0.5,  # 0.5 = equal weight vector + keyword
        "top_k": 5,
        "enable_thai_tokenization": True
    },
    "chunk_decoupling": {
        "enabled": False,  # Optional for atomic docs
        "top_k": 3
    },
    "planner": {
        "enabled": True,
        "max_sub_questions": 5,
        "enable_parallel": True,
        "max_depth": 3
    }
}

# Routing Configuration
ROUTING_CONFIG = {
    "enable_two_stage": True,
    "fallback_strategy": "vector",
    "min_confidence": 0.5,
    "enable_strategy_override": True,  # Allow manual strategy selection
}

# Complete configuration (embedding path resolved dynamically at runtime)
CONFIG = {
    "embedding_path": None,
    "models": MODEL_CONFIG,
    "retrieval": RETRIEVAL_CONFIG,
    "cache": CACHE_CONFIG,
    "metadata": METADATA_CONFIG,
    "strategies": STRATEGY_CONFIG,
    "routing": ROUTING_CONFIG,
}


def _list_embedding_dirs(base: Path) -> List[Path]:
    """Return timestamped embedding directories under the provided root."""
    if not base.exists():
        return []
    return sorted(
        (p.resolve() for p in base.glob("embeddings_sales_*") if p.is_dir()),
        reverse=True,
    )


def list_embedding_directories() -> List[Path]:
    """List all known embedding directories in priority order."""
    dirs: List[Path] = []
    for root in EMBEDDING_ROOT_CANDIDATES:
        dirs.extend(_list_embedding_dirs(root))
    # Remove duplicates while preserving order
    seen = set()
    unique_dirs = []
    for path in dirs:
        if path not in seen:
            seen.add(path)
            unique_dirs.append(path)
    return unique_dirs


def resolve_embedding_path(
    preferred: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Resolve the embedding directory to use for retrieval.

    Priority:
    1. Explicit `preferred` argument (CLI override)
    2. `EMBEDDING_DIR` environment variable
    3. Latest timestamped directory under known roots
    """
    if preferred:
        path = Path(preferred).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Embedding path override not found: {path}")
        return path

    env_path = os.getenv("EMBEDDING_DIR")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"EMBEDDING_DIR points to '{path}', but it does not exist."
            )
        return path

    candidates = list_embedding_directories()
    if not candidates:
        raise FileNotFoundError(
            "No embedding directories were found. "
            "Run the embedding pipeline first or set EMBEDDING_DIR explicitly."
        )

    return candidates[0]


def get_config() -> dict:
    """Get complete configuration dictionary."""
    return CONFIG


def get_strategy_config(strategy_name: str) -> dict:
    """Get configuration for specific strategy."""
    return STRATEGY_CONFIG.get(strategy_name, {})


def is_strategy_enabled(strategy_name: str) -> bool:
    """Check if a strategy is enabled."""
    strategy_cfg = STRATEGY_CONFIG.get(strategy_name, {})
    return strategy_cfg.get("enabled", False)
