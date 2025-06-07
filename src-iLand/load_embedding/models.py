"""
models.py - Configuration models and constants for iLand embedding loading

This module contains configuration classes, constants, and data models
used throughout the iLand embedding loading pipeline.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# ---------- CONFIGURATION CONSTANTS -----------------------------------------

# Paths - Use absolute path relative to project root
# Get the project root directory (go up from src-iLand/load_embedding/)
_current_file = Path(__file__)
_project_root = _current_file.parent.parent.parent
EMBEDDING_DIR = _project_root / "data" / "embedding"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-4o-mini"

# iLand specific constants
ILAND_BATCH_PREFIX = "embeddings_iland_"
# Thai provinces list - fallback if common module not available
try:
    from ..common.thai_provinces import THAI_PROVINCES
except ImportError:
    # Fallback Thai provinces list
    THAI_PROVINCES = [
        "กรุงเทพมหานคร", "เชียงใหม่", "เชียงราย", "กาญจนบุรี", "ระยอง", 
        "ชลบุรี", "ภูเก็ต", "สุราษฎร์ธานี", "นครราชสีมา", "อุบลราชธานี"
    ]
# Standard embedding types
EMBEDDING_TYPES = ["chunks", "indexnodes", "summaries"]

# ---------- CONFIGURATION CLASSES -------------------------------------------

class EmbeddingConfig:
    """Configuration class for embedding loading settings."""
    
    def __init__(
        self,
        embedding_dir: Path = EMBEDDING_DIR,
        embed_model: str = DEFAULT_EMBED_MODEL,
        llm_model: str = DEFAULT_LLM_MODEL,
        api_key: Optional[str] = None
    ):
        """Initialize embedding configuration."""
        self.embedding_dir = embedding_dir
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        if not self.embedding_dir.exists():
            raise ValueError(f"Embedding directory not found: {self.embedding_dir}")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "embedding_dir": str(self.embedding_dir),
            "embed_model": self.embed_model,
            "llm_model": self.llm_model,
            "has_api_key": bool(self.api_key)
        }

class FilterConfig:
    """Configuration class for embedding filtering options."""
    
    def __init__(
        self,
        provinces: Optional[list] = None,
        deed_types: Optional[list] = None,
        min_area_rai: Optional[float] = None,
        max_area_rai: Optional[float] = None,
        max_embeddings: Optional[int] = None
    ):
        """Initialize filter configuration."""
        self.provinces = provinces or []
        self.deed_types = deed_types or []
        self.min_area_rai = min_area_rai
        self.max_area_rai = max_area_rai
        self.max_embeddings = max_embeddings
    
    def has_filters(self) -> bool:
        """Check if any filters are configured."""
        return bool(
            self.provinces or 
            self.deed_types or 
            self.min_area_rai is not None or 
            self.max_area_rai is not None
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert filter configuration to dictionary."""
        return {
            "provinces": self.provinces,
            "deed_types": self.deed_types,
            "min_area_rai": self.min_area_rai,
            "max_area_rai": self.max_area_rai,
            "max_embeddings": self.max_embeddings,
            "has_filters": self.has_filters()
        }

class LoadingResult:
    """Data class for embedding loading results."""
    
    def __init__(
        self,
        embeddings: list,
        batch_path: Path,
        embedding_type: str,
        count: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize loading result."""
        self.embeddings = embeddings
        self.batch_path = batch_path
        self.embedding_type = embedding_type
        self.count = count
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "batch_name": self.batch_path.name,
            "embedding_type": self.embedding_type,
            "count": self.count,
            "metadata": self.metadata
        } 