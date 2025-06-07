"""
load_embedding - iLand embedding loading and index reconstruction module

This module provides utilities to load batch-processed iLand Thai land deed embeddings 
from disk and reconstruct various LlamaIndex indices for retrieval.

Main Components:
- EmbeddingConfig: Configuration for embedding loading
- FilterConfig: Configuration for filtering embeddings
- iLandEmbeddingLoader: Load embeddings from iLand processing pipeline
- iLandIndexReconstructor: Reconstruct LlamaIndex indices from embeddings
- Validation functions: Validate and analyze embeddings
- Utility functions: Convenient access to common operations
"""

# Import main classes and functions for public API
from .models import (
    EmbeddingConfig,
    FilterConfig,
    LoadingResult,
    EMBEDDING_DIR,
    THAI_PROVINCES,
    EMBEDDING_TYPES
)

from .embedding_loader import iLandEmbeddingLoader

from .index_reconstructor import iLandIndexReconstructor

from .validation import (
    validate_iland_embeddings,
    validate_embedding_consistency,
    analyze_thai_metadata_distribution,
    generate_validation_report
)

from .utils import (
    load_latest_iland_embeddings,
    load_all_latest_iland_embeddings,
    create_iland_index_from_latest_batch,
    get_iland_batch_summary
)

from .demo import demonstrate_iland_loading

# Version information
__version__ = "1.0.0"
__author__ = "iLand Team"

# Public API
__all__ = [
    # Configuration classes
    "EmbeddingConfig",
    "FilterConfig", 
    "LoadingResult",
    
    # Main classes
    "iLandEmbeddingLoader",
    "iLandIndexReconstructor",
    
    # Validation functions
    "validate_iland_embeddings",
    "validate_embedding_consistency", 
    "analyze_thai_metadata_distribution",
    "generate_validation_report",
    
    # Utility functions
    "load_latest_iland_embeddings",
    "load_all_latest_iland_embeddings",
    "create_iland_index_from_latest_batch",
    "get_iland_batch_summary",
    
    # Demo function
    "demonstrate_iland_loading",
    
    # Constants
    "EMBEDDING_DIR",
    "THAI_PROVINCES",
    "EMBEDDING_TYPES"
] 