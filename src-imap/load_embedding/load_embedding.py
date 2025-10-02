"""
load_embedding.py - Backward compatibility module for iLand embedding loading

This module maintains backward compatibility by re-exporting all functionality
from the new modular structure. The original 790-line file has been refactored
into focused modules following coding rules (each under 300 lines).

New modular structure:
- models.py: Configuration classes and constants
- embedding_loader.py: iLandEmbeddingLoader class
- index_reconstructor.py: iLandIndexReconstructor class  
- validation.py: Validation and analysis functions
- utils.py: Utility functions for common operations
- demo.py: Demonstration functions
"""

# Re-export everything from the modular structure for backward compatibility
from .models import *
from .embedding_loader import *
from .index_reconstructor import *
from .validation import *
from .utils import *
from .load_embedding_complete import *

# Maintain the original demonstration function name
demonstrate_iland_loading = demonstrate_iland_loading

# ---------- ENTRY POINT (BACKWARD COMPATIBILITY) ----------------------------

if __name__ == "__main__":
    demonstrate_iland_loading()
