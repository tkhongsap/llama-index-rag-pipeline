"""
CLI Utilities for iLand Retrieval System

Helper functions, imports, and utilities shared across CLI modules.
"""

import os
import sys
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import colorama
from colorama import Fore, Style, Back

# Initialize colorama for cross-platform color support
colorama.init(autoreset=True)

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def setup_imports():
    """Setup and validate all required imports."""
    imports = {}
    
    # Try to import retrieval components
    try:
        from .router import iLandRouterRetriever
        from .index_classifier import create_default_iland_classifier
        from .cache import iLandCacheManager
        from .parallel_executor import ParallelStrategyExecutor
        from .retrievers import (
            VectorRetrieverAdapter,
            SummaryRetrieverAdapter,
            RecursiveRetrieverAdapter,
            MetadataRetrieverAdapter,
            ChunkDecouplingRetrieverAdapter,
            HybridRetrieverAdapter,
            PlannerRetrieverAdapter
        )
        imports['retrieval'] = True
    except ImportError:
        # Fallback to absolute imports
        try:
            from retrieval.router import iLandRouterRetriever
            from retrieval.index_classifier import create_default_iland_classifier
            from retrieval.cache import iLandCacheManager
            from retrieval.parallel_executor import ParallelStrategyExecutor
            from retrieval.retrievers import (
                VectorRetrieverAdapter,
                SummaryRetrieverAdapter,
                RecursiveRetrieverAdapter,
                MetadataRetrieverAdapter,
                ChunkDecouplingRetrieverAdapter,
                HybridRetrieverAdapter,
                PlannerRetrieverAdapter
            )
            imports['retrieval'] = True
        except ImportError:
            print("Warning: Could not import retrieval components")
            imports['retrieval'] = False
    
    # Try to import embedding utilities
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from load_embedding import (
            load_latest_iland_embeddings,
            load_all_latest_iland_embeddings,
            get_iland_batch_summary
        )
        imports['embeddings'] = {
            'load_latest': load_latest_iland_embeddings,
            'load_all': load_all_latest_iland_embeddings,
            'batch_summary': get_iland_batch_summary
        }
    except ImportError:
        print("Warning: Could not import iLand embedding utilities")
        imports['embeddings'] = {
            'load_latest': None,
            'load_all': None,
            'batch_summary': None
        }
    
    # Try to import response synthesis
    try:
        from llama_index.core.response_synthesizers import ResponseMode
        from llama_index.core import get_response_synthesizer
        from llama_index.llms.openai import OpenAI
        imports['synthesis'] = {
            'ResponseMode': ResponseMode,
            'get_response_synthesizer': get_response_synthesizer,
            'OpenAI': OpenAI
        }
    except ImportError:
        print("Warning: Could not import response synthesis utilities")
        imports['synthesis'] = {
            'ResponseMode': None,
            'get_response_synthesizer': None,
            'OpenAI': None
        }
    
    return imports


def validate_api_key() -> Optional[str]:
    """Validate and return OpenAI API key."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in environment variables")
    return api_key


def print_colored_header(text: str, color: str = Fore.CYAN):
    """Print a colored header with separators."""
    print(f"\n{color}{text}{Style.RESET_ALL}")
    print(f"{color}{'-' * len(text)}{Style.RESET_ALL}")


def print_success(text: str):
    """Print success message in green."""
    print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")


def print_error(text: str):
    """Print error message in red."""
    print(f"{Fore.RED}❌ {text}{Style.RESET_ALL}")


def print_warning(text: str):
    """Print warning message in yellow."""
    print(f"{Fore.YELLOW}⚠️ {text}{Style.RESET_ALL}")


def format_execution_time(start_time: float) -> str:
    """Format execution time from start time."""
    return f"{time.time() - start_time:.2f}s"


def get_retrieval_components():
    """Get all retrieval component classes."""
    try:
        from .router import iLandRouterRetriever
        from .index_classifier import create_default_iland_classifier
        from .cache import iLandCacheManager
        from .parallel_executor import ParallelStrategyExecutor
        from .retrievers import (
            VectorRetrieverAdapter,
            SummaryRetrieverAdapter,
            RecursiveRetrieverAdapter,
            MetadataRetrieverAdapter,
            ChunkDecouplingRetrieverAdapter,
            HybridRetrieverAdapter,
            PlannerRetrieverAdapter
        )
        
        return {
            'router': iLandRouterRetriever,
            'classifier': create_default_iland_classifier,
            'cache': iLandCacheManager,
            'parallel': ParallelStrategyExecutor,
            'adapters': {
                'vector': VectorRetrieverAdapter,
                'summary': SummaryRetrieverAdapter,
                'recursive': RecursiveRetrieverAdapter,
                'metadata': MetadataRetrieverAdapter,
                'chunk_decoupling': ChunkDecouplingRetrieverAdapter,
                'hybrid': HybridRetrieverAdapter,
                'planner': PlannerRetrieverAdapter
            }
        }
    except ImportError:
        # Fallback to absolute imports
        from retrieval.router import iLandRouterRetriever
        from retrieval.index_classifier import create_default_iland_classifier
        from retrieval.cache import iLandCacheManager
        from retrieval.parallel_executor import ParallelStrategyExecutor
        from retrieval.retrievers import (
            VectorRetrieverAdapter,
            SummaryRetrieverAdapter,
            RecursiveRetrieverAdapter,
            MetadataRetrieverAdapter,
            ChunkDecouplingRetrieverAdapter,
            HybridRetrieverAdapter,
            PlannerRetrieverAdapter
        )
        
        return {
            'router': iLandRouterRetriever,
            'classifier': create_default_iland_classifier,
            'cache': iLandCacheManager,
            'parallel': ParallelStrategyExecutor,
            'adapters': {
                'vector': VectorRetrieverAdapter,
                'summary': SummaryRetrieverAdapter,
                'recursive': RecursiveRetrieverAdapter,
                'metadata': MetadataRetrieverAdapter,
                'chunk_decoupling': ChunkDecouplingRetrieverAdapter,
                'hybrid': HybridRetrieverAdapter,
                'planner': PlannerRetrieverAdapter
            }
        } 