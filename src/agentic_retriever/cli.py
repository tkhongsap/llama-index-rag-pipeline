"""
Agentic Retriever CLI

Command-line interface for the intelligent retrieval system that automatically
selects the best retrieval strategy and index for each query.
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

from .index_classifier import create_default_classifier, DEFAULT_INDICES
from .router import RouterRetriever
from .log_utils import log_retrieval_call

# Import all retrieval strategy adapters
from .retrievers import (
    VectorRetrieverAdapter,
    SummaryRetrieverAdapter,
    RecursiveRetrieverAdapter,
    MetadataRetrieverAdapter,
    ChunkDecouplingRetrieverAdapter,
    HybridRetrieverAdapter,
    PlannerRetrieverAdapter
)

# Import from parent src directory
try:
    from ..load_embeddings import EmbeddingLoader
except ImportError:
    # Fallback for when module is run directly
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from load_embeddings import EmbeddingLoader

# Load environment variables
load_dotenv(override=True)

# Global cache for router to avoid rebuilding indices
_cached_router = None
_cache_timestamp = None
_cache_duration = 3600  # 1 hour cache


def setup_models(api_key: Optional[str] = None):
    """Setup LLM and embedding models."""
    Settings.llm = OpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key or os.getenv("OPENAI_API_KEY")
    )
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=api_key or os.getenv("OPENAI_API_KEY")
    )


def create_strategy_adapters_optimized(embeddings_data: Any, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Create all retrieval strategy adapters with optimizations.
    
    Args:
        embeddings_data: Loaded embeddings data for adapters
        api_key: OpenAI API key
        
    Returns:
        Dict mapping strategy names to adapter instances
    """
    adapters = {}
    
    try:
        print("🚀 Creating optimized strategy adapters...")
        
        # Only create essential adapters to start with
        essential_adapters = ["vector", "metadata", "summary"]
        
        for adapter_name in essential_adapters:
            try:
                if adapter_name == "vector":
                    adapters["vector"] = VectorRetrieverAdapter.from_embeddings(embeddings_data, api_key)
                elif adapter_name == "metadata":
                    adapters["metadata"] = MetadataRetrieverAdapter.from_embeddings(embeddings_data, api_key)
                elif adapter_name == "summary":
                    adapters["summary"] = SummaryRetrieverAdapter.from_embeddings(embeddings_data, api_key)
                
                print(f"✅ Created {adapter_name} adapter")
            except Exception as e:
                print(f"⚠️  Failed to create {adapter_name} adapter: {e}")
                continue
        
        # Only create additional adapters if essential ones work
        if len(adapters) >= 2:
            additional_adapters = {
                "recursive": RecursiveRetrieverAdapter,
                "chunk_decoupling": ChunkDecouplingRetrieverAdapter,
                "hybrid": HybridRetrieverAdapter,
                "planner": PlannerRetrieverAdapter
            }
            
            for adapter_name, adapter_class in additional_adapters.items():
                try:
                    adapters[adapter_name] = adapter_class.from_embeddings(embeddings_data, api_key)
                    print(f"✅ Created {adapter_name} adapter")
                except Exception as e:
                    print(f"⚠️  Failed to create {adapter_name} adapter: {e}")
                    continue
        
        print(f"✅ Created {len(adapters)} strategy adapters: {list(adapters.keys())}")
        
    except Exception as e:
        print(f"⚠️  Error creating strategy adapters: {e}")
        # Fallback to just vector adapter
        try:
            adapters["vector"] = VectorRetrieverAdapter.from_embeddings(embeddings_data, api_key)
            print("✅ Fallback: Created vector adapter only")
        except Exception as fallback_error:
            print(f"❌ Failed to create even vector adapter: {fallback_error}")
    
    return adapters


def get_cached_router(api_key: Optional[str] = None, force_refresh: bool = False) -> RouterRetriever:
    """Get cached router or create new one if cache is invalid."""
    global _cached_router, _cache_timestamp
    
    current_time = time.time()
    
    # Check if cache is valid
    if (not force_refresh and 
        _cached_router is not None and 
        _cache_timestamp is not None and 
        (current_time - _cache_timestamp) < _cache_duration):
        print("🚀 Using cached router (performance optimization)")
        return _cached_router
    
    # Create new router
    print("🔄 Creating new router...")
    router = create_agentic_router(api_key)
    
    if router:
        _cached_router = router
        _cache_timestamp = current_time
        print("✅ Router cached for future queries")
    
    return router


def create_agentic_router(api_key: Optional[str] = None) -> RouterRetriever:
    """Create a full agentic router with multiple indices and strategies."""
    try:
        # Find embedding directory
        possible_paths = [
            Path("data/embedding"),     # From project root
            Path("../data/embedding"),  # From src directory
            Path("./data/embedding")    # Current directory
        ]
        
        embedding_dir = None
        for path in possible_paths:
            if path.exists():
                embedding_dir = path
                break
        
        if not embedding_dir:
            print("⚠️  No embedding directory found. Please run the embedding pipeline first.")
            return None
            
        loader = EmbeddingLoader(embedding_dir)
        latest_batch = loader.get_latest_batch()
        
        if not latest_batch:
            print("⚠️  No embedding batches found. Please run the embedding pipeline first.")
            return None
        
        # Load all embedding types from the latest batch
        all_embeddings = loader.load_all_embeddings_from_batch(latest_batch)
        
        # Combine all embeddings from all sub-batches
        full_data = []
        for sub_batch, emb_types in all_embeddings.items():
            for emb_type, embeddings in emb_types.items():
                full_data.extend(embeddings)
        
        if not full_data:
            print("⚠️  No embedding data found in latest batch.")
            return None
            
        print(f"📊 Loaded {len(full_data)} total embeddings from {len(all_embeddings)} sub-batches")
        
        # Create strategy adapters
        strategy_adapters = create_strategy_adapters_optimized(full_data, api_key)
        
        if not strategy_adapters:
            print("⚠️  No strategy adapters created.")
            return None
        
        # Create retrievers for each index with all available strategies
        retrievers = {}
        
        for index_name, index_description in DEFAULT_INDICES.items():
            retrievers[index_name] = {}
            
            # Add all available strategy adapters for each index
            for strategy_name, adapter in strategy_adapters.items():
                try:
                    # Create a copy/instance for this specific index
                    retrievers[index_name][strategy_name] = adapter
                except Exception as e:
                    print(f"⚠️  Failed to add {strategy_name} strategy for {index_name}: {e}")
                    continue
            
            print(f"📋 Index '{index_name}': {len(retrievers[index_name])} strategies available")
        
        # Create index classifier
        index_classifier = create_default_classifier(api_key)
        
        # Create router with all retrievers
        router = RouterRetriever(
            retrievers=retrievers,
            index_classifier=index_classifier,
            strategy_selector="llm",
            api_key=api_key
        )
        
        print(f"🎯 Router created with {len(retrievers)} indices and {len(strategy_adapters)} strategies")
        return router
        
    except Exception as e:
        print(f"❌ Error creating agentic router: {e}")
        return None


def create_simple_router(api_key: Optional[str] = None) -> RouterRetriever:
    """Create a simple router with vector strategy only (fallback)."""
    try:
        # Find embedding directory
        possible_paths = [
            Path("data/embedding"),
            Path("../data/embedding"),
            Path("./data/embedding")
        ]
        
        embedding_dir = None
        for path in possible_paths:
            if path.exists():
                embedding_dir = path
                break
        
        if not embedding_dir:
            print("⚠️  No embedding directory found.")
            return None
            
        loader = EmbeddingLoader(embedding_dir)
        latest_batch = loader.get_latest_batch()
        
        if not latest_batch:
            print("⚠️  No embedding batches found.")
            return None
        
        full_data, _, _ = loader.load_embeddings_from_files(
            latest_batch, "batch_1", "chunks"
        )
        
        if not full_data:
            print("⚠️  No embedding data found.")
            return None
        
        # Create simple router with just vector strategy
        retrievers = {
            "general_docs": {
                "vector": VectorRetrieverAdapter.from_embeddings(full_data, api_key)
            }
        }
        
        # Simple classifier with just one index
        from .index_classifier import IndexClassifier
        simple_indices = {"general_docs": "General document collection"}
        index_classifier = IndexClassifier(simple_indices, api_key, "llm")
        
        router = RouterRetriever(
            retrievers=retrievers,
            index_classifier=index_classifier,
            strategy_selector="llm",
            api_key=api_key
        )
        
        print("✅ Simple router created with vector strategy")
        return router
        
    except Exception as e:
        print(f"❌ Error creating simple router: {e}")
        return None


def query_agentic_retriever(
    query: str,
    top_k: int = 5,
    api_key: Optional[str] = None,
    fast_mode: bool = False
) -> dict:
    """
    Query the agentic retriever and return results.
    
    Args:
        query: The user query
        top_k: Number of results to retrieve
        api_key: OpenAI API key
        fast_mode: If True, prioritize speed over completeness
        
    Returns:
        Dict with response and metadata
    """
    start_time = time.time()
    
    # Setup models (only once per session)
    if not hasattr(Settings, 'llm') or Settings.llm is None:
        setup_models(api_key)
    
    # Create router (try cached first for performance)
    router = get_cached_router(api_key)
    if not router:
        print("🔄 Falling back to simple router...")
        router = create_simple_router(api_key)
        
    if not router:
        return {
            "error": "Could not initialize any router",
            "response": None,
            "metadata": {
                "query": query,
                "total_time_ms": round((time.time() - start_time) * 1000, 2)
            }
        }
    
    try:
        # Create query engine with response synthesizer
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
            use_async=False
        )
        
        query_engine = RetrieverQueryEngine(
            retriever=router,
            response_synthesizer=response_synthesizer
        )
        
        # Execute query with timing
        query_start = time.time()
        response = query_engine.query(query)
        query_time = time.time() - query_start
        
        # Extract metadata from retrieved nodes and router
        metadata = {
            "query": query,
            "top_k": top_k,
            "total_time_ms": round((time.time() - start_time) * 1000, 2),
            "query_time_ms": round(query_time * 1000, 2)
        }
        
        # Get routing information from router's last decision
        if hasattr(router, 'last_routing_info'):
            routing_info = router.last_routing_info
            metadata.update({
                "index": routing_info.get('selected_index', 'unknown'),
                "strategy": routing_info.get('selected_strategy', 'unknown'),
                "index_confidence": routing_info.get('index_confidence'),
                "strategy_confidence": routing_info.get('strategy_confidence')
            })
        
        # Get information from source nodes
        if hasattr(response, 'source_nodes') and response.source_nodes:
            metadata["num_sources"] = len(response.source_nodes)
            
            # Try to get strategy from first node metadata
            first_node = response.source_nodes[0]
            if hasattr(first_node, 'node') and hasattr(first_node.node, 'metadata'):
                node_metadata = first_node.node.metadata
                if not metadata.get("strategy"):
                    metadata["strategy"] = node_metadata.get('strategy', 'unknown')
                if not metadata.get("index"):
                    metadata["index"] = node_metadata.get('index', 'unknown')
        
        # Set defaults if still missing
        metadata.setdefault("index", "unknown")
        metadata.setdefault("strategy", "unknown")
        metadata.setdefault("num_sources", 0)
        
        return {
            "response": str(response),
            "metadata": metadata,
            "error": None
        }
        
    except Exception as e:
        error_time = round((time.time() - start_time) * 1000, 2)
        print(f"❌ Query execution error: {e}")
        
        return {
            "error": str(e),
            "response": None,
            "metadata": {
                "query": query,
                "total_time_ms": error_time,
                "index": "error",
                "strategy": "error"
            }
        }


def format_output(result: dict):
    """Format the output for display according to PRD requirements."""
    if result["error"]:
        print(f"❌ Error: {result['error']}")
        return
    
    # Print the response
    print("\n📝 Response:")
    print("=" * 60)
    print(result["response"])
    print("=" * 60)
    
    # Print routing information in PRD format
    metadata = result["metadata"]
    
    # Main routing info line as specified in PRD: "index = finance_docs | strategy = summary | latency = 320 ms"
    index = metadata.get("index", "unknown")
    strategy = metadata.get("strategy", "unknown")
    latency = metadata.get("total_time_ms", 0)
    
    print(f"\nindex = {index} | strategy = {strategy} | latency = {latency} ms")
    
    # Additional details if available and verbose mode
    additional_info = []
    
    if metadata.get("num_sources"):
        additional_info.append(f"sources = {metadata['num_sources']}")
    
    if metadata.get("index_confidence") is not None:
        additional_info.append(f"index_confidence = {metadata['index_confidence']:.3f}")
    
    if metadata.get("strategy_confidence") is not None:
        additional_info.append(f"strategy_confidence = {metadata['strategy_confidence']:.3f}")
    
    if metadata.get("query_time_ms"):
        additional_info.append(f"query_time = {metadata['query_time_ms']} ms")
    
    if additional_info:
        print("Additional details: " + " | ".join(additional_info))


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Agentic Retrieval CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m agentic_retriever.cli -q "What are the main topics?"
  python -m agentic_retriever.cli -q "Summarize Q4 revenue growth" --top_k 10
        """
    )
    
    parser.add_argument(
        "-q", "--query",
        required=True,
        help="The query to process"
    )
    
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of results to retrieve (default: 5)"
    )
    
    parser.add_argument(
        "--api_key",
        help="OpenAI API key (uses environment variable if not provided)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"🔍 Processing query: {args.query}")
        print(f"📊 Top K: {args.top_k}")
    
    # Execute query
    result = query_agentic_retriever(
        query=args.query,
        top_k=args.top_k,
        api_key=args.api_key
    )
    
    # Format and display output
    format_output(result)


if __name__ == "__main__":
    main() 