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
import pickle
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

# Cache configuration
CACHE_DIR = Path(__file__).parent.parent.parent / ".cache"  # Project root/.cache
EMBEDDINGS_CACHE_DURATION = 7200  # 2 hours
ADAPTERS_CACHE_DURATION = 7200    # 2 hours  
ROUTER_CACHE_DURATION = 3600      # 1 hour

# Ensure cache directory exists
CACHE_DIR.mkdir(exist_ok=True)


class PersistentCache:
    """Persistent file-based cache system for embeddings, adapters, and router."""
    
    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.embeddings_file = self.cache_dir / "embeddings.pkl"
        self.adapters_file = self.cache_dir / "adapters.pkl"
        self.router_file = self.cache_dir / "router.pkl"
        
        # Timestamp files
        self.embeddings_timestamp_file = self.cache_dir / "embeddings_timestamp.txt"
        self.adapters_timestamp_file = self.cache_dir / "adapters_timestamp.txt"
        self.router_timestamp_file = self.cache_dir / "router_timestamp.txt"
    
    def _is_cache_valid(self, timestamp_file: Path, cache_duration: int) -> bool:
        """Check if cache is still valid based on timestamp."""
        if not timestamp_file.exists():
            return False
        
        try:
            with open(timestamp_file, 'r') as f:
                cache_timestamp = float(f.read().strip())
            
            current_time = time.time()
            return (current_time - cache_timestamp) < cache_duration
        except:
            return False
    
    def _save_timestamp(self, timestamp_file: Path):
        """Save current timestamp to file."""
        with open(timestamp_file, 'w') as f:
            f.write(str(time.time()))
    
    def _get_cache_age_info(self, timestamp_file: Path) -> Dict[str, Any]:
        """Get cache age information for display."""
        if not timestamp_file.exists():
            return {"exists": False}
        
        try:
            with open(timestamp_file, 'r') as f:
                cache_timestamp = float(f.read().strip())
            
            current_time = time.time()
            age_seconds = current_time - cache_timestamp
            age_hours = age_seconds / 3600
            
            return {
                "exists": True,
                "age_seconds": age_seconds,
                "age_hours": age_hours,
                "timestamp": cache_timestamp
            }
        except:
            return {"exists": False}
    
    def load_embeddings(self) -> Optional[List[Any]]:
        """Load embeddings from cache if valid, otherwise return None."""
        if (self._is_cache_valid(self.embeddings_timestamp_file, EMBEDDINGS_CACHE_DURATION) and 
            self.embeddings_file.exists()):
            
            try:
                start_time = time.time()
                with open(self.embeddings_file, 'rb') as f:
                    embeddings = pickle.load(f)
                load_time = time.time() - start_time
                
                age_info = self._get_cache_age_info(self.embeddings_timestamp_file)
                print(f"⚡ Using cached embeddings ({len(embeddings)} items, age: {age_info['age_hours']:.1f}h, load time: {load_time*1000:.0f}ms)")
                return embeddings
            except Exception as e:
                print(f"⚠️ Failed to load embeddings cache: {e}")
                return None
        
        return None
    
    def save_embeddings(self, embeddings: List[Any]):
        """Save embeddings to cache."""
        try:
            start_time = time.time()
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(embeddings, f)
            self._save_timestamp(self.embeddings_timestamp_file)
            save_time = time.time() - start_time
            
            print(f"💾 Cached {len(embeddings)} embeddings to disk (save time: {save_time*1000:.0f}ms)")
        except Exception as e:
            print(f"⚠️ Failed to save embeddings cache: {e}")
    
    def load_adapters(self) -> Optional[Dict[str, Any]]:
        """Load strategy adapters from cache if valid, otherwise return None."""
        if (self._is_cache_valid(self.adapters_timestamp_file, ADAPTERS_CACHE_DURATION) and 
            self.adapters_file.exists()):
            
            try:
                start_time = time.time()
                with open(self.adapters_file, 'rb') as f:
                    adapters = pickle.load(f)
                load_time = time.time() - start_time
                
                age_info = self._get_cache_age_info(self.adapters_timestamp_file)
                print(f"⚡ Using cached strategy adapters ({len(adapters)} adapters, age: {age_info['age_hours']:.1f}h, load time: {load_time*1000:.0f}ms)")
                return adapters
            except Exception as e:
                print(f"⚠️ Failed to load adapters cache: {e}")
                return None
        
        return None
    
    def save_adapters(self, adapters: Dict[str, Any]):
        """Save strategy adapters to cache."""
        try:
            start_time = time.time()
            with open(self.adapters_file, 'wb') as f:
                pickle.dump(adapters, f)
            self._save_timestamp(self.adapters_timestamp_file)
            save_time = time.time() - start_time
            
            print(f"💾 Cached {len(adapters)} strategy adapters to disk (save time: {save_time*1000:.0f}ms)")
        except Exception as e:
            print(f"⚠️ Failed to save adapters cache: {e}")
    
    def load_router(self) -> Optional[RouterRetriever]:
        """Load router from cache if valid, otherwise return None."""
        if (self._is_cache_valid(self.router_timestamp_file, ROUTER_CACHE_DURATION) and 
            self.router_file.exists()):
            
            try:
                start_time = time.time()
                with open(self.router_file, 'rb') as f:
                    router = pickle.load(f)
                load_time = time.time() - start_time
                
                age_info = self._get_cache_age_info(self.router_timestamp_file)
                print(f"⚡ Using cached router (age: {age_info['age_hours']:.1f}h, load time: {load_time*1000:.0f}ms)")
                return router
            except Exception as e:
                print(f"⚠️ Failed to load router cache: {e}")
                return None
        
        return None
    
    def save_router(self, router: RouterRetriever):
        """Save router to cache."""
        try:
            start_time = time.time()
            with open(self.router_file, 'wb') as f:
                pickle.dump(router, f)
            self._save_timestamp(self.router_timestamp_file)
            save_time = time.time() - start_time
            
            print(f"💾 Cached router to disk (save time: {save_time*1000:.0f}ms)")
        except Exception as e:
            print(f"⚠️ Failed to save router cache: {e}")
    
    def clear_all_caches(self):
        """Clear all cache files."""
        cache_files = [
            self.embeddings_file, self.embeddings_timestamp_file,
            self.adapters_file, self.adapters_timestamp_file,
            self.router_file, self.router_timestamp_file
        ]
        
        cleared_count = 0
        for cache_file in cache_files:
            if cache_file.exists():
                cache_file.unlink()
                cleared_count += 1
        
        print(f"🗑️ Cleared {cleared_count} cache files")
    
    def show_status(self):
        """Display current cache status."""
        print("\n🔍 Persistent Cache Status:")
        print("=" * 50)
        
        # Embeddings cache
        emb_info = self._get_cache_age_info(self.embeddings_timestamp_file)
        if emb_info["exists"]:
            remaining_hours = (EMBEDDINGS_CACHE_DURATION - emb_info["age_seconds"]) / 3600
            print(f"📊 Embeddings: Age {emb_info['age_hours']:.1f}h | Expires in {remaining_hours:.1f}h | Valid: {self._is_cache_valid(self.embeddings_timestamp_file, EMBEDDINGS_CACHE_DURATION)}")
        else:
            print("📊 Embeddings: Not cached")
        
        # Adapters cache
        adapt_info = self._get_cache_age_info(self.adapters_timestamp_file)
        if adapt_info["exists"]:
            remaining_hours = (ADAPTERS_CACHE_DURATION - adapt_info["age_seconds"]) / 3600
            print(f"🔧 Adapters: Age {adapt_info['age_hours']:.1f}h | Expires in {remaining_hours:.1f}h | Valid: {self._is_cache_valid(self.adapters_timestamp_file, ADAPTERS_CACHE_DURATION)}")
        else:
            print("🔧 Adapters: Not cached")
        
        # Router cache
        router_info = self._get_cache_age_info(self.router_timestamp_file)
        if router_info["exists"]:
            remaining_hours = (ROUTER_CACHE_DURATION - router_info["age_seconds"]) / 3600
            print(f"🎯 Router: Age {router_info['age_hours']:.1f}h | Expires in {remaining_hours:.1f}h | Valid: {self._is_cache_valid(self.router_timestamp_file, ROUTER_CACHE_DURATION)}")
        else:
            print("🎯 Router: Not cached")
        
        print("=" * 50)


# Global cache instance
cache = PersistentCache()


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


def load_embeddings_from_disk() -> Optional[List[Any]]:
    """Load embeddings from disk with timing."""
    print("💾 Loading embeddings from disk...")
    start_time = time.time()
    
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
        
        # Load all embeddings
        all_embeddings = loader.load_all_embeddings_from_batch(latest_batch)
        
        # Combine all embeddings
        full_data = []
        for sub_batch, emb_types in all_embeddings.items():
            for emb_type, embeddings in emb_types.items():
                full_data.extend(embeddings)
        
        load_time = time.time() - start_time
        
        if full_data:
            print(f"✅ Loaded {len(full_data)} embeddings from disk (load time: {load_time*1000:.0f}ms)")
            # Cache the embeddings
            cache.save_embeddings(full_data)
            
        return full_data
        
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        return None


def get_cached_embeddings() -> Optional[List[Any]]:
    """Get cached embeddings or load from disk if cache is invalid."""
    # Try to load from cache first
    embeddings = cache.load_embeddings()
    if embeddings is not None:
        return embeddings
    
    # Load from disk if cache miss
    return load_embeddings_from_disk()


def create_strategy_adapters_from_cache_or_disk(embeddings_data: List[Any], api_key: Optional[str] = None) -> Dict[str, Any]:
    """Create strategy adapters with timing."""
    print("🔧 Creating strategy adapters...")
    start_time = time.time()
    
    adapters = create_strategy_adapters_optimized(embeddings_data, api_key)
    
    creation_time = time.time() - start_time
    
    if adapters:
        print(f"✅ Created {len(adapters)} strategy adapters (creation time: {creation_time*1000:.0f}ms)")
        # Cache the adapters
        cache.save_adapters(adapters)
    
    return adapters


def get_cached_strategy_adapters(embeddings_data: List[Any], api_key: Optional[str] = None) -> Dict[str, Any]:
    """Get cached strategy adapters or create new ones if cache is invalid."""
    # Try to load from cache first
    adapters = cache.load_adapters()
    if adapters is not None:
        return adapters
    
    # Create new adapters if cache miss
    return create_strategy_adapters_from_cache_or_disk(embeddings_data, api_key)


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


def create_agentic_router_from_scratch(api_key: Optional[str] = None) -> RouterRetriever:
    """Create a full agentic router from scratch with timing."""
    print("🔄 Creating router from scratch...")
    start_time = time.time()
    
    try:
        # Step 1: Get embeddings (cached or from disk)
        full_data = get_cached_embeddings()
        if not full_data:
            print("⚠️  No embedding data found.")
            return None
            
        print(f"📊 Using {len(full_data)} embeddings")
        
        # Step 2: Get strategy adapters (cached or create new)
        strategy_adapters = get_cached_strategy_adapters(full_data, api_key)
        if not strategy_adapters:
            print("⚠️  No strategy adapters created.")
            return None
        
        # Step 3: Create retrievers for each index (this is fast)
        retrievers = {}
        for index_name, index_description in DEFAULT_INDICES.items():
            retrievers[index_name] = {}
            
            # Reference the same adapter instances (no copying needed)
            for strategy_name, adapter in strategy_adapters.items():
                retrievers[index_name][strategy_name] = adapter
            
            print(f"📋 Index '{index_name}': {len(retrievers[index_name])} strategies")
        
        # Step 4: Create index classifier (lightweight)
        index_classifier = create_default_classifier(api_key)
        
        # Step 5: Create router (lightweight)
        router = RouterRetriever(
            retrievers=retrievers,
            index_classifier=index_classifier,
            strategy_selector="llm",
            api_key=api_key
        )
        
        creation_time = time.time() - start_time
        print(f"✅ Router created from scratch (creation time: {creation_time*1000:.0f}ms)")
        
        # Cache the router
        cache.save_router(router)
        
        return router
        
    except Exception as e:
        print(f"❌ Error creating router: {e}")
        return None


def get_cached_router(api_key: Optional[str] = None, force_refresh: bool = False) -> RouterRetriever:
    """Get cached router or create new one if cache is invalid."""
    if not force_refresh:
        # Try to load from cache first
        router = cache.load_router()
        if router is not None:
            return router
    
    # Create new router if cache miss or force refresh
    return create_agentic_router_from_scratch(api_key)


def show_cache_status():
    """Display current cache status for debugging and monitoring."""
    cache.show_status()


def create_agentic_router(api_key: Optional[str] = None) -> RouterRetriever:
    """Create a full agentic router (wrapper for new cache system)."""
    return get_cached_router(api_key)


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
    fast_mode: bool = False,
    show_performance: bool = False
) -> dict:
    """
    Query the agentic retriever and return results.
    
    Args:
        query: The user query
        top_k: Number of results to retrieve
        api_key: OpenAI API key
        fast_mode: If True, prioritize speed over completeness
        show_performance: If True, show detailed performance breakdown
        
    Returns:
        Dict with response and metadata
    """
    overall_start_time = time.time()
    
    # Performance tracking
    performance_stages = {}
    
    # Setup models (only once per session)
    if not hasattr(Settings, 'llm') or Settings.llm is None:
        setup_start = time.time()
        setup_models(api_key)
        performance_stages['setup_models'] = (time.time() - setup_start) * 1000
    
    # Create router (try cached first for performance)
    router_start = time.time()
    router = get_cached_router(api_key)
    performance_stages['router_initialization'] = (time.time() - router_start) * 1000
    
    if not router:
        print("🔄 Falling back to simple router...")
        fallback_start = time.time()
        router = create_simple_router(api_key)
        performance_stages['fallback_router'] = (time.time() - fallback_start) * 1000
        
    if not router:
        total_time = (time.time() - overall_start_time) * 1000
        return {
            "error": "Could not initialize any router",
            "response": None,
            "metadata": {
                "query": query,
                "total_time_ms": round(total_time, 2),
                "performance_stages": performance_stages
            }
        }
    
    try:
        # Create query engine with response synthesizer
        query_engine_start = time.time()
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
            use_async=False
        )
        
        query_engine = RetrieverQueryEngine(
            retriever=router,
            response_synthesizer=response_synthesizer
        )
        performance_stages['query_engine_creation'] = (time.time() - query_engine_start) * 1000
        
        # Execute query with timing
        query_start = time.time()
        response = query_engine.query(query)
        query_time = (time.time() - query_start) * 1000
        performance_stages['query_execution'] = query_time
        
        total_time = (time.time() - overall_start_time) * 1000
        
        # Extract metadata from retrieved nodes and router
        metadata = {
            "query": query,
            "top_k": top_k,
            "total_time_ms": round(total_time, 2),
            "query_time_ms": round(query_time, 2),
            "performance_stages": performance_stages
        }
        
        # Determine if this was a cold start or warm cache
        cache_type = "warm" if performance_stages['router_initialization'] < 100 else "cold"
        metadata["cache_type"] = cache_type
        
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
        
        # Show performance summary if requested
        if show_performance:
            print_performance_summary(metadata)
        
        # Log the retrieval call
        try:
            log_retrieval_call(
                query=query,
                selected_index=metadata.get("index", "unknown"),
                selected_strategy=metadata.get("strategy", "unknown"),
                latency_ms=metadata.get("total_time_ms", 0),
                confidence=metadata.get("index_confidence"),
                error=None
            )
        except Exception as log_error:
            print(f"⚠️ Failed to log retrieval call: {log_error}")
        
        return {
            "response": str(response),
            "metadata": metadata,
            "error": None
        }
        
    except Exception as e:
        error_time = round((time.time() - overall_start_time) * 1000, 2)
        print(f"❌ Query execution error: {e}")
        
        # Log the failed call
        try:
            log_retrieval_call(
                query=query,
                selected_index="error",
                selected_strategy="error",
                latency_ms=error_time,
                error=str(e)
            )
        except Exception as log_error:
            print(f"⚠️ Failed to log error: {log_error}")
        
        return {
            "error": str(e),
            "response": None,
            "metadata": {
                "query": query,
                "total_time_ms": error_time,
                "index": "error",
                "strategy": "error",
                "performance_stages": performance_stages
            }
        }


def print_performance_summary(metadata: dict):
    """Print a detailed performance summary."""
    cache_type = metadata.get("cache_type", "unknown")
    total_time = metadata.get("total_time_ms", 0)
    stages = metadata.get("performance_stages", {})
    
    print(f"\n⚡ Performance Summary ({cache_type.upper()} CACHE):")
    print("=" * 45)
    
    if cache_type == "cold":
        print("🔥 Cold Start - Initial Load Times:")
        if 'setup_models' in stages:
            print(f"   Model Setup: {stages['setup_models']:.0f}ms")
        print(f"   Router Init: {stages.get('router_initialization', 0):.0f}ms")
    else:
        print("⚡ Warm Cache - Lightning Fast:")
        print(f"   Router Init: {stages.get('router_initialization', 0):.0f}ms (cached)")
    
    if 'query_engine_creation' in stages:
        print(f"   Query Engine: {stages['query_engine_creation']:.0f}ms")
    
    print(f"   Query Execution: {stages.get('query_execution', 0):.0f}ms")
    print(f"   TOTAL: {total_time:.0f}ms")
    
    # Performance improvement message
    if cache_type == "cold":
        print("\n💡 Next query will be ~10-20x faster due to caching!")
    else:
        print("\n🚀 Optimized performance - cache hit!")
    
    print("=" * 45)


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
  python -m agentic_retriever.cli --cache-status
        """
    )
    
    parser.add_argument(
        "-q", "--query",
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
    
    parser.add_argument(
        "--cache-status",
        action="store_true",
        help="Show current cache status and exit"
    )
    
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh of all caches"
    )
    
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all caches and exit"
    )
    
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Show detailed performance breakdown"
    )
    
    args = parser.parse_args()
    
    # Handle clear cache command
    if args.clear_cache:
        cache.clear_all_caches()
        print("✅ All caches cleared successfully")
        return
    
    # Handle cache status command
    if args.cache_status:
        show_cache_status()
        return
    
    # Query is required unless showing cache status or clearing cache
    if not args.query:
        parser.error("Query (-q/--query) is required unless using --cache-status or --clear-cache")
    
    if args.verbose:
        print(f"🔍 Processing query: {args.query}")
        print(f"📊 Top K: {args.top_k}")
        if args.force_refresh:
            print("🔄 Force refresh enabled - will rebuild all caches")
        if args.performance:
            print("📈 Performance tracking enabled")
    
    # Force refresh caches if requested
    if args.force_refresh:
        cache.clear_all_caches()
        print("🔄 All caches cleared")
    
    # Execute query
    result = query_agentic_retriever(
        query=args.query,
        top_k=args.top_k,
        api_key=args.api_key,
        show_performance=args.performance or args.verbose
    )
    
    # Format and display output
    format_output(result)
    
    # Show cache status if verbose
    if args.verbose:
        show_cache_status()


if __name__ == "__main__":
    main() 