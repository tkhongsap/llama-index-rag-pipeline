"""
Agentic Retriever CLI

Command-line interface for the agentic retrieval system.
Usage: python -m agentic_retriever.cli -q "your question here"
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from llama_index.core import Settings, QueryBundle
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from .router import RouterRetriever, create_default_router
from .index_classifier import create_default_classifier
from .retrievers.vector import VectorRetrieverAdapter
from load_embeddings import EmbeddingLoader


def setup_models(api_key: Optional[str] = None):
    """Setup LLM and embedding models."""
    Settings.llm = OpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key
    )
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=api_key
    )


def create_simple_router(api_key: Optional[str] = None) -> RouterRetriever:
    """Create a simple router with available data."""
    try:
        # Try to load embeddings from the latest batch
        from pathlib import Path
        # Try different possible paths for embedding directory
        possible_paths = [
            Path("../data/embedding"),  # From src directory
            Path("data/embedding"),     # From project root
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
        
        if latest_batch:
            # Load chunk embeddings from the latest batch
            try:
                full_data, _, _ = loader.load_embeddings_from_files(
                    latest_batch, "batch_1", "chunks"
                )
                
                if full_data:
                    # Create a simple router with vector retrieval
                    retrievers = {
                        "general_docs": {
                            "vector": VectorRetrieverAdapter.from_embeddings(full_data, api_key)
                        }
                    }
                    
                    return RouterRetriever.from_retrievers(
                        retrievers=retrievers,
                        api_key=api_key,
                        strategy_selector="llm"
                    )
                else:
                    print("⚠️  No embedding data found in latest batch.")
                    return None
                    
            except Exception as e:
                print(f"⚠️  Error loading embeddings from batch: {e}")
                return None
        else:
            print("⚠️  No embedding batches found. Please run the embedding pipeline first.")
            return None
            
    except Exception as e:
        print(f"⚠️  Error loading embeddings: {e}")
        print("Please ensure the embedding pipeline has been run.")
        return None


def query_agentic_retriever(
    query: str,
    top_k: int = 5,
    api_key: Optional[str] = None
) -> dict:
    """
    Query the agentic retriever and return results.
    
    Args:
        query: The user query
        top_k: Number of results to retrieve
        api_key: OpenAI API key
        
    Returns:
        Dict with response and metadata
    """
    start_time = time.time()
    
    # Setup models
    setup_models(api_key)
    
    # Create router
    router = create_simple_router(api_key)
    if not router:
        return {
            "error": "Could not initialize router",
            "response": None,
            "metadata": {}
        }
    
    try:
        # Create query engine
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
            use_async=False
        )
        
        query_engine = RetrieverQueryEngine(
            retriever=router,
            response_synthesizer=response_synthesizer
        )
        
        # Execute query
        response = query_engine.query(query)
        
        # Extract metadata from retrieved nodes
        metadata = {
            "query": query,
            "top_k": top_k,
            "total_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        # Get routing information from the first node if available
        if hasattr(response, 'source_nodes') and response.source_nodes:
            first_node = response.source_nodes[0]
            if hasattr(first_node.node, 'metadata'):
                node_metadata = first_node.node.metadata
                metadata.update({
                    "index": node_metadata.get('selected_index', 'unknown'),
                    "strategy": node_metadata.get('selected_strategy', 'unknown'),
                    "index_confidence": node_metadata.get('index_confidence'),
                    "strategy_confidence": node_metadata.get('strategy_confidence'),
                    "num_sources": len(response.source_nodes)
                })
        
        return {
            "response": str(response),
            "metadata": metadata,
            "error": None
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "response": None,
            "metadata": {
                "query": query,
                "total_time_ms": round((time.time() - start_time) * 1000, 2)
            }
        }


def format_output(result: dict):
    """Format the output for display."""
    if result["error"]:
        print(f"❌ Error: {result['error']}")
        return
    
    # Print the response
    print("\n📝 Response:")
    print("=" * 60)
    print(result["response"])
    print("=" * 60)
    
    # Print metadata
    metadata = result["metadata"]
    
    print(f"\n📊 Routing Information:")
    index = metadata.get("index", "unknown")
    strategy = metadata.get("strategy", "unknown")
    latency = metadata.get("total_time_ms", 0)
    
    print(f"index = {index} | strategy = {strategy} | latency = {latency} ms")
    
    # Additional details if available
    if metadata.get("num_sources"):
        print(f"sources = {metadata['num_sources']}")
    
    if metadata.get("index_confidence") is not None:
        print(f"index_confidence = {metadata['index_confidence']:.3f}")
    
    if metadata.get("strategy_confidence") is not None:
        print(f"strategy_confidence = {metadata['strategy_confidence']:.3f}")


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