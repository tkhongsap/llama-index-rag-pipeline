"""
Command Line Interface for iLand Retrieval System

Provides CLI commands for loading embeddings, testing retrieval strategies, and performance analysis.
Adapted from src/agentic_retriever/cli.py for Thai land deed data.
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from router import iLandRouterRetriever
from index_classifier import create_default_iland_classifier
from retrievers import (
    VectorRetrieverAdapter,
    SummaryRetrieverAdapter,
    RecursiveRetrieverAdapter,
    MetadataRetrieverAdapter,
    ChunkDecouplingRetrieverAdapter,
    HybridRetrieverAdapter,
    PlannerRetrieverAdapter
)

# Import iLand embedding utilities
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from load_embedding import (
        load_latest_iland_embeddings,
        load_all_latest_iland_embeddings,
        get_iland_batch_summary
    )
except ImportError:
    print("Warning: Could not import iLand embedding utilities")
    load_latest_iland_embeddings = None
    load_all_latest_iland_embeddings = None
    get_iland_batch_summary = None


class iLandRetrievalCLI:
    """Command line interface for iLand retrieval system."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.router = None
        self.adapters = {}
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables")
    
    def load_embeddings(self, embedding_type: str = "all") -> bool:
        """
        Load iLand embeddings and create retriever adapters.
        
        Args:
            embedding_type: Type of embeddings to load ("all", "latest", "specific")
            
        Returns:
            True if successful, False otherwise
        """
        print(f"Loading iLand embeddings (type: {embedding_type})...")
        
        try:
            if embedding_type == "all" and load_all_latest_iland_embeddings:
                embeddings_data = load_all_latest_iland_embeddings()
            elif embedding_type == "latest" and load_latest_iland_embeddings:
                embeddings_data = load_latest_iland_embeddings()
            else:
                print(f"Embedding type '{embedding_type}' not supported or utilities not available")
                return False
            
            if not embeddings_data:
                print("No embedding data loaded")
                return False
            
            print(f"Loaded {len(embeddings_data)} embeddings")
            
            # Create adapters for the main iLand index
            index_name = "iland_land_deeds"
            self.adapters[index_name] = {}
            
            print("Creating retriever adapters...")
            
            # Vector adapter
            self.adapters[index_name]["vector"] = VectorRetrieverAdapter.from_iland_embeddings(
                embeddings_data, api_key=self.api_key
            )
            print("✓ Vector adapter created")
            
            # Summary adapter (using same embeddings for now)
            self.adapters[index_name]["summary"] = SummaryRetrieverAdapter.from_iland_embeddings(
                embeddings_data, api_key=self.api_key
            )
            print("✓ Summary adapter created")
            
            # Metadata adapter
            self.adapters[index_name]["metadata"] = MetadataRetrieverAdapter.from_iland_embeddings(
                embeddings_data, api_key=self.api_key
            )
            print("✓ Metadata adapter created")
            
            # Hybrid adapter
            self.adapters[index_name]["hybrid"] = HybridRetrieverAdapter.from_iland_embeddings(
                embeddings_data, api_key=self.api_key
            )
            print("✓ Hybrid adapter created")
            
            # Planner adapter
            self.adapters[index_name]["planner"] = PlannerRetrieverAdapter.from_iland_embeddings(
                embeddings_data, api_key=self.api_key
            )
            print("✓ Planner adapter created")
            
            # Chunk decoupling adapter (using same embeddings for both chunk and context)
            self.adapters[index_name]["chunk_decoupling"] = ChunkDecouplingRetrieverAdapter.from_iland_embeddings(
                embeddings_data, embeddings_data, api_key=self.api_key
            )
            print("✓ Chunk decoupling adapter created")
            
            # Recursive adapter (simplified - using vector index for both levels)
            vector_index = self.adapters[index_name]["vector"].index
            self.adapters[index_name]["recursive"] = RecursiveRetrieverAdapter.from_iland_indices(
                vector_index, vector_index
            )
            print("✓ Recursive adapter created")
            
            print(f"Successfully created {len(self.adapters[index_name])} adapters")
            return True
            
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return False
    
    def create_router(self, strategy_selector: str = "llm") -> bool:
        """
        Create the iLand router retriever.
        
        Args:
            strategy_selector: Strategy selection method
            
        Returns:
            True if successful, False otherwise
        """
        if not self.adapters:
            print("No adapters available. Please load embeddings first.")
            return False
        
        try:
            print(f"Creating iLand router with strategy selector: {strategy_selector}")
            
            # Create index classifier
            classifier = create_default_iland_classifier(api_key=self.api_key)
            
            # Create router
            self.router = iLandRouterRetriever(
                retrievers=self.adapters,
                index_classifier=classifier,
                strategy_selector=strategy_selector,
                api_key=self.api_key
            )
            
            print("✓ iLand router created successfully")
            return True
            
        except Exception as e:
            print(f"Error creating router: {e}")
            return False
    
    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Execute a query using the iLand router.
        
        Args:
            query_text: Query string (may contain Thai text)
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries
        """
        if not self.router:
            print("Router not initialized. Please create router first.")
            return []
        
        try:
            print(f"\nExecuting query: '{query_text}'")
            print("-" * 60)
            
            start_time = time.time()
            
            # Execute query
            from llama_index.core.schema import QueryBundle
            query_bundle = QueryBundle(query_str=query_text)
            nodes = self.router._retrieve(query_bundle)
            
            latency = time.time() - start_time
            
            # Format results
            results = []
            for i, node in enumerate(nodes[:top_k]):
                metadata = getattr(node.node, 'metadata', {})
                
                result = {
                    "rank": i + 1,
                    "score": node.score,
                    "text": node.node.text[:200] + "..." if len(node.node.text) > 200 else node.node.text,
                    "full_text": node.node.text,
                    "index": metadata.get("selected_index", "unknown"),
                    "strategy": metadata.get("selected_strategy", "unknown"),
                    "index_confidence": metadata.get("index_confidence", 0.0),
                    "strategy_confidence": metadata.get("strategy_confidence", 0.0),
                    "metadata": metadata
                }
                results.append(result)
            
            # Print results
            print(f"Found {len(results)} results in {latency:.2f}s")
            
            if results:
                first_result = results[0]
                print(f"Routed to: {first_result['index']}/{first_result['strategy']}")
                print(f"Confidence: Index={first_result['index_confidence']:.2f}, Strategy={first_result['strategy_confidence']:.2f}")
                print()
                
                for result in results:
                    print(f"[{result['rank']}] Score: {result['score']:.3f}")
                    print(f"Text: {result['text']}")
                    print()
            
            return results
            
        except Exception as e:
            print(f"Error executing query: {e}")
            return []
    
    def test_strategies(self, test_queries: List[str], top_k: int = 3) -> Dict[str, Any]:
        """
        Test all strategies with a set of queries.
        
        Args:
            test_queries: List of test queries
            top_k: Number of results per query
            
        Returns:
            Performance statistics
        """
        if not self.adapters:
            print("No adapters available. Please load embeddings first.")
            return {}
        
        print(f"\nTesting {len(test_queries)} queries across all strategies...")
        print("=" * 60)
        
        results = {}
        index_name = list(self.adapters.keys())[0]  # Use first available index
        
        for strategy_name, adapter in self.adapters[index_name].items():
            print(f"\nTesting strategy: {strategy_name}")
            print("-" * 40)
            
            strategy_results = []
            total_latency = 0
            
            for i, query in enumerate(test_queries):
                try:
                    start_time = time.time()
                    nodes = adapter.retrieve(query, top_k=top_k)
                    latency = time.time() - start_time
                    total_latency += latency
                    
                    query_result = {
                        "query": query,
                        "num_results": len(nodes),
                        "latency": latency,
                        "avg_score": sum(node.score for node in nodes) / len(nodes) if nodes else 0.0,
                        "top_score": nodes[0].score if nodes else 0.0
                    }
                    strategy_results.append(query_result)
                    
                    print(f"  Query {i+1}: {len(nodes)} results, {latency:.2f}s, top_score={query_result['top_score']:.3f}")
                    
                except Exception as e:
                    print(f"  Query {i+1}: ERROR - {e}")
                    strategy_results.append({
                        "query": query,
                        "num_results": 0,
                        "latency": 0.0,
                        "avg_score": 0.0,
                        "top_score": 0.0,
                        "error": str(e)
                    })
            
            # Calculate strategy statistics
            avg_latency = total_latency / len(test_queries)
            avg_results = sum(r["num_results"] for r in strategy_results) / len(strategy_results)
            avg_score = sum(r["avg_score"] for r in strategy_results) / len(strategy_results)
            
            results[strategy_name] = {
                "query_results": strategy_results,
                "avg_latency": avg_latency,
                "avg_results": avg_results,
                "avg_score": avg_score,
                "total_queries": len(test_queries)
            }
            
            print(f"  Summary: {avg_results:.1f} avg results, {avg_latency:.2f}s avg latency, {avg_score:.3f} avg score")
        
        return results
    
    def show_batch_summary(self):
        """Show summary of available iLand embedding batches."""
        if get_iland_batch_summary:
            try:
                summary = get_iland_batch_summary()
                print("\niLand Embedding Batch Summary:")
                print("=" * 40)
                print(summary)
            except Exception as e:
                print(f"Error getting batch summary: {e}")
        else:
            print("Batch summary utility not available")
    
    def interactive_mode(self):
        """Start interactive query mode."""
        print("\niLand Retrieval Interactive Mode")
        print("=" * 40)
        print("Enter queries to test the retrieval system.")
        print("Commands:")
        print("  /quit - Exit interactive mode")
        print("  /help - Show this help")
        print("  /summary - Show batch summary")
        print("  /strategies <query> - Test query with all strategies")
        print()
        
        while True:
            try:
                query = input("iLand> ").strip()
                
                if not query:
                    continue
                elif query == "/quit":
                    break
                elif query == "/help":
                    print("Commands: /quit, /help, /summary, /strategies <query>")
                elif query == "/summary":
                    self.show_batch_summary()
                elif query.startswith("/strategies "):
                    test_query = query[12:]  # Remove "/strategies "
                    if test_query:
                        self.test_strategies([test_query], top_k=3)
                else:
                    self.query(query)
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="iLand Retrieval System CLI")
    
    parser.add_argument("--load-embeddings", choices=["all", "latest"], 
                       help="Load iLand embeddings")
    parser.add_argument("--strategy-selector", choices=["llm", "heuristic", "round_robin"],
                       default="llm", help="Strategy selection method")
    parser.add_argument("--query", type=str, help="Execute a single query")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    parser.add_argument("--test-queries", nargs="+", help="Test multiple queries")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--batch-summary", action="store_true", help="Show batch summary")
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = iLandRetrievalCLI()
    
    # Show batch summary if requested
    if args.batch_summary:
        cli.show_batch_summary()
        return
    
    # Load embeddings if requested
    if args.load_embeddings:
        if not cli.load_embeddings(args.load_embeddings):
            print("Failed to load embeddings")
            return
        
        # Create router
        if not cli.create_router(args.strategy_selector):
            print("Failed to create router")
            return
    
    # Execute single query
    if args.query:
        if not cli.router:
            print("Router not initialized. Use --load-embeddings first.")
            return
        cli.query(args.query, args.top_k)
    
    # Test multiple queries
    elif args.test_queries:
        if not cli.router:
            print("Router not initialized. Use --load-embeddings first.")
            return
        cli.test_strategies(args.test_queries, args.top_k)
    
    # Interactive mode
    elif args.interactive:
        if not cli.router:
            print("Router not initialized. Use --load-embeddings first.")
            return
        cli.interactive_mode()
    
    # Default: show help
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 