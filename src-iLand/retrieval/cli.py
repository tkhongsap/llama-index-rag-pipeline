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
except ImportError:
    # Fallback to absolute imports when running from different directory
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

# Import response synthesis for natural language responses
try:
    from llama_index.core.response_synthesizers import ResponseMode
    from llama_index.core import get_response_synthesizer
    from llama_index.llms.openai import OpenAI
except ImportError:
    print("Warning: Could not import response synthesis utilities")
    ResponseMode = None
    get_response_synthesizer = None
    OpenAI = None


class iLandRetrievalCLI:
    """Command line interface for iLand retrieval system."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.router = None
        self.adapters = {}
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.cache_manager = None
        self.parallel_executor = None
        self.response_synthesizer = None
        
        if not self.api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables")
        
        # Initialize response synthesizer for natural language responses
        self._initialize_response_synthesizer()
    
    def _initialize_response_synthesizer(self):
        """Initialize the response synthesizer for generating natural language responses."""
        if not self.api_key or not get_response_synthesizer or not OpenAI:
            print("Warning: Response synthesis not available (missing API key or dependencies)")
            return
        
        try:
            llm = OpenAI(model="gpt-4.1-mini", api_key=self.api_key)
            self.response_synthesizer = get_response_synthesizer(
                response_mode=ResponseMode.COMPACT,
                llm=llm
            )
        except Exception as e:
            print(f"Warning: Could not initialize response synthesizer: {e}")
    
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
                embeddings_data, batch_path = load_all_latest_iland_embeddings()
            elif embedding_type == "latest" and load_latest_iland_embeddings:
                embeddings_data, batch_path = load_latest_iland_embeddings()
            else:
                print(f"Embedding type '{embedding_type}' not supported or utilities not available")
                return False
            
            if not embeddings_data:
                print("No embedding data loaded")
                return False
            
            print(f"Loaded {len(embeddings_data)} embeddings from {batch_path}")
            
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
                
                # Generate natural language response
                if self.response_synthesizer and nodes:
                    try:
                        print("🤖 Natural Language Response:")
                        print("-" * 40)
                        response = self.response_synthesizer.synthesize(query_text, nodes)
                        print(response.response)
                        print()
                    except Exception as e:
                        print(f"⚠️ Response generation failed: {str(e)[:100]}...")
                        print()
                
                print("📄 Retrieved Documents:")
                print("-" * 40)
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
                    
                    # Generate RAG response for the first query or if specifically requested
                    if i == 0 and self.response_synthesizer and nodes:
                        try:
                            response = self.response_synthesizer.synthesize(query, nodes)
                            print(f"    🤖 RAG Response: {response.response[:150]}...")
                        except Exception as e:
                            print(f"    ⚠️ Response generation failed: {str(e)[:50]}...")
                    
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
    
    def setup_performance_optimizations(self, enable_caching: bool = True, 
                                       enable_parallel: bool = True) -> bool:
        """
        Setup performance optimization features.
        
        Args:
            enable_caching: Enable query result caching
            enable_parallel: Enable parallel strategy execution
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Setup cache manager
            if enable_caching:
                self.cache_manager = iLandCacheManager.from_env()
                print("✓ Cache manager initialized")
            
            # Setup parallel executor
            if enable_parallel:
                self.parallel_executor = ParallelStrategyExecutor(max_workers=3)
                print("✓ Parallel executor initialized")
            
            return True
            
        except Exception as e:
            print(f"Error setting up performance optimizations: {e}")
            return False
    
    def test_parallel_strategies(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Test query using parallel strategy execution.
        
        Args:
            query: Query string
            top_k: Number of results per strategy
            
        Returns:
            Parallel execution results
        """
        if not self.adapters or not self.parallel_executor:
            print("Adapters or parallel executor not available")
            return {}
        
        index_name = list(self.adapters.keys())[0]
        strategies = self.adapters[index_name]
        
        print(f"\nExecuting parallel strategies for: '{query}'")
        print("-" * 60)
        
        # Test different parallel execution modes
        results = {}
        
        # Mode 1: Best strategy selection
        print("Mode 1: Best strategy selection")
        result_best = self.parallel_executor.execute_strategies_parallel(
            query=query,
            strategies=strategies,
            top_k=top_k,
            return_strategy="best",
            combine_results=False
        )
        results["best"] = result_best
        print(f"  Selected: {result_best['selected_strategy']}")
        print(f"  Results: {len(result_best['results'])}")
        print(f"  Latency: {result_best['execution_stats']['total_latency']:.2f}s")
        
        # Mode 2: Fastest strategy
        print("\nMode 2: Fastest strategy")
        result_fastest = self.parallel_executor.execute_strategies_parallel(
            query=query,
            strategies=strategies,
            top_k=top_k,
            return_strategy="fastest",
            combine_results=False
        )
        results["fastest"] = result_fastest
        print(f"  Selected: {result_fastest['selected_strategy']}")
        print(f"  Results: {len(result_fastest['results'])}")
        print(f"  Latency: {result_fastest['execution_stats']['total_latency']:.2f}s")
        
        # Mode 3: Combined results
        print("\nMode 3: Combined results")
        result_combined = self.parallel_executor.execute_strategies_parallel(
            query=query,
            strategies=strategies,
            top_k=top_k,
            return_strategy="best",
            combine_results=True
        )
        results["combined"] = result_combined
        print(f"  Selected: {result_combined['selected_strategy']}")
        print(f"  Results: {len(result_combined['results'])}")
        print(f"  Latency: {result_combined['execution_stats']['total_latency']:.2f}s")
        
        # Show execution statistics
        print(f"\nExecution Statistics:")
        stats = self.parallel_executor.get_stats()
        print(f"  Total executions: {stats['total_executions']}")
        print(f"  Successful: {stats['successful_executions']}")
        print(f"  Failed: {stats['failed_executions']}")
        print(f"  Average latency: {stats['average_latency']:.2f}s")
        
        return results
    
    def show_cache_stats(self):
        """Show cache performance statistics."""
        if not self.cache_manager:
            print("Cache manager not initialized")
            return
        
        stats = self.cache_manager.get_stats()
        print("\nCache Performance Statistics:")
        print("=" * 40)
        
        query_stats = stats.get("query_cache", {})
        print(f"Query Cache:")
        print(f"  Hit rate: {query_stats.get('hit_rate', 0):.2%}")
        print(f"  Total queries: {query_stats.get('total_queries', 0)}")
        print(f"  Cache size: {query_stats.get('size', 0)}/{query_stats.get('max_size', 0)}")
        print(f"  TTL: {query_stats.get('ttl_seconds', 0)}s")
    
    def clear_caches(self):
        """Clear all caches."""
        if self.cache_manager:
            self.cache_manager.clear_all_caches()
            print("✓ All caches cleared")
        else:
            print("Cache manager not initialized")
    
    def detailed_rag_response(self, query_text: str):
        """
        Generate a detailed RAG response for a query.
        
        Args:
            query_text: Query string (may contain Thai text)
        """
        if not self.router:
            print("Router not initialized. Please load embeddings first.")
            return
        
        if not self.response_synthesizer:
            print("Response synthesizer not available. Check API key and dependencies.")
            return
        
        try:
            print(f"\n🤖 Generating RAG Response for: '{query_text}'")
            print("=" * 60)
            
            start_time = time.time()
            
            # Execute query to get retrieved documents
            from llama_index.core.schema import QueryBundle
            query_bundle = QueryBundle(query_str=query_text)
            nodes = self.router._retrieve(query_bundle)
            
            retrieval_time = time.time() - start_time
            
            if not nodes:
                print("❌ No documents retrieved for this query.")
                return
            
            # Generate response
            synthesis_start = time.time()
            response = self.response_synthesizer.synthesize(query_text, nodes)
            synthesis_time = time.time() - synthesis_start
            
            total_time = time.time() - start_time
            
            # Display results
            print(f"📊 Performance Metrics:")
            print(f"  Documents retrieved: {len(nodes)}")
            print(f"  Retrieval time: {retrieval_time:.2f}s")
            print(f"  Response synthesis time: {synthesis_time:.2f}s")
            print(f"  Total time: {total_time:.2f}s")
            print()
            
            # Show routing information
            if nodes:
                metadata = getattr(nodes[0].node, 'metadata', {})
                index = metadata.get("selected_index", "unknown")
                strategy = metadata.get("selected_strategy", "unknown")
                print(f"🎯 Routing Information:")
                print(f"  Selected index: {index}")
                print(f"  Selected strategy: {strategy}")
                print()
            
            print(f"🤖 Generated Response:")
            print("-" * 40)
            print(response.response)
            print()
            
            print(f"📚 Source Documents:")
            print("-" * 40)
            for i, node in enumerate(nodes[:3], 1):  # Show top 3 sources
                metadata = node.node.metadata
                province = metadata.get('province', 'N/A')
                district = metadata.get('district', 'N/A')
                ownership = metadata.get('deed_holding_type', 'N/A')
                
                print(f"[{i}] Score: {node.score:.3f}")
                print(f"    Location: {province}, {district}")
                print(f"    Ownership: {ownership}")
                print(f"    Text: {node.node.text[:150]}...")
                print()
            
        except Exception as e:
            print(f"❌ Error generating RAG response: {e}")
    
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
        print("  /parallel <query> - Test parallel strategy execution")
        print("  /response <query> - Get detailed RAG response for query")
        print("  /cache-stats - Show cache statistics")
        print("  /clear-cache - Clear all caches")
        print()
        
        while True:
            try:
                query = input("iLand> ").strip()
                
                if not query:
                    continue
                elif query == "/quit":
                    break
                elif query == "/help":
                    print("Commands: /quit, /help, /summary, /strategies <query>, /parallel <query>, /response <query>, /cache-stats, /clear-cache")
                elif query == "/summary":
                    self.show_batch_summary()
                elif query == "/cache-stats":
                    self.show_cache_stats()
                elif query == "/clear-cache":
                    self.clear_caches()
                elif query.startswith("/strategies "):
                    test_query = query[12:]  # Remove "/strategies "
                    if test_query:
                        self.test_strategies([test_query], top_k=3)
                elif query.startswith("/parallel "):
                    test_query = query[10:]  # Remove "/parallel "
                    if test_query:
                        self.test_parallel_strategies(test_query, top_k=5)
                elif query.startswith("/response "):
                    test_query = query[10:]  # Remove "/response "
                    if test_query:
                        self.detailed_rag_response(test_query)
                else:
                    self.query(query)
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def test_retrieval_strategies(self, strategy_selector: str = "llm") -> Dict[str, Any]:
        """
        Run comprehensive retrieval strategy tests.
        
        Args:
            strategy_selector: Strategy selection method to test
            
        Returns:
            Test results
        """
        try:
            # Import the test suite
            import sys
            from pathlib import Path
            tests_dir = Path(__file__).parent.parent.parent / 'tests'
            sys.path.insert(0, str(tests_dir))
            
            from test_iland_retrieval_strategies import iLandRetrievalStrategyTester
            
            print(f"\nRunning iLand Retrieval Strategy Tests")
            print(f"Strategy Selector: {strategy_selector}")
            print("=" * 60)
            
            # Create and run tester
            tester = iLandRetrievalStrategyTester()
            results = tester.run_all_tests(strategy_selector)
            
            return results
            
        except ImportError as e:
            print(f"Error importing test suite: {e}")
            return {}
        except Exception as e:
            print(f"Error running strategy tests: {e}")
            return {}


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="iLand Retrieval System CLI")
    
    parser.add_argument("--load-embeddings", choices=["all", "latest"], 
                       help="Load iLand embeddings")
    parser.add_argument("--strategy-selector", choices=["llm", "heuristic", "round_robin"],
                       default="llm", help="Strategy selection method")
    parser.add_argument("--query", type=str, help="Execute a single query")
    parser.add_argument("--rag-response", type=str, help="Generate detailed RAG response for query")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    parser.add_argument("--test-queries", nargs="+", help="Test multiple queries")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--batch-summary", action="store_true", help="Show batch summary")
    parser.add_argument("--enable-cache", action="store_true", help="Enable query result caching")
    parser.add_argument("--enable-parallel", action="store_true", help="Enable parallel strategy execution")
    parser.add_argument("--parallel-query", type=str, help="Test parallel execution with query")
    parser.add_argument("--cache-stats", action="store_true", help="Show cache statistics")
    parser.add_argument("--clear-cache", action="store_true", help="Clear all caches")
    parser.add_argument("--test-retrieval-strategies", action="store_true", 
                       help="Run comprehensive retrieval strategy tests")
    
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
        
        # Setup performance optimizations
        if args.enable_cache or args.enable_parallel:
            cli.setup_performance_optimizations(
                enable_caching=args.enable_cache,
                enable_parallel=args.enable_parallel
            )
    
    # Execute single query
    if args.query:
        if not cli.router:
            print("Router not initialized. Use --load-embeddings first.")
            return
        cli.query(args.query, args.top_k)
    
    # Generate detailed RAG response
    elif args.rag_response:
        if not cli.router:
            print("Router not initialized. Use --load-embeddings first.")
            return
        cli.detailed_rag_response(args.rag_response)
    
    # Test multiple queries
    elif args.test_queries:
        if not cli.router:
            print("Router not initialized. Use --load-embeddings first.")
            return
        cli.test_strategies(args.test_queries, args.top_k)
    
    # Test parallel execution
    elif args.parallel_query:
        if not cli.router:
            print("Router not initialized. Use --load-embeddings first.")
            return
        if not cli.parallel_executor:
            cli.setup_performance_optimizations(enable_parallel=True)
        cli.test_parallel_strategies(args.parallel_query, args.top_k)
    
    # Show cache statistics
    elif args.cache_stats:
        if not cli.cache_manager:
            cli.setup_performance_optimizations(enable_caching=True)
        cli.show_cache_stats()
    
    # Clear caches
    elif args.clear_cache:
        if not cli.cache_manager:
            cli.setup_performance_optimizations(enable_caching=True)
        cli.clear_caches()
    
    # Interactive mode
    elif args.interactive:
        if not cli.router:
            print("Router not initialized. Use --load-embeddings first.")
            return
        cli.interactive_mode()
    
    # Test retrieval strategies
    elif args.test_retrieval_strategies:
        cli.test_retrieval_strategies(args.strategy_selector)
    
    # Default: show help
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 