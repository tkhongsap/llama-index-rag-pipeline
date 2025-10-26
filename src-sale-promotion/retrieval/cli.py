"""
Sales Promotion CLI - Interactive Query Interface

Simple command-line interface for querying Thai sales promotions.
Supports interactive mode, single-query mode, and automated testing.

Usage:
    python -m src_sale_promotion.retrieval.cli --interactive
    python -m src_sale_promotion.retrieval.cli --query "โปรโมชันเบียร์"
    python -m src_sale_promotion.retrieval.cli --run-tests
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment
load_dotenv()

# Import Phase 3 components
from retrieval.loader import EmbeddingLoader
from retrieval.fast_metadata_index import FastMetadataIndex
from retrieval.router import PromotionRouter
from retrieval.query_engine import PromotionQueryEngine
from retrieval.index_classifier import IndexClassifier
from retrieval.strategy_selector import StrategySelector
from retrieval.cache import QueryCache
from retrieval.retrievers import (
    VectorRetrieverAdapter,
    SummaryRetrieverAdapter,
    MetadataRetrieverAdapter,
    RecursiveRetrieverAdapter,
    HybridRetrieverAdapter,
    ChunkDecouplingRetrieverAdapter,
    PlannerRetrieverAdapter,
)
from llama_index.llms.openai import OpenAI


class SalesPromotionCLI:
    """
    Main CLI interface for sales promotion queries.
    
    Provides three modes:
    1. Interactive mode - Continuous query loop
    2. Single query mode - One query and exit
    3. Test mode - Run automated test suite
    """
    
    def __init__(self, verbose: bool = False, debug: bool = False):
        """
        Initialize CLI.
        
        Args:
            verbose: Enable verbose output
            debug: Enable debug output
        """
        self.verbose = verbose
        self.debug = debug
        self.query_engine = None
        self.router = None
        self.session_queries = 0
        self.session_start = time.time()
    
    def load_system(self):
        """Load embeddings, build indices, and initialize query engine."""
        print("\n" + "="*70)
        print("  Sales Promotion Query System v4.0")
        print("="*70)
        print("\nInitializing system...")
        
        # Load embeddings
        print("  [1/5] Loading embeddings...", end="", flush=True)
        loader = EmbeddingLoader()
        indexnode_embs, summary_embs = loader.load_all_embeddings()
        print(" ✓")
        
        # Validate
        print("  [2/5] Validating embeddings...", end="", flush=True)
        validation = loader.validate_embeddings(indexnode_embs, summary_embs)
        if not validation["validation_passed"]:
            raise ValueError("Embedding validation failed")
        print(" ✓")
        
        # Build vector index
        print("  [3/5] Building vector index...", end="", flush=True)
        vector_index = loader.build_vector_index(indexnode_embs, summary_embs)
        print(" ✓")
        
        # Build metadata index
        print("  [4/5] Building metadata index...", end="", flush=True)
        metadata_index = FastMetadataIndex()
        metadata_index.build_index(loader.nodes)
        print(" ✓")
        
        # Initialize query system
        print("  [5/5] Initializing query engine...", end="", flush=True)
        
        # LLM
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        llm = OpenAI(model="gpt-5", api_key=api_key, temperature=0.0)
        
        # Retrievers
        retrievers = {
            "vector": VectorRetrieverAdapter(vector_index),
            "summary": SummaryRetrieverAdapter(vector_index),
            "metadata": MetadataRetrieverAdapter(vector_index, metadata_index, llm),
            "recursive": RecursiveRetrieverAdapter(vector_index),
            "hybrid": HybridRetrieverAdapter(vector_index, loader.nodes),
            "chunk_decoupling": ChunkDecouplingRetrieverAdapter(vector_index),
            "planner": PlannerRetrieverAdapter(vector_index, llm)
        }
        
        # Router
        self.router = PromotionRouter(
            index_classifier=IndexClassifier(llm),
            strategy_selector=StrategySelector(llm),
            retrievers=retrievers,
            cache=QueryCache(max_size=100, ttl=3600),
            enable_caching=True,
            debug_logging=self.debug
        )
        
        # Query engine
        self.query_engine = PromotionQueryEngine(
            router=self.router,
            llm=llm,
            verbose=self.debug
        )
        
        print(" ✓")
        
        print("\n✅ System ready!")
        print(f"   - Embeddings: 142 (71 promotions)")
        print(f"   - Strategies: 7 available")
        print(f"   - Models: GPT-5 + text-embedding-3-large")
        print("="*70)
    
    def interactive_mode(self, top_k: int = 5, strategy_override: Optional[str] = None):
        """
        Run interactive query loop.
        
        Args:
            top_k: Number of results to return
            strategy_override: Force specific strategy
        """
        print("\nInteractive Mode")
        print("Type your question in Thai or English. Type 'exit' or 'quit' to stop.\n")
        
        while True:
            try:
                # Get user input
                query = input("Query: ").strip()
                
                # Check for exit
                if query.lower() in ['exit', 'quit', 'q']:
                    self._print_session_summary()
                    break
                
                # Skip empty queries
                if not query:
                    print("Please enter a query.\n")
                    continue
                
                # Process query
                print()
                self.process_query(query, top_k, strategy_override)
                print("\n" + "-"*70 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user")
                self._print_session_summary()
                break
            
            except Exception as e:
                self.handle_error(e)
                print()
    
    def single_query_mode(
        self,
        query: str,
        top_k: int = 5,
        strategy_override: Optional[str] = None
    ):
        """
        Process single query and exit.
        
        Args:
            query: Query string
            top_k: Number of results
            strategy_override: Force specific strategy
        """
        print()
        self.process_query(query, top_k, strategy_override)
        print()
    
    def test_mode(self):
        """Run automated test suite."""
        print("\n" + "="*70)
        print("  Running Automated Test Suite")
        print("="*70)
        
        try:
            # Import test runner
            from retrieval.test_runner import TestRunner
            
            # Run tests
            runner = TestRunner(self.query_engine)
            results = runner.run_all_tests()
            
            # Generate reports
            runner.generate_report(results)
            
            # Display summary
            passed = sum(1 for r in results if r.overall_pass)
            total = len(results)
            pass_rate = (passed / total * 100) if total > 0 else 0
            
            print("\n" + "="*70)
            print(f"Test Results: {passed}/{total} passed ({pass_rate:.0f}%)")
            print("="*70)
            
        except ImportError:
            print("❌ Error: test_runner.py not found. Please implement test framework first.")
        except Exception as e:
            print(f"❌ Error running tests: {e}")
    
    def process_query(
        self,
        query: str,
        top_k: int = 5,
        strategy_override: Optional[str] = None
    ):
        """
        Process a single query through the pipeline.
        
        Args:
            query: User query
            top_k: Number of results
            strategy_override: Force specific strategy
        """
        try:
            start_time = time.time()
            
            # Execute query
            result = self.query_engine.query(
                query,
                top_k=top_k,
                strategy_override=strategy_override,
                return_nodes=True
            )
            
            # Format and display output
            self.format_output(result)
            
            # Track session stats
            self.session_queries += 1
            
        except Exception as e:
            self.handle_error(e)
    
    def format_output(self, result: Dict[str, Any]):
        """
        Format and display query results.
        
        Args:
            result: Query result dictionary from query_engine
        """
        metadata = result.get("metadata", {})
        
        # Header
        print(f"Query: {result['query']}")
        print(f"Strategy: {metadata.get('selected_strategy', 'unknown')} "
              f"(confidence: {metadata.get('strategy_confidence', 0):.2f})")
        print()
        
        # Response
        print(result['response'])
        print()
        
        # Basic stats
        print(f"Promotions Found: {metadata.get('num_results', 0)}")
        
        # Verbose mode
        if self.verbose:
            print()
            print("Performance Metrics:")
            latency = metadata.get('latency', {})
            print(f"  Total Time: {latency.get('total_seconds', 0):.2f}s")
            print(f"  Retrieval: {latency.get('retrieval_seconds', 0):.3f}s")
            print(f"  Synthesis: {latency.get('synthesis_seconds', 0):.2f}s")
            print(f"  Cache: {'HIT' if metadata.get('cache_hit') else 'MISS'}")
            
            if metadata.get('top_scores'):
                print(f"  Top-3 Scores: {[f'{s:.3f}' for s in metadata['top_scores'][:3]]}")
        else:
            print(f"Response Time: {metadata.get('latency', {}).get('total_seconds', 0):.1f}s")
        
        # Debug mode
        if self.debug and 'nodes' in result:
            print("\nDebug Info:")
            print(f"  Retrieved {len(result['nodes'])} nodes")
            for i, node in enumerate(result['nodes'][:3], 1):
                print(f"  {i}. {node.node.metadata.get('document_id', 'N/A')} "
                      f"(score: {node.score:.3f})")
    
    def handle_error(self, error: Exception):
        """
        Handle and display errors gracefully.
        
        Args:
            error: Exception that occurred
        """
        print(f"\n❌ Error: {str(error)}")
        
        if self.debug:
            import traceback
            traceback.print_exc()
    
    def _print_session_summary(self):
        """Print session statistics on exit."""
        if self.session_queries > 0:
            session_time = time.time() - self.session_start
            print(f"\n{'='*70}")
            print("Session Summary:")
            print(f"  Queries: {self.session_queries}")
            print(f"  Session time: {session_time:.1f}s")
            
            if self.router:
                cache_stats = self.router.get_cache_stats()
                print(f"  Cache hit rate: {cache_stats['hit_rate_percent']:.1f}%")
            
            print("="*70)
        
        print("\n👋 Goodbye!\n")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sales Promotion Query System - CLI Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python -m src_sale_promotion.retrieval.cli --interactive
  
  # Single query
  python -m src_sale_promotion.retrieval.cli --query "โปรโมชันเบียร์"
  
  # Verbose output
  python -m src_sale_promotion.retrieval.cli -q "โปรโมชันเบียร์" -v
  
  # Force strategy
  python -m src_sale_promotion.retrieval.cli -q "โปรโมชันเบียร์" -s metadata
  
  # Run tests
  python -m src_sale_promotion.retrieval.cli --run-tests
        """
    )
    
    # Modes
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Start interactive query mode"
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Run single query and exit"
    )
    parser.add_argument(
        "-t", "--run-tests",
        action="store_true",
        help="Run automated test suite"
    )
    
    # Options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output with performance metrics"
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Show debug information"
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    parser.add_argument(
        "-s", "--strategy",
        type=str,
        choices=["vector", "metadata", "hybrid", "summary", "recursive", "chunk_decoupling", "planner"],
        help="Force specific retrieval strategy"
    )
    
    return parser.parse_args()


def main():
    """CLI entry point."""
    args = parse_arguments()
    
    # Check if no mode specified
    if not (args.interactive or args.query or args.run_tests):
        print("Error: No mode specified. Use --help for options.\n")
        print("Quick start:")
        print("  Interactive mode: python -m src_sale_promotion.retrieval.cli --interactive")
        print("  Single query:     python -m src_sale_promotion.retrieval.cli --query 'โปรโมชันเบียร์'")
        print("  Run tests:        python -m src_sale_promotion.retrieval.cli --run-tests")
        sys.exit(1)
    
    try:
        # Initialize CLI
        cli = SalesPromotionCLI(verbose=args.verbose, debug=args.debug)
        
        # Load system
        cli.load_system()
        
        # Execute requested mode
        if args.run_tests:
            cli.test_mode()
        
        elif args.interactive:
            cli.interactive_mode(top_k=args.top_k, strategy_override=args.strategy)
        
        elif args.query:
            cli.single_query_mode(args.query, top_k=args.top_k, strategy_override=args.strategy)
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user\n")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
