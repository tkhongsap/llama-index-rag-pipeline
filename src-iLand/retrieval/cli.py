"""
Command Line Interface for iLand Retrieval System

Main entry point for CLI commands for loading embeddings, testing retrieval strategies, 
and performance analysis. Refactored for maintainability and modularity.
"""

import argparse
from .cli_handlers import iLandRetrievalCLI


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