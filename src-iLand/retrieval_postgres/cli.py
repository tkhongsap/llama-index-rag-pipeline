"""
Command Line Interface for iLand PostgreSQL Retrieval System

Main entry point for CLI commands for testing PostgreSQL retrieval strategies, 
performance analysis, and system management. Based on local CLI pattern.
"""

import argparse
from .cli_handlers_postgres import iLandPostgresRetrievalCLI


def main():
    """Main CLI entry point for PostgreSQL retrieval."""
    parser = argparse.ArgumentParser(description="iLand PostgreSQL Retrieval System CLI")
    
    # Connection and setup
    parser.add_argument("--init-system", action="store_true", 
                       help="Initialize PostgreSQL retrieval system")
    parser.add_argument("--test-connection", action="store_true", 
                       help="Test PostgreSQL connection and table existence")
    
    # Query operations
    parser.add_argument("--query", type=str, help="Execute a single query")
    parser.add_argument("--rag-response", type=str, help="Generate detailed RAG response for query")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    parser.add_argument("--test-queries", nargs="+", help="Test multiple queries")
    
    # Strategy configuration
    parser.add_argument("--strategy", type=str, 
                       choices=["auto", "vector", "hybrid", "recursive", "chunk_decoupling", 
                               "planner", "metadata", "summary"],
                       default="auto", help="Retrieval strategy to use")
    parser.add_argument("--strategy-selector", choices=["llm", "heuristic", "round_robin"],
                       default="llm", help="Strategy selection method for router")
    
    # Query parameters
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--similarity-threshold", type=float, help="Similarity threshold for filtering")
    
    # Performance and testing
    parser.add_argument("--benchmark", action="store_true", 
                       help="Run benchmark tests on retrieval strategies")
    parser.add_argument("--test-all-strategies", action="store_true", 
                       help="Test all retrieval strategies with sample queries")
    parser.add_argument("--compare-strategies", nargs="+", 
                       help="Compare specific strategies with a query")
    
    # Analytics and monitoring
    parser.add_argument("--analytics", action="store_true", 
                       help="Show query analytics and statistics")
    parser.add_argument("--popular-queries", action="store_true", 
                       help="Show most popular queries")
    parser.add_argument("--clear-analytics", action="store_true", 
                       help="Clear analytics data")
    
    # Database management
    parser.add_argument("--db-status", action="store_true", 
                       help="Show database status and table information")
    parser.add_argument("--check-indices", action="store_true", 
                       help="Check and analyze database indices")
    parser.add_argument("--metadata-stats", action="store_true", 
                       help="Show metadata distribution statistics")
    
    # Configuration
    parser.add_argument("--config-show", action="store_true", 
                       help="Show current configuration")
    parser.add_argument("--config-test", action="store_true", 
                       help="Test configuration settings")
    
    # Filters and advanced options
    parser.add_argument("--province", type=str, help="Filter by Thai province")
    parser.add_argument("--district", type=str, help="Filter by district")
    parser.add_argument("--deed-type", type=str, help="Filter by deed type")
    parser.add_argument("--year", type=int, help="Filter by year")
    
    # Output options
    parser.add_argument("--output-format", choices=["table", "json", "detailed"], 
                       default="table", help="Output format")
    parser.add_argument("--export", type=str, help="Export results to file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = iLandPostgresRetrievalCLI(verbose=args.verbose)
    
    # Basic dependency check first
    if hasattr(cli, 'check_dependencies') and not cli.check_dependencies():
        print("\n❌ Missing required dependencies. Please install:")
        print("   pip install psycopg2-binary asyncpg")
        return
    
    # Connection and setup commands
    if args.test_connection:
        cli.test_connection()
        return
    
    if args.init_system:
        cli.init_system()
        return
    
    # Configuration commands
    if args.config_show:
        cli.show_config()
        return
    
    if args.config_test:
        cli.test_config()
        return
    
    # Database management commands
    if args.db_status:
        cli.show_db_status()
        return
    
    if args.check_indices:
        cli.check_indices()
        return
    
    if args.metadata_stats:
        cli.show_metadata_stats()
        return
    
    # Analytics commands
    if args.analytics:
        cli.show_analytics()
        return
    
    if args.popular_queries:
        cli.show_popular_queries()
        return
    
    if args.clear_analytics:
        cli.clear_analytics()
        return
    
    # Initialize router for query operations
    if not cli.init_router(
        strategy_selector=args.strategy_selector,
        similarity_threshold=args.similarity_threshold
    ):
        print("Failed to initialize router")
        return
    
    # Build filters from arguments
    filters = {}
    if args.province:
        filters['province'] = args.province
    if args.district:
        filters['district'] = args.district
    if args.deed_type:
        filters['deed_type'] = args.deed_type
    if args.year:
        filters['year'] = args.year
    
    # Query operations
    if args.query:
        cli.query(
            args.query, 
            top_k=args.top_k, 
            strategy=args.strategy,
            filters=filters,
            output_format=args.output_format,
            export_file=args.export
        )
    
    elif args.rag_response:
        cli.detailed_rag_response(args.rag_response, filters=filters)
    
    elif args.test_queries:
        cli.test_queries(
            args.test_queries, 
            top_k=args.top_k, 
            strategy=args.strategy,
            filters=filters
        )
    
    elif args.compare_strategies:
        if not args.query:
            print("--compare-strategies requires --query")
            return
        cli.compare_strategies(
            args.query,
            args.compare_strategies,
            top_k=args.top_k,
            filters=filters
        )
    
    elif args.benchmark:
        cli.run_benchmark()
    
    elif args.test_all_strategies:
        cli.test_all_strategies(top_k=args.top_k)
    
    elif args.interactive:
        cli.interactive_mode(strategy=args.strategy, top_k=args.top_k)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 