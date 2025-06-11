"""
Command-line interface for PostgreSQL-based retrieval operations.

This CLI provides commands for testing, benchmarking, and using the PostgreSQL
retrieval system for iLand data.
"""

import argparse
import asyncio
import json
import sys
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from .config import PostgresConfig
from .utils.db_connection import PostgresConnectionManager
from .retrievers import (
    BasicPostgresRetriever,
    RecursivePostgresRetriever,
    MetadataFilterPostgresRetriever
)
from .query_engine import PostgresQueryEngine


def test_connection(args) -> None:
    """Test PostgreSQL database connection."""
    print("Testing PostgreSQL connection...")
    
    try:
        config = PostgresConfig.from_env()
        config.validate()
        print(f"✓ Configuration loaded successfully")
        print(f"  Database: {config.db_name}")
        print(f"  Host: {config.db_host}:{config.db_port}")
        print(f"  User: {config.db_user}")
        
        with PostgresConnectionManager(config) as conn_manager:
            conn = conn_manager.get_connection()
            cursor = conn.cursor()
            
            # Test basic connection
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            print(f"✓ Connected to PostgreSQL: {version}")
            
            # Test pgvector extension
            cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            if cursor.fetchone():
                print("✓ pgvector extension is available")
            else:
                print("⚠ pgvector extension not found")
            
            # Test table existence
            tables = [config.chunks_table, config.summaries_table, config.source_table]
            for table in tables:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    );
                """, (table,))
                exists = cursor.fetchone()[0]
                status = "✓" if exists else "✗"
                print(f"{status} Table '{table}': {'exists' if exists else 'missing'}")
            
            conn_manager.return_connection(conn)
            print("\n✓ Connection test completed successfully!")
            
    except Exception as e:
        print(f"✗ Connection test failed: {e}")
        sys.exit(1)


def retrieve_documents(args) -> None:
    """Retrieve documents using specified strategy."""
    print(f"Retrieving documents for query: '{args.query}'")
    print(f"Strategy: {args.strategy}")
    print(f"Top-k: {args.top_k}")
    
    try:
        config = PostgresConfig.from_env()
        
        # Initialize retriever based on strategy
        if args.strategy == "basic":
            retriever = BasicPostgresRetriever(config)
        elif args.strategy == "recursive":
            retriever = RecursivePostgresRetriever(config)
        elif args.strategy == "metadata":
            retriever = MetadataFilterPostgresRetriever(config)
        else:
            print(f"✗ Unknown strategy: {args.strategy}")
            sys.exit(1)
        
        # Perform retrieval
        start_time = time.time()
        results = retriever.retrieve(args.query, top_k=args.top_k)
        end_time = time.time()
        
        print(f"\n✓ Retrieved {len(results)} documents in {end_time - start_time:.2f}s")
        
        # Display results
        for i, result in enumerate(results, 1):
            print(f"\n--- Document {i} ---")
            print(f"Score: {result.score:.4f}")
            print(f"Content: {result.text[:200]}...")
            if hasattr(result, 'metadata') and result.metadata:
                print(f"Metadata: {json.dumps(result.metadata, indent=2, ensure_ascii=False)}")
        
        # Save results if output file specified
        if args.output:
            output_data = {
                "query": args.query,
                "strategy": args.strategy,
                "top_k": args.top_k,
                "retrieval_time": end_time - start_time,
                "results": [
                    {
                        "score": result.score,
                        "text": result.text,
                        "metadata": getattr(result, 'metadata', {})
                    }
                    for result in results
                ]
            }
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Results saved to {args.output}")
            
    except Exception as e:
        print(f"✗ Retrieval failed: {e}")
        sys.exit(1)


def query_with_llm(args) -> None:
    """Query using LLM synthesis with PostgreSQL retrieval."""
    print(f"Querying with LLM synthesis: '{args.query}'")
    
    try:
        config = PostgresConfig.from_env()
        
        if not config.openai_api_key:
            print("✗ OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")
            sys.exit(1)
        
        query_engine = PostgresQueryEngine(config)
        
        start_time = time.time()
        response = query_engine.query(args.query)
        end_time = time.time()
        
        print(f"\n✓ Query completed in {end_time - start_time:.2f}s")
        print(f"\nResponse:\n{response}")
        
        if args.output:
            output_data = {
                "query": args.query,
                "response": response,
                "query_time": end_time - start_time
            }
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Response saved to {args.output}")
            
    except Exception as e:
        print(f"✗ Query failed: {e}")
        sys.exit(1)


def benchmark_retrievers(args) -> None:
    """Benchmark different retrieval strategies."""
    print("Benchmarking retrieval strategies...")
    
    try:
        config = PostgresConfig.from_env()
        
        # Test queries
        test_queries = [
            "ข้อมูลเกี่ยวกับที่ดินในจังหวัดชัยนาท",
            "กฎหมายเกี่ยวกับการซื้อขายที่ดิน",
            "ประเภทของโฉนดที่ดิน",
            "ขั้นตอนการจดทะเบียนที่ดิน"
        ]
        
        if args.queries:
            # Load queries from file
            with open(args.queries, 'r', encoding='utf-8') as f:
                test_queries = [line.strip() for line in f if line.strip()]
        
        strategies = {
            "basic": BasicPostgresRetriever(config),
            "recursive": RecursivePostgresRetriever(config),
            "metadata": MetadataFilterPostgresRetriever(config)
        }
        
        results = {}
        
        for strategy_name, retriever in strategies.items():
            print(f"\nTesting {strategy_name} strategy...")
            strategy_results = []
            
            for query in test_queries:
                start_time = time.time()
                docs = retriever.retrieve(query, top_k=args.top_k)
                end_time = time.time()
                
                strategy_results.append({
                    "query": query,
                    "num_results": len(docs),
                    "time": end_time - start_time,
                    "avg_score": sum(doc.score for doc in docs) / len(docs) if docs else 0
                })
                
                print(f"  Query: {query[:50]}... -> {len(docs)} docs in {end_time - start_time:.2f}s")
            
            results[strategy_name] = strategy_results
        
        # Print summary
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        for strategy_name, strategy_results in results.items():
            avg_time = sum(r["time"] for r in strategy_results) / len(strategy_results)
            avg_results = sum(r["num_results"] for r in strategy_results) / len(strategy_results)
            avg_score = sum(r["avg_score"] for r in strategy_results) / len(strategy_results)
            
            print(f"\n{strategy_name.upper()} Strategy:")
            print(f"  Average time: {avg_time:.3f}s")
            print(f"  Average results: {avg_results:.1f}")
            print(f"  Average score: {avg_score:.4f}")
        
        # Save detailed results
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Detailed results saved to {args.output}")
            
    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PostgreSQL-based retrieval CLI for iLand data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test database connection
  python -m retrieval_postgres.cli test-connection
  
  # Retrieve documents
  python -m retrieval_postgres.cli retrieve "ที่ดินในชัยนาท" --strategy basic --top-k 5
  
  # Query with LLM synthesis
  python -m retrieval_postgres.cli query "อธิบายกฎหมายที่ดิน" --output response.json
  
  # Benchmark strategies
  python -m retrieval_postgres.cli benchmark --top-k 10 --output benchmark.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Test connection command
    test_parser = subparsers.add_parser('test-connection', help='Test PostgreSQL connection')
    test_parser.set_defaults(func=test_connection)
    
    # Retrieve command
    retrieve_parser = subparsers.add_parser('retrieve', help='Retrieve documents')
    retrieve_parser.add_argument('query', help='Search query')
    retrieve_parser.add_argument('--strategy', choices=['basic', 'recursive', 'metadata'], 
                                default='basic', help='Retrieval strategy')
    retrieve_parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    retrieve_parser.add_argument('--output', help='Output file for results (JSON)')
    retrieve_parser.set_defaults(func=retrieve_documents)
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query with LLM synthesis')
    query_parser.add_argument('query', help='Search query')
    query_parser.add_argument('--output', help='Output file for response (JSON)')
    query_parser.set_defaults(func=query_with_llm)
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark retrieval strategies')
    benchmark_parser.add_argument('--queries', help='File with test queries (one per line)')
    benchmark_parser.add_argument('--top-k', type=int, default=5, help='Number of results per query')
    benchmark_parser.add_argument('--output', help='Output file for results (JSON)')
    benchmark_parser.set_defaults(func=benchmark_retrievers)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main() 