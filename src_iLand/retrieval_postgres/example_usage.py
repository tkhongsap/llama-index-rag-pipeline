"""
Example usage of PostgreSQL retrieval system for iLand data.

This script demonstrates how to use the PostgreSQL-based retrieval
with complete parity to the local file-based system.
"""

import os
from typing import Dict, Any

from retrieval_postgres import (
    PostgresRetrievalConfig,
    PostgresRouterRetriever,
    PostgresVectorRetriever,
    PostgresHybridRetriever,
    create_postgres_adapter,
    HybridModeAdapter
)


def example_basic_vector_search():
    """Example of basic vector similarity search."""
    print("\n=== PostgreSQL Vector Search Example ===")
    
    # Configure PostgreSQL connection
    config = PostgresRetrievalConfig(
        db_host="localhost",
        db_port=5432,
        db_name="iland_embeddings",
        db_user="postgres",
        db_password=os.getenv("POSTGRES_PASSWORD", ""),
        default_top_k=5
    )
    
    # Create vector retriever
    vector_retriever = PostgresVectorRetriever(
        config=config,
        use_bge_embeddings=True  # Use BGE-M3 embeddings
    )
    
    # Example queries
    queries = [
        "โฉนดที่ดินในจังหวัดชัยนาท",
        "การโอนกรรมสิทธิ์ที่ดิน",
        "นส.3 อำเภอเมือง"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        from llama_index.core.schema import QueryBundle
        results = vector_retriever._retrieve(QueryBundle(query_str=query))
        
        for i, node in enumerate(results[:3]):
            print(f"\nResult {i+1}:")
            print(f"  Score: {node.score:.3f}")
            print(f"  Text: {node.node.text[:100]}...")
            print(f"  Document: {node.node.metadata.get('document_title', 'Unknown')}")


def example_hybrid_search():
    """Example of hybrid search combining vector and keyword."""
    print("\n=== PostgreSQL Hybrid Search Example ===")
    
    config = PostgresRetrievalConfig()
    
    # Create hybrid retriever
    hybrid_retriever = PostgresHybridRetriever(
        config=config,
        alpha=0.7,  # 70% vector, 30% keyword
        use_bge_embeddings=True
    )
    
    # Search with Thai keywords
    query = "โฉนด นส.3 จังหวัดชัยนาท อำเภอวัดสิงห์"
    
    # Basic search
    from llama_index.core.schema import QueryBundle
    results = hybrid_retriever._retrieve(QueryBundle(query_str=query))
    
    print(f"\nHybrid search for: {query}")
    for i, node in enumerate(results[:3]):
        print(f"\nResult {i+1}:")
        print(f"  Hybrid Score: {node.score:.3f}")
        print(f"  Vector Score: {node.node.metadata.get('vector_score', 0):.3f}")
        print(f"  Keyword Score: {node.node.metadata.get('keyword_score', 0):.3f}")
        print(f"  Text: {node.node.text[:100]}...")


def example_router_retriever():
    """Example of using the router with multiple strategies."""
    print("\n=== PostgreSQL Router Retriever Example ===")
    
    config = PostgresRetrievalConfig()
    
    # Create router retriever
    router = PostgresRouterRetriever.from_config(
        config=config,
        api_key=os.getenv("OPENAI_API_KEY"),
        strategy_selector="llm",  # Use LLM for strategy selection
        llm_strategy_mode="enhanced"
    )
    
    # The router needs retriever adapters registered
    # In practice, these would be set up during initialization
    
    # Create adapters for different strategies
    vector_adapter = create_postgres_adapter(
        PostgresVectorRetriever,
        config=config
    )
    
    hybrid_adapter = create_postgres_adapter(
        PostgresHybridRetriever,
        config=config
    )
    
    # Register adapters with router
    router.add_retriever("iland_land_deeds", "vector", vector_adapter)
    router.add_retriever("iland_land_deeds", "hybrid", hybrid_adapter)
    
    # Example queries that would trigger different strategies
    queries = [
        "What are the key features of Thai land deeds?",  # Likely vector
        "โฉนด นส.3 จังหวัดชัยนาท ราคา",  # Likely hybrid due to Thai keywords
        "Show me land deeds from Chai Nat province"  # Likely metadata
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        from llama_index.core.schema import QueryBundle
        results = router._retrieve(QueryBundle(query_str=query))
        
        if results:
            first_result = results[0]
            metadata = first_result.node.metadata
            print(f"  Selected Strategy: {metadata.get('selected_strategy', 'unknown')}")
            print(f"  Strategy Confidence: {metadata.get('strategy_confidence', 0):.2f}")
            print(f"  Results: {len(results)} nodes retrieved")


def example_hybrid_mode():
    """Example of hybrid mode supporting both PostgreSQL and local files."""
    print("\n=== Hybrid Mode Example (PostgreSQL + Local) ===")
    
    config = PostgresRetrievalConfig(
        enable_hybrid_mode=True,
        local_index_path="/path/to/local/index"
    )
    
    # Create PostgreSQL adapter
    postgres_adapter = create_postgres_adapter(
        PostgresVectorRetriever,
        config=config
    )
    
    # In practice, you would also create a local adapter
    # local_adapter = create_local_adapter(...)
    
    # Create hybrid mode adapter
    hybrid_mode = HybridModeAdapter(
        postgres_adapter=postgres_adapter,
        local_adapter=None,  # Would be the local adapter
        mode="postgres_first",  # Try PostgreSQL first, fallback to local
        fallback_on_error=True
    )
    
    # Use hybrid mode for retrieval
    query = "Land ownership transfer procedures"
    results = hybrid_mode.retrieve(query)
    
    print(f"\nHybrid mode search: {query}")
    print(f"Mode: postgres_first with fallback")
    print(f"Results: {len(results)} nodes")
    
    # Check metrics
    metrics = hybrid_mode.get_metrics()
    print(f"\nMetrics:")
    print(f"  PostgreSQL queries: {metrics['postgres_queries']}")
    print(f"  Local queries: {metrics['local_queries']}")
    print(f"  Fallbacks: {metrics['fallbacks']}")


def example_query_analytics():
    """Example of using query analytics features."""
    print("\n=== Query Analytics Example ===")
    
    config = PostgresRetrievalConfig()
    
    # Create router with query logging enabled
    router = PostgresRouterRetriever.from_config(
        config=config,
        enable_query_logging=True
    )
    
    # Get query statistics
    stats = router.get_query_stats(hours=24)
    
    print("\nQuery Statistics (Last 24 hours):")
    print(f"  Total queries: {stats['overall'].get('total_queries', 0)}")
    print(f"  Unique queries: {stats['overall'].get('unique_queries', 0)}")
    print(f"  Avg latency: {stats['overall'].get('avg_latency_ms', 0):.1f}ms")
    print(f"  Cache hit rate: {stats['overall'].get('cache_hit_rate', 0):.2%}")
    
    print("\nStrategy Distribution:")
    for strategy in stats.get('by_strategy', [])[:5]:
        print(f"  {strategy['selected_strategy']}: {strategy['count']} queries")
    
    # Get popular queries
    popular = router.get_popular_queries(limit=5)
    print("\nPopular Queries:")
    for i, query_info in enumerate(popular):
        print(f"  {i+1}. {query_info['query'][:50]}... ({query_info['count']} times)")


if __name__ == "__main__":
    print("PostgreSQL Retrieval System Examples")
    print("=" * 50)
    
    # Run examples based on what's available
    try:
        # Check if PostgreSQL is accessible
        import psycopg2
        config = PostgresRetrievalConfig()
        conn = psycopg2.connect(config.connection_string)
        conn.close()
        
        # Run examples
        example_basic_vector_search()
        example_hybrid_search()
        example_router_retriever()
        example_hybrid_mode()
        example_query_analytics()
        
    except Exception as e:
        print(f"\nNote: Could not connect to PostgreSQL: {e}")
        print("Make sure PostgreSQL is running and configured properly.")
        print("\nExample configuration:")
        print("  POSTGRES_HOST=localhost")
        print("  POSTGRES_PORT=5432")
        print("  POSTGRES_DB=iland_embeddings")
        print("  POSTGRES_USER=postgres")
        print("  POSTGRES_PASSWORD=your_password")