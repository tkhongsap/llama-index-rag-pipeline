#!/usr/bin/env python3
"""
Test script for PostgreSQL retrieval implementation

Simple test to verify the retrieval_postgres module works correctly
with the existing PostgreSQL infrastructure.
"""

import os
import sys
from pathlib import Path

# Add src-iLand to path
src_iland_path = Path(__file__).parent / "src-iLand"
sys.path.insert(0, str(src_iland_path))

def test_configuration():
    """Test PostgreSQL configuration loading."""
    print("🔧 Testing PostgreSQL configuration...")
    
    try:
        from retrieval_postgres.config import PostgresConfig
        
        config = PostgresConfig.from_env()
        print(f"✅ Configuration loaded successfully")
        print(f"   Database: {config.db_name}")
        print(f"   Host: {config.db_host}:{config.db_port}")
        print(f"   Chunks table: {config.chunks_table}")
        print(f"   Summaries table: {config.summaries_table}")
        
        # Test validation
        config.validate()
        print("✅ Configuration validation passed")
        
        return config
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return None


def test_connection(config):
    """Test database connection."""
    print("\n🔗 Testing database connection...")
    
    try:
        from retrieval_postgres.utils.db_connection import ConnectionManager
        
        with ConnectionManager(config) as conn_mgr:
            # Test basic query
            result = conn_mgr.execute_query("SELECT 1 as test", fetch_results=True)
            
            if result and result[0][0] == 1:
                print("✅ Database connection successful")
            else:
                print("❌ Database connection failed")
                return False
                
            # Test table existence
            tables = [config.chunks_table, config.summaries_table]
            
            for table in tables:
                try:
                    count_result = conn_mgr.execute_query(
                        f"SELECT COUNT(*) FROM {table} LIMIT 1",
                        fetch_results=True
                    )
                    count = count_result[0][0] if count_result else 0
                    print(f"✅ Table {table}: {count} records available")
                except Exception as e:
                    print(f"⚠️  Table {table}: {e}")
            
            return True
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False


def test_vector_operations():
    """Test vector operations."""
    print("\n🧮 Testing vector operations...")
    
    try:
        from retrieval_postgres.utils.vector_ops import VectorOperations
        
        vector_ops = VectorOperations()
        
        # Test embedding generation
        test_text = "ที่ดินในกรุงเทพมหานคร"
        embedding = vector_ops.get_embedding(test_text)
        
        print(f"✅ Generated embedding for Thai text")
        print(f"   Text: {test_text}")
        print(f"   Embedding dimension: {len(embedding)}")
        
        # Test cosine similarity
        similarity = vector_ops.cosine_similarity(embedding, embedding)
        print(f"✅ Self-similarity: {similarity:.4f} (should be ~1.0)")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector operations test failed: {e}")
        return False


def test_basic_retriever(config):
    """Test basic PostgreSQL retriever."""
    print("\n🔍 Testing basic retriever...")
    
    try:
        from retrieval_postgres.retrievers.basic_postgres import BasicPostgresRetriever
        
        retriever = BasicPostgresRetriever(config)
        
        # Test retrieval with Thai query
        query = "ที่ดินในกรุงเทพ"
        results = retriever.retrieve(query, top_k=3)
        
        print(f"✅ Basic retrieval successful")
        print(f"   Query: {query}")
        print(f"   Results: {len(results)} nodes")
        
        if results:
            for i, node_with_score in enumerate(results[:2], 1):
                score = node_with_score.score
                text_preview = node_with_score.node.text[:100] + "..." if len(node_with_score.node.text) > 100 else node_with_score.node.text
                print(f"   Result {i}: Score {score:.4f}, Text: {text_preview}")
        
        retriever.close()
        return True
        
    except Exception as e:
        print(f"❌ Basic retriever test failed: {e}")
        return False


def test_router(config):
    """Test PostgreSQL router."""
    print("\n🧭 Testing router retriever...")
    
    try:
        from retrieval_postgres.router import PostgresRouterRetriever
        
        router = PostgresRouterRetriever(config, strategy_selector="heuristic")
        
        # Test with Thai geographic query
        query = "โฉนดที่ดินจังหวัดเชียงใหม่"
        results = router.retrieve(query, top_k=3)
        
        print(f"✅ Router retrieval successful")
        print(f"   Query: {query}")
        print(f"   Results: {len(results)} nodes")
        
        if results:
            # Check if routing metadata is present
            first_node = results[0].node
            if hasattr(first_node, 'metadata'):
                strategy = first_node.metadata.get('selected_strategy', 'unknown')
                method = first_node.metadata.get('selection_method', 'unknown')
                print(f"   Selected strategy: {strategy} ({method})")
        
        router.close()
        return True
        
    except Exception as e:
        print(f"❌ Router test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing PostgreSQL Retrieval Implementation")
    print("=" * 60)
    
    # Test configuration
    config = test_configuration()
    if not config:
        sys.exit(1)
    
    # Test connection
    if not test_connection(config):
        sys.exit(1)
    
    # Test vector operations
    if not test_vector_operations():
        sys.exit(1)
    
    # Test basic retriever
    if not test_basic_retriever(config):
        sys.exit(1)
    
    # Test router
    if not test_router(config):
        sys.exit(1)
    
    print("\n🎉 All tests passed! PostgreSQL retrieval implementation is working correctly.")
    print("\n📋 Next steps:")
    print("   1. Run: cd src-iLand/retrieval_postgres && python -m cli test-connection")
    print("   2. Try: python -m cli retrieve -q 'ที่ดินในกรุงเทพ' -s auto")
    print("   3. Benchmark: python -m cli benchmark -q 'โฉนดที่ดินจังหวัดเชียงใหม่'")


if __name__ == "__main__":
    main() 