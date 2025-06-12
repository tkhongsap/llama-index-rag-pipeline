#!/usr/bin/env python3
"""Test script to verify PostgreSQL planner and summary retrievers."""

import os
import sys

# Add src-iLand to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src-iLand'))

def test_imports():
    """Test that retrievers can be imported."""
    print("Testing imports...")
    
    try:
        from retrieval_postgres.retrievers import PostgresPlannerRetriever
        print("✓ PostgresPlannerRetriever imported successfully")
    except Exception as e:
        print(f"✗ Failed to import PostgresPlannerRetriever: {e}")
        return False
    
    try:
        from retrieval_postgres.retrievers import PostgresSummaryRetriever
        print("✓ PostgresSummaryRetriever imported successfully")
    except Exception as e:
        print(f"✗ Failed to import PostgresSummaryRetriever: {e}")
        return False
    
    return True


def test_instantiation():
    """Test that retrievers can be instantiated."""
    print("\nTesting instantiation...")
    
    from retrieval_postgres.config import PostgresRetrievalConfig
    
    # Create test config
    config = PostgresRetrievalConfig(
        db_host="localhost",
        db_port=5432,
        db_name="test_db",
        db_user="test_user",
        db_password="test_pass"
    )
    
    try:
        from retrieval_postgres.retrievers import PostgresPlannerRetriever
        planner = PostgresPlannerRetriever(config=config)
        print("✓ PostgresPlannerRetriever instantiated successfully")
    except Exception as e:
        print(f"✗ Failed to instantiate PostgresPlannerRetriever: {e}")
        return False
    
    try:
        from retrieval_postgres.retrievers import PostgresSummaryRetriever
        summary = PostgresSummaryRetriever(config=config)
        print("✓ PostgresSummaryRetriever instantiated successfully")
    except Exception as e:
        print(f"✗ Failed to instantiate PostgresSummaryRetriever: {e}")
        return False
    
    return True


def main():
    """Run tests."""
    print("PostgreSQL Retriever Tests")
    print("=" * 50)
    
    if not test_imports():
        print("\nImport tests failed!")
        return 1
    
    if not test_instantiation():
        print("\nInstantiation tests failed!")
        return 1
    
    print("\n✓ All tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())