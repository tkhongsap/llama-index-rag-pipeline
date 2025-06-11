"""
Command-line interface for PostgreSQL-based retrieval operations.

This CLI provides commands for testing, debugging, and using the PostgreSQL retrieval system for iLand data.
"""

import argparse
import logging
import sys
import time
from typing import Dict, Any, Optional, List

from retrieval_postgres.config import PostgresConfig
from retrieval_postgres.utils.db_connection import PostgresConnectionManager
from retrieval_postgres.router import PostgresRouterRetriever
from retrieval_postgres.retrievers import (
    BasicPostgresRetriever,
    RecursivePostgresRetriever,
    MetadataFilterPostgresRetriever,
    SentenceWindowPostgresRetriever,
    AutoMergePostgresRetriever,
    EnsemblePostgresRetriever,
    AgenticPostgresRetriever
)

logger = logging.getLogger(__name__)


def test_connect_postgres(config: PostgresConfig) -> None:
    """
    Debug function: test connection to PostgreSQL and print table names (from config) and their existence.
    """
    print("Testing PostgreSQL connection and table existence...")
    try:
        conn_manager = PostgresConnectionManager(config)
        with conn_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            print(f"✓ Connected to PostgreSQL: {version}")

            # List of table names from config
            tables = [
                ("chunks_table", config.chunks_table),
                ("summaries_table", config.summaries_table),
                ("indexnodes_table", config.indexnodes_table),
                ("combined_table", config.combined_table),
                ("source_table", config.source_table)
            ]
            for (name, table) in tables:
                cursor.execute("""
                    SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s);
                """, (table,))
                exists = cursor.fetchone()[0]
                print(f"{'✓' if exists else '✗'} Table {name} ({table}): {'exists' if exists else 'missing'}")

        print("\n✓ Debug test completed.")
    except Exception as e:
        print(f"✗ Debug test failed: {e}")
        sys.exit(1)


def retrieve_documents(query: str, top_k: int, strategy: Optional[str] = None, filters: Optional[Dict[str, Any]] = None) -> None:
    """
    Retrieve documents using the router (auto strategy) or a forced strategy.
    """
    config = PostgresConfig.from_env()
    if strategy:
        # Force a specific strategy (e.g. metadata, basic, recursive, etc.)
        retriever_map = {
            "basic": BasicPostgresRetriever(config),
            "recursive": RecursivePostgresRetriever(config),
            "metadata": MetadataFilterPostgresRetriever(config),
            "window": SentenceWindowPostgresRetriever(config),
            "auto_merge": AutoMergePostgresRetriever(config),
            "ensemble": EnsemblePostgresRetriever(config),
            "agentic": AgenticPostgresRetriever(config)
        }
        if strategy not in retriever_map:
            print(f"✗ Unknown strategy: {strategy}. Available strategies: {list(retriever_map.keys())}")
            sys.exit(1)
        retriever = retriever_map[strategy]
    else:
        # Use router (auto strategy)
        retriever = PostgresRouterRetriever(config)

    print(f"Retrieving documents for query: '{query}' (strategy: {strategy or 'auto'}, top_k: {top_k})")
    try:
        start = time.time()
        results = retriever.retrieve(query, top_k=top_k, filters=filters)
        elapsed = time.time() - start
        print(f"\n✓ Retrieved {len(results)} documents in {elapsed:.2f}s")
        for (i, result) in enumerate(results, 1):
            print(f"\n--- Document {i} ---")
            print(f"Score: {result.score:.4f}")
            print(f"Content: {result.text[:200]}...")
            if hasattr(result, "metadata") and result.metadata:
                print(f"Metadata: {result.metadata}")
    except Exception as e:
        if ("does not exist" in str(e) or "UndefinedTable" in str(e)):
            print("❗ ตาราง embedding (เช่น data_iland_chunks) ยังไม่ถูกสร้างใน database นี้\nโปรดแจ้งทีม data/infra ให้สร้างตารางก่อนใช้งาน retrieval/query")
        else:
            print(f"✗ Retrieval failed: {e}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="iLand PostgreSQL Retrieval CLI (follows PRD: 13-postgres-retrieval-migration-prd.md)")
    parser.add_argument("--query", type=str, help="Execute a single query (uses router if --strategy not provided)")
    parser.add_argument("--strategy", type=str, choices=["basic", "recursive", "metadata", "window", "auto_merge", "ensemble", "agentic"], help="Force a specific retrieval strategy (e.g. metadata, basic, recursive)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return (default: 5)")
    parser.add_argument("--debug", action="store_true", help="Run debug test (test connection and table existence)")
    args = parser.parse_args()

    config = PostgresConfig.from_env()
    if args.debug:
        test_connect_postgres(config)
    elif args.query:
        retrieve_documents(args.query, args.top_k, args.strategy)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 