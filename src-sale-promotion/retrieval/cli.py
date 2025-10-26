#!/usr/bin/env python3
"""
CLI for sales promotion retrieval system.

Usage:
    python -m src_sale_promotion.retrieval.cli --interactive
    python -m src_sale_promotion.retrieval.cli --query "Show summer campaigns"
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Sales Promotion Retrieval CLI"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive query mode"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to execute"
    )
    parser.add_argument(
        "--load-embeddings",
        type=str,
        default="latest",
        help="Embeddings to load (default: latest)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["vector", "hybrid", "metadata", "recursive", "summary", "chunk_decoupling", "planner"],
        help="Force specific retrieval strategy"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Sales Promotion Retrieval System")
    print("=" * 60)
    print()

    # TODO: Implement CLI
    print("⚠️  CLI implementation coming soon!")
    print()
    print("Planned features:")
    print("  ✓ Interactive query mode")
    print("  ✓ Single query execution")
    print("  ✓ Batch query processing")
    print("  ✓ Strategy comparison")
    print("  ✓ Performance benchmarking")
    print()
    print("Reference implementation: src-imap/retrieval/cli.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
