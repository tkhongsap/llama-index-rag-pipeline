"""
Stats Script

Analyzes agentic retrieval logs and prints summary statistics.
Usage: python -m agentic_retriever.stats
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter
import statistics



from .log_utils import read_log_entries, get_compressed_logs, read_compressed_log


def calculate_token_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculate the cost of tokens based on GPT-4o-mini pricing.
    
    Args:
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        
    Returns:
        Cost in USD
    """
    # GPT-4o-mini pricing (as of 2024)
    PROMPT_COST_PER_1K = 0.00015  # $0.15 per 1K tokens
    COMPLETION_COST_PER_1K = 0.0006  # $0.60 per 1K tokens
    
    prompt_cost = (prompt_tokens / 1000) * PROMPT_COST_PER_1K
    completion_cost = (completion_tokens / 1000) * COMPLETION_COST_PER_1K
    
    return prompt_cost + completion_cost


def analyze_logs(include_compressed: bool = False, limit: int = None) -> Dict[str, Any]:
    """
    Analyze log entries and return statistics.
    
    Args:
        include_compressed: Whether to include compressed log files
        limit: Maximum number of entries to analyze (most recent first)
        
    Returns:
        Dictionary with analysis results
    """
    # Read current log entries
    entries = read_log_entries(limit=limit)
    
    # Read compressed logs if requested
    if include_compressed:
        compressed_logs = get_compressed_logs()
        for log_file in compressed_logs:
            compressed_entries = read_compressed_log(log_file)
            entries.extend(compressed_entries)
        
        # Re-sort by timestamp and apply limit
        entries.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        if limit:
            entries = entries[:limit]
    
    if not entries:
        return {
            "total_queries": 0,
            "error": "No log entries found"
        }
    
    # Initialize counters and lists
    total_queries = len(entries)
    successful_queries = 0
    error_queries = 0
    
    latencies = []
    costs = []
    
    index_counter = Counter()
    strategy_counter = Counter()
    error_counter = Counter()
    
    # Analyze each entry
    for entry in entries:
        # Count by status
        if entry.get('error'):
            error_queries += 1
            error_counter[entry.get('error', 'Unknown error')] += 1
        else:
            successful_queries += 1
        
        # Collect latencies
        latency = entry.get('latency_ms', 0)
        if latency > 0:
            latencies.append(latency)
        
        # Calculate costs
        prompt_tokens = entry.get('prompt_tokens', 0)
        completion_tokens = entry.get('completion_tokens', 0)
        if prompt_tokens > 0 or completion_tokens > 0:
            cost = calculate_token_cost(prompt_tokens, completion_tokens)
            costs.append(cost)
        
        # Count indices and strategies
        index = entry.get('index', 'unknown')
        strategy = entry.get('strategy', 'unknown')
        
        if index != 'error':
            index_counter[index] += 1
        if strategy != 'error':
            strategy_counter[strategy] += 1
    
    # Calculate statistics
    stats = {
        "total_queries": total_queries,
        "successful_queries": successful_queries,
        "error_queries": error_queries,
        "success_rate": round(successful_queries / total_queries * 100, 1) if total_queries > 0 else 0
    }
    
    # Latency statistics
    if latencies:
        stats["latency"] = {
            "mean_ms": round(statistics.mean(latencies), 2),
            "median_ms": round(statistics.median(latencies), 2),
            "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
            "min_ms": round(min(latencies), 2),
            "max_ms": round(max(latencies), 2)
        }
    else:
        stats["latency"] = {"error": "No latency data available"}
    
    # Cost statistics
    if costs:
        stats["cost"] = {
            "total_usd": round(sum(costs), 6),
            "mean_per_query_usd": round(statistics.mean(costs), 6),
            "median_per_query_usd": round(statistics.median(costs), 6)
        }
    else:
        stats["cost"] = {"error": "No cost data available"}
    
    # Top indices and strategies
    stats["top_indices"] = dict(index_counter.most_common(5))
    stats["top_strategies"] = dict(strategy_counter.most_common(5))
    
    # Error analysis
    if error_counter:
        stats["top_errors"] = dict(error_counter.most_common(3))
    
    return stats


def format_stats_output(stats: Dict[str, Any]):
    """Format statistics for display."""
    if stats.get("error"):
        print(f"❌ {stats['error']}")
        return
    
    print("Agentic Retrieval Statistics")
    print("=" * 50)
    
    # Basic counts
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Successful: {stats['successful_queries']}")
    print(f"Errors: {stats['error_queries']}")
    print(f"Success Rate: {stats['success_rate']}%")
    
    # Latency statistics
    print(f"\nLatency Statistics:")
    latency = stats.get("latency", {})
    if "error" in latency:
        print(f"  {latency['error']}")
    else:
        print(f"  Mean: {latency['mean_ms']} ms")
        print(f"  Median: {latency['median_ms']} ms")
        print(f"  P95: {latency['p95_ms']} ms")
        print(f"  Range: {latency['min_ms']} - {latency['max_ms']} ms")
    
    # Cost statistics
    print(f"\nCost Statistics:")
    cost = stats.get("cost", {})
    if "error" in cost:
        print(f"  {cost['error']}")
    else:
        print(f"  Total Cost: ${cost['total_usd']:.6f}")
        print(f"  Mean per Query: ${cost['mean_per_query_usd']:.6f}")
        print(f"  Median per Query: ${cost['median_per_query_usd']:.6f}")
    
    # Top indices
    print(f"\nTop Indices:")
    top_indices = stats.get("top_indices", {})
    if top_indices:
        for index, count in top_indices.items():
            percentage = round(count / stats['total_queries'] * 100, 1)
            print(f"  {index}: {count} ({percentage}%)")
    else:
        print("  No index data available")
    
    # Top strategies
    print(f"\nTop Strategies:")
    top_strategies = stats.get("top_strategies", {})
    if top_strategies:
        for strategy, count in top_strategies.items():
            percentage = round(count / stats['total_queries'] * 100, 1)
            print(f"  {strategy}: {count} ({percentage}%)")
    else:
        print("  No strategy data available")
    
    # Error analysis
    if stats.get("top_errors"):
        print(f"\nTop Errors:")
        for error, count in stats["top_errors"].items():
            print(f"  {error}: {count}")


def main():
    """Main stats function."""
    parser = argparse.ArgumentParser(
        description="Analyze agentic retrieval logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m agentic_retriever.stats
  python -m agentic_retriever.stats --include-compressed
  python -m agentic_retriever.stats --limit 1000
        """
    )
    
    parser.add_argument(
        "--include-compressed",
        action="store_true",
        help="Include compressed log files in analysis"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of entries to analyze (most recent first)"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    args = parser.parse_args()
    
    # Analyze logs
    stats = analyze_logs(
        include_compressed=args.include_compressed,
        limit=args.limit
    )
    
    # Output results
    if args.json:
        import json
        print(json.dumps(stats, indent=2))
    else:
        format_stats_output(stats)


if __name__ == "__main__":
    main() 