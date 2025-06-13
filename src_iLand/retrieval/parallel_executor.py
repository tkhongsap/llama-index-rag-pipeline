"""
Parallel Strategy Execution for iLand Retrieval

Enables concurrent execution of multiple retrieval strategies for performance optimization.
Uses ThreadPoolExecutor for efficient parallel processing.
"""

import time
import concurrent.futures
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from llama_index.core.schema import NodeWithScore

from .retrievers.base import BaseRetrieverAdapter


class ParallelStrategyExecutor:
    """Executes multiple retrieval strategies in parallel."""
    
    def __init__(self, max_workers: int = 3, timeout_seconds: float = 30.0):
        """
        Initialize parallel executor.
        
        Args:
            max_workers: Maximum number of concurrent strategy executions
            timeout_seconds: Timeout for individual strategy execution
        """
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "timeout_executions": 0,
            "average_latency": 0.0
        }
    
    def execute_strategies_parallel(self, 
                                  query: str,
                                  strategies: Dict[str, BaseRetrieverAdapter],
                                  top_k: int = 5,
                                  return_strategy: str = "best",
                                  combine_results: bool = False) -> Dict[str, Any]:
        """
        Execute multiple strategies in parallel.
        
        Args:
            query: Query string to execute
            strategies: Dictionary of strategy_name -> adapter
            top_k: Number of results per strategy
            return_strategy: How to select results ("best", "fastest", "most_results")
            combine_results: Whether to combine results from all strategies
            
        Returns:
            Dict with selected results and execution metadata
        """
        if not strategies:
            return {
                "results": [],
                "selected_strategy": None,
                "execution_stats": {"error": "No strategies provided"}
            }
        
        start_time = time.time()
        strategy_results = {}
        strategy_latencies = {}
        strategy_errors = {}
        
        # Execute strategies in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all strategy executions
            future_to_strategy = {
                executor.submit(self._execute_single_strategy, name, adapter, query, top_k): name
                for name, adapter in strategies.items()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_strategy, timeout=self.timeout_seconds):
                strategy_name = future_to_strategy[future]
                
                try:
                    result = future.result(timeout=self.timeout_seconds)
                    strategy_results[strategy_name] = result["nodes"]
                    strategy_latencies[strategy_name] = result["latency"]
                    self.execution_stats["successful_executions"] += 1
                    
                except concurrent.futures.TimeoutError:
                    strategy_errors[strategy_name] = "Timeout"
                    self.execution_stats["timeout_executions"] += 1
                    
                except Exception as e:
                    strategy_errors[strategy_name] = str(e)
                    self.execution_stats["failed_executions"] += 1
                
                self.execution_stats["total_executions"] += 1
        
        total_latency = time.time() - start_time
        
        # Select results based on return strategy
        selected_results, selected_strategy = self._select_results(
            strategy_results, strategy_latencies, return_strategy, combine_results
        )
        
        # Update average latency
        self._update_average_latency(total_latency)
        
        return {
            "results": selected_results,
            "selected_strategy": selected_strategy,
            "execution_stats": {
                "total_latency": total_latency,
                "strategy_latencies": strategy_latencies,
                "strategy_errors": strategy_errors,
                "strategies_executed": len(strategies),
                "successful_strategies": len(strategy_results),
                "failed_strategies": len(strategy_errors)
            }
        }
    
    def _execute_single_strategy(self, 
                                strategy_name: str,
                                adapter: BaseRetrieverAdapter, 
                                query: str, 
                                top_k: int) -> Dict[str, Any]:
        """
        Execute a single strategy and return results with timing.
        
        Args:
            strategy_name: Name of the strategy
            adapter: Strategy adapter to execute
            query: Query string
            top_k: Number of results to retrieve
            
        Returns:
            Dict with nodes and execution latency
        """
        start_time = time.time()
        
        try:
            nodes = adapter.retrieve(query, top_k)
            latency = time.time() - start_time
            
            # Tag nodes with parallel execution metadata
            for node in nodes:
                if hasattr(node.node, 'metadata'):
                    node.node.metadata.update({
                        "parallel_execution": True,
                        "strategy_latency": latency,
                        "execution_mode": "parallel"
                    })
            
            return {
                "nodes": nodes,
                "latency": latency,
                "success": True
            }
            
        except Exception as e:
            latency = time.time() - start_time
            return {
                "nodes": [],
                "latency": latency,
                "success": False,
                "error": str(e)
            }
    
    def _select_results(self, 
                       strategy_results: Dict[str, List[NodeWithScore]],
                       strategy_latencies: Dict[str, float],
                       return_strategy: str,
                       combine_results: bool) -> Tuple[List[NodeWithScore], str]:
        """
        Select which results to return based on selection strategy.
        
        Args:
            strategy_results: Results from each strategy
            strategy_latencies: Latency for each strategy
            return_strategy: Selection method
            combine_results: Whether to combine all results
            
        Returns:
            Tuple of (selected_results, selected_strategy_name)
        """
        if not strategy_results:
            return [], "none"
        
        if combine_results:
            # Combine and deduplicate results from all strategies
            combined_results = []
            seen_content = set()
            
            for strategy_name, results in strategy_results.items():
                for node in results:
                    # Simple deduplication based on content hash
                    content_hash = hash(node.node.text[:200] if node.node.text else "")
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        # Tag with contributing strategy
                        if hasattr(node.node, 'metadata'):
                            node.node.metadata["contributing_strategy"] = strategy_name
                        combined_results.append(node)
            
            # Sort by score and return top results
            combined_results.sort(key=lambda x: x.score or 0, reverse=True)
            return combined_results, "combined"
        
        # Single strategy selection
        if return_strategy == "fastest":
            # Return results from fastest strategy
            fastest_strategy = min(strategy_latencies.keys(), key=lambda k: strategy_latencies[k])
            return strategy_results[fastest_strategy], fastest_strategy
            
        elif return_strategy == "most_results":
            # Return results from strategy with most results
            best_strategy = max(strategy_results.keys(), key=lambda k: len(strategy_results[k]))
            return strategy_results[best_strategy], best_strategy
            
        else:  # "best" or default
            # Return results from strategy with highest average score
            best_strategy = None
            best_avg_score = -1
            
            for strategy_name, results in strategy_results.items():
                if results:
                    avg_score = sum(node.score or 0 for node in results) / len(results)
                    if avg_score > best_avg_score:
                        best_avg_score = avg_score
                        best_strategy = strategy_name
            
            if best_strategy:
                return strategy_results[best_strategy], best_strategy
            else:
                # Fallback to first available results
                first_strategy = next(iter(strategy_results))
                return strategy_results[first_strategy], first_strategy
    
    def _update_average_latency(self, latency: float):
        """Update running average latency."""
        total_executions = self.execution_stats["total_executions"]
        if total_executions > 0:
            current_avg = self.execution_stats["average_latency"]
            self.execution_stats["average_latency"] = (current_avg * (total_executions - 1) + latency) / total_executions
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return self.execution_stats.copy()
    
    def reset_stats(self):
        """Reset execution statistics."""
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "timeout_executions": 0,
            "average_latency": 0.0
        } 