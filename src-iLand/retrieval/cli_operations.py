"""
CLI Operations for iLand Retrieval System

Contains query execution, strategy testing, and other retrieval operations.
"""

import time
from typing import Dict, List, Any
from colorama import Fore, Style

from .cli_utils import (
    print_colored_header, print_success, print_error, print_warning,
    format_execution_time, get_retrieval_components
)


class iLandCLIOperations:
    """Handles CLI operations for iLand retrieval system."""
    
    def __init__(self, router, adapters, response_synthesizer=None, 
                 cache_manager=None, parallel_executor=None):
        """Initialize CLI operations."""
        self.router = router
        self.adapters = adapters
        self.response_synthesizer = response_synthesizer
        self.cache_manager = cache_manager
        self.parallel_executor = parallel_executor
    
    def execute_query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Execute a query using the iLand router.
        
        Args:
            query_text: Query string (may contain Thai text)
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries
        """
        if not self.router:
            print_error("Router not initialized. Please create router first.")
            return []
        
        try:
            print_colored_header(f"Executing query: '{query_text}'")
            
            start_time = time.time()
            
            # Execute query
            from llama_index.core.schema import QueryBundle
            query_bundle = QueryBundle(query_str=query_text)
            nodes = self.router._retrieve(query_bundle)
            
            latency = time.time() - start_time
            
            # Format results
            results = self._format_query_results(nodes, top_k)
            
            # Generate natural language response first
            self._generate_natural_response(query_text, nodes)
            
            # Print execution summary
            print(f"{Fore.GREEN}Found {len(results)} results in {latency:.2f}s{Style.RESET_ALL}")
            
            if results:
                self._print_routing_info(results[0])
                self._print_retrieved_documents(results)
            
            return results
            
        except Exception as e:
            print_error(f"Error executing query: {e}")
            return []
    
    def _format_query_results(self, nodes, top_k: int) -> List[Dict[str, Any]]:
        """Format query results into structured dictionaries."""
        results = []
        for i, node in enumerate(nodes[:top_k]):
            metadata = getattr(node.node, 'metadata', {})
            
            result = {
                "rank": i + 1,
                "score": node.score,
                "text": node.node.text[:200] + "..." if len(node.node.text) > 200 else node.node.text,
                "full_text": node.node.text,
                "index": metadata.get("selected_index", "unknown"),
                "strategy": metadata.get("selected_strategy", "unknown"),
                "index_confidence": metadata.get("index_confidence", 0.0),
                "strategy_confidence": metadata.get("strategy_confidence", 0.0),
                "metadata": metadata
            }
            results.append(result)
        
        return results
    
    def _generate_natural_response(self, query_text: str, nodes):
        """Generate and print natural language response."""
        if self.response_synthesizer and nodes:
            try:
                print_colored_header("🤖 Natural Language Response:", Fore.MAGENTA)
                response = self.response_synthesizer.synthesize(query_text, nodes)
                print(f"{Fore.MAGENTA}{response.response}{Style.RESET_ALL}")
                print()
            except Exception as e:
                print_warning(f"Response generation failed: {str(e)[:100]}...")
                print()
    
    def _print_routing_info(self, first_result: Dict[str, Any]):
        """Print routing information."""
        print(f"{Fore.YELLOW}Routed to: {first_result['index']}/{first_result['strategy']}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Confidence: Index={first_result['index_confidence']:.2f}, Strategy={first_result['strategy_confidence']:.2f}{Style.RESET_ALL}")
        print()
    
    def _print_retrieved_documents(self, results: List[Dict[str, Any]]):
        """Print retrieved documents summary."""
        print_colored_header("📄 Retrieved Documents:")
        for result in results:
            print(f"{Fore.WHITE}[{result['rank']}] Score: {result['score']:.3f}{Style.RESET_ALL}")
            print(f"{Fore.WHITE}Text: {result['text']}{Style.RESET_ALL}")
            print()
    
    def test_all_strategies(self, test_queries: List[str], top_k: int = 3) -> Dict[str, Any]:
        """
        Test all strategies with a set of queries.
        
        Args:
            test_queries: List of test queries
            top_k: Number of results per query
            
        Returns:
            Performance statistics
        """
        if not self.adapters:
            print_error("No adapters available. Please load embeddings first.")
            return {}
        
        print_colored_header(f"Testing {len(test_queries)} queries across all strategies")
        
        results = {}
        index_name = list(self.adapters.keys())[0]  # Use first available index
        
        for strategy_name, adapter in self.adapters[index_name].items():
            print(f"\nTesting strategy: {strategy_name}")
            print("-" * 40)
            
            strategy_results = self._test_single_strategy(
                strategy_name, adapter, test_queries, top_k
            )
            results[strategy_name] = strategy_results
        
        return results
    
    def _test_single_strategy(self, strategy_name: str, adapter, 
                            test_queries: List[str], top_k: int) -> Dict[str, Any]:
        """Test a single strategy with queries."""
        strategy_results = []
        total_latency = 0
        
        for i, query in enumerate(test_queries):
            try:
                start_time = time.time()
                nodes = adapter.retrieve(query, top_k=top_k)
                latency = time.time() - start_time
                total_latency += latency
                
                query_result = {
                    "query": query,
                    "num_results": len(nodes),
                    "latency": latency,
                    "avg_score": sum(node.score for node in nodes) / len(nodes) if nodes else 0.0,
                    "top_score": nodes[0].score if nodes else 0.0
                }
                strategy_results.append(query_result)
                
                print(f"  Query {i+1}: {len(nodes)} results, {latency:.2f}s, top_score={query_result['top_score']:.3f}")
                
                # Generate RAG response for the first query
                if i == 0 and self.response_synthesizer and nodes:
                    self._show_sample_response(query, nodes)
                
            except Exception as e:
                print(f"  Query {i+1}: ERROR - {e}")
                strategy_results.append({
                    "query": query,
                    "num_results": 0,
                    "latency": 0.0,
                    "avg_score": 0.0,
                    "top_score": 0.0,
                    "error": str(e)
                })
        
        # Calculate strategy statistics
        return self._calculate_strategy_stats(strategy_results, total_latency, len(test_queries))
    
    def _show_sample_response(self, query: str, nodes):
        """Show sample RAG response for a query."""
        try:
            response = self.response_synthesizer.synthesize(query, nodes)
            print(f"    🤖 RAG Response: {response.response[:150]}...")
        except Exception as e:
            print(f"    ⚠️ Response generation failed: {str(e)[:50]}...")
    
    def _calculate_strategy_stats(self, strategy_results: List[Dict], 
                                total_latency: float, num_queries: int) -> Dict[str, Any]:
        """Calculate statistics for a strategy."""
        avg_latency = total_latency / num_queries
        avg_results = sum(r["num_results"] for r in strategy_results) / len(strategy_results)
        avg_score = sum(r["avg_score"] for r in strategy_results) / len(strategy_results)
        
        stats = {
            "query_results": strategy_results,
            "avg_latency": avg_latency,
            "avg_results": avg_results,
            "avg_score": avg_score,
            "total_queries": num_queries
        }
        
        print(f"  Summary: {avg_results:.1f} avg results, {avg_latency:.2f}s avg latency, {avg_score:.3f} avg score")
        return stats
    
    def test_parallel_strategies(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Test query using parallel strategy execution.
        
        Args:
            query: Query string
            top_k: Number of results per strategy
            
        Returns:
            Parallel execution results
        """
        if not self.adapters or not self.parallel_executor:
            print_error("Adapters or parallel executor not available")
            return {}
        
        index_name = list(self.adapters.keys())[0]
        strategies = self.adapters[index_name]
        
        print_colored_header(f"Executing parallel strategies for: '{query}'")
        
        results = {}
        
        # Test different parallel execution modes
        modes = [
            ("best", "Best strategy selection"),
            ("fastest", "Fastest strategy"),
            ("combined", "Combined results")
        ]
        
        for mode, description in modes:
            print(f"\nMode: {description}")
            
            combine_results = (mode == "combined")
            return_strategy = "best" if mode == "combined" else mode
            
            result = self.parallel_executor.execute_strategies_parallel(
                query=query,
                strategies=strategies,
                top_k=top_k,
                return_strategy=return_strategy,
                combine_results=combine_results
            )
            
            results[mode] = result
            self._print_parallel_result(result)
        
        # Show execution statistics
        self._show_parallel_stats()
        
        return results
    
    def _print_parallel_result(self, result: Dict[str, Any]):
        """Print parallel execution result."""
        print(f"  Selected: {result['selected_strategy']}")
        print(f"  Results: {len(result['results'])}")
        print(f"  Latency: {result['execution_stats']['total_latency']:.2f}s")
    
    def _show_parallel_stats(self):
        """Show parallel execution statistics."""
        if not self.parallel_executor:
            return
        
        print_colored_header("Execution Statistics:")
        stats = self.parallel_executor.get_stats()
        print(f"  Total executions: {stats['total_executions']}")
        print(f"  Successful: {stats['successful_executions']}")
        print(f"  Failed: {stats['failed_executions']}")
        print(f"  Average latency: {stats['average_latency']:.2f}s")
    
    def generate_detailed_rag_response(self, query_text: str):
        """
        Generate a detailed RAG response for a query.
        
        Args:
            query_text: Query string (may contain Thai text)
        """
        if not self.router:
            print_error("Router not initialized. Please load embeddings first.")
            return
        
        if not self.response_synthesizer:
            print_error("Response synthesizer not available. Check API key and dependencies.")
            return
        
        try:
            print_colored_header(f"🤖 Generating RAG Response for: '{query_text}'")
            
            start_time = time.time()
            
            # Execute query to get retrieved documents
            from llama_index.core.schema import QueryBundle
            query_bundle = QueryBundle(query_str=query_text)
            nodes = self.router._retrieve(query_bundle)
            
            retrieval_time = time.time() - start_time
            
            if not nodes:
                print_error("No documents retrieved for this query.")
                return
            
            # Generate response
            synthesis_start = time.time()
            response = self.response_synthesizer.synthesize(query_text, nodes)
            synthesis_time = time.time() - synthesis_start
            
            total_time = time.time() - start_time
            
            # Display results
            self._show_rag_metrics(len(nodes), retrieval_time, synthesis_time, total_time)
            self._show_rag_routing_info(nodes)
            self._show_rag_response(response)
            self._show_rag_sources(nodes)
            
        except Exception as e:
            print_error(f"Error generating RAG response: {e}")
    
    def _show_rag_metrics(self, num_docs: int, retrieval_time: float, 
                         synthesis_time: float, total_time: float):
        """Show RAG performance metrics."""
        print(f"📊 Performance Metrics:")
        print(f"  Documents retrieved: {num_docs}")
        print(f"  Retrieval time: {retrieval_time:.2f}s")
        print(f"  Response synthesis time: {synthesis_time:.2f}s")
        print(f"  Total time: {total_time:.2f}s")
        print()
    
    def _show_rag_routing_info(self, nodes):
        """Show RAG routing information."""
        if nodes:
            metadata = getattr(nodes[0].node, 'metadata', {})
            index = metadata.get("selected_index", "unknown")
            strategy = metadata.get("selected_strategy", "unknown")
            print(f"🎯 Routing Information:")
            print(f"  Selected index: {index}")
            print(f"  Selected strategy: {strategy}")
            print()
    
    def _show_rag_response(self, response):
        """Show generated RAG response."""
        print(f"🤖 Generated Response:")
        print("-" * 40)
        print(response.response)
        print()
    
    def _show_rag_sources(self, nodes):
        """Show source documents for RAG response."""
        print(f"📚 Source Documents:")
        print("-" * 40)
        for i, node in enumerate(nodes[:3], 1):  # Show top 3 sources
            metadata = node.node.metadata
            province = metadata.get('province', 'N/A')
            district = metadata.get('district', 'N/A')
            ownership = metadata.get('deed_holding_type', 'N/A')
            
            print(f"[{i}] Score: {node.score:.3f}")
            print(f"    Location: {province}, {district}")
            print(f"    Ownership: {ownership}")
            print(f"    Text: {node.node.text[:150]}...")
            print() 