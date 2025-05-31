"""
18_complete_pipeline_demo.py - Complete Retrieval Pipeline Demonstration

This script provides a comprehensive demonstration of the complete retrieval pipeline,
showcasing all implemented strategies and their capabilities in a unified interface.

Purpose:
- Demonstrate all retrieval strategies
- Provide performance comparisons
- Show real-world usage examples
- Offer interactive testing capabilities
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import importlib.util

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import our retrieval modules
from load_embeddings import create_index_from_latest_batch

# Import numbered modules using importlib
def import_module_from_file(module_name, file_path):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import all retrieval strategy modules
basic_query_engine = import_module_from_file("basic_query_engine", Path(__file__).parent / "10_basic_query_engine.py")
document_summary_retriever = import_module_from_file("document_summary_retriever", Path(__file__).parent / "11_document_summary_retriever.py")
recursive_retriever = import_module_from_file("recursive_retriever", Path(__file__).parent / "12_recursive_retriever.py")
metadata_filtering = import_module_from_file("metadata_filtering", Path(__file__).parent / "14_metadata_filtering.py")
chunk_decoupling = import_module_from_file("chunk_decoupling", Path(__file__).parent / "15_chunk_decoupling.py")
hybrid_search = import_module_from_file("hybrid_search", Path(__file__).parent / "16_hybrid_search.py")
query_planning_agent = import_module_from_file("query_planning_agent", Path(__file__).parent / "17_query_planning_agent.py")

# ---------- CONFIGURATION ---------------------------------------------------

# Load environment variables
load_dotenv(override=True)

# Strategy descriptions
STRATEGY_DESCRIPTIONS = {
    "basic": {
        "name": "Basic RAG Query Engine",
        "description": "Simple vector similarity search with response synthesis",
        "best_for": "General queries, quick responses",
        "complexity": "Low"
    },
    "document_summary": {
        "name": "Document Summary Retriever",
        "description": "Hierarchical retrieval using document summaries",
        "best_for": "Document-level analysis, overview queries",
        "complexity": "Medium"
    },
    "recursive": {
        "name": "Recursive Document Retriever",
        "description": "Multi-level retrieval with recursive refinement",
        "best_for": "Detailed analysis, specific information",
        "complexity": "Medium"
    },
    "metadata_filtering": {
        "name": "Metadata Filtering",
        "description": "LLM-inferred metadata filtering for targeted retrieval",
        "best_for": "Filtered searches, specific document types",
        "complexity": "Medium"
    },
    "hybrid_search": {
        "name": "Hybrid Search",
        "description": "Combines semantic and keyword search with result fusion",
        "best_for": "Keyword-specific queries, comprehensive search",
        "complexity": "High"
    },
    "chunk_decoupling": {
        "name": "Chunk Decoupling",
        "description": "Sentence-window retrieval with context expansion",
        "best_for": "Fine-grained retrieval, context-aware responses",
        "complexity": "High"
    },
    "query_planning": {
        "name": "Query Planning Agent",
        "description": "Query decomposition with parallel execution",
        "best_for": "Complex multi-part queries, comprehensive analysis",
        "complexity": "Very High"
    }
}

# Test queries for different complexity levels
TEST_QUERIES = {
    "simple": [
        "What educational qualifications are mentioned?",
        "What are the salary ranges in the profiles?",
        "How many years of experience do candidates have?"
    ],
    "moderate": [
        "Which profiles mention specific technical skills?",
        "Compare work experience levels across different companies",
        "What assessment scores are mentioned and how do they vary?"
    ],
    "complex": [
        "Compare educational qualifications and salary expectations between different experience levels",
        "What are the most common educational institutions, their graduates' salary ranges, and assessment score patterns?",
        "Analyze the relationship between technical skills, work experience, and compensation across all profiles"
    ]
}

# ---------- COMPLETE PIPELINE DEMO CLASS ------------------------------------

class CompletePipelineDemo:
    """Complete demonstration of all retrieval strategies."""
    
    def __init__(self):
        """Initialize all retrieval strategies."""
        print("🚀 COMPLETE RETRIEVAL PIPELINE DEMONSTRATION")
        print("=" * 80)
        print("Initializing all retrieval strategies...")
        
        self.strategies = {}
        self.performance_metrics = {}
        self._initialize_all_strategies()
    
    def _initialize_all_strategies(self):
        """Initialize all available retrieval strategies."""
        try:
            # Basic RAG Query Engine
            print("\n🔧 Initializing Basic RAG Query Engine...")
            basic_index = create_index_from_latest_batch(use_chunks=True)
            self.strategies['basic'] = basic_query_engine.BasicRAGQueryEngine(basic_index)
            
            # Document Summary Retriever
            print("🔧 Initializing Document Summary Retriever...")
            self.strategies['document_summary'] = document_summary_retriever.create_document_summary_retriever_from_latest_batch()
            
            # Recursive Document Retriever
            print("🔧 Initializing Recursive Document Retriever...")
            self.strategies['recursive'] = recursive_retriever.create_recursive_retriever_from_latest_batch()
            
            # Metadata Filtering
            print("🔧 Initializing Metadata Filtering...")
            self.strategies['metadata_filtering'] = metadata_filtering.create_metadata_filtered_retriever_from_latest_batch()
            
            # Hybrid Search
            print("🔧 Initializing Hybrid Search...")
            self.strategies['hybrid_search'] = hybrid_search.create_hybrid_search_engine_from_latest_batch()
            
            # Chunk Decoupling
            print("🔧 Initializing Chunk Decoupling...")
            self.strategies['chunk_decoupling'] = chunk_decoupling.create_advanced_decoupler_from_latest_batch()
            
            # Query Planning Agent
            print("🔧 Initializing Query Planning Agent...")
            self.strategies['query_planning'] = query_planning_agent.create_query_planning_agent()
            
            print(f"\n✅ Successfully initialized {len(self.strategies)} retrieval strategies!")
            
        except Exception as e:
            print(f"\n❌ Error initializing strategies: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def show_strategy_overview(self):
        """Display overview of all available strategies."""
        print("\n📋 RETRIEVAL STRATEGY OVERVIEW")
        print("=" * 80)
        
        for strategy_key, strategy_info in STRATEGY_DESCRIPTIONS.items():
            if strategy_key in self.strategies:
                print(f"\n🔹 {strategy_info['name']}")
                print(f"   Description: {strategy_info['description']}")
                print(f"   Best for: {strategy_info['best_for']}")
                print(f"   Complexity: {strategy_info['complexity']}")
    
    def benchmark_all_strategies(self, query: str) -> Dict[str, Any]:
        """Benchmark all strategies with a single query."""
        print(f"\n🏁 BENCHMARKING ALL STRATEGIES")
        print(f"Query: {query}")
        print("=" * 80)
        
        results = {}
        
        for strategy_name, strategy in self.strategies.items():
            print(f"\n📊 Testing {STRATEGY_DESCRIPTIONS[strategy_name]['name']}...")
            
            start_time = time.time()
            
            try:
                # Execute query based on strategy type
                if strategy_name == 'document_summary':
                    result = strategy.hierarchical_retrieve(query)
                    formatted_result = {
                        'response': f"Found {result['metadata']['total_docs_found']} relevant documents",
                        'sources': len(result['relevant_chunks']),
                        'metadata': result['metadata']
                    }
                elif strategy_name == 'recursive':
                    result = strategy.recursive_query(query, show_details=False)
                    formatted_result = result
                elif strategy_name == 'metadata_filtering':
                    result = strategy.query_with_auto_filters(query, show_filters=False)
                    formatted_result = result
                elif strategy_name == 'hybrid_search':
                    result = strategy.query(query, show_details=False)
                    formatted_result = result
                elif strategy_name == 'chunk_decoupling':
                    result = strategy.query_with_decoupled_chunks(query, show_details=False)
                    formatted_result = result
                elif strategy_name == 'query_planning':
                    result = strategy.plan_and_execute_query(query, show_details=False)
                    formatted_result = {
                        'response': result['final_response'],
                        'sources': 0,  # Query planning aggregates sources differently
                        'metadata': result['metadata']
                    }
                else:
                    result = strategy.query(query, show_sources=False, show_timing=False)
                    formatted_result = result
                
                end_time = time.time()
                execution_time = round(end_time - start_time, 2)
                
                results[strategy_name] = {
                    'success': True,
                    'execution_time': execution_time,
                    'response_length': len(formatted_result.get('response', '')),
                    'num_sources': formatted_result.get('sources', 0) if isinstance(formatted_result.get('sources'), int) else len(formatted_result.get('sources', [])),
                    'response': formatted_result.get('response', '')[:200] + "..."
                }
                
                print(f"  ✅ Completed in {execution_time}s")
                
            except Exception as e:
                end_time = time.time()
                execution_time = round(end_time - start_time, 2)
                
                results[strategy_name] = {
                    'success': False,
                    'execution_time': execution_time,
                    'error': str(e)
                }
                
                print(f"  ❌ Error: {str(e)}")
        
        return results
    
    def display_benchmark_results(self, results: Dict[str, Any]):
        """Display benchmark results in a formatted table."""
        print(f"\n📊 BENCHMARK RESULTS")
        print("-" * 100)
        print(f"{'Strategy':<25} {'Status':<10} {'Time (s)':<10} {'Sources':<10} {'Response Len':<15}")
        print("-" * 100)
        
        for strategy_name, result in results.items():
            strategy_display = STRATEGY_DESCRIPTIONS[strategy_name]['name'][:24]
            
            if result['success']:
                status = "✅ Success"
                time_str = f"{result['execution_time']:.2f}"
                sources_str = str(result['num_sources'])
                response_len_str = str(result['response_length'])
            else:
                status = "❌ Error"
                time_str = f"{result['execution_time']:.2f}"
                sources_str = "-"
                response_len_str = "-"
            
            print(f"{strategy_display:<25} {status:<10} {time_str:<10} {sources_str:<10} {response_len_str:<15}")
        
        print("-" * 100)
    
    def run_complexity_analysis(self):
        """Run analysis across different query complexity levels."""
        print("\n🎯 QUERY COMPLEXITY ANALYSIS")
        print("=" * 80)
        
        for complexity_level, queries in TEST_QUERIES.items():
            print(f"\n📈 {complexity_level.upper()} QUERIES")
            print("-" * 60)
            
            for i, query in enumerate(queries, 1):
                print(f"\n🔍 Query {i}: {query}")
                
                # Run benchmark for this query
                results = self.benchmark_all_strategies(query)
                
                # Show quick summary
                successful_strategies = [name for name, result in results.items() if result['success']]
                avg_time = sum(result['execution_time'] for result in results.values() if result['success']) / len(successful_strategies) if successful_strategies else 0
                
                print(f"📊 Summary: {len(successful_strategies)}/{len(results)} strategies successful, avg time: {avg_time:.2f}s")
    
    def interactive_demo(self):
        """Run interactive demonstration."""
        print("\n💬 INTERACTIVE RETRIEVAL PIPELINE DEMO")
        print("=" * 80)
        print("Available commands:")
        print("  • 'overview' - Show strategy overview")
        print("  • 'benchmark <query>' - Benchmark all strategies with a query")
        print("  • 'test <strategy> <query>' - Test specific strategy")
        print("  • 'complexity' - Run complexity analysis")
        print("  • 'quit' - Exit")
        
        while True:
            try:
                user_input = input("\n❓ Enter command: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'overview':
                    self.show_strategy_overview()
                
                elif user_input.lower() == 'complexity':
                    self.run_complexity_analysis()
                
                elif user_input.startswith('benchmark '):
                    query = user_input[10:].strip()
                    if query:
                        results = self.benchmark_all_strategies(query)
                        self.display_benchmark_results(results)
                    else:
                        print("❌ Please provide a query after 'benchmark'")
                
                elif user_input.startswith('test '):
                    parts = user_input[5:].strip().split(' ', 1)
                    if len(parts) == 2:
                        strategy, query = parts
                        if strategy in self.strategies:
                            results = self.benchmark_all_strategies(query)
                            if strategy in results:
                                result = results[strategy]
                                print(f"\n📝 {STRATEGY_DESCRIPTIONS[strategy]['name']} RESULT:")
                                if result['success']:
                                    print(f"Response: {result['response']}")
                                    print(f"Time: {result['execution_time']}s")
                                    print(f"Sources: {result['num_sources']}")
                                else:
                                    print(f"Error: {result['error']}")
                        else:
                            print(f"❌ Unknown strategy: {strategy}")
                            print(f"Available strategies: {', '.join(self.strategies.keys())}")
                    else:
                        print("❌ Usage: test <strategy> <query>")
                
                else:
                    print("❌ Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Demo interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")

# ---------- DEMONSTRATION FUNCTIONS -----------------------------------------

def run_complete_demo():
    """Run the complete pipeline demonstration."""
    try:
        # Initialize demo
        demo = CompletePipelineDemo()
        
        # Show strategy overview
        demo.show_strategy_overview()
        
        # Run a quick benchmark
        print("\n🚀 QUICK BENCHMARK")
        print("=" * 80)
        sample_query = "What educational qualifications are mentioned in the profiles?"
        results = demo.benchmark_all_strategies(sample_query)
        demo.display_benchmark_results(results)
        
        # Offer interactive mode
        print("\n🎮 Would you like to try interactive mode? (y/n)")
        response = input().strip().lower()
        if response in ['y', 'yes']:
            demo.interactive_demo()
        
    except Exception as e:
        print(f"\n❌ Demo error: {str(e)}")
        import traceback
        traceback.print_exc()

def quick_benchmark():
    """Run a quick benchmark of all strategies."""
    print("⚡ QUICK BENCHMARK")
    print("=" * 80)
    
    try:
        demo = CompletePipelineDemo()
        test_query = "What are the salary ranges mentioned in the profiles?"
        results = demo.benchmark_all_strategies(test_query)
        demo.display_benchmark_results(results)
        
    except Exception as e:
        print(f"❌ Quick benchmark error: {str(e)}")

# ---------- ENTRY POINT -----------------------------------------------------

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_benchmark()
    elif len(sys.argv) > 1 and sys.argv[1] == "interactive":
        demo = CompletePipelineDemo()
        demo.interactive_demo()
    else:
        run_complete_demo() 