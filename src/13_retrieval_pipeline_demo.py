"""
13_retrieval_pipeline_demo.py - Complete Retrieval Pipeline Demonstration

This script demonstrates the complete retrieval pipeline implementation,
showcasing all the different retrieval strategies and their capabilities.

Purpose:
- Demonstrate all retrieval strategies in one place
- Compare performance and results across strategies
- Provide interactive testing capabilities
- Show real-world usage examples
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
from load_embeddings import (
    EmbeddingLoader,
    create_index_from_latest_batch
)

# Import numbered modules using importlib
def import_module_from_file(module_name, file_path):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import the numbered modules
basic_query_engine = import_module_from_file("basic_query_engine", Path(__file__).parent / "10_basic_query_engine.py")
document_summary_retriever = import_module_from_file("document_summary_retriever", Path(__file__).parent / "11_document_summary_retriever.py")
recursive_retriever = import_module_from_file("recursive_retriever", Path(__file__).parent / "12_recursive_retriever.py")

# ---------- CONFIGURATION ---------------------------------------------------

# Load environment variables
load_dotenv(override=True)

# Demo queries for comprehensive testing
DEMO_QUERIES = [
    {
        "query": "What educational backgrounds are represented in the data?",
        "category": "Educational Analysis",
        "expected_strategy": "hierarchical"
    },
    {
        "query": "What are the salary ranges mentioned across profiles?",
        "category": "Compensation Analysis", 
        "expected_strategy": "document_summary"
    },
    {
        "query": "Which profiles mention specific assessment scores?",
        "category": "Assessment Analysis",
        "expected_strategy": "recursive"
    },
    {
        "query": "What types of companies and industries are represented?",
        "category": "Industry Analysis",
        "expected_strategy": "basic"
    },
    {
        "query": "How do work experience levels vary across profiles?",
        "category": "Experience Analysis",
        "expected_strategy": "hierarchical"
    }
]

# ---------- RETRIEVAL PIPELINE DEMO CLASS -----------------------------------

class RetrievalPipelineDemo:
    """Complete demonstration of the retrieval pipeline."""
    
    def __init__(self):
        """Initialize all retrieval strategies."""
        print("🚀 INITIALIZING RETRIEVAL PIPELINE DEMO")
        print("=" * 80)
        
        self.strategies = {}
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize all retrieval strategies."""
        try:
            # Strategy 1: Basic RAG Query Engine
            print("\n🔧 Initializing Basic RAG Query Engine...")
            basic_index = create_index_from_latest_batch(
                use_chunks=True,
                use_summaries=False,
                max_embeddings=None
            )
            self.strategies['basic'] = basic_query_engine.BasicRAGQueryEngine(basic_index)
            print("✅ Basic RAG Query Engine ready")
            
            # Strategy 2: Document Summary Retriever
            print("\n🔧 Initializing Document Summary Retriever...")
            self.strategies['document_summary'] = document_summary_retriever.create_document_summary_retriever_from_latest_batch()
            print("✅ Document Summary Retriever ready")
            
            # Strategy 3: Recursive Document Retriever
            print("\n🔧 Initializing Recursive Document Retriever...")
            self.strategies['recursive'] = recursive_retriever.create_recursive_retriever_from_latest_batch()
            print("✅ Recursive Document Retriever ready")
            
            # Strategy 4: Combined Index (all embedding types)
            print("\n🔧 Initializing Combined Index Strategy...")
            combined_index = create_index_from_latest_batch(
                use_chunks=True,
                use_summaries=True,
                use_indexnodes=True,
                max_embeddings=None
            )
            self.strategies['combined'] = basic_query_engine.BasicRAGQueryEngine(combined_index)
            print("✅ Combined Index Strategy ready")
            
            print(f"\n✅ All {len(self.strategies)} retrieval strategies initialized!")
            
        except Exception as e:
            print(f"\n❌ Error initializing strategies: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def query_all_strategies(
        self,
        query: str,
        show_details: bool = True
    ) -> Dict[str, Any]:
        """Query all retrieval strategies and compare results."""
        if show_details:
            print(f"\n🔍 QUERYING ALL STRATEGIES")
            print(f"Query: {query}")
            print("=" * 60)
        
        results = {}
        
        for strategy_name, strategy in self.strategies.items():
            if show_details:
                print(f"\n📊 Testing {strategy_name.upper()} strategy...")
            
            start_time = time.time()
            
            try:
                if strategy_name == 'document_summary':
                    # Use hierarchical retrieval for document summary
                    result = strategy.hierarchical_retrieve(query)
                    formatted_result = {
                        'query': query,
                        'response': f"Found {result['metadata']['total_docs_found']} relevant documents with {result['metadata']['total_chunks_found']} chunks",
                        'sources': result['relevant_chunks'][:5],  # Top 5 chunks
                        'metadata': result['metadata']
                    }
                elif strategy_name == 'recursive':
                    # Use recursive query
                    result = strategy.recursive_query(query, show_details=False)
                    formatted_result = result
                else:
                    # Use basic query for basic and combined strategies
                    result = strategy.query(query, show_sources=True, show_timing=False)
                    formatted_result = result
                
                end_time = time.time()
                formatted_result['metadata']['total_time'] = round(end_time - start_time, 2)
                
                results[strategy_name] = formatted_result
                
                if show_details:
                    print(f"  ✅ Completed in {formatted_result['metadata']['total_time']}s")
                    
            except Exception as e:
                if show_details:
                    print(f"  ❌ Error: {str(e)}")
                results[strategy_name] = {
                    'query': query,
                    'error': str(e),
                    'metadata': {'total_time': time.time() - start_time}
                }
        
        return results
    
    def compare_strategies(
        self,
        query: str
    ) -> None:
        """Compare all strategies for a single query."""
        print(f"\n🔬 STRATEGY COMPARISON")
        print(f"Query: {query}")
        print("=" * 80)
        
        results = self.query_all_strategies(query, show_details=False)
        
        # Display comparison table
        print(f"\n📊 RESULTS SUMMARY:")
        print("-" * 80)
        print(f"{'Strategy':<20} {'Time (s)':<10} {'Sources':<10} {'Response Length':<15}")
        print("-" * 80)
        
        for strategy_name, result in results.items():
            if 'error' not in result:
                time_taken = result['metadata'].get('total_time', 0)
                num_sources = len(result.get('sources', []))
                response_len = len(result.get('response', ''))
                
                print(f"{strategy_name:<20} {time_taken:<10.2f} {num_sources:<10} {response_len:<15}")
            else:
                print(f"{strategy_name:<20} {'ERROR':<10} {'-':<10} {'-':<15}")
        
        print("-" * 80)
        
        # Show detailed results
        for strategy_name, result in results.items():
            if 'error' not in result:
                print(f"\n📝 {strategy_name.upper()} RESPONSE:")
                print(f"{result['response'][:300]}...")
                
                if result.get('sources'):
                    print(f"\n📚 Top sources ({len(result['sources'])}):")
                    for i, source in enumerate(result['sources'][:3], 1):
                        if isinstance(source, dict):
                            preview = source.get('text_preview', source.get('text', ''))[:100]
                            print(f"  {i}. {preview}...")
            else:
                print(f"\n❌ {strategy_name.upper()} ERROR: {result['error']}")
    
    def run_comprehensive_demo(self):
        """Run comprehensive demonstration with all demo queries."""
        print("\n🎯 COMPREHENSIVE RETRIEVAL PIPELINE DEMONSTRATION")
        print("=" * 80)
        
        for i, demo_query in enumerate(DEMO_QUERIES, 1):
            print(f"\n🔍 Demo Query {i}/{len(DEMO_QUERIES)}")
            print(f"Category: {demo_query['category']}")
            print(f"Expected Best Strategy: {demo_query['expected_strategy']}")
            
            self.compare_strategies(demo_query['query'])
            
            if i < len(DEMO_QUERIES):
                print("\n" + "="*80)
        
        print("\n✅ Comprehensive demonstration complete!")
    
    def interactive_demo(self):
        """Run interactive demonstration."""
        print("\n💬 INTERACTIVE RETRIEVAL PIPELINE DEMO")
        print("=" * 80)
        print("Available strategies:")
        for strategy in self.strategies.keys():
            print(f"  • {strategy}")
        print("\nCommands:")
        print("  • 'compare <query>' - Compare all strategies")
        print("  • 'test <strategy> <query>' - Test specific strategy")
        print("  • 'demo' - Run comprehensive demo")
        print("  • 'quit' - Exit")
        
        while True:
            try:
                user_input = input("\n❓ Enter command: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'demo':
                    self.run_comprehensive_demo()
                
                elif user_input.startswith('compare '):
                    query = user_input[8:].strip()
                    if query:
                        self.compare_strategies(query)
                    else:
                        print("❌ Please provide a query after 'compare'")
                
                elif user_input.startswith('test '):
                    parts = user_input[5:].strip().split(' ', 1)
                    if len(parts) == 2:
                        strategy, query = parts
                        if strategy in self.strategies:
                            results = self.query_all_strategies(query, show_details=False)
                            if strategy in results:
                                result = results[strategy]
                                print(f"\n📝 {strategy.upper()} RESULT:")
                                print(f"Response: {result.get('response', 'No response')}")
                                print(f"Time: {result['metadata'].get('total_time', 0):.2f}s")
                        else:
                            print(f"❌ Unknown strategy: {strategy}")
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

def demonstrate_retrieval_pipeline():
    """Main demonstration function."""
    try:
        # Initialize demo
        demo = RetrievalPipelineDemo()
        
        # Run comprehensive demo
        demo.run_comprehensive_demo()
        
        # Offer interactive mode
        print("\n🎮 Would you like to try interactive mode? (y/n)")
        response = input().strip().lower()
        if response in ['y', 'yes']:
            demo.interactive_demo()
        
    except Exception as e:
        print(f"\n❌ Demo error: {str(e)}")
        import traceback
        traceback.print_exc()

def quick_strategy_test():
    """Quick test of all strategies with a single query."""
    print("⚡ QUICK STRATEGY TEST")
    print("=" * 80)
    
    try:
        demo = RetrievalPipelineDemo()
        test_query = "What educational qualifications are mentioned in the profiles?"
        demo.compare_strategies(test_query)
        
    except Exception as e:
        print(f"❌ Quick test error: {str(e)}")

# ---------- ENTRY POINT -----------------------------------------------------

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_strategy_test()
    elif len(sys.argv) > 1 and sys.argv[1] == "interactive":
        demo = RetrievalPipelineDemo()
        demo.interactive_demo()
    else:
        demonstrate_retrieval_pipeline() 