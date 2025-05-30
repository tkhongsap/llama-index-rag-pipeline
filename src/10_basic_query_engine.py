"""
10_basic_query_engine.py - Create basic query engine for testing

This script creates a simple query engine using loaded embeddings to test
basic RAG functionality with top-k retrieval and response synthesis.

Purpose:
- Build basic VectorStoreIndex query engine
- Implement simple top-k retrieval
- Add basic response synthesis
- Test with sample queries
"""

import os
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import LlamaIndex components
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    QueryBundle,
    Response
)
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Import our loader utilities
from load_embeddings import (
    EmbeddingLoader,
    IndexReconstructor,
    create_index_from_latest_batch,
    validate_loaded_embeddings
)

# ---------- CONFIGURATION ---------------------------------------------------

# Load environment variables
load_dotenv(override=True)

# Query engine settings
DEFAULT_TOP_K = 5
DEFAULT_RESPONSE_MODE = ResponseMode.TREE_SUMMARIZE
DEFAULT_TEMPERATURE = 0.0

# Sample queries for testing
SAMPLE_QUERIES = [
    "What are the main topics discussed in these documents?",
    "Explain the key concepts mentioned in the content.",
    "What are the most important findings or conclusions?",
    "How do the documents relate to each other?",
    "Summarize the main ideas in simple terms."
]

# ---------- BASIC QUERY ENGINE CLASS ----------------------------------------

class BasicRAGQueryEngine:
    """Basic RAG query engine with simple top-k retrieval."""
    
    def __init__(
        self,
        index: VectorStoreIndex,
        top_k: int = DEFAULT_TOP_K,
        response_mode: ResponseMode = DEFAULT_RESPONSE_MODE,
        temperature: float = DEFAULT_TEMPERATURE
    ):
        """Initialize basic query engine."""
        self.index = index
        self.top_k = top_k
        self.response_mode = response_mode
        self.temperature = temperature
        self._setup_query_engine()
    
    def _setup_query_engine(self):
        """Set up the query engine with specified parameters."""
        # Create query engine
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=self.top_k,
            response_mode=self.response_mode,
            temperature=self.temperature,
            verbose=True  # Show retrieval details
        )
    
    def query(
        self, 
        query_text: str,
        show_sources: bool = True,
        show_timing: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a query and return structured results.
        
        Args:
            query_text: The question to ask
            show_sources: Include source nodes in results
            show_timing: Include timing information
            
        Returns:
            Dictionary with response, sources, and metadata
        """
        start_time = time.time()
        
        # Execute query
        response = self.query_engine.query(query_text)
        
        # Calculate timing
        end_time = time.time()
        query_time = end_time - start_time
        
        # Extract results
        result = {
            "query": query_text,
            "response": str(response),
            "metadata": {
                "top_k": self.top_k,
                "response_mode": str(self.response_mode),
                "temperature": self.temperature
            }
        }
        
        # Add sources if requested
        if show_sources and hasattr(response, 'source_nodes'):
            sources = []
            for i, node in enumerate(response.source_nodes):
                source_info = {
                    "rank": i + 1,
                    "score": node.score if hasattr(node, 'score') else None,
                    "text_preview": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                    "metadata": node.metadata if hasattr(node, 'metadata') else {}
                }
                sources.append(source_info)
            result["sources"] = sources
        
        # Add timing if requested
        if show_timing:
            result["timing"] = {
                "query_time_seconds": round(query_time, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        
        return result
    
    def batch_query(
        self, 
        queries: List[str],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """Execute multiple queries and return all results."""
        results = []
        
        for i, query in enumerate(queries):
            if show_progress:
                print(f"\n🔄 Processing query {i+1}/{len(queries)}: {query[:50]}...")
            
            try:
                result = self.query(query, show_sources=True, show_timing=True)
                results.append(result)
                
                if show_progress:
                    print(f"✅ Query completed in {result['timing']['query_time_seconds']}s")
            
            except Exception as e:
                print(f"❌ Error processing query: {str(e)}")
                results.append({
                    "query": query,
                    "error": str(e),
                    "response": None
                })
        
        return results

# ---------- QUERY ENGINE TESTING FUNCTIONS ----------------------------------

def test_basic_retrieval(engine: BasicRAGQueryEngine):
    """Test basic retrieval functionality."""
    print("\n🧪 TESTING BASIC RETRIEVAL")
    print("=" * 60)
    
    test_query = "What is the main topic?"
    print(f"Query: {test_query}")
    
    result = engine.query(test_query)
    
    print(f"\n📝 Response: {result['response'][:300]}...")
    print(f"\n📊 Retrieved {len(result.get('sources', []))} sources")
    
    if result.get('sources'):
        print("\nTop sources:")
        for source in result['sources'][:3]:
            score_str = f"{source['score']:.3f}" if source['score'] is not None else 'N/A'
            print(f"  • Rank {source['rank']} (score: {score_str})")
            print(f"    Preview: {source['text_preview'][:100]}...")

def compare_retrieval_settings(index: VectorStoreIndex):
    """Compare different retrieval settings."""
    print("\n🔬 COMPARING RETRIEVAL SETTINGS")
    print("=" * 60)
    
    test_query = "What are the key findings?"
    settings = [
        {"top_k": 3, "response_mode": ResponseMode.COMPACT},
        {"top_k": 5, "response_mode": ResponseMode.TREE_SUMMARIZE},
        {"top_k": 10, "response_mode": ResponseMode.REFINE}
    ]
    
    for setting in settings:
        print(f"\n📊 Testing with top_k={setting['top_k']}, mode={setting['response_mode']}...")
        
        engine = BasicRAGQueryEngine(
            index=index,
            top_k=setting['top_k'],
            response_mode=setting['response_mode']
        )
        
        result = engine.query(test_query, show_timing=True)
        
        print(f"  • Response length: {len(result['response'])} chars")
        print(f"  • Query time: {result['timing']['query_time_seconds']}s")
        print(f"  • Sources used: {len(result.get('sources', []))}")

def analyze_retrieval_quality(engine: BasicRAGQueryEngine, queries: List[str]):
    """Analyze retrieval quality across multiple queries."""
    print("\n📈 ANALYZING RETRIEVAL QUALITY")
    print("=" * 60)
    
    results = engine.batch_query(queries, show_progress=True)
    
    # Calculate statistics
    total_time = sum(r['timing']['query_time_seconds'] for r in results if 'timing' in r)
    avg_time = total_time / len(results) if results else 0
    avg_response_length = sum(len(r['response']) for r in results if r.get('response')) / len(results)
    
    print(f"\n📊 Summary Statistics:")
    print(f"  • Total queries: {len(queries)}")
    print(f"  • Average query time: {avg_time:.2f}s")
    print(f"  • Average response length: {avg_response_length:.0f} chars")
    
    # Show sample responses
    print("\n📝 Sample Responses:")
    for i, result in enumerate(results[:3]):
        if result.get('response'):
            print(f"\nQuery {i+1}: {result['query']}")
            print(f"Response: {result['response'][:200]}...")

# ---------- DEMONSTRATION FUNCTIONS -----------------------------------------

def demonstrate_basic_query_engine():
    """Demonstrate basic query engine functionality."""
    print("🚀 BASIC QUERY ENGINE DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Load embeddings and create index
        print("\n🔄 Loading embeddings and creating index...")
        index = create_index_from_latest_batch(
            use_chunks=True,
            use_summaries=False,
            use_indexnodes=False,
            max_embeddings=100  # Limit for demo
        )
        print("✅ Index created successfully")
        
        # Create basic query engine
        print("\n🔧 Creating basic query engine...")
        engine = BasicRAGQueryEngine(
            index=index,
            top_k=DEFAULT_TOP_K,
            response_mode=DEFAULT_RESPONSE_MODE
        )
        print("✅ Query engine ready")
        
        # Test basic retrieval
        test_basic_retrieval(engine)
        
        # Compare settings
        compare_retrieval_settings(index)
        
        # Analyze quality
        analyze_retrieval_quality(engine, SAMPLE_QUERIES[:3])
        
        print("\n✅ Basic query engine demonstration complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

def interactive_query_session():
    """Run an interactive query session."""
    print("💬 INTERACTIVE QUERY SESSION")
    print("=" * 80)
    print("🎯 You can now ask questions about your documents!")
    print("\n📋 Available commands:")
    print("  • Type any question to get an answer")
    print("  • 'quit' or 'exit' - End the session")
    print("  • 'help' - Show available commands")
    print("  • 'settings' - Show current engine settings")
    print("  • 'top_k=N' - Change number of retrieved documents (e.g., top_k=10)")
    print("\n💡 Example questions:")
    print("  • What are the main topics in these documents?")
    print("  • Summarize the key findings")
    print("  • What educational qualifications are mentioned?")
    
    try:
        # Create index and engine
        print("\n🔄 Setting up query engine...")
        print("   Loading embeddings from latest batch...")
        index = create_index_from_latest_batch(
            use_chunks=True,
            use_summaries=True,  # Include summaries for better context
            max_embeddings=None  # Use all available embeddings
        )
        
        engine = BasicRAGQueryEngine(index=index)
        print("✅ Query engine ready!")
        print(f"📊 Loaded index with {len(index.docstore.docs)} documents")
        print(f"⚙️  Current settings: top_k={engine.top_k}, response_mode={engine.response_mode}")
        
        while True:
            # Get user input
            query = input("\n❓ Enter your question: ").strip()
            
            # Check for commands
            if query.lower() in ['quit', 'exit']:
                print("👋 Thanks for using the query engine! Goodbye!")
                break
            
            elif query.lower() == 'help':
                print("\n📋 Available commands:")
                print("  • quit/exit - End session")
                print("  • help - Show this help")
                print("  • settings - Show current settings")
                print("  • top_k=N - Change number of retrieved documents")
                print("  • clear - Clear screen (if supported)")
                continue
                
            elif query.lower() == 'settings':
                print(f"\n⚙️ Current settings:")
                print(f"  • Top K: {engine.top_k}")
                print(f"  • Response Mode: {engine.response_mode}")
                print(f"  • Temperature: {engine.temperature}")
                print(f"  • Documents in index: {len(index.docstore.docs)}")
                continue
                
            elif query.startswith('top_k='):
                try:
                    new_k = int(query.split('=')[1])
                    if new_k > 0:
                        engine.top_k = new_k
                        engine._setup_query_engine()
                        print(f"✅ Updated top_k to {new_k}")
                    else:
                        print("❌ top_k must be greater than 0")
                except:
                    print("❌ Invalid format. Use: top_k=5")
                continue
                
            elif query.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            
            # Execute query
            if query:
                print("\n🔄 Processing your question...")
                start_time = time.time()
                result = engine.query(query)
                
                print(f"\n📝 Answer:")
                print("-" * 50)
                print(result['response'])
                print("-" * 50)
                
                # Show sources
                if result.get('sources'):
                    print(f"\n📚 Sources used ({len(result['sources'])} documents retrieved):")
                    for i, source in enumerate(result['sources'][:3], 1):
                        score_str = f" (relevance: {source['score']:.3f})" if source['score'] is not None else ""
                        print(f"\n  📄 Source {i}{score_str}:")
                        print(f"     {source['text_preview'][:150]}...")
                
                print(f"\n⏱️ Query completed in {result['timing']['query_time_seconds']}s")
            else:
                print("❌ Please enter a question or command.")
    
    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error setting up query engine: {str(e)}")
        print("💡 Make sure you have embeddings available in the data directory.")

# ---------- UTILITY FUNCTIONS -----------------------------------------------

def create_optimized_query_engine(
    use_all_embeddings: bool = True,
    custom_top_k: Optional[int] = None,
    custom_temperature: Optional[float] = None
) -> BasicRAGQueryEngine:
    """
    Create an optimized query engine with custom settings.
    
    This is a utility function for use in other scripts.
    """
    # Load embeddings
    index = create_index_from_latest_batch(
        use_chunks=True,
        use_summaries=use_all_embeddings,
        use_indexnodes=use_all_embeddings,
        max_embeddings=None
    )
    
    # Create engine with custom settings
    engine = BasicRAGQueryEngine(
        index=index,
        top_k=custom_top_k or DEFAULT_TOP_K,
        temperature=custom_temperature or DEFAULT_TEMPERATURE
    )
    
    return engine

# ---------- ENTRY POINT -----------------------------------------------------

def main_menu():
    """Display main menu and handle user choice."""
    print("🚀 BASIC QUERY ENGINE")
    print("=" * 50)
    print("\nChoose an option:")
    print("1. Interactive Mode - Type your own questions")
    print("2. Sample Queries - Run demonstration with predefined queries")
    print("3. Exit")
    
    while True:
        try:
            choice = input("\n❓ Enter your choice (1, 2, or 3): ").strip()
            
            if choice == "1":
                print("\n🎯 Starting Interactive Mode...")
                interactive_query_session()
                break
            elif choice == "2":
                print("\n🎯 Running Sample Queries Demonstration...")
                demonstrate_basic_query_engine()
                break
            elif choice == "3":
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    import sys
    
    # Check for command line arguments for backward compatibility
    if len(sys.argv) > 1:
        if sys.argv[1] == "interactive":
            interactive_query_session()
        elif sys.argv[1] == "demo":
            demonstrate_basic_query_engine()
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Available arguments: 'interactive', 'demo'")
    else:
        # Default behavior: show menu
        main_menu() 