"""
11_document_summary_retriever.py - Document-level retrieval using summaries

This script implements document-level retrieval using summaries as a first-pass
filter before retrieving detailed chunks. This provides better context awareness
and more relevant results for document-level questions.

Purpose:
- Create DocumentSummaryIndex from saved data
- Implement summary-first retrieval strategy
- Test document-level vs chunk-level retrieval
- Compare retrieval strategies
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import LlamaIndex components
from llama_index.core import (
    VectorStoreIndex,
    DocumentSummaryIndex,
    Settings,
    Document,
    StorageContext
)
from llama_index.core.schema import TextNode, IndexNode, NodeWithScore
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Import our utilities
from load_embeddings import (
    EmbeddingLoader,
    IndexReconstructor,
    validate_loaded_embeddings
)

# ---------- CONFIGURATION ---------------------------------------------------

# Load environment variables
load_dotenv(override=True)

# Retrieval settings
DEFAULT_TOP_K = 5
DEFAULT_SUMMARY_TOP_K = 3
DEFAULT_CHUNK_TOP_K = 8

# Sample queries for testing document-level retrieval
DOCUMENT_LEVEL_QUERIES = [
    "What documents discuss educational background and career progression?",
    "Which profiles mention specific assessment scores or evaluations?",
    "What are the different types of compensation mentioned across documents?",
    "Which documents contain information about demographic details?",
    "What documents discuss work experience and career achievements?"
]

# ---------- DOCUMENT SUMMARY RETRIEVER CLASS --------------------------------

class DocumentSummaryRetriever:
    """Document-level retriever using summaries for initial filtering."""
    
    def __init__(
        self,
        summary_embeddings: List[Dict[str, Any]],
        chunk_embeddings: List[Dict[str, Any]],
        api_key: Optional[str] = None
    ):
        """Initialize document summary retriever."""
        self.summary_embeddings = summary_embeddings
        self.chunk_embeddings = chunk_embeddings
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        # Setup LLM and embedding model
        self._setup_models()
        
        # Create indices
        self._create_indices()
        
        # Setup response synthesizer
        self._setup_response_synthesizer()
    
    def _setup_models(self):
        """Setup LLM and embedding models."""
        Settings.llm = OpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=self.api_key
        )
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=self.api_key
        )
    
    def _setup_response_synthesizer(self):
        """Setup response synthesizer for generating answers."""
        self.response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.TREE_SUMMARIZE,
            use_async=False
        )
    
    def _create_indices(self):
        """Create summary and chunk indices."""
        print("🔄 Creating document summary index...")
        
        # Create summary index
        reconstructor = IndexReconstructor(self.api_key)
        self.summary_index = reconstructor.create_vector_index_from_embeddings(
            self.summary_embeddings,
            show_progress=True
        )
        
        # Create chunk index
        print("🔄 Creating chunk index...")
        self.chunk_index = reconstructor.create_vector_index_from_embeddings(
            self.chunk_embeddings,
            show_progress=True
        )
        
        # Create document mapping (summary -> chunks)
        self._create_document_mapping()
    
    def _create_document_mapping(self):
        """Create mapping from document IDs to their chunks."""
        self.doc_to_chunks = {}
        
        for chunk in self.chunk_embeddings:
            doc_id = chunk.get('doc_id')
            if doc_id:
                if doc_id not in self.doc_to_chunks:
                    self.doc_to_chunks[doc_id] = []
                self.doc_to_chunks[doc_id].append(chunk)
        
        print(f"📊 Created mapping for {len(self.doc_to_chunks)} documents")
    
    def retrieve_documents_by_summary(
        self,
        query: str,
        top_k: int = DEFAULT_SUMMARY_TOP_K
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using summary-level search."""
        # Query summary index
        summary_retriever = self.summary_index.as_retriever(
            similarity_top_k=top_k
        )
        
        summary_nodes = summary_retriever.retrieve(query)
        
        # Extract document information
        relevant_docs = []
        for node in summary_nodes:
            doc_info = {
                'node_id': node.node.id_,
                'score': node.score,
                'summary_text': node.node.text,
                'metadata': node.node.metadata,
                'doc_id': node.node.metadata.get('doc_id'),
                'doc_title': node.node.metadata.get('doc_title', 'Unknown')
            }
            relevant_docs.append(doc_info)
        
        return relevant_docs
    
    def retrieve_chunks_for_documents(
        self,
        doc_ids: List[str],
        query: str,
        chunks_per_doc: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve specific chunks from identified documents."""
        relevant_chunks = []
        
        for doc_id in doc_ids:
            if doc_id in self.doc_to_chunks:
                doc_chunks = self.doc_to_chunks[doc_id]
                
                # Create temporary index for this document's chunks
                doc_chunk_embeddings = [
                    chunk for chunk in self.chunk_embeddings
                    if chunk.get('doc_id') == doc_id
                ]
                
                if doc_chunk_embeddings:
                    # Create mini-index for this document
                    reconstructor = IndexReconstructor(self.api_key)
                    doc_index = reconstructor.create_vector_index_from_embeddings(
                        doc_chunk_embeddings,
                        show_progress=False
                    )
                    
                    # Retrieve best chunks from this document
                    doc_retriever = doc_index.as_retriever(
                        similarity_top_k=min(chunks_per_doc, len(doc_chunk_embeddings))
                    )
                    
                    doc_nodes = doc_retriever.retrieve(query)
                    
                    for node in doc_nodes:
                        chunk_info = {
                            'node_id': node.node.id_,
                            'score': node.score,
                            'text': node.node.text,
                            'metadata': node.node.metadata,
                            'doc_id': doc_id,
                            'doc_title': node.node.metadata.get('doc_title', 'Unknown')
                        }
                        relevant_chunks.append(chunk_info)
        
        return relevant_chunks
    
    def synthesize_response(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Synthesize a response from retrieved chunks.
        
        Args:
            query: The original query
            retrieved_chunks: List of retrieved chunk dictionaries
            
        Returns:
            Synthesized response string
        """
        if not retrieved_chunks:
            return "No relevant information found to answer your question."
        
        # Convert chunks to NodeWithScore objects (what the synthesizer expects)
        nodes_with_scores = []
        for chunk in retrieved_chunks:
            text_node = TextNode(
                text=chunk['text'],
                metadata=chunk.get('metadata', {}),
                id_=chunk.get('node_id', '')
            )
            
            # Create NodeWithScore object
            node_with_score = NodeWithScore(
                node=text_node,
                score=chunk.get('score', 0.0)
            )
            nodes_with_scores.append(node_with_score)
        
        # Synthesize response
        try:
            response = self.response_synthesizer.synthesize(query, nodes_with_scores)
            return str(response)
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def hierarchical_retrieve_with_response(
        self,
        query: str,
        summary_top_k: int = DEFAULT_SUMMARY_TOP_K,
        chunks_per_doc: int = 3
    ) -> Dict[str, Any]:
        """
        Perform hierarchical retrieval and generate a response.
        
        Returns:
            Dictionary with retrieval results, synthesized response, and metadata
        """
        # Get retrieval results
        retrieval_result = self.hierarchical_retrieve(
            query, summary_top_k, chunks_per_doc
        )
        
        # Synthesize response from retrieved chunks
        response = self.synthesize_response(query, retrieval_result['relevant_chunks'])
        
        # Add response to results
        retrieval_result['synthesized_response'] = response
        
        return retrieval_result

    def hierarchical_retrieve(
        self,
        query: str,
        summary_top_k: int = DEFAULT_SUMMARY_TOP_K,
        chunks_per_doc: int = 3
    ) -> Dict[str, Any]:
        """
        Perform hierarchical retrieval: summaries first, then chunks.
        
        Returns:
            Dictionary with summary results, chunk results, and metadata
        """
        start_time = time.time()
        
        # Step 1: Retrieve relevant documents using summaries
        print(f"🔍 Step 1: Retrieving top {summary_top_k} documents by summary...")
        relevant_docs = self.retrieve_documents_by_summary(query, summary_top_k)
        
        # Step 2: Get detailed chunks from relevant documents
        doc_ids = [doc['doc_id'] for doc in relevant_docs if doc['doc_id']]
        print(f"🔍 Step 2: Retrieving chunks from {len(doc_ids)} documents...")
        
        relevant_chunks = self.retrieve_chunks_for_documents(
            doc_ids, query, chunks_per_doc
        )
        
        end_time = time.time()
        
        return {
            'query': query,
            'relevant_documents': relevant_docs,
            'relevant_chunks': relevant_chunks,
            'metadata': {
                'summary_top_k': summary_top_k,
                'chunks_per_doc': chunks_per_doc,
                'total_docs_found': len(relevant_docs),
                'total_chunks_found': len(relevant_chunks),
                'retrieval_time': round(end_time - start_time, 2)
            }
        }
    
    def compare_retrieval_strategies(
        self,
        query: str
    ) -> Dict[str, Any]:
        """Compare different retrieval strategies."""
        results = {}
        
        # Strategy 1: Direct chunk retrieval
        print("📊 Testing direct chunk retrieval...")
        start_time = time.time()
        chunk_retriever = self.chunk_index.as_retriever(similarity_top_k=DEFAULT_TOP_K)
        chunk_nodes = chunk_retriever.retrieve(query)
        chunk_time = time.time() - start_time
        
        results['direct_chunks'] = {
            'results': [
                {
                    'score': node.score,
                    'text_preview': node.node.text[:200] + "...",
                    'doc_title': node.node.metadata.get('doc_title', 'Unknown')
                }
                for node in chunk_nodes
            ],
            'count': len(chunk_nodes),
            'time': round(chunk_time, 2)
        }
        
        # Strategy 2: Summary-first hierarchical retrieval
        print("📊 Testing summary-first hierarchical retrieval...")
        hierarchical_result = self.hierarchical_retrieve(query)
        
        results['hierarchical'] = {
            'documents': hierarchical_result['relevant_documents'],
            'chunks': [
                {
                    'score': chunk['score'],
                    'text_preview': chunk['text'][:200] + "...",
                    'doc_title': chunk['doc_title']
                }
                for chunk in hierarchical_result['relevant_chunks']
            ],
            'doc_count': hierarchical_result['metadata']['total_docs_found'],
            'chunk_count': hierarchical_result['metadata']['total_chunks_found'],
            'time': hierarchical_result['metadata']['retrieval_time']
        }
        
        # Strategy 3: Summary-only retrieval
        print("📊 Testing summary-only retrieval...")
        start_time = time.time()
        summary_docs = self.retrieve_documents_by_summary(query, DEFAULT_TOP_K)
        summary_time = time.time() - start_time
        
        results['summary_only'] = {
            'results': [
                {
                    'score': doc['score'],
                    'text_preview': doc['summary_text'][:200] + "...",
                    'doc_title': doc['doc_title']
                }
                for doc in summary_docs
            ],
            'count': len(summary_docs),
            'time': round(summary_time, 2)
        }
        
        return results

# ---------- INTERACTIVE FUNCTIONS -------------------------------------------

def interactive_query_session():
    """Run an interactive query session with document summary retrieval."""
    print("💬 INTERACTIVE DOCUMENT SUMMARY RETRIEVAL SESSION")
    print("=" * 80)
    print("🎯 You can now ask document-level questions!")
    print("\n📋 Available commands:")
    print("  • Type any question to get hierarchical retrieval results")
    print("  • 'quit' or 'exit' - End the session")
    print("  • 'help' - Show available commands")
    print("  • 'settings' - Show current retriever settings")
    print("  • 'summary_k=N' - Change number of documents retrieved by summary (e.g., summary_k=5)")
    print("  • 'chunks_per_doc=N' - Change chunks per document (e.g., chunks_per_doc=4)")
    print("  • 'compare' - Compare retrieval strategies for last query")
    print("\n💡 Example questions:")
    print("  • What documents discuss educational background and career progression?")
    print("  • Which profiles mention specific assessment scores or evaluations?")
    print("  • What are the different types of compensation mentioned across documents?")
    print("  • Which documents contain information about demographic details?")
    
    try:
        # Create document summary retriever
        print("\n🔄 Setting up document summary retriever...")
        print("   Loading embeddings from latest batch...")
        retriever = create_document_summary_retriever_from_latest_batch()
        print("✅ Document summary retriever ready!")
        print(f"📊 Loaded {len(retriever.summary_embeddings)} summaries and {len(retriever.chunk_embeddings)} chunks")
        print(f"📄 Mapped {len(retriever.doc_to_chunks)} documents")
        
        # Current settings
        current_summary_k = DEFAULT_SUMMARY_TOP_K
        current_chunks_per_doc = 3
        last_query = None
        
        print(f"⚙️  Current settings: summary_k={current_summary_k}, chunks_per_doc={current_chunks_per_doc}")
        
        while True:
            # Get user input
            query = input("\n❓ Enter your question: ").strip()
            
            # Check for commands
            if query.lower() in ['quit', 'exit']:
                print("👋 Thanks for using the document summary retriever! Goodbye!")
                break
            
            elif query.lower() == 'help':
                print("\n📋 Available commands:")
                print("  • quit/exit - End session")
                print("  • help - Show this help")
                print("  • settings - Show current settings")
                print("  • summary_k=N - Change number of documents retrieved by summary")
                print("  • chunks_per_doc=N - Change chunks per document")
                print("  • compare - Compare retrieval strategies for last query")
                print("  • clear - Clear screen (if supported)")
                continue
                
            elif query.lower() == 'settings':
                print(f"\n⚙️ Current settings:")
                print(f"  • Summary Top K: {current_summary_k}")
                print(f"  • Chunks per Document: {current_chunks_per_doc}")
                print(f"  • Total Summaries: {len(retriever.summary_embeddings)}")
                print(f"  • Total Chunks: {len(retriever.chunk_embeddings)}")
                print(f"  • Documents Mapped: {len(retriever.doc_to_chunks)}")
                continue
                
            elif query.startswith('summary_k='):
                try:
                    new_k = int(query.split('=')[1])
                    if new_k > 0:
                        current_summary_k = new_k
                        print(f"✅ Updated summary_k to {new_k}")
                    else:
                        print("❌ summary_k must be greater than 0")
                except:
                    print("❌ Invalid format. Use: summary_k=5")
                continue
                
            elif query.startswith('chunks_per_doc='):
                try:
                    new_chunks = int(query.split('=')[1])
                    if new_chunks > 0:
                        current_chunks_per_doc = new_chunks
                        print(f"✅ Updated chunks_per_doc to {new_chunks}")
                    else:
                        print("❌ chunks_per_doc must be greater than 0")
                except:
                    print("❌ Invalid format. Use: chunks_per_doc=4")
                continue
                
            elif query.lower() == 'compare':
                if last_query:
                    print(f"\n🔬 Comparing retrieval strategies for: '{last_query}'")
                    comparison = retriever.compare_retrieval_strategies(last_query)
                    
                    print(f"\n📊 COMPARISON RESULTS:")
                    print("-" * 40)
                    
                    for strategy, results in comparison.items():
                        print(f"\n{strategy.upper().replace('_', ' ')}:")
                        print(f"  • Results: {results.get('count', results.get('chunk_count', 0))}")
                        print(f"  • Time: {results['time']}s")
                        
                        if 'results' in results and results['results']:
                            top_result = results['results'][0]
                            print(f"  • Top score: {top_result['score']:.3f}")
                            print(f"  • Preview: {top_result['text_preview'][:100]}...")
                        elif 'chunks' in results:
                            print(f"  • Documents found: {results['doc_count']}")
                            if results['chunks']:
                                top_chunk = results['chunks'][0]
                                print(f"  • Top chunk score: {top_chunk['score']:.3f}")
                else:
                    print("❌ No previous query to compare. Ask a question first!")
                continue
                
            elif query.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            
            # Execute hierarchical retrieval
            if query:
                print("\n🔄 Processing your question with hierarchical retrieval...")
                last_query = query
                
                result = retriever.hierarchical_retrieve_with_response(
                    query,
                    summary_top_k=current_summary_k,
                    chunks_per_doc=current_chunks_per_doc
                )
                
                # Show synthesized response first
                print(f"\n💬 ANSWER:")
                print("=" * 50)
                print(result['synthesized_response'])
                print("=" * 50)
                
                print(f"\n📋 DOCUMENT-LEVEL RESULTS:")
                print("-" * 50)
                print(f"Found {result['metadata']['total_docs_found']} relevant documents:")
                
                for i, doc in enumerate(result['relevant_documents'], 1):
                    print(f"\n  📄 Document {i}: {doc['doc_title']}")
                    print(f"     Relevance Score: {doc['score']:.3f}")
                    print(f"     Summary Preview: {doc['summary_text'][:150]}...")
                
                print(f"\n📄 DETAILED CHUNKS:")
                print("-" * 50)
                print(f"Retrieved {result['metadata']['total_chunks_found']} chunks from relevant documents:")
                
                for i, chunk in enumerate(result['relevant_chunks'], 1):
                    print(f"\n  📝 Chunk {i} from '{chunk['doc_title']}':")
                    print(f"     Relevance Score: {chunk['score']:.3f}")
                    print(f"     Content: {chunk['text'][:200]}...")
                
                print(f"\n⏱️ Retrieval completed in {result['metadata']['retrieval_time']}s")
                print(f"💡 Tip: Type 'compare' to see how this compares to other retrieval strategies")
                
            else:
                print("❌ Please enter a question or command.")
    
    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error setting up document summary retriever: {str(e)}")
        print("💡 Make sure you have embeddings available in the data directory.")
        import traceback
        traceback.print_exc()

def sample_query_demonstration():
    """Run demonstration with predefined sample queries."""
    print("🧪 SAMPLE QUERY DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Create retriever
        print("🔄 Setting up document summary retriever...")
        retriever = create_document_summary_retriever_from_latest_batch()
        print("✅ Retriever ready!")
        
        # Test with sample queries
        for i, query in enumerate(DOCUMENT_LEVEL_QUERIES, 1):
            print(f"\n🔍 Sample Query {i}: {query}")
            print("-" * 60)
            
            # Perform hierarchical retrieval
            result = retriever.hierarchical_retrieve(query)
            
            print(f"📋 Found {result['metadata']['total_docs_found']} relevant documents:")
            for doc in result['relevant_documents']:
                print(f"  • {doc['doc_title']} (score: {doc['score']:.3f})")
            
            print(f"\n📄 Retrieved {result['metadata']['total_chunks_found']} chunks:")
            for chunk in result['relevant_chunks'][:3]:  # Show top 3
                print(f"  • {chunk['doc_title']}: {chunk['text'][:100]}...")
            
            print(f"\n⏱️ Retrieval time: {result['metadata']['retrieval_time']}s")
            
            # Pause between queries for readability
            if i < len(DOCUMENT_LEVEL_QUERIES):
                input("\nPress Enter to continue to next query...")
        
        # Show strategy comparison for last query
        print(f"\n🔬 STRATEGY COMPARISON for last query:")
        print("=" * 60)
        comparison = retriever.compare_retrieval_strategies(DOCUMENT_LEVEL_QUERIES[-1])
        
        for strategy, results in comparison.items():
            print(f"\n{strategy.upper().replace('_', ' ')}:")
            print(f"  • Results: {results.get('count', results.get('chunk_count', 0))}")
            print(f"  • Time: {results['time']}s")
        
        print("\n✅ Sample query demonstration complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

def main_menu():
    """Display main menu and handle user choice."""
    print("🚀 DOCUMENT SUMMARY RETRIEVER")
    print("=" * 50)
    print("\nChoose an option:")
    print("1. Interactive Mode - Type your own document-level questions")
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
                sample_query_demonstration()
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

# ---------- TESTING AND DEMONSTRATION FUNCTIONS ----------------------------

def test_document_summary_retrieval():
    """Test document summary retrieval functionality."""
    print("🧪 TESTING DOCUMENT SUMMARY RETRIEVAL")
    print("=" * 80)
    
    try:
        # Load embeddings
        print("🔄 Loading embeddings...")
        loader = EmbeddingLoader(Path("data/embedding"))
        latest_batch = loader.get_latest_batch()
        
        if not latest_batch:
            print("❌ No embedding batches found!")
            return
        
        # Load summary and chunk embeddings from all sub-batches
        all_embeddings = loader.load_all_embeddings_from_batch(latest_batch)
        
        # Combine all summaries and chunks
        all_summaries = []
        all_chunks = []
        
        for sub_batch, emb_types in all_embeddings.items():
            all_summaries.extend(emb_types.get('summaries', []))
            all_chunks.extend(emb_types.get('chunks', []))
        
        print(f"📊 Loaded {len(all_summaries)} summaries and {len(all_chunks)} chunks")
        
        # Create document summary retriever
        print("🔧 Creating document summary retriever...")
        retriever = DocumentSummaryRetriever(
            summary_embeddings=all_summaries,
            chunk_embeddings=all_chunks
        )
        
        # Test with sample queries
        for i, query in enumerate(DOCUMENT_LEVEL_QUERIES[:3], 1):
            print(f"\n🔍 Query {i}: {query}")
            print("-" * 60)
            
            # Perform hierarchical retrieval
            result = retriever.hierarchical_retrieve(query)
            
            print(f"📋 Found {result['metadata']['total_docs_found']} relevant documents:")
            for doc in result['relevant_documents']:
                print(f"  • {doc['doc_title']} (score: {doc['score']:.3f})")
            
            print(f"\n📄 Retrieved {result['metadata']['total_chunks_found']} chunks:")
            for chunk in result['relevant_chunks'][:3]:  # Show top 3
                print(f"  • {chunk['doc_title']}: {chunk['text'][:100]}...")
            
            print(f"\n⏱️ Retrieval time: {result['metadata']['retrieval_time']}s")
        
        print("\n✅ Document summary retrieval test complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

def compare_retrieval_strategies():
    """Compare different retrieval strategies."""
    print("\n🔬 COMPARING RETRIEVAL STRATEGIES")
    print("=" * 80)
    
    try:
        # Load embeddings
        loader = EmbeddingLoader(Path("data/embedding"))
        latest_batch = loader.get_latest_batch()
        all_embeddings = loader.load_all_embeddings_from_batch(latest_batch)
        
        # Combine embeddings
        all_summaries = []
        all_chunks = []
        for sub_batch, emb_types in all_embeddings.items():
            all_summaries.extend(emb_types.get('summaries', []))
            all_chunks.extend(emb_types.get('chunks', []))
        
        # Create retriever
        retriever = DocumentSummaryRetriever(
            summary_embeddings=all_summaries,
            chunk_embeddings=all_chunks
        )
        
        # Test query
        test_query = "What information is available about educational background?"
        print(f"🔍 Test query: {test_query}")
        
        # Compare strategies
        comparison = retriever.compare_retrieval_strategies(test_query)
        
        print(f"\n📊 COMPARISON RESULTS:")
        print("-" * 40)
        
        for strategy, results in comparison.items():
            print(f"\n{strategy.upper().replace('_', ' ')}:")
            print(f"  • Results: {results.get('count', results.get('chunk_count', 0))}")
            print(f"  • Time: {results['time']}s")
            
            if 'results' in results:
                print("  • Top result:")
                if results['results']:
                    top_result = results['results'][0]
                    print(f"    - Score: {top_result['score']:.3f}")
                    print(f"    - Preview: {top_result['text_preview'][:100]}...")
            
            elif 'chunks' in results:
                print(f"  • Documents found: {results['doc_count']}")
                print(f"  • Chunks found: {results['chunk_count']}")
                if results['chunks']:
                    top_chunk = results['chunks'][0]
                    print(f"    - Top chunk score: {top_chunk['score']:.3f}")
                    print(f"    - Preview: {top_chunk['text_preview'][:100]}...")
        
        print("\n✅ Strategy comparison complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

def demonstrate_document_summary_retrieval():
    """Main demonstration function (legacy - now calls sample_query_demonstration)."""
    print("🚀 DOCUMENT SUMMARY RETRIEVAL DEMONSTRATION")
    print("=" * 80)
    
    # Test document summary retrieval
    test_document_summary_retrieval()
    
    # Compare retrieval strategies
    compare_retrieval_strategies()

# ---------- UTILITY FUNCTIONS -----------------------------------------------

def create_document_summary_retriever_from_latest_batch() -> DocumentSummaryRetriever:
    """Create a document summary retriever from the latest batch."""
    loader = EmbeddingLoader(Path("data/embedding"))
    latest_batch = loader.get_latest_batch()
    
    if not latest_batch:
        raise RuntimeError("No embedding batches found")
    
    all_embeddings = loader.load_all_embeddings_from_batch(latest_batch)
    
    # Combine embeddings
    all_summaries = []
    all_chunks = []
    for sub_batch, emb_types in all_embeddings.items():
        all_summaries.extend(emb_types.get('summaries', []))
        all_chunks.extend(emb_types.get('chunks', []))
    
    return DocumentSummaryRetriever(
        summary_embeddings=all_summaries,
        chunk_embeddings=all_chunks
    )

# ---------- ENTRY POINT -----------------------------------------------------

if __name__ == "__main__":
    import sys
    
    # Check for command line arguments for backward compatibility
    if len(sys.argv) > 1:
        if sys.argv[1] == "interactive":
            interactive_query_session()
        elif sys.argv[1] == "demo":
            sample_query_demonstration()
        elif sys.argv[1] == "legacy":
            demonstrate_document_summary_retrieval()
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Available arguments: 'interactive', 'demo', 'legacy'")
    else:
        # Default behavior: show menu
        main_menu() 