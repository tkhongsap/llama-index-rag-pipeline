"""
12_recursive_retriever.py - Recursive retrieval with IndexNodes

This script implements recursive retrieval using IndexNodes for hierarchical
document access. It creates query engines for individual documents and builds
a multi-level retrieval system (summary → document → chunks).

Purpose:
- Implement recursive retrieval using IndexNodes
- Create query engines for individual documents
- Build hierarchical retrieval (summary → document → chunks)
- Compare recursive vs flat retrieval strategies
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
    Settings,
    QueryBundle,
    get_response_synthesizer
)
from llama_index.core.schema import TextNode, IndexNode
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
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

# Recursive retrieval settings
DEFAULT_TOP_K = 5
DEFAULT_RECURSIVE_TOP_K = 5
DEFAULT_CHUNK_TOP_K = 15

# Sample queries for testing recursive retrieval
RECURSIVE_QUERIES = [
    "What are the educational qualifications mentioned in the profiles?",
    "How do work experience levels vary across different profiles?",
    "What assessment scores are documented in the data?",
    "What types of companies and industries are represented?",
    "What salary ranges are mentioned across the profiles?"
]

# ---------- RECURSIVE RETRIEVER CLASS ---------------------------------------

class RecursiveDocumentRetriever:
    """Recursive retriever using IndexNodes for hierarchical access."""
    
    def __init__(
        self,
        indexnode_embeddings: List[Dict[str, Any]],
        chunk_embeddings: List[Dict[str, Any]],
        api_key: Optional[str] = None
    ):
        """Initialize recursive document retriever."""
        self.indexnode_embeddings = indexnode_embeddings
        self.chunk_embeddings = chunk_embeddings
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        # Setup models
        self._setup_models()
        
        # Create indices and retriever
        self._create_recursive_structure()
    
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
    
    def _create_recursive_structure(self):
        """Create the recursive retrieval structure."""
        print("🔄 Creating recursive retrieval structure...")
        
        # Create document-level indices for each document
        self._create_document_indices()
        
        # Create top-level index with IndexNodes
        self._create_top_level_index()
        
        # Create recursive retriever
        self._create_recursive_retriever()
    
    def _create_document_indices(self):
        """Create individual indices for each document."""
        print("📄 Creating individual document indices...")
        
        self.document_indices = {}
        reconstructor = IndexReconstructor(self.api_key)
        
        # Group chunks by document
        doc_chunks = {}
        for chunk in self.chunk_embeddings:
            doc_id = chunk.get('doc_id')
            if doc_id:
                if doc_id not in doc_chunks:
                    doc_chunks[doc_id] = []
                doc_chunks[doc_id].append(chunk)
        
        # Create index for each document
        for doc_id, chunks in doc_chunks.items():
            if chunks:  # Only create index if there are chunks
                doc_index = reconstructor.create_vector_index_from_embeddings(
                    chunks,
                    show_progress=False
                )
                self.document_indices[doc_id] = doc_index
        
        print(f"📊 Created {len(self.document_indices)} document indices")
    
    def _create_top_level_index(self):
        """Create top-level index using IndexNodes."""
        print("🔝 Creating top-level IndexNode index...")
        
        # Create IndexNodes that point to document indices
        index_nodes = []
        reconstructor = IndexReconstructor(self.api_key)
        
        for indexnode_emb in self.indexnode_embeddings:
            doc_id = indexnode_emb.get('metadata', {}).get('doc_id')
            
            if doc_id and doc_id in self.document_indices:
                # Create IndexNode with reference to document index
                index_node = IndexNode(
                    text=indexnode_emb.get('text', ''),
                    index_id=indexnode_emb.get('index_id', f"doc_{doc_id}"),
                    metadata=indexnode_emb.get('metadata', {}),
                    embedding=indexnode_emb.get('embedding_vector'),
                    id_=indexnode_emb.get('node_id')
                )
                index_nodes.append(index_node)
        
        # Create top-level index
        if index_nodes:
            self.top_level_index = reconstructor.create_vector_index_from_embeddings(
                [
                    {
                        'node_id': node.id_,
                        'text': node.text,
                        'metadata': node.metadata,
                        'embedding_vector': node.embedding,
                        'type': 'indexnode',
                        'index_id': node.index_id
                    }
                    for node in index_nodes
                ],
                show_progress=True
            )
        else:
            raise ValueError("No valid IndexNodes found for top-level index")
    
    def _create_recursive_retriever(self):
        """Create the recursive retriever using manual hierarchical logic."""
        print("🔄 Creating hierarchical retriever...")
        
        # Store retrievers for manual hierarchical retrieval
        self.top_level_retriever = self.top_level_index.as_retriever(
            similarity_top_k=DEFAULT_RECURSIVE_TOP_K
        )
        
        self.document_retrievers = {}
        for doc_id, doc_index in self.document_indices.items():
            self.document_retrievers[doc_id] = doc_index.as_retriever(
                similarity_top_k=DEFAULT_CHUNK_TOP_K
            )
        
        print("✅ Hierarchical retriever created successfully")
    
    def recursive_query(
        self,
        query: str,
        show_details: bool = True
    ) -> Dict[str, Any]:
        """Perform recursive query using manual hierarchical logic."""
        start_time = time.time()
        
        if show_details:
            print(f"🔍 Performing hierarchical query: {query}")
        
        # Step 1: Query top-level index to find relevant documents
        if show_details:
            print("🔍 Step 1: Finding relevant documents...")
        
        top_level_nodes = self.top_level_retriever.retrieve(query)
        
        # Step 2: Query individual document indices
        if show_details:
            print(f"🔍 Step 2: Querying {len(top_level_nodes)} document indices...")
        
        all_chunks = []
        for node in top_level_nodes:
            doc_id = node.node.metadata.get('doc_id')
            if doc_id and doc_id in self.document_retrievers:
                doc_retriever = self.document_retrievers[doc_id]
                doc_chunks = doc_retriever.retrieve(query)
                
                # Add document context to chunks
                for chunk in doc_chunks:
                    chunk.node.metadata['source_doc_score'] = node.score
                    all_chunks.append(chunk)
        
        # Step 3: Create response using LLM
        if show_details:
            print(f"🔍 Step 3: Synthesizing response from {len(all_chunks)} chunks...")
        
        # Create context from retrieved chunks
        context_str = "\n\n".join([
            f"Document {chunk.node.metadata.get('doc_id', 'Unknown')}:\n{chunk.node.text}"
            for chunk in all_chunks[:10]  # Limit context size
        ])
        
        # Generate response using LLM
        llm = Settings.llm
        prompt = f"""Based on the following context, answer the question: {query}

Context:
{context_str}

Answer:"""
        
        response_text = llm.complete(prompt).text
        
        end_time = time.time()
        
        # Extract source information
        sources = []
        for chunk in all_chunks:
            source_info = {
                'node_id': chunk.node.id_,
                'score': chunk.score if hasattr(chunk, 'score') else None,
                'text_preview': chunk.node.text[:200] + "..." if len(chunk.node.text) > 200 else chunk.node.text,
                'metadata': chunk.node.metadata if hasattr(chunk.node, 'metadata') else {},
                'doc_id': chunk.node.metadata.get('doc_id') if hasattr(chunk.node, 'metadata') else None,
                'source_doc_score': chunk.node.metadata.get('source_doc_score')
            }
            sources.append(source_info)
        
        return {
            'query': query,
            'response': response_text,
            'sources': sources,
            'metadata': {
                'query_time': round(end_time - start_time, 2),
                'num_sources': len(sources),
                'num_documents_searched': len(top_level_nodes),
                'retrieval_method': 'hierarchical'
            }
        }
    
    def compare_with_flat_retrieval(
        self,
        query: str
    ) -> Dict[str, Any]:
        """Compare recursive retrieval with flat retrieval."""
        print(f"🔬 Comparing retrieval methods for: {query}")
        
        results = {}
        
        # Recursive retrieval
        print("📊 Testing recursive retrieval...")
        recursive_result = self.recursive_query(query, show_details=False)
        results['recursive'] = recursive_result
        
        # Flat retrieval (all chunks in one index)
        print("📊 Testing flat retrieval...")
        start_time = time.time()
        
        # Create flat index with all chunks
        reconstructor = IndexReconstructor(self.api_key)
        flat_index = reconstructor.create_vector_index_from_embeddings(
            self.chunk_embeddings,
            show_progress=False
        )
        
        flat_query_engine = flat_index.as_query_engine(
            similarity_top_k=DEFAULT_TOP_K,
            response_mode=ResponseMode.TREE_SUMMARIZE
        )
        
        flat_response = flat_query_engine.query(query)
        end_time = time.time()
        
        # Extract flat sources
        flat_sources = []
        if hasattr(flat_response, 'source_nodes'):
            for node in flat_response.source_nodes:
                source_info = {
                    'node_id': node.node.id_,
                    'score': node.score if hasattr(node, 'score') else None,
                    'text_preview': node.node.text[:200] + "..." if len(node.node.text) > 200 else node.node.text,
                    'metadata': node.node.metadata if hasattr(node.node, 'metadata') else {},
                    'doc_id': node.node.metadata.get('doc_id') if hasattr(node.node, 'metadata') else None
                }
                flat_sources.append(source_info)
        
        results['flat'] = {
            'query': query,
            'response': str(flat_response),
            'sources': flat_sources,
            'metadata': {
                'query_time': round(end_time - start_time, 2),
                'num_sources': len(flat_sources),
                'retrieval_method': 'flat'
            }
        }
        
        return results

# ---------- INTERACTIVE FUNCTIONS -------------------------------------------

def interactive_query_session():
    """Run an interactive query session with recursive retrieval."""
    print("💬 INTERACTIVE RECURSIVE RETRIEVAL SESSION")
    print("=" * 80)
    print("🎯 You can now ask questions using recursive hierarchical retrieval!")
    print("\n📋 Available commands:")
    print("  • Type any question to get recursive retrieval results")
    print("  • 'quit' or 'exit' - End the session")
    print("  • 'help' - Show available commands")
    print("  • 'settings' - Show current retriever settings")
    print("  • 'top_k=N' - Change number of documents retrieved at top level (e.g., top_k=5)")
    print("  • 'chunk_k=N' - Change chunks per document (e.g., chunk_k=8)")
    print("  • 'compare' - Compare recursive vs flat retrieval for last query")
    print("\n💡 Example questions:")
    print("  • What are the educational qualifications mentioned in the profiles?")
    print("  • How do work experience levels vary across different profiles?")
    print("  • What assessment scores are documented in the data?")
    print("  • What types of companies and industries are represented?")
    
    try:
        # Create recursive retriever
        print("\n🔄 Setting up recursive retriever...")
        print("   Loading embeddings from latest batch...")
        retriever = create_recursive_retriever_from_latest_batch()
        print("✅ Recursive retriever ready!")
        print(f"📊 Loaded {len(retriever.indexnode_embeddings)} IndexNodes and {len(retriever.chunk_embeddings)} chunks")
        print(f"📄 Created {len(retriever.document_indices)} document indices")
        
        # Current settings
        current_top_k = DEFAULT_RECURSIVE_TOP_K
        current_chunk_k = DEFAULT_CHUNK_TOP_K
        last_query = None
        
        print(f"⚙️  Current settings: top_k={current_top_k}, chunk_k={current_chunk_k}")
        
        while True:
            # Get user input
            query = input("\n❓ Enter your question: ").strip()
            
            # Check for commands
            if query.lower() in ['quit', 'exit']:
                print("👋 Thanks for using the recursive retriever! Goodbye!")
                break
            
            elif query.lower() == 'help':
                print("\n📋 Available commands:")
                print("  • quit/exit - End session")
                print("  • help - Show this help")
                print("  • settings - Show current settings")
                print("  • top_k=N - Change number of documents retrieved at top level")
                print("  • chunk_k=N - Change chunks per document")
                print("  • compare - Compare recursive vs flat retrieval for last query")
                print("  • clear - Clear screen (if supported)")
                continue
                
            elif query.lower() == 'settings':
                print(f"\n⚙️ Current settings:")
                print(f"  • Top Level K: {current_top_k}")
                print(f"  • Chunks per Document K: {current_chunk_k}")
                print(f"  • Total IndexNodes: {len(retriever.indexnode_embeddings)}")
                print(f"  • Total Chunks: {len(retriever.chunk_embeddings)}")
                print(f"  • Document Indices: {len(retriever.document_indices)}")
                continue
                
            elif query.startswith('top_k='):
                try:
                    new_k = int(query.split('=')[1])
                    if new_k > 0:
                        current_top_k = new_k
                        # Update retriever settings
                        retriever.top_level_retriever = retriever.top_level_index.as_retriever(
                            similarity_top_k=new_k
                        )
                        print(f"✅ Updated top_k to {new_k}")
                    else:
                        print("❌ top_k must be greater than 0")
                except:
                    print("❌ Invalid format. Use: top_k=5")
                continue
                
            elif query.startswith('chunk_k='):
                try:
                    new_k = int(query.split('=')[1])
                    if new_k > 0:
                        current_chunk_k = new_k
                        # Update document retrievers
                        for doc_id, doc_index in retriever.document_indices.items():
                            retriever.document_retrievers[doc_id] = doc_index.as_retriever(
                                similarity_top_k=new_k
                            )
                        print(f"✅ Updated chunk_k to {new_k}")
                    else:
                        print("❌ chunk_k must be greater than 0")
                except:
                    print("❌ Invalid format. Use: chunk_k=8")
                continue
                
            elif query.lower() == 'compare':
                if last_query:
                    print(f"\n🔬 Comparing retrieval methods for: '{last_query}'")
                    comparison = retriever.compare_with_flat_retrieval(last_query)
                    
                    print(f"\n📊 COMPARISON RESULTS:")
                    print("-" * 40)
                    
                    for method, result in comparison.items():
                        print(f"\n{method.upper()} RETRIEVAL:")
                        print(f"  • Response length: {len(result['response'])} chars")
                        print(f"  • Query time: {result['metadata']['query_time']}s")
                        print(f"  • Sources found: {result['metadata']['num_sources']}")
                        
                        if result['sources']:
                            print("  • Top source:")
                            top_source = result['sources'][0]
                            score = top_source.get('score')
                            score_str = f"{score:.3f}" if score is not None else 'N/A'
                            print(f"    - Score: {score_str}")
                            print(f"    - Preview: {top_source['text_preview'][:100]}...")
                        
                        print(f"  • Response preview: {result['response'][:150]}...")
                else:
                    print("❌ No previous query to compare. Ask a question first!")
                continue
                
            elif query.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            
            # Execute recursive retrieval
            if query:
                print("\n🔄 Processing your question with recursive retrieval...")
                last_query = query
                
                result = retriever.recursive_query(query, show_details=False)
                
                # Show response first
                print(f"\n💬 ANSWER:")
                print("=" * 50)
                print(result['response'])
                print("=" * 50)
                
                print(f"\n⏱️ Query completed in {result['metadata']['query_time']}s")
                print(f"📊 Retrieved {result['metadata']['num_sources']} chunks from {result['metadata']['num_documents_searched']} documents")
                print(f"💡 Tip: Type 'compare' to see how this compares to flat retrieval")
                
            else:
                print("❌ Please enter a question or command.")
    
    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error setting up recursive retriever: {str(e)}")
        print("💡 Make sure you have embeddings available in the data directory.")
        import traceback
        traceback.print_exc()

def sample_query_demonstration():
    """Run demonstration with predefined sample queries."""
    print("🧪 SAMPLE QUERY DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Create retriever
        print("🔄 Setting up recursive retriever...")
        retriever = create_recursive_retriever_from_latest_batch()
        print("✅ Retriever ready!")
        
        # Test with sample queries
        for i, query in enumerate(RECURSIVE_QUERIES, 1):
            print(f"\n🔍 Sample Query {i}: {query}")
            print("-" * 60)
            
            # Perform recursive retrieval
            result = retriever.recursive_query(query, show_details=False)
            
            print(f"💬 Response: {result['response'][:300]}...")
            print(f"📊 Sources: {result['metadata']['num_sources']}")
            print(f"📄 Documents searched: {result['metadata']['num_documents_searched']}")
            print(f"⏱️ Query time: {result['metadata']['query_time']}s")
            
            # Show top sources
            if result['sources']:
                print("\nTop sources:")
                for j, source in enumerate(result['sources'][:3], 1):
                    doc_id = source.get('doc_id', 'Unknown')
                    score = source.get('score')
                    score_str = f"{score:.3f}" if score is not None else 'N/A'
                    print(f"  {j}. Doc: {doc_id} (score: {score_str})")
                    print(f"     Preview: {source['text_preview'][:100]}...")
            
            # Pause between queries for readability
            if i < len(RECURSIVE_QUERIES):
                input("\nPress Enter to continue to next query...")
        
        # Show method comparison for last query
        print(f"\n🔬 METHOD COMPARISON for last query:")
        print("=" * 60)
        comparison = retriever.compare_with_flat_retrieval(RECURSIVE_QUERIES[-1])
        
        for method, result in comparison.items():
            print(f"\n{method.upper()} RETRIEVAL:")
            print(f"  • Response length: {len(result['response'])} chars")
            print(f"  • Query time: {result['metadata']['query_time']}s")
            print(f"  • Sources found: {result['metadata']['num_sources']}")
        
        print("\n✅ Sample query demonstration complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

def main_menu():
    """Display main menu and handle user choice."""
    print("🚀 RECURSIVE RETRIEVER")
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

def test_recursive_retrieval():
    """Test recursive retrieval functionality."""
    print("🧪 TESTING RECURSIVE RETRIEVAL")
    print("=" * 80)
    
    try:
        # Load embeddings
        print("🔄 Loading embeddings...")
        loader = EmbeddingLoader(Path("data/embedding"))
        latest_batch = loader.get_latest_batch()
        
        if not latest_batch:
            print("❌ No embedding batches found!")
            return
        
        # Load all embedding types
        all_embeddings = loader.load_all_embeddings_from_batch(latest_batch)
        
        # Combine embeddings from all sub-batches
        all_indexnodes = []
        all_chunks = []
        
        for sub_batch, emb_types in all_embeddings.items():
            all_indexnodes.extend(emb_types.get('indexnodes', []))
            all_chunks.extend(emb_types.get('chunks', []))
        
        print(f"📊 Loaded {len(all_indexnodes)} IndexNodes and {len(all_chunks)} chunks")
        
        # Create recursive retriever
        print("🔧 Creating recursive document retriever...")
        retriever = RecursiveDocumentRetriever(
            indexnode_embeddings=all_indexnodes,
            chunk_embeddings=all_chunks
        )
        
        # Test with sample queries
        for i, query in enumerate(RECURSIVE_QUERIES[:3], 1):
            print(f"\n🔍 Query {i}: {query}")
            print("-" * 60)
            
            result = retriever.recursive_query(query)
            
            print(f"📝 Response: {result['response'][:300]}...")
            print(f"📊 Sources: {result['metadata']['num_sources']}")
            print(f"⏱️ Query time: {result['metadata']['query_time']}s")
            
            # Show top sources
            if result['sources']:
                print("\nTop sources:")
                for j, source in enumerate(result['sources'][:3], 1):
                    doc_id = source.get('doc_id', 'Unknown')
                    score = source.get('score')
                    score_str = f"{score:.3f}" if score is not None else 'N/A'
                    print(f"  {j}. Doc: {doc_id} (score: {score_str})")
                    print(f"     Preview: {source['text_preview'][:100]}...")
        
        print("\n✅ Recursive retrieval test complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

def compare_retrieval_methods():
    """Compare recursive vs flat retrieval methods."""
    print("\n🔬 COMPARING RETRIEVAL METHODS")
    print("=" * 80)
    
    try:
        # Load embeddings
        loader = EmbeddingLoader(Path("data/embedding"))
        latest_batch = loader.get_latest_batch()
        all_embeddings = loader.load_all_embeddings_from_batch(latest_batch)
        
        # Combine embeddings
        all_indexnodes = []
        all_chunks = []
        for sub_batch, emb_types in all_embeddings.items():
            all_indexnodes.extend(emb_types.get('indexnodes', []))
            all_chunks.extend(emb_types.get('chunks', []))
        
        # Create recursive retriever
        retriever = RecursiveDocumentRetriever(
            indexnode_embeddings=all_indexnodes,
            chunk_embeddings=all_chunks
        )
        
        # Test query
        test_query = "What educational degrees and majors are mentioned?"
        print(f"🔍 Test query: {test_query}")
        
        # Compare methods
        comparison = retriever.compare_with_flat_retrieval(test_query)
        
        print(f"\n📊 COMPARISON RESULTS:")
        print("-" * 40)
        
        for method, result in comparison.items():
            print(f"\n{method.upper()} RETRIEVAL:")
            print(f"  • Response length: {len(result['response'])} chars")
            print(f"  • Query time: {result['metadata']['query_time']}s")
            print(f"  • Sources found: {result['metadata']['num_sources']}")
            
            if result['sources']:
                print("  • Top source:")
                top_source = result['sources'][0]
                score = top_source.get('score')
                score_str = f"{score:.3f}" if score is not None else 'N/A'
                print(f"    - Score: {score_str}")
                print(f"    - Preview: {top_source['text_preview'][:100]}...")
            
            print(f"  • Response preview: {result['response'][:150]}...")
        
        print("\n✅ Method comparison complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

def demonstrate_recursive_retrieval():
    """Main demonstration function."""
    print("🚀 RECURSIVE RETRIEVAL DEMONSTRATION")
    print("=" * 80)
    
    # Test recursive retrieval
    test_recursive_retrieval()
    
    # Compare retrieval methods
    compare_retrieval_methods()

# ---------- UTILITY FUNCTIONS -----------------------------------------------

def create_recursive_retriever_from_latest_batch() -> RecursiveDocumentRetriever:
    """Create a recursive retriever from the latest batch."""
    loader = EmbeddingLoader(Path("data/embedding"))
    latest_batch = loader.get_latest_batch()
    
    if not latest_batch:
        raise RuntimeError("No embedding batches found")
    
    all_embeddings = loader.load_all_embeddings_from_batch(latest_batch)
    
    # Combine embeddings
    all_indexnodes = []
    all_chunks = []
    for sub_batch, emb_types in all_embeddings.items():
        all_indexnodes.extend(emb_types.get('indexnodes', []))
        all_chunks.extend(emb_types.get('chunks', []))
    
    return RecursiveDocumentRetriever(
        indexnode_embeddings=all_indexnodes,
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
            demonstrate_recursive_retrieval()
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Available arguments: 'interactive', 'demo', 'legacy'")
    else:
        # Default behavior: show menu
        main_menu() 