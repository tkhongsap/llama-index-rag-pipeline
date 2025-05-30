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
from llama_index.core.schema import TextNode, IndexNode
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
    """Main demonstration function."""
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
    demonstrate_document_summary_retrieval() 