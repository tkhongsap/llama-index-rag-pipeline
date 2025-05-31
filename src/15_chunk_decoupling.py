"""
15_chunk_decoupling.py - Decouple retrieval chunks from synthesis chunks

This script implements sentence-window retrieval and metadata replacement postprocessing
to create fine-grained retrieval with context expansion for better response quality.

Purpose:
- Implement sentence-window retrieval
- Add metadata replacement postprocessing
- Create fine-grained retrieval with context expansion
- Optimize chunk retrieval system
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dotenv import load_dotenv
import pandas as pd

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import LlamaIndex components
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    Document,
    QueryBundle,
    Response
)
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceEmbeddingOptimizer,
    SimilarityPostprocessor
)
from llama_index.core.node_parser import (
    SentenceWindowNodeParser,
    SentenceSplitter
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Import our utilities
from load_embeddings import (
    EmbeddingLoader,
    create_index_from_latest_batch
)

# ---------- CONFIGURATION ---------------------------------------------------

# Load environment variables
load_dotenv(override=True)

# Configure LlamaIndex settings
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.0)

# Sentence window configuration
DEFAULT_WINDOW_SIZE = 3  # Number of sentences before and after
DEFAULT_WINDOW_METADATA_KEY = "window"
DEFAULT_ORIGINAL_TEXT_METADATA_KEY = "original_text"

# Chunk optimization settings
DEFAULT_SIMILARITY_THRESHOLD = 0.7
DEFAULT_PERCENTILE_CUTOFF = 0.5

# ---------- SENTENCE WINDOW RETRIEVAL ---------------------------------------

class SentenceWindowRetriever:
    """Retriever that uses sentence-window approach for fine-grained retrieval."""
    
    def __init__(
        self,
        documents: List[Document],
        window_size: int = DEFAULT_WINDOW_SIZE,
        sentence_splitter_chunk_size: int = 1024,
        sentence_splitter_chunk_overlap: int = 20
    ):
        """Initialize sentence window retriever."""
        self.documents = documents
        self.window_size = window_size
        self.sentence_splitter_chunk_size = sentence_splitter_chunk_size
        self.sentence_splitter_chunk_overlap = sentence_splitter_chunk_overlap
        
        # Create sentence window node parser
        self.sentence_window_parser = SentenceWindowNodeParser.from_defaults(
            window_size=window_size,
            window_metadata_key=DEFAULT_WINDOW_METADATA_KEY,
            original_text_metadata_key=DEFAULT_ORIGINAL_TEXT_METADATA_KEY,
        )
        
        # Create sentence splitter for base nodes
        self.sentence_splitter = SentenceSplitter(
            chunk_size=sentence_splitter_chunk_size,
            chunk_overlap=sentence_splitter_chunk_overlap,
        )
        
        # Build the index
        self.index = self._build_sentence_window_index()
        
        # Create retriever
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=5
        )
        
        # Create postprocessor for metadata replacement
        self.postprocessor = MetadataReplacementPostProcessor(
            target_metadata_key=DEFAULT_WINDOW_METADATA_KEY
        )
    
    def _build_sentence_window_index(self) -> VectorStoreIndex:
        """Build sentence window index from documents."""
        print(f"🔧 Building sentence window index with window size {self.window_size}...")
        
        # First, split documents into base nodes
        base_nodes = self.sentence_splitter.get_nodes_from_documents(self.documents)
        print(f"📄 Created {len(base_nodes)} base nodes")
        
        # Then create sentence window nodes
        sentence_window_nodes = self.sentence_window_parser.get_nodes_from_documents(
            [Document(text=node.text, metadata=node.metadata) for node in base_nodes]
        )
        print(f"🪟 Created {len(sentence_window_nodes)} sentence window nodes")
        
        # Build index from sentence window nodes
        index = VectorStoreIndex(sentence_window_nodes)
        print("✅ Sentence window index built successfully")
        
        return index
    
    def retrieve(
        self,
        query: str,
        similarity_top_k: int = 5,
        apply_postprocessing: bool = True
    ) -> List[NodeWithScore]:
        """Retrieve with sentence window approach."""
        # Update retriever similarity_top_k
        self.retriever.similarity_top_k = similarity_top_k
        
        # Retrieve nodes
        retrieved_nodes = self.retriever.retrieve(query)
        
        # Apply postprocessing to replace with window content
        if apply_postprocessing:
            retrieved_nodes = self.postprocessor.postprocess_nodes(retrieved_nodes)
        
        return retrieved_nodes
    
    def query(
        self,
        query: str,
        similarity_top_k: int = 5,
        apply_postprocessing: bool = True,
        show_details: bool = True
    ) -> Dict[str, Any]:
        """Query with sentence window retrieval."""
        start_time = time.time()
        
        if show_details:
            print(f"🔍 Sentence window query: {query}")
            print(f"📊 Window size: {self.window_size}, Top-k: {similarity_top_k}")
        
        # Retrieve nodes
        retrieved_nodes = self.retrieve(query, similarity_top_k, apply_postprocessing)
        
        # Create query engine and get response
        query_engine = self.index.as_query_engine(
            similarity_top_k=similarity_top_k,
            node_postprocessors=[self.postprocessor] if apply_postprocessing else [],
            response_mode="tree_summarize"
        )
        
        response = query_engine.query(query)
        
        end_time = time.time()
        
        # Extract source information
        sources = []
        for node in retrieved_nodes:
            source_info = {
                'text_preview': node.text[:200] + "..." if len(node.text) > 200 else node.text,
                'score': getattr(node, 'score', 0.0),
                'metadata': node.metadata if hasattr(node, 'metadata') else {},
                'has_window': DEFAULT_WINDOW_METADATA_KEY in (node.metadata or {}),
                'window_size': self.window_size
            }
            sources.append(source_info)
        
        return {
            'query': query,
            'response': str(response),
            'sources': sources,
            'metadata': {
                'total_time': round(end_time - start_time, 2),
                'num_sources': len(sources),
                'window_size': self.window_size,
                'postprocessing_applied': apply_postprocessing,
                'retrieval_method': 'sentence_window'
            }
        }

# ---------- ADVANCED CHUNK DECOUPLING ---------------------------------------

class AdvancedChunkDecoupler:
    """Advanced chunk decoupling with multiple optimization strategies."""
    
    def __init__(
        self,
        index: VectorStoreIndex,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        percentile_cutoff: float = DEFAULT_PERCENTILE_CUTOFF
    ):
        """Initialize advanced chunk decoupler."""
        self.index = index
        self.similarity_threshold = similarity_threshold
        self.percentile_cutoff = percentile_cutoff
        
        # Create multiple postprocessors
        self.metadata_replacer = MetadataReplacementPostProcessor(
            target_metadata_key=DEFAULT_WINDOW_METADATA_KEY
        )
        
        self.similarity_filter = SimilarityPostprocessor(
            similarity_cutoff=similarity_threshold
        )
        
        # Create optimized retriever
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=10  # Retrieve more, then filter
        )
    
    def retrieve_with_context_expansion(
        self,
        query: str,
        base_similarity_top_k: int = 5,
        expansion_factor: int = 2,
        apply_similarity_filter: bool = True
    ) -> List[NodeWithScore]:
        """Retrieve with context expansion strategy."""
        # First retrieval: get base nodes
        self.retriever.similarity_top_k = base_similarity_top_k
        base_nodes = self.retriever.retrieve(query)
        
        # Context expansion: get additional nodes around high-scoring ones
        expanded_nodes = []
        
        for node in base_nodes:
            expanded_nodes.append(node)
            
            # Try to get neighboring nodes (simplified approach)
            # In a real implementation, you'd use document structure
            if hasattr(node.node, 'metadata') and 'doc_id' in node.node.metadata:
                doc_id = node.node.metadata['doc_id']
                
                # Get more nodes from the same document
                self.retriever.similarity_top_k = base_similarity_top_k * expansion_factor
                doc_nodes = self.retriever.retrieve(f"document {doc_id} content")
                
                # Add relevant nodes that aren't already included
                for doc_node in doc_nodes:
                    if doc_node.node.id_ not in [n.node.id_ for n in expanded_nodes]:
                        expanded_nodes.append(doc_node)
        
        # Apply similarity filtering if requested
        if apply_similarity_filter:
            expanded_nodes = self.similarity_filter.postprocess_nodes(expanded_nodes)
        
        # Sort by score and limit
        expanded_nodes.sort(key=lambda x: getattr(x, 'score', 0.0), reverse=True)
        return expanded_nodes[:base_similarity_top_k * 2]  # Return up to 2x the requested amount
    
    def retrieve_with_hierarchical_chunking(
        self,
        query: str,
        chunk_sizes: List[int] = [512, 1024, 2048],
        top_k_per_size: int = 3
    ) -> Dict[str, List[NodeWithScore]]:
        """Retrieve using multiple chunk sizes for hierarchical context."""
        results = {}
        
        for chunk_size in chunk_sizes:
            # Create temporary nodes with different chunk sizes
            # This is a simplified approach - in practice, you'd pre-process with different sizes
            self.retriever.similarity_top_k = top_k_per_size
            nodes = self.retriever.retrieve(query)
            
            # Filter nodes by approximate chunk size (based on text length)
            size_filtered_nodes = [
                node for node in nodes 
                if abs(len(node.text) - chunk_size) < chunk_size * 0.3
            ]
            
            results[f"chunk_{chunk_size}"] = size_filtered_nodes[:top_k_per_size]
        
        return results
    
    def query_with_decoupled_chunks(
        self,
        query: str,
        strategy: str = "context_expansion",
        show_details: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Query with decoupled chunk strategies."""
        start_time = time.time()
        
        if show_details:
            print(f"🔧 Decoupled chunk query using {strategy} strategy")
            print(f"🔍 Query: {query}")
        
        if strategy == "context_expansion":
            retrieved_nodes = self.retrieve_with_context_expansion(query, **kwargs)
            
        elif strategy == "hierarchical":
            hierarchical_results = self.retrieve_with_hierarchical_chunking(query, **kwargs)
            # Flatten results for response synthesis
            retrieved_nodes = []
            for chunk_size, nodes in hierarchical_results.items():
                retrieved_nodes.extend(nodes)
            
        else:
            # Default retrieval
            retrieved_nodes = self.retriever.retrieve(query)
        
        # Create response using retrieved nodes
        query_engine = self.index.as_query_engine(
            similarity_top_k=len(retrieved_nodes),
            response_mode="tree_summarize"
        )
        
        response = query_engine.query(query)
        
        end_time = time.time()
        
        # Extract source information
        sources = []
        for node in retrieved_nodes:
            source_info = {
                'text_preview': node.text[:200] + "..." if len(node.text) > 200 else node.text,
                'score': getattr(node, 'score', 0.0),
                'text_length': len(node.text),
                'metadata': node.metadata if hasattr(node, 'metadata') else {}
            }
            sources.append(source_info)
        
        return {
            'query': query,
            'response': str(response),
            'sources': sources,
            'metadata': {
                'total_time': round(end_time - start_time, 2),
                'num_sources': len(sources),
                'strategy': strategy,
                'retrieval_method': 'decoupled_chunks',
                'avg_chunk_length': sum(len(s['text_preview']) for s in sources) / len(sources) if sources else 0
            }
        }

# ---------- COMPARISON AND EVALUATION ---------------------------------------

class ChunkDecouplingSuite:
    """Suite for comparing different chunk decoupling strategies."""
    
    def __init__(self, documents: List[Document]):
        """Initialize chunk decoupling suite."""
        self.documents = documents
        
        # Create different retrievers
        print("🔧 Initializing chunk decoupling suite...")
        
        # Standard retriever
        standard_index = create_index_from_latest_batch(use_chunks=True)
        self.standard_retriever = VectorIndexRetriever(index=standard_index)
        
        # Sentence window retriever
        self.sentence_window_retriever = SentenceWindowRetriever(
            documents=documents,
            window_size=3
        )
        
        # Advanced decoupler
        self.advanced_decoupler = AdvancedChunkDecoupler(standard_index)
        
        print("✅ All retrievers initialized")
    
    def compare_strategies(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Compare all chunk decoupling strategies."""
        print(f"\n🔬 COMPARING CHUNK DECOUPLING STRATEGIES")
        print(f"Query: {query}")
        print("=" * 80)
        
        results = {}
        
        # Strategy 1: Standard retrieval
        print("\n📊 Testing standard retrieval...")
        start_time = time.time()
        standard_nodes = self.standard_retriever.retrieve(query)
        standard_time = time.time() - start_time
        
        results['standard'] = {
            'nodes': standard_nodes[:top_k],
            'time': round(standard_time, 2),
            'avg_chunk_length': sum(len(node.text) for node in standard_nodes[:top_k]) / min(top_k, len(standard_nodes))
        }
        
        # Strategy 2: Sentence window
        print("📊 Testing sentence window retrieval...")
        sentence_result = self.sentence_window_retriever.query(
            query, 
            similarity_top_k=top_k, 
            show_details=False
        )
        
        results['sentence_window'] = {
            'result': sentence_result,
            'time': sentence_result['metadata']['total_time'],
            'avg_chunk_length': sentence_result['metadata'].get('avg_chunk_length', 0)
        }
        
        # Strategy 3: Context expansion
        print("📊 Testing context expansion...")
        expansion_result = self.advanced_decoupler.query_with_decoupled_chunks(
            query,
            strategy="context_expansion",
            show_details=False,
            base_similarity_top_k=top_k
        )
        
        results['context_expansion'] = {
            'result': expansion_result,
            'time': expansion_result['metadata']['total_time'],
            'avg_chunk_length': expansion_result['metadata'].get('avg_chunk_length', 0)
        }
        
        # Strategy 4: Hierarchical chunking
        print("📊 Testing hierarchical chunking...")
        hierarchical_result = self.advanced_decoupler.query_with_decoupled_chunks(
            query,
            strategy="hierarchical",
            show_details=False,
            top_k_per_size=2
        )
        
        results['hierarchical'] = {
            'result': hierarchical_result,
            'time': hierarchical_result['metadata']['total_time'],
            'avg_chunk_length': hierarchical_result['metadata'].get('avg_chunk_length', 0)
        }
        
        # Display comparison
        self._display_comparison(results)
        
        return results
    
    def _display_comparison(self, results: Dict[str, Any]):
        """Display comparison results."""
        print(f"\n📊 STRATEGY COMPARISON RESULTS")
        print("-" * 80)
        print(f"{'Strategy':<20} {'Time (s)':<10} {'Sources':<10} {'Avg Length':<12}")
        print("-" * 80)
        
        for strategy, data in results.items():
            if strategy == 'standard':
                time_taken = data['time']
                num_sources = len(data['nodes'])
                avg_length = int(data['avg_chunk_length'])
            else:
                time_taken = data['time']
                num_sources = data['result']['metadata']['num_sources']
                avg_length = int(data['avg_chunk_length'])
            
            print(f"{strategy:<20} {time_taken:<10.2f} {num_sources:<10} {avg_length:<12}")
        
        print("-" * 80)

# ---------- DEMONSTRATION FUNCTIONS -----------------------------------------

def demonstrate_chunk_decoupling():
    """Demonstrate chunk decoupling capabilities."""
    print("🔧 CHUNK DECOUPLING DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Load real documents from CSV file
        print("\n📚 Loading real documents from CSV file...")
        
        # Load the CSV file
        csv_path = Path("data/input_docs/input_dataset.csv")
        if not csv_path.exists():
            print(f"❌ CSV file not found: {csv_path}")
            return
        
        # Read CSV and create documents
        df = pd.read_csv(csv_path)
        print(f"📊 Loaded {len(df)} rows from CSV")
        
        # Create documents from CSV content (using first 10 rows for demo)
        documents = []
        for idx, row in df.head(10).iterrows():
            # Combine relevant text columns into document content
            text_parts = []
            for col in df.columns:
                if pd.notna(row[col]) and isinstance(row[col], str):
                    text_parts.append(f"{col}: {row[col]}")
            
            if text_parts:
                doc_text = "\n".join(text_parts)
                documents.append(Document(
                    text=doc_text,
                    metadata={"row_id": idx, "source": "input_dataset.csv"}
                ))
        
        print(f"📄 Created {len(documents)} documents from CSV data")
        
        # Create sentence window retriever
        print("\n🪟 Creating sentence window retriever...")
        sentence_retriever = SentenceWindowRetriever(
            documents=documents,
            window_size=3
        )
        
        # Create advanced decoupler using existing embeddings
        print("\n🔧 Creating advanced decoupler from existing embeddings...")
        try:
            index = create_index_from_latest_batch(use_chunks=True, max_embeddings=50)
            advanced_decoupler = AdvancedChunkDecoupler(index)
            has_existing_index = True
        except Exception as e:
            print(f"⚠️ Could not load existing embeddings: {e}")
            print("📄 Creating new index from CSV documents...")
            # Create a simple index from our CSV documents
            index = VectorStoreIndex.from_documents(documents[:5])  # Use first 5 for demo
            advanced_decoupler = AdvancedChunkDecoupler(index)
            has_existing_index = False
        
        # Test queries relevant to typical CSV data
        test_queries = [
            "What information is available in the dataset?",
            "What are the main data fields or categories?",
            "What patterns can be found in the data?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Test Query {i}: {query}")
            print("-" * 60)
            
            # Test sentence window retrieval
            print("\n🪟 Sentence Window Retrieval:")
            try:
                sw_result = sentence_retriever.query(query, show_details=False)
                print(f"Response: {sw_result['response'][:200]}...")
                print(f"Time: {sw_result['metadata']['total_time']}s")
                print(f"Sources: {sw_result['metadata']['num_sources']}")
                print(f"Window size: {sw_result['metadata']['window_size']}")
            except Exception as e:
                print(f"❌ Sentence window error: {e}")
            
            # Test context expansion (only if we have existing embeddings)
            if has_existing_index:
                print("\n🔧 Context Expansion:")
                try:
                    ce_result = advanced_decoupler.query_with_decoupled_chunks(
                        query, 
                        strategy="context_expansion",
                        show_details=False
                    )
                    print(f"Response: {ce_result['response'][:200]}...")
                    print(f"Time: {ce_result['metadata']['total_time']}s")
                    print(f"Sources: {ce_result['metadata']['num_sources']}")
                except Exception as e:
                    print(f"❌ Context expansion error: {e}")
            else:
                print("\n🔧 Standard Retrieval (fallback):")
                try:
                    query_engine = index.as_query_engine(similarity_top_k=3)
                    response = query_engine.query(query)
                    print(f"Response: {str(response)[:200]}...")
                except Exception as e:
                    print(f"❌ Standard retrieval error: {e}")
        
        print("\n✅ Chunk decoupling demonstration complete!")
        
        # Show comparison of different window sizes
        print("\n🪟 TESTING DIFFERENT WINDOW SIZES")
        print("-" * 60)
        
        test_query = "What information is available?"
        window_sizes = [1, 2, 3, 5]
        
        for window_size in window_sizes:
            print(f"\n📊 Window size {window_size}:")
            try:
                temp_retriever = SentenceWindowRetriever(
                    documents=documents[:3],  # Use fewer docs for speed
                    window_size=window_size
                )
                result = temp_retriever.query(test_query, show_details=False)
                print(f"  Response length: {len(result['response'])} chars")
                print(f"  Time: {result['metadata']['total_time']}s")
                print(f"  Sources: {result['metadata']['num_sources']}")
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
    except Exception as e:
        print(f"\n❌ Error in demonstration: {str(e)}")
        import traceback
        traceback.print_exc()

def test_sentence_window_sizes():
    """Test different sentence window sizes."""
    print("\n🪟 SENTENCE WINDOW SIZE TESTING")
    print("=" * 80)
    
    try:
        # Create test documents
        documents = [
            Document(text="This is sentence one. This is sentence two. This is sentence three. This is sentence four. This is sentence five.")
        ]
        
        window_sizes = [1, 2, 3, 5]
        query = "What sentences are mentioned?"
        
        for window_size in window_sizes:
            print(f"\n📊 Testing window size {window_size}:")
            
            retriever = SentenceWindowRetriever(
                documents=documents,
                window_size=window_size
            )
            
            result = retriever.query(query, show_details=False)
            print(f"  Window size: {window_size}")
            print(f"  Response length: {len(result['response'])}")
            print(f"  Sources: {result['metadata']['num_sources']}")
            print(f"  Time: {result['metadata']['total_time']}s")
        
    except Exception as e:
        print(f"❌ Window size test error: {str(e)}")

# ---------- UTILITY FUNCTIONS -----------------------------------------------

def create_sentence_window_retriever_from_latest_batch(
    window_size: int = 3,
    chunk_size: int = 1024
) -> SentenceWindowRetriever:
    """Create sentence window retriever from latest batch."""
    # This is a simplified version - in practice, you'd load original documents
    documents = [
        Document(text="Sample document for sentence window retrieval testing.")
    ]
    
    return SentenceWindowRetriever(
        documents=documents,
        window_size=window_size,
        sentence_splitter_chunk_size=chunk_size
    )

def create_advanced_decoupler_from_latest_batch() -> AdvancedChunkDecoupler:
    """Create advanced decoupler from latest batch."""
    index = create_index_from_latest_batch(use_chunks=True)
    return AdvancedChunkDecoupler(index)

# ---------- ENTRY POINT -----------------------------------------------------

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "window":
        test_sentence_window_sizes()
    else:
        demonstrate_chunk_decoupling() 