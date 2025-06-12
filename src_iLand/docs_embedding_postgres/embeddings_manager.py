"""
Embeddings Manager for iLand BGE Embedding

This module provides functions for managing embeddings generation and index building
for the BGE embedding pipeline.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

# Import LlamaIndex for document processing
from llama_index.core import Document, DocumentSummaryIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode

# Import from internal modules
try:
    from .bge_embedding_processor import create_bge_embedding_processor
    from .file_storage import EmbeddingStorage
except ImportError:
    from bge_embedding_processor import create_bge_embedding_processor
    from file_storage import EmbeddingStorage

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class EmbeddingsManager:
    """Manages the process of creating embeddings from documents."""
    
    def __init__(
        self,
        # BGE model configuration
        bge_model: str = os.getenv("BGE_MODEL", "bge-small-en-v1.5"),
        cache_folder: str = os.getenv("CACHE_FOLDER", "./cache/bge_models"),
        
        # Processing configuration
        chunk_size: int = int(os.getenv("CHUNK_SIZE", "512")),
        chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50")),
        batch_size: int = int(os.getenv("BATCH_SIZE", "20"))
    ):
        """Initialize the Embeddings Manager."""
        # BGE model configuration
        self.bge_model = bge_model
        self.cache_folder = cache_folder
        
        # Processing configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        
        # Initialize components
        self.storage = EmbeddingStorage()
        self.embedding_processor = self._initialize_embedding_processor()
        
        # Create output dir for files
        self.output_dir = Path("./data/output_bge")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _initialize_embedding_processor(self):
        """Initialize the BGE embedding processor."""
        logger.info(f"Initializing BGE model: {self.bge_model} from cache: {self.cache_folder}")
        processor = create_bge_embedding_processor(
            model_name=self.bge_model,
            cache_folder=self.cache_folder
        )
        model_info = processor.get_model_info()
        logger.info(f"BGE model initialized: {model_info['model_name']}")
        logger.info(f"Embedding dimension: {model_info['dimension']}")
        logger.info(f"Max text length: {model_info['max_length']}")
        return processor
    
    def get_model_dimension(self) -> int:
        """Get the dimension of the embedding model."""
        model_info = self.embedding_processor.get_model_info()
        return model_info['dimension']
    
    def build_document_summary_index(self, documents: List[Dict[str, Any]]) -> DocumentSummaryIndex:
        """Build a DocumentSummaryIndex from the fetched documents."""
        logger.info(f"Building DocumentSummaryIndex from {len(documents)} documents...")
        
        # Convert to LlamaIndex Document objects
        llama_docs = []
        for doc in documents:
            llama_doc = Document(
                text=doc["content"],
                metadata=doc["metadata"]
            )
            llama_docs.append(llama_doc)
        
        # Configure models
        Settings.embed_model = self.embedding_processor.embed_model
        
        # Create node parser
        splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Build index (this will generate embeddings)
        doc_summary_index = DocumentSummaryIndex.from_documents(
            llama_docs,
            transformations=[splitter],
            show_progress=True,
        )
        
        logger.info(f"Built DocumentSummaryIndex with {len(doc_summary_index.docstore.docs)} nodes")
        return doc_summary_index
    
    def build_index_nodes(self, doc_summary_index: DocumentSummaryIndex) -> List[IndexNode]:
        """Build IndexNodes for recursive retrieval pattern."""
        logger.info(f"Building IndexNodes for recursive retrieval...")
        
        doc_index_nodes = []
        doc_ids = list(doc_summary_index.ref_doc_info.keys())
        
        for i, doc_id in enumerate(doc_ids):
            doc_info = doc_summary_index.ref_doc_info[doc_id]
            deed_id = doc_info.metadata.get("deed_id", f"Document {i+1}")
            
            try:
                doc_summary = doc_summary_index.get_document_summary(doc_id)
            except Exception:
                # Fallback if summary generation fails
                doc_summary = "Summary not available"
            
            # Get chunks for this document  
            doc_chunks = [node for node_id, node in doc_summary_index.docstore.docs.items()
                         if hasattr(node, 'ref_doc_id') and node.ref_doc_id == doc_id and
                         not getattr(node, 'is_summary', False)]
            
            if doc_chunks:
                # Preserve metadata for structured retrieval
                original_metadata = doc_info.metadata.copy()
                original_metadata.update({
                    "doc_id": doc_id,
                    "deed_id": deed_id,
                    "chunk_count": len(doc_chunks),
                    "type": "document_summary",
                    "embedding_provider": self.embedding_processor.provider,
                    "embedding_model": self.embedding_processor.config.get("model_name")
                })
                
                index_node = IndexNode(
                    text=f"Document: {deed_id}\n\nSummary: {doc_summary}",
                    index_id=f"doc_{i}",
                    metadata=original_metadata
                )
                doc_index_nodes.append(index_node)
        
        logger.info(f"Created {len(doc_index_nodes)} IndexNodes")
        return doc_index_nodes
    
    def extract_embeddings(self, doc_summary_index: DocumentSummaryIndex, doc_index_nodes: List[IndexNode]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Extract embeddings for chunks, summaries and index nodes."""
        logger.info(f"Extracting embeddings with BGE model: {self.bge_model}")
        
        # Extract chunk embeddings
        chunk_embeddings = self.embedding_processor.extract_chunk_embeddings(doc_summary_index, 1)
        logger.info(f"Extracted {len(chunk_embeddings)} chunk embeddings")
        
        # Extract summary embeddings
        summary_embeddings = self.embedding_processor.extract_summary_embeddings(doc_summary_index, 1)
        logger.info(f"Extracted {len(summary_embeddings)} summary embeddings")
        
        # Extract index node embeddings
        indexnode_embeddings = self.embedding_processor.extract_indexnode_embeddings(doc_index_nodes, 1)
        logger.info(f"Extracted {len(indexnode_embeddings)} index node embeddings")
        
        return chunk_embeddings, summary_embeddings, indexnode_embeddings
    
    def save_embeddings_to_files(self, chunk_embeddings: List[Dict], summary_embeddings: List[Dict], indexnode_embeddings: List[Dict]) -> None:
        """Save embeddings to local files as backup."""
        # Save to local files as backup
        self.storage.save_batch_embeddings(
            self.output_dir, 1, indexnode_embeddings, chunk_embeddings, summary_embeddings
        )
        logger.info(f"Saved embeddings to local files in {self.output_dir}")
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Process documents and extract embeddings."""
        # Step 1: Build document summary index
        doc_summary_index = self.build_document_summary_index(documents)
        
        # Step 2: Build index nodes
        doc_index_nodes = self.build_index_nodes(doc_summary_index)
        
        # Step 3: Extract embeddings
        chunk_embeddings, summary_embeddings, indexnode_embeddings = self.extract_embeddings(
            doc_summary_index, doc_index_nodes
        )
        
        # Step 4: Save embeddings to files (backup)
        self.save_embeddings_to_files(chunk_embeddings, summary_embeddings, indexnode_embeddings)
        
        return chunk_embeddings, summary_embeddings, indexnode_embeddings 