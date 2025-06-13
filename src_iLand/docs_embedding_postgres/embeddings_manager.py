"""
Enhanced Embeddings Manager for iLand BGE-M3 Embedding with Section-Based Chunking

This module provides the main management interface for the PRD v2.0 implementation:
- BGE-M3 model for 100% local processing
- Section-based chunking reducing chunks from ~169 to ~6 per document  
- Complete metadata preservation with security compliance
- Zero external data transmission for government deployment

Author: AI Assistant (PRD v2.0 Implementation)
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

# Import LlamaIndex for document processing
from llama_index.core import Document, Settings
from llama_index.core.schema import TextNode

# Import from internal modules
try:
    from .embedding_processor import BGEEmbeddingProcessor
    from .db_utils import PostgresManager
except ImportError:
    from embedding_processor import BGEEmbeddingProcessor
    from db_utils import PostgresManager

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class EmbeddingsManager:
    """
    Enhanced Embeddings Manager implementing PRD v2.0 specifications.
    
    Manages the complete pipeline from documents to embeddings using:
    - BGE-M3 model for multilingual Thai support
    - Section-based chunking for efficient document representation
    - 100% local processing with comprehensive security compliance
    """
    
    def __init__(
        self,
        # BGE-M3 model configuration (PRD v2.0 defaults)
        bge_model: str = os.getenv("BGE_MODEL", "bge-m3"),
        cache_folder: str = os.getenv("CACHE_FOLDER", "./cache/bge_models"),
        
        # Section-based chunking configuration
        chunk_size: int = int(os.getenv("CHUNK_SIZE", "512")),
        chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50")),
        min_section_size: int = 50,
        enable_section_chunking: bool = True,
        
        # Processing configuration
        batch_size: int = int(os.getenv("BATCH_SIZE", "32")),
        device: str = "auto",
        
        # Security configuration (PRD v2.0)
        allow_external_apis: bool = False,  # Enforce local-only processing
        audit_logging: bool = True
    ):
        """
        Initialize the Enhanced Embeddings Manager.
        
        Args:
            bge_model: BGE model to use (default: bge-m3 for Thai support)
            cache_folder: Directory to cache BGE models
            chunk_size: Maximum chunk size in tokens
            chunk_overlap: Overlap between chunks
            min_section_size: Minimum size for standalone sections
            enable_section_chunking: Enable section-based parsing
            batch_size: Batch size for processing
            device: Device to use (auto, cuda, cpu)
            allow_external_apis: Allow external API calls (DISABLED by default)
            audit_logging: Enable audit logging for compliance
        """
        # Configuration
        self.bge_model = bge_model
        self.cache_folder = cache_folder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_section_size = min_section_size
        self.enable_section_chunking = enable_section_chunking
        self.batch_size = batch_size
        self.device = device
        self.allow_external_apis = allow_external_apis
        self.audit_logging = audit_logging
        
        # Initialize BGE embedding processor (PRD v2.0)
        self.embedding_processor = self._initialize_embedding_processor()
        
        # Create output directory for file backups
        self.output_dir = Path("./data/output_bge")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
        # Initialize database manager
        self.db_manager = None
        
        logger.info("🎯 Enhanced Embeddings Manager initialized (PRD v2.0)")
        logger.info(f"   BGE Model: {self.bge_model}")
        logger.info(f"   Section chunking: {'ENABLED' if self.enable_section_chunking else 'DISABLED'}")
        logger.info(f"   External APIs: {'DISABLED' if not self.allow_external_apis else 'ENABLED'}")
        logger.info(f"   Cache folder: {self.cache_folder}")
    
    def _initialize_embedding_processor(self) -> BGEEmbeddingProcessor:
        """Initialize the BGE-M3 embedding processor with PRD v2.0 configuration."""
        logger.info(f"Initializing BGE-M3 processor: {self.bge_model}")
        
        processor = BGEEmbeddingProcessor(
            bge_model_key=self.bge_model,
            model_cache_dir=self.cache_folder,
            device=self.device,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            min_section_size=self.min_section_size,
            enable_section_chunking=self.enable_section_chunking,
            batch_size=self.batch_size,
            allow_external_apis=self.allow_external_apis,
            audit_logging=self.audit_logging
        )
        
        model_info = processor.get_model_info()
        logger.info(f"BGE-M3 processor initialized: {model_info['model_name']}")
        logger.info(f"Embedding dimension: {model_info['dimension']}")
        logger.info(f"Max text length: {model_info['max_length']}")
        logger.info(f"Local processing: {model_info['local_processing']}")
        
        return processor
    
    def get_model_dimension(self) -> int:
        """Get the dimension of the BGE embedding model."""
        return self.embedding_processor.get_model_dimension()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return self.embedding_processor.get_model_info()
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Process documents using BGE-M3 with section-based chunking.
        
        This method implements the complete PRD v2.0 pipeline and returns
        embeddings in the format expected by the database manager.
        
        Args:
            documents: List of document dictionaries with content and metadata
            
        Returns:
            Tuple of (chunk_embeddings, summary_embeddings, indexnode_embeddings)
            Note: In PRD v2.0, we focus on chunks with section-based processing
        """
        logger.info(f"🚀 Processing {len(documents)} documents with BGE-M3 + Section chunking")
        
        # Process documents to embedded nodes using the new processor
        embedded_nodes = self.embedding_processor.process_and_embed_documents(documents)
        
        if not embedded_nodes:
            logger.warning("No embedded nodes created from documents")
            return [], [], []
        
        # Convert TextNodes to the format expected by the database manager
        chunk_embeddings = self._convert_nodes_to_embeddings(embedded_nodes, "chunk")
        
        # For PRD v2.0, we focus on section-based chunks
        # Summary and index node embeddings can be generated from key_info chunks
        summary_embeddings = self._extract_summary_embeddings(embedded_nodes)
        indexnode_embeddings = self._extract_indexnode_embeddings(embedded_nodes)
        
        # Get processing statistics
        stats = self.embedding_processor.get_processing_statistics()
        
        # Log comprehensive results
        logger.info("🎉 BGE-M3 Processing completed successfully!")
        logger.info("=" * 60)
        logger.info(f"📊 PROCESSING RESULTS:")
        logger.info(f"   - Input documents: {len(documents)}")
        logger.info(f"   - Chunk embeddings: {len(chunk_embeddings)}")
        logger.info(f"   - Summary embeddings: {len(summary_embeddings)}")
        logger.info(f"   - Index node embeddings: {len(indexnode_embeddings)}")
        logger.info(f"   - Average chunks per doc: {stats['avg_chunks_per_doc']:.1f}")
        logger.info(f"   - Section-based chunks: {stats['section_chunks']}")
        logger.info(f"   - Fallback chunks: {stats['fallback_chunks']}")
        logger.info("=" * 60)
        
        # Verify PRD v2.0 compliance
        self._verify_prd_compliance(stats, len(documents))
        
        return chunk_embeddings, summary_embeddings, indexnode_embeddings
    
    def _convert_nodes_to_embeddings(self, nodes: List[TextNode], embedding_type: str) -> List[Dict]:
        """Convert TextNodes to embedding dictionaries for database storage."""
        embeddings = []
        
        for i, node in enumerate(nodes):
            if not hasattr(node, 'embedding') or node.embedding is None:
                logger.warning(f"Node {i} missing embedding, skipping")
                continue
            
            embedding_data = {
                "node_id": node.node_id,
                "text": node.text,
                "text_length": len(node.text),
                "embedding_vector": node.embedding,
                "embedding_dim": len(node.embedding),
                "metadata": dict(node.metadata),
                "type": embedding_type,
                
                # Additional PRD v2.0 metadata
                "embedding_model": self.bge_model,
                "processing_method": node.metadata.get("processing_method", "section_based"),
                "section_type": node.metadata.get("section", "unknown"),
                "chunk_type": node.metadata.get("chunk_type", "section"),
                "processed_locally": True,
                "prd_version": "2.0"
            }
            
            # Extract deed_id for database compatibility
            if "deed_id" in node.metadata:
                embedding_data["deed_id"] = node.metadata["deed_id"]
            
            embeddings.append(embedding_data)
        
        return embeddings
    
    def _extract_summary_embeddings(self, nodes: List[TextNode]) -> List[Dict]:
        """Extract summary embeddings from key_info chunks."""
        summary_embeddings = []
        
        # Group nodes by deed_id
        deed_nodes = {}
        for node in nodes:
            deed_id = node.metadata.get("deed_id", "unknown")
            if deed_id not in deed_nodes:
                deed_nodes[deed_id] = []
            deed_nodes[deed_id].append(node)
        
        # Create summary from key_info chunk for each document
        for deed_id, doc_nodes in deed_nodes.items():
            # Find key_info chunk (most important for summary)
            key_info_chunk = None
            for node in doc_nodes:
                if node.metadata.get("chunk_type") == "key_info":
                    key_info_chunk = node
                    break
            
            if key_info_chunk and hasattr(key_info_chunk, 'embedding'):
                summary_data = {
                    "node_id": f"summary_{deed_id}",
                    "deed_id": deed_id,
                    "summary_text": key_info_chunk.text,
                    "summary_length": len(key_info_chunk.text),
                    "embedding_vector": key_info_chunk.embedding,
                    "embedding_dim": len(key_info_chunk.embedding),
                    "metadata": {
                        **key_info_chunk.metadata,
                        "type": "summary",
                        "derived_from": "key_info_chunk",
                        "prd_version": "2.0"
                    },
                    "type": "summary",
                    "embedding_model": self.bge_model
                }
                summary_embeddings.append(summary_data)
        
        return summary_embeddings
    
    def _extract_indexnode_embeddings(self, nodes: List[TextNode]) -> List[Dict]:
        """Extract index node embeddings from primary chunks."""
        indexnode_embeddings = []
        
        # Group nodes by deed_id
        deed_nodes = {}
        for node in nodes:
            deed_id = node.metadata.get("deed_id", "unknown")
            if deed_id not in deed_nodes:
                deed_nodes[deed_id] = []
            deed_nodes[deed_id].append(node)
        
        # Create index node from primary chunk for each document
        for deed_id, doc_nodes in deed_nodes.items():
            # Find primary chunk
            primary_chunk = None
            for node in doc_nodes:
                if node.metadata.get("is_primary_chunk") or node.metadata.get("chunk_type") == "key_info":
                    primary_chunk = node
                    break
            
            if not primary_chunk and doc_nodes:
                # Use first chunk as fallback
                primary_chunk = doc_nodes[0]
            
            if primary_chunk and hasattr(primary_chunk, 'embedding'):
                indexnode_data = {
                    "node_id": f"indexnode_{deed_id}",
                    "deed_id": deed_id,
                    "text": f"Document: {deed_id}\n\n{primary_chunk.text}",
                    "text_length": len(primary_chunk.text),
                    "embedding_vector": primary_chunk.embedding,
                    "embedding_dim": len(primary_chunk.embedding),
                    "metadata": {
                        **primary_chunk.metadata,
                        "type": "indexnode",
                        "derived_from": "primary_chunk",
                        "prd_version": "2.0"
                    },
                    "type": "indexnode",
                    "embedding_model": self.bge_model
                }
                indexnode_embeddings.append(indexnode_data)
        
        return indexnode_embeddings
    
    def _verify_prd_compliance(self, stats: Dict[str, Any], input_docs: int):
        """Verify that processing meets PRD v2.0 compliance requirements."""
        compliance_issues = []
        
        # Check chunk efficiency (should be ≤ 15 chunks per doc)
        avg_chunks = stats.get("avg_chunks_per_doc", 0)
        if avg_chunks > 15:
            compliance_issues.append(f"High chunk count: {avg_chunks:.1f} per doc (target: ≤15)")
        
        # Check section-based processing
        if not stats.get("section_chunking_enabled", False):
            compliance_issues.append("Section-based chunking disabled")
        
        # Check fallback usage
        fallback_chunks = stats.get("fallback_chunks", 0)
        if fallback_chunks > 0:
            compliance_issues.append(f"{fallback_chunks} fallback chunks used (target: 0)")
        
        # Check local processing
        if not stats.get("local_processing_only", False):
            compliance_issues.append("External API usage detected")
        
        # Report compliance status
        if compliance_issues:
            logger.warning("⚠️ PRD v2.0 COMPLIANCE ISSUES:")
            for issue in compliance_issues:
                logger.warning(f"   - {issue}")
        else:
            logger.info("✅ PRD v2.0 COMPLIANCE: All requirements satisfied")
            logger.info("   - Efficient chunking achieved")
            logger.info("   - Section-based processing used")
            logger.info("   - 100% local processing verified")
            logger.info("   - Zero external API calls")
    
    def save_embeddings_to_files(self, chunk_embeddings: List[Dict], summary_embeddings: List[Dict], indexnode_embeddings: List[Dict]) -> None:
        """Save embeddings to local files as backup (PRD v2.0 compliance)."""
        try:
            import json
            
            # Save comprehensive backup with metadata
            backup_data = {
                "metadata": {
                    "prd_version": "2.0",
                    "model": self.bge_model,
                    "processing_timestamp": self.embedding_processor.processing_stats.get("processing_timestamp"),
                    "total_documents": self.embedding_processor.processing_stats.get("documents_processed", 0),
                    "total_chunks": len(chunk_embeddings),
                    "section_chunking_enabled": self.enable_section_chunking,
                    "local_processing_only": not self.allow_external_apis,
                    "avg_chunks_per_doc": self.embedding_processor.processing_stats.get("avg_chunks_per_doc", 0)
                },
                "chunk_embeddings": chunk_embeddings,
                "summary_embeddings": summary_embeddings,
                "indexnode_embeddings": indexnode_embeddings
            }
            
            backup_file = self.output_dir / f"bge_embeddings_backup_{self.bge_model.replace('/', '_')}.json"
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"💾 Embeddings backup saved to: {backup_file}")
            logger.info(f"   - Chunk embeddings: {len(chunk_embeddings)}")
            logger.info(f"   - Summary embeddings: {len(summary_embeddings)}")
            logger.info(f"   - Index node embeddings: {len(indexnode_embeddings)}")
            
        except Exception as e:
            logger.error(f"Failed to save embeddings backup: {e}")
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        return self.embedding_processor.get_processing_statistics()


# Factory function for backward compatibility
def create_embeddings_manager(**kwargs) -> EmbeddingsManager:
    """
    Factory function to create an EmbeddingsManager with PRD v2.0 defaults.
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        EmbeddingsManager instance
    """
    return EmbeddingsManager(**kwargs) 