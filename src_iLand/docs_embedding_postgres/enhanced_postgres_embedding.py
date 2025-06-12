"""
Enhanced PostgreSQL Embedding Pipeline for iLand RAG System

This module provides a comprehensive embedding pipeline that loads documents from PostgreSQL,
processes them with the same sophisticated techniques as the local version (including
section-based chunking, metadata extraction), and stores embeddings systematically in
PostgreSQL with proper indexing for fast retrieval.

Key Features:
- Multi-model embedding support (BGE-M3 + OpenAI fallback)
- Section-based chunking for Thai land deed documents
- Rich metadata extraction and categorization
- Systematic storage with proper database schema
- Hierarchical document structure (summaries, chunks, index nodes)
- Production-ready error handling and logging
"""

import os
import sys
import time
import logging
import argparse
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import psycopg2
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LlamaIndex imports
from llama_index.core import Document, DocumentSummaryIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, IndexNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Import local components
try:
    from .metadata_extractor import iLandMetadataExtractor
    from .standalone_section_parser import StandaloneLandDeedSectionParser
    from .db_utils import PostgresManager
    from .multi_model_embedding_processor import MultiModelEmbeddingProcessor
    from .embedding_config import get_embedding_config, get_config_from_environment
except ImportError:
    from metadata_extractor import iLandMetadataExtractor
    from standalone_section_parser import StandaloneLandDeedSectionParser
    from db_utils import PostgresManager
    try:
        from multi_model_embedding_processor import MultiModelEmbeddingProcessor
        from embedding_config import get_embedding_config, get_config_from_environment
        MULTI_MODEL_SUPPORT = True
    except ImportError:
        MULTI_MODEL_SUPPORT = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedPostgresEmbeddingPipeline:
    """
    Enhanced PostgreSQL embedding pipeline with systematic storage and processing.
    Follows the same patterns as the local embedding pipeline but stores everything in PostgreSQL.
    """
    
    def __init__(
        self,
        # Database configuration
        db_name: str = os.getenv("DB_NAME", "iland-vector-dev"),
        db_user: str = os.getenv("DB_USER", "vector_user_dev"), 
        db_password: str = os.getenv("DB_PASSWORD", "akqVvIJvVqe7Jr1"),
        db_host: str = os.getenv("DB_HOST", "10.4.102.11"),
        db_port: int = int(os.getenv("DB_PORT", "5432")),
        
        # Processing configuration
        chunk_size: int = int(os.getenv("CHUNK_SIZE", "512")),
        chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50")),
        batch_size: int = int(os.getenv("API_BATCH_SIZE", "20")),
        
        # Embedding configuration
        embed_model_name: str = os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        llm_model_name: str = os.getenv("LLM_MODEL", "gpt-4o-mini"),
        
        # Enhanced processing settings
        enable_section_chunking: bool = True,
        enable_multi_model: bool = True,
        min_section_size: int = 50
    ):
        """Initialize the Enhanced PostgreSQL Embedding Pipeline."""
        
        # Database configuration
        self.db_config = {
            "db_name": db_name,
            "db_user": db_user,
            "db_password": db_password,
            "db_host": db_host,
            "db_port": db_port
        }
        
        # Processing configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.enable_section_chunking = enable_section_chunking
        self.enable_multi_model = enable_multi_model and MULTI_MODEL_SUPPORT
        self.min_section_size = min_section_size
        
        # Embedding configuration
        self.embed_model_name = embed_model_name
        self.llm_model_name = llm_model_name
        
        # Validate API keys
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not found")
        
        # Initialize components
        self._initialize_components()
        
        # Processing statistics
        self.stats = {
            "documents_processed": 0,
            "nodes_created": 0,
            "section_chunks": 0,
            "sentence_chunks": 0,
            "summaries_generated": 0,
            "index_nodes_created": 0,
            "embeddings_generated": 0,
            "db_insertions": 0
        }
        
        logger.info(f"Enhanced PostgreSQL Embedding Pipeline initialized")
        logger.info(f"Multi-model support: {'Enabled' if self.enable_multi_model else 'Disabled'}")
        logger.info(f"Section chunking: {'Enabled' if self.enable_section_chunking else 'Disabled'}")
    
    def _initialize_components(self):
        """Initialize all processing components."""
        
        # Initialize database manager
        self.db_manager = PostgresManager(**self.db_config)
        
        # Initialize metadata extractor
        self.metadata_extractor = iLandMetadataExtractor()
        
        # Initialize multi-model embedding processor if available
        if self.enable_multi_model:
            try:
                # Get embedding configuration with environment overrides
                embedding_config = {
                    "default_provider": "BGE_M3",
                    "providers": {
                        "BGE_M3": {
                            "model_name": "BAAI/bge-m3",
                            "device": "auto",
                            "batch_size": 32,
                            "normalize": True
                        },
                        "OPENAI": {
                            "model_name": self.embed_model_name,
                            "api_key_env": "OPENAI_API_KEY",
                            "batch_size": self.batch_size
                        }
                    },
                    "fallback_enabled": True,
                    "fallback_order": ["BGE_M3", "OPENAI"]
                }
                
                # Apply environment overrides
                env_overrides = get_config_from_environment()
                if env_overrides:
                    embedding_config.update(env_overrides)
                
                self.embedding_processor = MultiModelEmbeddingProcessor(embedding_config)
                self.embed_model = self.embedding_processor.get_primary_embed_model()
                logger.info(f"Multi-model embedding processor initialized")
                
            except Exception as e:
                logger.warning(f"Could not initialize multi-model processor: {e}")
                logger.info("Falling back to OpenAI-only embedding")
                self.enable_multi_model = False
                self._initialize_openai_embedding()
        else:
            self._initialize_openai_embedding()
        
        # Initialize section parser if enabled
        if self.enable_section_chunking:
            try:
                self.section_parser = StandaloneLandDeedSectionParser(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    min_section_size=self.min_section_size
                )
                logger.info("Section-based chunking parser initialized")
            except Exception as e:
                logger.warning(f"Could not initialize section parser: {e}")
                logger.info("Falling back to sentence splitting")
                self.enable_section_chunking = False
                self._initialize_sentence_parser()
        else:
            self._initialize_sentence_parser()
        
        # Initialize LLM for summaries
        self.llm = OpenAI(model=self.llm_model_name, api_key=os.getenv("OPENAI_API_KEY"))
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        logger.info("All components initialized successfully")
    
    def _initialize_openai_embedding(self):
        """Initialize OpenAI-only embedding model."""
        self.embed_model = OpenAIEmbedding(
            model=self.embed_model_name,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.embedding_processor = None
        logger.info(f"OpenAI embedding model initialized: {self.embed_model_name}")
    
    def _initialize_sentence_parser(self):
        """Initialize sentence-based parser as fallback."""
        self.sentence_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        logger.info("Sentence splitting parser initialized")
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension for database setup."""
        if self.enable_multi_model and self.embedding_processor:
            return self.embedding_processor.get_embedding_dimension()
        else:
            # Standard OpenAI dimensions
            dimensions = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536
            }
            return dimensions.get(self.embed_model_name, 1536)
    
    def setup_database(self) -> bool:
        """Setup database tables with proper schema."""
        logger.info("Setting up database schema...")
        
        if not self.db_manager.connect():
            logger.error("Failed to connect to database")
            return False
        
        embed_dim = self.get_embedding_dimension()
        logger.info(f"Setting up tables with embedding dimension: {embed_dim}")
        
        if self.db_manager.setup_tables(embed_dim):
            logger.info("Database schema setup completed")
            return True
        else:
            logger.error("Failed to setup database schema")
            return False
    
    def fetch_documents(self, limit: Optional[int] = None, status_filter: str = None) -> List[Dict[str, Any]]:
        """
        Fetch documents from the source database with optional filtering.
        
        Args:
            limit: Maximum number of documents to fetch
            status_filter: Filter by embedding_status ('pending', 'completed', 'failed')
        """
        logger.info(f"Fetching documents from database (limit: {limit}, status: {status_filter})")
        
        if not self.db_manager.connection and not self.db_manager.connect():
            return []
        
        documents = []
        try:
            cursor = self.db_manager.connection.cursor()
            
            # Build query with optional filters
            query = """
                SELECT deed_id, md_string, raw_metadata, extracted_metadata, 
                       province, district, land_use_category, deed_type_category, area_category
                FROM iland_md_data
            """
            
            conditions = []
            params = []
            
            if status_filter:
                conditions.append("embedding_status = %s")
                params.append(status_filter)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY created_at"
            
            if limit:
                query += " LIMIT %s"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            for row in rows:
                deed_id, md_string, raw_metadata, extracted_metadata, province, district, land_use_cat, deed_type_cat, area_cat = row
                
                # Parse metadata
                try:
                    raw_meta = json.loads(raw_metadata) if raw_metadata else {}
                    extracted_meta = json.loads(extracted_metadata) if extracted_metadata else {}
                except json.JSONDecodeError:
                    raw_meta = {}
                    extracted_meta = {}
                
                # Combine metadata
                combined_metadata = {
                    **raw_meta,
                    **extracted_meta,
                    "deed_id": deed_id,
                    "province": province,
                    "district": district,
                    "land_use_category": land_use_cat,
                    "deed_type_category": deed_type_cat,
                    "area_category": area_cat,
                    "source": "postgresql_enhanced_pipeline"
                }
                
                doc = {
                    "deed_id": deed_id,
                    "content": md_string,
                    "metadata": combined_metadata
                }
                documents.append(doc)
            
            logger.info(f"Fetched {len(documents)} documents from database")
            cursor.close()
            return documents
            
        except Exception as e:
            logger.error(f"Error fetching documents: {e}")
            return []
    
    def process_documents_to_nodes(self, documents: List[Dict[str, Any]]) -> Tuple[List[TextNode], List[IndexNode], DocumentSummaryIndex]:
        """
        Process documents into nodes using enhanced processing with section-based chunking.
        Returns (chunk_nodes, index_nodes, document_summary_index)
        """
        logger.info(f"Processing {len(documents)} documents to nodes...")
        
        # Convert to LlamaIndex Documents
        llama_docs = []
        for doc in documents:
            llama_doc = Document(
                text=doc['content'],
                metadata=doc['metadata']
            )
            llama_docs.append(llama_doc)
        
        # Build DocumentSummaryIndex for hierarchical structure
        logger.info("Building DocumentSummaryIndex...")
        doc_summary_index = DocumentSummaryIndex.from_documents(
            llama_docs,
            llm=self.llm,
            embed_model=self.embed_model,
            show_progress=True
        )
        
        self.stats["summaries_generated"] = len(doc_summary_index.ref_doc_info)
        logger.info(f"Generated {self.stats['summaries_generated']} document summaries")
        
        # Extract nodes using enhanced processing
        all_chunk_nodes = []
        
        for i, doc in enumerate(documents):
            logger.info(f"Processing document {i+1}/{len(documents)}: {doc['deed_id']}")
            
            try:
                # Enhanced metadata extraction
                enhanced_metadata = self.metadata_extractor.extract_from_content(doc['content'])
                doc['metadata'].update(enhanced_metadata)
                
                # Process with section-based chunking if enabled
                if self.enable_section_chunking:
                    nodes = self.section_parser.parse_document_to_sections(
                        doc['content'],
                        doc['metadata']
                    )
                    
                    section_count = sum(1 for node in nodes if node.metadata.get("chunk_type") == "section")
                    self.stats["section_chunks"] += section_count
                    
                    logger.info(f"  Created {len(nodes)} section-based chunks ({section_count} sections)")
                else:
                    # Fallback to sentence splitting
                    llama_doc = Document(text=doc['content'], metadata=doc['metadata'])
                    nodes = self.sentence_parser.get_nodes_from_documents([llama_doc])
                    
                    # Add enhanced metadata
                    for node in nodes:
                        node.metadata.update(doc['metadata'])
                        node.metadata["chunk_type"] = "sentence"
                        node.metadata["chunking_strategy"] = "sentence_splitting"
                    
                    self.stats["sentence_chunks"] += len(nodes)
                    logger.info(f"  Created {len(nodes)} sentence-based chunks")
                
                all_chunk_nodes.extend(nodes)
                self.stats["nodes_created"] += len(nodes)
                self.stats["documents_processed"] += 1
                
            except Exception as e:
                logger.error(f"Error processing document {doc['deed_id']}: {e}")
                continue
        
        # Build IndexNodes for recursive retrieval
        logger.info("Building IndexNodes for recursive retrieval...")
        index_nodes = self._build_index_nodes(doc_summary_index)
        
        logger.info(f"Enhanced processing completed:")
        logger.info(f"  Documents processed: {self.stats['documents_processed']}")
        logger.info(f"  Total chunk nodes: {len(all_chunk_nodes)}")
        logger.info(f"  Section chunks: {self.stats['section_chunks']}")
        logger.info(f"  Sentence chunks: {self.stats['sentence_chunks']}")
        logger.info(f"  Index nodes: {len(index_nodes)}")
        
        return all_chunk_nodes, index_nodes, doc_summary_index
    
    def _build_index_nodes(self, doc_summary_index: DocumentSummaryIndex) -> List[IndexNode]:
        """Build IndexNodes for recursive retrieval pattern."""
        index_nodes = []
        doc_ids = list(doc_summary_index.ref_doc_info.keys())
        
        for i, doc_id in enumerate(doc_ids):
            doc_info = doc_summary_index.ref_doc_info[doc_id]
            deed_id = doc_info.metadata.get("deed_id", f"Document {i+1}")
            
            try:
                doc_summary = doc_summary_index.get_document_summary(doc_id)
            except Exception:
                doc_summary = "Summary not available"
            
            # Get chunks for this document
            doc_chunks = [
                node for node_id, node in doc_summary_index.docstore.docs.items()
                if hasattr(node, 'ref_doc_id') and node.ref_doc_id == doc_id
                and not getattr(node, 'is_summary', False)
            ]
            
            if doc_chunks:
                # Enhanced metadata for index node
                index_metadata = doc_info.metadata.copy()
                index_metadata.update({
                    "doc_id": doc_id,
                    "deed_id": deed_id,
                    "chunk_count": len(doc_chunks),
                    "type": "document_summary",
                    "processing_timestamp": datetime.now().isoformat(),
                    "embedding_provider": "multi_model" if self.enable_multi_model else "openai"
                })
                
                index_node = IndexNode(
                    text=f"Document: {deed_id}\n\nSummary: {doc_summary}",
                    index_id=f"idx_doc_{i}",
                    metadata=index_metadata
                )
                index_nodes.append(index_node)
        
        self.stats["index_nodes_created"] = len(index_nodes)
        return index_nodes
    
    def generate_embeddings(self, chunk_nodes: List[TextNode], index_nodes: List[IndexNode], doc_summary_index: DocumentSummaryIndex) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Generate embeddings for all node types."""
        logger.info("Generating embeddings for all node types...")
        
        if self.enable_multi_model and self.embedding_processor:
            # Use multi-model embedding processor
            chunk_embeddings = self.embedding_processor.extract_chunk_embeddings_from_nodes(chunk_nodes, 1)
            index_embeddings = self.embedding_processor.extract_indexnode_embeddings(index_nodes, 1)
            summary_embeddings = self.embedding_processor.extract_summary_embeddings(doc_summary_index, 1)
        else:
            # Use OpenAI-only embedding
            chunk_embeddings = self._extract_openai_embeddings(chunk_nodes, "chunk")
            index_embeddings = self._extract_openai_embeddings(index_nodes, "indexnode") 
            summary_embeddings = self._extract_summary_embeddings_openai(doc_summary_index)
        
        self.stats["embeddings_generated"] = len(chunk_embeddings) + len(index_embeddings) + len(summary_embeddings)
        
        logger.info(f"Generated embeddings:")
        logger.info(f"  Chunk embeddings: {len(chunk_embeddings)}")
        logger.info(f"  Index embeddings: {len(index_embeddings)}")
        logger.info(f"  Summary embeddings: {len(summary_embeddings)}")
        
        return chunk_embeddings, index_embeddings, summary_embeddings
    
    def _extract_openai_embeddings(self, nodes: List, node_type: str) -> List[Dict]:
        """Extract embeddings using OpenAI for fallback."""
        embeddings = []
        
        for i, node in enumerate(nodes):
            try:
                if node_type == "indexnode":
                    text = node.text
                    node_id = node.index_id
                else:
                    text = node.text
                    node_id = node.node_id
                
                embedding_vector = self.embed_model.get_text_embedding(text)
                
                embedding_data = {
                    "node_id": node_id,
                    "text": text,
                    "text_length": len(text),
                    "embedding_vector": embedding_vector,
                    "embedding_dim": len(embedding_vector),
                    "metadata": dict(node.metadata),
                    "type": node_type,
                    "embedding_model": self.embed_model_name
                }
                
                embeddings.append(embedding_data)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"  Processed {i + 1}/{len(nodes)} {node_type} embeddings")
                    
            except Exception as e:
                logger.error(f"Error generating embedding for {node_type} {i}: {e}")
        
        return embeddings
    
    def _extract_summary_embeddings_openai(self, doc_summary_index: DocumentSummaryIndex) -> List[Dict]:
        """Extract summary embeddings using OpenAI."""
        embeddings = []
        doc_ids = list(doc_summary_index.ref_doc_info.keys())
        
        for i, doc_id in enumerate(doc_ids):
            try:
                doc_info = doc_summary_index.ref_doc_info[doc_id]
                summary_text = doc_summary_index.get_document_summary(doc_id)
                
                embedding_vector = self.embed_model.get_text_embedding(summary_text)
                
                embedding_data = {
                    "doc_id": doc_id,
                    "summary_text": summary_text,
                    "text_length": len(summary_text),
                    "embedding_vector": embedding_vector,
                    "embedding_dim": len(embedding_vector),
                    "metadata": dict(doc_info.metadata),
                    "type": "summary",
                    "embedding_model": self.embed_model_name
                }
                
                embeddings.append(embedding_data)
                
            except Exception as e:
                logger.error(f"Error generating summary embedding for doc {i}: {e}")
        
        return embeddings
    
    def save_embeddings_to_database(self, chunk_embeddings: List[Dict], index_embeddings: List[Dict], summary_embeddings: List[Dict]) -> bool:
        """Save all embeddings to PostgreSQL with systematic organization."""
        logger.info("Saving embeddings to PostgreSQL database...")
        
        try:
            # Save to individual tables
            chunks_saved, summaries_saved, index_saved, combined_saved = self.db_manager.save_all_embeddings(
                chunk_embeddings, summary_embeddings, index_embeddings
            )
            
            self.stats["db_insertions"] = chunks_saved + summaries_saved + index_saved
            
            # Update embedding status in source table
            self._update_embedding_status(chunk_embeddings + index_embeddings + summary_embeddings)
            
            logger.info(f"Database storage completed:")
            logger.info(f"  Chunks saved: {chunks_saved}")
            logger.info(f"  Summaries saved: {summaries_saved}")
            logger.info(f"  Index nodes saved: {index_saved}")
            logger.info(f"  Combined table saved: {combined_saved}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving embeddings to database: {e}")
            return False
    
    def _update_embedding_status(self, all_embeddings: List[Dict]):
        """Update embedding status in the source table."""
        if not self.db_manager.connection:
            return
        
        try:
            cursor = self.db_manager.connection.cursor()
            
            # Get unique deed_ids from embeddings
            deed_ids = set()
            for emb in all_embeddings:
                deed_id = emb.get('metadata', {}).get('deed_id')
                if deed_id:
                    deed_ids.add(deed_id)
            
            # Update status for processed documents
            for deed_id in deed_ids:
                cursor.execute(
                    """
                    UPDATE iland_md_data 
                    SET embedding_status = 'completed', embedding_timestamp = %s
                    WHERE deed_id = %s
                    """,
                    (datetime.now(), deed_id)
                )
            
            self.db_manager.connection.commit()
            logger.info(f"Updated embedding status for {len(deed_ids)} documents")
            
        except Exception as e:
            logger.error(f"Error updating embedding status: {e}")
        finally:
            if cursor:
                cursor.close()
    
    def run_pipeline(self, limit: Optional[int] = None, status_filter: str = "pending") -> Dict[str, Any]:
        """Run the complete enhanced embedding pipeline."""
        start_time = time.time()
        logger.info("Starting Enhanced PostgreSQL Embedding Pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Setup database
            if not self.setup_database():
                raise Exception("Failed to setup database")
            
            # Step 2: Fetch documents
            documents = self.fetch_documents(limit=limit, status_filter=status_filter)
            if not documents:
                logger.warning("No documents found to process")
                return {"success": False, "message": "No documents found"}
            
            # Step 3: Process documents to nodes
            chunk_nodes, index_nodes, doc_summary_index = self.process_documents_to_nodes(documents)
            
            # Step 4: Generate embeddings
            chunk_embeddings, index_embeddings, summary_embeddings = self.generate_embeddings(
                chunk_nodes, index_nodes, doc_summary_index
            )
            
            # Step 5: Save to database
            if not self.save_embeddings_to_database(chunk_embeddings, index_embeddings, summary_embeddings):
                raise Exception("Failed to save embeddings to database")
            
            # Final statistics
            duration = time.time() - start_time
            
            logger.info("=" * 60)
            logger.info("Enhanced PostgreSQL Embedding Pipeline COMPLETED")
            logger.info(f"Total processing time: {duration:.2f} seconds")
            logger.info(f"Documents processed: {self.stats['documents_processed']}")
            logger.info(f"Total nodes created: {self.stats['nodes_created']}")
            logger.info(f"Total embeddings generated: {self.stats['embeddings_generated']}")
            logger.info(f"Database insertions: {self.stats['db_insertions']}")
            logger.info("=" * 60)
            
            return {
                "success": True,
                "stats": self.stats,
                "duration": duration,
                "message": f"Successfully processed {self.stats['documents_processed']} documents"
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {"success": False, "error": str(e), "stats": self.stats}
        
        finally:
            # Cleanup
            if hasattr(self, 'db_manager'):
                self.db_manager.close()


def main():
    """Command-line interface for the enhanced embedding pipeline."""
    parser = argparse.ArgumentParser(
        description='Enhanced PostgreSQL Embedding Pipeline for iLand documents'
    )
    
    # Processing arguments
    parser.add_argument('--limit', type=int, default=None,
                        help='Maximum number of documents to process (default: all)')
    parser.add_argument('--status-filter', type=str, default="pending",
                        choices=['pending', 'completed', 'failed', None],
                        help='Filter documents by embedding status (default: pending)')
    
    # Model arguments
    parser.add_argument('--embed-model', type=str, default="text-embedding-3-small",
                        help='OpenAI embedding model (default: text-embedding-3-small)')
    parser.add_argument('--llm-model', type=str, default="gpt-4o-mini",
                        help='LLM model for summaries (default: gpt-4o-mini)')
    
    # Processing arguments
    parser.add_argument('--chunk-size', type=int, default=512,
                        help='Chunk size for text splitting (default: 512)')
    parser.add_argument('--chunk-overlap', type=int, default=50,
                        help='Chunk overlap for text splitting (default: 50)')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='Batch size for API calls (default: 20)')
    
    # Feature flags
    parser.add_argument('--disable-section-chunking', action='store_true',
                        help='Disable section-based chunking')
    parser.add_argument('--disable-multi-model', action='store_true',
                        help='Disable multi-model embedding support')
    
    # Database arguments
    parser.add_argument('--db-host', type=str, default="10.4.102.11",
                        help='Database host (default: 10.4.102.11)')
    parser.add_argument('--db-port', type=int, default=5432,
                        help='Database port (default: 5432)')
    
    args = parser.parse_args()
    
    logger.info("Enhanced PostgreSQL Embedding Pipeline")
    logger.info("=" * 50)
    logger.info(f"Embedding model: {args.embed_model}")
    logger.info(f"LLM model: {args.llm_model}")
    logger.info(f"Chunk size: {args.chunk_size}, Overlap: {args.chunk_overlap}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Document limit: {args.limit or 'All'}")
    logger.info(f"Status filter: {args.status_filter}")
    logger.info(f"Section chunking: {'Disabled' if args.disable_section_chunking else 'Enabled'}")
    logger.info(f"Multi-model: {'Disabled' if args.disable_multi_model else 'Enabled'}")
    logger.info(f"Database: {args.db_host}:{args.db_port}")
    logger.info("=" * 50)
    
    # Create and run pipeline
    pipeline = EnhancedPostgresEmbeddingPipeline(
        db_host=args.db_host,
        db_port=args.db_port,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        embed_model_name=args.embed_model,
        llm_model_name=args.llm_model,
        enable_section_chunking=not args.disable_section_chunking,
        enable_multi_model=not args.disable_multi_model
    )
    
    result = pipeline.run_pipeline(
        limit=args.limit,
        status_filter=args.status_filter
    )
    
    if result["success"]:
        logger.info(f"Pipeline completed successfully: {result['message']}")
        return 0
    else:
        logger.error(f"Pipeline failed: {result.get('error', 'Unknown error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())