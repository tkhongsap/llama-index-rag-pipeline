"""
BGE-based PostgreSQL Embedding Generator for iLand RAG Pipeline

This module loads documents from PostgreSQL database, generates embeddings
using BGE models (local processing, no API calls), and stores the vectors 
in PostgreSQL for vector search.

Uses the same sophisticated processing as the local version:
- Rich metadata extraction with 30+ Thai land deed fields
- Section-based chunking with semantic coherence  
- Enhanced document structure preservation
"""

import os
import sys
import time
import logging
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import psycopg2
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LlamaIndex imports
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo

# BGE embedding support (default)
try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    BGE_AVAILABLE = True
except ImportError:
    BGE_AVAILABLE = False
    print("⚠️ BGE embeddings not available. Install with: pip install llama-index-embeddings-huggingface sentence-transformers")

# OpenAI embedding support (fallback)
try:
    from llama_index.embeddings.openai import OpenAIEmbedding
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI embeddings not available. Install with: pip install llama-index-embeddings-openai")

from llama_index.vector_stores.postgres import PGVectorStore

# Import local components for rich processing
try:
    from .metadata_extractor import iLandMetadataExtractor
    from .standalone_section_parser import StandaloneLandDeedSectionParser
    from .embedding_config import get_config, DEFAULT_CONFIG
except ImportError:
    from metadata_extractor import iLandMetadataExtractor
    from standalone_section_parser import StandaloneLandDeedSectionParser
    from embedding_config import get_config, DEFAULT_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BGEPostgresEmbeddingGenerator:
    """
    BGE-based PostgreSQL embedding generator that uses local BGE models
    with the same sophisticated processing as the local version.
    """
    
    # BGE model configurations
    BGE_MODELS = {
        "bge-small-en-v1.5": {
            "model_name": "BAAI/bge-small-en-v1.5",
            "dimension": 384,
            "max_length": 512,
            "description": "Lightweight, fast BGE model"
        },
        "bge-base-en-v1.5": {
            "model_name": "BAAI/bge-base-en-v1.5", 
            "dimension": 768,
            "max_length": 512,
            "description": "Balanced BGE model"
        },
        "bge-large-en-v1.5": {
            "model_name": "BAAI/bge-large-en-v1.5",
            "dimension": 1024,
            "max_length": 512,
            "description": "High-quality BGE model"
        },
        "bge-m3": {
            "model_name": "BAAI/bge-m3",
            "dimension": 1024,
            "max_length": 8192,
            "description": "Multilingual BGE model (supports Thai)"
        }
    }
    
    def __init__(
        self,
        # Source DB (where markdown content is stored)
        source_db_name: str = os.getenv("DB_NAME", "iland-vector-dev"),
        source_db_user: str = os.getenv("DB_USER", "vector_user_dev"),
        source_db_password: str = os.getenv("DB_PASSWORD", "akqVvIJvVqe7Jr1"),
        source_db_host: str = os.getenv("DB_HOST", "10.4.102.11"),
        source_db_port: int = int(os.getenv("DB_PORT", "5432")),
        source_table_name: str = os.getenv("SOURCE_TABLE", "iland_md_data"),
        
        # Destination vector DB (for embeddings)
        dest_db_name: str = os.getenv("DB_NAME", "iland-vector-dev"),
        dest_db_user: str = os.getenv("DB_USER", "vector_user_dev"),
        dest_db_password: str = os.getenv("DB_PASSWORD", "akqVvIJvVqe7Jr1"),
        dest_db_host: str = os.getenv("DB_HOST", "10.4.102.11"),
        dest_db_port: int = int(os.getenv("DB_PORT", "5432")),
        dest_table_name: str = os.getenv("DEST_TABLE", "iland_embeddings"),
        
        # BGE embedding settings
        bge_model_key: str = os.getenv("BGE_MODEL", "bge-m3"),  # Default to multilingual for Thai
        cache_folder: str = os.getenv("BGE_CACHE_FOLDER", "./cache/bge_models"),
        chunk_size: int = int(os.getenv("CHUNK_SIZE", "512")),
        chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50")),
        batch_size: int = int(os.getenv("BATCH_SIZE", "20")),
        
        # Enhanced processing settings
        enable_section_chunking: bool = True,
        min_section_size: int = 50,
        
        # Fallback to OpenAI if needed
        fallback_to_openai: bool = False
    ):
        """Initialize the BGE PostgreSQL Embedding Generator."""
        
        # Store all configuration
        self.source_db_name = source_db_name
        self.source_db_user = source_db_user
        self.source_db_password = source_db_password
        self.source_db_host = source_db_host
        self.source_db_port = source_db_port
        self.source_table_name = source_table_name
        self.source_conn = None
        
        self.dest_db_name = dest_db_name
        self.dest_db_user = dest_db_user
        self.dest_db_password = dest_db_password
        self.dest_db_host = dest_db_host
        self.dest_db_port = dest_db_port
        self.dest_table_name = dest_table_name
        
        self.bge_model_key = bge_model_key
        self.cache_folder = cache_folder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.fallback_to_openai = fallback_to_openai
        
        self.enable_section_chunking = enable_section_chunking
        self.min_section_size = min_section_size
        
        # Initialize embedding model
        self.embed_model, self.embed_dim = self._initialize_embedding_model()
        
        # Initialize enhanced processing components
        self.metadata_extractor = iLandMetadataExtractor()
        
        if self.enable_section_chunking:
            self.section_parser = StandaloneLandDeedSectionParser(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_section_size=min_section_size
            )
        else:
            self.node_parser = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        
        # Initialize vector store
        self.vector_store = self._initialize_vector_store()
        
        # Create vector store index
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model
        )
        
        # Statistics tracking
        self.processing_stats = {
            "documents_processed": 0,
            "nodes_created": 0,
            "section_chunks": 0,
            "fallback_chunks": 0,
            "metadata_fields_extracted": 0,
            "embedding_provider": "bge" if BGE_AVAILABLE else "openai",
            "model_name": self.bge_model_key
        }
        
        logger.info(f"🤗 Initialized BGE PostgreSQL Embedding Generator")
        logger.info(f"   Model: {self.bge_model_key} ({self.embed_dim}d)")
        logger.info(f"   Provider: {self.processing_stats['embedding_provider']}")
        logger.info(f"   Section chunking: {self.enable_section_chunking}")

    def _initialize_embedding_model(self):
        """Initialize BGE embedding model with fallback to OpenAI."""
        
        # Try BGE first (preferred for local processing)
        if BGE_AVAILABLE:
            try:
                if self.bge_model_key not in self.BGE_MODELS:
                    logger.warning(f"Unknown BGE model {self.bge_model_key}, using bge-m3")
                    self.bge_model_key = "bge-m3"
                
                model_config = self.BGE_MODELS[self.bge_model_key]
                
                logger.info(f"🤗 Initializing BGE model: {model_config['model_name']}")
                logger.info(f"   Dimension: {model_config['dimension']}")
                logger.info(f"   Max length: {model_config['max_length']}")
                logger.info(f"   Cache folder: {self.cache_folder}")
                
                embed_model = HuggingFaceEmbedding(
                    model_name=model_config["model_name"],
                    cache_folder=self.cache_folder,
                    max_length=model_config["max_length"]
                )
                
                return embed_model, model_config["dimension"]
                
            except Exception as e:
                logger.error(f"Failed to initialize BGE model: {e}")
                if not self.fallback_to_openai:
                    raise
        
        # Fallback to OpenAI if BGE fails or not available
        if self.fallback_to_openai and OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI fallback enabled but OPENAI_API_KEY not found")
            
            logger.warning("🔑 Falling back to OpenAI embeddings")
            embed_model = OpenAIEmbedding(
                model="text-embedding-3-small",
                api_key=api_key
            )
            return embed_model, 1536
        
        raise RuntimeError("No embedding model available. Install BGE or enable OpenAI fallback.")

    def _initialize_vector_store(self) -> PGVectorStore:
        """Initialize the PostgreSQL vector store."""
        vector_store = PGVectorStore.from_params(
            database=self.dest_db_name,
            host=self.dest_db_host,
            password=self.dest_db_password,
            port=self.dest_db_port,
            user=self.dest_db_user,
            table_name=self.dest_table_name,
            embed_dim=self.embed_dim
        )
        return vector_store

    def connect_to_source_db(self) -> bool:
        """Establish connection to source PostgreSQL database."""
        try:
            logger.info(f"Connecting to source database {self.source_db_name} at {self.source_db_host}:{self.source_db_port}")
            self.source_conn = psycopg2.connect(
                dbname=self.source_db_name,
                user=self.source_db_user,
                password=self.source_db_password,
                host=self.source_db_host,
                port=self.source_db_port
            )
            logger.info(f"Successfully connected to source database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to source database: {e}")
            return False

    def close_source_db(self) -> None:
        """Close connection to source PostgreSQL database."""
        if self.source_conn:
            self.source_conn.close()
            self.source_conn = None
            logger.info("Source database connection closed")

    def fetch_documents_from_db(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch documents from the source database."""
        if not self.source_conn:
            if not self.connect_to_source_db():
                return []

        documents = []
        try:
            cursor = self.source_conn.cursor()
            
            # Build query with optional limit
            query = f"SELECT deed_id, md_string FROM {self.source_table_name}"
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            for row in rows:
                deed_id, md_string = row
                doc = {
                    "deed_id": deed_id,
                    "content": md_string
                }
                documents.append(doc)
            
            logger.info(f"Fetched {len(documents)} documents from database")
            cursor.close()
            return documents
            
        except Exception as e:
            logger.error(f"Error fetching documents from database: {e}")
            return []

    def process_documents(self, documents: List[Dict[str, Any]]) -> List[TextNode]:
        """Process documents using enhanced processing with BGE embeddings."""
        all_nodes = []
        
        for i, doc in enumerate(documents):
            logger.info(f"Processing document {i+1}/{len(documents)} with deed_id: {doc['deed_id']}")
            
            try:
                # Extract rich metadata from document content
                extracted_metadata = self.metadata_extractor.extract_from_content(doc['content'])
                
                # Add derived categories
                enhanced_metadata = self.metadata_extractor.derive_categories(extracted_metadata)
                
                # Add document tracking metadata
                enhanced_metadata.update({
                    "deed_id": doc['deed_id'],
                    "source": "postgres_db",
                    "processing_timestamp": datetime.now().isoformat(),
                    "embedding_provider": self.processing_stats['embedding_provider'],
                    "embedding_model": self.bge_model_key
                })
                
                # Update stats
                self.processing_stats["metadata_fields_extracted"] += len(enhanced_metadata)
                
                # Create Document object
                llama_doc = Document(
                    text=doc['content'],
                    metadata=enhanced_metadata
                )
                
                # Parse document into nodes using section-based chunking
                if self.enable_section_chunking:
                    try:
                        # Use section-aware parsing for Thai land deeds
                        chunks = self.section_parser.parse_land_deed_document(doc['content'])
                        
                        nodes = []
                        for j, chunk in enumerate(chunks):
                            # Create TextNode with rich metadata
                            node = TextNode(
                                text=chunk["text"],
                                metadata={
                                    **enhanced_metadata,
                                    "chunk_index": j,
                                    "chunk_type": chunk["section_type"],
                                    "section_title": chunk.get("section_title", ""),
                                    "processing_method": "section_chunking"
                                }
                            )
                            nodes.append(node)
                        
                        logger.info(f"   Created {len(nodes)} section-based chunks")
                        self.processing_stats["section_chunks"] += len(nodes)
                        
                    except Exception as e:
                        logger.warning(f"   Section parsing failed: {e}, using fallback")
                        # Fallback to sentence splitting
                        nodes = self.node_parser.get_nodes_from_documents([llama_doc])
                        
                        # Add metadata to fallback nodes
                        for j, node in enumerate(nodes):
                            node.metadata.update({
                                **enhanced_metadata,
                                "chunk_index": j,
                                "processing_method": "sentence_splitting"
                            })
                        
                        self.processing_stats["fallback_chunks"] += len(nodes)
                else:
                    # Use sentence splitting
                    nodes = self.node_parser.get_nodes_from_documents([llama_doc])
                    
                    # Add metadata to nodes
                    for j, node in enumerate(nodes):
                        node.metadata.update({
                            **enhanced_metadata,
                            "chunk_index": j,
                            "processing_method": "sentence_splitting"
                        })
                    
                    self.processing_stats["fallback_chunks"] += len(nodes)
                
                all_nodes.extend(nodes)
                self.processing_stats["nodes_created"] += len(nodes)
                
                logger.info(f"   ✅ Processed deed_id {doc['deed_id']}: {len(nodes)} nodes")
                
            except Exception as e:
                logger.error(f"   ❌ Error processing document {doc['deed_id']}: {e}")
                continue
        
        self.processing_stats["documents_processed"] = len(documents)
        logger.info(f"📊 Processing complete: {len(all_nodes)} total nodes from {len(documents)} documents")
        
        return all_nodes

    def insert_nodes_to_vector_store(self, nodes: List[TextNode]) -> int:
        """Insert processed nodes into the vector store."""
        try:
            logger.info(f"Inserting {len(nodes)} nodes into vector store...")
            
            # Add nodes to the vector store
            for node in nodes:
                self.index.insert_nodes([node])
            
            logger.info(f"✅ Successfully inserted {len(nodes)} nodes into vector store")
            return len(nodes)
            
        except Exception as e:
            logger.error(f"❌ Error inserting nodes into vector store: {e}")
            return 0

    def run_pipeline(self, limit: Optional[int] = None) -> int:
        """Run the complete BGE embedding pipeline."""
        start_time = time.time()
        logger.info("🚀 Starting BGE PostgreSQL Embedding Pipeline")
        logger.info(f"   Model: {self.bge_model_key}")
        logger.info(f"   Limit: {limit or 'all documents'}")
        
        try:
            # Fetch documents from database
            documents = self.fetch_documents_from_db(limit)
            if not documents:
                logger.error("No documents found in database")
                return 0
            
            # Process documents into nodes
            nodes = self.process_documents(documents)
            if not nodes:
                logger.error("No nodes created from documents")
                return 0
            
            # Insert nodes into vector store
            inserted_count = self.insert_nodes_to_vector_store(nodes)
            
            # Print final statistics
            duration = time.time() - start_time
            logger.info(f"\n✅ BGE PostgreSQL Pipeline Complete!")
            logger.info(f"   ⏱️ Duration: {duration:.2f} seconds")
            logger.info(f"   📄 Documents processed: {self.processing_stats['documents_processed']}")
            logger.info(f"   🔗 Nodes created: {self.processing_stats['nodes_created']}")
            logger.info(f"   📊 Section chunks: {self.processing_stats['section_chunks']}")
            logger.info(f"   📊 Fallback chunks: {self.processing_stats['fallback_chunks']}")
            logger.info(f"   💾 Nodes inserted: {inserted_count}")
            logger.info(f"   📈 Metadata fields: {self.processing_stats['metadata_fields_extracted']}")
            logger.info(f"   🤗 Model: {self.bge_model_key} ({self.embed_dim}d)")
            
            return inserted_count
            
        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}")
            return 0
        finally:
            self.close_source_db()


def main():
    """CLI entry point for BGE PostgreSQL embedding pipeline."""
    parser = argparse.ArgumentParser(
        description='BGE-based PostgreSQL Embedding Pipeline for iLand Thai Land Deeds',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all documents with default BGE multilingual model
  python postgres_embedding_bge.py
  
  # Process 100 documents with specific BGE model
  python postgres_embedding_bge.py --limit 100 --model bge-large-en-v1.5
  
  # Use custom cache folder
  python postgres_embedding_bge.py --cache-folder /path/to/cache

Available BGE models:
  - bge-small-en-v1.5: Fast, 384 dimensions
  - bge-base-en-v1.5: Balanced, 768 dimensions  
  - bge-large-en-v1.5: High quality, 1024 dimensions
  - bge-m3: Multilingual (Thai support), 1024 dimensions
        """
    )
    
    parser.add_argument(
        '--limit', 
        type=int, 
        help='Limit number of documents to process (default: all)'
    )
    parser.add_argument(
        '--model', 
        default='bge-m3',
        choices=['bge-small-en-v1.5', 'bge-base-en-v1.5', 'bge-large-en-v1.5', 'bge-m3'],
        help='BGE model to use (default: bge-m3 for Thai support)'
    )
    parser.add_argument(
        '--cache-folder',
        default='./cache/bge_models',
        help='Cache folder for BGE models (default: ./cache/bge_models)'
    )
    parser.add_argument(
        '--enable-openai-fallback',
        action='store_true',
        help='Enable OpenAI fallback if BGE fails'
    )
    
    args = parser.parse_args()
    
    try:
        generator = BGEPostgresEmbeddingGenerator(
            bge_model_key=args.model,
            cache_folder=args.cache_folder,
            fallback_to_openai=args.enable_openai_fallback
        )
        
        result = generator.run_pipeline(limit=args.limit)
        
        if result > 0:
            print(f"\n🎉 Success! Processed {result} embeddings using BGE model: {args.model}")
        else:
            print(f"\n❌ Pipeline failed or no embeddings created")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n👋 Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 