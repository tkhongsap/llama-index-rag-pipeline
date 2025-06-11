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
        
        # Always create node_parser for fallback
        self.node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize multiple vector stores for different data types
        self.vector_stores = self._initialize_vector_stores()
        
        # Create vector store indices for each type
        self.indices = self._initialize_indices()
        
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

    def _initialize_vector_stores(self) -> Dict[str, PGVectorStore]:
        """Initialize multiple PostgreSQL vector stores for different data types."""
        vector_stores = {}
        
        # Define table names for different embedding types
        table_configs = {
            "chunks": f"{self.dest_table_name.replace('_embeddings', '')}_chunks",
            "summaries": f"{self.dest_table_name.replace('_embeddings', '')}_summaries", 
            "indexnodes": f"{self.dest_table_name.replace('_embeddings', '')}_indexnodes",
            "combined": f"{self.dest_table_name.replace('_embeddings', '')}_combined"
        }
        
        # Create pgVector store for each table
        for store_type, table_name in table_configs.items():
            try:
                vector_store = PGVectorStore.from_params(
                    database=self.dest_db_name,
                    host=self.dest_db_host,
                    password=self.dest_db_password,
                    port=self.dest_db_port,
                    user=self.dest_db_user,
                    table_name=table_name,
                    embed_dim=self.embed_dim
                )
                vector_stores[store_type] = vector_store
                logger.info(f"✅ Initialized pgVector store for {store_type}: {table_name}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize {store_type} vector store: {e}")
        
        return vector_stores
    
    def _initialize_indices(self) -> Dict[str, VectorStoreIndex]:
        """Initialize vector store indices for each data type."""
        indices = {}
        
        for store_type, vector_store in self.vector_stores.items():
            try:
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    embed_model=self.embed_model
                )
                indices[store_type] = index
                logger.info(f"✅ Initialized index for {store_type}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize {store_type} index: {e}")
        
        return indices

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

    def process_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, List[TextNode]]:
        """Process documents using enhanced processing with BGE embeddings and create multiple node types."""
        all_node_types = {
            "chunks": [],
            "summaries": [],
            "indexnodes": [],
            "combined": []
        }
        
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
                
                # 1. Create chunk nodes using section-based chunking
                chunk_nodes = self._create_chunk_nodes(doc, enhanced_metadata)
                all_node_types["chunks"].extend(chunk_nodes)
                
                # 2. Create summary node
                summary_node = self._create_summary_node(doc, enhanced_metadata)
                all_node_types["summaries"].append(summary_node)
                
                # 3. Create index node 
                index_node = self._create_index_node(doc, enhanced_metadata, len(chunk_nodes))
                all_node_types["indexnodes"].append(index_node)
                
                # 4. Create combined nodes (all chunks + summary + index)
                combined_nodes = chunk_nodes + [summary_node, index_node]
                all_node_types["combined"].extend(combined_nodes)
                
                self.processing_stats["nodes_created"] += len(chunk_nodes)
                
                logger.info(f"   ✅ Processed deed_id {doc['deed_id']}: {len(chunk_nodes)} chunks, 1 summary, 1 index")
                
            except Exception as e:
                logger.error(f"   ❌ Error processing document {doc['deed_id']}: {e}")
                continue
        
        self.processing_stats["documents_processed"] = len(documents)
        
        # Log statistics
        for node_type, nodes in all_node_types.items():
            logger.info(f"📊 {node_type}: {len(nodes)} nodes")
        
        return all_node_types
    
    def _create_chunk_nodes(self, doc: Dict[str, Any], metadata: Dict[str, Any]) -> List[TextNode]:
        """Create chunk nodes using section-based chunking."""
        if self.enable_section_chunking:
            try:
                # Use section-aware parsing for Thai land deeds
                nodes = self.section_parser.parse_document_to_sections(
                    document_text=doc['content'],
                    metadata=metadata
                )
                
                # Add chunk-specific metadata
                for node in nodes:
                    node.metadata.update({
                        "node_type": "chunk",
                        "table_type": "chunks"
                    })
                
                logger.info(f"   Created {len(nodes)} section-based chunks")
                self.processing_stats["section_chunks"] += len(nodes)
                return nodes
                
            except Exception as e:
                logger.warning(f"   Section parsing failed: {e}, using fallback")
                
        # Fallback to sentence splitting
        llama_doc = Document(text=doc['content'], metadata=metadata)
        nodes = self.node_parser.get_nodes_from_documents([llama_doc])
        
        # Add metadata to fallback nodes
        for j, node in enumerate(nodes):
            node.metadata.update({
                **metadata,
                "chunk_index": j,
                "processing_method": "sentence_splitting",
                "node_type": "chunk",
                "table_type": "chunks"
            })
        
        self.processing_stats["fallback_chunks"] += len(nodes)
        return nodes
    
    def _create_summary_node(self, doc: Dict[str, Any], metadata: Dict[str, Any]) -> TextNode:
        """Create summary node for the document."""
        # Create simple summary from first 500 chars + key metadata
        content = doc['content']
        summary_text = content[:500] + "..." if len(content) > 500 else content
        
        summary_metadata = {
            **metadata,
            "node_type": "summary",
            "table_type": "summaries",
            "summary_length": len(summary_text),
            "original_length": len(content)
        }
        
        summary_node = TextNode(
            text=f"# Document Summary - {metadata.get('deed_id', 'Unknown')}\n\n{summary_text}",
            metadata=summary_metadata
        )
        
        return summary_node
    
    def _create_index_node(self, doc: Dict[str, Any], metadata: Dict[str, Any], chunk_count: int) -> TextNode:
        """Create index node for the document."""
        # Create index with key information
        key_info = []
        if metadata.get('province'):
            key_info.append(f"Province: {metadata['province']}")
        if metadata.get('district'):
            key_info.append(f"District: {metadata['district']}")
        if metadata.get('deed_type'):
            key_info.append(f"Deed Type: {metadata['deed_type']}")
        if metadata.get('area_rai'):
            key_info.append(f"Area: {metadata['area_rai']} rai")
        
        index_text = f"# Document Index - {metadata.get('deed_id', 'Unknown')}\n\n"
        index_text += f"Document contains {chunk_count} chunks\n"
        index_text += "Key Information:\n" + "\n".join(f"- {info}" for info in key_info)
        
        index_metadata = {
            **metadata,
            "node_type": "index",
            "table_type": "indexnodes",
            "chunk_count": chunk_count,
            "index_info_count": len(key_info)
        }
        
        index_node = TextNode(
            text=index_text,
            metadata=index_metadata
        )
        
        return index_node

    def insert_nodes_to_vector_stores(self, all_node_types: Dict[str, List[TextNode]]) -> Dict[str, int]:
        """Insert processed nodes into multiple vector stores."""
        insertion_counts = {}
        
        for node_type, nodes in all_node_types.items():
            if not nodes:
                insertion_counts[node_type] = 0
                continue
                
            try:
                logger.info(f"Inserting {len(nodes)} {node_type} nodes...")
                
                # Get the appropriate index for this node type
                if node_type in self.indices:
                    index = self.indices[node_type]
                    
                    # Insert nodes in batches
                    batch_size = min(10, len(nodes))
                    inserted_count = 0
                    
                    for i in range(0, len(nodes), batch_size):
                        batch = nodes[i:i + batch_size]
                        try:
                            index.insert_nodes(batch)
                            inserted_count += len(batch)
                        except Exception as e:
                            logger.error(f"❌ Error inserting batch for {node_type}: {e}")
                    
                    insertion_counts[node_type] = inserted_count
                    logger.info(f"✅ Successfully inserted {inserted_count}/{len(nodes)} {node_type} nodes")
                else:
                    logger.warning(f"⚠️ No index found for {node_type}")
                    insertion_counts[node_type] = 0
                    
            except Exception as e:
                logger.error(f"❌ Error inserting {node_type} nodes: {e}")
                insertion_counts[node_type] = 0
        
        return insertion_counts

    def run_pipeline(self, limit: Optional[int] = None) -> int:
        """Run the complete BGE embedding pipeline with multi-table structure."""
        start_time = time.time()
        logger.info("🚀 Starting BGE PostgreSQL Multi-Table Embedding Pipeline")
        logger.info(f"   Model: {self.bge_model_key}")
        logger.info(f"   Limit: {limit or 'all documents'}")
        logger.info(f"   Tables: chunks, summaries, indexnodes, combined")
        
        try:
            # Fetch documents from database
            documents = self.fetch_documents_from_db(limit)
            if not documents:
                logger.error("No documents found in database")
                return 0
            
            # Process documents into multiple node types
            all_node_types = self.process_documents(documents)
            if not any(all_node_types.values()):
                logger.error("No nodes created from documents")
                return 0
            
            # Insert nodes into multiple vector stores
            insertion_counts = self.insert_nodes_to_vector_stores(all_node_types)
            
            # Calculate total insertions
            total_inserted = sum(insertion_counts.values())
            
            # Print final statistics
            duration = time.time() - start_time
            logger.info(f"\n✅ BGE PostgreSQL Multi-Table Pipeline Complete!")
            logger.info(f"   ⏱️ Duration: {duration:.2f} seconds")
            logger.info(f"   📄 Documents processed: {self.processing_stats['documents_processed']}")
            logger.info(f"   🔗 Chunk nodes created: {self.processing_stats['nodes_created']}")
            logger.info(f"   📊 Section chunks: {self.processing_stats['section_chunks']}")
            logger.info(f"   📊 Fallback chunks: {self.processing_stats['fallback_chunks']}")
            logger.info(f"   📈 Metadata fields: {self.processing_stats['metadata_fields_extracted']}")
            logger.info(f"   🤗 Model: {self.bge_model_key} ({self.embed_dim}d)")
            logger.info(f"\n🗄️ DATABASE INSERTIONS:")
            for table_type, count in insertion_counts.items():
                logger.info(f"   • {table_type}: {count} nodes")
            logger.info(f"   • Total: {total_inserted} nodes across all tables")
            
            return total_inserted
            
        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}")
            return 0
        finally:
            self.close_source_db()


def main():
    """CLI entry point for BGE PostgreSQL Multi-Table embedding pipeline."""
    parser = argparse.ArgumentParser(
        description='BGE-based PostgreSQL Multi-Table Embedding Pipeline for iLand Thai Land Deeds',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all documents with multi-table structure (default)
  python postgres_embedding_bge.py
  
  # Process 100 documents with specific BGE model
  python postgres_embedding_bge.py --limit 100 --model bge-large-en-v1.5
  
  # Use custom cache folder
  python postgres_embedding_bge.py --cache-folder /path/to/cache

Multi-Table Structure (pgVector):
  - chunks: Section-based document chunks (~6-10 per document)
  - summaries: Document summaries
  - indexnodes: Document index metadata
  - combined: All node types combined

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
    parser.add_argument(
        '--disable-section-chunking',
        action='store_true',
        help='Disable section-based chunking (use sentence splitting)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BGE POSTGRESQL MULTI-TABLE EMBEDDING PIPELINE - PRD v2.0")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Section chunking: {not args.disable_section_chunking}")
    print(f"Limit: {args.limit or 'all documents'}")
    print(f"Tables: chunks, summaries, indexnodes, combined (pgVector)")
    print("=" * 60)
    
    try:
        generator = BGEPostgresEmbeddingGenerator(
            bge_model_key=args.model,
            cache_folder=args.cache_folder,
            fallback_to_openai=args.enable_openai_fallback,
            enable_section_chunking=not args.disable_section_chunking
        )
        
        result = generator.run_pipeline(limit=args.limit)
        
        if result > 0:
            print(f"\n🎉 SUCCESS! Processed {result} embeddings across multi-table structure")
            print("🗄️ Created 4 pgVector tables: chunks, summaries, indexnodes, combined")  
            print(f"📊 Expected: ~6-10 chunks per document (vs ~169 traditional chunks)")
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