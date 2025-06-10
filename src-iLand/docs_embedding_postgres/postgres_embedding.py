"""
PostgreSQL Embedding Generator for iLand RAG Pipeline

This module loads documents from PostgreSQL database, generates embeddings
using OpenAI's embedding API, and stores the vectors in PostgreSQL for vector search.

It extends the iLand RAG pipeline by connecting the data processing module
with the vector search capabilities.
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
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PostgresEmbeddingGenerator:
    """Generates embeddings from PostgreSQL documents and stores them in a vector database."""
    
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
        
        # Embedding settings
        embed_model_name: str = os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        chunk_size: int = int(os.getenv("CHUNK_SIZE", "512")),
        chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50")),
        batch_size: int = int(os.getenv("API_BATCH_SIZE", "20"))
    ):
        """Initialize the PostgreSQL Embedding Generator with connection settings."""
        
        # Source database (markdown documents)
        self.source_db_name = source_db_name
        self.source_db_user = source_db_user
        self.source_db_password = source_db_password
        self.source_db_host = source_db_host
        self.source_db_port = source_db_port
        self.source_table_name = source_table_name
        self.source_conn = None
        
        # Destination database (vector embeddings)
        self.dest_db_name = dest_db_name
        self.dest_db_user = dest_db_user
        self.dest_db_password = dest_db_password
        self.dest_db_host = dest_db_host
        self.dest_db_port = dest_db_port
        self.dest_table_name = dest_table_name
        
        # Embedding configuration
        self.embed_model_name = embed_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        
        # Validate OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not found. Please set it in the .env file.")
        
        # Initialize the embedding model
        self.embed_model = OpenAIEmbedding(
            model=embed_model_name,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Get embedding dimensions based on model
        self.embed_dim = self._get_embed_dim(embed_model_name)
        
        # Initialize node parser for text chunking
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
    
    def _get_embed_dim(self, model_name: str) -> int:
        """Get embedding dimensions based on the model name."""
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return dimensions.get(model_name, 1536)
    
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
        """Process documents into TextNodes with embeddings."""
        all_nodes = []
        
        for i, doc in enumerate(documents):
            logger.info(f"Processing document {i+1}/{len(documents)} with deed_id: {doc['deed_id']}")
            
            # Create Document object
            llama_doc = Document(
                text=doc['content'],
                metadata={"deed_id": doc['deed_id']}
            )
            
            # Parse document into nodes
            nodes = self.node_parser.get_nodes_from_documents([llama_doc])
            
            # Add document metadata to each node
            for node in nodes:
                # Add deed_id to all nodes' metadata
                node.metadata["deed_id"] = doc['deed_id']
                
                # Add section info if available in the markdown content
                if "## " in node.text:
                    section = node.text.split("## ")[1].split("\n")[0].strip()
                    node.metadata["section"] = section
            
            all_nodes.extend(nodes)
            
            # Log progress for first few documents
            if i < 3:
                logger.info(f"  - Created {len(nodes)} chunks for deed_id: {doc['deed_id']}")
        
        logger.info(f"Created total of {len(all_nodes)} nodes from {len(documents)} documents")
        return all_nodes
    
    def insert_nodes_to_vector_store(self, nodes: List[TextNode]) -> int:
        """Insert nodes with embeddings into vector store."""
        try:
            start_time = time.time()
            
            # Process in batches to avoid overloading the API
            total_inserted = 0
            
            for i in range(0, len(nodes), self.batch_size):
                batch = nodes[i:i+self.batch_size]
                logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(nodes)-1)//self.batch_size + 1} ({len(batch)} nodes)")
                
                # Insert nodes with embeddings
                self.index.insert_nodes(batch)
                total_inserted += len(batch)
                
                # Small delay between batches
                if i + self.batch_size < len(nodes):
                    time.sleep(1)
            
            duration = time.time() - start_time
            logger.info(f"Successfully inserted {total_inserted} nodes with embeddings in {duration:.2f} seconds")
            return total_inserted
            
        except Exception as e:
            logger.error(f"Error inserting nodes to vector store: {e}")
            return 0
    
    def run_pipeline(self, limit: Optional[int] = None) -> int:
        """Run the full embedding pipeline."""
        try:
            start_time = time.time()
            logger.info("Starting embedding pipeline")
            
            # Step 1: Fetch documents
            documents = self.fetch_documents_from_db(limit=limit)
            if not documents:
                logger.error("No documents found in the database. Aborting.")
                return 0
            
            # Step 2: Process into nodes
            nodes = self.process_documents(documents)
            
            # Step 3: Insert to vector store
            inserted_count = self.insert_nodes_to_vector_store(nodes)
            
            # Log completion
            duration = time.time() - start_time
            logger.info(f"Pipeline completed in {duration:.2f} seconds")
            logger.info(f"Processed {len(documents)} documents into {len(nodes)} nodes")
            logger.info(f"Successfully inserted {inserted_count} nodes with embeddings")
            
            return inserted_count
            
        except Exception as e:
            logger.error(f"Error running pipeline: {e}")
            return 0
        finally:
            self.close_source_db()


def main():
    """Command-line interface for the embedding generator."""
    parser = argparse.ArgumentParser(description='Generate embeddings from PostgreSQL documents')
    
    # Add arguments
    parser.add_argument('--limit', type=int, default=None,
                        help='Maximum number of documents to process (default: all)')
    parser.add_argument('--chunk-size', type=int, default=512,
                        help='Chunk size for text splitting (default: 512)')
    parser.add_argument('--chunk-overlap', type=int, default=50,
                        help='Chunk overlap for text splitting (default: 50)')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='Batch size for API calls (default: 20)')
    parser.add_argument('--model', type=str, default="text-embedding-3-small",
                        help='OpenAI embedding model (default: text-embedding-3-small)')
    parser.add_argument('--source-host', type=str, default="10.4.102.11",
                        help='Source database host (default: 10.4.102.11)')
    parser.add_argument('--dest-host', type=str, default="10.4.102.11",
                        help='Destination database host (default: 10.4.102.11)')
    parser.add_argument('--dest-table', type=str, default="iland_embeddings",
                        help='Destination table name (default: iland_embeddings)')
    
    args = parser.parse_args()
    
    logger.info("====== iLand PostgreSQL Embedding Generator ======")
    logger.info(f"Model: {args.model}")
    logger.info(f"Chunk size: {args.chunk_size}, Overlap: {args.chunk_overlap}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Document limit: {args.limit or 'All'}")
    logger.info(f"Source DB: {args.source_host}")
    logger.info(f"Destination table: {args.dest_table}")
    logger.info("=" * 50)
    
    # Create and run the generator
    generator = PostgresEmbeddingGenerator(
        source_db_host=args.source_host,
        dest_db_host=args.dest_host,
        dest_table_name=args.dest_table,
        embed_model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size
    )
    
    inserted_count = generator.run_pipeline(limit=args.limit)
    
    if inserted_count > 0:
        logger.info(f"Success! {inserted_count} nodes with embeddings inserted into PostgreSQL vector store")
        return 0
    else:
        logger.error("Pipeline failed or no documents were processed")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 