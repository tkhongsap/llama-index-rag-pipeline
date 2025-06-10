"""
Database utilities for iLand BGE Embedding with PostgreSQL

This module provides database utilities for handling PostgreSQL operations
related to embedding storage and retrieval.
"""

import os
import logging
import json
import psycopg2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class PostgresManager:
    """Handles PostgreSQL database operations for BGE embeddings."""
    
    def __init__(
        self,
        # Database configuration
        db_name: str = os.getenv("DB_NAME", "iland-vector-dev"),
        db_user: str = os.getenv("DB_USER", "vector_user_dev"),
        db_password: str = os.getenv("DB_PASSWORD", "akqVvIJvVqe7Jr1"),
        db_host: str = os.getenv("DB_HOST", "10.4.102.11"),
        db_port: int = int(os.getenv("DB_PORT", "5432")),
        
        # Table names
        source_table: str = os.getenv("SOURCE_TABLE", "iland_md_data"),
        chunks_table: str = os.getenv("CHUNKS_TABLE", "iland_chunks"),
        summaries_table: str = os.getenv("SUMMARIES_TABLE", "iland_summaries"),
        indexnodes_table: str = os.getenv("INDEXNODES_TABLE", "iland_indexnodes"),
        combined_table: str = os.getenv("COMBINED_TABLE", "iland_combined")
    ):
        """Initialize the PostgreSQL Manager."""
        # Database configuration
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        self.connection = None
        
        # Table names
        self.source_table = source_table
        self.chunks_table = chunks_table
        self.summaries_table = summaries_table
        self.indexnodes_table = indexnodes_table
        self.combined_table = combined_table
    
    def connect(self) -> bool:
        """Establish connection to PostgreSQL database."""
        try:
            logger.info(f"Connecting to database {self.db_name} at {self.db_host}:{self.db_port}")
            self.connection = psycopg2.connect(
                dbname=self.db_name,
                user=self.db_user,
                password=self.db_password,
                host=self.db_host,
                port=self.db_port
            )
            logger.info(f"Successfully connected to database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def close(self) -> None:
        """Close connection to PostgreSQL database."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Database connection closed")
    
    def setup_tables(self, vector_dim: int) -> bool:
        """Create the required database tables if they don't exist."""
        if not self.connection and not self.connect():
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # Create extension if not exists (only needs to be done once per database)
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create chunks table
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.chunks_table} (
                id SERIAL PRIMARY KEY,
                deed_id TEXT NOT NULL,
                chunk_index INTEGER,
                text TEXT NOT NULL,
                metadata JSONB,
                embedding vector({vector_dim}),
                embedding_model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Create summaries table
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.summaries_table} (
                id SERIAL PRIMARY KEY,
                deed_id TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                metadata JSONB,
                embedding vector({vector_dim}),
                embedding_model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Create index nodes table
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.indexnodes_table} (
                id SERIAL PRIMARY KEY,
                deed_id TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata JSONB,
                embedding vector({vector_dim}),
                embedding_model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Create combined table for unified search
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.combined_table} (
                id SERIAL PRIMARY KEY,
                deed_id TEXT NOT NULL,
                type TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata JSONB,
                embedding vector({vector_dim}),
                embedding_model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Create indexes for faster queries
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.chunks_table}_deed_id ON {self.chunks_table} (deed_id)")
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.summaries_table}_deed_id ON {self.summaries_table} (deed_id)")
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.indexnodes_table}_deed_id ON {self.indexnodes_table} (deed_id)")
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.combined_table}_deed_id ON {self.combined_table} (deed_id)")
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.combined_table}_type ON {self.combined_table} (type)")
            
            # Create vector indexes for similarity search
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.chunks_table}_embedding ON {self.chunks_table} USING ivfflat (embedding vector_cosine_ops)")
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.summaries_table}_embedding ON {self.summaries_table} USING ivfflat (embedding vector_cosine_ops)")
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.indexnodes_table}_embedding ON {self.indexnodes_table} USING ivfflat (embedding vector_cosine_ops)")
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.combined_table}_embedding ON {self.combined_table} USING ivfflat (embedding vector_cosine_ops)")
            
            self.connection.commit()
            cursor.close()
            
            logger.info(f"Created tables: {self.chunks_table}, {self.summaries_table}, {self.indexnodes_table}, {self.combined_table}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up database tables: {e}")
            self.connection.rollback()
            return False
    
    def fetch_documents(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch documents from the source database table."""
        if not self.connection and not self.connect():
            return []
        
        documents = []
        try:
            cursor = self.connection.cursor()
            
            # Build query with optional limit
            query = f"SELECT deed_id, md_string FROM {self.source_table}"
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            for row in rows:
                deed_id, md_string = row
                doc = {
                    "deed_id": deed_id,
                    "content": md_string,
                    "metadata": {"deed_id": deed_id}
                }
                documents.append(doc)
            
            logger.info(f"Fetched {len(documents)} documents from database")
            cursor.close()
            return documents
            
        except Exception as e:
            logger.error(f"Error fetching documents from database: {e}")
            return []
    
    def save_chunk_embeddings(self, chunk_embeddings: List[Dict]) -> int:
        """Save chunk embeddings to the chunks table."""
        if not chunk_embeddings:
            return 0
        
        if not self.connection and not self.connect():
            return 0
            
        inserted_count = 0
        cursor = self.connection.cursor()
        
        try:
            for emb in chunk_embeddings:
                # Convert embedding vector to PostgreSQL vector format
                embedding_vector = np.array(emb['embedding_vector']).tolist()
                
                # Convert metadata to JSON
                metadata_json = json.dumps(emb['metadata'])
                
                # Insert into chunks table
                cursor.execute(
                    f"""
                    INSERT INTO {self.chunks_table} 
                    (deed_id, chunk_index, text, metadata, embedding, embedding_model)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        emb['metadata'].get('deed_id', 'unknown'),
                        emb.get('chunk_index', 0),
                        emb['text'],
                        metadata_json,
                        embedding_vector,
                        emb['embedding_model']
                    )
                )
                inserted_count += 1
                
                # Commit every 50 rows to avoid large transactions
                if inserted_count % 50 == 0:
                    self.connection.commit()
            
            # Final commit
            self.connection.commit()
            logger.info(f"Inserted {inserted_count} rows into {self.chunks_table}")
            return inserted_count
            
        except Exception as e:
            logger.error(f"Error inserting chunk embeddings: {e}")
            self.connection.rollback()
            return inserted_count
        finally:
            cursor.close()
    
    def save_summary_embeddings(self, summary_embeddings: List[Dict]) -> int:
        """Save summary embeddings to the summaries table."""
        if not summary_embeddings:
            return 0
        
        if not self.connection and not self.connect():
            return 0
            
        inserted_count = 0
        cursor = self.connection.cursor()
        
        try:
            for emb in summary_embeddings:
                # Convert embedding vector to PostgreSQL vector format
                embedding_vector = np.array(emb['embedding_vector']).tolist()
                
                # Convert metadata to JSON
                metadata_json = json.dumps(emb['metadata'])
                
                # Insert into summaries table
                cursor.execute(
                    f"""
                    INSERT INTO {self.summaries_table} 
                    (deed_id, summary_text, metadata, embedding, embedding_model)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        emb['metadata'].get('deed_id', 'unknown'),
                        emb['summary_text'],
                        metadata_json,
                        embedding_vector,
                        emb['embedding_model']
                    )
                )
                inserted_count += 1
                
                # Commit every 50 rows to avoid large transactions
                if inserted_count % 50 == 0:
                    self.connection.commit()
            
            # Final commit
            self.connection.commit()
            logger.info(f"Inserted {inserted_count} rows into {self.summaries_table}")
            return inserted_count
            
        except Exception as e:
            logger.error(f"Error inserting summary embeddings: {e}")
            self.connection.rollback()
            return inserted_count
        finally:
            cursor.close()
    
    def save_indexnode_embeddings(self, indexnode_embeddings: List[Dict]) -> int:
        """Save indexnode embeddings to the indexnodes table."""
        if not indexnode_embeddings:
            return 0
        
        if not self.connection and not self.connect():
            return 0
            
        inserted_count = 0
        cursor = self.connection.cursor()
        
        try:
            for emb in indexnode_embeddings:
                # Convert embedding vector to PostgreSQL vector format
                embedding_vector = np.array(emb['embedding_vector']).tolist()
                
                # Convert metadata to JSON
                metadata_json = json.dumps(emb['metadata'])
                
                # Insert into indexnodes table
                cursor.execute(
                    f"""
                    INSERT INTO {self.indexnodes_table} 
                    (deed_id, text, metadata, embedding, embedding_model)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        emb['metadata'].get('deed_id', 'unknown'),
                        emb['text'],
                        metadata_json,
                        embedding_vector,
                        emb['embedding_model']
                    )
                )
                inserted_count += 1
                
                # Commit every 50 rows to avoid large transactions
                if inserted_count % 50 == 0:
                    self.connection.commit()
            
            # Final commit
            self.connection.commit()
            logger.info(f"Inserted {inserted_count} rows into {self.indexnodes_table}")
            return inserted_count
            
        except Exception as e:
            logger.error(f"Error inserting indexnode embeddings: {e}")
            self.connection.rollback()
            return inserted_count
        finally:
            cursor.close()
    
    def save_combined_embeddings(
        self, 
        chunk_embeddings: List[Dict], 
        summary_embeddings: List[Dict], 
        indexnode_embeddings: List[Dict]
    ) -> int:
        """Save all embeddings to the combined table."""
        if not self.connection and not self.connect():
            return 0
            
        inserted_count = 0
        cursor = self.connection.cursor()
        
        try:
            # Process chunk embeddings
            for emb in chunk_embeddings:
                # Convert embedding vector to PostgreSQL vector format
                embedding_vector = np.array(emb['embedding_vector']).tolist()
                
                # Convert metadata to JSON
                metadata_json = json.dumps(emb['metadata'])
                
                # Insert into combined table
                cursor.execute(
                    f"""
                    INSERT INTO {self.combined_table} 
                    (deed_id, type, text, metadata, embedding, embedding_model)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        emb['metadata'].get('deed_id', 'unknown'),
                        'chunk',
                        emb['text'],
                        metadata_json,
                        embedding_vector,
                        emb['embedding_model']
                    )
                )
                inserted_count += 1
                
                # Commit every 50 rows to avoid large transactions
                if inserted_count % 50 == 0:
                    self.connection.commit()
            
            # Process summary embeddings
            for emb in summary_embeddings:
                # Convert embedding vector to PostgreSQL vector format
                embedding_vector = np.array(emb['embedding_vector']).tolist()
                
                # Convert metadata to JSON
                metadata_json = json.dumps(emb['metadata'])
                
                # Insert into combined table
                cursor.execute(
                    f"""
                    INSERT INTO {self.combined_table} 
                    (deed_id, type, text, metadata, embedding, embedding_model)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        emb['metadata'].get('deed_id', 'unknown'),
                        'summary',
                        emb['summary_text'],
                        metadata_json,
                        embedding_vector,
                        emb['embedding_model']
                    )
                )
                inserted_count += 1
                
                # Commit every 50 rows to avoid large transactions
                if inserted_count % 50 == 0:
                    self.connection.commit()
            
            # Process indexnode embeddings
            for emb in indexnode_embeddings:
                # Convert embedding vector to PostgreSQL vector format
                embedding_vector = np.array(emb['embedding_vector']).tolist()
                
                # Convert metadata to JSON
                metadata_json = json.dumps(emb['metadata'])
                
                # Insert into combined table
                cursor.execute(
                    f"""
                    INSERT INTO {self.combined_table} 
                    (deed_id, type, text, metadata, embedding, embedding_model)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        emb['metadata'].get('deed_id', 'unknown'),
                        'indexnode',
                        emb['text'],
                        metadata_json,
                        embedding_vector,
                        emb['embedding_model']
                    )
                )
                inserted_count += 1
                
                # Commit every 50 rows to avoid large transactions
                if inserted_count % 50 == 0:
                    self.connection.commit()
            
            # Final commit
            self.connection.commit()
            logger.info(f"Inserted {inserted_count} rows into {self.combined_table}")
            return inserted_count
            
        except Exception as e:
            logger.error(f"Error inserting combined embeddings: {e}")
            self.connection.rollback()
            return inserted_count
        finally:
            cursor.close()
    
    def save_all_embeddings(
        self, 
        chunk_embeddings: List[Dict], 
        summary_embeddings: List[Dict], 
        indexnode_embeddings: List[Dict]
    ) -> Tuple[int, int, int, int]:
        """Save embeddings to all tables and return counts."""
        # Track inserted counts
        chunks_inserted = self.save_chunk_embeddings(chunk_embeddings)
        summaries_inserted = self.save_summary_embeddings(summary_embeddings)
        indexnodes_inserted = self.save_indexnode_embeddings(indexnode_embeddings)
        
        # Then save to combined table
        combined_inserted = self.save_combined_embeddings(
            chunk_embeddings, summary_embeddings, indexnode_embeddings
        )
        
        logger.info(f"Saved embeddings to database:")
        logger.info(f"  - Chunks: {chunks_inserted}")
        logger.info(f"  - Summaries: {summaries_inserted}")
        logger.info(f"  - IndexNodes: {indexnodes_inserted}")
        logger.info(f"  - Combined: {combined_inserted}")
        
        return chunks_inserted, summaries_inserted, indexnodes_inserted, combined_inserted 