import os
import logging
import json
import psycopg2
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Handle both relative and absolute imports
try:
    from .models import SimpleDocument, DatasetConfig
    from .file_output import FileOutputManager
except ImportError:
    from models import SimpleDocument, DatasetConfig
    from file_output import FileOutputManager

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Handles database operations for storing documents in PostgreSQL"""
    
    def __init__(self, 
                 db_name: str = os.getenv("DB_NAME", "iland-vector-dev"), 
                 db_user: str = os.getenv("DB_USER", "vector_user_dev"), 
                 db_password: str = os.getenv("DB_PASSWORD", "akqVvIJvVqe7Jr1"),
                 db_host: str = os.getenv("DB_HOST", "10.4.102.11"),
                 db_port: int = int(os.getenv("DB_PORT", "5432"))):
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        self.connection = None
        
    def connect(self):
        """Establish connection to PostgreSQL database"""
        try:
            logger.info(f"Connecting to database {self.db_name} at {self.db_host}:{self.db_port} with user {self.db_user}")
            self.connection = psycopg2.connect(
                dbname=self.db_name,
                user=self.db_user,
                password=self.db_password,
                host=self.db_host,
                port=self.db_port
            )
            logger.info(f"Successfully connected to database: {self.db_name} at {self.db_host}:{self.db_port}")
            
            # Setup source table if it doesn't exist
            self.setup_source_table()
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def setup_source_table(self):
        """Create the source table (iland_md_data) if it doesn't exist"""
        if not self.connection:
            logger.error("No database connection available")
            return False
        
        try:
            cursor = self.connection.cursor()
            table_name = os.getenv("SOURCE_TABLE", "iland_md_data")
            
            # Create vector extension if not exists
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                logger.info("Vector extension enabled or already exists")
            except Exception as e:
                logger.warning(f"Could not create vector extension: {e}")
                logger.warning("Vector search functionality may be limited")
            
            # Check if table exists
            cursor.execute(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    AND table_name = '{table_name}'
                );
            """)
            
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                logger.info(f"Source table '{table_name}' does not exist. Creating...")
                
                # Create the table
                cursor.execute(f"""
                    CREATE TABLE {table_name} (
                        id SERIAL PRIMARY KEY,
                        deed_id TEXT NOT NULL,
                        md_string TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create index on deed_id for faster lookups
                cursor.execute(f"""
                    CREATE INDEX idx_{table_name}_deed_id ON {table_name} (deed_id);
                """)
                
                self.connection.commit()
                logger.info(f"Successfully created source table '{table_name}'")
                return True
            else:
                logger.info(f"Source table '{table_name}' already exists")
                return True
                
        except Exception as e:
            logger.error(f"Error setting up source table: {e}")
            self.connection.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Database connection closed")
    
    def insert_documents(self, documents: List[SimpleDocument], batch_size: int = 100, output_manager: Optional[FileOutputManager] = None) -> int:
        """
        Insert documents into the iland_md_data table
        Returns the number of successfully inserted documents
        """
        if not documents:
            logger.warning("No documents to insert into database")
            return 0
        
        if not self.connection:
            if not self.connect():
                logger.error("Failed to establish database connection")
                return 0
        
        successful_inserts = 0
        
        try:
            cursor = self.connection.cursor()
            
            # Log available metadata keys from the first document for debugging
            if documents:
                logger.info(f"Available metadata keys: {list(documents[0].metadata.keys())}")
            
            # Process documents in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                
                for doc in batch:
                    # Try multiple potential ID fields
                    deed_id = (
                        doc.metadata.get('deed_id') or 
                        doc.metadata.get('doc_id') or 
                        doc.metadata.get('id') or 
                        doc.metadata.get('row_index') or 
                        str(doc.id)  # Fall back to the document's id
                    )
                    
                    # Make sure we have a string
                    deed_id = str(deed_id)
                    
                    # Get enhanced markdown content with all metadata
                    if output_manager:
                        md_string = output_manager.generate_enhanced_markdown_content(doc)
                    else:
                        # Fallback if no output_manager is provided
                        md_string = doc.text
                    
                    try:
                        # Insert into the iland_md_data table
                        table_name = os.getenv("SOURCE_TABLE", "iland_md_data")
                        cursor.execute(
                            f"INSERT INTO {table_name} (deed_id, md_string) VALUES (%s, %s)",
                            (deed_id, md_string)
                        )
                        successful_inserts += 1
                        
                        if successful_inserts <= 3:  # Log details for first few documents
                            logger.info(f"Inserted document with deed_id: {deed_id}, text length: {len(md_string)} chars")
                            
                    except Exception as e:
                        logger.error(f"Error inserting document with deed_id {deed_id}: {e}")
                
                # Commit batch
                self.connection.commit()
                logger.info(f"Inserted batch of {len(batch)} documents - Total: {successful_inserts}/{len(documents)}")
            
            logger.info(f"Database insertion completed: {successful_inserts} documents successfully inserted")
            return successful_inserts
            
        except Exception as e:
            logger.error(f"Error during database insertion: {e}")
            self.connection.rollback()
            return successful_inserts
        finally:
            if cursor:
                cursor.close() 