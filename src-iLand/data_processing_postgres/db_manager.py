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
        """Create the source table (iland_md_data) if it doesn't exist with enhanced schema"""
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
                logger.info(f"Source table '{table_name}' does not exist. Creating enhanced schema...")
                
                # Create the enhanced table with metadata and processing status
                cursor.execute(f"""
                    CREATE TABLE {table_name} (
                        id SERIAL PRIMARY KEY,
                        deed_id TEXT NOT NULL UNIQUE,
                        md_string TEXT NOT NULL,
                        raw_metadata JSONB,
                        extracted_metadata JSONB,
                        province TEXT,
                        district TEXT,
                        land_use_category TEXT,
                        deed_type_category TEXT,
                        area_category TEXT,
                        processing_status TEXT DEFAULT 'pending',
                        processing_timestamp TIMESTAMP,
                        embedding_status TEXT DEFAULT 'pending',
                        embedding_timestamp TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create indexes for faster lookups and filtering
                cursor.execute(f"CREATE INDEX idx_{table_name}_deed_id ON {table_name} (deed_id)")
                cursor.execute(f"CREATE INDEX idx_{table_name}_province ON {table_name} (province)")
                cursor.execute(f"CREATE INDEX idx_{table_name}_district ON {table_name} (district)")
                cursor.execute(f"CREATE INDEX idx_{table_name}_land_use ON {table_name} (land_use_category)")
                cursor.execute(f"CREATE INDEX idx_{table_name}_deed_type ON {table_name} (deed_type_category)")
                cursor.execute(f"CREATE INDEX idx_{table_name}_area ON {table_name} (area_category)")
                cursor.execute(f"CREATE INDEX idx_{table_name}_processing_status ON {table_name} (processing_status)")
                cursor.execute(f"CREATE INDEX idx_{table_name}_embedding_status ON {table_name} (embedding_status)")
                
                # Create GIN index for JSONB metadata searching
                cursor.execute(f"CREATE INDEX idx_{table_name}_raw_metadata ON {table_name} USING GIN (raw_metadata)")
                cursor.execute(f"CREATE INDEX idx_{table_name}_extracted_metadata ON {table_name} USING GIN (extracted_metadata)")
                
                # Create trigger for updating timestamp
                cursor.execute(f"""
                    CREATE OR REPLACE FUNCTION update_timestamp()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        NEW.updated_at = CURRENT_TIMESTAMP;
                        RETURN NEW;
                    END;
                    $$ LANGUAGE plpgsql;
                """)
                
                cursor.execute(f"""
                    CREATE TRIGGER trigger_update_timestamp
                        BEFORE UPDATE ON {table_name}
                        FOR EACH ROW
                        EXECUTE FUNCTION update_timestamp();
                """)
                
                self.connection.commit()
                logger.info(f"Successfully created enhanced source table '{table_name}' with metadata indexing")
                return True
            else:
                # Check if the table needs to be updated with new columns
                self._update_table_schema(cursor, table_name)
                logger.info(f"Source table '{table_name}' exists and schema updated if needed")
                return True
                
        except Exception as e:
            logger.error(f"Error setting up source table: {e}")
            self.connection.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
    
    def _update_table_schema(self, cursor, table_name: str):
        """Update existing table schema with new columns if they don't exist"""
        # First, update table columns
        try:
            # Get existing columns
            cursor.execute(f"""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = '{table_name}' AND table_schema = 'public'
            """)
            existing_columns = {row[0] for row in cursor.fetchall()}
            
            # Define required columns
            required_columns = {
                'raw_metadata': 'JSONB',
                'extracted_metadata': 'JSONB', 
                'province': 'TEXT',
                'district': 'TEXT',
                'land_use_category': 'TEXT',
                'deed_type_category': 'TEXT',
                'area_category': 'TEXT',
                'processing_status': 'TEXT DEFAULT \'pending\'',
                'processing_timestamp': 'TIMESTAMP',
                'embedding_status': 'TEXT DEFAULT \'pending\'',
                'embedding_timestamp': 'TIMESTAMP',
                'updated_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
            }
            
            # Add missing columns
            columns_added = False
            for column_name, column_type in required_columns.items():
                if column_name not in existing_columns:
                    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
                    logger.info(f"Added column {column_name} to {table_name}")
                    columns_added = True
            
            # Check and add UNIQUE constraint on deed_id if it doesn't exist
            cursor.execute(f"""
                SELECT COUNT(*) FROM information_schema.table_constraints 
                WHERE table_name = '{table_name}' 
                AND constraint_type = 'UNIQUE' 
                AND constraint_name LIKE '%deed_id%'
            """)
            unique_constraint_exists = cursor.fetchone()[0] > 0
            
            if not unique_constraint_exists:
                try:
                    # First check if deed_id column exists
                    if 'deed_id' in existing_columns or columns_added:
                        cursor.execute(f"ALTER TABLE {table_name} ADD CONSTRAINT uq_{table_name}_deed_id UNIQUE (deed_id)")
                        logger.info(f"Added UNIQUE constraint on deed_id for {table_name}")
                        columns_added = True
                except Exception as e:
                    logger.warning(f"Could not add UNIQUE constraint on deed_id: {e}")
            
            # Commit column changes first
            if columns_added:
                self.connection.commit()
                logger.info(f"Successfully added missing columns to {table_name}")
            
        except Exception as e:
            logger.warning(f"Could not update table columns: {e}")
            self.connection.rollback()
        
        # Then, create indexes separately
        try:
            self._ensure_indexes_exist(cursor, table_name)
            self.connection.commit()
            logger.info(f"Index creation completed for {table_name}")
            
        except Exception as e:
            logger.warning(f"Could not create indexes: {e}")
            self.connection.rollback()
    
    def _ensure_indexes_exist(self, cursor, table_name: str):
        """Ensure all required indexes exist"""
        indexes_config = [
            (f"idx_{table_name}_province", "province"),
            (f"idx_{table_name}_district", "district"), 
            (f"idx_{table_name}_land_use_category", "land_use_category"),
            (f"idx_{table_name}_deed_type_category", "deed_type_category"),
            (f"idx_{table_name}_area_category", "area_category"),
            (f"idx_{table_name}_processing_status", "processing_status"),
            (f"idx_{table_name}_embedding_status", "embedding_status"),
            (f"idx_{table_name}_raw_metadata", "raw_metadata"),
            (f"idx_{table_name}_extracted_metadata", "extracted_metadata")
        ]
        
        for index_name, column_name in indexes_config:
            try:
                # Use separate transactions for each index to prevent cascade failures
                if 'metadata' in column_name:
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} USING GIN ({column_name})")
                else:
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({column_name})")
                # Don't commit here - let the caller handle it
            except Exception as e:
                logger.warning(f"Could not create index {index_name}: {e}")
                # Continue with other indexes even if one fails
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Database connection closed")
    
    def insert_documents(self, documents: List[SimpleDocument], batch_size: int = 100, output_manager: Optional[FileOutputManager] = None) -> int:
        """
        Insert documents into the enhanced iland_md_data table with metadata extraction
        Returns the number of successfully inserted documents
        """
        if not documents:
            logger.warning("No documents to insert into database")
            return 0
        
        if not self.connection:
            if not self.connect():
                logger.error("Failed to establish database connection")
                return 0
        
        # Import metadata extractor for enhanced processing
        try:
            from ..common.thai_provinces import ThaiProvinceMapper
        except ImportError:
            try:
                from thai_provinces import ThaiProvinceMapper
            except ImportError:
                ThaiProvinceMapper = None
                logger.warning("ThaiProvinceMapper not available - province mapping disabled")
        
        successful_inserts = 0
        
        try:
            cursor = self.connection.cursor()
            table_name = os.getenv("SOURCE_TABLE", "iland_md_data")
            
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
                    
                    # Prepare metadata for storage
                    raw_metadata = json.dumps(doc.metadata)
                    
                    # Extract key metadata fields for indexing
                    extracted_metadata = self._extract_metadata_fields(doc.metadata)
                    extracted_metadata_json = json.dumps(extracted_metadata)
                    
                    # Extract individual fields for direct querying
                    province = extracted_metadata.get('province')
                    district = extracted_metadata.get('district')
                    land_use_category = extracted_metadata.get('land_use_category')
                    deed_type_category = extracted_metadata.get('deed_type_category')
                    area_category = extracted_metadata.get('area_category')
                    
                    try:
                        # First check if the deed_id already exists
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE deed_id = %s", (deed_id,))
                        exists = cursor.fetchone()[0] > 0
                        
                        if exists:
                            # Update existing record
                            cursor.execute(
                                f"""
                                UPDATE {table_name} SET
                                    md_string = %s,
                                    raw_metadata = %s,
                                    extracted_metadata = %s,
                                    province = %s,
                                    district = %s,
                                    land_use_category = %s,
                                    deed_type_category = %s,
                                    area_category = %s,
                                    processing_status = %s,
                                    processing_timestamp = %s
                                WHERE deed_id = %s
                                """,
                                (md_string, raw_metadata, extracted_metadata_json,
                                 province, district, land_use_category, deed_type_category, area_category,
                                 'processed', datetime.now(), deed_id)
                            )
                        else:
                            # Insert new record
                            cursor.execute(
                                f"""
                                INSERT INTO {table_name} 
                                (deed_id, md_string, raw_metadata, extracted_metadata, 
                                 province, district, land_use_category, deed_type_category, area_category,
                                 processing_status, processing_timestamp)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                """,
                                (deed_id, md_string, raw_metadata, extracted_metadata_json,
                                 province, district, land_use_category, deed_type_category, area_category,
                                 'processed', datetime.now())
                            )
                        
                        successful_inserts += 1
                        
                        if successful_inserts <= 3:  # Log details for first few documents
                            action = "Updated" if exists else "Inserted"
                            logger.info(f"{action} document deed_id: {deed_id}, province: {province}, "
                                      f"land_use: {land_use_category}, text length: {len(md_string)} chars")
                            
                    except Exception as e:
                        logger.error(f"Error inserting document with deed_id {deed_id}: {e}")
                
                # Commit batch
                self.connection.commit()
                logger.info(f"Inserted batch of {len(batch)} documents - Total: {successful_inserts}/{len(documents)}")
            
            logger.info(f"Enhanced database insertion completed: {successful_inserts} documents successfully inserted")
            return successful_inserts
            
        except Exception as e:
            logger.error(f"Error during database insertion: {e}")
            self.connection.rollback()
            return successful_inserts
        finally:
            if cursor:
                cursor.close()
    
    def _extract_metadata_fields(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and categorize key metadata fields"""
        extracted = {}
        
        # Province extraction and normalization
        province_raw = metadata.get('province', '')
        if province_raw:
            # Try to normalize province name
            try:
                from ..common.thai_provinces import ThaiProvinceMapper
                mapper = ThaiProvinceMapper()
                province_normalized = mapper.normalize_province_name(province_raw)
                extracted['province'] = province_normalized
            except:
                extracted['province'] = str(province_raw).strip()
        
        # District extraction
        district_raw = metadata.get('district', '')
        if district_raw:
            extracted['district'] = str(district_raw).strip()
        
        # Land use categorization
        land_use_raw = metadata.get('land_use', metadata.get('land_main_category', ''))
        if land_use_raw:
            extracted['land_use_category'] = self._categorize_land_use(str(land_use_raw))
        
        # Deed type categorization
        deed_type_raw = metadata.get('deed_type', '')
        if deed_type_raw:
            extracted['deed_type_category'] = self._categorize_deed_type(str(deed_type_raw))
        
        # Area categorization
        area_raw = metadata.get('area', metadata.get('land_area', ''))
        if area_raw:
            extracted['area_category'] = self._categorize_area(area_raw)
        
        # Additional useful metadata
        for key in ['land_main_category', 'land_sub_category', 'owner_type', 'property_value']:
            if key in metadata and metadata[key]:
                extracted[key] = str(metadata[key]).strip()
        
        return extracted
    
    def _categorize_land_use(self, land_use: str) -> str:
        """Categorize land use into standard categories"""
        land_use_lower = land_use.lower()
        
        if any(term in land_use_lower for term in ['agricult', 'farm', 'crop', 'plantation']):
            return 'agricultural'
        elif any(term in land_use_lower for term in ['residential', 'house', 'home', 'living']):
            return 'residential'
        elif any(term in land_use_lower for term in ['commercial', 'business', 'office', 'shop', 'market']):
            return 'commercial'
        elif any(term in land_use_lower for term in ['industrial', 'factory', 'manufacturing']):
            return 'industrial'
        elif any(term in land_use_lower for term in ['forest', 'conservation', 'protected']):
            return 'conservation'
        else:
            return 'other'
    
    def _categorize_deed_type(self, deed_type: str) -> str:
        """Categorize deed type into standard categories"""
        deed_type_lower = deed_type.lower()
        
        if 'chanote' in deed_type_lower or 'ns4' in deed_type_lower or 'นส.4' in deed_type:
            return 'chanote'
        elif 'ns3' in deed_type_lower or 'นส.3' in deed_type:
            return 'nor_sor_3'
        elif 'sor kor' in deed_type_lower or 'สค.' in deed_type:
            return 'sor_kor'
        else:
            return 'other'
    
    def _categorize_area(self, area: Any) -> str:
        """Categorize area into size categories"""
        try:
            # Try to convert to float for comparison
            if isinstance(area, str):
                # Extract numeric part if area contains units
                import re
                area_match = re.search(r'[\d,]+\.?\d*', area.replace(',', ''))
                if area_match:
                    area_value = float(area_match.group())
                else:
                    return 'unknown'
            else:
                area_value = float(area)
            
            # Categorize based on rai (Thai land unit) - roughly 1600 sq meters
            if area_value < 1:
                return 'small'  # Less than 1 rai
            elif area_value < 10:
                return 'medium'  # 1-10 rai
            elif area_value < 50:
                return 'large'  # 10-50 rai
            else:
                return 'very_large'  # 50+ rai
                
        except (ValueError, TypeError):
            return 'unknown'
    
    def setup_chunks_table(self):
        """Create the chunks table for storing section-based chunks with embeddings"""
        if not self.connection:
            logger.error("No database connection available")
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # Create chunks table for BGE-M3 embeddings
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS iland_chunks (
                    id SERIAL PRIMARY KEY,
                    deed_id VARCHAR(255) NOT NULL,
                    parent_doc_id INTEGER REFERENCES iland_md_data(id),
                    
                    -- Chunk content
                    text TEXT NOT NULL,
                    embedding_vector REAL[],
                    
                    -- Section metadata
                    section_type VARCHAR(50) NOT NULL,
                    section_name VARCHAR(100),
                    chunk_type VARCHAR(20) NOT NULL,
                    is_primary_chunk BOOLEAN DEFAULT FALSE,
                    chunk_index INTEGER NOT NULL,
                    total_chunks INTEGER NOT NULL,
                    
                    -- BGE-M3 metadata
                    embedding_model VARCHAR(50) DEFAULT 'BAAI/bge-m3',
                    embedding_dim INTEGER DEFAULT 1024,
                    
                    -- Full metadata from document
                    metadata JSONB,
                    
                    -- Processing timestamps
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    CONSTRAINT check_embedding_dim CHECK (
                        embedding_vector IS NULL OR 
                        array_length(embedding_vector, 1) = embedding_dim
                    )
                );
            """)
            
            # Create indexes for efficient retrieval
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_deed_id ON iland_chunks(deed_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_parent_doc ON iland_chunks(parent_doc_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_section_type ON iland_chunks(section_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chunk_type ON iland_chunks(chunk_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_primary ON iland_chunks(is_primary_chunk) WHERE is_primary_chunk = TRUE")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_metadata ON iland_chunks USING GIN(metadata)")
            
            # Try to add pgvector support if available
            try:
                cursor.execute("ALTER TABLE iland_chunks ADD COLUMN IF NOT EXISTS embedding_pgvector vector(1024)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON iland_chunks USING ivfflat (embedding_pgvector vector_cosine_ops)")
                logger.info("pgVector support enabled for similarity search")
            except Exception as e:
                logger.warning(f"Could not add pgVector support: {e}")
            
            self.connection.commit()
            logger.info("Successfully created/updated chunks table for BGE-M3 embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up chunks table: {e}")
            self.connection.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
    
    def insert_chunks(self, chunks: List[Dict[str, Any]], parent_doc_id: int) -> int:
        """Insert section-based chunks into the chunks table"""
        if not chunks:
            return 0
        
        if not self.connection:
            if not self.connect():
                logger.error("Failed to establish database connection")
                return 0
        
        successful_inserts = 0
        
        try:
            cursor = self.connection.cursor()
            
            for chunk in chunks:
                try:
                    cursor.execute("""
                        INSERT INTO iland_chunks 
                        (deed_id, parent_doc_id, text, section_type, section_name, 
                         chunk_type, is_primary_chunk, chunk_index, total_chunks, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        chunk['deed_id'],
                        parent_doc_id,
                        chunk['text'],
                        chunk['section_type'],
                        chunk.get('section_name'),
                        chunk['chunk_type'],
                        chunk.get('is_primary', False),
                        chunk['chunk_index'],
                        chunk['total_chunks'],
                        json.dumps(chunk.get('metadata', {}))
                    ))
                    
                    chunk_id = cursor.fetchone()[0]
                    successful_inserts += 1
                    
                    if successful_inserts <= 3:
                        logger.info(f"Inserted chunk {chunk_id}: section={chunk['section_type']}, "
                                  f"type={chunk['chunk_type']}, index={chunk['chunk_index']}/{chunk['total_chunks']}")
                    
                except Exception as e:
                    logger.error(f"Error inserting chunk: {e}")
            
            self.connection.commit()
            logger.info(f"Successfully inserted {successful_inserts}/{len(chunks)} chunks")
            return successful_inserts
            
        except Exception as e:
            logger.error(f"Error during chunk insertion: {e}")
            self.connection.rollback()
            return successful_inserts
        finally:
            if cursor:
                cursor.close()
    
    def update_chunk_embeddings(self, chunk_embeddings: List[Dict[str, Any]]) -> int:
        """Update chunks with BGE-M3 embeddings"""
        if not chunk_embeddings:
            return 0
        
        if not self.connection:
            if not self.connect():
                return 0
        
        successful_updates = 0
        
        try:
            cursor = self.connection.cursor()
            
            for chunk_data in chunk_embeddings:
                try:
                    # Update with embedding vector
                    cursor.execute("""
                        UPDATE iland_chunks 
                        SET embedding_vector = %s,
                            embedding_model = %s,
                            embedding_dim = %s,
                            processed_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """, (
                        chunk_data['embedding_vector'],
                        chunk_data.get('embedding_model', 'BAAI/bge-m3'),
                        chunk_data.get('embedding_dim', 1024),
                        chunk_data['chunk_id']
                    ))
                    
                    # Also update pgvector column if available
                    if 'embedding_pgvector' in chunk_data:
                        cursor.execute("""
                            UPDATE iland_chunks 
                            SET embedding_pgvector = %s
                            WHERE id = %s
                        """, (chunk_data['embedding_pgvector'], chunk_data['chunk_id']))
                    
                    successful_updates += 1
                    
                except Exception as e:
                    logger.error(f"Error updating chunk {chunk_data.get('chunk_id')} with embedding: {e}")
            
            self.connection.commit()
            logger.info(f"Successfully updated {successful_updates}/{len(chunk_embeddings)} chunks with embeddings")
            return successful_updates
            
        except Exception as e:
            logger.error(f"Error during embedding update: {e}")
            self.connection.rollback()
            return successful_updates
        finally:
            if cursor:
                cursor.close()
    
    def get_pending_chunks_for_embedding(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get chunks that need embeddings"""
        if not self.connection:
            if not self.connect():
                return []
        
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                SELECT id, deed_id, text, section_type, chunk_type, metadata
                FROM iland_chunks
                WHERE embedding_vector IS NULL
                LIMIT %s
            """, (limit,))
            
            columns = [desc[0] for desc in cursor.description]
            chunks = []
            
            for row in cursor.fetchall():
                chunk = dict(zip(columns, row))
                # Parse JSONB metadata
                if chunk['metadata']:
                    chunk['metadata'] = chunk['metadata']
                chunks.append(chunk)
            
            logger.info(f"Found {len(chunks)} chunks pending embedding")
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting pending chunks: {e}")
            return []
        finally:
            if cursor:
                cursor.close() 