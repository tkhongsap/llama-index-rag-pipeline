# Product Requirements Document (PRD)
## PostgreSQL Embedding Module Enhancement with BGE-M3 for iLand RAG Pipeline

**Document Version:** 2.0  
**Last Updated:** June 10, 2025  
**Status:** Ready for Implementation  
**Priority:** Critical - Security & Performance

---

## 1. Executive Summary

This PRD outlines the critical enhancement of the `docs_embedding_postgres` module to:
- **Implement BGE-M3 as the primary embedding model** matching the main `docs_embedding` module
- **Integrate advanced section-based chunking** reducing chunks from ~169 to ~6 per document
- **Ensure zero external data transmission** for security compliance
- **Preserve complete metadata** throughout the embedding pipeline

The current PostgreSQL implementation's reliance on OpenAI embeddings and basic sentence splitting creates security risks and poor retrieval quality for Thai land deed documents.

---

## 2. Background & Problem Statement

### 2.1 Current State Analysis

| Issue | Current State | Impact |
|-------|--------------|--------|
| **Embedding Model** | Hardcoded OpenAI API | Sensitive data sent externally |
| **Chunking Strategy** | Simple SentenceSplitter | 169 chunks vs optimal 6 chunks |
| **Metadata Handling** | Basic metadata only | Lost section context |
| **Security** | Requires internet access | Non-compliant for government use |
| **Cost** | Pay-per-token API calls | Unnecessary operational costs |

### 2.2 Business Impact
- **Security Risk**: Cannot deploy in government/enterprise environments
- **Poor Search Quality**: 28x more chunks fragment context
- **Inefficient Storage**: Excessive database records
- **High Costs**: Unnecessary OpenAI API expenses

---

## 3. Goals & Objectives

### 3.1 Primary Goals
1. ✅ **Implement BGE-M3 embedding model** for 100% on-premise processing
2. ✅ **Integrate section-based chunking** from `standalone_section_parser.py`
3. ✅ **Zero external API calls** - complete data sovereignty
4. ✅ **Full metadata preservation** with section-aware storage

### 3.2 Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Chunks per document | ~169 | 6-10 |
| External API calls | All embeddings | 0 |
| Embedding dimensions | 1536 (OpenAI) | 1024 (BGE-M3) |
| Processing location | Cloud | 100% local |
| Metadata fields preserved | 5-6 | 15+ |

---

## 4. Detailed Technical Requirements

### 4.1 BGE-M3 Embedding Implementation

```python
# New postgres_embedding.py implementation
from FlagEmbedding import FlagModel
import torch
from typing import List, Dict, Optional
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PostgresEmbeddingGenerator:
    """
    Enhanced embedding generator using BGE-M3 model with section-based chunking
    """
    
    def __init__(
        self,
        # Model configuration
        model_name: str = "BAAI/bge-m3",
        device: str = "auto",
        use_fp16: bool = True,
        normalize_embeddings: bool = True,
        model_cache_dir: Optional[str] = None,
        
        # Chunking configuration
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_section_size: int = 50,
        
        # Processing configuration
        batch_size: int = 32,
        max_length: int = 8192,
        
        # Database configuration
        db_connection_string: str = None,
    ):
        """Initialize BGE-M3 embedding generator with section parser"""
        
        # Model settings
        self.model_name = model_name
        self.device = self._get_device(device)
        self.use_fp16 = use_fp16 and self.device != "cpu"
        self.normalize_embeddings = normalize_embeddings
        self.model_cache_dir = model_cache_dir or "./models"
        
        # Processing settings
        self.batch_size = batch_size
        self.max_length = max_length
        self.embedding_dim = 1024  # BGE-M3 dimension
        
        # Initialize components
        self._initialize_bge_model()
        self._initialize_section_parser(chunk_size, chunk_overlap, min_section_size)
        
        logger.info(f"Initialized BGE-M3 embedding generator on {self.device}")
    
    def _initialize_bge_model(self):
        """Initialize BGE-M3 model for local embeddings"""
        try:
            # Ensure model cache directory exists
            Path(self.model_cache_dir).mkdir(parents=True, exist_ok=True)
            
            # Initialize BGE-M3 with proper settings
            self.embed_model = FlagModel(
                self.model_name,
                query_instruction_for_retrieval=(
                    "Represent this query for retrieving relevant Thai land deed documents: "
                ),
                use_fp16=self.use_fp16,
                device=self.device,
                cache_dir=self.model_cache_dir
            )
            
            logger.info(f"BGE-M3 model loaded successfully from {self.model_cache_dir}")
            
        except Exception as e:
            logger.error(f"Failed to initialize BGE-M3 model: {e}")
            raise RuntimeError(f"BGE-M3 initialization failed: {e}")
    
    def _initialize_section_parser(self, chunk_size: int, chunk_overlap: int, min_section_size: int):
        """Initialize section-based parser for Thai land deeds"""
        from standalone_section_parser import StandaloneLandDeedSectionParser
        
        self.section_parser = StandaloneLandDeedSectionParser(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_section_size=min_section_size,
            include_metadata_in_chunks=True,
            preserve_formatting=True,
            language='th'  # Thai language support
        )
        
        logger.info("Section parser initialized for Thai land deed documents")
```

### 4.2 Section-Based Document Processing

```python
def process_documents(self, documents: List[Document]) -> List[Dict]:
    """
    Process documents using section-based chunking matching docs_embedding module
    """
    all_chunks = []
    processing_stats = {
        'total_documents': len(documents),
        'total_chunks': 0,
        'chunks_by_type': defaultdict(int),
        'sections_by_type': defaultdict(int)
    }
    
    for doc_idx, doc in enumerate(documents):
        deed_id = doc.metadata.get('deed_id', f'unknown_{doc_idx}')
        logger.info(f"Processing document {deed_id} ({doc_idx + 1}/{len(documents)})")
        
        try:
            # Parse document into semantic sections
            sections = self.section_parser.parse_document_to_sections(
                document_text=doc.text,
                metadata=doc.metadata
            )
            
            logger.info(f"Document {deed_id} parsed into {len(sections)} sections")
            
            # Process each section
            for section_idx, section in enumerate(sections):
                # Create chunk with full metadata
                chunk = self._create_chunk_from_section(
                    section=section,
                    document=doc,
                    section_index=section_idx,
                    total_sections=len(sections)
                )
                
                all_chunks.append(chunk)
                
                # Update statistics
                processing_stats['chunks_by_type'][section.chunk_type] += 1
                processing_stats['sections_by_type'][section.type] += 1
            
            processing_stats['total_chunks'] += len(sections)
            
        except Exception as e:
            logger.error(f"Error processing document {deed_id}: {e}")
            # Add fallback chunk for failed documents
            fallback_chunk = self._create_fallback_chunk(doc, str(e))
            all_chunks.append(fallback_chunk)
    
    # Log processing statistics
    self._log_processing_stats(processing_stats)
    
    return all_chunks

def _create_chunk_from_section(
    self, 
    section: Any, 
    document: Document, 
    section_index: int, 
    total_sections: int
) -> Dict:
    """Create a chunk with complete metadata from a section"""
    
    return {
        'text': section.text,
        'metadata': {
            # Document metadata (preserved from original)
            'deed_id': document.metadata.get('deed_id'),
            'land_id': document.metadata.get('land_id'),
            'province': document.metadata.get('province'),
            'district': document.metadata.get('district'),
            'subdistrict': document.metadata.get('subdistrict'),
            'deed_type': document.metadata.get('deed_type'),
            'deed_number': document.metadata.get('deed_number'),
            'registration_date': document.metadata.get('registration_date'),
            
            # Location metadata
            'location_hierarchy': document.metadata.get('location_hierarchy'),
            'coordinates_formatted': document.metadata.get('coordinates_formatted'),
            'longitude': document.metadata.get('longitude'),
            'latitude': document.metadata.get('latitude'),
            'google_maps_url': document.metadata.get('google_maps_url'),
            
            # Area metadata
            'area_rai': document.metadata.get('area_rai'),
            'area_ngan': document.metadata.get('area_ngan'),
            'area_wa': document.metadata.get('area_wa'),
            'area_formatted': document.metadata.get('area_formatted'),
            'area_total_sqm': document.metadata.get('area_total_sqm'),
            
            # Section metadata
            'section_type': section.type,  # deed_info, location, area_measurements, etc.
            'section_name': section.name,
            'chunk_type': section.chunk_type,  # key_info, section, fallback
            'is_primary_chunk': section.is_primary,
            'chunk_index': section_index,
            'total_chunks': total_sections,
            
            # Processing metadata
            'embedding_model': self.model_name,
            'embedding_dim': self.embedding_dim,
            'chunk_size': len(section.text),
            'processed_at': datetime.now().isoformat(),
            'processing_version': '2.0',
            
            # Security metadata
            'processed_locally': True,
            'external_apis_used': [],
            'data_transmitted_externally': False
        }
    }
```

### 4.3 Enhanced Database Schema

```sql
-- Drop old tables if migrating
-- DROP TABLE IF EXISTS iland_chunks CASCADE;

-- Enhanced chunks table with BGE-M3 support
CREATE TABLE IF NOT EXISTS iland_chunks (
    id SERIAL PRIMARY KEY,
    
    -- Document identifiers
    deed_id VARCHAR(255) NOT NULL,
    land_id VARCHAR(255),
    
    -- Chunk content
    text TEXT NOT NULL,
    embedding_vector REAL[] NOT NULL,
    
    -- Section metadata
    section_type VARCHAR(50) NOT NULL,
    section_name VARCHAR(100),
    chunk_type VARCHAR(20) NOT NULL, -- key_info, section, fallback
    is_primary_chunk BOOLEAN DEFAULT FALSE,
    chunk_index INTEGER NOT NULL,
    total_chunks INTEGER NOT NULL,
    
    -- Location metadata
    province VARCHAR(100),
    district VARCHAR(100),
    subdistrict VARCHAR(100),
    location_hierarchy TEXT,
    
    -- Coordinates
    longitude DOUBLE PRECISION,
    latitude DOUBLE PRECISION,
    coordinates_formatted VARCHAR(50),
    
    -- Area measurements
    area_rai DECIMAL(10,2),
    area_ngan DECIMAL(10,2),
    area_wa DECIMAL(10,2),
    area_formatted TEXT,
    area_total_sqm DECIMAL(15,2),
    
    -- Model metadata
    embedding_model VARCHAR(50) NOT NULL DEFAULT 'BAAI/bge-m3',
    embedding_dim INTEGER NOT NULL DEFAULT 1024,
    
    -- Processing metadata
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_version VARCHAR(10) DEFAULT '2.0',
    
    -- Full metadata JSONB for flexibility
    metadata JSONB,
    
    -- Constraints
    CONSTRAINT check_embedding_dim CHECK (array_length(embedding_vector, 1) = embedding_dim),
    CONSTRAINT check_bge_m3_dim CHECK (
        embedding_model != 'BAAI/bge-m3' OR embedding_dim = 1024
    )
);

-- Indexes for efficient retrieval
CREATE INDEX idx_chunks_deed_id ON iland_chunks(deed_id);
CREATE INDEX idx_chunks_section_type ON iland_chunks(section_type);
CREATE INDEX idx_chunks_chunk_type ON iland_chunks(chunk_type);
CREATE INDEX idx_chunks_primary ON iland_chunks(is_primary_chunk) WHERE is_primary_chunk = TRUE;
CREATE INDEX idx_chunks_location ON iland_chunks(province, district, subdistrict);
CREATE INDEX idx_chunks_coordinates ON iland_chunks(longitude, latitude);
CREATE INDEX idx_chunks_metadata ON iland_chunks USING GIN(metadata);

-- Vector similarity search support (requires pgvector extension)
-- CREATE EXTENSION IF NOT EXISTS vector;
-- ALTER TABLE iland_chunks ADD COLUMN embedding vector(1024);
-- CREATE INDEX idx_chunks_embedding ON iland_chunks USING ivfflat (embedding vector_cosine_ops);
```

### 4.4 Complete Embedding Pipeline

```python
# Enhanced embedding_processor.py
class EmbeddingProcessor:
    """Enhanced processor with BGE-M3 and section-based chunking"""
    
    def __init__(self):
        # Initialize with BGE-M3 only - no OpenAI
        self.embed_generator = PostgresEmbeddingGenerator(
            model_name="BAAI/bge-m3",
            device="auto",  # Use GPU if available
            use_fp16=True,  # Use half precision for efficiency
            normalize_embeddings=True,
            model_cache_dir=os.getenv("MODEL_CACHE_DIR", "./models"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
        )
        self.db_utils = DatabaseUtils()
    
    def process_and_store_embeddings(self, documents: List[Document]):
        """Process documents and store embeddings locally"""
        logger.info(f"Processing {len(documents)} documents with BGE-M3")
        
        # Step 1: Parse documents into semantic sections
        all_chunks = self.embed_generator.process_documents(documents)
        logger.info(f"Generated {len(all_chunks)} chunks from {len(documents)} documents")
        
        # Step 2: Generate embeddings locally with BGE-M3
        texts = [chunk['text'] for chunk in all_chunks]
        embeddings = self.embed_generator.get_text_embedding_batch(
            texts=texts,
            show_progress=True
        )
        
        # Step 3: Store embeddings with full metadata
        for chunk, embedding in zip(all_chunks, embeddings):
            chunk_data = {
                'deed_id': chunk['metadata'].get('deed_id'),
                'text': chunk['text'],
                'embedding_vector': embedding,  # 1024-dimensional BGE-M3 embedding
                'metadata': chunk['metadata']
            }
            
            self.db_utils.save_chunk_embedding(chunk_data)
        
        logger.info("All embeddings stored successfully in PostgreSQL")
```

---

## 5. Migration Strategy

### 5.1 Database Schema Migration

```sql
-- migration/01_add_bge_m3_support.sql
-- Update existing table for BGE-M3 embeddings

-- Step 1: Add new columns for section metadata
ALTER TABLE iland_chunks 
    ADD COLUMN IF NOT EXISTS section_type VARCHAR(50),
    ADD COLUMN IF NOT EXISTS section_name VARCHAR(100),
    ADD COLUMN IF NOT EXISTS chunk_type VARCHAR(20) DEFAULT 'fallback',
    ADD COLUMN IF NOT EXISTS is_primary_chunk BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS embedding_model VARCHAR(50) DEFAULT 'BAAI/bge-m3',
    ADD COLUMN IF NOT EXISTS embedding_dim INTEGER DEFAULT 1024,
    ADD COLUMN IF NOT EXISTS processing_version VARCHAR(10) DEFAULT '2.0';

-- Step 2: Create backup table
CREATE TABLE IF NOT EXISTS iland_chunks_backup AS 
SELECT * FROM iland_chunks;

-- Step 3: Add constraints for BGE-M3
ALTER TABLE iland_chunks 
    ADD CONSTRAINT check_bge_m3_dim CHECK (
        embedding_model != 'BAAI/bge-m3' OR embedding_dim = 1024
    );

-- Step 4: Create new indexes
CREATE INDEX IF NOT EXISTS idx_chunks_section_type ON iland_chunks(section_type);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_type ON iland_chunks(chunk_type);
CREATE INDEX IF NOT EXISTS idx_chunks_primary ON iland_chunks(is_primary_chunk) WHERE is_primary_chunk = TRUE;
CREATE INDEX IF NOT EXISTS idx_chunks_location ON iland_chunks(province, district, subdistrict);
```

### 5.2 Data Migration Script

```python
# migration/migrate_to_bge_m3.py
import logging
from typing import List, Dict
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

class BGE_M3_Migration:
    """Migrate existing embeddings to BGE-M3 with section-based chunking"""
    
    def __init__(self, batch_size: int = 50):
        self.batch_size = batch_size
        self.db_utils = DatabaseUtils()
        self.embedding_generator = PostgresEmbeddingGenerator(
            device="auto",
            batch_size=32
        )
        
    def migrate_database_schema(self):
        """Execute schema migration"""
        logger.info("Updating database schema for BGE-M3")
        
        migration_sql_file = Path(__file__).parent / "01_add_bge_m3_support.sql"
        with open(migration_sql_file, 'r') as f:
            migration_sql = f.read()
        
        self.db_utils.execute_sql(migration_sql)
        logger.info("Schema updated successfully")
    
    def reprocess_documents(self):
        """Reprocess all documents with BGE-M3 and section parser"""
        logger.info("Starting document reprocessing with BGE-M3")
        
        # Load documents from the original data source
        documents = self.db_utils.load_original_documents()
        total_docs = len(documents)
        
        logger.info(f"Found {total_docs} documents to reprocess")
        
        # Clear existing chunks
        self.db_utils.clear_existing_chunks()
        
        # Process in batches
        for i in range(0, total_docs, self.batch_size):
            batch = documents[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total_docs + self.batch_size - 1) // self.batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
            
            try:
                # Process with section-based chunking
                chunks = self.embedding_generator.process_documents(batch)
                
                # Generate BGE-M3 embeddings
                texts = [chunk['text'] for chunk in chunks]
                embeddings = self.embedding_generator.get_text_embedding_batch(texts)
                
                # Combine chunks with embeddings
                embedded_chunks = []
                for chunk, embedding in zip(chunks, embeddings):
                    chunk['embedding'] = embedding
                    embedded_chunks.append(chunk)
                
                # Save to database
                self.db_utils.save_chunk_embeddings(embedded_chunks)
                
                avg_chunks = len(embedded_chunks) / len(batch)
                logger.info(f"Batch {batch_num} processed: {len(batch)} docs -> {len(embedded_chunks)} chunks (avg: {avg_chunks:.1f})")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                raise
    
    def verify_migration(self) -> Dict:
        """Verify migration success"""
        stats = self.db_utils.get_embedding_statistics()
        
        verification_results = {
            'total_chunks': stats['total_chunks'],
            'avg_chunks_per_doc': stats['avg_chunks_per_doc'],
            'bge_m3_chunks': stats['bge_m3_chunks'],
            'primary_chunks': stats['primary_chunks'],
            'section_types': stats['section_types'],
            'migration_successful': stats['avg_chunks_per_doc'] < 20
        }
        
        logger.info("Migration verification:")
        logger.info(f"- Total chunks: {verification_results['total_chunks']}")
        logger.info(f"- Average chunks per document: {verification_results['avg_chunks_per_doc']:.1f}")
        logger.info(f"- BGE-M3 chunks: {verification_results['bge_m3_chunks']}")
        logger.info(f"- Primary chunks: {verification_results['primary_chunks']}")
        logger.info(f"- Section types found: {verification_results['section_types']}")
        
        if verification_results['migration_successful']:
            logger.info("✅ Migration successful!")
        else:
            logger.warning("⚠️ Migration may have issues - reviewing chunk distribution")
        
        return verification_results

if __name__ == "__main__":
    migration = BGE_M3_Migration()
    
    try:
        # Step 1: Update schema
        migration.migrate_database_schema()
        
        # Step 2: Reprocess documents
        migration.reprocess_documents()
        
        # Step 3: Verify
        stats = migration.verify_migration()
        
        if stats['migration_successful']:
            print("Migration completed successfully!")
        else:
            print("Migration completed with warnings. Please review the results.")
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise
```

---

## 6. Testing Strategy

### 6.1 Unit Tests

```python
# tests/test_bge_m3_embedding.py
import pytest
import numpy as np
from unittest.mock import Mock, patch

class TestBGEM3Embedding:
    """Test BGE-M3 embedding functionality"""
    
    def test_no_external_api_calls(self):
        """Ensure no external APIs are called"""
        with patch('requests.post') as mock_post:
            with patch('urllib.request.urlopen') as mock_urlopen:
                generator = PostgresEmbeddingGenerator()
                
                # Process test document
                test_doc = self._create_test_document()
                chunks = generator.process_documents([test_doc])
                
                # Generate embeddings
                texts = [chunk['text'] for chunk in chunks]
                embeddings = generator.get_text_embedding_batch(texts)
                
                # Verify no external calls
                mock_post.assert_not_called()
                mock_urlopen.assert_not_called()
                assert len(embeddings) == len(chunks)
    
    def test_embedding_dimensions(self):
        """Verify BGE-M3 produces 1024-dim embeddings"""
        generator = PostgresEmbeddingGenerator()
        result = generator.get_text_embedding("โฉนดที่ดินเลขที่ 12345")
        
        assert len(result) == 1024
        assert all(isinstance(x, float) for x in result)
        assert -1 <= min(result) <= max(result) <= 1  # Normalized
    
    def test_section_based_chunking(self):
        """Test section parser integration"""
        generator = PostgresEmbeddingGenerator()
        
        # Create test Thai land deed document
        test_doc = self._create_test_land_deed()
        chunks = generator.process_documents([test_doc])
        
        # Should produce 6-10 chunks, not 169
        assert 6 <= len(chunks) <= 10, f"Expected 6-10 chunks, got {len(chunks)}"
        
        # Check section types
        section_types = {c['metadata']['section_type'] for c in chunks}
        expected_sections = {'deed_info', 'location', 'area_measurements'}
        assert expected_sections.issubset(section_types)
        
        # Check metadata preservation
        for chunk in chunks:
            assert 'deed_id' in chunk['metadata']
            assert 'section_type' in chunk['metadata']
            assert 'chunk_type' in chunk['metadata']
            assert 'processed_locally' in chunk['metadata']
            assert chunk['metadata']['processed_locally'] is True
    
    def _create_test_document(self):
        """Create a test document"""
        return Document(
            text="Test document content",
            metadata={'deed_id': 'test_001'}
        )
    
    def _create_test_land_deed(self):
        """Create a realistic Thai land deed document"""
        return Document(
            text="""# บันทึกข้อมูลโฉนดที่ดิน (Land Deed Record)

## ข้อมูลโฉนด (Deed Information)
- รหัสโฉนด: 12345
- ประเภทโฉนด: โฉนดที่ดิน
- เลขที่โฉนด: 67890

## ที่ตั้ง (Location)
- ที่ตั้ง: นครราชสีมา > เมืองนครราชสีมา > ในเมือง
- จังหวัด: นครราชสีมา
- อำเภอ: เมืองนครราชสีมา
- ตำบล: ในเมือง

## ขนาดพื้นที่ (Area Measurements)
- เนื้อที่: 2 ไร่ 1 งาน 50 ตร.ว.
- พื้นที่รวม (ตร.ม.): 3400.0""",
            metadata={
                'deed_id': 'test_thai_001',
                'province': 'นครราชสีมา',
                'district': 'เมืองนครราชสีมา',
                'subdistrict': 'ในเมือง',
                'deed_type': 'โฉนดที่ดิน',
                'area_rai': 2,
                'area_ngan': 1,
                'area_wa': 50
            }
        )
```

### 6.2 Integration Tests

```python
# tests/test_postgres_integration.py
class TestPostgresIntegration:
    """Test full pipeline with PostgreSQL"""
    
    @pytest.fixture
    def test_database(self):
        """Create test database"""
        test_db = TestDatabaseUtils()
        test_db.setup_test_schema()
        yield test_db
        test_db.cleanup()
    
    def test_end_to_end_processing(self, test_database):
        """Test complete document processing pipeline"""
        # Load test document
        doc = self._create_test_land_deed()
        
        # Process with BGE-M3
        processor = EmbeddingProcessor()
        processor.process_and_store_embeddings([doc])
        
        # Verify in database
        chunks = test_database.get_chunks_for_deed(doc.metadata['deed_id'])
        
        # Verify chunking efficiency
        assert len(chunks) < 20, f"Too many chunks: {len(chunks)}"
        
        # Verify BGE-M3 embeddings
        for chunk in chunks:
            assert len(chunk['embedding_vector']) == 1024
            assert chunk['embedding_model'] == 'BAAI/bge-m3'
            assert chunk['embedding_dim'] == 1024
        
        # Verify section metadata
        section_types = {chunk['section_type'] for chunk in chunks}
        assert len(section_types) >= 3  # At least 3 different section types
        
        # Verify primary chunks exist
        primary_chunks = [c for c in chunks if c['is_primary_chunk']]
        assert len(primary_chunks) >= 1
    
    def test_search_quality_improvement(self, test_database):
        """Test that section-based chunking improves search"""
        # Process document with section-based chunking
        doc = self._create_test_land_deed()
        processor = EmbeddingProcessor()
        processor.process_and_store_embeddings([doc])
        
        # Perform search query
        query = "โฉนดที่ดินใน นครราชสีมา"
        results = test_database.similarity_search(query, top_k=5)
        
        # Verify relevant results returned
        assert len(results) > 0
        
        # Verify section context preserved
        location_results = [r for r in results if r['section_type'] == 'location']
        deed_info_results = [r for r in results if r['section_type'] == 'deed_info']
        
        assert len(location_results) > 0  # Location section should be found
        assert len(deed_info_results) > 0  # Deed info should be found
```

---

## 7. Performance Specifications

### 7.1 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 4 cores, 2.5GHz | 8 cores, 3.0GHz |
| **RAM** | 8GB | 16GB |
| **GPU** | None (CPU only) | NVIDIA GPU with 8GB VRAM |
| **Storage** | 10GB free space | 50GB SSD |
| **Network** | Internet for initial model download | Local network only |

### 7.2 Performance Benchmarks

| Metric | BGE-M3 (GPU) | BGE-M3 (CPU) | Current (OpenAI) |
|--------|--------------|--------------|------------------|
| **Model Load Time** | 10-15s | 15-20s | N/A (API) |
| **Documents/hour** | 1000+ | 200-300 | 500 |
| **Embedding Speed (100 docs)** | 2-3s | 15-20s | 5-10s |
| **Memory Usage** | ~4GB | ~2GB | ~100MB |
| **Batch Size** | 32 | 8 | 100 |
| **Chunks per Document** | 6-10 | 6-10 | 169 |
| **Storage per Document** | 90% less | 90% less | Current baseline |

### 7.3 Performance Monitoring

```python
# monitoring/performance_tracker.py
class PerformanceTracker:
    """Track embedding pipeline performance"""
    
    def __init__(self):
        self.metrics = {
            'documents_processed': 0,
            'total_chunks': 0,
            'processing_times': [],
            'memory_usage': [],
            'chunk_distribution': defaultdict(int),
            'section_distribution': defaultdict(int)
        }
    
    def log_document_processing(
        self, 
        doc_id: str, 
        chunks: int, 
        duration: float,
        memory_mb: float,
        section_types: List[str]
    ):
        self.metrics['documents_processed'] += 1
        self.metrics['total_chunks'] += chunks
        self.metrics['processing_times'].append(duration)
        self.metrics['memory_usage'].append(memory_mb)
        self.metrics['chunk_distribution'][chunks] += 1
        
        for section_type in section_types:
            self.metrics['section_distribution'][section_type] += 1
        
        # Alert on anomalies
        if chunks > 20:
            logger.warning(f"Document {doc_id} produced {chunks} chunks (expected < 20)")
        
        if duration > 5.0:
            logger.warning(f"Document {doc_id} took {duration:.1f}s to process")
        
        if memory_mb > 8000:
            logger.warning(f"High memory usage: {memory_mb:.1f}MB")
    
    def generate_report(self) -> Dict:
        """Generate performance report"""
        if not self.metrics['documents_processed']:
            return {'error': 'No documents processed yet'}
        
        return {
            'total_documents': self.metrics['documents_processed'],
            'total_chunks': self.metrics['total_chunks'],
            'avg_chunks_per_doc': self.metrics['total_chunks'] / self.metrics['documents_processed'],
            'avg_processing_time': np.mean(self.metrics['processing_times']),
            'max_processing_time': max(self.metrics['processing_times']),
            'peak_memory_mb': max(self.metrics['memory_usage']),
            'avg_memory_mb': np.mean(self.metrics['memory_usage']),
            'documents_per_hour': 3600 / np.mean(self.metrics['processing_times']),
            'chunk_distribution': dict(self.metrics['chunk_distribution']),
            'section_distribution': dict(self.metrics['section_distribution'])
        }
    
    def check_performance_health(self) -> Dict:
        """Check if performance is within expected bounds"""
        report = self.generate_report()
        
        health_checks = {
            'avg_chunks_reasonable': report['avg_chunks_per_doc'] < 15,
            'processing_speed_ok': report['avg_processing_time'] < 5.0,
            'memory_usage_ok': report['peak_memory_mb'] < 8000,
            'throughput_ok': report['documents_per_hour'] > 100
        }
        
        health_checks['overall_healthy'] = all(health_checks.values())
        
        return health_checks
```

---

## 8. Security & Compliance

### 8.1 Security Guarantees

| Requirement | Implementation | Verification Method |
|------------|----------------|-------------------|
| **No External Data Transfer** | BGE-M3 runs locally | Network monitoring during processing |
| **Data Sovereignty** | All processing on-premise | Audit logs show no external connections |
| **No API Keys Required** | Local model only | Configuration verification |
| **Encrypted Storage** | PostgreSQL with TLS | Database configuration check |
| **Access Control** | Database authentication | User permission audit |

### 8.2 Security Implementation

```python
# security/compliance_checker.py
class ComplianceChecker:
    """Verify security and compliance requirements"""
    
    def __init__(self):
        self.security_violations = []
        self.compliance_status = {}
    
    def check_no_external_apis(self, embedding_generator):
        """Verify no external API calls are made"""
        # Check if OpenAI client is initialized
        if hasattr(embedding_generator, 'openai_client'):
            self.security_violations.append("OpenAI client detected")
        
        # Verify BGE-M3 is local
        if not hasattr(embedding_generator, 'embed_model'):
            self.security_violations.append("Local embedding model not found")
        
        self.compliance_status['no_external_apis'] = len(self.security_violations) == 0
    
    def check_data_locality(self, chunk_metadata):
        """Verify all data processing is local"""
        for chunk in chunk_metadata:
            if not chunk.get('processed_locally', False):
                self.security_violations.append(f"Chunk {chunk['id']} not processed locally")
            
            if chunk.get('data_transmitted_externally', True):
                self.security_violations.append(f"Chunk {chunk['id']} data transmitted externally")
        
        self.compliance_status['data_locality'] = len(self.security_violations) == 0
    
    def check_model_storage(self, model_cache_dir):
        """Verify model is stored locally"""
        model_path = Path(model_cache_dir) / "BAAI" / "bge-m3"
        
        if not model_path.exists():
            self.security_violations.append("BGE-M3 model not found locally")
        
        self.compliance_status['local_model'] = model_path.exists()
    
    def generate_compliance_report(self) -> Dict:
        """Generate compliance report"""
        return {
            'compliant': len(self.security_violations) == 0,
            'violations': self.security_violations,
            'status_checks': self.compliance_status,
            'recommendation': self._get_recommendation()
        }
    
    def _get_recommendation(self) -> str:
        if len(self.security_violations) == 0:
            return "✅ All security requirements met. Safe for government deployment."
        else:
            return f"❌ {len(self.security_violations)} security violations found. Fix before deployment."
```

### 8.3 Audit Logging

```python
# security/audit_logger.py
class AuditLogger:
    """Log all operations for compliance auditing"""
    
    def __init__(self, log_file: str = "audit.log"):
        self.logger = logging.getLogger("audit")
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_model_initialization(self, model_name: str, cache_dir: str):
        """Log model initialization"""
        self.logger.info(f"MODEL_INIT: {model_name} loaded from {cache_dir}")
    
    def log_document_processing(self, deed_id: str, chunks_created: int):
        """Log document processing"""
        self.logger.info(f"DOC_PROCESS: {deed_id} -> {chunks_created} chunks")
    
    def log_embedding_generation(self, text_count: int, model: str):
        """Log embedding generation"""
        self.logger.info(f"EMBED_GEN: {text_count} texts processed with {model}")
    
    def log_database_operation(self, operation: str, table: str, rows: int):
        """Log database operations"""
        self.logger.info(f"DB_OP: {operation} {rows} rows in {table}")
    
    def log_security_check(self, check_type: str, result: bool):
        """Log security checks"""
        status = "PASS" if result else "FAIL"
        self.logger.info(f"SECURITY_CHECK: {check_type} - {status}")
```

---

## 9. Success Criteria & Validation

### 9.1 Technical Success Criteria

| Criterion | Target | Measurement | Status |
|-----------|--------|-------------|---------|
| **Local Processing** | 100% local | No external network calls | ✅ To verify |
| **Chunk Efficiency** | ≤ 10 chunks per doc | Average chunk count | ✅ To verify |
| **Embedding Quality** | 1024-dim vectors | Vector dimension check | ✅ To verify |
| **Processing Speed** | > 100 docs/hour | Throughput measurement | ✅ To verify |
| **Memory Usage** | < 8GB peak | Memory monitoring | ✅ To verify |
| **Storage Reduction** | 90% less chunks | Before/after comparison | ✅ To verify |

### 9.2 Business Success Criteria

| Criterion | Target | Measurement | Impact |
|-----------|--------|-------------|---------|
| **Security Compliance** | Government ready | Compliance audit | High |
| **Cost Reduction** | Zero API costs | Monthly billing | Medium |
| **Search Quality** | Better relevance | User testing | High |
| **Deployment Flexibility** | On-premise ready | Environment test | High |

### 9.3 Validation Plan

```python
# validation/success_criteria_validator.py
class SuccessCriteriaValidator:
    """Validate all success criteria"""
    
    def __init__(self):
        self.results = {}
        self.performance_tracker = PerformanceTracker()
        self.compliance_checker = ComplianceChecker()
    
    def validate_technical_criteria(self, embedding_processor):
        """Validate technical success criteria"""
        
        # Test local processing
        with NetworkMonitor() as monitor:
            test_docs = self._load_test_documents()
            embedding_processor.process_and_store_embeddings(test_docs)
            
            self.results['local_processing'] = len(monitor.external_calls) == 0
        
        # Test chunk efficiency
        performance_report = self.performance_tracker.generate_report()
        self.results['chunk_efficiency'] = performance_report['avg_chunks_per_doc'] <= 10
        
        # Test embedding quality
        sample_embedding = embedding_processor.embed_generator.get_text_embedding("test")
        self.results['embedding_quality'] = len(sample_embedding) == 1024
        
        # Test processing speed
        self.results['processing_speed'] = performance_report['documents_per_hour'] >= 100
        
        # Test memory usage
        self.results['memory_usage'] = performance_report['peak_memory_mb'] <= 8000
    
    def validate_business_criteria(self, db_utils):
        """Validate business success criteria"""
        
        # Test security compliance
        compliance_report = self.compliance_checker.generate_compliance_report()
        self.results['security_compliance'] = compliance_report['compliant']
        
        # Test cost reduction (no API costs)
        self.results['cost_reduction'] = not self._check_api_usage()
        
        # Test storage reduction
        old_chunks = db_utils.count_old_chunks()
        new_chunks = db_utils.count_new_chunks()
        reduction_ratio = (old_chunks - new_chunks) / old_chunks if old_chunks > 0 else 0
        self.results['storage_reduction'] = reduction_ratio >= 0.9
    
    def generate_validation_report(self) -> Dict:
        """Generate final validation report"""
        passed_criteria = sum(1 for result in self.results.values() if result)
        total_criteria = len(self.results)
        
        return {
            'overall_success': all(self.results.values()),
            'criteria_passed': passed_criteria,
            'total_criteria': total_criteria,
            'success_rate': passed_criteria / total_criteria,
            'detailed_results': self.results,
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []
        
        if not self.results.get('local_processing', True):
            recommendations.append("Fix external API calls - ensure complete local processing")
        
        if not self.results.get('chunk_efficiency', True):
            recommendations.append("Optimize chunking strategy - too many chunks per document")
        
        if not self.results.get('security_compliance', True):
            recommendations.append("Address security violations before deployment")
        
        if not self.results.get('processing_speed', True):
            recommendations.append("Optimize processing pipeline for better performance")
        
        if len(recommendations) == 0:
            recommendations.append("✅ All criteria met - ready for production deployment")
        
        return recommendations
```

---

## 10. Rollout Plan

### Phase 1: Development Setup (Week 1)
**Objectives**: Establish BGE-M3 environment and basic functionality

**Tasks**:
- [ ] Install BGE-M3 model locally
- [ ] Update `postgres_embedding.py` with BGE-M3 implementation
- [ ] Update database schema for section metadata
- [ ] Create migration scripts
- [ ] Run initial unit tests

**Success Criteria**:
- BGE-M3 model loads successfully
- 1024-dimensional embeddings generated
- Database schema updated without errors

### Phase 2: Section Integration (Week 2)
**Objectives**: Integrate section-based chunking

**Tasks**:
- [ ] Import `standalone_section_parser.py` to PostgreSQL module
- [ ] Update document processing pipeline
- [ ] Test section-based chunking on sample documents
- [ ] Validate metadata preservation
- [ ] Performance benchmarking

**Success Criteria**:
- Documents produce 6-10 chunks (not 169)
- All section types properly identified
- Complete metadata preservation verified

### Phase 3: Testing & Validation (Week 3)
**Objectives**: Comprehensive testing and validation

**Tasks**:
- [ ] Run full test suite
- [ ] Process 100 sample documents
- [ ] Validate search quality improvements
- [ ] Security compliance verification
- [ ] Performance optimization

**Success Criteria**:
- All tests pass
- Performance meets benchmarks
- Security requirements validated

### Phase 4: Production Deployment (Week 4)
**Objectives**: Deploy to production environment

**Tasks**:
- [ ] Backup existing data
- [ ] Execute database migration
- [ ] Process production documents
- [ ] Monitor performance
- [ ] Conduct final validation

**Success Criteria**:
- Zero data loss during migration
- All success criteria met
- System ready for use

---

## 11. Troubleshooting Guide

### 11.1 Common Issues & Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Model Download Fails** | Connection errors during initialization | Check internet connection, use cached model |
| **Out of Memory** | Process crashes with memory error | Reduce batch_size, use CPU instead of GPU |
| **Slow Processing** | Documents taking > 10s each | Enable GPU, use FP16, increase batch_size |
| **Wrong Chunk Count** | Still getting 100+ chunks per doc | Verify section parser is loaded correctly |
| **Database Errors** | Connection or constraint failures | Check database schema, verify credentials |
| **Embedding Dimension Mismatch** | Constraint violation on insert | Verify BGE-M3 is producing 1024-dim vectors |

### 11.2 Debugging Commands

```bash
# Check BGE-M3 model status
python -c "from FlagEmbedding import FlagModel; print('BGE-M3 available')"

# Verify database schema
psql -U postgres -d iland_embeddings -c "\d iland_chunks"

# Test embedding generation
python -c "
from src_iland.docs_embedding_postgres.postgres_embedding import PostgresEmbeddingGenerator
gen = PostgresEmbeddingGenerator()
emb = gen.get_text_embedding('test')
print(f'Embedding dimension: {len(emb)}')
"

# Check chunk distribution
psql -U postgres -d iland_embeddings -c "
SELECT 
  deed_id, 
  COUNT(*) as chunk_count,
  STRING_AGG(DISTINCT section_type, ', ') as sections
FROM iland_chunks 
GROUP BY deed_id 
ORDER BY chunk_count DESC 
LIMIT 10;
"

# Monitor memory usage
python -c "
import psutil
import time
print(f'Memory usage: {psutil.virtual_memory().percent}%')
"
```

### 11.3 Performance Optimization

```yaml
# config/optimization.yaml
performance:
  # For high-performance deployments
  gpu_optimized:
    device: "cuda"
    use_fp16: true
    batch_size: 64
    
  # For memory-constrained environments  
  memory_optimized:
    device: "cpu"
    use_fp16: false
    batch_size: 8
    
  # For balanced performance
  balanced:
    device: "auto"
    use_fp16: true
    batch_size: 32
```

---

## 12. Appendix

### 12.1 Configuration Template

```yaml
# config/embedding_config.yaml
embedding:
  model:
    name: "BAAI/bge-m3"
    cache_dir: "./cache/bge_models"
    device: "auto"  # auto, cuda, cpu
    use_fp16: true
    normalize_embeddings: true
    
  chunking:
    strategy: "section-based"
    chunk_size: 512
    chunk_overlap: 50
    min_section_size: 50
    preserve_formatting: true
    language: "th"
    
  processing:
    batch_size: 32
    max_length: 8192
    max_workers: 4
    
  database:
    host: "localhost"
    port: 5432
    database: "iland_embeddings"
    user: "postgres"
    pool_size: 10
    
  security:
    allow_external_apis: false
    log_all_operations: true
    encrypt_at_rest: true
    audit_logging: true
    
  monitoring:
    enable_performance_tracking: true
    log_level: "INFO"
    metrics_collection: true
```

### 12.2 Dependencies

```txt
# requirements_bge_m3.txt
# Core embedding dependencies
FlagEmbedding>=1.2.0
torch>=2.0.0
transformers>=4.36.0
sentence-transformers>=2.3.0

# Database dependencies
psycopg2-binary>=2.9.0
sqlalchemy>=2.0.0

# Processing dependencies
numpy>=1.24.0
pandas>=2.0.0

# Monitoring dependencies
psutil>=5.9.0
```

---

**End of Document**

This enhanced PRD provides a comprehensive plan for updating the PostgreSQL embedding module to use BGE-M3 with section-based chunking, ensuring security compliance and improved performance for Thai land deed document processing.