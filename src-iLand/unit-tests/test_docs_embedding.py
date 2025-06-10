"""
Comprehensive unit tests for the docs_embedding module.

Tests cover batch embedding pipeline, document loading, metadata extraction,
embedding processing, and file storage functionality.
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import the modules to test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docs_embedding.document_loader import iLandDocumentLoader
from docs_embedding.metadata_extractor import iLandMetadataExtractor
from docs_embedding.file_storage import EmbeddingStorage
from docs_embedding.embedding_config import EmbeddingConfiguration, get_embedding_config
from docs_embedding.embedding_processor import EmbeddingProcessor

# Mock the imports that require external dependencies
try:
    from docs_embedding.batch_embedding import iLandBatchEmbeddingPipeline, CONFIG
    BATCH_EMBEDDING_AVAILABLE = True
except ImportError:
    BATCH_EMBEDDING_AVAILABLE = False
    
try:
    from docs_embedding.bge_embedding_processor import BGEEmbeddingProcessor
    BGE_AVAILABLE = True
except ImportError:
    BGE_AVAILABLE = False


class TestiLandDocumentLoader(unittest.TestCase):
    """Test iLandDocumentLoader functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = iLandDocumentLoader()
        
        # Create test markdown files
        self.test_files = []
        for i in range(3):
            file_path = Path(self.temp_dir) / f"test_doc_{i}.md"
            content = f"""# Test Document {i}
## ข้อมูลโฉนด (Deed Information)
- ประเภทโฉนด: โฉนดที่ดิน
- เลขที่โฉนด: {1000 + i}

## ที่ตั้ง (Location)  
- จังหวัด: กรุงเทพมหานคร
- อำเภอ: บางกะปิ
- ตำบล: หัวหมาก

## ขนาดพื้นที่ (Area)
- เนื้อที่: {i+1} ไร่ 2 งาน 50 ตร.ว.
"""
            file_path.write_text(content, encoding='utf-8')
            self.test_files.append(file_path)
    
    def test_load_documents_from_directory(self):
        """Test loading documents from directory."""
        documents = self.loader.load_documents_from_directory(self.temp_dir)
        
        self.assertEqual(len(documents), 3)
        
        # Check document content
        for i, doc in enumerate(documents):
            self.assertIn(f"Test Document {i}", doc.text)
            self.assertEqual(doc.metadata['file_name'], f"test_doc_{i}.md")
            self.assertIn('file_path', doc.metadata)
            self.assertIn('file_size', doc.metadata)
            self.assertIn('created_at', doc.metadata)
    
    def test_load_documents_with_filter(self):
        """Test loading documents with file pattern filter."""
        # Create non-markdown file
        txt_file = Path(self.temp_dir) / "ignore_me.txt"
        txt_file.write_text("This should be ignored")
        
        documents = self.loader.load_documents_from_directory(
            self.temp_dir, 
            file_pattern="*.md"
        )
        
        # Should only load markdown files
        self.assertEqual(len(documents), 3)
        for doc in documents:
            self.assertTrue(doc.metadata['file_name'].endswith('.md'))
    
    def test_load_single_document(self):
        """Test loading a single document."""
        test_file = self.test_files[0]
        doc = self.loader.load_single_document(test_file)
        
        self.assertIn("Test Document 0", doc.text)
        self.assertEqual(doc.metadata['file_name'], "test_doc_0.md")
        self.assertIn('file_size', doc.metadata)
    
    def test_extract_title_from_content(self):
        """Test title extraction from markdown content."""
        content = "# Main Title\nSome content\n## Subtitle"
        title = self.loader._extract_title_from_content(content)
        self.assertEqual(title, "Main Title")
        
        # Test without title
        content_no_title = "Just some content without title"
        title = self.loader._extract_title_from_content(content_no_title)
        self.assertEqual(title, "Untitled Document")
    
    def test_get_documents_in_batches(self):
        """Test getting documents in batches."""
        batches = self.loader.get_documents_in_batches(
            self.temp_dir, 
            batch_size=2
        )
        
        self.assertEqual(len(batches), 2)  # 3 files / batch_size=2 = 2 batches
        self.assertEqual(len(batches[0]), 2)  # First batch: 2 documents
        self.assertEqual(len(batches[1]), 1)  # Second batch: 1 document


class TestiLandMetadataExtractor(unittest.TestCase):
    """Test iLandMetadataExtractor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = iLandMetadataExtractor()
        
        # Sample Thai land deed content
        self.sample_content = """# บันทึกข้อมูลโฉนดที่ดิน
        
## ข้อมูลโฉนด (Deed Information)
- ประเภทโฉนด: โฉนดที่ดิน
- เลขที่โฉนด: 12345
- เล่มที่: 100
- หน้าที่: 25

## ที่ตั้ง (Location)
- จังหวัด: กรุงเทพมหานคร
- อำเภอ: บางกะปิ
- ตำบล: หัวหมาก
- พิกัด: 13.7563, 100.5014

## ขนาดพื้นที่ (Area)
- เนื้อที่: 2 ไร่ 1 งาน 50 ตร.ว.
- พื้นที่รวม (ตร.ม.): 3400

## รายละเอียดที่ดิน (Land Details)  
- ประเภทการใช้ที่ดิน: ที่อยู่อาศัย
- หมวดหมู่หลัก: ที่ดินเพื่ออยู่อาศัย
"""
    
    def test_extract_thai_provinces(self):
        """Test extraction of Thai provinces."""
        provinces = self.extractor.extract_thai_provinces(self.sample_content)
        self.assertIn('กรุงเทพมหานคร', provinces)
    
    def test_extract_deed_information(self):
        """Test extraction of deed information."""
        deed_info = self.extractor.extract_deed_information(self.sample_content)
        
        self.assertEqual(deed_info['deed_type'], 'โฉนดที่ดิน')
        self.assertEqual(deed_info['deed_serial_no'], '12345')
        self.assertEqual(deed_info['deed_book_no'], '100')
        self.assertEqual(deed_info['deed_page_no'], '25')
    
    def test_extract_location_information(self):
        """Test extraction of location information."""
        location_info = self.extractor.extract_location_information(self.sample_content)
        
        self.assertEqual(location_info['province'], 'กรุงเทพมหานคร')
        self.assertEqual(location_info['district'], 'บางกะปิ')
        self.assertEqual(location_info['subdistrict'], 'หัวหมาก')
        self.assertIn('location_hierarchy', location_info)
    
    def test_extract_area_measurements(self):
        """Test extraction of area measurements."""
        area_info = self.extractor.extract_area_measurements(self.sample_content)
        
        self.assertIn('area_rai', area_info)
        self.assertIn('area_ngan', area_info)
        self.assertIn('area_wa', area_info)
        self.assertIn('area_formatted', area_info)
        self.assertEqual(area_info['area_formatted'], '2 ไร่ 1 งาน 50 ตร.ว.')
    
    def test_extract_coordinates(self):
        """Test extraction of coordinates."""
        coords = self.extractor.extract_coordinates(self.sample_content)
        
        self.assertIn('latitude', coords)
        self.assertIn('longitude', coords)
        self.assertEqual(coords['latitude'], 13.7563)
        self.assertEqual(coords['longitude'], 100.5014)
    
    def test_extract_land_use_types(self):
        """Test extraction of land use types."""
        land_use = self.extractor.extract_land_use_types(self.sample_content)
        
        self.assertIn('ที่อยู่อาศัย', land_use)
    
    def test_extract_comprehensive_metadata(self):
        """Test comprehensive metadata extraction."""
        metadata = self.extractor.extract_comprehensive_metadata(self.sample_content)
        
        # Check all sections are present
        self.assertIn('deed_info', metadata)
        self.assertIn('location_info', metadata)
        self.assertIn('area_info', metadata)
        self.assertIn('land_use_types', metadata)
        self.assertIn('coordinates', metadata)
        
        # Check metadata structure
        self.assertEqual(metadata['deed_info']['deed_type'], 'โฉนดที่ดิน')
        self.assertEqual(metadata['location_info']['province'], 'กรุงเทพมหานคร')
        
        # Check computed fields
        self.assertIn('search_keywords', metadata)
        self.assertIn('กรุงเทพมหานคร', metadata['search_keywords'])
    
    def test_thai_text_detection(self):
        """Test Thai text detection."""
        self.assertTrue(self.extractor._contains_thai_text('กรุงเทพมหานคร'))
        self.assertTrue(self.extractor._contains_thai_text('Mixed Thai ข้อความ'))
        self.assertFalse(self.extractor._contains_thai_text('English only'))
        self.assertFalse(self.extractor._contains_thai_text('123'))


class TestEmbeddingStorage(unittest.TestCase):
    """Test EmbeddingStorage functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = EmbeddingStorage(self.temp_dir)
        
        # Create mock embedding data
        self.mock_embeddings = {
            'chunk_embeddings': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            'metadata': [
                {'id': 'chunk_1', 'text': 'Sample text 1'},
                {'id': 'chunk_2', 'text': 'Sample text 2'}
            ],
            'embedding_model': 'text-embedding-3-small',
            'created_at': datetime.now().isoformat(),
            'batch_info': {
                'batch_id': 'test_batch_001',
                'total_chunks': 2,
                'source_files': ['test1.md', 'test2.md']
            }
        }
    
    def test_save_and_load_embeddings(self):
        """Test saving and loading embeddings."""
        # Save embeddings
        saved_path = self.storage.save_embeddings(
            self.mock_embeddings, 
            batch_id='test_batch_001'
        )
        
        self.assertTrue(Path(saved_path).exists())
        
        # Load embeddings
        loaded_embeddings = self.storage.load_embeddings('test_batch_001')
        
        self.assertEqual(len(loaded_embeddings['chunk_embeddings']), 2)
        self.assertEqual(loaded_embeddings['embedding_model'], 'text-embedding-3-small')
        self.assertEqual(loaded_embeddings['batch_info']['batch_id'], 'test_batch_001')
    
    def test_list_available_batches(self):
        """Test listing available embedding batches."""
        # Save multiple batches
        batch_ids = ['batch_001', 'batch_002', 'batch_003']
        for batch_id in batch_ids:
            self.storage.save_embeddings(self.mock_embeddings, batch_id)
        
        # List batches
        available_batches = self.storage.list_available_batches()
        
        self.assertEqual(len(available_batches), 3)
        for batch_info in available_batches:
            self.assertIn(batch_info['batch_id'], batch_ids)
            self.assertIn('created_at', batch_info)
            self.assertIn('file_size', batch_info)
    
    def test_get_batch_summary(self):
        """Test getting batch summary."""
        # Save embeddings
        self.storage.save_embeddings(self.mock_embeddings, 'summary_test')
        
        # Get summary
        summary = self.storage.get_batch_summary('summary_test')
        
        self.assertEqual(summary['batch_id'], 'summary_test')
        self.assertEqual(summary['total_chunks'], 2)
        self.assertEqual(summary['embedding_model'], 'text-embedding-3-small')
        self.assertIn('file_size', summary)
    
    def test_delete_batch(self):
        """Test deleting embedding batch."""
        # Save and then delete
        self.storage.save_embeddings(self.mock_embeddings, 'delete_test')
        
        # Verify exists
        self.assertTrue(self.storage.batch_exists('delete_test'))
        
        # Delete
        self.storage.delete_batch('delete_test')
        
        # Verify deleted
        self.assertFalse(self.storage.batch_exists('delete_test'))
    
    def test_create_backup(self):
        """Test creating backup of embeddings."""
        # Save embeddings
        self.storage.save_embeddings(self.mock_embeddings, 'backup_test')
        
        # Create backup
        backup_path = self.storage.create_backup('backup_test')
        
        self.assertTrue(Path(backup_path).exists())
        self.assertIn('backup', str(backup_path))


class TestEmbeddingConfiguration(unittest.TestCase):
    """Test EmbeddingConfiguration functionality."""
    
    def test_default_configuration(self):
        """Test default configuration creation."""
        config = EmbeddingConfiguration()
        
        self.assertEqual(config.embedding_model, 'text-embedding-3-small')
        self.assertEqual(config.llm_model, 'gpt-4o-mini')
        self.assertEqual(config.chunk_size, 512)
        self.assertEqual(config.chunk_overlap, 50)
        self.assertTrue(config.enable_sentence_window)
    
    def test_custom_configuration(self):
        """Test custom configuration."""
        config = EmbeddingConfiguration(
            embedding_model='text-embedding-3-large',
            chunk_size=1024,
            enable_hierarchical_retrieval=False
        )
        
        self.assertEqual(config.embedding_model, 'text-embedding-3-large')
        self.assertEqual(config.chunk_size, 1024)
        self.assertFalse(config.enable_hierarchical_retrieval)
    
    def test_validate_configuration(self):
        """Test configuration validation."""
        # Valid configuration
        config = EmbeddingConfiguration()
        self.assertTrue(config.validate())
        
        # Invalid configuration
        config.chunk_size = -1
        self.assertFalse(config.validate())
    
    def test_to_dict_conversion(self):
        """Test converting configuration to dictionary."""
        config = EmbeddingConfiguration()
        config_dict = config.to_dict()
        
        self.assertIn('embedding_model', config_dict)
        self.assertIn('chunk_size', config_dict)
        self.assertEqual(config_dict['embedding_model'], 'text-embedding-3-small')
    
    @patch.dict(os.environ, {'EMBEDDING_MODEL': 'custom-model', 'CHUNK_SIZE': '256'})
    def test_get_config_from_environment(self):
        """Test loading configuration from environment variables."""
        config = get_embedding_config()
        
        self.assertEqual(config.embedding_model, 'custom-model')
        self.assertEqual(config.chunk_size, 256)


class TestEmbeddingProcessor(unittest.TestCase):
    """Test EmbeddingProcessor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock configuration
        self.config = EmbeddingConfiguration(
            embedding_model='text-embedding-3-small',
            output_dir=self.temp_dir
        )
        
        # Mock documents
        self.mock_documents = [
            Mock(text="Test document 1", metadata={'id': 'doc1'}),
            Mock(text="Test document 2", metadata={'id': 'doc2'})
        ]
    
    @patch('docs_embedding.embedding_processor.OpenAIEmbedding')
    @patch('docs_embedding.embedding_processor.VectorStoreIndex')
    def test_process_documents(self, mock_index, mock_embedding):
        """Test document processing."""
        # Setup mocks
        mock_embedding_instance = Mock()
        mock_embedding.return_value = mock_embedding_instance
        
        mock_index_instance = Mock()
        mock_index.from_documents.return_value = mock_index_instance
        
        processor = EmbeddingProcessor(self.config)
        result = processor.process_documents(self.mock_documents, 'test_batch')
        
        # Verify calls
        mock_embedding.assert_called_once()
        mock_index.from_documents.assert_called_once()
        
        self.assertIn('index', result)
        self.assertIn('batch_id', result)
    
    @patch('docs_embedding.embedding_processor.SentenceSplitter')
    def test_chunk_documents(self, mock_splitter):
        """Test document chunking."""
        mock_splitter_instance = Mock()
        mock_splitter.return_value = mock_splitter_instance
        mock_splitter_instance.get_nodes_from_documents.return_value = [
            Mock(text="Chunk 1", metadata={}),
            Mock(text="Chunk 2", metadata={})
        ]
        
        processor = EmbeddingProcessor(self.config)
        chunks = processor._chunk_documents(self.mock_documents)
        
        self.assertEqual(len(chunks), 2)
        mock_splitter.assert_called_once_with(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
    
    def test_extract_thai_keywords(self):
        """Test Thai keyword extraction."""
        processor = EmbeddingProcessor(self.config)
        
        text = "กรุงเทพมหานคร บางกะปิ โฉนดที่ดิน ที่อยู่อาศัย"
        keywords = processor._extract_thai_keywords(text)
        
        self.assertIn('กรุงเทพมหานคร', keywords)
        self.assertIn('บางกะปิ', keywords)
        self.assertIn('โฉนดที่ดิน', keywords)


@unittest.skipUnless(BATCH_EMBEDDING_AVAILABLE, "Batch embedding module not available")
class TestBatchEmbeddingPipeline(unittest.TestCase):
    """Test iLandBatchEmbeddingPipeline functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test markdown files
        for i in range(3):
            file_path = Path(self.temp_dir) / f"deed_{i}.md"
            content = f"""# โฉนดที่ดิน {i}
## ข้อมูลโฉนด
- เลขที่: {1000 + i}
- จังหวัด: กรุงเทพมหานคร
"""
            file_path.write_text(content, encoding='utf-8')
    
    @patch.dict('docs_embedding.batch_embedding.CONFIG', {'data_dir': None})
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        with patch.dict('docs_embedding.batch_embedding.CONFIG', 
                       {'data_dir': Path(self.temp_dir)}):
            pipeline = iLandBatchEmbeddingPipeline()
            self.assertIsInstance(pipeline.config, dict)
            self.assertIn('batch_size', pipeline.config)
    
    @patch.dict('docs_embedding.batch_embedding.CONFIG', {'data_dir': None})
    def test_get_markdown_files_in_batches(self):
        """Test getting markdown files in batches."""
        with patch.dict('docs_embedding.batch_embedding.CONFIG', 
                       {'data_dir': Path(self.temp_dir)}):
            pipeline = iLandBatchEmbeddingPipeline()
            batches = pipeline.get_markdown_files_in_batches()
            
            self.assertGreater(len(batches), 0)
            total_files = sum(len(batch) for batch in batches)
            self.assertEqual(total_files, 3)


@unittest.skipUnless(BGE_AVAILABLE, "BGE embedding processor not available")
class TestBGEEmbeddingProcessor(unittest.TestCase):
    """Test BGEEmbeddingProcessor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock documents
        self.mock_documents = [
            Mock(text="Thai land deed document", metadata={'id': 'doc1'}),
            Mock(text="Another deed document", metadata={'id': 'doc2'})
        ]
    
    @patch('docs_embedding.bge_embedding_processor.BGEM3FlagModel')
    def test_bge_processor_initialization(self, mock_bge_model):
        """Test BGE processor initialization."""
        mock_model_instance = Mock()
        mock_bge_model.return_value = mock_model_instance
        
        processor = BGEEmbeddingProcessor()
        
        mock_bge_model.assert_called_once_with('BAAI/bge-m3', use_fp16=True)
        self.assertEqual(processor.model, mock_model_instance)
    
    @patch('docs_embedding.bge_embedding_processor.BGEM3FlagModel')
    def test_process_documents_with_bge(self, mock_bge_model):
        """Test document processing with BGE model."""
        mock_model_instance = Mock()
        mock_model_instance.encode.return_value = {
            'dense_vecs': [[0.1, 0.2], [0.3, 0.4]],
            'sparse_vecs': [{}, {}]
        }
        mock_bge_model.return_value = mock_model_instance
        
        processor = BGEEmbeddingProcessor()
        result = processor.process_documents(self.mock_documents, 'bge_test_batch')
        
        self.assertIn('embeddings', result)
        self.assertIn('batch_id', result)
        mock_model_instance.encode.assert_called()


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete docs_embedding pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample documents
        self.sample_docs = [
            (Path(self.temp_dir) / "deed_001.md", """# โฉนดที่ดิน 001
## ข้อมูลโฉนด
- ประเภทโฉนด: โฉนดที่ดิน
- เลขที่โฉนด: 12345

## ที่ตั้ง
- จังหวัด: กรุงเทพมหานคร
- อำเภอ: บางกะปิ
- พิกัด: 13.7563, 100.5014

## ขนาดพื้นที่
- เนื้อที่: 2 ไร่ 1 งาน 50 ตร.ว.
"""),
            (Path(self.temp_dir) / "deed_002.md", """# โฉนดที่ดิน 002
## ข้อมูลโฉนด
- ประเภทโฉนด: โฉนดที่ดิน
- เลขที่โฉนด: 67890

## ที่ตั้ง
- จังหวัด: เชียงใหม่
- อำเภอ: เมือง
- พิกัด: 18.7883, 98.9817

## ขนาดพื้นที่
- เนื้อที่: 5 ไร่
""")
        ]
        
        # Write test files
        for file_path, content in self.sample_docs:
            file_path.write_text(content, encoding='utf-8')
    
    def test_complete_embedding_workflow(self):
        """Test the complete embedding workflow."""
        # 1. Load documents
        loader = iLandDocumentLoader()
        documents = loader.load_documents_from_directory(self.temp_dir)
        
        self.assertEqual(len(documents), 2)
        
        # 2. Extract metadata
        extractor = iLandMetadataExtractor()
        
        for doc in documents:
            metadata = extractor.extract_comprehensive_metadata(doc.text)
            
            # Verify metadata extraction
            self.assertIn('deed_info', metadata)
            self.assertIn('location_info', metadata)
            
            # Update document metadata
            doc.metadata.update({
                'extracted_metadata': metadata,
                'thai_provinces': extractor.extract_thai_provinces(doc.text),
                'search_keywords': metadata.get('search_keywords', [])
            })
        
        # 3. Test storage (without actual embedding)
        storage = EmbeddingStorage(self.temp_dir)
        
        # Mock embedding data
        mock_embeddings = {
            'chunk_embeddings': [[0.1] * 1536] * len(documents),
            'metadata': [doc.metadata for doc in documents],
            'embedding_model': 'text-embedding-3-small',
            'created_at': datetime.now().isoformat(),
            'batch_info': {
                'batch_id': 'integration_test',
                'total_chunks': len(documents),
                'source_files': [doc.metadata['file_name'] for doc in documents]
            }
        }
        
        # Save and verify
        saved_path = storage.save_embeddings(mock_embeddings, 'integration_test')
        self.assertTrue(Path(saved_path).exists())
        
        # Load and verify
        loaded_embeddings = storage.load_embeddings('integration_test')
        self.assertEqual(len(loaded_embeddings['chunk_embeddings']), 2)
        self.assertEqual(len(loaded_embeddings['metadata']), 2)
        
        # 4. Verify metadata enrichment
        doc1_metadata = loaded_embeddings['metadata'][0]['extracted_metadata']
        self.assertIn('deed_info', doc1_metadata)
        self.assertEqual(doc1_metadata['deed_info']['deed_serial_no'], '12345')
        self.assertEqual(doc1_metadata['location_info']['province'], 'กรุงเทพมหานคร')
    
    def test_batch_processing_workflow(self):
        """Test batch processing workflow."""
        loader = iLandDocumentLoader()
        
        # Test batch processing
        batches = loader.get_documents_in_batches(self.temp_dir, batch_size=1)
        self.assertEqual(len(batches), 2)
        
        # Process each batch
        extractor = iLandMetadataExtractor()
        storage = EmbeddingStorage(self.temp_dir)
        
        for i, batch in enumerate(batches):
            batch_id = f"batch_{i+1:03d}"
            
            # Extract metadata for batch
            for doc in batch:
                metadata = extractor.extract_comprehensive_metadata(doc.text)
                doc.metadata['extracted_metadata'] = metadata
            
            # Mock save batch
            mock_batch_embeddings = {
                'chunk_embeddings': [[0.1] * 1536] * len(batch),
                'metadata': [doc.metadata for doc in batch],
                'embedding_model': 'text-embedding-3-small',
                'created_at': datetime.now().isoformat(),
                'batch_info': {
                    'batch_id': batch_id,
                    'total_chunks': len(batch),
                    'source_files': [doc.metadata['file_name'] for doc in batch]
                }
            }
            
            storage.save_embeddings(mock_batch_embeddings, batch_id)
        
        # Verify all batches saved
        available_batches = storage.list_available_batches()
        self.assertEqual(len(available_batches), 2)
        
        # Verify batch summaries
        for batch_info in available_batches:
            summary = storage.get_batch_summary(batch_info['batch_id'])
            self.assertEqual(summary['total_chunks'], 1)
            self.assertIn('file_size', summary)


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestiLandDocumentLoader,
        TestiLandMetadataExtractor,
        TestEmbeddingStorage,
        TestEmbeddingConfiguration,
        TestEmbeddingProcessor,
        TestIntegration
    ]
    
    # Add conditional test classes
    if BATCH_EMBEDDING_AVAILABLE:
        test_classes.append(TestBatchEmbeddingPipeline)
    
    if BGE_AVAILABLE:
        test_classes.append(TestBGEEmbeddingProcessor)
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY - Docs Embedding Module")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}")