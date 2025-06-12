"""
Comprehensive unit tests for the data_processing module.

Tests cover DocumentProcessor, models, and all data processing functionality
including Thai text handling, metadata extraction, and document generation.
"""

import unittest
import pandas as pd
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

# Import the modules to test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.models import FieldMapping, DatasetConfig, SimpleDocument
from data_processing.document_processor import DocumentProcessor
from data_processing.file_output import FileOutputManager
from data_processing.csv_analyzer import CSVAnalyzer
from data_processing.config_manager import DatasetConfigManager


class TestFieldMapping(unittest.TestCase):
    """Test FieldMapping dataclass functionality."""
    
    def test_field_mapping_creation(self):
        """Test basic field mapping creation."""
        mapping = FieldMapping(
            csv_column='province',
            metadata_key='province',
            field_type='location'
        )
        
        self.assertEqual(mapping.csv_column, 'province')
        self.assertEqual(mapping.metadata_key, 'province')
        self.assertEqual(mapping.field_type, 'location')
        self.assertEqual(mapping.data_type, 'string')
        self.assertEqual(mapping.aliases, [])
        
    def test_field_mapping_with_aliases(self):
        """Test field mapping with aliases."""
        mapping = FieldMapping(
            csv_column='prov',
            metadata_key='province',
            field_type='location',
            aliases=['province', 'จังหวัด']
        )
        
        self.assertEqual(mapping.aliases, ['province', 'จังหวัด'])


class TestDatasetConfig(unittest.TestCase):
    """Test DatasetConfig functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mappings = [
            FieldMapping('province', 'province', 'location'),
            FieldMapping('district', 'district', 'location'),
            FieldMapping('deed_type', 'deed_type', 'deed_info')
        ]
    
    def test_dataset_config_creation(self):
        """Test dataset config creation."""
        config = DatasetConfig(
            name='test_config',
            description='Test configuration',
            field_mappings=self.mappings
        )
        
        self.assertEqual(config.name, 'test_config')
        self.assertEqual(len(config.field_mappings), 3)
        self.assertIn('deed_type', config.embedding_fields)
        self.assertIn('province', config.embedding_fields)
    
    def test_custom_embedding_fields(self):
        """Test custom embedding fields."""
        config = DatasetConfig(
            name='test_config',
            description='Test configuration',
            field_mappings=self.mappings,
            embedding_fields=['province', 'district']
        )
        
        self.assertEqual(config.embedding_fields, ['province', 'district'])


class TestSimpleDocument(unittest.TestCase):
    """Test SimpleDocument functionality."""
    
    def test_document_creation(self):
        """Test document creation with metadata."""
        metadata = {
            'province': 'กรุงเทพมหานคร',
            'district': 'บางกะปิ',
            'deed_type': 'โฉนดที่ดิน'
        }
        
        doc = SimpleDocument(
            text='# Test Document\nThis is a test document.',
            metadata=metadata
        )
        
        self.assertEqual(doc.text, '# Test Document\nThis is a test document.')
        self.assertEqual(doc.metadata['province'], 'กรุงเทพมหานคร')
        self.assertIsNotNone(doc.id)
    
    def test_document_id_generation(self):
        """Test document ID generation."""
        metadata = {'test': 'value'}
        doc = SimpleDocument('Test text', metadata)
        
        self.assertIsInstance(doc.id, str)
        self.assertEqual(len(doc.id), 16)  # MD5 hash truncated to 16 chars
    
    def test_document_with_custom_id(self):
        """Test document with custom ID in metadata."""
        metadata = {'doc_id': 'custom_id_123'}
        doc = SimpleDocument('Test text', metadata)
        
        self.assertEqual(doc.id, 'custom_id_123')


class TestDocumentProcessor(unittest.TestCase):
    """Test DocumentProcessor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mappings = [
            FieldMapping('province', 'province', 'location'),
            FieldMapping('district', 'district', 'location'),
            FieldMapping('subdistrict', 'subdistrict', 'location'),
            FieldMapping('deed_type', 'deed_type', 'deed_info'),
            FieldMapping('area_rai', 'area_rai', 'area_measurements', 'numeric'),
            FieldMapping('area_ngan', 'area_ngan', 'area_measurements', 'numeric'),
            FieldMapping('area_wa', 'area_wa', 'area_measurements', 'numeric'),
            FieldMapping('land_geom_point', 'land_geom_point', 'geolocation'),
            FieldMapping('land_use_type', 'land_use_type', 'land_details'),
            FieldMapping('owner_date', 'owner_date', 'dates', 'date')
        ]
        
        self.config = DatasetConfig(
            name='test_land_deeds',
            description='Test Thai land deed configuration',
            field_mappings=self.mappings
        )
        
        self.processor = DocumentProcessor(self.config)
    
    def test_clean_value_basic(self):
        """Test basic value cleaning."""
        # Valid values
        self.assertEqual(self.processor.clean_value('Test'), 'Test')
        self.assertEqual(self.processor.clean_value('  Test  '), 'Test')
        
        # Invalid/placeholder values
        self.assertIsNone(self.processor.clean_value(''))
        self.assertIsNone(self.processor.clean_value('nan'))
        self.assertIsNone(self.processor.clean_value('ไม่ระบุ'))
        self.assertIsNone(self.processor.clean_value('-'))
        self.assertIsNone(self.processor.clean_value(pd.NA))
    
    def test_clean_value_thai_punctuation(self):
        """Test Thai punctuation normalization."""
        # Thai punctuation should be normalized
        self.assertEqual(self.processor.clean_value('"ทดสอบ"'), '"ทดสอบ"')
        self.assertEqual(self.processor.clean_value('ทดสอบ，ต่อไป'), 'ทดสอบ,ต่อไป')
        self.assertEqual(self.processor.clean_value('ทดสอบ　ห่างไกล'), 'ทดสอบ ห่างไกล')
    
    def test_format_area_for_display(self):
        """Test Thai area formatting."""
        # Complete area
        result = self.processor.format_area_for_display(2, 1, 50)
        self.assertEqual(result, '2 ไร่ 1 งาน 50 ตร.ว.')
        
        # Partial area
        result = self.processor.format_area_for_display(5, None, None)
        self.assertEqual(result, '5 ไร่')
        
        # No area
        result = self.processor.format_area_for_display(None, None, None)
        self.assertEqual(result, 'ไม่ระบุ')
    
    def test_parse_geom_point_valid(self):
        """Test geolocation parsing with valid coordinates."""
        # Standard POINT format
        result = self.processor.parse_geom_point('POINT(100.5014 13.7563)')
        self.assertIn('longitude', result)
        self.assertIn('latitude', result)
        self.assertEqual(result['longitude'], 100.5014)
        self.assertEqual(result['latitude'], 13.7563)
        self.assertIn('google_maps_url', result)
        
        # POINT with comma
        result = self.processor.parse_geom_point('POINT(100.5014,13.7563)')
        self.assertEqual(result['longitude'], 100.5014)
        self.assertEqual(result['latitude'], 13.7563)
        
        # Simple coordinate pair
        result = self.processor.parse_geom_point('13.7563 100.5014')
        self.assertEqual(result['longitude'], 100.5014)
        self.assertEqual(result['latitude'], 13.7563)
    
    def test_parse_geom_point_invalid(self):
        """Test geolocation parsing with invalid coordinates."""
        # Invalid coordinates
        result = self.processor.parse_geom_point('POINT(300 500)')
        self.assertEqual(result, {})
        
        # Malformed string
        result = self.processor.parse_geom_point('invalid string')
        self.assertEqual(result, {})
        
        # Empty string
        result = self.processor.parse_geom_point('')
        self.assertEqual(result, {})
    
    def test_extract_metadata_from_row(self):
        """Test metadata extraction from CSV row."""
        row_data = {
            'province': 'กรุงเทพมหานคร',
            'district': 'บางกะปิ',
            'subdistrict': 'หัวหมาก',
            'deed_type': 'โฉนดที่ดิน',
            'area_rai': 2,
            'area_ngan': 1,
            'area_wa': 50,
            'land_geom_point': 'POINT(100.5014 13.7563)',
            'land_use_type': 'ที่อยู่อาศัย',
            'owner_date': '2023-01-15'
        }
        
        row = pd.Series(row_data)
        metadata = self.processor.extract_metadata_from_row(row)
        
        # Check basic metadata
        self.assertEqual(metadata['province'], 'กรุงเทพมหานคร')
        self.assertEqual(metadata['district'], 'บางกะปิ')
        self.assertEqual(metadata['deed_type'], 'โฉนดที่ดิน')
        
        # Check computed metadata
        self.assertIn('location_hierarchy', metadata)
        self.assertEqual(metadata['location_hierarchy'], 'กรุงเทพมหานคร > บางกะปิ > หัวหมาก')
        
        # Check area formatting
        self.assertIn('area_formatted', metadata)
        self.assertEqual(metadata['area_formatted'], '2 ไร่ 1 งาน 50 ตร.ว.')
        
        # Check geolocation
        self.assertIn('longitude', metadata)
        self.assertIn('latitude', metadata)
        self.assertEqual(metadata['longitude'], 100.5014)
        self.assertEqual(metadata['latitude'], 13.7563)
        
        # Check search text
        self.assertIn('search_text', metadata)
        self.assertIn('กรุงเทพมหานคร', metadata['search_text'])
    
    def test_extract_metadata_with_aliases(self):
        """Test metadata extraction using field aliases."""
        # Add alias mapping
        alias_mapping = FieldMapping(
            csv_column='prov',
            metadata_key='province',
            field_type='location',
            aliases=['province_name', 'จังหวัด']
        )
        
        config_with_alias = DatasetConfig(
            name='test_alias',
            description='Test with aliases',
            field_mappings=[alias_mapping]
        )
        
        processor = DocumentProcessor(config_with_alias)
        
        # Test with alias column
        row = pd.Series({'province_name': 'เชียงใหม่'})
        metadata = processor.extract_metadata_from_row(row)
        
        self.assertEqual(metadata['province'], 'เชียงใหม่')
    
    def test_convert_row_to_document(self):
        """Test complete row to document conversion."""
        row_data = {
            'province': 'กรุงเทพมหานคร',
            'district': 'บางกะปิ',
            'deed_type': 'โฉนดที่ดิน',
            'area_rai': 2,
            'land_use_type': 'ที่อยู่อาศัย'
        }
        
        row = pd.Series(row_data)
        doc = self.processor.convert_row_to_document(row, row_index=5)
        
        # Check document structure
        self.assertIsInstance(doc, SimpleDocument)
        self.assertIn('# บันทึกข้อมูลโฉนดที่ดิน', doc.text)
        
        # Check metadata
        self.assertEqual(doc.metadata['province'], 'กรุงเทพมหานคร')
        self.assertEqual(doc.metadata['doc_type'], 'land_deed_record')
        self.assertEqual(doc.metadata['row_index'], 5)
        self.assertIn('created_at', doc.metadata)
        self.assertIn('doc_id', doc.metadata)
    
    def test_generate_document_text_structure(self):
        """Test document text generation structure."""
        metadata = {
            'province': 'กรุงเทพมหานคร',
            'district': 'บางกะปิ',
            'deed_type': 'โฉนดที่ดิน',
            'area_formatted': '2 ไร่',
            'location_hierarchy': 'กรุงเทพมหานคร > บางกะปิ'
        }
        
        row = pd.Series(metadata)
        text = self.processor.generate_document_text(row, metadata)
        
        # Check structure
        self.assertIn('# บันทึกข้อมูลโฉนดที่ดิน', text)
        self.assertIn('## ข้อมูลโฉนด', text)
        self.assertIn('## ที่ตั้ง', text)
        self.assertIn('กรุงเทพมหานคร', text)
        self.assertIn('โฉนดที่ดิน', text)


class TestFileOutputManager(unittest.TestCase):
    """Test FileOutputManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = DatasetConfig(
            name='test_config',
            description='Test configuration',
            field_mappings=[]
        )
        self.manager = FileOutputManager(self.temp_dir, self.config)
        
        # Create test documents
        self.test_docs = [
            SimpleDocument('Test doc 1', {'id': 'doc1', 'province': 'กรุงเทพ'}),
            SimpleDocument('Test doc 2', {'id': 'doc2', 'province': 'เชียงใหม่'})
        ]
    
    def test_save_documents_as_jsonl(self):
        """Test saving documents as JSONL format."""
        output_file = self.manager.save_documents_as_jsonl(self.test_docs, 'test.jsonl')
        
        self.assertTrue(Path(output_file).exists())
        
        # Verify content
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)
            
            # Parse first line
            doc_data = json.loads(lines[0])
            self.assertEqual(doc_data['text'], 'Test doc 1')
            self.assertEqual(doc_data['metadata']['province'], 'กรุงเทพ')
    
    def test_save_documents_as_markdown_files(self):
        """Test saving documents as individual markdown files."""
        md_files = self.manager.save_documents_as_markdown_files(
            self.test_docs, 
            prefix='test_doc', 
            batch_size=1
        )
        
        self.assertEqual(len(md_files), 2)
        
        # Check files exist
        for file_path in md_files:
            self.assertTrue(Path(file_path).exists())
            
            # Check content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.assertIn('Test doc', content)


class TestCSVAnalyzer(unittest.TestCase):
    """Test CSVAnalyzer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample CSV data
        self.sample_data = pd.DataFrame({
            'province': ['กรุงเทพมหานคร', 'เชียงใหม่', 'ภูเก็ต'],
            'district': ['บางกะปิ', 'เมือง', 'เมือง'],
            'area_rai': [2, 5, 1],
            'deed_type': ['โฉนดที่ดิน', 'โฉนดที่ดิน', 'ใบอนุญาต'],
            'empty_col': [None, '', 'ไม่ระบุ']
        })
        
        self.analyzer = CSVAnalyzer()
    
    def test_analyze_column_types(self):
        """Test column type analysis."""
        analysis = self.analyzer.analyze_dataframe_structure(self.sample_data)
        
        # Check column types
        self.assertIn('province', analysis['columns'])
        self.assertIn('area_rai', analysis['columns'])
        
        # Check data types
        province_info = analysis['columns']['province']
        self.assertEqual(province_info['data_type'], 'object')
        self.assertEqual(province_info['non_null_count'], 3)
        
        area_info = analysis['columns']['area_rai']
        self.assertEqual(area_info['data_type'], 'int64')
    
    def test_suggest_field_mappings(self):
        """Test field mapping suggestions."""
        suggestions = self.analyzer.suggest_field_mappings(self.sample_data)
        
        # Should suggest mappings for key columns
        mapping_dict = {s.csv_column: s for s in suggestions}
        
        self.assertIn('province', mapping_dict)
        self.assertEqual(mapping_dict['province'].field_type, 'location')
        
        self.assertIn('area_rai', mapping_dict)
        self.assertEqual(mapping_dict['area_rai'].field_type, 'area_measurements')
    
    def test_detect_thai_content(self):
        """Test Thai content detection."""
        # Thai text
        self.assertTrue(self.analyzer._contains_thai_text('กรุงเทพมหานคร'))
        self.assertTrue(self.analyzer._contains_thai_text('Mixed Thai ข้อความ'))
        
        # Non-Thai text
        self.assertFalse(self.analyzer._contains_thai_text('Bangkok'))
        self.assertFalse(self.analyzer._contains_thai_text('123'))


class TestDatasetConfigManager(unittest.TestCase):
    """Test DatasetConfigManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = DatasetConfigManager(self.temp_dir)
        
        # Create test config
        self.test_config = DatasetConfig(
            name='test_dataset',
            description='Test dataset configuration',
            field_mappings=[
                FieldMapping('province', 'province', 'location'),
                FieldMapping('deed_type', 'deed_type', 'deed_info')
            ]
        )
    
    def test_save_and_load_config(self):
        """Test saving and loading configurations."""
        # Save config
        config_file = self.manager.save_config(self.test_config)
        self.assertTrue(Path(config_file).exists())
        
        # Load config
        loaded_config = self.manager.load_config('test_dataset')
        
        self.assertEqual(loaded_config.name, 'test_dataset')
        self.assertEqual(len(loaded_config.field_mappings), 2)
        self.assertEqual(loaded_config.field_mappings[0].csv_column, 'province')
    
    def test_list_available_configs(self):
        """Test listing available configurations."""
        # Save multiple configs
        self.manager.save_config(self.test_config)
        
        config2 = DatasetConfig(
            name='another_dataset',
            description='Another test dataset',
            field_mappings=[]
        )
        self.manager.save_config(config2)
        
        # List configs
        configs = self.manager.list_available_configs()
        config_names = [c['name'] for c in configs]
        
        self.assertIn('test_dataset', config_names)
        self.assertIn('another_dataset', config_names)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete data processing pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample CSV data
        self.sample_data = pd.DataFrame({
            'province': ['กรุงเทพมหานคร', 'เชียงใหม่'],
            'district': ['บางกะปิ', 'เมือง'],
            'deed_type': ['โฉนดที่ดิน', 'โฉนดที่ดิน'],
            'area_rai': [2, 5],
            'land_geom_point': ['POINT(100.5014 13.7563)', 'POINT(98.9817 18.7883)'],
            'land_use_type': ['ที่อยู่อาศัย', 'เกษตรกรรม']
        })
    
    def test_complete_processing_pipeline(self):
        """Test the complete data processing pipeline."""
        # 1. Analyze CSV structure
        analyzer = CSVAnalyzer()
        suggested_mappings = analyzer.suggest_field_mappings(self.sample_data)
        
        # 2. Create configuration
        config = DatasetConfig(
            name='integration_test',
            description='Integration test configuration',
            field_mappings=suggested_mappings
        )
        
        # 3. Process documents
        processor = DocumentProcessor(config)
        documents = []
        
        for idx, (_, row) in enumerate(self.sample_data.iterrows()):
            doc = processor.convert_row_to_document(row, idx)
            documents.append(doc)
        
        # 4. Save outputs
        file_manager = FileOutputManager(self.temp_dir, config)
        
        # Save as JSONL
        jsonl_file = file_manager.save_documents_as_jsonl(documents, 'integration_test.jsonl')
        self.assertTrue(Path(jsonl_file).exists())
        
        # Save as markdown
        md_files = file_manager.save_documents_as_markdown_files(documents, 'integration', batch_size=10)
        self.assertEqual(len(md_files), 1)  # Should be one batch file
        
        # 5. Verify document content
        self.assertEqual(len(documents), 2)
        
        # Check first document
        doc1 = documents[0]
        self.assertIn('กรุงเทพมหานคร', doc1.text)  
        self.assertEqual(doc1.metadata['province'], 'กรุงเทพมหานคร')
        self.assertIn('longitude', doc1.metadata)
        self.assertIn('location_hierarchy', doc1.metadata)
        
        # Check geolocation parsing
        self.assertEqual(doc1.metadata['longitude'], 100.5014)
        self.assertEqual(doc1.metadata['latitude'], 13.7563)


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestFieldMapping,
        TestDatasetConfig,
        TestSimpleDocument,
        TestDocumentProcessor,
        TestFileOutputManager,
        TestCSVAnalyzer,
        TestDatasetConfigManager,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY - Data Processing Module")
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