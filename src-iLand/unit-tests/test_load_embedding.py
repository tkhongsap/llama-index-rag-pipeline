"""
Comprehensive unit tests for the load_embedding module.

Tests cover embedding loading, index reconstruction, validation,
and utility functions for iLand embedding management.
"""

import unittest
import tempfile
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import the modules to test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_embedding.models import EmbeddingConfig, FilterConfig, EMBEDDING_DIR, THAI_PROVINCES
from load_embedding.embedding_loader import iLandEmbeddingLoader
from load_embedding.index_reconstructor import iLandIndexReconstructor
from load_embedding.validation import validate_iland_embeddings, generate_validation_report
from load_embedding.utils import (
    load_latest_iland_embeddings,
    load_all_latest_iland_embeddings,
    create_iland_index_from_latest_batch,
    get_iland_batch_summary
)

# Mock LlamaIndex imports that might not be available
try:
    from llama_index.core import VectorStoreIndex, DocumentSummaryIndex
    from llama_index.core.schema import NodeWithScore
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False
    VectorStoreIndex = Mock
    DocumentSummaryIndex = Mock
    NodeWithScore = Mock


class TestEmbeddingConfig(unittest.TestCase):
    """Test EmbeddingConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = EmbeddingConfig()
        
        self.assertEqual(config.embedding_dir, EMBEDDING_DIR)
        self.assertEqual(config.embed_model, "text-embedding-3-small")
        self.assertEqual(config.llm_model, "gpt-4o-mini")
        self.assertIsNone(config.api_key)
    
    def test_custom_config(self):
        """Test custom configuration."""
        custom_dir = Path("/custom/path")
        config = EmbeddingConfig(
            embedding_dir=custom_dir,
            embed_model="text-embedding-3-large",
            llm_model="gpt-4",
            api_key="test_key"
        )
        
        self.assertEqual(config.embedding_dir, custom_dir)
        self.assertEqual(config.embed_model, "text-embedding-3-large")
        self.assertEqual(config.llm_model, "gpt-4")
        self.assertEqual(config.api_key, "test_key")
    
    def test_validate_config(self):
        """Test configuration validation."""
        # Valid config
        config = EmbeddingConfig()
        self.assertTrue(config.validate())
        
        # Invalid model
        config.embed_model = ""
        self.assertFalse(config.validate())
        
        # Invalid directory
        config.embed_model = "valid-model"
        config.embedding_dir = Path("/nonexistent/path/that/should/not/exist")
        # Should still be valid as directory might be created later
        self.assertTrue(config.validate())
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = EmbeddingConfig()
        config_dict = config.to_dict()
        
        self.assertIn('embedding_dir', config_dict)
        self.assertIn('embed_model', config_dict)
        self.assertIn('llm_model', config_dict)
        self.assertEqual(config_dict['embed_model'], "text-embedding-3-small")


class TestFilterConfig(unittest.TestCase):
    """Test FilterConfig functionality."""
    
    def test_province_filter(self):
        """Test province-based filtering."""
        filter_config = FilterConfig(provinces=['กรุงเทพมหานคร', 'เชียงใหม่'])
        
        self.assertEqual(len(filter_config.provinces), 2)
        self.assertIn('กรุงเทพมหานคร', filter_config.provinces)
        self.assertIn('เชียงใหม่', filter_config.provinces)
    
    def test_deed_type_filter(self):
        """Test deed type filtering."""
        filter_config = FilterConfig(deed_types=['โฉนดที่ดิน', 'ใบอนุญาต'])
        
        self.assertEqual(len(filter_config.deed_types), 2)
        self.assertIn('โฉนดที่ดิน', filter_config.deed_types)
    
    def test_area_range_filter(self):
        """Test area range filtering."""
        filter_config = FilterConfig(
            min_area_sqm=1000,
            max_area_sqm=5000
        )
        
        self.assertEqual(filter_config.min_area_sqm, 1000)
        self.assertEqual(filter_config.max_area_sqm, 5000)
    
    def test_date_range_filter(self):
        """Test date range filtering."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        filter_config = FilterConfig(
            start_date=start_date,
            end_date=end_date
        )
        
        self.assertEqual(filter_config.start_date, start_date)
        self.assertEqual(filter_config.end_date, end_date)
    
    def test_combined_filters(self):
        """Test combining multiple filters."""
        filter_config = FilterConfig(
            provinces=['กรุงเทพมหานคร'],
            deed_types=['โฉนดที่ดิน'],
            min_area_sqm=1000,
            land_use_types=['ที่อยู่อาศัย']
        )
        
        # Test filter application
        sample_metadata = {
            'province': 'กรุงเทพมหานคร',
            'deed_type': 'โฉนดที่ดิน',
            'area_total_sqm': 1500,
            'land_use_type': 'ที่อยู่อาศัย'
        }
        
        self.assertTrue(filter_config.matches(sample_metadata))
        
        # Test non-matching metadata
        non_matching = {
            'province': 'เชียงใหม่',  # Different province
            'deed_type': 'โฉนดที่ดิน',
            'area_total_sqm': 1500,
            'land_use_type': 'ที่อยู่อาศัย'
        }
        
        self.assertFalse(filter_config.matches(non_matching))


class TestiLandEmbeddingLoader(unittest.TestCase):
    """Test iLandEmbeddingLoader functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = EmbeddingConfig(embedding_dir=Path(self.temp_dir))
        self.loader = iLandEmbeddingLoader(self.config)
        
        # Create mock embedding files
        self.mock_embeddings = {
            'chunk_embeddings': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            'metadata': [
                {
                    'id': 'chunk_1',
                    'text': 'โฉนดที่ดินในกรุงเทพมหานคร',
                    'province': 'กรุงเทพมหานคร',
                    'deed_type': 'โฉนดที่ดิน',
                    'area_total_sqm': 1600
                },
                {
                    'id': 'chunk_2',
                    'text': 'โฉนดที่ดินในเชียงใหม่',
                    'province': 'เชียงใหม่',
                    'deed_type': 'โฉนดที่ดิน',
                    'area_total_sqm': 3200
                }
            ],
            'embedding_model': 'text-embedding-3-small',
            'created_at': datetime.now().isoformat(),
            'batch_info': {
                'batch_id': 'test_batch_001',
                'total_chunks': 2
            }
        }
        
        # Save mock embedding file
        self.batch_file = Path(self.temp_dir) / "embeddings_iland_test_batch_001.json"
        with open(self.batch_file, 'w', encoding='utf-8') as f:
            json.dump(self.mock_embeddings, f, ensure_ascii=False)
    
    def test_load_batch(self):
        """Test loading embedding batch."""
        batch_data = self.loader.load_batch('test_batch_001')
        
        self.assertIsNotNone(batch_data)
        self.assertEqual(len(batch_data['chunk_embeddings']), 2)
        self.assertEqual(len(batch_data['metadata']), 2)
        self.assertEqual(batch_data['embedding_model'], 'text-embedding-3-small')
    
    def test_load_batch_not_found(self):
        """Test loading non-existent batch."""
        batch_data = self.loader.load_batch('nonexistent_batch')
        self.assertIsNone(batch_data)
    
    def test_list_available_batches(self):
        """Test listing available batches."""
        # Create additional batch files
        batch2_file = Path(self.temp_dir) / "embeddings_iland_test_batch_002.json"
        with open(batch2_file, 'w', encoding='utf-8') as f:
            json.dump(self.mock_embeddings, f)
        
        batches = self.loader.list_available_batches()
        
        self.assertEqual(len(batches), 2)
        batch_ids = [b['batch_id'] for b in batches]
        self.assertIn('test_batch_001', batch_ids)
        self.assertIn('test_batch_002', batch_ids)
    
    def test_get_latest_batch(self):
        """Test getting latest batch."""
        # Create newer batch
        newer_embeddings = self.mock_embeddings.copy()
        newer_embeddings['created_at'] = datetime.now().isoformat()
        newer_embeddings['batch_info']['batch_id'] = 'test_batch_newer'
        
        batch_newer = Path(self.temp_dir) / "embeddings_iland_test_batch_newer.json"
        with open(batch_newer, 'w', encoding='utf-8') as f:
            json.dump(newer_embeddings, f)
        
        latest = self.loader.get_latest_batch()
        
        self.assertIsNotNone(latest)
        # Should get the newer batch due to filename sorting
        self.assertIn('test_batch', latest['batch_info']['batch_id'])
    
    def test_load_embeddings_with_filter(self):
        """Test loading embeddings with filtering."""
        filter_config = FilterConfig(provinces=['กรุงเทพมหานคร'])
        
        embeddings, metadata = self.loader.load_embeddings_with_filter(
            'test_batch_001', 
            filter_config
        )
        
        # Should only return Bangkok entries
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(len(metadata), 1)
        self.assertEqual(metadata[0]['province'], 'กรุงเทพมหานคร')
    
    def test_get_batch_statistics(self):
        """Test getting batch statistics."""
        stats = self.loader.get_batch_statistics('test_batch_001')
        
        self.assertIn('total_chunks', stats)
        self.assertIn('unique_provinces', stats)
        self.assertIn('deed_types', stats)
        self.assertIn('area_statistics', stats)
        
        self.assertEqual(stats['total_chunks'], 2)
        self.assertEqual(len(stats['unique_provinces']), 2)
        self.assertIn('กรุงเทพมหานคร', stats['unique_provinces'])
    
    def test_search_batches_by_metadata(self):
        """Test searching batches by metadata."""
        # Create batch with specific metadata
        special_embeddings = self.mock_embeddings.copy()
        special_embeddings['batch_info']['batch_id'] = 'special_batch'
        special_embeddings['metadata'][0]['special_field'] = 'special_value'
        
        special_file = Path(self.temp_dir) / "embeddings_iland_special_batch.json"
        with open(special_file, 'w', encoding='utf-8') as f:
            json.dump(special_embeddings, f)
        
        # Search for batches containing Bangkok
        results = self.loader.search_batches_by_metadata('กรุงเทพมหานคร')
        
        self.assertGreater(len(results), 0)
        self.assertTrue(any('test_batch' in r['batch_id'] for r in results))


class TestiLandIndexReconstructor(unittest.TestCase):
    """Test iLandIndexReconstructor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = EmbeddingConfig(embedding_dir=Path(self.temp_dir))
        self.reconstructor = iLandIndexReconstructor(self.config)
        
        # Create mock embeddings with proper structure
        self.mock_embeddings = {
            'chunk_embeddings': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            'metadata': [
                {
                    'id': 'chunk_1',
                    'text': 'Text content 1',
                    'province': 'กรุงเทพมหานคร',
                    'deed_type': 'โฉนดที่ดิน'
                },
                {
                    'id': 'chunk_2', 
                    'text': 'Text content 2',
                    'province': 'เชียงใหม่',
                    'deed_type': 'โฉนดที่ดิน'
                }
            ]
        }
    
    @patch('load_embedding.index_reconstructor.VectorStoreIndex')
    def test_create_vector_index(self, mock_vector_index):
        """Test creating vector store index."""
        mock_index = Mock()
        mock_vector_index.from_documents.return_value = mock_index
        
        index = self.reconstructor.create_vector_index(
            self.mock_embeddings['chunk_embeddings'],
            self.mock_embeddings['metadata']
        )
        
        self.assertIsNotNone(index)
        mock_vector_index.from_documents.assert_called_once()
    
    @patch('load_embedding.index_reconstructor.DocumentSummaryIndex')
    def test_create_summary_index(self, mock_summary_index):
        """Test creating document summary index."""
        mock_index = Mock()
        mock_summary_index.from_documents.return_value = mock_index
        
        index = self.reconstructor.create_summary_index(
            self.mock_embeddings['metadata']
        )
        
        self.assertIsNotNone(index)
        mock_summary_index.from_documents.assert_called_once()
    
    def test_create_nodes_from_embeddings(self):
        """Test creating nodes from embeddings."""
        nodes = self.reconstructor._create_nodes_from_embeddings(
            self.mock_embeddings['chunk_embeddings'],
            self.mock_embeddings['metadata']
        )
        
        self.assertEqual(len(nodes), 2)
        
        # Check node structure
        node1 = nodes[0]
        self.assertEqual(node1.text, 'Text content 1')
        self.assertEqual(node1.metadata['province'], 'กรุงเทพมหานคร')
        self.assertIsNotNone(node1.embedding)
    
    def test_create_hierarchical_index(self):
        """Test creating hierarchical index structure."""
        with patch.multiple(
            self.reconstructor,
            create_vector_index=Mock(return_value=Mock()),
            create_summary_index=Mock(return_value=Mock())
        ):
            hierarchy = self.reconstructor.create_hierarchical_index(
                self.mock_embeddings['chunk_embeddings'],
                self.mock_embeddings['metadata']
            )
            
            self.assertIn('vector_index', hierarchy)
            self.assertIn('summary_index', hierarchy)
            self.assertIn('metadata', hierarchy)
    
    def test_group_by_province(self):
        """Test grouping embeddings by province."""
        grouped = self.reconstructor._group_by_province(
            self.mock_embeddings['chunk_embeddings'],
            self.mock_embeddings['metadata']
        )
        
        self.assertIn('กรุงเทพมหานคร', grouped)
        self.assertIn('เชียงใหม่', grouped)
        
        bangkok_group = grouped['กรุงเทพมหานคร']
        self.assertEqual(len(bangkok_group['embeddings']), 1)
        self.assertEqual(len(bangkok_group['metadata']), 1)
    
    @patch.multiple(
        'load_embedding.index_reconstructor',
        VectorStoreIndex=Mock(),
        DocumentSummaryIndex=Mock()
    )
    def test_create_province_specific_indexes(self):
        """Test creating province-specific indexes."""
        province_indexes = self.reconstructor.create_province_specific_indexes(
            self.mock_embeddings['chunk_embeddings'],
            self.mock_embeddings['metadata']
        )
        
        self.assertIn('กรุงเทพมหานคร', province_indexes)
        self.assertIn('เชียงใหม่', province_indexes)
        
        bangkok_index = province_indexes['กรุงเทพมหานคร']
        self.assertIn('vector_index', bangkok_index)
        self.assertIn('chunk_count', bangkok_index)


class TestValidation(unittest.TestCase):
    """Test validation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create valid embedding data
        self.valid_embeddings = {
            'chunk_embeddings': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            'metadata': [
                {
                    'id': 'chunk_1',
                    'text': 'Valid text content',
                    'province': 'กรุงเทพมหานคร'
                },
                {
                    'id': 'chunk_2',
                    'text': 'Another valid text',
                    'province': 'เชียงใหม่'
                }
            ],
            'embedding_model': 'text-embedding-3-small',
            'created_at': datetime.now().isoformat(),
            'batch_info': {
                'batch_id': 'valid_batch',
                'total_chunks': 2
            }
        }
        
        # Create invalid embedding data
        self.invalid_embeddings = {
            'chunk_embeddings': [[0.1, 0.2], [0.4, 0.5, 0.6]],  # Inconsistent dimensions
            'metadata': [
                {'id': 'chunk_1', 'text': 'Text without province'},
                # Missing second metadata entry
            ],
            'embedding_model': '',  # Empty model name
            'batch_info': {}  # Missing batch_id
        }
    
    def test_validate_valid_embeddings(self):
        """Test validation of valid embeddings."""
        result = validate_iland_embeddings(self.valid_embeddings)
        
        self.assertTrue(result['is_valid'])
        self.assertEqual(len(result['errors']), 0)
        self.assertEqual(len(result['warnings']), 0)
    
    def test_validate_invalid_embeddings(self):
        """Test validation of invalid embeddings."""
        result = validate_iland_embeddings(self.invalid_embeddings)
        
        self.assertFalse(result['is_valid'])
        self.assertGreater(len(result['errors']), 0)
        
        # Check specific errors
        error_messages = ' '.join(result['errors'])
        self.assertIn('embedding dimensions', error_messages.lower())
        self.assertIn('metadata count', error_messages.lower())
    
    def test_validate_missing_fields(self):
        """Test validation with missing required fields."""
        incomplete_embeddings = {
            'chunk_embeddings': [[0.1, 0.2, 0.3]],
            # Missing metadata
            'embedding_model': 'test-model'
            # Missing batch_info
        }
        
        result = validate_iland_embeddings(incomplete_embeddings)
        
        self.assertFalse(result['is_valid'])
        self.assertTrue(any('metadata' in error.lower() for error in result['errors']))
    
    def test_validate_thai_content(self):
        """Test validation of Thai content in metadata."""
        thai_embeddings = self.valid_embeddings.copy()
        thai_embeddings['metadata'][0]['text'] = 'English only text'
        
        result = validate_iland_embeddings(thai_embeddings)
        
        # Should have warnings about missing Thai content
        self.assertTrue(any('thai' in warning.lower() for warning in result['warnings']))
    
    def test_generate_validation_report(self):
        """Test generation of validation report."""
        # Save valid embeddings to file
        batch_file = Path(self.temp_dir) / "embeddings_iland_test_batch.json"
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(self.valid_embeddings, f)
        
        # Generate report
        report = generate_validation_report(self.temp_dir)
        
        self.assertIn('total_batches', report)
        self.assertIn('valid_batches', report)
        self.assertIn('batch_reports', report)
        
        self.assertEqual(report['total_batches'], 1)
        self.assertEqual(report['valid_batches'], 1)
        
        batch_report = report['batch_reports'][0]
        self.assertEqual(batch_report['batch_id'], 'test_batch')
        self.assertTrue(batch_report['is_valid'])
    
    def test_validate_embeddings_consistency(self):
        """Test validation of embedding consistency across batches."""
        # Create two batches with different embedding models
        batch1 = self.valid_embeddings.copy()
        batch1['embedding_model'] = 'model-1'
        batch1['batch_info']['batch_id'] = 'batch_1'
        
        batch2 = self.valid_embeddings.copy()
        batch2['embedding_model'] = 'model-2'  # Different model
        batch2['batch_info']['batch_id'] = 'batch_2'
        
        # Save both batches
        for i, batch in enumerate([batch1, batch2], 1):
            batch_file = Path(self.temp_dir) / f"embeddings_iland_batch_{i}.json"
            with open(batch_file, 'w', encoding='utf-8') as f:
                json.dump(batch, f)
        
        # Generate report
        report = generate_validation_report(self.temp_dir)
        
        # Should have warnings about inconsistent embedding models
        self.assertTrue(any('model' in warning.lower() 
                          for warnings in [br.get('warnings', []) 
                                         for br in report['batch_reports']]
                          for warning in warnings))


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test embedding files
        self.embeddings_data = [
            {
                'chunk_embeddings': [[0.1, 0.2, 0.3]],
                'metadata': [{'id': 'chunk_1', 'province': 'กรุงเทพมหานคร'}],
                'embedding_model': 'text-embedding-3-small',
                'created_at': '2023-01-01T00:00:00',
                'batch_info': {'batch_id': 'batch_001', 'total_chunks': 1}
            },
            {
                'chunk_embeddings': [[0.4, 0.5, 0.6]],
                'metadata': [{'id': 'chunk_2', 'province': 'เชียงใหม่'}],
                'embedding_model': 'text-embedding-3-small',
                'created_at': '2023-01-02T00:00:00',
                'batch_info': {'batch_id': 'batch_002', 'total_chunks': 1}
            }
        ]
        
        # Save embedding files
        for i, data in enumerate(self.embeddings_data, 1):
            file_path = Path(self.temp_dir) / f"embeddings_iland_batch_{i:03d}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
    
    def test_load_latest_iland_embeddings(self):
        """Test loading latest embeddings."""
        latest = load_latest_iland_embeddings(self.temp_dir)
        
        self.assertIsNotNone(latest)
        self.assertEqual(latest['batch_info']['batch_id'], 'batch_002')  # Latest by filename
    
    def test_load_all_latest_iland_embeddings(self):
        """Test loading all latest embeddings."""
        all_latest = load_all_latest_iland_embeddings(self.temp_dir)
        
        self.assertEqual(len(all_latest), 2)
        
        # Check that all batches are included
        batch_ids = [batch['batch_info']['batch_id'] for batch in all_latest]
        self.assertIn('batch_001', batch_ids)
        self.assertIn('batch_002', batch_ids)
    
    @patch('load_embedding.utils.iLandIndexReconstructor')
    def test_create_iland_index_from_latest_batch(self, mock_reconstructor):
        """Test creating index from latest batch."""
        mock_reconstructor_instance = Mock()
        mock_reconstructor.return_value = mock_reconstructor_instance
        mock_reconstructor_instance.create_vector_index.return_value = Mock()
        
        config = EmbeddingConfig(embedding_dir=Path(self.temp_dir))
        index = create_iland_index_from_latest_batch(config)
        
        self.assertIsNotNone(index)
        mock_reconstructor.assert_called_once_with(config)
    
    def test_get_iland_batch_summary(self):
        """Test getting batch summary."""
        summary = get_iland_batch_summary(self.temp_dir)
        
        self.assertIn('total_batches', summary)
        self.assertIn('total_chunks', summary)
        self.assertIn('unique_provinces', summary)
        self.assertIn('embedding_models', summary)
        self.assertIn('batch_details', summary)
        
        self.assertEqual(summary['total_batches'], 2)
        self.assertEqual(summary['total_chunks'], 2)
        self.assertEqual(len(summary['unique_provinces']), 2)
        self.assertIn('กรุงเทพมหานคร', summary['unique_provinces'])
        self.assertIn('เชียงใหม่', summary['unique_provinces'])
    
    def test_get_iland_batch_summary_empty_dir(self):
        """Test getting batch summary from empty directory."""
        empty_dir = tempfile.mkdtemp()
        summary = get_iland_batch_summary(empty_dir)
        
        self.assertEqual(summary['total_batches'], 0)
        self.assertEqual(summary['total_chunks'], 0)
        self.assertEqual(len(summary['unique_provinces']), 0)
    
    def test_filter_embeddings_by_province(self):
        """Test filtering embeddings by province."""
        from load_embedding.utils import filter_embeddings_by_province
        
        # Load all embeddings
        all_embeddings = load_all_latest_iland_embeddings(self.temp_dir)
        
        # Filter by Bangkok
        filtered = filter_embeddings_by_province(all_embeddings, ['กรุงเทพมหานคร'])
        
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['metadata'][0]['province'], 'กรุงเทพมหานคร')
    
    def test_merge_embedding_batches(self):
        """Test merging multiple embedding batches."""
        from load_embedding.utils import merge_embedding_batches
        
        all_embeddings = load_all_latest_iland_embeddings(self.temp_dir)
        merged = merge_embedding_batches(all_embeddings)
        
        self.assertEqual(len(merged['chunk_embeddings']), 2)
        self.assertEqual(len(merged['metadata']), 2)
        self.assertIn('merged_batch_info', merged)
        self.assertEqual(merged['merged_batch_info']['total_batches'], 2)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete load_embedding pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = EmbeddingConfig(embedding_dir=Path(self.temp_dir))
        
        # Create comprehensive test data
        self.test_batches = [
            {
                'chunk_embeddings': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                'metadata': [
                    {
                        'id': 'bangkok_1',
                        'text': 'โฉนดที่ดินในกรุงเทพมหานคร บางกะปิ',
                        'province': 'กรุงเทพมหานคร',
                        'district': 'บางกะปิ',
                        'deed_type': 'โฉนดที่ดิน',
                        'area_total_sqm': 1600
                    },
                    {
                        'id': 'bangkok_2',
                        'text': 'โฉนดที่ดินในกรุงเทพมหานคร ห้วยขวาง',
                        'province': 'กรุงเทพมหานคร',
                        'district': 'ห้วยขวาง',
                        'deed_type': 'โฉนดที่ดิน',
                        'area_total_sqm': 3200
                    }
                ],
                'embedding_model': 'text-embedding-3-small',
                'created_at': '2023-01-01T00:00:00',
                'batch_info': {'batch_id': 'bangkok_batch', 'total_chunks': 2}
            },
            {
                'chunk_embeddings': [[0.7, 0.8, 0.9]],
                'metadata': [
                    {
                        'id': 'chiangmai_1',
                        'text': 'โฉนดที่ดินในเชียงใหม่ เมือง',
                        'province': 'เชียงใหม่',
                        'district': 'เมือง',
                        'deed_type': 'โฉนดที่ดิน',
                        'area_total_sqm': 6400
                    }
                ],
                'embedding_model': 'text-embedding-3-small',
                'created_at': '2023-01-02T00:00:00',
                'batch_info': {'batch_id': 'chiangmai_batch', 'total_chunks': 1}
            }
        ]
        
        # Save test batches
        for batch in self.test_batches:
            batch_id = batch['batch_info']['batch_id']
            file_path = Path(self.temp_dir) / f"embeddings_iland_{batch_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(batch, f, ensure_ascii=False)
    
    def test_complete_loading_workflow(self):
        """Test the complete embedding loading workflow."""
        # 1. Initialize loader
        loader = iLandEmbeddingLoader(self.config)
        
        # 2. List available batches
        batches = loader.list_available_batches()
        self.assertEqual(len(batches), 2)
        
        # 3. Load specific batch
        bangkok_batch = loader.load_batch('bangkok_batch')
        self.assertIsNotNone(bangkok_batch)
        self.assertEqual(len(bangkok_batch['chunk_embeddings']), 2)
        
        # 4. Get batch statistics
        stats = loader.get_batch_statistics('bangkok_batch')
        self.assertEqual(stats['total_chunks'], 2)
        self.assertEqual(len(stats['unique_provinces']), 1)
        self.assertIn('กรุงเทพมหานคร', stats['unique_provinces'])
        
        # 5. Apply filters
        filter_config = FilterConfig(
            provinces=['กรุงเทพมหานคร'],
            min_area_sqm=2000
        )
        
        filtered_embeddings, filtered_metadata = loader.load_embeddings_with_filter(
            'bangkok_batch', 
            filter_config
        )
        
        # Should only get the larger Bangkok property
        self.assertEqual(len(filtered_embeddings), 1)
        self.assertEqual(filtered_metadata[0]['area_total_sqm'], 3200)
    
    def test_index_reconstruction_workflow(self):
        """Test the index reconstruction workflow."""
        # 1. Load embeddings
        loader = iLandEmbeddingLoader(self.config)
        batch_data = loader.load_batch('bangkok_batch')
        
        # 2. Initialize reconstructor
        reconstructor = iLandIndexReconstructor(self.config)
        
        # 3. Create nodes from embeddings
        nodes = reconstructor._create_nodes_from_embeddings(
            batch_data['chunk_embeddings'],
            batch_data['metadata']
        )
        
        self.assertEqual(len(nodes), 2)
        for node in nodes:
            self.assertIsNotNone(node.text)
            self.assertIsNotNone(node.embedding)
            self.assertIn('province', node.metadata)
        
        # 4. Group by province
        grouped = reconstructor._group_by_province(
            batch_data['chunk_embeddings'],
            batch_data['metadata']
        )
        
        self.assertIn('กรุงเทพมหานคร', grouped)
        bangkok_group = grouped['กรุงเทพมหานคร']
        self.assertEqual(len(bangkok_group['embeddings']), 2)
    
    def test_validation_workflow(self):
        """Test the validation workflow."""
        # 1. Validate individual batch
        loader = iLandEmbeddingLoader(self.config)
        batch_data = loader.load_batch('bangkok_batch')
        
        validation_result = validate_iland_embeddings(batch_data)
        self.assertTrue(validation_result['is_valid'])
        self.assertEqual(len(validation_result['errors']), 0)
        
        # 2. Generate comprehensive report
        report = generate_validation_report(self.temp_dir)
        
        self.assertEqual(report['total_batches'], 2)
        self.assertEqual(report['valid_batches'], 2)
        
        # Check batch-specific reports
        for batch_report in report['batch_reports']:
            self.assertTrue(batch_report['is_valid'])
            self.assertIn('total_chunks', batch_report)
            self.assertIn('embedding_model', batch_report)
    
    def test_utility_functions_workflow(self):
        """Test the utility functions workflow."""
        # 1. Get overall summary
        summary = get_iland_batch_summary(self.temp_dir)
        
        self.assertEqual(summary['total_batches'], 2)
        self.assertEqual(summary['total_chunks'], 3)
        self.assertEqual(len(summary['unique_provinces']), 2)
        self.assertIn('กรุงเทพมหานคร', summary['unique_provinces'])
        self.assertIn('เชียงใหม่', summary['unique_provinces'])
        
        # 2. Load latest embeddings
        latest = load_latest_iland_embeddings(self.temp_dir)
        self.assertIsNotNone(latest)
        
        # 3. Load all embeddings
        all_embeddings = load_all_latest_iland_embeddings(self.temp_dir)
        self.assertEqual(len(all_embeddings), 2)
        
        # 4. Filter and merge
        from load_embedding.utils import (
            filter_embeddings_by_province, 
            merge_embedding_batches
        )
        
        # Filter by province
        bangkok_only = filter_embeddings_by_province(
            all_embeddings, 
            ['กรุงเทพมหานคร']
        )
        self.assertEqual(len(bangkok_only), 1)
        
        # Merge all batches
        merged = merge_embedding_batches(all_embeddings)
        self.assertEqual(len(merged['chunk_embeddings']), 3)
        self.assertEqual(len(merged['metadata']), 3)


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestEmbeddingConfig,
        TestFilterConfig,
        TestiLandEmbeddingLoader,
        TestiLandIndexReconstructor,
        TestValidation,
        TestUtils,
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
    print(f"TEST SUMMARY - Load Embedding Module")
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