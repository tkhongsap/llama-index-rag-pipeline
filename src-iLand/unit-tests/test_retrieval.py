"""
Comprehensive unit tests for the retrieval module.

Tests cover all retrieval strategies, CLI functionality, caching,
parallel execution, and the router system for iLand data.
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

from retrieval.retrievers.base import BaseRetrieverAdapter
from retrieval.retrievers.vector import VectorRetrieverAdapter
from retrieval.retrievers.metadata import MetadataRetrieverAdapter
from retrieval.retrievers.hybrid import HybridRetrieverAdapter
from retrieval.retrievers.summary import SummaryRetrieverAdapter
from retrieval.retrievers.recursive import RecursiveRetrieverAdapter
from retrieval.retrievers.chunk_decoupling import ChunkDecouplingRetrieverAdapter
from retrieval.retrievers.section_retriever import SectionRetrieverAdapter
from retrieval.router import iLandQueryRouter
from retrieval.cache import RetrievalCache
from retrieval.parallel_executor import ParallelRetrieverExecutor
from retrieval.fast_metadata_index import FastMetadataIndex
from retrieval.index_classifier import IndexClassifier

# Mock LlamaIndex imports
try:
    from llama_index.core.schema import NodeWithScore
    from llama_index.core import VectorStoreIndex, DocumentSummaryIndex
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False
    NodeWithScore = Mock
    VectorStoreIndex = Mock
    DocumentSummaryIndex = Mock


class TestBaseRetrieverAdapter(unittest.TestCase):
    """Test BaseRetrieverAdapter functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a concrete implementation for testing
        class TestRetriever(BaseRetrieverAdapter):
            def retrieve(self, query: str, top_k=None):
                # Mock implementation
                mock_node = Mock()
                mock_node.node = Mock()
                mock_node.node.metadata = {}
                mock_node.score = 0.8
                return [mock_node]
        
        self.retriever = TestRetriever("test_strategy")
    
    def test_initialization(self):
        """Test base retriever initialization."""
        self.assertEqual(self.retriever.strategy_name, "test_strategy")
        self.assertEqual(self.retriever.name, "test_strategy")
    
    def test_tag_nodes_with_strategy(self):
        """Test tagging nodes with strategy information."""
        # Create mock nodes
        mock_nodes = []
        for i in range(2):
            mock_node = Mock()
            mock_node.node = Mock()
            mock_node.node.metadata = {'original_field': f'value_{i}'}
            mock_nodes.append(mock_node)
        
        # Tag nodes
        tagged_nodes = self.retriever._tag_nodes_with_strategy(mock_nodes)
        
        # Verify tagging
        for node in tagged_nodes:
            self.assertEqual(node.node.metadata['retrieval_strategy'], 'test_strategy')
            self.assertEqual(node.node.metadata['data_source'], 'iland')
            self.assertIn('original_field', node.node.metadata)
    
    def test_retrieve_abstract_method(self):
        """Test that retrieve method is implemented."""
        result = self.retriever.retrieve("test query")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)


class TestVectorRetrieverAdapter(unittest.TestCase):
    """Test VectorRetrieverAdapter functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_index = Mock()
        self.retriever = VectorRetrieverAdapter(self.mock_index)
    
    @patch('retrieval.retrievers.vector.VectorStoreIndex')
    def test_retrieve_basic(self, mock_vector_index):
        """Test basic vector retrieval."""
        # Setup mock
        mock_retriever = Mock()
        mock_response = Mock()
        mock_response.source_nodes = [Mock(), Mock()]
        
        self.mock_index.as_retriever.return_value = mock_retriever
        mock_retriever.retrieve.return_value = mock_response.source_nodes
        
        # Test retrieval
        result = self.retriever.retrieve("test query", top_k=5)
        
        # Verify calls
        self.mock_index.as_retriever.assert_called_once_with(similarity_top_k=5)
        mock_retriever.retrieve.assert_called_once_with("test query")
        
        # Verify result
        self.assertEqual(len(result), 2)
    
    def test_retrieve_with_similarity_threshold(self):
        """Test retrieval with similarity threshold."""
        # Setup mock with scores
        mock_nodes = []
        for score in [0.9, 0.7, 0.4, 0.2]:
            mock_node = Mock()
            mock_node.score = score
            mock_node.node = Mock()
            mock_node.node.metadata = {}
            mock_nodes.append(mock_node)
        
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = mock_nodes
        self.mock_index.as_retriever.return_value = mock_retriever
        
        # Test with threshold
        result = self.retriever.retrieve(
            "test query", 
            top_k=10, 
            similarity_threshold=0.6
        )
        
        # Should only return nodes with score >= 0.6
        self.assertEqual(len(result), 2)
        for node in result:
            self.assertGreaterEqual(node.score, 0.6)
    
    def test_retrieve_thai_query_preprocessing(self):
        """Test Thai query preprocessing."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = []
        self.mock_index.as_retriever.return_value = mock_retriever
        
        # Test with Thai query
        thai_query = "โฉนดที่ดินในกรุงเทพมหานคร"
        self.retriever.retrieve(thai_query)
        
        # Verify the query was passed through
        mock_retriever.retrieve.assert_called_once_with(thai_query)


class TestMetadataRetrieverAdapter(unittest.TestCase):
    """Test MetadataRetrieverAdapter functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_index = Mock()
        self.retriever = MetadataRetrieverAdapter(self.mock_index)
    
    def test_retrieve_with_province_filter(self):
        """Test retrieval with province metadata filter."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = []
        self.mock_index.as_retriever.return_value = mock_retriever
        
        # Test with province filter
        query = "โฉนดที่ดิน province:กรุงเทพมหานคร"
        self.retriever.retrieve(query)
        
        # Verify retriever was created with filters
        self.mock_index.as_retriever.assert_called_once()
        call_args = self.mock_index.as_retriever.call_args
        self.assertIn('filters', call_args[1])
    
    def test_retrieve_with_deed_type_filter(self):
        """Test retrieval with deed type filter."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = []
        self.mock_index.as_retriever.return_value = mock_retriever
        
        # Test with deed type filter
        query = "ที่ดิน deed_type:โฉนดที่ดิน"
        self.retriever.retrieve(query)
        
        # Verify filtering
        self.mock_index.as_retriever.assert_called_once()
    
    def test_parse_metadata_filters(self):
        """Test parsing metadata filters from query."""
        # Test various filter formats
        test_cases = [
            ("province:กรุงเทพมหานคร", "province", "กรุงเทพมหานคร"),
            ("deed_type:โฉนดที่ดิน", "deed_type", "โฉนดที่ดิน"),
            ("area_min:1000", "area_min", "1000")
        ]
        
        for query_part, expected_key, expected_value in test_cases:
            filters = self.retriever._parse_metadata_filters(f"test {query_part}")
            self.assertTrue(any(f.key == expected_key and f.value == expected_value 
                             for f in filters))
    
    def test_extract_clean_query(self):
        """Test extracting clean query without filters."""
        original_query = "โฉนดที่ดิน province:กรุงเทพมหานคร deed_type:โฉนดที่ดิน"
        clean_query = self.retriever._extract_clean_query(original_query)
        
        self.assertEqual(clean_query.strip(), "โฉนดที่ดิน")
        self.assertNotIn("province:", clean_query)
        self.assertNotIn("deed_type:", clean_query)


class TestHybridRetrieverAdapter(unittest.TestCase):
    """Test HybridRetrieverAdapter functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_vector_index = Mock()
        self.mock_keyword_index = Mock()
        self.retriever = HybridRetrieverAdapter(
            self.mock_vector_index, 
            self.mock_keyword_index
        )
    
    def test_retrieve_hybrid_combination(self):
        """Test hybrid retrieval combining vector and keyword results."""
        # Setup mock vector results
        vector_nodes = []
        for i in range(3):
            node = Mock()
            node.node = Mock()
            node.node.id = f"vector_{i}"
            node.node.metadata = {}
            node.score = 0.8 - (i * 0.1)
            vector_nodes.append(node)
        
        # Setup mock keyword results
        keyword_nodes = []
        for i in range(2):
            node = Mock()
            node.node = Mock()
            node.node.id = f"keyword_{i}"
            node.node.metadata = {}
            node.score = 0.7 - (i * 0.1)
            keyword_nodes.append(node)
        
        # Mock retrievers
        mock_vector_retriever = Mock()
        mock_vector_retriever.retrieve.return_value = vector_nodes
        self.mock_vector_index.as_retriever.return_value = mock_vector_retriever
        
        mock_keyword_retriever = Mock()
        mock_keyword_retriever.retrieve.return_value = keyword_nodes
        self.mock_keyword_index.as_retriever.return_value = mock_keyword_retriever
        
        # Test hybrid retrieval
        result = self.retriever.retrieve("test query", top_k=4)
        
        # Should combine and re-rank results
        self.assertLessEqual(len(result), 4)
        
        # Verify both retrievers were called
        mock_vector_retriever.retrieve.assert_called_once()
        mock_keyword_retriever.retrieve.assert_called_once()
    
    def test_hybrid_score_combination(self):
        """Test hybrid score combination strategies."""
        # Test with different alpha values
        alpha_values = [0.3, 0.5, 0.7]
        
        for alpha in alpha_values:
            retriever = HybridRetrieverAdapter(
                self.mock_vector_index,
                self.mock_keyword_index,
                alpha=alpha
            )
            
            # Test score combination
            vector_score = 0.8
            keyword_score = 0.6
            combined = retriever._combine_scores(vector_score, keyword_score)
            
            expected = alpha * vector_score + (1 - alpha) * keyword_score
            self.assertAlmostEqual(combined, expected, places=3)
    
    def test_deduplicate_nodes(self):
        """Test deduplication of nodes across retrieval methods."""
        # Create overlapping nodes
        nodes = []
        
        # Same node from different retrievers
        for retriever_type in ['vector', 'keyword']:
            node = Mock()
            node.node = Mock()
            node.node.id = "duplicate_node"
            node.node.metadata = {}
            node.score = 0.8 if retriever_type == 'vector' else 0.6
            nodes.append(node)
        
        # Different nodes
        for i in range(2):
            node = Mock()
            node.node = Mock()
            node.node.id = f"unique_{i}"
            node.node.metadata = {}
            node.score = 0.7
            nodes.append(node)
        
        # Test deduplication
        deduplicated = self.retriever._deduplicate_nodes(nodes)
        
        # Should have 3 unique nodes (1 deduplicated + 2 unique)
        self.assertEqual(len(deduplicated), 3)
        
        # Should keep the higher score for duplicates
        duplicate_node = next(n for n in deduplicated if n.node.id == "duplicate_node")
        self.assertEqual(duplicate_node.score, 0.8)


class TestSummaryRetrieverAdapter(unittest.TestCase):
    """Test SummaryRetrieverAdapter functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_summary_index = Mock()
        self.retriever = SummaryRetrieverAdapter(self.mock_summary_index)
    
    def test_retrieve_document_summaries(self):
        """Test retrieving document summaries."""
        mock_query_engine = Mock()
        mock_response = Mock()
        mock_response.source_nodes = [Mock(), Mock()]
        mock_query_engine.query.return_value = mock_response
        
        self.mock_summary_index.as_query_engine.return_value = mock_query_engine
        
        # Test retrieval
        result = self.retriever.retrieve("summarize land deeds in Bangkok")
        
        # Verify summary query engine was used
        self.mock_summary_index.as_query_engine.assert_called_once()
        mock_query_engine.query.assert_called_once()
        
        # Verify result
        self.assertEqual(len(result), 2)
    
    def test_detect_summary_query(self):
        """Test detection of summary-type queries."""
        summary_queries = [
            "สรุปโฉนดที่ดินในกรุงเทพ",
            "summarize all deeds",
            "overview of land types",
            "ภาพรวมที่ดิน"
        ]
        
        non_summary_queries = [
            "โฉนดที่ดินเลขที่ 12345",
            "specific deed information",
            "ที่ดินขนาด 2 ไร่"
        ]
        
        for query in summary_queries:
            self.assertTrue(self.retriever._is_summary_query(query))
        
        for query in non_summary_queries:
            self.assertFalse(self.retriever._is_summary_query(query))


class TestRecursiveRetrieverAdapter(unittest.TestCase):
    """Test RecursiveRetrieverAdapter functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_recursive_index = Mock()
        self.retriever = RecursiveRetrieverAdapter(self.mock_recursive_index)
    
    def test_retrieve_hierarchical(self):
        """Test hierarchical recursive retrieval."""
        mock_retriever = Mock()
        mock_nodes = [Mock(), Mock()]
        mock_retriever.retrieve.return_value = mock_nodes
        
        self.mock_recursive_index.as_retriever.return_value = mock_retriever
        
        # Test retrieval
        result = self.retriever.retrieve("complex hierarchical query")
        
        # Verify recursive retriever was used
        self.mock_recursive_index.as_retriever.assert_called_once()
        mock_retriever.retrieve.assert_called_once()
        
        # Verify result
        self.assertEqual(len(result), 2)
    
    def test_chunk_expansion(self):
        """Test chunk expansion functionality."""
        # Mock nodes with chunk relationships
        parent_chunk = Mock()
        parent_chunk.node = Mock()
        parent_chunk.node.id = "parent_chunk"
        parent_chunk.node.metadata = {
            'child_chunks': ['child_1', 'child_2'],
            'chunk_type': 'parent'
        }
        
        # Test expansion
        expanded = self.retriever._expand_chunks([parent_chunk])
        
        # Should include the parent chunk in result
        self.assertGreaterEqual(len(expanded), 1)


class TestChunkDecouplingRetrieverAdapter(unittest.TestCase):
    """Test ChunkDecouplingRetrieverAdapter functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_chunk_index = Mock()
        self.mock_doc_index = Mock()
        self.retriever = ChunkDecouplingRetrieverAdapter(
            self.mock_chunk_index,
            self.mock_doc_index
        )
    
    def test_chunk_to_document_retrieval(self):
        """Test chunk-to-document decoupling retrieval."""
        # Mock chunk retrieval
        chunk_nodes = []
        for i in range(2):
            node = Mock()
            node.node = Mock()
            node.node.metadata = {'document_id': f'doc_{i}'}
            node.score = 0.8
            chunk_nodes.append(node)
        
        mock_chunk_retriever = Mock()
        mock_chunk_retriever.retrieve.return_value = chunk_nodes
        self.mock_chunk_index.as_retriever.return_value = mock_chunk_retriever
        
        # Mock document retrieval
        doc_nodes = [Mock(), Mock()]
        mock_doc_retriever = Mock()
        mock_doc_retriever.retrieve.return_value = doc_nodes
        self.mock_doc_index.as_retriever.return_value = mock_doc_retriever
        
        # Test retrieval
        result = self.retriever.retrieve("test query")
        
        # Should combine chunk and document information
        self.assertGreaterEqual(len(result), 2)
        
        # Verify both retrievers were used
        mock_chunk_retriever.retrieve.assert_called_once()
    
    def test_document_context_enrichment(self):
        """Test document context enrichment."""
        chunk_node = Mock()
        chunk_node.node = Mock()
        chunk_node.node.metadata = {
            'document_id': 'doc_123',
            'chunk_text': 'Original chunk text'
        }
        
        doc_context = "Full document context with additional information"
        
        enriched = self.retriever._enrich_with_document_context(
            chunk_node, 
            doc_context
        )
        
        self.assertIn('enriched_context', enriched.node.metadata)


class TestSectionRetrieverAdapter(unittest.TestCase):
    """Test SectionRetrieverAdapter functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_section_index = Mock()
        self.retriever = SectionRetrieverAdapter(self.mock_section_index)
    
    def test_section_based_retrieval(self):
        """Test section-based retrieval."""
        # Mock section nodes
        section_nodes = []
        sections = ['deed_info', 'location', 'area_measurements']
        
        for section in sections:
            node = Mock()
            node.node = Mock()
            node.node.metadata = {
                'section_type': section,
                'section_content': f'Content for {section}'
            }
            node.score = 0.8
            section_nodes.append(node)
        
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = section_nodes
        self.mock_section_index.as_retriever.return_value = mock_retriever
        
        # Test retrieval
        result = self.retriever.retrieve("ข้อมูลที่ตั้งและขนาดพื้นที่")
        
        # Should return relevant sections
        self.assertEqual(len(result), 3)
        
        # Check section information is preserved
        for node in result:
            self.assertIn('section_type', node.node.metadata)
    
    def test_section_relevance_scoring(self):
        """Test section relevance scoring."""
        query = "ข้อมูลที่ตั้ง"  # Location information
        
        sections = [
            ('location', 0.9),     # Highly relevant
            ('deed_info', 0.3),    # Less relevant
            ('area_measurements', 0.2)  # Least relevant
        ]
        
        for section_type, expected_relevance in sections:
            relevance = self.retriever._calculate_section_relevance(
                query, 
                section_type
            )
            
            # Should assign higher relevance to location section
            if section_type == 'location':
                self.assertGreater(relevance, 0.7)


class TestiLandQueryRouter(unittest.TestCase):
    """Test iLandQueryRouter functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock retrievers
        self.mock_retrievers = {
            'vector': Mock(),
            'metadata': Mock(),
            'hybrid': Mock(),
            'summary': Mock(),
            'recursive': Mock()
        }
        
        self.router = iLandQueryRouter(self.mock_retrievers)
    
    def test_route_vector_query(self):
        """Test routing to vector retriever."""
        vector_queries = [
            "โฉนดที่ดินที่คล้ายกัน",
            "similar land deeds",
            "semantic search for properties"
        ]
        
        for query in vector_queries:
            strategy = self.router.route_query(query)
            self.assertEqual(strategy, 'vector')
    
    def test_route_metadata_query(self):
        """Test routing to metadata retriever."""
        metadata_queries = [
            "province:กรุงเทพมหานคร",
            "deed_type:โฉนดที่ดิน",
            "area > 1000"
        ]
        
        for query in metadata_queries:
            strategy = self.router.route_query(query)
            self.assertEqual(strategy, 'metadata')
    
    def test_route_summary_query(self):
        """Test routing to summary retriever."""
        summary_queries = [
            "สรุปโฉนดที่ดินทั้งหมด",
            "overview of all deeds",
            "ภาพรวมที่ดิน"
        ]
        
        for query in summary_queries:
            strategy = self.router.route_query(query)
            self.assertEqual(strategy, 'summary')
    
    def test_route_hybrid_query(self):
        """Test routing to hybrid retriever."""
        hybrid_queries = [
            "โฉนดที่ดินในกรุงเทพขนาดใหญ่",  # Combines semantic + metadata
            "large properties in Bangkok",
            "ที่อยู่อาศัยใกล้รถไฟฟ้า"
        ]
        
        for query in hybrid_queries:
            strategy = self.router.route_query(query)
            self.assertIn(strategy, ['hybrid', 'vector'])  # Could route to either
    
    def test_execute_routed_query(self):
        """Test executing routed query."""
        # Setup mock response
        mock_nodes = [Mock(), Mock()]
        self.mock_retrievers['vector'].retrieve.return_value = mock_nodes
        
        # Test execution
        result = self.router.execute_query("test vector query")
        
        # Verify correct retriever was called
        self.mock_retrievers['vector'].retrieve.assert_called_once()
        
        # Verify result
        self.assertEqual(len(result), 2)
    
    def test_query_analysis(self):
        """Test query analysis functionality."""
        test_query = "โฉนดที่ดินในกรุงเทพมหานคร ขนาดมากกว่า 2 ไร่"
        
        analysis = self.router._analyze_query(test_query)
        
        self.assertIn('has_thai_text', analysis)
        self.assertIn('has_location_filter', analysis)
        self.assertIn('has_area_filter', analysis)
        self.assertIn('query_type', analysis)
        
        self.assertTrue(analysis['has_thai_text'])
        self.assertTrue(analysis['has_location_filter'])
        self.assertTrue(analysis['has_area_filter'])


class TestRetrievalCache(unittest.TestCase):
    """Test RetrievalCache functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache = RetrievalCache(max_size=100, ttl_seconds=300)
    
    def test_cache_set_and_get(self):
        """Test caching and retrieving results."""
        query = "test query"
        results = [Mock(), Mock()]
        
        # Cache results
        self.cache.set(query, results)
        
        # Retrieve results
        cached_results = self.cache.get(query)
        
        self.assertIsNotNone(cached_results)
        self.assertEqual(len(cached_results), 2)
    
    def test_cache_miss(self):
        """Test cache miss for non-existent query."""
        result = self.cache.get("non-existent query")
        self.assertIsNone(result)
    
    def test_cache_expiration(self):
        """Test cache expiration based on TTL."""
        import time
        
        # Use very short TTL for testing
        short_cache = RetrievalCache(max_size=10, ttl_seconds=0.1)
        
        query = "expiring query"
        results = [Mock()]
        
        # Cache results
        short_cache.set(query, results)
        
        # Should be available immediately
        self.assertIsNotNone(short_cache.get(query))
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be expired
        self.assertIsNone(short_cache.get(query))
    
    def test_cache_size_limit(self):
        """Test cache size limiting."""
        small_cache = RetrievalCache(max_size=2, ttl_seconds=300)
        
        # Add items up to limit
        for i in range(3):
            query = f"query_{i}"
            results = [Mock()]
            small_cache.set(query, results)
        
        # First item should be evicted
        self.assertIsNone(small_cache.get("query_0"))
        self.assertIsNotNone(small_cache.get("query_1"))
        self.assertIsNotNone(small_cache.get("query_2"))
    
    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        query = "stats test query"
        results = [Mock()]
        
        # Cache and retrieve
        self.cache.set(query, results)
        self.cache.get(query)  # Cache hit
        self.cache.get("non-existent")  # Cache miss
        
        stats = self.cache.get_statistics()
        
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
        self.assertGreater(stats['hit_rate'], 0)


class TestParallelRetrieverExecutor(unittest.TestCase):
    """Test ParallelRetrieverExecutor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock retrievers
        self.mock_retrievers = {
            'vector': Mock(),
            'metadata': Mock(),
            'hybrid': Mock()
        }
        
        self.executor = ParallelRetrieverExecutor(self.mock_retrievers)
    
    def test_parallel_execution(self):
        """Test parallel execution of multiple retrievers."""
        # Setup mock responses
        for name, retriever in self.mock_retrievers.items():
            mock_nodes = [Mock() for _ in range(2)]
            for i, node in enumerate(mock_nodes):
                node.node = Mock()
                node.node.metadata = {'source': name, 'index': i}
                node.score = 0.8
            retriever.retrieve.return_value = mock_nodes
        
        # Execute parallel retrieval
        results = self.executor.execute_parallel("test query", ['vector', 'metadata'])
        
        # Verify both retrievers were called
        self.mock_retrievers['vector'].retrieve.assert_called_once()
        self.mock_retrievers['metadata'].retrieve.assert_called_once()
        self.mock_retrievers['hybrid'].retrieve.assert_not_called()
        
        # Check combined results
        self.assertIn('vector', results)
        self.assertIn('metadata', results)
        self.assertEqual(len(results['vector']), 2)
        self.assertEqual(len(results['metadata']), 2)
    
    def test_merge_parallel_results(self):
        """Test merging results from parallel execution."""
        parallel_results = {
            'vector': [Mock(), Mock()],
            'metadata': [Mock()]
        }
        
        # Set up mock nodes with scores
        for strategy, nodes in parallel_results.items():
            for i, node in enumerate(nodes):
                node.score = 0.8 - (i * 0.1)
                node.node = Mock()
                node.node.metadata = {'strategy': strategy}
        
        # Merge results
        merged = self.executor.merge_results(parallel_results, top_k=2)
        
        # Should return top 2 results across all strategies
        self.assertEqual(len(merged), 2)
        
        # Should be sorted by score
        self.assertGreaterEqual(merged[0].score, merged[1].score)
    
    def test_error_handling(self):
        """Test error handling in parallel execution."""
        # Make one retriever raise an exception
        self.mock_retrievers['vector'].retrieve.side_effect = Exception("Retrieval error")
        self.mock_retrievers['metadata'].retrieve.return_value = [Mock()]
        
        # Execute with error handling
        results = self.executor.execute_parallel(
            "test query", 
            ['vector', 'metadata'],
            handle_errors=True
        )
        
        # Should return results from successful retrievers only
        self.assertNotIn('vector', results)
        self.assertIn('metadata', results)
    
    def test_timeout_handling(self):
        """Test timeout handling for slow retrievers."""
        import time
        
        # Make one retriever slow
        def slow_retrieve(query, **kwargs):
            time.sleep(1)
            return [Mock()]
        
        self.mock_retrievers['vector'].retrieve.side_effect = slow_retrieve
        self.mock_retrievers['metadata'].retrieve.return_value = [Mock()]
        
        # Execute with short timeout
        start_time = time.time()
        results = self.executor.execute_parallel(
            "test query",
            ['vector', 'metadata'],
            timeout=0.5
        )
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time
        self.assertLess(execution_time, 2.0)


class TestFastMetadataIndex(unittest.TestCase):
    """Test FastMetadataIndex functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.index = FastMetadataIndex(self.temp_dir)
        
        # Create sample metadata
        self.sample_metadata = [
            {
                'id': 'doc_1',
                'province': 'กรุงเทพมหานคร',
                'district': 'บางกะปิ',
                'deed_type': 'โฉนดที่ดิน',
                'area_total_sqm': 1600
            },
            {
                'id': 'doc_2',
                'province': 'เชียงใหม่',
                'district': 'เมือง',
                'deed_type': 'โฉนดที่ดิน',
                'area_total_sqm': 3200
            },
            {
                'id': 'doc_3',
                'province': 'กรุงเทพมหานคร',
                'district': 'ห้วยขวาง',
                'deed_type': 'ใบอนุญาต',
                'area_total_sqm': 800
            }
        ]
    
    def test_build_index(self):
        """Test building metadata index."""
        self.index.build_index(self.sample_metadata)
        
        # Verify index was built
        self.assertTrue(self.index.is_built())
        
        # Check province index
        bangkok_docs = self.index.get_by_province('กรุงเทพมหานคร')
        self.assertEqual(len(bangkok_docs), 2)
        
        # Check deed type index
        chanote_docs = self.index.get_by_deed_type('โฉนดที่ดิน')
        self.assertEqual(len(chanote_docs), 2)
    
    def test_range_queries(self):
        """Test range queries on numeric fields."""
        self.index.build_index(self.sample_metadata)
        
        # Test area range query
        large_properties = self.index.get_by_area_range(min_area=2000)
        self.assertEqual(len(large_properties), 1)
        self.assertEqual(large_properties[0]['id'], 'doc_2')
        
        # Test area range with max
        medium_properties = self.index.get_by_area_range(
            min_area=1000, 
            max_area=2000
        )
        self.assertEqual(len(medium_properties), 1)
        self.assertEqual(medium_properties[0]['id'], 'doc_1')
    
    def test_compound_queries(self):
        """Test compound metadata queries."""
        self.index.build_index(self.sample_metadata)
        
        # Query: Bangkok + โฉนดที่ดิน
        results = self.index.query_compound({
            'province': 'กรุงเทพมหานคร',
            'deed_type': 'โฉนดที่ดิน'
        })
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['id'], 'doc_1')
    
    def test_save_and_load_index(self):
        """Test saving and loading index to/from disk."""
        self.index.build_index(self.sample_metadata)
        
        # Save index
        index_file = self.index.save_index()
        self.assertTrue(Path(index_file).exists())
        
        # Create new index and load
        new_index = FastMetadataIndex(self.temp_dir)
        new_index.load_index()
        
        # Verify loaded index works
        self.assertTrue(new_index.is_built())
        bangkok_docs = new_index.get_by_province('กรุงเทพมหานคร')
        self.assertEqual(len(bangkok_docs), 2)
    
    def test_update_index(self):
        """Test updating index with new metadata."""
        self.index.build_index(self.sample_metadata[:2])
        
        # Add new document
        new_doc = {
            'id': 'doc_4',
            'province': 'ภูเก็ต',
            'district': 'เมือง',
            'deed_type': 'โฉนดที่ดิน',
            'area_total_sqm': 2400
        }
        
        self.index.update_index([new_doc])
        
        # Verify update
        phuket_docs = self.index.get_by_province('ภูเก็ต')
        self.assertEqual(len(phuket_docs), 1)
        self.assertEqual(phuket_docs[0]['id'], 'doc_4')


class TestIndexClassifier(unittest.TestCase):
    """Test IndexClassifier functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = IndexClassifier()
    
    def test_classify_query_type(self):
        """Test query type classification."""
        test_cases = [
            ("โฉนดที่ดินเลขที่ 12345", "specific"),
            ("สรุปที่ดินทั้งหมด", "summary"),
            ("ที่ดินที่คล้ายกัน", "similarity"),
            ("province:กรุงเทพมหานคร", "filter"),
            ("ที่ดินในกรุงเทพขนาดใหญ่", "complex")
        ]
        
        for query, expected_type in test_cases:
            result = self.classifier.classify_query(query)
            self.assertEqual(result['query_type'], expected_type)
    
    def test_recommend_strategy(self):
        """Test strategy recommendation based on query."""
        test_cases = [
            ("โฉนดที่ดินเลขที่ 12345", "metadata"),
            ("สรุปที่ดินทั้งหมด", "summary"),
            ("ที่ดินที่คล้ายกัน", "vector"),
            ("province:กรุงเทพมหานคร area>1000", "metadata"),
            ("ที่ดินในกรุงเทพขนาดใหญ่ใกล้รถไฟฟ้า", "hybrid")
        ]
        
        for query, expected_strategy in test_cases:
            recommendation = self.classifier.recommend_strategy(query)
            self.assertEqual(recommendation['primary_strategy'], expected_strategy)
    
    def test_detect_thai_patterns(self):
        """Test detection of Thai language patterns."""
        thai_queries = [
            "โฉนดที่ดินในกรุงเทพมหานคร",
            "ที่ดินขนาด 2 ไร่",
            "ประเภทการใช้ที่ดิน"
        ]
        
        english_queries = [
            "land deed in Bangkok",
            "property size 2 rai",
            "land use type"
        ]
        
        for query in thai_queries:
            analysis = self.classifier._analyze_language(query)
            self.assertTrue(analysis['has_thai'])
        
        for query in english_queries:
            analysis = self.classifier._analyze_language(query)
            self.assertFalse(analysis['has_thai'])
    
    def test_complexity_assessment(self):
        """Test query complexity assessment."""
        simple_queries = [
            "โฉนดที่ดินเลขที่ 12345",
            "province:กรุงเทพมหานคร"
        ]
        
        complex_queries = [
            "ที่ดินในกรุงเทพขนาดใหญ่ใกล้รถไฟฟ้าที่มีศักยภาพเพิ่มมูลค่า",
            "properties in Bangkok over 2 rai with good transportation access"
        ]
        
        for query in simple_queries:
            complexity = self.classifier._assess_complexity(query)
            self.assertLess(complexity['score'], 0.5)
        
        for query in complex_queries:
            complexity = self.classifier._assess_complexity(query)
            self.assertGreater(complexity['score'], 0.5)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete retrieval pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock indexes
        self.mock_indexes = {
            'vector': Mock(),
            'summary': Mock(),
            'recursive': Mock()
        }
        
        # Create mock metadata
        self.sample_metadata = [
            {
                'id': 'doc_1',
                'text': 'โฉนดที่ดินในกรุงเทพมหานคร บางกะปิ',
                'province': 'กรุงเทพมหานคร',
                'district': 'บางกะปิ',
                'deed_type': 'โฉนดที่ดิน'
            },
            {
                'id': 'doc_2',
                'text': 'โฉนดที่ดินในเชียงใหม่ เมือง',
                'province': 'เชียงใหม่',
                'district': 'เมือง',
                'deed_type': 'โฉนดที่ดิน'
            }
        ]
    
    @patch.multiple(
        'retrieval.retrievers.vector',
        VectorStoreIndex=Mock()
    )
    def test_complete_retrieval_workflow(self):
        """Test the complete retrieval workflow."""
        # 1. Initialize components
        fast_metadata = FastMetadataIndex(self.temp_dir)
        fast_metadata.build_index(self.sample_metadata)
        
        cache = RetrievalCache(max_size=100, ttl_seconds=300)
        
        # 2. Create retrievers
        vector_retriever = VectorRetrieverAdapter(self.mock_indexes['vector'])
        metadata_retriever = MetadataRetrieverAdapter(self.mock_indexes['vector'])
        
        retrievers = {
            'vector': vector_retriever,
            'metadata': metadata_retriever
        }
        
        # 3. Setup router
        router = iLandQueryRouter(retrievers)
        
        # 4. Setup parallel executor
        executor = ParallelRetrieverExecutor(retrievers)
        
        # 5. Mock retriever responses
        mock_nodes = []
        for i in range(2):
            node = Mock()
            node.node = Mock()
            node.node.metadata = self.sample_metadata[i]
            node.score = 0.8 - (i * 0.1)
            mock_nodes.append(node)
        
        vector_retriever.mock_index.as_retriever.return_value.retrieve.return_value = mock_nodes
        metadata_retriever.mock_index.as_retriever.return_value.retrieve.return_value = mock_nodes[:1]
        
        # 6. Test single retrieval
        query = "โฉนดที่ดินในกรุงเทพมหานคร"
        
        # Check cache miss
        cached_result = cache.get(query)
        self.assertIsNone(cached_result)
        
        # Execute query through router
        result = router.execute_query(query)
        
        # Cache result
        cache.set(query, result)
        
        # Verify result
        self.assertGreater(len(result), 0)
        
        # Check cache hit
        cached_result = cache.get(query)
        self.assertIsNotNone(cached_result)
        
        # 7. Test parallel execution
        parallel_results = executor.execute_parallel(
            query, 
            ['vector', 'metadata']
        )
        
        self.assertIn('vector', parallel_results)
        self.assertIn('metadata', parallel_results)
        
        # 8. Test metadata filtering
        bangkok_docs = fast_metadata.get_by_province('กรุงเทพมหานคร')
        self.assertEqual(len(bangkok_docs), 1)
        self.assertEqual(bangkok_docs[0]['id'], 'doc_1')
    
    def test_query_routing_integration(self):
        """Test query routing integration with different query types."""
        # Setup components
        retrievers = {
            'vector': VectorRetrieverAdapter(self.mock_indexes['vector']),
            'metadata': MetadataRetrieverAdapter(self.mock_indexes['vector']),
            'summary': SummaryRetrieverAdapter(self.mock_indexes['summary'])
        }
        
        router = iLandQueryRouter(retrievers)
        classifier = IndexClassifier()
        
        # Test different query types
        test_queries = [
            ("โฉนดที่ดินเลขที่ 12345", "metadata"),
            ("สรุปที่ดินในกรุงเทพ", "summary"),
            ("ที่ดินที่คล้ายกัน", "vector")
        ]
        
        for query, expected_strategy in test_queries:
            # Classify query
            classification = classifier.classify_query(query)
            recommendation = classifier.recommend_strategy(query)
            
            # Route query
            routed_strategy = router.route_query(query)
            
            # Verify routing is consistent with classification
            self.assertIn(routed_strategy, [expected_strategy, recommendation['primary_strategy']])
    
    def test_caching_integration(self):
        """Test caching integration with retrieval pipeline."""
        cache = RetrievalCache(max_size=10, ttl_seconds=300)
        
        # Create retriever with cache integration
        class CachedVectorRetriever(VectorRetrieverAdapter):
            def __init__(self, index, cache):
                super().__init__(index)
                self.cache = cache
            
            def retrieve(self, query, **kwargs):
                # Check cache first
                cached = self.cache.get(query)
                if cached:
                    return cached
                
                # Execute retrieval
                result = super().retrieve(query, **kwargs)
                
                # Cache result
                self.cache.set(query, result)
                
                return result
        
        # Test cached retrieval
        cached_retriever = CachedVectorRetriever(
            self.mock_indexes['vector'], 
            cache
        )
        
        # Mock retriever response
        mock_nodes = [Mock(), Mock()]
        cached_retriever.mock_index.as_retriever.return_value.retrieve.return_value = mock_nodes
        
        query = "test caching query"
        
        # First call - should hit index
        result1 = cached_retriever.retrieve(query)
        
        # Second call - should hit cache
        result2 = cached_retriever.retrieve(query)
        
        # Results should be the same
        self.assertEqual(len(result1), len(result2))
        
        # Verify cache statistics
        stats = cache.get_statistics()
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestBaseRetrieverAdapter,
        TestVectorRetrieverAdapter,
        TestMetadataRetrieverAdapter,
        TestHybridRetrieverAdapter,
        TestSummaryRetrieverAdapter,
        TestRecursiveRetrieverAdapter,
        TestChunkDecouplingRetrieverAdapter,
        TestSectionRetrieverAdapter,
        TestiLandQueryRouter,
        TestRetrievalCache,
        TestParallelRetrieverExecutor,
        TestFastMetadataIndex,
        TestIndexClassifier,
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
    print(f"TEST SUMMARY - Retrieval Module")
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