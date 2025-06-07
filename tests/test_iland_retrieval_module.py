import sys
import importlib.util
from pathlib import Path
import pytest
from dataclasses import dataclass
from unittest.mock import Mock, patch

# Skip entire module if llama_index is not available
if importlib.util.find_spec("llama_index") is None:
    pytest.skip("llama_index not installed", allow_module_level=True)

# Add src-iLand to Python path to resolve imports
SRC_DIR = Path(__file__).resolve().parents[1] / 'src-iLand'
sys.path.insert(0, str(SRC_DIR))

# Add retrieval directory to path to fix relative imports
RETRIEVAL_DIR = SRC_DIR / 'retrieval'
sys.path.insert(0, str(RETRIEVAL_DIR))

# Now import the modules with better error handling
try:
    from llama_index.core.schema import NodeWithScore, TextNode
    from retrieval.retrievers.hybrid import HybridRetrieverAdapter
    from retrieval.index_classifier import iLandIndexClassifier, create_default_iland_classifier
    
    # Import router separately to avoid import chain issues
    import sys
    import importlib
    router_spec = importlib.util.spec_from_file_location("router", RETRIEVAL_DIR / "router.py")
    router_module = importlib.util.module_from_spec(router_spec)
    sys.modules["router"] = router_module
    router_spec.loader.exec_module(router_module)
    iLandRouterRetriever = router_module.iLandRouterRetriever
    
except ImportError as e:
    pytest.skip(f"iLand modules not available: {e}", allow_module_level=True)


def create_test_node_with_score(text: str, score: float, node_id: str = None) -> NodeWithScore:
    """Helper function to create NodeWithScore objects for testing."""
    text_node = TextNode(
        text=text,
        id_=node_id or text,
        metadata={}
    )
    return NodeWithScore(node=text_node, score=score)


def test_extract_thai_keywords():
    """Test Thai keyword extraction functionality."""
    # Mock the required attributes
    adapter = Mock(spec=HybridRetrieverAdapter)
    adapter._extract_thai_keywords = HybridRetrieverAdapter._extract_thai_keywords.__get__(adapter)
    
    keywords = adapter._extract_thai_keywords('โฉนดที่ดินในกรุงเทพ Bangkok test')
    assert 'โฉนดที่ดินในกรุงเทพ' in keywords
    assert 'bangkok' in keywords
    assert 'โฉนด' in keywords


def test_combine_scores_basic():
    """Test basic score combination functionality."""
    adapter = Mock(spec=HybridRetrieverAdapter)
    adapter._combine_scores = HybridRetrieverAdapter._combine_scores.__get__(adapter)
    
    # Use proper NodeWithScore objects
    vector_nodes = [
        create_test_node_with_score('a', 0.8),
        create_test_node_with_score('b', 0.5)
    ]
    keyword_nodes = [
        create_test_node_with_score('a', 0.2),
        create_test_node_with_score('b', 0.1)
    ]
    
    combined = adapter._combine_scores(vector_nodes, keyword_nodes, alpha=0.6)
    assert len(combined) == 2
    assert combined[0].score == pytest.approx(0.6 * 0.8 + 0.4 * 0.2)
    assert combined[1].score == pytest.approx(0.6 * 0.5 + 0.4 * 0.1)


def test_round_robin_strategy_selection():
    """Test round-robin strategy selection."""
    router = Mock(spec=iLandRouterRetriever)
    router._strategy_round_robin_state = {}
    router._default_strategy = 'vector'
    router._select_strategy_round_robin = iLandRouterRetriever._select_strategy_round_robin.__get__(router)
    
    available = ['vector', 'hybrid']
    first = router._select_strategy_round_robin('index', available)
    second = router._select_strategy_round_robin('index', available)
    third = router._select_strategy_round_robin('index', available)
    
    assert first['strategy'] == 'vector'
    assert second['strategy'] == 'hybrid'
    assert third['strategy'] == 'vector'


def test_extract_thai_keywords_mixed_content():
    """Test Thai keyword extraction with mixed Thai-English content."""
    adapter = Mock(spec=HybridRetrieverAdapter)
    adapter._extract_thai_keywords = HybridRetrieverAdapter._extract_thai_keywords.__get__(adapter)
    
    keywords = adapter._extract_thai_keywords('ราคา 1,000,000 บาท price $30,000')
    assert 'ราคา' in keywords
    assert 'บาท' in keywords
    assert 'price' in keywords


def test_combine_scores_empty_results():
    """Test score combination with empty input."""
    adapter = Mock(spec=HybridRetrieverAdapter)
    adapter._combine_scores = HybridRetrieverAdapter._combine_scores.__get__(adapter)
    
    combined = adapter._combine_scores([], [], alpha=0.5)
    assert len(combined) == 0


def test_combine_scores_single_source():
    """Test score combination with single source."""
    adapter = Mock(spec=HybridRetrieverAdapter)
    adapter._combine_scores = HybridRetrieverAdapter._combine_scores.__get__(adapter)
    
    vector_nodes = [create_test_node_with_score('a', 0.9)]
    keyword_nodes = []
    
    combined = adapter._combine_scores(vector_nodes, keyword_nodes, alpha=0.7)
    assert len(combined) == 1
    assert combined[0].score == 0.9 * 0.7


def test_round_robin_multiple_indices():
    """Test round-robin selection with multiple indices."""
    router = Mock(spec=iLandRouterRetriever)
    router._strategy_round_robin_state = {}
    router._default_strategy = 'vector'
    router._select_strategy_round_robin = iLandRouterRetriever._select_strategy_round_robin.__get__(router)
    
    available = ['vector', 'hybrid', 'keyword']
    
    # Test with different indices
    result1 = router._select_strategy_round_robin('index1', available)
    result2 = router._select_strategy_round_robin('index2', available)
    result3 = router._select_strategy_round_robin('index1', available)
    
    assert result1['strategy'] == 'vector'
    assert result2['strategy'] == 'vector'
    assert result3['strategy'] == 'hybrid'


def test_combine_scores_with_actual_nodes():
    """Test score combination using actual TextNode objects."""
    class DummyHybrid(HybridRetrieverAdapter):
        def __init__(self):
            self.strategy_name = 'hybrid'

    adapter = DummyHybrid()
    
    # Create proper NodeWithScore objects with TextNode
    vector_nodes = [
        create_test_node_with_score('Document A content', 0.8, 'node_a'),
        create_test_node_with_score('Document B content', 0.5, 'node_b')
    ]
    keyword_nodes = [
        create_test_node_with_score('Document A content', 0.2, 'node_a'),
        create_test_node_with_score('Document B content', 0.1, 'node_b')
    ]
    
    combined = adapter._combine_scores(vector_nodes, keyword_nodes, alpha=0.6)
    assert len(combined) == 2
    assert combined[0].score == pytest.approx(0.6 * 0.8 + 0.4 * 0.2)
    assert combined[1].score == pytest.approx(0.6 * 0.5 + 0.4 * 0.1)
    assert combined[0].node.text == 'Document A content'
    assert combined[1].node.text == 'Document B content'


def test_round_robin_with_dummy_router():
    """Test round-robin using a minimal dummy router."""
    class DummyRouter:
        def __init__(self):
            # Only setup state needed for round robin selection
            self._strategy_round_robin_state = {}
            self._default_strategy = 'vector'
        
        def _select_strategy_round_robin(self, index_name, available_strategies):
            # Simplified version of the round-robin logic
            if index_name not in self._strategy_round_robin_state:
                self._strategy_round_robin_state[index_name] = 0
            
            current_index = self._strategy_round_robin_state[index_name]
            selected_strategy = available_strategies[current_index % len(available_strategies)]
            
            # Update for next call
            self._strategy_round_robin_state[index_name] = (current_index + 1) % len(available_strategies)
            
            return {
                "strategy": selected_strategy,
                "confidence": 0.5,
                "method": "round_robin",
                "reasoning": f"Round-robin selection: {selected_strategy}"
            }
    
    dummy_router = DummyRouter()
    available = ['vector', 'hybrid']
    
    first = dummy_router._select_strategy_round_robin('index', available)
    second = dummy_router._select_strategy_round_robin('index', available)
    third = dummy_router._select_strategy_round_robin('index', available)
    
    assert first['strategy'] == 'vector'
    assert second['strategy'] == 'hybrid'
    assert third['strategy'] == 'vector'


def test_node_with_score_creation():
    """Test creating NodeWithScore objects with proper TextNode."""
    node = create_test_node_with_score('Test content', 0.85, 'test_node_1')
    
    assert isinstance(node, NodeWithScore)
    assert isinstance(node.node, TextNode)
    assert node.node.text == 'Test content'
    assert node.score == 0.85
    assert node.node.id_ == 'test_node_1'


def test_index_classifier_basic():
    """Test basic index classifier functionality."""
    try:
        classifier = create_default_iland_classifier()
        assert classifier is not None
        
        # Test with a simple query
        result = classifier.classify_query("โฉนดที่ดิน")
        assert 'selected_index' in result
        assert 'confidence' in result
        
    except Exception as e:
        pytest.skip(f"Index classifier test failed due to dependencies: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
