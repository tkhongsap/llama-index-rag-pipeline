import importlib.util
from pathlib import Path
import pytest
from dataclasses import dataclass

# Skip entire module if llama_index is not available
if importlib.util.find_spec("llama_index") is None:
    pytest.skip("llama_index not installed", allow_module_level=True)

# Load retrieval modules from src-iLand
BASE_DIR = Path(__file__).resolve().parents[1] / 'src-iLand' / 'retrieval'

# Load hybrid adapter
hybrid_spec = importlib.util.spec_from_file_location('hybrid', BASE_DIR / 'retrievers' / 'hybrid.py')
hybrid = importlib.util.module_from_spec(hybrid_spec)
hybrid_spec.loader.exec_module(hybrid)

# Load router
router_spec = importlib.util.spec_from_file_location('router', BASE_DIR / 'router.py')
router = importlib.util.module_from_spec(router_spec)
router_spec.loader.exec_module(router)


def test_extract_thai_keywords():
    class DummyHybrid(hybrid.HybridRetrieverAdapter):
        def __init__(self):
            # Skip parent initialisation
            self.strategy_name = 'hybrid'
    adapter = DummyHybrid()
    keywords = adapter._extract_thai_keywords('โฉนดที่ดินในกรุงเทพ Bangkok test')
    assert 'โฉนดที่ดินในกรุงเทพ' in keywords
    assert 'bangkok' in keywords
    assert 'โฉนด' in keywords


def test_combine_scores():
    @dataclass
    class SimpleNode:
        text: str
    @dataclass
    class NodeWithScore:
        node: SimpleNode
        score: float

    class DummyHybrid(hybrid.HybridRetrieverAdapter):
        def __init__(self):
            self.strategy_name = 'hybrid'

    adapter = DummyHybrid()
    vector_nodes = [NodeWithScore(SimpleNode('a'), 0.8), NodeWithScore(SimpleNode('b'), 0.5)]
    keyword_nodes = [NodeWithScore(SimpleNode('a'), 0.2), NodeWithScore(SimpleNode('b'), 0.1)]
    combined = adapter._combine_scores(vector_nodes, keyword_nodes, alpha=0.6)
    assert len(combined) == 2
    assert combined[0].score == pytest.approx(0.6 * 0.8 + 0.4 * 0.2)
    assert combined[1].score == pytest.approx(0.6 * 0.5 + 0.4 * 0.1)


def test_round_robin_strategy_selection():
    class DummyRouter(router.iLandRouterRetriever):
        def __init__(self):
            # Only setup state needed for round robin selection
            self._strategy_round_robin_state = {}
            self._default_strategy = 'vector'
        def _setup_models(self):
            pass
    dummy_router = DummyRouter()
    available = ['vector', 'hybrid']
    first = dummy_router._select_strategy_round_robin('index', available)
    second = dummy_router._select_strategy_round_robin('index', available)
    third = dummy_router._select_strategy_round_robin('index', available)
    assert first['strategy'] == 'vector'
    assert second['strategy'] == 'hybrid'
    assert third['strategy'] == 'vector'
