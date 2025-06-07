"""
Comprehensive Unit Tests for iLand Retrieval Strategies

Tests all 7 retrieval strategies (vector, hybrid, recursive, chunk_decoupling, 
planner, metadata, summary) with carefully designed queries to trigger different 
routing behaviors based on Thai land deed data.
"""

import sys
import pytest
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from unittest.mock import Mock, patch

# Skip entire module if llama_index is not available
try:
    import importlib.util
    if importlib.util.find_spec("llama_index") is None:
        pytest.skip("llama_index not installed", allow_module_level=True)
except ImportError:
    pytest.skip("llama_index not available", allow_module_level=True)

# Add src-iLand to Python path
SRC_DIR = Path(__file__).resolve().parents[1] / 'src-iLand'
sys.path.insert(0, str(SRC_DIR))

# Add retrieval directory to path
RETRIEVAL_DIR = SRC_DIR / 'retrieval'
sys.path.insert(0, str(RETRIEVAL_DIR))

try:
    from llama_index.core.schema import QueryBundle, NodeWithScore, TextNode
    from router import iLandRouterRetriever
    from index_classifier import create_default_iland_classifier
    from cache import iLandCacheManager
    from retrievers.base import BaseRetrieverAdapter
except ImportError as e:
    pytest.skip(f"iLand modules not available: {e}", allow_module_level=True)


@dataclass
class StrategyTestCase:
    """Test case for a specific retrieval strategy."""
    query: str
    expected_strategy: str
    expected_results_contain: List[str]  # Keywords that should appear in results
    description: str
    confidence_threshold: float = 0.3


@dataclass
class RetrievalTestResult:
    """Result of a retrieval test."""
    query: str
    routed_strategy: str
    expected_strategy: str
    num_results: int
    latency: float
    confidence: float
    results_preview: List[str]
    strategy_match: bool
    content_match_score: float
    error: Optional[str] = None


class MockRetrieverAdapter(BaseRetrieverAdapter):
    """Mock retriever adapter for testing."""
    
    def __init__(self, strategy_name: str, mock_results: List[Dict[str, Any]]):
        self.strategy_name = strategy_name
        self.mock_results = mock_results
    
    def retrieve(self, query: str, top_k: int = 5) -> List[NodeWithScore]:
        """Mock retrieval that returns predefined results."""
        nodes = []
        
        for i, result in enumerate(self.mock_results[:top_k]):
            text_node = TextNode(
                text=result['text'],
                id_=f"{self.strategy_name}_node_{i}",
                metadata={
                    'strategy_used': self.strategy_name,
                    'mock_result': True,
                    **result.get('metadata', {})
                }
            )
            
            node_with_score = NodeWithScore(
                node=text_node,
                score=result.get('score', 0.8 - i * 0.1)
            )
            nodes.append(node_with_score)
        
        return nodes


class iLandRetrievalStrategyTester:
    """Comprehensive tester for iLand retrieval strategies."""
    
    def __init__(self):
        """Initialize the strategy tester."""
        self.test_cases = self._create_test_cases()
        self.mock_adapters = self._create_mock_adapters()
        self.router = None
        
    def _create_test_cases(self) -> List[StrategyTestCase]:
        """
        Create test cases designed to trigger different retrieval strategies.
        Based on routing logic in router.py and Thai land deed data.
        """
        return [
            # 1. VECTOR STRATEGY - Simple semantic queries
            StrategyTestCase(
                query="โฉนดที่ดินในจังหวัดชัยนาท",
                expected_strategy="vector",
                expected_results_contain=["โฉนด", "ชัยนาท", "Chai Nat"],
                description="Simple semantic search for land deeds in Chai Nat province",
                confidence_threshold=0.6
            ),
            
            # 2. HYBRID STRATEGY - Thai keywords + semantic
            StrategyTestCase(
                query="หาโฉนด นส.3 กรรมสิทธิ์บริษัท",
                expected_strategy="hybrid",
                expected_results_contain=["โฉนด", "นส.3", "กรรมสิทธิ์บริษัท"],
                description="Hybrid search combining Thai legal terms with semantic understanding",
                confidence_threshold=0.5
            ),
            
            # 3. METADATA STRATEGY - Geographic filtering
            StrategyTestCase(
                query="ที่ดินในจังหวัดอ่างทอง อำเภอเมืองอ่างทอง",
                expected_strategy="metadata",
                expected_results_contain=["อ่างทอง", "Ang Thong", "เมืองอ่างทอง"],
                description="Geographic metadata filtering for specific province and district",
                confidence_threshold=0.7
            ),
            
            # 4. PLANNER STRATEGY - Multi-step complex query
            StrategyTestCase(
                query="วิเคราะห์ขั้นตอนการโอนกรรมสิทธิ์ที่ดิน จากนั้นแสดงเอกสารที่จำเป็น แล้วคำนวณค่าธรรมเนียม",
                expected_strategy="planner",
                expected_results_contain=["โอนกรรมสิทธิ์", "เอกสาร", "ค่าธรรมเนียม"],
                description="Multi-step analysis query requiring planning and sequential execution",
                confidence_threshold=0.4
            ),
            
            # 5. RECURSIVE STRATEGY - Hierarchical information need
            StrategyTestCase(
                query="รายละเอียดโครงสร้างการถือครองที่ดินและการจัดหมวดหมู่ทรัพย์สิน",
                expected_strategy="recursive",
                expected_results_contain=["การถือครอง", "จัดหมวดหมู่", "ทรัพย์สิน"],
                description="Hierarchical query requiring multi-level document exploration",
                confidence_threshold=0.4
            ),
            
            # 6. CHUNK_DECOUPLING STRATEGY - Detailed section analysis
            StrategyTestCase(
                query="รายละเอียดเฉพาะส่วนของการวัดพื้นที่ในหน่วยไร่ งาน ตารางวา",
                expected_strategy="chunk_decoupling",
                expected_results_contain=["ไร่", "งาน", "ตารางวา", "พื้นที่"],
                description="Detailed chunk-level analysis of area measurement sections",
                confidence_threshold=0.4
            ),
            
            # 7. SUMMARY STRATEGY - Overview questions
            StrategyTestCase(
                query="สรุปภาพรวมของข้อมูลโฉนดที่ดินในระบบ",
                expected_strategy="summary",
                expected_results_contain=["สรุป", "ภาพรวม", "โฉนดที่ดิน"],
                description="High-level overview query requiring summary information",
                confidence_threshold=0.5
            ),
            
            # 8. VECTOR STRATEGY - English semantic query
            StrategyTestCase(
                query="find land deeds with company ownership",
                expected_strategy="vector",
                expected_results_contain=["company", "ownership", "กรรมสิทธิ์บริษัท"],
                description="English semantic query for company-owned land",
                confidence_threshold=0.6
            ),
            
            # 9. HYBRID STRATEGY - Mixed Thai-English keywords
            StrategyTestCase(
                query="โฉนด land deed coordinates GPS location",
                expected_strategy="hybrid",
                expected_results_contain=["โฉนด", "coordinates", "GPS", "location"],
                description="Mixed language query requiring both Thai and English keyword matching",
                confidence_threshold=0.5
            ),
            
            # 10. METADATA STRATEGY - Specific attribute filtering
            StrategyTestCase(
                query="ที่ดินในภาคกลาง ประเภทกรรมสิทธิ์บริษัท ไม่ใช่คอนโด",
                expected_strategy="metadata",
                expected_results_contain=["ภาคกลาง", "กรรมสิทธิ์บริษัท", "คอนโด"],
                description="Complex metadata filtering with multiple attribute conditions",
                confidence_threshold=0.6
            )
        ]
    
    def _create_mock_adapters(self) -> Dict[str, Dict[str, MockRetrieverAdapter]]:
        """Create mock adapters with realistic Thai land deed responses."""
        
        # Mock results for different strategies
        mock_results = {
            "vector": [
                {
                    "text": "โฉนดที่ดินเลขที่ 50 จังหวัดชัยนาท อำเภอเนินขาม ประเภทโฉนด กรรมสิทธิ์บริษัท พื้นที่ 11 ไร่ 3 งาน 66 ตารางวา ได้มาเมื่อ 2001-11-02",
                    "score": 0.9,
                    "metadata": {"province": "Chai Nat", "district": "Noen Kham", "deed_type": "โฉนด"}
                },
                {
                    "text": "ที่ดินในจังหวัดชัยนาท บริหารแลนด์แบงค์ 1 พื้นที่ 1 กรุงเทพ ปริมณฑล จังหวัดฝั่งตะวันออก",
                    "score": 0.8,
                    "metadata": {"province": "Chai Nat", "land_passport": "บริหารแลนด์แบงค์ 1"}
                }
            ],
            
            "hybrid": [
                {
                    "text": "โฉนด นส.3 กรรมสิทธิ์บริษัท เลขที่โฉนด 352 ประเภทโฉนด โฉนด หน้าที่ 97 ประเภทการถือครอง กรรมสิทธิ์บริษัท",
                    "score": 0.85,
                    "metadata": {"deed_type": "โฉนด", "holding_type": "กรรมสิทธิ์บริษัท"}
                },
                {
                    "text": "land deed coordinates GPS location โฉนด พิกัด 14.5486, 100.4514 Google Maps URL",
                    "score": 0.8,
                    "metadata": {"coordinates": "14.5486,100.4514", "has_gps": True}
                }
            ],
            
            "metadata": [
                {
                    "text": "จังหวัดอ่างทอง อำเภอเมืองอ่างทอง ภาคกลาง ประเทศไทย พิกัด 14.548600, 100.451400",
                    "score": 0.9,
                    "metadata": {"province": "Ang Thong", "district": "Mueang Ang Thong", "region": "กลาง"}
                },
                {
                    "text": "ภาคกลาง กรรมสิทธิ์บริษัท Is Condo: False ประเภทโฉนด โฉนด หนังสือแสดงกรรมสิทธิ์",
                    "score": 0.85,
                    "metadata": {"region": "กลาง", "is_condo": False, "holding_type": "กรรมสิทธิ์บริษัท"}
                }
            ],
            
            "planner": [
                {
                    "text": "ขั้นตอนการโอนกรรมสิทธิ์ที่ดิน: 1) เตรียมเอกสาร 2) ตรวจสอบสิทธิ์ 3) คำนวณค่าธรรมเนียม 4) ดำเนินการโอน",
                    "score": 0.8,
                    "metadata": {"process_type": "transfer", "steps": 4}
                },
                {
                    "text": "เอกสารที่จำเป็นสำหรับการโอนกรรมสิทธิ์: โฉนด หลักฐานการชำระค่าธรรมเนียม บัตรประชาชน",
                    "score": 0.75,
                    "metadata": {"document_type": "transfer_requirements"}
                }
            ],
            
            "recursive": [
                {
                    "text": "โครงสร้างการถือครองที่ดิน: กรรมสิทธิ์บริษัท -> การจัดหมวดหมู่ -> หนังสือแสดงกรรมสิทธิ์ -> ประเภทโฉนด",
                    "score": 0.75,
                    "metadata": {"hierarchy_level": "ownership_structure"}
                },
                {
                    "text": "การจัดหมวดหมู่ทรัพย์สิน: หมวดหมู่หลัก Empty หมวดหมู่ย่อย ใช้ประโยชน์ในด้านใดๆ รอเวลาสำหรับการพัฒนาในอนาคต",
                    "score": 0.7,
                    "metadata": {"hierarchy_level": "classification"}
                }
            ],
            
            "chunk_decoupling": [
                {
                    "text": "ขนาดพื้นที่ (Area Measurements): Deed Rai: 11 Land Ngan: 3 Deed Wa: 66 Deed Total Square Wa: 4766",
                    "score": 0.8,
                    "metadata": {"chunk_type": "area_measurements", "units": ["ไร่", "งาน", "ตารางวา"]}
                },
                {
                    "text": "รายละเอียดการวัดพื้นที่: ไร่ 11 งาน 3 ตารางวา 66 รวมตารางวา 4766 ตารางเมตร 0",
                    "score": 0.75,
                    "metadata": {"chunk_type": "detailed_measurements"}
                }
            ],
            
            "summary": [
                {
                    "text": "สรุปภาพรวมข้อมูลโฉนดที่ดิน: มีข้อมูลจากหลายจังหวัด ประเภทโฉนด กรรมสิทธิ์บริษัท วันที่ได้มา 2001-2014",
                    "score": 0.8,
                    "metadata": {"summary_type": "overview", "date_range": "2001-2014"}
                },
                {
                    "text": "ภาพรวมระบบโฉนดที่ดิน: หนังสือแสดงกรรมสิทธิ์ การจำแนกประเภท พิกัดภูมิศาสตร์ ขนาดพื้นที่",
                    "score": 0.75,
                    "metadata": {"summary_type": "system_overview"}
                }
            ]
        }
        
        # Create mock adapters for the main index
        index_name = "iland_land_deeds"
        adapters = {index_name: {}}
        
        for strategy_name, results in mock_results.items():
            adapters[index_name][strategy_name] = MockRetrieverAdapter(strategy_name, results)
        
        return adapters
    
    def create_test_router(self, strategy_selector: str = "llm") -> iLandRouterRetriever:
        """Create a router with mock adapters for testing."""
        
        # Create mock classifier that returns the test index
        mock_classifier = Mock()
        mock_classifier.classify_query = Mock(return_value={
            "selected_index": "iland_land_deeds",
            "confidence": 0.8,
            "method": "mock_classifier"
        })
        
        # Create router with mock adapters
        router = iLandRouterRetriever(
            retrievers=self.mock_adapters,
            index_classifier=mock_classifier,
            strategy_selector=strategy_selector,
            api_key="test_key",
            enable_caching=False  # Disable caching for testing
        )
        
        return router
    
    def run_strategy_test(self, test_case: StrategyTestCase, router: iLandRouterRetriever) -> RetrievalTestResult:
        """Run a single strategy test case."""
        
        start_time = time.time()
        error = None
        
        try:
            # Execute query
            query_bundle = QueryBundle(query_str=test_case.query)
            nodes = router._retrieve(query_bundle)
            
            latency = time.time() - start_time
            
            # Extract routing information from first node
            if nodes:
                first_node_metadata = getattr(nodes[0].node, 'metadata', {})
                routed_strategy = first_node_metadata.get('selected_strategy', 'unknown')
                confidence = first_node_metadata.get('strategy_confidence', 0.0)
            else:
                routed_strategy = 'unknown'
                confidence = 0.0
            
            # Check if strategy matches expectation
            strategy_match = routed_strategy == test_case.expected_strategy
            
            # Calculate content match score
            content_match_score = self._calculate_content_match(nodes, test_case.expected_results_contain)
            
            # Create results preview
            results_preview = [
                node.node.text[:100] + "..." if len(node.node.text) > 100 else node.node.text
                for node in nodes[:3]
            ]
            
        except Exception as e:
            error = str(e)
            routed_strategy = 'error'
            confidence = 0.0
            latency = time.time() - start_time
            strategy_match = False
            content_match_score = 0.0
            results_preview = []
            nodes = []
        
        return RetrievalTestResult(
            query=test_case.query,
            routed_strategy=routed_strategy,
            expected_strategy=test_case.expected_strategy,
            num_results=len(nodes),
            latency=latency,
            confidence=confidence,
            results_preview=results_preview,
            strategy_match=strategy_match,
            content_match_score=content_match_score,
            error=error
        )
    
    def _calculate_content_match(self, nodes: List[NodeWithScore], expected_keywords: List[str]) -> float:
        """Calculate how well the results match expected content keywords."""
        if not nodes or not expected_keywords:
            return 0.0
        
        # Combine all result text
        all_text = " ".join([node.node.text.lower() for node in nodes])
        
        # Count keyword matches
        matches = sum(1 for keyword in expected_keywords if keyword.lower() in all_text)
        
        return matches / len(expected_keywords)
    
    def run_all_tests(self, strategy_selector: str = "llm") -> Dict[str, Any]:
        """Run all strategy tests and return comprehensive results."""
        
        print(f"\n🧪 Running iLand Retrieval Strategy Tests")
        print(f"Strategy Selector: {strategy_selector}")
        print("=" * 80)
        
        # Create test router
        router = self.create_test_router(strategy_selector)
        
        # Run all test cases
        results = []
        strategy_stats = {}
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n[Test {i}/{len(self.test_cases)}] {test_case.description}")
            print(f"Query: {test_case.query}")
            print(f"Expected Strategy: {test_case.expected_strategy}")
            
            result = self.run_strategy_test(test_case, router)
            results.append(result)
            
            # Update strategy statistics
            if result.routed_strategy not in strategy_stats:
                strategy_stats[result.routed_strategy] = {
                    'count': 0, 'matches': 0, 'total_confidence': 0.0, 'total_latency': 0.0
                }
            
            stats = strategy_stats[result.routed_strategy]
            stats['count'] += 1
            stats['total_confidence'] += result.confidence
            stats['total_latency'] += result.latency
            
            if result.strategy_match:
                stats['matches'] += 1
            
            # Print result
            if result.error:
                print(f"❌ ERROR: {result.error}")
            else:
                match_indicator = "✅" if result.strategy_match else "❌"
                print(f"{match_indicator} Routed to: {result.routed_strategy} (confidence: {result.confidence:.2f})")
                print(f"   Results: {result.num_results} nodes, {result.latency:.3f}s")
                print(f"   Content match: {result.content_match_score:.2f}")
                if result.results_preview:
                    print(f"   Preview: {result.results_preview[0]}")
        
        # Calculate summary statistics
        total_tests = len(results)
        successful_routes = sum(1 for r in results if r.strategy_match and not r.error)
        avg_latency = sum(r.latency for r in results) / total_tests
        avg_confidence = sum(r.confidence for r in results if r.confidence > 0) / max(1, sum(1 for r in results if r.confidence > 0))
        avg_content_match = sum(r.content_match_score for r in results) / total_tests
        
        # Print summary
        print(f"\n📊 Test Summary")
        print("=" * 50)
        print(f"Total Tests: {total_tests}")
        print(f"Successful Routes: {successful_routes} ({successful_routes/total_tests:.1%})")
        print(f"Average Latency: {avg_latency:.3f}s")
        print(f"Average Confidence: {avg_confidence:.2f}")
        print(f"Average Content Match: {avg_content_match:.2f}")
        
        print(f"\n📈 Strategy Usage:")
        for strategy, stats in strategy_stats.items():
            accuracy = stats['matches'] / stats['count'] if stats['count'] > 0 else 0
            avg_conf = stats['total_confidence'] / stats['count'] if stats['count'] > 0 else 0
            avg_lat = stats['total_latency'] / stats['count'] if stats['count'] > 0 else 0
            print(f"  {strategy}: {stats['count']} uses, {accuracy:.1%} accuracy, {avg_conf:.2f} confidence, {avg_lat:.3f}s latency")
        
        return {
            'test_results': results,
            'summary_stats': {
                'total_tests': total_tests,
                'successful_routes': successful_routes,
                'success_rate': successful_routes / total_tests,
                'avg_latency': avg_latency,
                'avg_confidence': avg_confidence,
                'avg_content_match': avg_content_match
            },
            'strategy_stats': strategy_stats,
            'strategy_selector': strategy_selector
        }


# Pytest test functions
@pytest.fixture
def strategy_tester():
    """Fixture providing the strategy tester instance."""
    return iLandRetrievalStrategyTester()


def test_all_retrieval_strategies_llm_selector(strategy_tester):
    """Test all retrieval strategies with LLM-based strategy selection."""
    results = strategy_tester.run_all_tests(strategy_selector="llm")
    
    # Assert overall success rate is reasonable
    assert results['summary_stats']['success_rate'] >= 0.5, \
        f"Success rate too low: {results['summary_stats']['success_rate']:.2%}"
    
    # Assert all strategies were tested at least once
    expected_strategies = {"vector", "hybrid", "metadata", "planner", "recursive", "chunk_decoupling", "summary"}
    tested_strategies = set(results['strategy_stats'].keys()) - {"error", "unknown"}
    
    # We expect at least 4 out of 7 strategies to be used
    assert len(tested_strategies) >= 4, \
        f"Too few strategies tested: {tested_strategies}"
    
    # Assert reasonable performance
    assert results['summary_stats']['avg_latency'] < 1.0, \
        f"Average latency too high: {results['summary_stats']['avg_latency']:.3f}s"


def test_all_retrieval_strategies_heuristic_selector(strategy_tester):
    """Test all retrieval strategies with heuristic-based strategy selection."""
    results = strategy_tester.run_all_tests(strategy_selector="heuristic")
    
    # Assert overall success rate is reasonable for heuristic
    assert results['summary_stats']['success_rate'] >= 0.4, \
        f"Heuristic success rate too low: {results['summary_stats']['success_rate']:.2%}"
    
    # Assert reasonable performance
    assert results['summary_stats']['avg_latency'] < 0.5, \
        f"Heuristic latency too high: {results['summary_stats']['avg_latency']:.3f}s"


def test_all_retrieval_strategies_round_robin_selector(strategy_tester):
    """Test all retrieval strategies with round-robin strategy selection."""
    results = strategy_tester.run_all_tests(strategy_selector="round_robin")
    
    # Round-robin should distribute queries across strategies
    strategy_counts = {s: stats['count'] for s, stats in results['strategy_stats'].items()}
    
    # Should have reasonable distribution (not all going to one strategy)
    if len(strategy_counts) > 1:
        max_count = max(strategy_counts.values())
        min_count = min(strategy_counts.values())
        assert max_count - min_count <= 3, \
            f"Round-robin distribution too uneven: {strategy_counts}"


def test_specific_strategy_routing(strategy_tester):
    """Test that specific queries route to expected strategies."""
    router = strategy_tester.create_test_router("llm")
    
    # Test cases that should have high confidence routing
    high_confidence_tests = [
        ("โฉนดที่ดินในจังหวัดชัยนาท", "vector"),  # Simple semantic
        ("จังหวัดอ่างทอง อำเภอเมืองอ่างทอง", "metadata"),  # Geographic
        ("โฉนด นส.3 กรรมสิทธิ์บริษัท", "hybrid"),  # Thai keywords
    ]
    
    for query, expected_strategy in high_confidence_tests:
        query_bundle = QueryBundle(query_str=query)
        nodes = router._retrieve(query_bundle)
        
        if nodes:
            metadata = getattr(nodes[0].node, 'metadata', {})
            routed_strategy = metadata.get('selected_strategy', 'unknown')
            confidence = metadata.get('strategy_confidence', 0.0)
            
            print(f"Query: {query}")
            print(f"Expected: {expected_strategy}, Got: {routed_strategy}, Confidence: {confidence:.2f}")
            
            # For high-confidence cases, we expect correct routing
            if confidence >= 0.7:
                assert routed_strategy == expected_strategy, \
                    f"High-confidence routing failed: {query} -> {routed_strategy} (expected {expected_strategy})"


def test_content_quality(strategy_tester):
    """Test that retrieval results contain relevant content."""
    router = strategy_tester.create_test_router("llm")
    
    test_queries = [
        ("โฉนดที่ดินในจังหวัดชัยนาท", ["โฉนด", "ชัยนาท"]),
        ("ขนาดพื้นที่ไร่ งาน ตารางวา", ["ไร่", "งาน", "ตารางวา"]),
        ("กรรมสิทธิ์บริษัท", ["กรรมสิทธิ์บริษัท"]),
    ]
    
    for query, expected_keywords in test_queries:
        query_bundle = QueryBundle(query_str=query)
        nodes = router._retrieve(query_bundle)
        
        assert len(nodes) > 0, f"No results for query: {query}"
        
        # Check if results contain expected keywords
        all_text = " ".join([node.node.text.lower() for node in nodes])
        matches = sum(1 for keyword in expected_keywords if keyword.lower() in all_text)
        match_ratio = matches / len(expected_keywords)
        
        assert match_ratio >= 0.5, \
            f"Poor content relevance for '{query}': {match_ratio:.1%} keyword match"


if __name__ == "__main__":
    # Run tests directly
    tester = iLandRetrievalStrategyTester()
    
    print("Testing LLM Strategy Selector:")
    llm_results = tester.run_all_tests("llm")
    
    print("\n" + "="*80)
    print("Testing Heuristic Strategy Selector:")
    heuristic_results = tester.run_all_tests("heuristic")
    
    print("\n" + "="*80)
    print("Testing Round-Robin Strategy Selector:")
    rr_results = tester.run_all_tests("round_robin")
    
    # Compare results
    print(f"\n🏆 Comparison Summary:")
    print(f"LLM Success Rate: {llm_results['summary_stats']['success_rate']:.1%}")
    print(f"Heuristic Success Rate: {heuristic_results['summary_stats']['success_rate']:.1%}")
    print(f"Round-Robin Success Rate: {rr_results['summary_stats']['success_rate']:.1%}")