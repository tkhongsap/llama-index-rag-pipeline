"""
Test Set for Sales Promotion RAG Pipeline

10 comprehensive test questions covering all 7 retrieval strategies.
Validates strategy selection, result quality, and system performance.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class TestQuestion:
    """Single test question definition."""
    
    test_id: int
    query: str
    query_english: str
    language: str  # "Thai" or "English"
    
    # Expected routing
    expected_strategy: str
    alternative_strategies: List[str] = field(default_factory=list)
    
    # Expected filters (for metadata strategy)
    expected_filters: Optional[Dict] = None
    
    # Expected results
    expected_count_min: int = 1
    expected_count_max: int = 100
    expected_doc_ids: Optional[List[str]] = None
    
    # Validation criteria
    validation_rules: Optional[Dict] = None
    
    # Performance
    timeout: int = 30  # seconds


# 10 Test Questions
TEST_SET: List[TestQuestion] = [
    # Test 1: Customer Type Filtering (Metadata Strategy)
    TestQuestion(
        test_id=1,
        query="โปรโมชันสำหรับเอเย่นต์",
        query_english="Promotions for agents",
        language="Thai",
        expected_strategy="metadata",
        alternative_strategies=["vector"],
        expected_filters={"customer_type": "เอเย่นต์"},
        expected_count_min=3,
        expected_count_max=10,
        expected_doc_ids=["PROMO_001", "PROMO_025"],
        validation_rules={
            "top_3_relevance_threshold": 0.50
        },
        timeout=30
    ),
    
    # Test 2: Product Category Search (Vector Strategy)
    TestQuestion(
        test_id=2,
        query="โปรโมชันน้ำดื่มช้าง",
        query_english="Chang drinking water promotions",
        language="Thai",
        expected_strategy="vector",
        alternative_strategies=["hybrid"],
        expected_count_min=3,
        expected_count_max=10,
        expected_doc_ids=["PROMO_001", "PROMO_008", "PROMO_013"],
        validation_rules={
            "top_3_relevance_threshold": 0.50
        },
        timeout=30
    ),
    
    # Test 3: Discount Range Filtering (Metadata Strategy)
    TestQuestion(
        test_id=3,
        query="โปรโมชันที่ให้ส่วนลดมากกว่า 25%",
        query_english="Promotions with discount greater than 25%",
        language="Thai",
        expected_strategy="metadata",
        alternative_strategies=["vector"],
        expected_filters={"discount_percentage": {"$gt": 25}},
        expected_count_min=3,
        expected_count_max=10,
        validation_rules={
            "top_3_relevance_threshold": 0.50
        },
        timeout=30
    ),
    
    # Test 4: Active Status Filter (Metadata Strategy)
    TestQuestion(
        test_id=4,
        query="โปรโมชันที่กำลังใช้งานอยู่",
        query_english="Currently active promotions",
        language="Thai",
        expected_strategy="metadata",
        alternative_strategies=["vector"],
        expected_filters={"is_active": True},
        expected_count_min=3,
        expected_count_max=10,
        validation_rules={
            "top_3_relevance_threshold": 0.50
        },
        timeout=30
    ),
    
    # Test 5: Multi-Criteria Query (Hybrid/Metadata Strategy)
    TestQuestion(
        test_id=5,
        query="โปรโมชันเบียร์สำหรับซับเอเย่นต์",
        query_english="Beer promotions for sub-agents",
        language="Thai",
        expected_strategy="hybrid",
        alternative_strategies=["metadata", "vector"],
        expected_filters={"customer_type": "ซับเอเย่นต์", "product_category": "beer"},
        expected_count_min=2,
        expected_count_max=10,
        validation_rules={
            "top_3_relevance_threshold": 0.50
        },
        timeout=30
    ),
    
    # Test 6: Free Gift Queries (Vector Strategy)
    TestQuestion(
        test_id=6,
        query="โปรโมชันที่ให้ของแถม",
        query_english="Promotions with free gifts",
        language="Thai",
        expected_strategy="vector",
        alternative_strategies=["summary", "metadata"],
        expected_count_min=3,
        expected_count_max=10,
        validation_rules={
            "top_3_relevance_threshold": 0.50
        },
        timeout=30
    ),
    
    # Test 7: Brand-Specific Search (Vector/Hybrid Strategy)
    TestQuestion(
        test_id=7,
        query="โปรโมชันสินค้าแบรนด์ช้าง",
        query_english="Chang brand product promotions",
        language="Thai",
        expected_strategy="vector",
        alternative_strategies=["hybrid", "metadata"],
        expected_count_min=3,
        expected_count_max=10,
        validation_rules={
            "top_3_relevance_threshold": 0.50
        },
        timeout=30
    ),
    
    # Test 8: Size Comparison Query (Summary/Planner Strategy)
    TestQuestion(
        test_id=8,
        query="Compare promotions for 600 CC vs 1500 CC water bottles",
        query_english="Compare promotions for 600 CC vs 1500 CC water bottles",
        language="English",
        expected_strategy="planner",
        alternative_strategies=["summary", "metadata", "vector"],
        expected_count_min=3,
        expected_count_max=30,
        validation_rules={
            "response_mentions_both_sizes": True
        },
        timeout=40
    ),
    
    # Test 9: Expired Promotions (Metadata Strategy)
    TestQuestion(
        test_id=9,
        query="โปรโมชันที่หมดอายุแล้ว",
        query_english="Expired promotions",
        language="Thai",
        expected_strategy="metadata",
        alternative_strategies=["vector"],
        expected_filters={"is_active": False},
        expected_count_min=0,  # May return 0 if no expired
        expected_count_max=35,
        validation_rules={
            "top_3_relevance_threshold": 0.50
        },
        timeout=30
    ),
    
    # Test 10: Best Value Analysis (Planner/Summary Strategy)
    TestQuestion(
        test_id=10,
        query="Which promotion offers the best value for wholesale customers?",
        query_english="Which promotion offers the best value for wholesale customers?",
        language="English",
        expected_strategy="planner",
        alternative_strategies=["summary", "metadata", "vector"],
        expected_count_min=3,
        expected_count_max=10,
        validation_rules={
            "response_includes_analysis": True,
            "mentions_discount_percentage": True
        },
        timeout=40
    ),
]


def get_test_set() -> List[TestQuestion]:
    """
    Get complete test set.
    
    Returns:
        List of 10 TestQuestion objects
    """
    return TEST_SET


def get_test_by_id(test_id: int) -> Optional[TestQuestion]:
    """
    Get specific test by ID.
    
    Args:
        test_id: Test identifier (1-10)
        
    Returns:
        TestQuestion or None if not found
    """
    return next((t for t in TEST_SET if t.test_id == test_id), None)


def get_tests_by_strategy(strategy: str) -> List[TestQuestion]:
    """
    Get all tests for a specific strategy.
    
    Args:
        strategy: Strategy name
        
    Returns:
        List of tests expecting that strategy
    """
    return [
        t for t in TEST_SET
        if t.expected_strategy == strategy or strategy in t.alternative_strategies
    ]


def print_test_summary():
    """Print summary of test set."""
    print("\n" + "="*70)
    print("Test Set Summary")
    print("="*70)
    
    print(f"\nTotal Tests: {len(TEST_SET)}")
    
    # By strategy
    strategy_counts = {}
    for test in TEST_SET:
        strategy = test.expected_strategy
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    print("\nBy Expected Strategy:")
    for strategy, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
        print(f"  - {strategy}: {count}")
    
    # By language
    thai_count = sum(1 for t in TEST_SET if t.language == "Thai")
    english_count = sum(1 for t in TEST_SET if t.language == "English")
    
    print(f"\nBy Language:")
    print(f"  - Thai: {thai_count}")
    print(f"  - English: {english_count}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    print_test_summary()
    
    print("\nTest Questions:")
    for test in TEST_SET:
        print(f"\n{test.test_id}. {test.query}")
        print(f"   Expected: {test.expected_strategy}")
        print(f"   Timeout: {test.timeout}s")

