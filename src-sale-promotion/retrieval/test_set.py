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
    # Test 1: Customer Identity Question (Metadata Strategy)
    TestQuestion(
        test_id=1,
        query="ผมเป็นเอเย่นต์ มีโปรโมชันอะไรบ้าง",
        query_english="I'm an agent, what promotions can I use?",
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
    
    # Test 2: Product Interest Question (Vector/Hybrid Strategy)
    TestQuestion(
        test_id=2,
        query="มีโปรอะไรสำหรับน้ำดื่มช้างบ้าง",
        query_english="What deals do you have for Chang water?",
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
    
    # Test 3: Value-Seeking Question (Metadata Strategy)
    TestQuestion(
        test_id=3,
        query="โปรไหนให้ส่วนลดเยอะที่สุด",
        query_english="Which promotion gives the best discount?",
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
    
    # Test 4: Time-Based Question (Metadata Strategy)
    TestQuestion(
        test_id=4,
        query="ตอนนี้มีโปรอะไรใช้ได้บ้าง",
        query_english="What promotions are available right now?",
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
    
    # Test 5: Purchase Intent Question (Hybrid Strategy)
    TestQuestion(
        test_id=5,
        query="ถ้าผมซื้อเบียร์ จะได้อะไรแถมบ้าง",
        query_english="If I buy beer, what do I get free?",
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
    
    # Test 6: Free Gift Question (Vector Strategy)
    TestQuestion(
        test_id=6,
        query="มีโปรแถมของฟรีไหม",
        query_english="Are there any promotions with free items?",
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
    
    # Test 7: Brand Preference Question (Vector/Hybrid Strategy)
    TestQuestion(
        test_id=7,
        query="แบรนด์ช้างมีโปรอะไรดีๆ บ้าง",
        query_english="What good promotions does Chang brand have?",
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
    
    # Test 8: Comparison Question (Planner Strategy)
    TestQuestion(
        test_id=8,
        query="Which is better value - small or large bottles?",
        query_english="Which is better value - small or large bottles?",
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
    
    # Test 9: Urgency Question (Metadata Strategy)
    TestQuestion(
        test_id=9,
        query="มีโปรไหนใกล้หมดอายุบ้างไหม",
        query_english="Are there any promotions ending soon?",
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
    
    # Test 10: Order Planning Question (Planner Strategy)
    TestQuestion(
        test_id=10,
        query="If I order 100 packs of water, which promotion should I use?",
        query_english="If I order 100 packs of water, which promotion should I use?",
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

