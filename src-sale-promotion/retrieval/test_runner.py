"""
Test Runner for Sales Promotion RAG Pipeline

Automated test execution, validation, and report generation.
Runs 10-question test set and produces quality metrics.
"""

import time
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

from .test_set import TEST_SET, get_test_set, TestQuestion


@dataclass
class TestResult:
    """Result of a single test execution."""
    
    test_id: int
    query: str
    query_english: str
    
    # Execution results
    selected_strategy: str
    strategy_match: bool
    strategy_confidence: float
    
    results_count: int
    results_count_valid: bool
    
    response_time: float
    response_time_valid: bool
    
    # Validation results
    validation_passed: Dict[str, bool] = field(default_factory=dict)
    overall_pass: bool = False
    
    # Retrieved documents
    retrieved_doc_ids: List[str] = field(default_factory=list)
    top_3_scores: List[float] = field(default_factory=list)
    
    # Response
    response_text: str = ""
    
    # Error info
    error: Optional[str] = None


class TestRunner:
    """
    Automated test execution and validation.
    
    Executes all 10 tests, validates results, and generates reports.
    """
    
    def __init__(self, query_engine):
        """
        Initialize test runner.
        
        Args:
            query_engine: PromotionQueryEngine instance
        """
        self.query_engine = query_engine
        self.results = []
    
    def run_all_tests(self) -> List[TestResult]:
        """
        Run all 10 tests.
        
        Returns:
            List of TestResult objects
        """
        print(f"\n{'='*70}")
        print(f"Running {len(TEST_SET)} Tests")
        print(f"{'='*70}\n")
        
        results = []
        
        for test in TEST_SET:
            print(f"{'─'*70}")
            print(f"Test {test.test_id}/10: {test.query_english}")
            print(f"{'─'*70}")
            print(f"Query: {test.query}")
            
            result = self.run_single_test(test)
            results.append(result)
            
            # Display result
            status_icon = "✅" if result.overall_pass else "❌"
            print(f"\n{status_icon} {'PASS' if result.overall_pass else 'FAIL'}")
            print(f"  Strategy: {result.selected_strategy} "
                  f"(expected: {test.expected_strategy}) "
                  f"{'✓' if result.strategy_match else '✗'}")
            print(f"  Results: {result.results_count} documents")
            print(f"  Time: {result.response_time:.1f}s")
            
            if result.error:
                print(f"  Error: {result.error}")
            
            print()
        
        self.results = results
        return results
    
    def run_single_test(self, test: TestQuestion) -> TestResult:
        """
        Run a single test and validate.
        
        Args:
            test: TestQuestion to execute
            
        Returns:
            TestResult with validation
        """
        start_time = time.time()
        
        try:
            # Execute query
            result = self.query_engine.query(
                test.query,
                top_k=5,
                return_nodes=True
            )
            
            response_time = time.time() - start_time
            
            # Extract data
            metadata = result.get('metadata', {})
            nodes = result.get('nodes', [])
            
            selected_strategy = metadata.get('selected_strategy', 'unknown')
            strategy_confidence = metadata.get('strategy_confidence')
            if strategy_confidence is None:
                strategy_confidence = 0.0
            
            # Validate strategy selection
            strategy_match = (
                selected_strategy == test.expected_strategy or
                selected_strategy in test.alternative_strategies
            )
            
            # Validate result count
            results_count = len(nodes)
            results_count_valid = (
                test.expected_count_min <= results_count <= test.expected_count_max
            )
            
            # Validate response time
            response_time_valid = response_time <= test.timeout
            
            # Run custom validation rules
            validation_passed = self.validate_rules(test, result)
            
            # Overall pass/fail
            overall_pass = (
                strategy_match and
                results_count_valid and
                response_time_valid and
                (not validation_passed or all(validation_passed.values()))
            )
            
            return TestResult(
                test_id=test.test_id,
                query=test.query,
                query_english=test.query_english,
                selected_strategy=selected_strategy,
                strategy_match=strategy_match,
                strategy_confidence=strategy_confidence,
                results_count=results_count,
                results_count_valid=results_count_valid,
                response_time=response_time,
                response_time_valid=response_time_valid,
                validation_passed=validation_passed,
                overall_pass=overall_pass,
                retrieved_doc_ids=[node.node.id_ for node in nodes[:10]],
                top_3_scores=[node.score for node in nodes[:3]],
                response_text=result.get('response', '')
            )
        
        except Exception as e:
            response_time = time.time() - start_time
            
            return TestResult(
                test_id=test.test_id,
                query=test.query,
                query_english=test.query_english,
                selected_strategy="ERROR",
                strategy_match=False,
                strategy_confidence=0.0,
                results_count=0,
                results_count_valid=False,
                response_time=response_time,
                response_time_valid=False,
                validation_passed={},
                overall_pass=False,
                retrieved_doc_ids=[],
                top_3_scores=[],
                error=str(e)
            )
    
    def validate_rules(self, test: TestQuestion, result: Dict) -> Dict[str, bool]:
        """
        Validate custom rules for test.
        
        Args:
            test: TestQuestion with validation_rules
            result: Query result dictionary
            
        Returns:
            Dictionary of rule_name -> passed (bool)
        """
        validation = {}
        
        if not test.validation_rules:
            return validation
        
        nodes = result.get('nodes', [])
        response = result.get('response', '')
        
        # Rule: all_results_match_customer_type
        if "all_results_match_customer_type" in test.validation_rules:
            expected_type = test.expected_filters.get("customer_type") if test.expected_filters else None
            validation["all_results_match_customer_type"] = all(
                node.node.metadata.get("customer_type") == expected_type
                for node in nodes
            )
        
        # Rule: all_results_are_active
        if "all_results_are_active" in test.validation_rules:
            validation["all_results_are_active"] = all(
                node.node.metadata.get("is_active") == True
                for node in nodes
            )
        
        # Rule: all_results_are_inactive
        if "all_results_are_inactive" in test.validation_rules:
            validation["all_results_are_inactive"] = all(
                node.node.metadata.get("is_active") == False
                for node in nodes
            )
        
        # Rule: all_results_discount_above_25
        if "all_results_discount_above_25" in test.validation_rules:
            validation["all_results_discount_above_25"] = all(
                node.node.metadata.get("discount_percentage", 0) > 25
                for node in nodes
            )
        
        # Rule: top_3_relevance_threshold
        if "top_3_relevance_threshold" in test.validation_rules:
            threshold = test.validation_rules["top_3_relevance_threshold"]
            top_3_scores = [node.score for node in nodes[:3]]
            validation["top_3_relevance_threshold"] = (
                len(top_3_scores) > 0 and
                min(top_3_scores) >= threshold
            )
        
        # Rule: top_5_all_water_category
        if "top_5_all_water_category" in test.validation_rules:
            validation["top_5_all_water_category"] = all(
                node.node.metadata.get("product_category") == "water"
                for node in nodes[:5]
            )
        
        # Rule: top_5_all_chang_brand
        if "top_5_all_chang_brand" in test.validation_rules:
            validation["top_5_all_chang_brand"] = all(
                node.node.metadata.get("product_brand") == "ช้าง"
                for node in nodes[:5]
            )
        
        # Rule: response_mentions_both_sizes
        if "response_mentions_both_sizes" in test.validation_rules:
            validation["response_mentions_both_sizes"] = (
                "600" in response and "1500" in response
            )
        
        # Rule: response_includes_analysis
        if "response_includes_analysis" in test.validation_rules:
            validation["response_includes_analysis"] = len(response) > 100
        
        # Rule: mentions_discount_percentage
        if "mentions_discount_percentage" in test.validation_rules:
            validation["mentions_discount_percentage"] = (
                "discount" in response.lower() or
                "ส่วนลด" in response or
                "%" in response
            )
        
        return validation
    
    def generate_report(self, results: List[TestResult]):
        """
        Generate JSON and Markdown reports.
        
        Args:
            results: List of TestResult objects
        """
        # Calculate summary statistics
        total = len(results)
        passed = sum(1 for r in results if r.overall_pass)
        failed = total - passed
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        avg_response_time = sum(r.response_time for r in results) / total if total > 0 else 0
        
        strategy_accuracy = sum(
            1 for r in results if r.strategy_match
        ) / total * 100 if total > 0 else 0
        
        result_count_accuracy = sum(
            1 for r in results if r.results_count_valid
        ) / total * 100 if total > 0 else 0
        
        # Build JSON report
        json_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_run_id": f"run_{time.strftime('%Y%m%d_%H%M%S')}",
            "system_info": {
                "phase": "Phase 4",
                "total_promotions": 71,
                "total_embeddings": 142,
                "embedding_model": "text-embedding-3-large",
                "llm_model": "gpt-5",
                "strategies_available": 7
            },
            "summary": {
                "total_tests": total,
                "passed": passed,
                "failed": failed,
                "pass_rate_percent": round(pass_rate, 1),
                "avg_response_time": round(avg_response_time, 2),
                "min_response_time": round(min(r.response_time for r in results), 2) if results else 0,
                "max_response_time": round(max(r.response_time for r in results), 2) if results else 0,
                "strategy_accuracy_percent": round(strategy_accuracy, 1),
                "result_count_accuracy_percent": round(result_count_accuracy, 1)
            },
            "test_results": [
                {
                    "test_id": r.test_id,
                    "query": r.query,
                    "query_english": r.query_english,
                    "status": "PASS" if r.overall_pass else "FAIL",
                    "selected_strategy": r.selected_strategy,
                    "expected_strategy": TEST_SET[r.test_id - 1].expected_strategy,
                    "strategy_match": r.strategy_match,
                    "strategy_confidence": round(r.strategy_confidence, 3),
                    "results_count": r.results_count,
                    "response_time": round(r.response_time, 2),
                    "validation": r.validation_passed,
                    "top_3_scores": [round(s, 3) for s in r.top_3_scores],
                    "error": r.error
                }
                for r in results
            ],
            "failed_tests": [
                {
                    "test_id": r.test_id,
                    "query": r.query,
                    "reason": self._get_failure_reason(r, TEST_SET[r.test_id - 1])
                }
                for r in results if not r.overall_pass
            ]
        }
        
        # Save reports
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        json_path = output_dir / f"test_results_{timestamp}.json"
        md_path = output_dir / f"test_summary_{timestamp}.md"
        
        # Save JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False)
        
        # Save Markdown
        md_content = self._generate_markdown_summary(results, json_report)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        
        print(f"\n{'='*70}")
        print("Reports Generated:")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {md_path}")
        print(f"{'='*70}")
        
        return json_report
    
    def _get_failure_reason(self, result: TestResult, test: TestQuestion) -> str:
        """Get human-readable failure reason."""
        reasons = []
        
        if not result.strategy_match:
            reasons.append(f"Strategy mismatch: expected '{test.expected_strategy}', got '{result.selected_strategy}'")
        
        if not result.results_count_valid:
            reasons.append(f"Result count out of range: {result.results_count} (expected: {test.expected_count_min}-{test.expected_count_max})")
        
        if not result.response_time_valid:
            reasons.append(f"Timeout exceeded: {result.response_time:.1f}s > {test.timeout}s")
        
        for rule, passed in result.validation_passed.items():
            if not passed:
                reasons.append(f"Validation failed: {rule}")
        
        if result.error:
            reasons.append(f"Error: {result.error}")
        
        return "; ".join(reasons) if reasons else "Unknown failure"
    
    def _generate_markdown_summary(self, results: List[TestResult], json_report: Dict) -> str:
        """
        Generate Markdown summary report.
        
        Args:
            results: List of TestResult objects
            json_report: JSON report dictionary
            
        Returns:
            Markdown formatted string
        """
        summary = json_report["summary"]
        
        md = f"""# Test Run Summary

**Date:** {json_report["timestamp"]}  
**Test Run ID:** {json_report["test_run_id"]}  
**Phase:** 4 - CLI Implementation & Test Set

---

## Overall Results

| Metric | Value |
|--------|-------|
| **Total Tests** | {summary["total_tests"]} |
| **Passed** | {summary["passed"]} ✅ |
| **Failed** | {summary["failed"]} ❌ |
| **Pass Rate** | {summary["pass_rate_percent"]}% |
| **Avg Response Time** | {summary["avg_response_time"]}s |
| **Min Response Time** | {summary["min_response_time"]}s |
| **Max Response Time** | {summary["max_response_time"]}s |
| **Strategy Accuracy** | {summary["strategy_accuracy_percent"]}% |
| **Result Count Accuracy** | {summary["result_count_accuracy_percent"]}% |

---

## Test Results

"""
        
        for r in results:
            test = TEST_SET[r.test_id - 1]
            status_icon = "✅" if r.overall_pass else "❌"
            
            md += f"\n### Test {r.test_id}: {test.query_english} {status_icon}\n\n"
            md += f"**Query:** {r.query}\n\n"
            md += f"- **Language:** {test.language}\n"
            md += f"- **Expected Strategy:** {test.expected_strategy}\n"
            md += f"- **Selected Strategy:** {r.selected_strategy} "
            md += f"({'✓' if r.strategy_match else '✗'})\n"
            md += f"- **Confidence:** {r.strategy_confidence:.2f}\n"
            md += f"- **Results Count:** {r.results_count} "
            md += f"(expected: {test.expected_count_min}-{test.expected_count_max}) "
            md += f"({'✓' if r.results_count_valid else '✗'})\n"
            md += f"- **Response Time:** {r.response_time:.2f}s "
            md += f"(limit: {test.timeout}s) "
            md += f"({'✓' if r.response_time_valid else '✗'})\n"
            
            if r.validation_passed:
                md += f"\n**Validation Rules:**\n"
                for rule, passed in r.validation_passed.items():
                    icon = "✓" if passed else "✗"
                    md += f"- {icon} {rule}\n"
            
            if r.top_3_scores:
                md += f"\n**Top-3 Scores:** {[f'{s:.3f}' for s in r.top_3_scores]}\n"
            
            if r.error:
                md += f"\n**Error:** {r.error}\n"
            
            md += f"\n**Status:** {'PASS' if r.overall_pass else 'FAIL'}\n"
            md += "\n---\n"
        
        # Failed tests summary
        if json_report.get("failed_tests"):
            md += "\n## Failed Tests Analysis\n\n"
            for failed in json_report["failed_tests"]:
                md += f"### Test {failed['test_id']}\n\n"
                md += f"**Reason:** {failed['reason']}\n\n"
                md += "---\n\n"
        
        # Recommendations
        md += "\n## Recommendations\n\n"
        if summary["pass_rate_percent"] >= 90:
            md += "✅ Excellent pass rate! System is production-ready.\n"
        elif summary["pass_rate_percent"] >= 80:
            md += "✅ Good pass rate. Minor improvements recommended.\n"
        else:
            md += "⚠️ Below target pass rate. Review failed tests and improve system.\n"
        
        return md


def main():
    """Standalone test runner for debugging."""
    from retrieval.query_engine import PromotionQueryEngine
    from retrieval.router import PromotionRouter
    
    print("Test Set Summary:")
    from retrieval.test_set import print_test_summary
    print_test_summary()


if __name__ == "__main__":
    main()

