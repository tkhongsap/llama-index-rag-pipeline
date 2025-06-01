#!/usr/bin/env python3
"""
Test script for the Agentic Retriever CLI

This script tests the agentic retrieval system with predefined questions
based on the candidate profile data in the example folder.
"""

import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from agentic_retriever.cli import query_agentic_retriever, format_output


class RetrieverTestSuite:
    """Test suite for the agentic retriever with predefined questions."""
    
    def __init__(self, api_key: str = None, save_markdown: bool = True):
        """Initialize the test suite."""
        self.api_key = api_key
        self.save_markdown = save_markdown
        self.test_questions = self._create_test_questions()
        self.results = []
        self.markdown_content = []
    
    def _create_test_questions(self) -> List[Dict[str, Any]]:
        """Create predefined test questions based on the candidate profile data."""
        return [
            # === BASIC VECTOR STRATEGY TESTS ===
            {
                "id": 1,
                "category": "Basic Vector - Simple",
                "question": "salary compensation THB",
                "description": "Test basic vector search with simple keywords",
                "expected_topics": ["salary", "compensation", "THB"],
                "expected_strategy": "vector"
            },
            {
                "id": 2,
                "category": "Basic Vector - Education",
                "question": "university degree bachelor master education",
                "description": "Test vector search for educational content",
                "expected_topics": ["university", "degree", "bachelor", "master", "education"],
                "expected_strategy": "vector"
            },
            {
                "id": 3,
                "category": "Basic Vector - Career",
                "question": "job position experience years career",
                "description": "Test vector search for career-related content",
                "expected_topics": ["job", "position", "experience", "years", "career"],
                "expected_strategy": "vector"
            },
            
            # === STRATEGY-SPECIFIC ROUTER TESTS ===
            {
                "id": 4,
                "category": "Router - Index Classification",
                "question": "What are the salary ranges for different positions?",
                "description": "Test router index classification - should route to compensation_docs",
                "expected_topics": ["salary", "ranges", "positions"],
                "expected_strategy": "vector"
            },
            {
                "id": 5,
                "category": "Router - Education Index",
                "question": "What educational backgrounds do the candidates have?",
                "description": "Test router index classification - should route to education_career",
                "expected_topics": ["educational", "backgrounds", "candidates"],
                "expected_strategy": "vector"
            },
            {
                "id": 6,
                "category": "Router - Candidate Profiles",
                "question": "Show me candidate profiles from Bangkok",
                "description": "Test router index classification - should route to candidate_profiles",
                "expected_topics": ["candidate", "profiles", "bangkok"],
                "expected_strategy": "vector"
            },
            
            # === SEMANTIC SEARCH TESTS ===
            {
                "id": 7,
                "category": "Semantic - Descriptive",
                "question": "Find candidates with management experience in technology companies",
                "description": "Test semantic understanding of role and industry requirements",
                "expected_topics": ["management", "experience", "technology", "companies"],
                "expected_strategy": "vector"
            },
            {
                "id": 8,
                "category": "Semantic - Skills",
                "question": "Who has expertise in human resources and recruitment?",
                "description": "Test semantic search for skills and expertise",
                "expected_topics": ["expertise", "human resources", "recruitment"],
                "expected_strategy": "vector"
            },
            
            # === SPECIFIC DATA RETRIEVAL ===
            {
                "id": 9,
                "category": "Data Retrieval - Location",
                "question": "candidates in กรุงเทพมหานคร Bangkok",
                "description": "Test location-based search with Thai and English terms",
                "expected_topics": ["กรุงเทพมหานคร", "bangkok", "candidates"],
                "expected_strategy": "vector"
            },
            {
                "id": 10,
                "category": "Data Retrieval - Industry",
                "question": "manufacturing oil industry positions",
                "description": "Test industry-specific search",
                "expected_topics": ["manufacturing", "oil", "industry", "positions"],
                "expected_strategy": "vector"
            },
            
            # === SUMMARY STRATEGY TESTS ===
            {
                "id": 11,
                "category": "Summary Strategy",
                "question": "Provide an overview of all candidate qualifications and experience levels",
                "description": "Test summary-first retrieval for comprehensive overviews",
                "expected_topics": ["overview", "qualifications", "experience", "levels"],
                "expected_strategy": "summary"
            },
            {
                "id": 12,
                "category": "Summary Strategy",
                "question": "Summarize the compensation structure across all positions",
                "description": "Test summary strategy for compensation overview",
                "expected_topics": ["summarize", "compensation", "structure", "positions"],
                "expected_strategy": "summary"
            },
            
            # === RECURSIVE STRATEGY TESTS ===
            {
                "id": 13,
                "category": "Recursive Strategy",
                "question": "Break down the career progression from entry to senior levels with specific examples",
                "description": "Test recursive retrieval for hierarchical information",
                "expected_topics": ["career", "progression", "entry", "senior", "examples"],
                "expected_strategy": "recursive"
            },
            
            # === HYBRID STRATEGY TESTS ===
            {
                "id": 14,
                "category": "Hybrid Strategy",
                "question": "Find exact match for 'Training Officer' position AND similar roles",
                "description": "Test hybrid search combining exact and semantic matching",
                "expected_topics": ["training officer", "position", "similar", "roles"],
                "expected_strategy": "hybrid"
            },
            
            # === METADATA FILTERING TESTS ===
            {
                "id": 15,
                "category": "Metadata Strategy",
                "question": "Filter candidates with salary > 40000 AND experience 5-10 years",
                "description": "Test metadata filtering with specific criteria",
                "expected_topics": ["filter", "salary", "40000", "experience", "5-10"],
                "expected_strategy": "metadata"
            },
            
            # === PLANNER STRATEGY TESTS ===
            {
                "id": 16,
                "category": "Planner Strategy",
                "question": "First find the highest paid positions, then analyze their required qualifications, and finally compare with entry-level requirements",
                "description": "Test query planning for multi-step analysis",
                "expected_topics": ["highest paid", "qualifications", "compare", "entry-level"],
                "expected_strategy": "planner"
            },
            
            # === STRESS TESTS ===
            {
                "id": 17,
                "category": "Stress Test - Simple",
                "question": "salary",
                "description": "Test single keyword retrieval",
                "expected_topics": ["salary"],
                "expected_strategy": "vector"
            },
            {
                "id": 18,
                "category": "Stress Test - Complex",
                "question": "What are the detailed compensation packages including base salary, bonuses, and benefits for senior management positions in technology and manufacturing industries, specifically for candidates located in Bangkok with master's degrees and more than 10 years of experience?",
                "description": "Test very complex query handling",
                "expected_topics": ["compensation", "salary", "bonuses", "benefits", "senior", "management", "technology", "manufacturing", "bangkok", "master", "10 years"],
                "expected_strategy": "vector"
            },
            
            # === MULTILINGUAL TESTS ===
            {
                "id": 19,
                "category": "Multilingual - Thai",
                "question": "เงินเดือน ประสบการณ์ กรุงเทพมหานคร",
                "description": "Test Thai language query handling",
                "expected_topics": ["เงินเดือน", "ประสบการณ์", "กรุงเทพมหานคร"],
                "expected_strategy": "vector"
            },
            {
                "id": 20,
                "category": "Multilingual - Mixed",
                "question": "candidates in กรุงเทพมหานคร with bachelor degree salary above 30000 THB",
                "description": "Test mixed Thai-English query",
                "expected_topics": ["candidates", "กรุงเทพมหานคร", "bachelor", "salary", "30000"],
                "expected_strategy": "vector"
            }
        ]
    
    def _add_to_markdown(self, content: str):
        """Add content to markdown output."""
        if self.save_markdown:
            self.markdown_content.append(content)
    
    def run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case."""
        test_header = f"\n{'='*80}\n🧪 TEST {test_case['id']}: {test_case['category']}\n{'='*80}"
        print(test_header)
        
        question_info = f"❓ Question: {test_case['question']}\n📝 Description: {test_case['description']}\n🎯 Expected Topics: {', '.join(test_case['expected_topics'])}"
        print(question_info)
        print(f"{'='*80}")
        
        # Add to markdown
        self._add_to_markdown(f"## TEST {test_case['id']}: {test_case['category']}\n")
        self._add_to_markdown(f"**Question:** {test_case['question']}\n")
        self._add_to_markdown(f"**Description:** {test_case['description']}\n")
        self._add_to_markdown(f"**Expected Topics:** {', '.join(test_case['expected_topics'])}\n")
        
        start_time = time.time()
        
        # Execute the query using the CLI function
        result = query_agentic_retriever(
            query=test_case['question'],
            top_k=5,
            api_key=self.api_key
        )
        
        test_duration = time.time() - start_time
        
        # Format and display the output
        format_output(result)
        
        # Add response to markdown
        if result.get("response"):
            self._add_to_markdown(f"**Response:**\n```\n{result['response']}\n```\n")
        
        # Add routing info to markdown
        metadata = result.get("metadata", {})
        routing_info = f"**Routing Information:**\n"
        routing_info += f"- Index: {metadata.get('index', 'unknown')}\n"
        routing_info += f"- Strategy: {metadata.get('strategy', 'unknown')}\n"
        routing_info += f"- Latency: {metadata.get('total_time_ms', 0)} ms\n"
        routing_info += f"- Sources: {metadata.get('num_sources', 0)}\n"
        if metadata.get('index_confidence') is not None:
            routing_info += f"- Index Confidence: {metadata['index_confidence']:.3f}\n"
        if metadata.get('strategy_confidence') is not None:
            routing_info += f"- Strategy Confidence: {metadata['strategy_confidence']:.3f}\n"
        
        self._add_to_markdown(routing_info)
        
        # Analyze the result
        analysis = self._analyze_result(result, test_case)
        
        analysis_output = f"\n📊 TEST ANALYSIS:\n   • Success: {'✅ YES' if analysis['success'] else '❌ NO'}\n   • Response Quality: {analysis['quality_score']}/5\n   • Topic Coverage: {analysis['topic_coverage']:.1%}\n   • Test Duration: {test_duration:.2f}s"
        print(analysis_output)
        
        # Add analysis to markdown
        analysis_md = f"**Analysis:**\n"
        analysis_md += f"- Success: {'✅ YES' if analysis['success'] else '❌ NO'}\n"
        analysis_md += f"- Response Quality: {analysis['quality_score']}/5\n"
        analysis_md += f"- Topic Coverage: {analysis['topic_coverage']:.1%}\n"
        analysis_md += f"- Test Duration: {test_duration:.2f}s\n"
        
        if analysis['issues']:
            issues_text = f"   • Issues: {', '.join(analysis['issues'])}"
            print(issues_text)
            analysis_md += f"- Issues: {', '.join(analysis['issues'])}\n"
        
        self._add_to_markdown(analysis_md + "\n---\n")
        
        # Store result for summary
        test_result = {
            **test_case,
            "result": result,
            "analysis": analysis,
            "duration": test_duration,
            "timestamp": time.time()
        }
        
        self.results.append(test_result)
        return test_result
    
    def _analyze_result(self, result: Dict[str, Any], test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the quality of a test result."""
        analysis = {
            "success": False,
            "quality_score": 0,
            "topic_coverage": 0.0,
            "issues": [],
            "routing_quality": "unknown"
        }
        
        if result["error"]:
            analysis["issues"].append(f"Error: {result['error']}")
            return analysis
        
        # Check if response is effectively empty
        response = result.get("response", "")
        if not response or response.strip() == "Empty Response" or len(response.strip()) < 10:
            analysis["issues"].append("Empty or minimal response")
            analysis["routing_quality"] = "poor"
            return analysis
        
        analysis["success"] = True
        response_text = response.lower()
        
        # Check topic coverage
        expected_topics = test_case["expected_topics"]
        covered_topics = sum(1 for topic in expected_topics if topic.lower() in response_text)
        analysis["topic_coverage"] = covered_topics / len(expected_topics) if expected_topics else 0
        
        # Enhanced quality scoring (1-5 scale)
        quality_score = 1  # Base score for successful response
        
        # Content quality checks
        if analysis["topic_coverage"] >= 0.7:
            quality_score += 2  # Excellent topic coverage
        elif analysis["topic_coverage"] >= 0.4:
            quality_score += 1  # Good topic coverage
        
        # Response depth and usefulness
        if len(response) >= 300:
            quality_score += 1  # Comprehensive response
        elif len(response) >= 150:
            quality_score += 0.5  # Adequate response
        
        # Performance metrics
        latency = result["metadata"].get("total_time_ms", 0)
        if latency > 0:
            if latency < 3000:
                quality_score += 1  # Fast response
            elif latency < 8000:
                quality_score += 0.5  # Reasonable response time
        
        # Source availability
        num_sources = result["metadata"].get("num_sources", 0)
        if num_sources > 0:
            quality_score += 0.5
            if num_sources >= 3:
                quality_score += 0.5  # Multiple sources
        
        analysis["quality_score"] = min(round(quality_score, 1), 5.0)
        
        # Enhanced issue detection
        if analysis["topic_coverage"] < 0.2:
            analysis["issues"].append("Very low topic coverage")
        elif analysis["topic_coverage"] < 0.4:
            analysis["issues"].append("Low topic coverage")
        
        if len(response) < 50:
            analysis["issues"].append("Response too short")
        elif len(response) < 100:
            analysis["issues"].append("Response somewhat short")
        
        if latency > 15000:
            analysis["issues"].append("Very slow response time")
        elif latency > 8000:
            analysis["issues"].append("Slow response time")
        
        if num_sources == 0:
            analysis["issues"].append("No sources retrieved")
        
        # Routing quality assessment
        selected_strategy = result["metadata"].get("strategy", "unknown")
        expected_strategy = test_case.get("expected_strategy", "unknown")
        
        if selected_strategy == expected_strategy:
            analysis["routing_quality"] = "excellent"
        elif selected_strategy in ["vector", "summary"]:  # Common fallback strategies
            analysis["routing_quality"] = "good"
        else:
            analysis["routing_quality"] = "poor"
            analysis["issues"].append(f"Strategy mismatch: got {selected_strategy}, expected {expected_strategy}")
        
        return analysis
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test cases and return summary."""
        suite_header = "🚀 AGENTIC RETRIEVER TEST SUITE\n" + "=" * 80 + "\nTesting the agentic retrieval system with predefined questions\nbased on candidate profile data from the example folder.\n" + "=" * 80
        print(suite_header)
        
        # Add header to markdown
        self._add_to_markdown("# Agentic Retriever Test Suite Results\n\n")
        self._add_to_markdown(f"**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        self._add_to_markdown("Testing the agentic retrieval system with predefined questions based on candidate profile data.\n\n")
        
        start_time = time.time()
        
        # Run each test
        for test_case in self.test_questions:
            try:
                self.run_single_test(test_case)
                time.sleep(1)  # Brief pause between tests
            except Exception as e:
                error_msg = f"❌ Test {test_case['id']} failed with error: {str(e)}"
                print(error_msg)
                self._add_to_markdown(f"**ERROR in Test {test_case['id']}:** {str(e)}\n\n")
                self.results.append({
                    **test_case,
                    "result": {"error": str(e), "response": None, "metadata": {}},
                    "analysis": {"success": False, "issues": [f"Test execution error: {str(e)}"]},
                    "duration": 0,
                    "timestamp": time.time()
                })
        
        total_duration = time.time() - start_time
        
        # Generate summary
        summary = self._generate_summary(total_duration)
        self._print_summary(summary)
        
        # Add summary to markdown
        self._add_summary_to_markdown(summary)
        
        # Save markdown file
        if self.save_markdown:
            self._save_markdown_file()
        
        return summary
    
    def _generate_summary(self, total_duration: float) -> Dict[str, Any]:
        """Generate test summary statistics."""
        if not self.results:
            return {"error": "No test results available"}
        
        successful_tests = [r for r in self.results if r["analysis"]["success"]]
        failed_tests = [r for r in self.results if not r["analysis"]["success"]]
        
        avg_quality = sum(r["analysis"]["quality_score"] for r in successful_tests) / len(successful_tests) if successful_tests else 0
        avg_coverage = sum(r["analysis"]["topic_coverage"] for r in successful_tests) / len(successful_tests) if successful_tests else 0
        avg_duration = sum(r["duration"] for r in self.results) / len(self.results)
        
        # Strategy distribution analysis
        strategy_distribution = {}
        routing_quality_distribution = {}
        
        for result in self.results:
            strategy = result["result"]["metadata"].get("strategy", "unknown")
            strategy_distribution[strategy] = strategy_distribution.get(strategy, 0) + 1
            
            routing_quality = result["analysis"].get("routing_quality", "unknown")
            routing_quality_distribution[routing_quality] = routing_quality_distribution.get(routing_quality, 0) + 1
        
        return {
            "total_tests": len(self.results),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "success_rate": len(successful_tests) / len(self.results),
            "average_quality_score": avg_quality,
            "average_topic_coverage": avg_coverage,
            "average_test_duration": avg_duration,
            "total_duration": total_duration,
            "issues_by_category": self._categorize_issues(),
            "strategy_distribution": strategy_distribution,
            "routing_quality_distribution": routing_quality_distribution
        }
    
    def _categorize_issues(self) -> Dict[str, int]:
        """Categorize and count issues across all tests."""
        issue_counts = {}
        for result in self.results:
            for issue in result["analysis"].get("issues", []):
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        return issue_counts
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print the test summary."""
        print(f"\n{'='*80}")
        print("📊 TEST SUITE SUMMARY")
        print(f"{'='*80}")
        
        if "error" in summary:
            print(f"❌ {summary['error']}")
            return
        
        print(f"📈 Overall Results:")
        print(f"   • Total Tests: {summary['total_tests']}")
        print(f"   • Successful: {summary['successful_tests']} ✅")
        print(f"   • Failed: {summary['failed_tests']} ❌")
        print(f"   • Success Rate: {summary['success_rate']:.1%}")
        
        print(f"\n🎯 Quality Metrics:")
        print(f"   • Average Quality Score: {summary['average_quality_score']:.1f}/5")
        print(f"   • Average Topic Coverage: {summary['average_topic_coverage']:.1%}")
        print(f"   • Average Test Duration: {summary['average_test_duration']:.2f}s")
        print(f"   • Total Suite Duration: {summary['total_duration']:.2f}s")
        
        if summary['strategy_distribution']:
            print(f"\n🎯 Strategy Usage:")
            for strategy, count in summary['strategy_distribution'].items():
                percentage = (count / summary['total_tests']) * 100
                print(f"   • {strategy}: {count} ({percentage:.1f}%)")
        
        if summary['routing_quality_distribution']:
            print(f"\n🧭 Routing Quality:")
            for quality, count in summary['routing_quality_distribution'].items():
                percentage = (count / summary['total_tests']) * 100
                print(f"   • {quality}: {count} ({percentage:.1f}%)")
        
        if summary['issues_by_category']:
            print(f"\n⚠️  Common Issues:")
            for issue, count in summary['issues_by_category'].items():
                print(f"   • {issue}: {count} test(s)")
        
        # Enhanced overall assessment
        excellent_routing = summary['routing_quality_distribution'].get('excellent', 0) / summary['total_tests']
        strategy_variety = len(summary['strategy_distribution'])
        
        if summary['success_rate'] >= 0.8 and summary['average_quality_score'] >= 3.5 and excellent_routing >= 0.6:
            print(f"\n🎉 EXCELLENT! The agentic retriever is performing very well.")
        elif summary['success_rate'] >= 0.6 and summary['average_quality_score'] >= 2.5 and strategy_variety >= 3:
            print(f"\n👍 GOOD! The agentic retriever is working well with room for improvement.")
        elif summary['success_rate'] >= 0.4:
            print(f"\n🔧 FAIR! The system is functional but needs optimization.")
        else:
            print(f"\n⚠️  NEEDS IMPROVEMENT! Consider reviewing the retrieval configuration.")
    
    def _add_summary_to_markdown(self, summary: Dict[str, Any]):
        """Add summary to markdown content."""
        if "error" in summary:
            self._add_to_markdown(f"## Summary\n\n❌ {summary['error']}\n")
            return
        
        summary_md = "## Test Suite Summary\n\n"
        summary_md += "### Overall Results\n"
        summary_md += f"- **Total Tests:** {summary['total_tests']}\n"
        summary_md += f"- **Successful:** {summary['successful_tests']} ✅\n"
        summary_md += f"- **Failed:** {summary['failed_tests']} ❌\n"
        summary_md += f"- **Success Rate:** {summary['success_rate']:.1%}\n\n"
        
        summary_md += "### Quality Metrics\n"
        summary_md += f"- **Average Quality Score:** {summary['average_quality_score']:.1f}/5\n"
        summary_md += f"- **Average Topic Coverage:** {summary['average_topic_coverage']:.1%}\n"
        summary_md += f"- **Average Test Duration:** {summary['average_test_duration']:.2f}s\n"
        summary_md += f"- **Total Suite Duration:** {summary['total_duration']:.2f}s\n\n"
        
        if summary['issues_by_category']:
            summary_md += "### Common Issues\n"
            for issue, count in summary['issues_by_category'].items():
                summary_md += f"- **{issue}:** {count} test(s)\n"
            summary_md += "\n"
        
        # Overall assessment
        if summary['success_rate'] >= 0.8 and summary['average_quality_score'] >= 3.5:
            summary_md += "### Assessment\n🎉 **EXCELLENT!** The agentic retriever is performing very well.\n"
        elif summary['success_rate'] >= 0.6 and summary['average_quality_score'] >= 2.5:
            summary_md += "### Assessment\n👍 **GOOD!** The agentic retriever is working well with room for improvement.\n"
        else:
            summary_md += "### Assessment\n⚠️ **NEEDS IMPROVEMENT!** Consider reviewing the retrieval configuration.\n"
        
        self._add_to_markdown(summary_md)
    
    def _save_markdown_file(self):
        """Save the markdown content to a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure attached_assets directory exists
        assets_dir = Path("attached_assets")
        assets_dir.mkdir(exist_ok=True)
        
        filename = assets_dir / f"test_results_{timestamp}.md"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.markdown_content))
            print(f"\n📄 Test results saved to: {filename}")
        except Exception as e:
            print(f"\n❌ Error saving markdown file: {e}")


def main():
    """Main function to run the retriever tests."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test the Agentic Retriever CLI with predefined questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_retriever.py                    # Run all tests
  python test_retriever.py --test_id 1        # Run specific test
  python test_retriever.py --category "Basic Vector"  # Run tests by category
  python test_retriever.py --no_markdown      # Skip markdown output
  python test_retriever.py --quick            # Run only first 5 tests (quick check)
  python test_retriever.py --diagnostic       # Run diagnostic tests only
        """
    )
    
    parser.add_argument(
        "--test_id",
        type=int,
        help="Run a specific test by ID (1-5)"
    )
    
    parser.add_argument(
        "--category",
        help="Run tests from a specific category"
    )
    
    parser.add_argument(
        "--api_key",
        help="OpenAI API key (uses environment variable if not provided)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--no_markdown",
        action="store_true",
        help="Skip saving results to markdown file"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only first 5 tests for quick validation"
    )
    
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Run diagnostic tests (simple queries to check basic functionality)"
    )
    
    args = parser.parse_args()
    
    # Initialize test suite
    test_suite = RetrieverTestSuite(
        api_key=args.api_key, 
        save_markdown=not args.no_markdown
    )
    
    if args.test_id:
        # Run specific test
        test_case = next((t for t in test_suite.test_questions if t["id"] == args.test_id), None)
        if test_case:
            test_suite.run_single_test(test_case)
            if test_suite.save_markdown:
                test_suite._save_markdown_file()
        else:
            print(f"❌ Test ID {args.test_id} not found. Available IDs: 1-5")
    
    elif args.category:
        # Run tests by category
        matching_tests = [t for t in test_suite.test_questions if t["category"].lower() == args.category.lower()]
        if matching_tests:
            # Add header for category tests
            test_suite._add_to_markdown(f"# {args.category} Tests\n\n")
            for test_case in matching_tests:
                test_suite.run_single_test(test_case)
                time.sleep(1)
            if test_suite.save_markdown:
                test_suite._save_markdown_file()
        else:
            available_categories = list(set(t["category"] for t in test_suite.test_questions))
            print(f"❌ Category '{args.category}' not found. Available categories: {', '.join(available_categories)}")
    
    elif args.quick:
        # Run first 5 tests only
        test_suite._add_to_markdown(f"# Quick Test Results\n\n")
        for test_case in test_suite.test_questions[:5]:
            test_suite.run_single_test(test_case)
            time.sleep(1)
        if test_suite.save_markdown:
            test_suite._save_markdown_file()
    
    elif args.diagnostic:
        # Run only basic diagnostic tests
        diagnostic_tests = [t for t in test_suite.test_questions if "Basic Vector" in t["category"] or "Router" in t["category"]]
        test_suite._add_to_markdown(f"# Diagnostic Test Results\n\n")
        for test_case in diagnostic_tests:
            test_suite.run_single_test(test_case)
            time.sleep(1)
        if test_suite.save_markdown:
            test_suite._save_markdown_file()
    
    else:
        # Run all tests
        test_suite.run_all_tests()


if __name__ == "__main__":
    main()