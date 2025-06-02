#!/usr/bin/env python3
"""
Enhanced Test script for the Agentic Retriever CLI

This script tests the agentic retrieval system with diverse questions
designed to trigger different retrieval strategies.
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


class EnhancedRetrieverTestSuite:
    """Enhanced test suite for the agentic retriever with diverse questions."""
    
    def __init__(self, api_key: str = None, save_markdown: bool = True):
        """Initialize the test suite."""
        self.api_key = api_key
        self.save_markdown = save_markdown
        self.test_questions = self._create_test_questions()
        self.results = []
        self.markdown_content = []
    
    def _create_test_questions(self) -> List[Dict[str, Any]]:
        """Create diverse test questions designed to trigger different retrieval strategies."""
        return [
            # Basic semantic queries (should trigger vector strategy)
            {
                "id": 1,
                "category": "Demographics - Semantic",
                "question": "What age groups are represented in the candidate profiles?",
                "description": "Test basic semantic retrieval of demographic information",
                "expected_topics": ["age", "demographics", "mid-career", "senior professionals"],
                "expected_strategy": "vector",
                "query_type": "semantic"
            },
            {
                "id": 2,
                "category": "Education - Semantic",
                "question": "What are the most common educational backgrounds and degrees among candidates?",
                "description": "Test semantic retrieval of educational information and degree patterns",
                "expected_topics": ["education", "degree", "major", "university", "bachelor", "master"],
                "expected_strategy": "vector",
                "query_type": "semantic"
            },
            
            # Specific filtering queries (should trigger metadata strategy)
            {
                "id": 3,
                "category": "Compensation - Filtered",
                "question": "Show me all candidates with salary above 50,000 THB in the Human Resources job family",
                "description": "Test metadata-filtered retrieval with specific criteria",
                "expected_topics": ["salary", "50000", "human resources", "THB", "job family"],
                "expected_strategy": "metadata",
                "query_type": "filtered"
            },
            {
                "id": 4,
                "category": "Geographic - Filtered",
                "question": "Find candidates located specifically in กรุงเทพมหานคร (Bangkok) region R1",
                "description": "Test metadata filtering for specific geographic criteria",
                "expected_topics": ["กรุงเทพมหานคร", "bangkok", "R1", "region"],
                "expected_strategy": "metadata",
                "query_type": "filtered"
            },
            
            # Hierarchical/complex queries (should trigger recursive strategy)
            {
                "id": 5,
                "category": "Career Progression - Hierarchical",
                "question": "Analyze the career progression patterns from entry-level to senior positions across different industries",
                "description": "Test recursive retrieval for hierarchical career analysis",
                "expected_topics": ["career", "progression", "entry-level", "senior", "industries"],
                "expected_strategy": "recursive",
                "query_type": "hierarchical"
            },
            {
                "id": 6,
                "category": "Education Hierarchy",
                "question": "Break down the educational pathways from bachelor's to master's degrees and their impact on career advancement",
                "description": "Test recursive retrieval for educational hierarchy analysis",
                "expected_topics": ["bachelor", "master", "educational", "pathways", "career advancement"],
                "expected_strategy": "recursive",
                "query_type": "hierarchical"
            },
            
            # Mixed semantic and exact queries (should trigger hybrid strategy if available)
            {
                "id": 7,
                "category": "Compensation Analysis - Hybrid",
                "question": "What is the exact salary range for 'Training Officer' positions and how does it compare to similar roles?",
                "description": "Test hybrid retrieval combining exact job title matching with semantic comparison",
                "expected_topics": ["training officer", "salary range", "similar roles", "compare"],
                "expected_strategy": "hybrid",
                "query_type": "hybrid"
            },
            {
                "id": 8,
                "category": "Industry Comparison - Hybrid",
                "question": "Compare the 'Manufacturing' industry compensation with 'Oil' industry for similar experience levels",
                "description": "Test hybrid retrieval for exact industry matching with semantic comparison",
                "expected_topics": ["manufacturing", "oil", "industry", "compensation", "experience"],
                "expected_strategy": "hybrid",
                "query_type": "hybrid"
            },
            
            # Complex multi-step queries (should trigger planner strategy if available)
            {
                "id": 9,
                "category": "Complex Analysis - Planning",
                "question": "First identify the top 3 industries by candidate count, then analyze their average compensation, and finally compare their educational requirements",
                "description": "Test query planning for multi-step analysis",
                "expected_topics": ["top industries", "candidate count", "average compensation", "educational requirements"],
                "expected_strategy": "planner",
                "query_type": "multi-step"
            },
            {
                "id": 10,
                "category": "Regional Analysis - Planning",
                "question": "Determine which provinces have the highest concentration of candidates, analyze their job families, and identify compensation trends by region",
                "description": "Test query planning for complex regional analysis",
                "expected_topics": ["provinces", "concentration", "job families", "compensation trends", "region"],
                "expected_strategy": "planner",
                "query_type": "multi-step"
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
        
        question_info = f"❓ Question: {test_case['question']}\n📝 Description: {test_case['description']}\n🎯 Expected Topics: {', '.join(test_case['expected_topics'])}\n🔧 Expected Strategy: {test_case.get('expected_strategy', 'unknown')}\n📊 Query Type: {test_case.get('query_type', 'unknown')}"
        print(question_info)
        print(f"{'='*80}")
        
        # Add to markdown
        self._add_to_markdown(f"## TEST {test_case['id']}: {test_case['category']}\n")
        self._add_to_markdown(f"**Question:** {test_case['question']}\n")
        self._add_to_markdown(f"**Description:** {test_case['description']}\n")
        self._add_to_markdown(f"**Expected Topics:** {', '.join(test_case['expected_topics'])}\n")
        self._add_to_markdown(f"**Expected Strategy:** {test_case.get('expected_strategy', 'unknown')}\n")
        self._add_to_markdown(f"**Query Type:** {test_case.get('query_type', 'unknown')}\n")
        
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
        actual_strategy = metadata.get('strategy', 'unknown')
        expected_strategy = test_case.get('expected_strategy', 'unknown')
        
        routing_info = f"**Routing Information:**\n"
        routing_info += f"- Index: {metadata.get('index', 'unknown')}\n"
        routing_info += f"- Strategy: {actual_strategy}\n"
        routing_info += f"- Expected Strategy: {expected_strategy}\n"
        routing_info += f"- Strategy Match: {'✅ YES' if actual_strategy == expected_strategy else '❌ NO'}\n"
        routing_info += f"- Latency: {metadata.get('total_time_ms', 0)} ms\n"
        routing_info += f"- Sources: {metadata.get('num_sources', 0)}\n"
        if metadata.get('index_confidence') is not None:
            routing_info += f"- Index Confidence: {metadata['index_confidence']:.3f}\n"
        if metadata.get('strategy_confidence') is not None:
            routing_info += f"- Strategy Confidence: {metadata['strategy_confidence']:.3f}\n"
        
        self._add_to_markdown(routing_info)
        
        # Analyze the result
        analysis = self._analyze_result(result, test_case)
        
        # Check strategy matching
        strategy_match = actual_strategy == expected_strategy
        if not strategy_match:
            analysis['issues'].append(f"Strategy mismatch: expected {expected_strategy}, got {actual_strategy}")
        
        analysis_output = f"\n📊 TEST ANALYSIS:\n   • Success: {'✅ YES' if analysis['success'] else '❌ NO'}\n   • Response Quality: {analysis['quality_score']}/5\n   • Topic Coverage: {analysis['topic_coverage']:.1%}\n   • Strategy Match: {'✅ YES' if strategy_match else '❌ NO'}\n   • Test Duration: {test_duration:.2f}s"
        print(analysis_output)
        
        # Add analysis to markdown
        analysis_md = f"**Analysis:**\n"
        analysis_md += f"- Success: {'✅ YES' if analysis['success'] else '❌ NO'}\n"
        analysis_md += f"- Response Quality: {analysis['quality_score']}/5\n"
        analysis_md += f"- Topic Coverage: {analysis['topic_coverage']:.1%}\n"
        analysis_md += f"- Strategy Match: {'✅ YES' if strategy_match else '❌ NO'}\n"
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
            "strategy_match": strategy_match,
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
            "issues": []
        }
        
        if result["error"]:
            analysis["issues"].append(f"Error: {result['error']}")
            return analysis
        
        if not result["response"]:
            analysis["issues"].append("No response generated")
            return analysis
        
        analysis["success"] = True
        response_text = result["response"].lower()
        
        # Check topic coverage
        expected_topics = test_case["expected_topics"]
        covered_topics = sum(1 for topic in expected_topics if topic.lower() in response_text)
        analysis["topic_coverage"] = covered_topics / len(expected_topics) if expected_topics else 0
        
        # Quality scoring (1-5 scale)
        quality_score = 1  # Base score for successful response
        
        # +1 for good topic coverage
        if analysis["topic_coverage"] >= 0.5:
            quality_score += 1
        
        # +1 for comprehensive response (length check)
        if len(result["response"]) >= 200:
            quality_score += 1
        
        # +1 for fast response
        if result["metadata"].get("total_time_ms", 0) < 5000:
            quality_score += 1
        
        # +1 for having source information
        if result["metadata"].get("num_sources", 0) > 0:
            quality_score += 1
        
        analysis["quality_score"] = min(quality_score, 5)
        
        # Check for potential issues
        if analysis["topic_coverage"] < 0.3:
            analysis["issues"].append("Low topic coverage")
        
        if len(result["response"]) < 100:
            analysis["issues"].append("Response too short")
        
        if result["metadata"].get("total_time_ms", 0) > 10000:
            analysis["issues"].append("Slow response time")
        
        return analysis
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test cases and return summary."""
        suite_header = "🚀 ENHANCED AGENTIC RETRIEVER TEST SUITE\n" + "=" * 80 + "\nTesting the agentic retrieval system with diverse questions\ndesigned to trigger different retrieval strategies.\n" + "=" * 80
        print(suite_header)
        
        # Add header to markdown
        self._add_to_markdown("# Enhanced Agentic Retriever Test Suite Results\n\n")
        self._add_to_markdown(f"**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        self._add_to_markdown("Testing the agentic retrieval system with diverse questions designed to trigger different retrieval strategies.\n\n")
        
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
                    "strategy_match": False,
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
        strategy_matches = [r for r in self.results if r.get("strategy_match", False)]
        
        avg_quality = sum(r["analysis"]["quality_score"] for r in successful_tests) / len(successful_tests) if successful_tests else 0
        avg_coverage = sum(r["analysis"]["topic_coverage"] for r in successful_tests) / len(successful_tests) if successful_tests else 0
        avg_duration = sum(r["duration"] for r in self.results) / len(self.results)
        
        # Strategy analysis
        strategy_stats = {}
        for result in self.results:
            actual_strategy = result["result"].get("metadata", {}).get("strategy", "unknown")
            expected_strategy = result.get("expected_strategy", "unknown")
            
            if expected_strategy not in strategy_stats:
                strategy_stats[expected_strategy] = {"total": 0, "matched": 0, "actual_strategies": {}}
            
            strategy_stats[expected_strategy]["total"] += 1
            if actual_strategy == expected_strategy:
                strategy_stats[expected_strategy]["matched"] += 1
            
            if actual_strategy not in strategy_stats[expected_strategy]["actual_strategies"]:
                strategy_stats[expected_strategy]["actual_strategies"][actual_strategy] = 0
            strategy_stats[expected_strategy]["actual_strategies"][actual_strategy] += 1
        
        return {
            "total_tests": len(self.results),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "strategy_matches": len(strategy_matches),
            "success_rate": len(successful_tests) / len(self.results),
            "strategy_match_rate": len(strategy_matches) / len(self.results),
            "average_quality_score": avg_quality,
            "average_topic_coverage": avg_coverage,
            "average_test_duration": avg_duration,
            "total_duration": total_duration,
            "strategy_stats": strategy_stats,
            "issues_by_category": self._categorize_issues()
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
        print("📊 ENHANCED TEST SUITE SUMMARY")
        print(f"{'='*80}")
        
        if "error" in summary:
            print(f"❌ {summary['error']}")
            return
        
        print(f"📈 Overall Results:")
        print(f"   • Total Tests: {summary['total_tests']}")
        print(f"   • Successful: {summary['successful_tests']} ✅")
        print(f"   • Failed: {summary['failed_tests']} ❌")
        print(f"   • Success Rate: {summary['success_rate']:.1%}")
        print(f"   • Strategy Matches: {summary['strategy_matches']} ✅")
        print(f"   • Strategy Match Rate: {summary['strategy_match_rate']:.1%}")
        
        print(f"\n🎯 Quality Metrics:")
        print(f"   • Average Quality Score: {summary['average_quality_score']:.1f}/5")
        print(f"   • Average Topic Coverage: {summary['average_topic_coverage']:.1%}")
        print(f"   • Average Test Duration: {summary['average_test_duration']:.2f}s")
        print(f"   • Total Suite Duration: {summary['total_duration']:.2f}s")
        
        # Strategy analysis
        print(f"\n🔧 Strategy Analysis:")
        for expected_strategy, stats in summary['strategy_stats'].items():
            match_rate = stats['matched'] / stats['total'] if stats['total'] > 0 else 0
            print(f"   • {expected_strategy}: {stats['matched']}/{stats['total']} ({match_rate:.1%})")
            for actual_strategy, count in stats['actual_strategies'].items():
                if actual_strategy != expected_strategy:
                    print(f"     → Got {actual_strategy}: {count} times")
        
        if summary['issues_by_category']:
            print(f"\n⚠️  Common Issues:")
            for issue, count in summary['issues_by_category'].items():
                print(f"   • {issue}: {count} test(s)")
        
        # Overall assessment
        if summary['success_rate'] >= 0.8 and summary['strategy_match_rate'] >= 0.6:
            print(f"\n🎉 EXCELLENT! The agentic retriever is performing very well with good strategy selection.")
        elif summary['success_rate'] >= 0.6 and summary['strategy_match_rate'] >= 0.4:
            print(f"\n👍 GOOD! The agentic retriever is working well but strategy selection needs improvement.")
        else:
            print(f"\n⚠️  NEEDS IMPROVEMENT! Consider reviewing the retrieval configuration and strategy selection.")
    
    def _add_summary_to_markdown(self, summary: Dict[str, Any]):
        """Add summary to markdown content."""
        if "error" in summary:
            self._add_to_markdown(f"## Summary\n\n❌ {summary['error']}\n")
            return
        
        summary_md = "## Enhanced Test Suite Summary\n\n"
        summary_md += "### Overall Results\n"
        summary_md += f"- **Total Tests:** {summary['total_tests']}\n"
        summary_md += f"- **Successful:** {summary['successful_tests']} ✅\n"
        summary_md += f"- **Failed:** {summary['failed_tests']} ❌\n"
        summary_md += f"- **Success Rate:** {summary['success_rate']:.1%}\n"
        summary_md += f"- **Strategy Matches:** {summary['strategy_matches']} ✅\n"
        summary_md += f"- **Strategy Match Rate:** {summary['strategy_match_rate']:.1%}\n\n"
        
        summary_md += "### Quality Metrics\n"
        summary_md += f"- **Average Quality Score:** {summary['average_quality_score']:.1f}/5\n"
        summary_md += f"- **Average Topic Coverage:** {summary['average_topic_coverage']:.1%}\n"
        summary_md += f"- **Average Test Duration:** {summary['average_test_duration']:.2f}s\n"
        summary_md += f"- **Total Suite Duration:** {summary['total_duration']:.2f}s\n\n"
        
        # Strategy analysis
        summary_md += "### Strategy Analysis\n"
        for expected_strategy, stats in summary['strategy_stats'].items():
            match_rate = stats['matched'] / stats['total'] if stats['total'] > 0 else 0
            summary_md += f"- **{expected_strategy}:** {stats['matched']}/{stats['total']} ({match_rate:.1%})\n"
            for actual_strategy, count in stats['actual_strategies'].items():
                if actual_strategy != expected_strategy:
                    summary_md += f"  - Got {actual_strategy}: {count} times\n"
        summary_md += "\n"
        
        if summary['issues_by_category']:
            summary_md += "### Common Issues\n"
            for issue, count in summary['issues_by_category'].items():
                summary_md += f"- **{issue}:** {count} test(s)\n"
            summary_md += "\n"
        
        # Overall assessment
        if summary['success_rate'] >= 0.8 and summary['strategy_match_rate'] >= 0.6:
            summary_md += "### Assessment\n🎉 **EXCELLENT!** The agentic retriever is performing very well with good strategy selection.\n"
        elif summary['success_rate'] >= 0.6 and summary['strategy_match_rate'] >= 0.4:
            summary_md += "### Assessment\n👍 **GOOD!** The agentic retriever is working well but strategy selection needs improvement.\n"
        else:
            summary_md += "### Assessment\n⚠️ **NEEDS IMPROVEMENT!** Consider reviewing the retrieval configuration and strategy selection.\n"
        
        self._add_to_markdown(summary_md)
    
    def _save_markdown_file(self):
        """Save the markdown content to a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure attached_assets directory exists
        assets_dir = Path("attached_assets")
        assets_dir.mkdir(exist_ok=True)
        
        filename = assets_dir / f"enhanced_test_results_{timestamp}.md"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.markdown_content))
            print(f"\n📄 Enhanced test results saved to: {filename}")
        except Exception as e:
            print(f"\n❌ Error saving markdown file: {e}")


def main():
    """Main function to run the enhanced retriever tests."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhanced Test for the Agentic Retriever CLI with diverse questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_retriever_enhanced.py                    # Run all tests
  python test_retriever_enhanced.py --test_id 1        # Run specific test
  python test_retriever_enhanced.py --category "Semantic"  # Run tests by category
  python test_retriever_enhanced.py --no_markdown      # Skip markdown output
        """
    )
    
    parser.add_argument(
        "--test_id",
        type=int,
        help="Run a specific test by ID (1-10)"
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
    
    args = parser.parse_args()
    
    # Initialize test suite
    test_suite = EnhancedRetrieverTestSuite(
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
            print(f"❌ Test ID {args.test_id} not found. Available IDs: 1-10")
    
    elif args.category:
        # Run tests by category
        matching_tests = [t for t in test_suite.test_questions if args.category.lower() in t["category"].lower()]
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
    
    else:
        # Run all tests
        test_suite.run_all_tests()


if __name__ == "__main__":
    main() 