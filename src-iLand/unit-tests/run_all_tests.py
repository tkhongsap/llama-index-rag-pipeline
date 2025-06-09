#!/usr/bin/env python3
"""
Test runner for all iLand module unit tests.

This script runs all unit tests for the iLand modules and provides
a comprehensive test report.
"""

import unittest
import sys
import time
from pathlib import Path
from datetime import datetime

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all test modules
try:
    from test_data_processing import *
    DATA_PROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import data_processing tests: {e}")
    DATA_PROCESSING_AVAILABLE = False

try:
    from test_docs_embedding import *
    DOCS_EMBEDDING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import docs_embedding tests: {e}")
    DOCS_EMBEDDING_AVAILABLE = False

try:
    from test_load_embedding import *
    LOAD_EMBEDDING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import load_embedding tests: {e}")
    LOAD_EMBEDDING_AVAILABLE = False

try:
    from test_retrieval import *
    RETRIEVAL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import retrieval tests: {e}")
    RETRIEVAL_AVAILABLE = False


def run_module_tests(module_name, test_classes, verbosity=1):
    """Run tests for a specific module."""
    print(f"\n{'='*60}")
    print(f"RUNNING {module_name.upper()} TESTS")
    print(f"{'='*60}")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes for this module
    for test_class in test_classes:
        try:
            tests = loader.loadTestsFromTestCase(test_class)
            suite.addTests(tests)
        except Exception as e:
            print(f"Warning: Could not load tests from {test_class.__name__}: {e}")
    
    if suite.countTestCases() == 0:
        print(f"No tests found for {module_name}")
        return None
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Calculate statistics
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_count = total_tests - failures - errors
    success_rate = (success_count / total_tests * 100) if total_tests > 0 else 0
    execution_time = end_time - start_time
    
    return {
        'module': module_name,
        'total_tests': total_tests,
        'successes': success_count,
        'failures': failures,
        'errors': errors,
        'success_rate': success_rate,
        'execution_time': execution_time,
        'result': result
    }


def print_summary_report(module_results):
    """Print a comprehensive summary report."""
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE TEST SUMMARY REPORT")
    print(f"{'='*80}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Overall statistics
    total_tests = sum(r['total_tests'] for r in module_results if r)
    total_successes = sum(r['successes'] for r in module_results if r)
    total_failures = sum(r['failures'] for r in module_results if r)
    total_errors = sum(r['errors'] for r in module_results if r)
    total_time = sum(r['execution_time'] for r in module_results if r)
    overall_success_rate = (total_successes / total_tests * 100) if total_tests > 0 else 0
    
    print(f"OVERALL STATISTICS:")
    print(f"{'─' * 40}")
    print(f"Total Tests:      {total_tests}")
    print(f"Successes:        {total_successes}")
    print(f"Failures:         {total_failures}")
    print(f"Errors:           {total_errors}")
    print(f"Success Rate:     {overall_success_rate:.1f}%")
    print(f"Total Time:       {total_time:.2f} seconds")
    print()
    
    # Module breakdown
    print(f"MODULE BREAKDOWN:")
    print(f"{'─' * 40}")
    print(f"{'Module':<20} {'Tests':<8} {'Success':<8} {'Fail':<6} {'Error':<6} {'Rate':<8} {'Time':<8}")
    print(f"{'─' * 80}")
    
    for result in module_results:
        if result:
            print(f"{result['module']:<20} "
                  f"{result['total_tests']:<8} "
                  f"{result['successes']:<8} "
                  f"{result['failures']:<6} "
                  f"{result['errors']:<6} "
                  f"{result['success_rate']:<7.1f}% "
                  f"{result['execution_time']:<7.2f}s")
    
    print()
    
    # Detailed failure/error reporting
    has_issues = any(r and (r['failures'] > 0 or r['errors'] > 0) for r in module_results)
    
    if has_issues:
        print(f"DETAILED ISSUE REPORT:")
        print(f"{'─' * 40}")
        
        for result in module_results:
            if result and (result['failures'] > 0 or result['errors'] > 0):
                print(f"\n{result['module'].upper()} ISSUES:")
                
                # Print failures
                if result['failures'] > 0:
                    print(f"\nFailures ({len(result['result'].failures)}):")
                    for i, (test, traceback) in enumerate(result['result'].failures, 1):
                        print(f"  {i}. {test}")
                        # Print first few lines of traceback
                        traceback_lines = traceback.split('\n')[:3]
                        for line in traceback_lines:
                            if line.strip():
                                print(f"     {line}")
                
                # Print errors
                if result['errors'] > 0:
                    print(f"\nErrors ({len(result['result'].errors)}):")
                    for i, (test, traceback) in enumerate(result['result'].errors, 1):
                        print(f"  {i}. {test}")
                        # Print first few lines of traceback
                        traceback_lines = traceback.split('\n')[:3]
                        for line in traceback_lines:
                            if line.strip():
                                print(f"     {line}")
    
    # Recommendations
    print(f"\nRECOMMENDations:")
    print(f"{'─' * 40}")
    
    if overall_success_rate >= 95:
        print("✅ Excellent! All modules are well tested.")
    elif overall_success_rate >= 80:
        print("✅ Good test coverage. Review and fix failing tests.")
    elif overall_success_rate >= 60:
        print("⚠️  Moderate test coverage. Significant improvements needed.")
    else:
        print("❌ Poor test coverage. Major test improvements required.")
    
    if total_failures > 0:
        print(f"🔧 Fix {total_failures} failing test(s)")
    
    if total_errors > 0:
        print(f"🐛 Resolve {total_errors} error(s) in test setup/execution")
    
    print(f"\n📊 Test execution completed in {total_time:.2f} seconds")


def main():
    """Main test execution function."""
    print("🧪 iLAND MODULE COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print(f"Starting test execution at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define test modules and their test classes
    test_modules = []
    
    if DATA_PROCESSING_AVAILABLE:
        # Get all data processing test classes
        data_processing_classes = [
            TestFieldMapping, TestDatasetConfig, TestSimpleDocument,
            TestDocumentProcessor, TestFileOutputManager, TestCSVAnalyzer,
            TestDatasetConfigManager, TestIntegration
        ]
        test_modules.append(('data_processing', data_processing_classes))
    
    if DOCS_EMBEDDING_AVAILABLE:
        # Get all docs embedding test classes
        docs_embedding_classes = [
            TestiLandDocumentLoader, TestiLandMetadataExtractor,
            TestEmbeddingStorage, TestEmbeddingConfiguration,
            TestEmbeddingProcessor, TestIntegration
        ]
        
        # Add conditional test classes if available
        if 'TestBatchEmbeddingPipeline' in globals():
            docs_embedding_classes.append(TestBatchEmbeddingPipeline)
        if 'TestBGEEmbeddingProcessor' in globals():
            docs_embedding_classes.append(TestBGEEmbeddingProcessor)
        
        test_modules.append(('docs_embedding', docs_embedding_classes))
    
    if LOAD_EMBEDDING_AVAILABLE:
        # Get all load embedding test classes
        load_embedding_classes = [
            TestEmbeddingConfig, TestFilterConfig, TestiLandEmbeddingLoader,
            TestiLandIndexReconstructor, TestValidation, TestUtils,
            TestIntegration
        ]
        test_modules.append(('load_embedding', load_embedding_classes))
    
    if RETRIEVAL_AVAILABLE:
        # Get all retrieval test classes
        retrieval_classes = [
            TestBaseRetrieverAdapter, TestVectorRetrieverAdapter,
            TestMetadataRetrieverAdapter, TestHybridRetrieverAdapter,
            TestSummaryRetrieverAdapter, TestRecursiveRetrieverAdapter,
            TestChunkDecouplingRetrieverAdapter, TestSectionRetrieverAdapter,
            TestiLandQueryRouter, TestRetrievalCache,
            TestParallelRetrieverExecutor, TestFastMetadataIndex,
            TestIndexClassifier, TestIntegration
        ]
        test_modules.append(('retrieval', retrieval_classes))
    
    if not test_modules:
        print("❌ No test modules available for execution!")
        return 1
    
    # Run tests for each module
    module_results = []
    for module_name, test_classes in test_modules:
        result = run_module_tests(module_name, test_classes, verbosity=2)
        module_results.append(result)
    
    # Print comprehensive summary
    print_summary_report(module_results)
    
    # Determine exit code
    total_failures = sum(r['failures'] for r in module_results if r)
    total_errors = sum(r['errors'] for r in module_results if r)
    
    if total_failures > 0 or total_errors > 0:
        print(f"\n❌ Tests completed with issues. Exit code: 1")
        return 1
    else:
        print(f"\n✅ All tests passed successfully! Exit code: 0")
        return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)