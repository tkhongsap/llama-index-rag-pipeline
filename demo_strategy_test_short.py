#!/usr/bin/env python3
"""
Short demo of iLand retrieval strategy testing - quick verification
"""

import sys
from pathlib import Path

# Add tests directory to path
sys.path.insert(0, str(Path(__file__).parent / 'tests'))

def main():
    try:
        from test_iland_retrieval_strategies import iLandRetrievalStrategyTester
        from llama_index.core.schema import QueryBundle
        
        print('🧪 Quick iLand Strategy Test Verification')
        print('=' * 50)

        tester = iLandRetrievalStrategyTester()
        router = tester.create_test_router('round_robin')

        # Test 5 key queries that target different strategies
        test_queries = [
            ("โฉนดที่ดินในจังหวัดอ่างทอง", "vector"),
            ("ค้นหาโฉนด นส.3 ที่มีกรรมสิทธิ์บริษัท", "hybrid"), 
            ("พิกัด 14.5486,100.4514 มีโฉนดไหนบ้าง", "metadata"),
            ("สรุปจำนวนที่ดินทั้งหมดที่ถือครองโดยบริษัท", "planner"),
            ("เปรียบเทียบพื้นที่ (ไร่-งาน-วา) ระหว่างโฉนด", "chunk_decoupling")
        ]

        success_count = 0
        
        for i, (query, expected_strategy) in enumerate(test_queries, 1):
            try:
                result = router._retrieve(QueryBundle(query_str=query))
                if result:
                    metadata = result[0].node.metadata
                    actual_strategy = metadata.get('selected_strategy', 'unknown')
                    
                    status = "✅" if actual_strategy == expected_strategy else "❌"
                    if actual_strategy == expected_strategy:
                        success_count += 1
                    
                    print(f'{status} Test {i}: {query[:50]}...')
                    print(f'    Expected: {expected_strategy} | Got: {actual_strategy}')
                else:
                    print(f'❌ Test {i}: No results for "{query[:50]}..."')
                    
            except Exception as e:
                print(f'❌ Test {i}: Error - {str(e)[:50]}...')

        print(f'\n📊 Results: {success_count}/{len(test_queries)} tests passed')
        
        if success_count == len(test_queries):
            print('🎉 All tests passed! Strategy routing is working correctly.')
        elif success_count >= len(test_queries) // 2:
            print('⚠️  Most tests passed. Some strategies may need tuning.')
        else:
            print('❌ Multiple test failures. Check strategy routing logic.')
            
    except ImportError as e:
        print(f'❌ Import error: {e}')
        print('Make sure llama-index is installed and test files are present.')
    except Exception as e:
        print(f'❌ Unexpected error: {e}')

if __name__ == "__main__":
    main() 