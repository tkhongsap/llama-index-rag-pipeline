#!/usr/bin/env python3
"""
Quick demo of iLand retrieval strategy testing
"""

import sys
from pathlib import Path

# Add tests directory to path
sys.path.insert(0, str(Path(__file__).parent / 'tests'))

from test_iland_retrieval_strategies import iLandRetrievalStrategyTester

def main():
    print('🧪 Quick iLand Strategy Test Demo')
    print('=' * 50)

    tester = iLandRetrievalStrategyTester()
    router = tester.create_test_router('round_robin')

    # Test a few queries
    test_queries = [
        # --- Basic keyword / vector ------------------------------------------------
        "โฉนดที่ดินในจังหวัดอ่างทอง",
        
        # Simple hybrid (keyword + semantic) with an extra filter
        "ค้นหาโฉนด นส.3 ที่มีกรรมสิทธิ์บริษัทในอำเภอเมืองอ่างทอง",
        
        # Metadata-driven query using explicit coordinates
        "พิกัด 14.5486,100.4514 มีโฉนดไหนบ้าง",
        
        # High-level overview suited to the summary adapter
        "สรุปจำนวนที่ดินทั้งหมดที่ถือครองโดยบริษัทในจังหวัดชัยนาท",
        
        # Procedural/legal explanation (vector first, may escalate to planner)
        "อธิบายขั้นตอนการโอนกรรมสิทธิ์ที่ดินจากบริษัทไปยังบุคคลภายนอก",
        
        # Focused comparison likely to trigger chunk-decoupling
        "เปรียบเทียบพื้นที่ (ไร่-งาน-วา) ระหว่างโฉนด 123 และ 124",
        
        # Recursive need: overview + per-deed drill-down sorted by size
        "สรุปภาพรวมและให้รายละเอียดขนาดพื้นที่ของทุกโฉนดในตำบลโพสะ เรียงตามขนาดจากมากไปน้อย",
        
        # Planner-style multi-step aggregation
        "สำหรับโฉนดทั้งหมดในชุดข้อมูล ให้สร้างรายงาน: (1) กลุ่มตามหมวดหมู่หลัก, (2) รวมจำนวนไร่ต่อกลุ่ม, (3) ระบุโฉนดผืนที่ใหญ่ที่สุดในแต่ละกลุ่ม",
        
        # Hybrid with metadata filters and a date condition
        "ค้นหาโฉนดประเภท 'ทาวน์โฮม' ที่ได้รับการโอนหลังปี 2024 และอยู่ในภาคกลาง",
        
        # Open-ended reasoning—should push the planner to chain tools
        "ถ้าต้องการพัฒนาโครงการที่อยู่อาศัยใหม่ในอำเภอเมืองอ่างทอง ควรเลือกโฉนดใดเป็นตัวเลือกอันดับแรก พร้อมเหตุผล"
    ]

    for i, query in enumerate(test_queries, 1):
        from llama_index.core.schema import QueryBundle
        result = router._retrieve(QueryBundle(query_str=query))
        if result:
            metadata = result[0].node.metadata
            strategy = metadata.get('selected_strategy', 'unknown')
            content_preview = result[0].node.text[:60] + "..." if len(result[0].node.text) > 60 else result[0].node.text
            print(f'{i}. Query: {query[:120]}...')
            print(f'   → Strategy: {strategy}')
            print(f'   → Preview: {content_preview}')
            print()
        else:
            print(f'{i}. {query[:40]}... → No results')

    print('✅ Demo completed successfully!')
    print('\nTo run full tests, use:')
    print('python tests/test_iland_retrieval_strategies.py')

if __name__ == "__main__":
    main() 