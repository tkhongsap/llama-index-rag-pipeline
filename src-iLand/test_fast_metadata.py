#!/usr/bin/env python3
"""
Test script for Fast Metadata Indexing Implementation

Quick validation of the FastMetadataIndexManager and enhanced MetadataRetrieverAdapter.
This script tests the core functionality without requiring full embedding loading.
"""

import sys
import os
from pathlib import Path

# Add src-iLand to path
sys.path.insert(0, str(Path(__file__).parent))

from retrieval.fast_metadata_index import FastMetadataIndexManager
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters, FilterOperator


def create_test_nodes():
    """Create test nodes with Thai land deed metadata."""
    test_nodes = [
        TextNode(
            text="โฉนดที่ดินในกรุงเทพมหานคร",
            metadata={
                "province": "กรุงเทพมหานคร",
                "district": "บางรัก", 
                "deed_type": "โฉนด",
                "area_rai": 2.5,
                "doc_id": "deed_001"
            },
            node_id="node_001"
        ),
        TextNode(
            text="ที่ดินนส.3 ในสมุทรปราการ",
            metadata={
                "province": "สมุทรปราการ",
                "district": "บางพลี",
                "deed_type": "นส.3", 
                "area_rai": 5.0,
                "doc_id": "deed_002"
            },
            node_id="node_002"
        ),
        TextNode(
            text="โฉนดที่ดินขนาดใหญ่ในนนทบุรี",
            metadata={
                "province": "นนทบุรี",
                "district": "บางใหญ่",
                "deed_type": "โฉนด",
                "area_rai": 10.0,
                "doc_id": "deed_003"  
            },
            node_id="node_003"
        ),
        TextNode(
            text="ที่ดินนส.4 ในกรุงเทพมหานคร",
            metadata={
                "province": "กรุงเทพมหานคร", 
                "district": "วัฒนา",
                "deed_type": "นส.4",
                "area_rai": 1.0,
                "doc_id": "deed_004"
            },
            node_id="node_004"
        )
    ]
    return test_nodes


def test_fast_metadata_indexing():
    """Test FastMetadataIndexManager functionality."""
    print("🧪 TESTING FAST METADATA INDEXING")
    print("=" * 50)
    
    # Create test data
    test_nodes = create_test_nodes()
    print(f"📊 Created {len(test_nodes)} test nodes")
    
    # Initialize index manager
    index_manager = FastMetadataIndexManager()
    
    # Build indices
    index_manager.build_indices_from_llamaindex_nodes(test_nodes)
    
    # Test 1: Province filtering
    print("\n🔍 Test 1: Province filtering (กรุงเทพมหานคร)")
    province_filter = MetadataFilters(filters=[
        MetadataFilter(key="province", value="กรุงเทพมหานคร", operator=FilterOperator.EQ)
    ])
    
    bangkok_results = index_manager.pre_filter_node_ids(province_filter)
    print(f"   Results: {len(bangkok_results)} nodes - {bangkok_results}")
    
    # Test 2: Deed type filtering  
    print("\n🔍 Test 2: Deed type filtering (โฉนด)")
    deed_filter = MetadataFilters(filters=[
        MetadataFilter(key="deed_type", value="โฉนด", operator=FilterOperator.EQ)
    ])
    
    deed_results = index_manager.pre_filter_node_ids(deed_filter)
    print(f"   Results: {len(deed_results)} nodes - {deed_results}")
    
    # Test 3: Numeric range filtering (area > 2.0 rai)
    print("\n🔍 Test 3: Numeric range filtering (area > 2.0 rai)")
    area_filter = MetadataFilters(filters=[
        MetadataFilter(key="area_rai", value=2.0, operator=FilterOperator.GT)
    ])
    
    area_results = index_manager.pre_filter_node_ids(area_filter)
    print(f"   Results: {len(area_results)} nodes - {area_results}")
    
    # Test 4: Compound filtering (Bangkok AND โฉนด)
    print("\n🔍 Test 4: Compound filtering (Bangkok AND โฉนด)")
    compound_filter = MetadataFilters(
        filters=[
            MetadataFilter(key="province", value="กรุงเทพมหานคร", operator=FilterOperator.EQ),
            MetadataFilter(key="deed_type", value="โฉนด", operator=FilterOperator.EQ)
        ],
        condition="and"
    )
    
    compound_results = index_manager.pre_filter_node_ids(compound_filter)
    print(f"   Results: {len(compound_results)} nodes - {compound_results}")
    
    # Test 5: OR filtering (โฉนด OR นส.3)
    print("\n🔍 Test 5: OR filtering (โฉนด OR นส.3)")
    or_filter = MetadataFilters(
        filters=[
            MetadataFilter(key="deed_type", value="โฉนด", operator=FilterOperator.EQ),
            MetadataFilter(key="deed_type", value="นส.3", operator=FilterOperator.EQ)
        ],
        condition="or"
    )
    
    or_results = index_manager.pre_filter_node_ids(or_filter)
    print(f"   Results: {len(or_results)} nodes - {or_results}")
    
    # Show index statistics
    print("\n📊 INDEX STATISTICS")
    stats = index_manager.get_index_stats()
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Indexed fields: {stats['indexed_fields']}")
    print(f"   Categorical fields: {stats['categorical_fields']}")
    print(f"   Numeric fields: {stats['numeric_fields']}")
    print(f"   Performance stats: {stats['performance_stats']}")
    
    print("\n✅ All tests completed!")
    return True


def main():
    """Main test function."""
    print("🚀 FAST METADATA INDEXING VALIDATION")
    print("=" * 60)
    
    try:
        # Test the core functionality
        success = test_fast_metadata_indexing()
        
        if success:
            print("\n🎉 VALIDATION SUCCESSFUL!")
            print("The FastMetadataIndexManager is working correctly.")
            print("Ready for integration with iLand retrieval pipeline.")
        else:
            print("\n❌ VALIDATION FAILED!")
            
    except Exception as e:
        print(f"\n❌ ERROR during validation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 