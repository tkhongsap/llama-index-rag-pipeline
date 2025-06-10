#!/usr/bin/env python
"""
Demo script for Google Maps Location Integration features

Showcases the new location display functionality in the iLand CLI operations.
"""

import sys
from pathlib import Path
from unittest.mock import Mock

# Add the current directory to the path to import retrieval modules
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from retrieval.cli_operations import iLandCLIOperations


def create_sample_results():
    """Create sample results with location data for demonstration."""
    sample_results = [
        {
            'rank': 1,
            'score': 0.823,
            'text': 'ที่ดิน น.ส.3 ติดถนนสายหลัก 121 จังหวัดเชียงใหม่ อำเภอเมืองเชียงใหม่ ตำบลศรีภูมิ เนื้อที่ 2 ไร่ 1 งาน 50 ตารางวา...',
            'metadata': {
                'province': 'เชียงใหม่',
                'district': 'เมืองเชียงใหม่',
                'subdistrict': 'ศรีภูมิ',
                'google_maps_url': 'https://maps.google.com/maps?q=18.7883,98.9853',
                'latitude': 18.7883,
                'longitude': 98.9853,
                'coordinates_formatted': '18.788300, 98.985300',
                'deed_holding_type': 'น.ส.3'
            }
        },
        {
            'rank': 2,
            'score': 0.756,
            'text': 'โฉนดที่ดิน เลขที่ 12345 ติดถนนซุปเปอร์ไฮเวย์ จังหวัดเชียงใหม่ อำเภอสันทราย เนื้อที่ 5 ไร่...',
            'metadata': {
                'province': 'เชียงใหม่',
                'district': 'สันทราย',
                'latitude': 18.7456,
                'longitude': 99.0234,
                'deed_holding_type': 'โฉนดที่ดิน'
            }
        },
        {
            'rank': 3,
            'score': 0.692,
            'text': 'ที่ดินติดถนนใหญ่ จังหวัดเชียงใหม่ อำเภอหางดง ตำบลหางดง เนื้อที่ 1 ไร่ 2 งาน...',
            'metadata': {
                'province': 'เชียงใหม่',
                'district': 'หางดง',
                'subdistrict': 'หางดง',
                'coordinates_formatted': '18.689200, 98.926700',
                'deed_holding_type': 'น.ส.3'
            }
        }
    ]
    return sample_results


def create_sample_nodes():
    """Create sample nodes for RAG sources demonstration."""
    class MockNode:
        def __init__(self, text, metadata):
            self.text = text
            self.metadata = metadata
    
    class MockScoredNode:
        def __init__(self, node, score):
            self.node = node
            self.score = score
    
    nodes = [
        MockScoredNode(
            MockNode(
                'ที่ดิน น.ส.3 ติดถนนสายหลัก 121 จังหวัดเชียงใหม่ อำเภอเมืองเชียงใหม่ ตำบลศรีภูมิ เนื้อที่ 2 ไร่ 1 งาน 50 ตารางวา ได้มาเมื่อ 15 มกราคม 2565',
                {
                    'province': 'เชียงใหม่',
                    'district': 'เมืองเชียงใหม่',
                    'subdistrict': 'ศรีภูมิ',
                    'google_maps_url': 'https://maps.google.com/maps?q=18.7883,98.9853',
                    'latitude': 18.7883,
                    'longitude': 98.9853,
                    'coordinates_formatted': '18.788300, 98.985300',
                    'deed_holding_type': 'น.ส.3'
                }
            ),
            0.823
        ),
        MockScoredNode(
            MockNode(
                'โฉนดที่ดิน เลขที่ 12345 ติดถนนซุปเปอร์ไฮเวย์ จังหวัดเชียงใหม่ อำเภอสันทราย เนื้อที่ 5 ไร่ ประเภทที่อยู่อาศัย',
                {
                    'province': 'เชียงใหม่',
                    'district': 'สันทราย',
                    'latitude': 18.7456,
                    'longitude': 99.0234,
                    'deed_holding_type': 'โฉนดที่ดิน'
                }
            ),
            0.756
        ),
        MockScoredNode(
            MockNode(
                'ที่ดินติดถนนใหญ่ จังหวัดเชียงใหม่ อำเภอหางดง ตำบลหางดง เนื้อที่ 1 ไร่ 2 งาน ประเภทเกษตรกรรม',
                {
                    'province': 'เชียงใหม่',
                    'district': 'หางดง',
                    'subdistrict': 'หางดง',
                    'coordinates_formatted': '18.689200, 98.926700',
                    'latitude': 18.6892,
                    'longitude': 98.9267,
                    'deed_holding_type': 'น.ส.3'
                }
            ),
            0.692
        )
    ]
    return nodes


def demo_location_features():
    """Demonstrate the new location integration features."""
    print("🌟 Google Maps Location Integration Demo")
    print("=" * 50)
    
    # Create CLI operations instance
    mock_router = Mock()
    cli_ops = iLandCLIOperations(router=mock_router, adapters={})
    
    # Demo 1: Retrieved Documents with Location Information
    print("\n📄 Demo 1: Enhanced Retrieved Documents Display")
    print("-" * 45)
    
    sample_results = create_sample_results()
    cli_ops._print_retrieved_documents(sample_results)
    
    # Demo 2: RAG Sources with Location Information
    print("\n📚 Demo 2: Enhanced RAG Sources Display")
    print("-" * 40)
    
    sample_nodes = create_sample_nodes()
    cli_ops._show_rag_sources(sample_nodes)
    
    # Demo 3: Primary Location JSON Extraction (NEW)
    print("\n🗺️  Demo 3: Primary Location JSON for Mapping Integration")
    print("-" * 55)
    
    primary_location = cli_ops._extract_primary_location_json(sample_nodes)
    if primary_location:
        print("Primary Location Data (for mapping backend):")
        import json
        print(json.dumps(primary_location, indent=2, ensure_ascii=False))
    else:
        print("No location data available")
    
    # Demo 4: Location Information Extraction
    print("\n🔍 Demo 4: Location Information Extraction")
    print("-" * 45)
    
    test_metadata = {
        'province': 'กรุงเทพมหานคร',
        'district': 'บางกะปิ',
        'subdistrict': 'หัวหมาก',
        'google_maps_url': 'https://maps.google.com/maps?q=13.7563,100.5014',
        'latitude': 13.7563,
        'longitude': 100.5014,
        'coordinates_formatted': '13.756300, 100.501400'
    }
    
    location_info = cli_ops._extract_location_info(test_metadata)
    print(f"Location Display: {location_info['location_display']}")
    print(f"Maps Link: {location_info['maps_link']}")
    print(f"Maps Display: {location_info['maps_display']}")
    print(f"Coordinates: {location_info['coordinates']}")
    
    # Demo 5: Unique Locations Collection
    print("\n📍 Demo 5: Unique Locations Collection")
    print("-" * 40)
    
    unique_locations = cli_ops._collect_unique_locations(sample_results)
    print(f"Found {len(unique_locations)} unique locations:")
    for i, location in enumerate(unique_locations, 1):
        print(f"  {i}. {location['display']}: {location['link']}")
    
    print("\n✅ Demo completed successfully!")
    print("\nFeatures demonstrated:")
    print("  📍 Location hierarchy display (Province > District > Subdistrict)")
    print("  🗺️  Google Maps URL integration")
    print("  📍 Coordinate fallback when Maps URL unavailable")
    print("  🎯 Unique location collection for related locations")
    print("  🌐 Unicode Thai character support")
    print("  ⚡ Graceful handling of missing location data")
    print("  🗺️  Primary location JSON extraction for mapping integration")


if __name__ == '__main__':
    demo_location_features() 