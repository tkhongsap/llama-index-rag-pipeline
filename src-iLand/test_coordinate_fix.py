"""
Test script to verify coordinate extraction fix works with actual metadata structure
"""

import sys
from pathlib import Path

# Add the current directory to the path to import retrieval modules
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from retrieval.cli_operations import iLandCLIOperations

def test_coordinate_extraction():
    """Test coordinate extraction with various metadata formats."""
    print("🔍 Testing Coordinate Extraction Fix")
    print("=" * 50)
    
    cli_ops = iLandCLIOperations(router=None, adapters={})
    
    # Test cases based on the actual document structure
    test_cases = [
        {
            'name': 'Expected format from document',
            'metadata': {
                'Province': 'Chai Nat',
                'District': 'Noen Kham', 
                'Latitude': 14.967,
                'Longitude': 99.907,
                'Coordinates Formatted': '14.967000, 99.907000'
            }
        },
        {
            'name': 'With "**:" prefixes',
            'metadata': {
                'province': '**: Chai Nat',
                'district': '**: Noen Kham',
                'latitude': 14.967,
                'longitude': 99.907
            }
        },
        {
            'name': 'String coordinates',
            'metadata': {
                'Province': 'Chai Nat',
                'District': 'Noen Kham',
                'Latitude': '14.967',
                'Longitude': '99.907'
            }
        },
        {
            'name': 'Lowercase fields',
            'metadata': {
                'province': 'Chai Nat',
                'district': 'Noen Kham',
                'latitude': 14.967,
                'longitude': 99.907
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📍 Testing: {test_case['name']}")
        print("-" * 30)
        print(f"Input metadata: {test_case['metadata']}")
        
        location_info = cli_ops._extract_location_info(test_case['metadata'])
        
        print(f"Location Display: '{location_info['location_display']}'")
        print(f"Coordinates: {location_info['coordinates']}")
        print(f"Maps Link: {location_info['maps_link']}")
        
        # Check if coordinates were extracted
        if location_info['coordinates']:
            print("✅ Coordinates extracted successfully!")
        else:
            print("❌ No coordinates extracted")

def test_primary_location_json():
    """Test primary location JSON extraction."""
    print(f"\n🗺️ Testing Primary Location JSON")
    print("=" * 40)
    
    cli_ops = iLandCLIOperations(router=None, adapters={})
    
    # Create mock nodes with fixed metadata
    class MockNode:
        def __init__(self, metadata):
            self.metadata = metadata
    
    class MockScoredNode:
        def __init__(self, node):
            self.node = node
    
    # Test with the expected document format
    metadata = {
        'Province': 'Chai Nat',
        'District': 'Noen Kham',
        'Latitude': 14.967,
        'Longitude': 99.907,
        'Coordinates Formatted': '14.967000, 99.907000'
    }
    
    mock_nodes = [MockScoredNode(MockNode(metadata))]
    location_json = cli_ops._extract_primary_location_json(mock_nodes)
    
    print(f"Primary Location JSON: {location_json}")
    
    if location_json and 'location' in location_json:
        location = location_json['location']
        print(f"✅ Latitude: {location.get('latitude')}")
        print(f"✅ Longitude: {location.get('longitude')}")
        print(f"✅ Display: {location.get('display')}")
        print(f"✅ Maps URL: {location.get('maps_url')}")
    else:
        print("❌ No location data extracted")
        print("Debug: Let's check why...")
        
        # Debug the extraction step by step
        location_info = cli_ops._extract_location_info(metadata)
        print(f"Location info extracted: {location_info}")
        
        # Check coordinate extraction specifically
        latitude = metadata.get('Latitude')
        longitude = metadata.get('Longitude')
        print(f"Direct Latitude from metadata: {latitude} (type: {type(latitude)})")
        print(f"Direct Longitude from metadata: {longitude} (type: {type(longitude)})")

if __name__ == "__main__":
    test_coordinate_extraction()
    test_primary_location_json() 