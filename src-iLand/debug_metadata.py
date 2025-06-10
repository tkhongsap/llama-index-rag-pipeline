"""
Debug script to examine metadata structure and coordinate availability
"""

import sys
from pathlib import Path
import json
from pprint import pprint

# Add the current directory to the path to import retrieval modules
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def debug_metadata_structure():
    """Debug the metadata structure to understand coordinate field names."""
    print("🔍 Metadata Structure Debug")
    print("=" * 40)
    
    # Let's check what a typical metadata structure might look like
    sample_metadata_variations = [
        {
            'province': 'Chai Nat',
            'district': 'Noen Kham', 
            'latitude': 14.967,
            'longitude': 99.907
        },
        {
            'province': '**: Chai Nat',
            'district': '**: Noen Kham',
            'lat': 14.967,
            'lng': 99.907
        },
        {
            'province': '**: Chai Nat', 
            'district': '**: Noen Kham',
            'coordinates': '14.967,99.907'
        },
        {
            'province': '**: Chai Nat',
            'district': '**: Noen Kham'
            # No coordinates at all
        }
    ]
    
    print("Testing different metadata structures:")
    print("-" * 40)
    
    from retrieval.cli_operations import iLandCLIOperations
    cli_ops = iLandCLIOperations(router=None, adapters={})
    
    for i, metadata in enumerate(sample_metadata_variations, 1):
        print(f"\nSample {i}: {metadata}")
        location_info = cli_ops._extract_location_info(metadata)
        print(f"Result: {location_info}")
        print(f"Display: '{location_info['location_display']}'")
        print(f"Coordinates: {location_info['coordinates']}")
        print(f"Maps Link: {location_info['maps_link']}")

def debug_coordinate_field_variations():
    """Test different possible coordinate field names."""
    print("\n🌍 Testing Coordinate Field Variations")
    print("=" * 40)
    
    from retrieval.cli_operations import iLandCLIOperations
    cli_ops = iLandCLIOperations(router=None, adapters={})
    
    coordinate_variations = [
        {'latitude': 14.967, 'longitude': 99.907},
        {'lat': 14.967, 'lng': 99.907},
        {'lat': 14.967, 'lon': 99.907},
        {'coordinates': '14.967,99.907'},
        {'coords': '14.967,99.907'},
        {'location': '14.967,99.907'},
        {'google_maps_url': 'https://maps.google.com/maps?q=14.967,99.907'},
    ]
    
    base_metadata = {'province': '**: Chai Nat', 'district': '**: Noen Kham'}
    
    for i, coords in enumerate(coordinate_variations, 1):
        metadata = {**base_metadata, **coords}
        print(f"\nVariation {i}: {coords}")
        location_info = cli_ops._extract_location_info(metadata)
        print(f"Coordinates extracted: {location_info['coordinates']}")
        print(f"Maps link: {location_info['maps_link']}")

def create_metadata_inspector():
    """Create a function to inspect actual metadata from CLI operations."""
    print("\n🔍 Metadata Inspector Function")
    print("=" * 40)
    print("""
To debug your actual metadata, add this to your CLI operations:

def debug_actual_metadata(self, nodes):
    if nodes:
        top_metadata = getattr(nodes[0].node, 'metadata', {})
        print("\\n🔍 ACTUAL METADATA DEBUG:")
        print("-" * 30)
        for key, value in top_metadata.items():
            print(f"{key}: {value} (type: {type(value).__name__})")
        print("-" * 30)
        return top_metadata

Then call this in _generate_natural_response() before extracting location info.
    """)

if __name__ == "__main__":
    debug_metadata_structure()
    debug_coordinate_field_variations()
    create_metadata_inspector() 