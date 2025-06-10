"""
Demo: Google Maps Location Integration for Mapping Apps

This script demonstrates how to extract location data from iLand CLI operations
for integration with mapping applications.
"""

import sys
from pathlib import Path
import json

# Add the current directory to the path to import retrieval modules
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from retrieval.cli_operations import iLandCLIOperations


class MappingIntegrationDemo:
    """Demonstrates location data extraction for mapping applications."""
    
    def __init__(self):
        """Initialize demo with mock CLI operations."""
        # In real usage, these would be properly initialized
        self.cli_ops = iLandCLIOperations(
            router=None,  # Would be actual router
            adapters={},  # Would be actual adapters
            response_synthesizer=None  # Would be actual synthesizer
        )
    
    def demo_location_extraction(self):
        """Demo location data extraction from metadata."""
        print("🗺️  Google Maps Location Integration Demo")
        print("=" * 50)
        
        # Sample metadata from land deed documents
        sample_metadata = {
            'province': 'Chai Nat',
            'district': 'Noen Kham',
            'subdistrict': 'Ban Kluai',
            'latitude': 14.967,
            'longitude': 99.907,
            'google_maps_url': 'https://maps.google.com/maps?q=14.967,99.907',
            'deed_holding_type': 'บริษัท',
            'deed_number': '792'
        }
        
        print("\n1. 📍 Location Info Extraction:")
        print("-" * 30)
        location_info = self.cli_ops._extract_location_info(sample_metadata)
        
        print(f"Display: {location_info['location_display']}")
        print(f"Coordinates: {location_info['coordinates']}")
        print(f"Maps Link: {location_info['maps_link']}")
        print(f"Maps Display: {location_info['maps_display']}")
        
        return location_info
    
    def demo_primary_location_json(self):
        """Demo primary location JSON extraction for mapping backends."""
        print("\n2. 📊 Primary Location JSON for Mapping:")
        print("-" * 40)
        
        # Create mock nodes with location data
        class MockNode:
            def __init__(self, metadata):
                self.metadata = metadata
        
        class MockScoredNode:
            def __init__(self, node):
                self.node = node
        
        # Sample data from different locations
        locations = [
            {
                'province': 'Chai Nat',
                'district': 'Noen Kham',
                'latitude': 14.967,
                'longitude': 99.907,
                'google_maps_url': 'https://maps.google.com/maps?q=14.967,99.907'
            },
            {
                'province': 'Bangkok',
                'district': 'Bang Kapi',
                'latitude': 13.7563,
                'longitude': 100.5014,
                'coordinates_formatted': '13.756300, 100.501400'
            }
        ]
        
        for i, metadata in enumerate(locations, 1):
            print(f"\nLocation {i}:")
            mock_nodes = [MockScoredNode(MockNode(metadata))]
            location_json = self.cli_ops._extract_primary_location_json(mock_nodes)
            
            if location_json:
                print(json.dumps(location_json, indent=2, ensure_ascii=False))
            else:
                print("No location data available")
    
    def demo_mapping_app_integration(self):
        """Demo how a mapping app would use the location data."""
        print("\n3. 🗺️  Mapping App Integration Example:")
        print("-" * 40)
        
        # Simulate query results with location data
        mock_results = [
            {
                'rank': 1,
                'score': 0.85,
                'text': 'Sample land deed from Chai Nat...',
                'metadata': {
                    'province': 'Chai Nat',
                    'district': 'Noen Kham',
                    'latitude': 14.967,
                    'longitude': 99.907,
                    'google_maps_url': 'https://maps.google.com/maps?q=14.967,99.907'
                },
                'primary_location': {
                    'location': {
                        'latitude': 14.967,
                        'longitude': 99.907,
                        'display': 'Chai Nat, Noen Kham',
                        'maps_url': 'https://maps.google.com/maps?q=14.967,99.907'
                    }
                }
            }
        ]
        
        print("Backend API Response:")
        for result in mock_results:
            if 'primary_location' in result:
                location = result['primary_location']['location']
                
                print(f"Document: {result['text'][:50]}...")
                print(f"Confidence: {result['score']:.2f}")
                print(f"Primary Location: {location['display']}")
                print(f"Coordinates: {location['latitude']}, {location['longitude']}")
                print(f"Google Maps URL: {location['maps_url']}")
                
                # Show how mapping app would use this data
                print("\nMapping App Integration:")
                print(f"  → Set map center: [{location['latitude']}, {location['longitude']}]")
                print(f"  → Add marker at coordinates")
                print(f"  → Display location: {location['display']}")
                print(f"  → Link to Google Maps: {location['maps_url']}")
    
    def demo_coordinate_formats(self):
        """Demo different coordinate formats and their handling."""
        print("\n4. 📐 Coordinate Format Handling:")
        print("-" * 35)
        
        coordinate_examples = [
            {
                'name': 'Decimal Degrees',
                'metadata': {'latitude': 14.967, 'longitude': 99.907}
            },
            {
                'name': 'Formatted String',
                'metadata': {'coordinates_formatted': '14.967000, 99.907000'}
            },
            {
                'name': 'Mixed Format',
                'metadata': {
                    'latitude': 13.7563,
                    'longitude': 100.5014,
                    'coordinates_formatted': '13°45\'23"N, 100°30\'05"E'
                }
            }
        ]
        
        for example in coordinate_examples:
            print(f"\n{example['name']}:")
            location_info = self.cli_ops._extract_location_info(example['metadata'])
            
            if location_info['coordinates']:
                print(f"  Coordinates: {location_info['coordinates']}")
            if location_info['maps_link']:
                print(f"  Generated Maps URL: {location_info['maps_link']}")
            else:
                print("  No Maps URL generated (missing lat/lon)")
    
    def run_demo(self):
        """Run the complete mapping integration demo."""
        try:
            self.demo_location_extraction()
            self.demo_primary_location_json()
            self.demo_mapping_app_integration()
            self.demo_coordinate_formats()
            
            print("\n✅ Demo completed successfully!")
            print("\nKey takeaways for mapping app integration:")
            print("• Primary location coordinates are automatically extracted from top search result")
            print("• JSON format ready for backend consumption")
            print("• Google Maps URLs generated from coordinates when not available")
            print("• Multiple coordinate formats supported")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Run the mapping integration demo."""
    demo = MappingIntegrationDemo()
    demo.run_demo()


if __name__ == "__main__":
    main() 