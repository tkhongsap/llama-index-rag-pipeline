#!/usr/bin/env python
"""
Test script for mapping integration features with real data

This script demonstrates how the enhanced CLI now provides location data
for mapping backend integration.
"""

import json
import sys
from pathlib import Path

# Add the current directory to the path to import retrieval modules
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from retrieval.cli_handlers import iLandRetrievalCLI


def test_mapping_integration():
    """Test the mapping integration features with real data."""
    
    print("🗺️ Testing Mapping Integration Features")
    print("=" * 50)
    
    # Initialize CLI
    cli = iLandRetrievalCLI()
    
    print("\n1. Loading embeddings...")
    if not cli.load_embeddings("latest"):
        print("❌ Failed to load embeddings")
        return
    
    print("\n2. Creating router...")
    if not cli.create_router("llm"):
        print("❌ Failed to create router")
        return
    
    print("\n3. Testing query with location extraction...")
    test_query = "land deeds in Chai Nat"
    
    # Execute query and capture results
    results = cli.query(test_query, top_k=3)
    
    print(f"\n4. Analyzing results for mapping integration...")
    if results:
        # Check if primary location data is included
        first_result = results[0]
        if 'primary_location' in first_result:
            print("\n✅ Primary location data found!")
            location_data = first_result['primary_location']
            print("📊 Location Data for Mapping Backend:")
            print(json.dumps(location_data, indent=2, ensure_ascii=False))
            
            # Extract coordinates for mapping
            if 'location' in location_data:
                lat = location_data['location']['latitude']
                lng = location_data['location']['longitude']
                print(f"\n🎯 Ready to send to mapping backend:")
                print(f"   Latitude: {lat}")
                print(f"   Longitude: {lng}")
                print(f"   Maps URL: {location_data['location'].get('maps_url', 'N/A')}")
        else:
            print("ℹ️  No primary location data found in results")
    else:
        print("❌ No results returned")
    
    print(f"\n5. Summary:")
    print(f"   Query: '{test_query}'")
    print(f"   Results: {len(results)} documents")
    print(f"   Location integration: {'✅ Active' if results and 'primary_location' in results[0] else '❌ Not found'}")


if __name__ == '__main__':
    test_mapping_integration() 