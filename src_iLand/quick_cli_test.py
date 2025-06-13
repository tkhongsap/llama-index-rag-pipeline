"""
Quick CLI test to verify coordinate display is working
"""

import sys
from pathlib import Path

# Add the current directory to the path to import retrieval modules
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from retrieval.cli_operations import iLandCLIOperations

def test_coordinate_display():
    """Test coordinate display functionality."""
    print("🗺️ Quick Coordinate Display Test")
    print("=" * 40)
    
    cli_ops = iLandCLIOperations(router=None, adapters={})
    
    # Mock response synthesizer
    class MockResponse:
        def __init__(self):
            self.response = "Test response about land deeds."
    
    cli_ops.response_synthesizer = type('MockSynthesizer', (), {
        'synthesize': lambda self, query, nodes: MockResponse()
    })()
    
    # Create mock nodes with the document structure format
    class MockNode:
        def __init__(self, metadata):
            self.metadata = metadata
            self.text = "Sample document text about land deeds..."
    
    class MockScoredNode:
        def __init__(self, node):
            self.node = node
            self.score = 0.85
    
    # Test with the actual document format
    metadata = {
        'Province': 'Chai Nat',
        'District': 'Noen Kham',
        'Latitude': 14.967,
        'Longitude': 99.907,
        'Coordinates Formatted': '14.967000, 99.907000'
    }
    
    mock_nodes = [MockScoredNode(MockNode(metadata))]
    
    print("Testing with document format metadata:")
    print(f"Input: {metadata}")
    print("\nExpected output should include:")
    print("📍 Primary Location: Chai Nat, Noen Kham (14.967, 99.907)")
    print("🗺️ Maps Link: https://maps.google.com/maps?q=14.967,99.907")
    print("📊 Location Data (JSON): {...}")
    
    print("\n" + "="*50)
    print("ACTUAL CLI OUTPUT:")
    print("="*50)
    
    # This should now show coordinates inline
    cli_ops._generate_natural_response("test query", mock_nodes)

if __name__ == "__main__":
    test_coordinate_display() 