"""
Unit tests for Google Maps Location Integration features in CLI Operations

Tests the location display functionality added to iLand CLI operations.
"""

import unittest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add the parent directory to the path to import retrieval modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from retrieval.cli_operations import iLandCLIOperations


class TestLocationIntegrationFeatures(unittest.TestCase):
    """Test Google Maps location integration features."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_router = Mock()
        self.mock_adapters = {}
        self.cli_ops = iLandCLIOperations(
            router=self.mock_router,
            adapters=self.mock_adapters
        )
    
    def test_extract_location_info_complete_data(self):
        """Test location extraction with complete metadata."""
        metadata = {
            'province': 'เชียงใหม่',
            'district': 'เมืองเชียงใหม่',
            'subdistrict': 'ศรีภูมิ',
            'google_maps_url': 'https://maps.google.com/maps?q=18.7883,98.9853',
            'latitude': 18.7883,
            'longitude': 98.9853,
            'coordinates_formatted': '18.788300, 98.985300'
        }
        
        result = self.cli_ops._extract_location_info(metadata)
        
        self.assertEqual(result['location_display'], 'เชียงใหม่, เมืองเชียงใหม่, ศรีภูมิ')
        self.assertEqual(result['maps_link'], 'https://maps.google.com/maps?q=18.7883,98.9853')
        self.assertEqual(result['maps_display'], '🗺️  Google Maps')
        self.assertEqual(result['coordinates'], '18.788300, 98.985300')
    
    def test_extract_location_info_no_maps_url_with_coordinates(self):
        """Test location extraction without Maps URL but with coordinates."""
        metadata = {
            'province': 'กรุงเทพมหานคร',
            'district': 'บางกะปิ',
            'latitude': 13.7563,
            'longitude': 100.5014
        }
        
        result = self.cli_ops._extract_location_info(metadata)
        
        self.assertEqual(result['location_display'], 'กรุงเทพมหานคร, บางกะปิ')
        self.assertEqual(result['maps_link'], 'https://maps.google.com/maps?q=13.7563,100.5014')
        self.assertEqual(result['maps_display'], '🗺️  Google Maps')
        self.assertEqual(result['coordinates'], '13.7563, 100.5014')
    
    def test_extract_location_info_partial_data(self):
        """Test location extraction with partial metadata."""
        metadata = {
            'province': 'ภูเก็ต',
            # No district, subdistrict, or coordinates
        }
        
        result = self.cli_ops._extract_location_info(metadata)
        
        self.assertEqual(result['location_display'], 'ภูเก็ต')
        self.assertIsNone(result['maps_link'])
        self.assertIsNone(result['maps_display'])
        self.assertIsNone(result['coordinates'])
    
    def test_extract_location_info_no_data(self):
        """Test location extraction with empty metadata."""
        metadata = {}
        
        result = self.cli_ops._extract_location_info(metadata)
        
        self.assertEqual(result['location_display'], 'N/A')
        self.assertIsNone(result['maps_link'])
        self.assertIsNone(result['maps_display'])
        self.assertIsNone(result['coordinates'])
    
    def test_collect_unique_locations(self):
        """Test collection of unique locations from results."""
        results = [
            {
                'metadata': {
                    'province': 'เชียงใหม่',
                    'district': 'เมืองเชียงใหม่',
                    'google_maps_url': 'https://maps.google.com/maps?q=18.7883,98.9853'
                }
            },
            {
                'metadata': {
                    'province': 'เชียงใหม่',
                    'district': 'สันทราย',
                    'latitude': 18.7456,
                    'longitude': 99.0234
                }
            },
            {
                'metadata': {
                    'province': 'เชียงใหม่',
                    'district': 'เมืองเชียงใหม่',  # Duplicate location
                    'google_maps_url': 'https://maps.google.com/maps?q=18.7883,98.9853'
                }
            }
        ]
        
        unique_locations = self.cli_ops._collect_unique_locations(results)
        
        self.assertEqual(len(unique_locations), 2)  # Should have 2 unique locations
        
        location_displays = [loc['display'] for loc in unique_locations]
        self.assertIn('เชียงใหม่, เมืองเชียงใหม่', location_displays)
        self.assertIn('เชียงใหม่, สันทราย', location_displays)
    
    def test_collect_unique_locations_max_three(self):
        """Test that collection returns maximum of 3 locations."""
        results = []
        for i in range(5):  # Create 5 different locations
            results.append({
                'metadata': {
                    'province': f'จังหวัด{i}',
                    'district': f'อำเภอ{i}',
                    'latitude': 18.0 + i,
                    'longitude': 99.0 + i
                }
            })
        
        unique_locations = self.cli_ops._collect_unique_locations(results)
        
        self.assertLessEqual(len(unique_locations), 3)  # Should return max 3 locations
    
    def test_extract_location_info_unicode_compatibility(self):
        """Test location extraction with Unicode Thai characters."""
        metadata = {
            'province': 'นครราชสีมา',
            'district': 'เมืองนครราชสีมา',
            'subdistrict': 'ในเมือง',
            'google_maps_url': 'https://maps.google.com/maps?q=14.9799,102.0977'
        }
        
        result = self.cli_ops._extract_location_info(metadata)
        
        # Test that Unicode Thai characters are handled correctly
        self.assertIn('นครราชสีมา', result['location_display'])
        self.assertIn('เมืองนครราชสีมา', result['location_display'])
        self.assertIn('ในเมือง', result['location_display'])
        self.assertIsNotNone(result['maps_link'])
        self.assertEqual(result['maps_display'], '🗺️  Google Maps')
    
    def test_location_fallback_coordinates_formatted(self):
        """Test coordinate fallback with formatted coordinates."""
        metadata = {
            'province': 'สงขลา',
            'district': 'หาดใหญ่',
            'coordinates_formatted': '7.0187, 100.4933',
            # No direct google_maps_url or separate lat/lon
        }
        
        result = self.cli_ops._extract_location_info(metadata)
        
        self.assertEqual(result['location_display'], 'สงขลา, หาดใหญ่')
        self.assertEqual(result['coordinates'], '7.0187, 100.4933')
        self.assertIsNone(result['maps_link'])  # No coordinates to generate URL
        self.assertIsNone(result['maps_display'])
    
    @patch('builtins.print')
    def test_print_retrieved_documents_with_location(self, mock_print):
        """Test that retrieved documents display includes location information."""
        results = [
            {
                'rank': 1,
                'score': 0.85,
                'text': 'Sample land deed text...',
                'metadata': {
                    'province': 'เชียงใหม่',
                    'district': 'เมืองเชียงใหม่',
                    'google_maps_url': 'https://maps.google.com/maps?q=18.7883,98.9853'
                }
            }
        ]
        
        self.cli_ops._print_retrieved_documents(results)
        
        # Verify that print was called with location information
        print_calls = [str(call) for call in mock_print.call_args_list]
        print_output = ' '.join(print_calls)
        
        self.assertIn('📍 Location', print_output)
    
    def test_primary_location_coordinates_inline_display(self):
        """Test that primary location includes coordinates for mapping integration."""
        # Test the core functionality directly with capitalized field names (actual document format)
        metadata = {
            'Province': 'Chai Nat',
            'District': 'Noen Kham',
            'Latitude': 14.967,
            'Longitude': 99.907,
            'Coordinates Formatted': '14.967000, 99.907000'
        }
        
        # Create mock nodes
        class MockNode:
            def __init__(self, metadata):
                self.metadata = metadata
        
        class MockScoredNode:
            def __init__(self, node):
                self.node = node
        
        mock_nodes = [MockScoredNode(MockNode(metadata))]
        
        # Test primary location JSON extraction
        location_json = self.cli_ops._extract_primary_location_json(mock_nodes)
        
        self.assertIsNotNone(location_json)
        self.assertIn('location', location_json)
        
        location = location_json['location']
        self.assertEqual(location['latitude'], 14.967)
        self.assertEqual(location['longitude'], 99.907)
        self.assertEqual(location['display'], 'Chai Nat, Noen Kham')
        self.assertEqual(location['maps_url'], 'https://maps.google.com/maps?q=14.967,99.907')
        self.assertEqual(location['province'], 'Chai Nat')
        self.assertEqual(location['district'], 'Noen Kham')
    
    def test_prefix_cleanup_functionality(self):
        """Test that "**:" prefixes are properly cleaned up."""
        metadata = {
            'province': '**: Chai Nat',
            'district': '**: Noen Kham',
            'latitude': 14.967,
            'longitude': 99.907
        }
        
        result = self.cli_ops._extract_location_info(metadata)
        
        # Should clean up the "**:" prefixes
        self.assertEqual(result['location_display'], 'Chai Nat, Noen Kham')
        self.assertEqual(result['coordinates'], '14.967, 99.907')
        self.assertIsNotNone(result['maps_link'])
    
    def test_extract_primary_location_json_complete_data(self):
        """Test primary location JSON extraction with complete data."""
        class MockNode:
            def __init__(self, metadata):
                self.metadata = metadata
        
        class MockScoredNode:
            def __init__(self, node):
                self.node = node
        
        nodes = [
            MockScoredNode(
                MockNode({
                    'province': 'เชียงใหม่',
                    'district': 'เมืองเชียงใหม่',
                    'subdistrict': 'ศรีภูมิ',
                    'latitude': 18.7883,
                    'longitude': 98.9853,
                    'coordinates_formatted': '18.788300, 98.985300',
                    'google_maps_url': 'https://maps.google.com/maps?q=18.7883,98.9853'
                })
            )
        ]
        
        result = self.cli_ops._extract_primary_location_json(nodes)
        
        self.assertIn('location', result)
        location_data = result['location']
        
        self.assertEqual(location_data['latitude'], 18.7883)
        self.assertEqual(location_data['longitude'], 98.9853)
        self.assertEqual(location_data['display'], 'เชียงใหม่, เมืองเชียงใหม่, ศรีภูมิ')
        self.assertEqual(location_data['province'], 'เชียงใหม่')
        self.assertEqual(location_data['district'], 'เมืองเชียงใหม่')
        self.assertEqual(location_data['subdistrict'], 'ศรีภูมิ')
        self.assertEqual(location_data['maps_url'], 'https://maps.google.com/maps?q=18.7883,98.9853')
        self.assertEqual(location_data['coordinates_formatted'], '18.788300, 98.985300')
    
    def test_extract_primary_location_json_no_coordinates(self):
        """Test primary location JSON extraction without coordinates."""
        class MockNode:
            def __init__(self, metadata):
                self.metadata = metadata
        
        class MockScoredNode:
            def __init__(self, node):
                self.node = node
        
        nodes = [
            MockScoredNode(
                MockNode({
                    'province': 'เชียงใหม่',
                    'district': 'เมืองเชียงใหม่',
                    # No latitude/longitude
                })
            )
        ]
        
        result = self.cli_ops._extract_primary_location_json(nodes)
        
        self.assertEqual(result, {})  # Should return empty dict when no coordinates
    
    def test_extract_primary_location_json_empty_nodes(self):
        """Test primary location JSON extraction with empty nodes."""
        result = self.cli_ops._extract_primary_location_json([])
        
        self.assertEqual(result, {})  # Should return empty dict when no nodes
    
    def test_extract_primary_location_json_coordinate_generation(self):
        """Test coordinate generation when formatted coordinates are missing."""
        class MockNode:
            def __init__(self, metadata):
                self.metadata = metadata
        
        class MockScoredNode:
            def __init__(self, node):
                self.node = node
        
        nodes = [
            MockScoredNode(
                MockNode({
                    'province': 'กรุงเทพมหานคร',
                    'district': 'บางกะปิ',
                    'latitude': 13.7563,
                    'longitude': 100.5014,
                    # No coordinates_formatted
                })
            )
        ]
        
        result = self.cli_ops._extract_primary_location_json(nodes)
        
        self.assertIn('location', result)
        location_data = result['location']
        
        self.assertEqual(location_data['latitude'], 13.7563)
        self.assertEqual(location_data['longitude'], 100.5014)
        self.assertEqual(location_data['coordinates_formatted'], '13.7563, 100.5014')


if __name__ == '__main__':
    unittest.main() 