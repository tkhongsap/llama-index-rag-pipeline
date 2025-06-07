import importlib.util
import sys
from pathlib import Path

# Load modules directly from src-iLand/data_processing
base = Path(__file__).resolve().parents[1] / 'src-iLand' / 'data_processing'
models_spec = importlib.util.spec_from_file_location('models', base / 'models.py')
models = importlib.util.module_from_spec(models_spec)
sys.modules['models'] = models
models_spec.loader.exec_module(models)

docproc_spec = importlib.util.spec_from_file_location('docproc', base / 'document_processor.py')
docproc = importlib.util.module_from_spec(docproc_spec)
docproc_spec.loader.exec_module(docproc)

processor = docproc.DocumentProcessor(models.DatasetConfig(name='t', description='', field_mappings=[]))

def test_parse_point_with_commas():
    result = processor.parse_geom_point("POINT(100.4514,14.5486)")
    assert result == {
        'longitude': 100.4514,
        'latitude': 14.5486,
        'coordinates_formatted': '14.548600, 100.451400',
        'google_maps_url': 'https://www.google.com/maps?q=14.5486,100.4514'
    }

def test_parse_lat_lon_string():
    result = processor.parse_geom_point("14.5486 100.4514")
    assert result == {
        'longitude': 100.4514,
        'latitude': 14.5486,
        'coordinates_formatted': '14.548600, 100.451400',
        'google_maps_url': 'https://www.google.com/maps?q=14.5486,100.4514'
    }

def test_invalid_range_returns_empty():
    assert processor.parse_geom_point("POINT(200 95)") == {}
