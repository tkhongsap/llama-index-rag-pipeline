import importlib.util
import sys
from pathlib import Path
import pandas as pd

# Load modules from src-iLand/data_processing
BASE_DIR = Path(__file__).resolve().parents[1] / 'data_processing'

models_spec = importlib.util.spec_from_file_location('models', BASE_DIR / 'models.py')
models = importlib.util.module_from_spec(models_spec)
models_spec.loader.exec_module(models)
sys.modules['models'] = models

processor_spec = importlib.util.spec_from_file_location('document_processor', BASE_DIR / 'document_processor.py')
document_processor = importlib.util.module_from_spec(processor_spec)
processor_spec.loader.exec_module(document_processor)


def _create_processor():
    mappings = [
        models.FieldMapping(csv_column='province', metadata_key='province', field_type='location'),
        models.FieldMapping(csv_column='district', metadata_key='district', field_type='location'),
    ]
    config = models.DatasetConfig(
        name='test',
        description='test',
        field_mappings=mappings,
        embedding_fields=['province']
    )
    return document_processor.DocumentProcessor(config)


def test_clean_value_and_area_formatting():
    proc = _create_processor()
    # Placeholder values removed
    assert proc.clean_value('ไม่ระบุ') is None
    assert proc.clean_value(' N/A ') is None
    # Thai punctuation normalized
    assert proc.clean_value('“ทดสอบ”') == '"ทดสอบ"'
    # Format area
    display = proc.format_area_for_display(1, 2, 3)
    assert '1 ไร่' in display and '2 งาน' in display and '3 ตร.ว.' in display


def test_parse_geom_point():
    proc = _create_processor()
    result = proc.parse_geom_point('POINT(100.5 13.7)')
    assert result['latitude'] == 13.7
    assert result['longitude'] == 100.5
    assert 'google.com/maps' in result['google_maps_url']
