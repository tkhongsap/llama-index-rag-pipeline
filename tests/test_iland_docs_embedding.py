import importlib.util
from pathlib import Path
import pytest

# Load metadata_extractor module
base = Path(__file__).resolve().parents[1] / 'src-iLand' / 'docs_embedding'
meta_spec = importlib.util.spec_from_file_location('meta', base / 'metadata_extractor.py')
meta = importlib.util.module_from_spec(meta_spec)
meta_spec.loader.exec_module(meta)


def test_metadata_extraction_and_categories():
    extractor = meta.iLandMetadataExtractor()
    content = """Deed Serial No: 123\nProvince: Bangkok\nLand Rai: 5\nDeed Total Square Wa: 800\nDeed Type: โฉนด\nLand Main Category: agricultural\nDeed Holding Type: Company\nRegion: ภาคกลาง"""
    data = extractor.extract_from_content(content)
    assert data['deed_serial_no'] == '123'
    assert data['province'] == 'Bangkok'
    assert data['land_rai'] == 5.0
    cats = extractor.classify_content_types(content)
    assert 'land_deed' in cats
    derived = extractor.derive_categories(data)
    assert derived['area_category'] == 'medium'
    assert derived['region_category'] == 'central'
    title = extractor.extract_document_title(data, 1)
    assert 'Bangkok' in title
