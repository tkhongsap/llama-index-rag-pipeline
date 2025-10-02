import importlib.util
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1] / 'docs_embedding'

extractor_spec = importlib.util.spec_from_file_location('metadata_extractor', BASE_DIR / 'metadata_extractor.py')
metadata_extractor = importlib.util.module_from_spec(extractor_spec)
extractor_spec.loader.exec_module(metadata_extractor)


def test_metadata_extraction_and_title():
    extractor = metadata_extractor.iLandMetadataExtractor()
    content = (
        'Deed Type: Chanote\n'
        'Province: Bangkok\n'
        'District: Bang Kapi\n'
        'Deed Serial No: 1234\n'
        'Land Rai: 2\n'
        'Deed Total Square Wa: 200\n'
    )
    meta = extractor.extract_from_content(content)
    assert meta['deed_type'] == 'Chanote'
    assert meta['province'] == 'Bangkok'
    types = extractor.classify_content_types(content)
    assert 'land_deed' in types
    derived = extractor.derive_categories(meta)
    assert derived['area_category'] == 'small'
    title = extractor.extract_document_title(meta, 1)
    assert 'Chanote' in title and 'Bangkok' in title
