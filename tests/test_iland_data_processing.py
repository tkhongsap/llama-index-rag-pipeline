import importlib.util
from pathlib import Path
import pytest

pd = pytest.importorskip("pandas")

# Load modules from src-iLand/data_processing
base = Path(__file__).resolve().parents[1] / 'src-iLand' / 'data_processing'
models_spec = importlib.util.spec_from_file_location('models', base / 'models.py')
models = importlib.util.module_from_spec(models_spec)
models_spec.loader.exec_module(models)

docproc_spec = importlib.util.spec_from_file_location('docproc', base / 'document_processor.py')
docproc = importlib.util.module_from_spec(docproc_spec)
docproc_spec.loader.exec_module(docproc)

fileout_spec = importlib.util.spec_from_file_location('fileout', base / 'file_output.py')
fileout = importlib.util.module_from_spec(fileout_spec)
fileout_spec.loader.exec_module(fileout)


def create_processor():
    mappings = [
        models.FieldMapping(csv_column='province', metadata_key='province', field_type='location'),
        models.FieldMapping(csv_column='district', metadata_key='district', field_type='location'),
        models.FieldMapping(csv_column='area_rai', metadata_key='area_rai', field_type='area_measurements'),
        models.FieldMapping(csv_column='deed_type', metadata_key='deed_type', field_type='deed_info'),
        models.FieldMapping(csv_column='land_use_type', metadata_key='land_use_type', field_type='land_details'),
    ]
    config = models.DatasetConfig(
        name='test',
        description='test',
        field_mappings=mappings,
        embedding_fields=['deed_type', 'province']
    )
    return docproc.DocumentProcessor(config)


def test_convert_row_to_document(tmp_path):
    processor = create_processor()
    row = pd.Series({
        'province': 'Bangkok',
        'district': 'Bang Kapi',
        'area_rai': 2,
        'deed_type': 'Chanote',
        'land_use_type': 'residential'
    })
    doc = processor.convert_row_to_document(row)

    assert isinstance(doc, models.SimpleDocument)
    assert doc.metadata['province'] == 'Bangkok'
    assert doc.metadata['location_hierarchy'] == 'Bangkok > Bang Kapi'
    assert doc.metadata['doc_type'] == 'land_deed_record'
    assert doc.metadata['area_formatted'].startswith('2')
    assert doc.text.startswith('#')

    out = tmp_path / 'out'
    out.mkdir()
    fm = fileout.FileOutputManager(str(out), processor.dataset_config)
    jsonl = fm.save_documents_as_jsonl([doc], 'docs.jsonl')
    assert Path(jsonl).exists()
    md_files = fm.save_documents_as_markdown_files([doc], prefix='doc', batch_size=1)
    assert md_files
