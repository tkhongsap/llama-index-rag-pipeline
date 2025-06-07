import importlib.util
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1] / 'load_embedding'

models_spec = importlib.util.spec_from_file_location('models', BASE_DIR / 'models.py')
models = importlib.util.module_from_spec(models_spec)
models_spec.loader.exec_module(models)
sys.modules['models'] = models

loader_spec = importlib.util.spec_from_file_location('embedding_loader', BASE_DIR / 'embedding_loader.py')
embedding_loader = importlib.util.module_from_spec(loader_spec)
loader_spec.loader.exec_module(embedding_loader)


def _create_loader(tmp_path):
    config = models.EmbeddingConfig(embedding_dir=tmp_path)
    tmp_path.mkdir(exist_ok=True)
    return embedding_loader.iLandEmbeddingLoader(config)


def test_filtering_functions(tmp_path):
    loader = _create_loader(tmp_path)
    embeddings = [
        {"metadata": {"province": "Bangkok", "deed_type_category": "chanote", "search_text": "5 ไร่"}},
        {"metadata": {"province": "Chiang Mai", "deed_type_category": "nor_sor_3", "search_text": "1 ไร่"}},
    ]
    by_province = loader.filter_embeddings_by_province(embeddings, "Bangkok")
    assert len(by_province) == 1
    by_deed = loader.filter_embeddings_by_deed_type(embeddings, ["nor_sor_3"])
    assert len(by_deed) == 1
    by_area = loader.filter_embeddings_by_area_range(embeddings, min_area_rai=2)
    assert len(by_area) == 1
    area = loader._extract_area_from_text("10 ไร่")
    assert area == 10.0
