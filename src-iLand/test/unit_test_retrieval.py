import importlib.util
from pathlib import Path
import sys
import types
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1] / 'retrieval'

spec = importlib.util.spec_from_file_location('index_classifier', BASE_DIR / 'index_classifier.py')
index_classifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(index_classifier)


class DummyEmbed:
    def get_text_embedding(self, text):
        # return small deterministic vector
        return [0.1, 0.2, 0.3]


def test_classifier_embedding_mode(monkeypatch):
    monkeypatch.setattr(index_classifier, 'OpenAIEmbedding', lambda *a, **k: DummyEmbed())
    classifier = index_classifier.create_default_iland_classifier(api_key='test', mode='embedding')
    result = classifier.classify_query('ที่ดินกรุงเทพ')
    assert 'selected_index' in result
    assert result['confidence'] >= 0
