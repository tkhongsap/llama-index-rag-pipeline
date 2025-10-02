import importlib.util
from pathlib import Path
import sys
import types
import numpy as np

# Create dummy llama_index modules to satisfy imports
fake_core = types.ModuleType('llama_index.core')
fake_core.Settings = types.SimpleNamespace(llm=None, embed_model=None)
class _DummyClass:
    def __init__(self, *args, **kwargs):
        pass

fake_llms_openai = types.ModuleType('llama_index.llms.openai')
fake_llms_openai.OpenAI = _DummyClass
fake_embeddings_openai = types.ModuleType('llama_index.embeddings.openai')
fake_embeddings_openai.OpenAIEmbedding = _DummyClass
sys.modules['llama_index'] = types.ModuleType('llama_index')
sys.modules['llama_index.core'] = fake_core
sys.modules['llama_index.llms'] = types.ModuleType('llama_index.llms')
sys.modules['llama_index.llms.openai'] = fake_llms_openai
sys.modules['llama_index.embeddings'] = types.ModuleType('llama_index.embeddings')
sys.modules['llama_index.embeddings.openai'] = fake_embeddings_openai

BASE_DIR = Path(__file__).resolve().parents[1] / 'retrieval'

spec = importlib.util.spec_from_file_location('index_classifier', BASE_DIR / 'index_classifier.py')
index_classifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(index_classifier)


class DummyEmbed:
    def get_text_embedding(self, text):
        # return small deterministic vector
        return [0.1, 0.2, 0.3]


class DummyLLM:
    def complete(self, prompt):
        class Resp:
            text = "iland_land_deeds"

        return Resp()


def test_classifier_embedding_mode(monkeypatch):
    monkeypatch.setattr(index_classifier, 'OpenAIEmbedding', lambda *a, **k: DummyEmbed())
    monkeypatch.setattr(index_classifier, 'OpenAI', lambda *a, **k: DummyLLM())
    classifier = index_classifier.create_default_iland_classifier(api_key='test', mode='embedding')
    result = classifier.classify_query('ที่ดินกรุงเทพ')
    assert 'selected_index' in result
    assert result['confidence'] >= 0
