"""Simple evaluation for the iLand RAG pipeline.

This script loads a small QA dataset and measures how often the
index classifier selects the expected index. It can be extended to
cover full retrieval evaluation once embeddings and indices are
available.
"""
from pathlib import Path
import json
from typing import List, Dict

import sys
import types

# Setup minimal llama_index stubs if package is unavailable
if 'llama_index.llms.openai' not in sys.modules:
    class _DummyClass:
        def __init__(self, *a, **k):
            pass
        def get_text_embedding(self, *a, **k):
            return [0.1, 0.2, 0.3]
        def complete(self, prompt):
            class R:
                text = "iland_land_deeds"
            return R()

    fake_core = types.ModuleType('llama_index.core')
    fake_core.Settings = types.SimpleNamespace(llm=_DummyClass(), embed_model=_DummyClass())
    fake_llms = types.ModuleType('llama_index.llms')
    fake_llms_openai = types.ModuleType('llama_index.llms.openai')
    fake_llms_openai.OpenAI = _DummyClass
    fake_embeddings = types.ModuleType('llama_index.embeddings')
    fake_embeddings_openai = types.ModuleType('llama_index.embeddings.openai')
    fake_embeddings_openai.OpenAIEmbedding = _DummyClass
    sys.modules.update({
        'llama_index': types.ModuleType('llama_index'),
        'llama_index.core': fake_core,
        'llama_index.llms': fake_llms,
        'llama_index.llms.openai': fake_llms_openai,
        'llama_index.embeddings': fake_embeddings,
        'llama_index.embeddings.openai': fake_embeddings_openai,
    })

import importlib.util

_SPEC = importlib.util.spec_from_file_location(
    "index_classifier",
    Path(__file__).parent / "retrieval" / "index_classifier.py",
)
index_classifier = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(index_classifier)
create_default_iland_classifier = index_classifier.create_default_iland_classifier


def load_dataset(path: Path) -> List[Dict[str, str]]:
    data = []
    with path.open() as f:
        for line in f:
            data.append(json.loads(line))
    return data


def evaluate_classifier(dataset_path: Path) -> float:
    classifier = create_default_iland_classifier(mode="embedding", api_key="test")
    total = 0
    correct = 0
    for item in load_dataset(dataset_path):
        result = classifier.classify_query(item["query"])
        if result.get("selected_index") == item["expected_index"]:
            correct += 1
        total += 1
    return correct / total if total else 0.0


def main():
    dataset_path = Path(__file__).parent / "test" / "eval_dataset.jsonl"
    accuracy = evaluate_classifier(dataset_path)
    print(f"Classification accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
