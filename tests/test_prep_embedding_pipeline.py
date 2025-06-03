import importlib.util
from pathlib import Path
import pytest

# ---------- Prep Data Tests ----------

def test_flexible_csv_converter(tmp_path):
    pandas = pytest.importorskip("pandas")

    module_path = Path(__file__).resolve().parents[1] / "src" / "02_prep_doc_for_embedding.py"
    spec = importlib.util.spec_from_file_location("prep_module", module_path)
    prep_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep_module)

    df = pandas.DataFrame({
        "id": [1, 2],
        "age": [25, 30],
        "degree": ["BSc", "MBA"],
        "company": ["A", "B"]
    })
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)

    output_dir = tmp_path / "out"
    converter = prep_module.FlexibleCSVConverter(str(csv_path), str(output_dir))
    converter.setup_configuration(config_name="test", auto_generate=True)
    docs = converter.process_csv_to_documents(batch_size=10)

    assert len(docs) == 2
    for doc in docs:
        assert doc.text
        assert doc.metadata.get("doc_type") == "csv_record"

    jsonl = converter.save_documents_as_jsonl(docs)
    assert Path(jsonl).exists()

# ---------- Embedding Tests ----------

class DummyEmbed:
    def get_text_embedding(self, text: str):
        return [float(len(text))] * 3

class DummyNode:
    def __init__(self, node_id: str, index_id: str, text: str, metadata: dict):
        self.node_id = node_id
        self.index_id = index_id
        self.text = text
        self.metadata = metadata


def test_structured_markdown_loader_and_embedding(tmp_path):
    pytest.importorskip("llama_index")
    pytest.importorskip("dotenv")

    module_path = Path(__file__).resolve().parents[1] / "src" / "09_enhanced_batch_embeddings.py"
    spec = importlib.util.spec_from_file_location("emb_module", module_path)
    emb_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(emb_module)

    md_content = """\
Record ID: 1
Age: 30
Province: Bangkok
Education Level: Bachelor
Position: Engineer
Salary: 50000
"""
    md_path = tmp_path / "doc.md"
    md_path.write_text(md_content, encoding="utf-8")

    loader = emb_module.StructuredMarkdownLoader()
    docs = loader.load_documents_from_files([md_path])
    assert len(docs) == 1
    doc = docs[0]
    assert doc.metadata.get("age_group") == "30"
    assert doc.metadata.get("province") == "Bangkok"
    node = DummyNode("n1", "i1", doc.text, doc.metadata)

    embeddings = emb_module.extract_indexnode_embeddings_batch([node], DummyEmbed(), 1)
    assert len(embeddings) == 1
    emb = embeddings[0]
    assert emb["embedding_vector"] == [float(len(doc.text))] * 3
    assert emb["metadata"]["province"] == "Bangkok"
