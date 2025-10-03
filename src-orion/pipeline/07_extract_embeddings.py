"""Export embeddings from Orion indexes for downstream use."""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

from dotenv import load_dotenv
from llama_index.core import DocumentSummaryIndex, StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from paths import DATASETS, dataset_choices, ensure_directories

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("orion.embeddings")

SUMMARY_EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"


def validate_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not configured. Please set it before running.")
    return api_key


def extract_vectors(vector_store) -> List[Dict]:
    records: List[Dict] = []

    store_dict = None
    if hasattr(vector_store, "to_dict"):
        store_dict = vector_store.to_dict()

    if store_dict and isinstance(store_dict, dict):
        items = store_dict.get("embedding_dict") or store_dict.get("data") or {}
        for node_id, payload in items.items():
            embedding = payload.get("embedding") or payload.get("vector")
            metadata = payload.get("metadata") or payload.get("extra_info") or {}
            records.append({"node_id": node_id, "embedding": embedding, "metadata": metadata})
        if records:
            return records

    data_attr = getattr(vector_store, "data", None)
    if isinstance(data_attr, dict):
        for node_id, payload in data_attr.items():
            embedding = getattr(payload, "embedding", None)
            if embedding is None and isinstance(payload, dict):
                embedding = payload.get("embedding")
            metadata = getattr(payload, "metadata", None)
            if metadata is None and hasattr(payload, "extra_info"):
                metadata = payload.extra_info
            if metadata is None and isinstance(payload, dict):
                metadata = payload.get("metadata") or payload.get("extra_info")
            records.append({"node_id": node_id, "embedding": embedding, "metadata": metadata})
    return records


def export_summary_embeddings(dataset_key: str, llm: OpenAI, embed_model: OpenAIEmbedding) -> List[Dict]:
    dataset = DATASETS[dataset_key]
    storage_context = StorageContext.from_defaults(persist_dir=str(dataset.summary_index_dir))
    index = DocumentSummaryIndex.from_storage(storage_context, llm=llm, embed_model=embed_model)
    return extract_vectors(index.storage_context.vector_store)


def export_vector_index_embeddings(persist_dir: Path, llm: OpenAI, embed_model: OpenAIEmbedding) -> List[Dict]:
    storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
    index = VectorStoreIndex.from_storage(storage_context, llm=llm, embed_model=embed_model)
    return extract_vectors(index.storage_context.vector_store)


def save_embeddings(output_path: Path, records: List[Dict]) -> None:
    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "vector_count": len(records),
        "vectors": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma separated dataset keys to export (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    ensure_directories()

    args = parse_args()
    api_key = validate_api_key()

    llm = OpenAI(model=LLM_MODEL, temperature=0, api_key=api_key)
    embed_model = OpenAIEmbedding(model=SUMMARY_EMBED_MODEL, api_key=api_key)

    for dataset in dataset_choices(args.datasets):
        if dataset.summary_index_dir.exists():
            summary_vectors = export_summary_embeddings(dataset.key, llm, embed_model)
            if summary_vectors:
                save_embeddings(dataset.embedding_dir / "summary_embeddings.json", summary_vectors)
                LOGGER.info("Exported %s summary vectors for %s", len(summary_vectors), dataset.title)

        if dataset.recursive_index_dir.exists():
            recursive_vectors = export_vector_index_embeddings(dataset.recursive_index_dir, llm, embed_model)
            if recursive_vectors:
                save_embeddings(dataset.embedding_dir / "recursive_embeddings.json", recursive_vectors)
                LOGGER.info("Exported %s recursive vectors for %s", len(recursive_vectors), dataset.title)

        if dataset.chunk_index_base_dir.exists():
            for chunk_dir in sorted(dataset.chunk_index_base_dir.iterdir()):
                if not chunk_dir.is_dir():
                    continue
                chunk_vectors = export_vector_index_embeddings(chunk_dir, llm, embed_model)
                if chunk_vectors:
                    output_path = dataset.embedding_dir / "chunks" / f"{chunk_dir.name}.json"
                    save_embeddings(output_path, chunk_vectors)
            LOGGER.info("Chunk embeddings exported for %s", dataset.title)


if __name__ == "__main__":
    main()
