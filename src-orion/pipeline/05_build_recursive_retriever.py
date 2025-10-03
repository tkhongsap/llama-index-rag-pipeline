"""Build recursive retrieval indexes for Orion summary indexes."""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from llama_index.core import (
    DocumentSummaryIndex,
    Settings,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import IndexNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from paths import DATASETS, dataset_choices, ensure_directories

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("orion.recursive")

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 50
SUMMARY_EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"


def validate_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not configured. Please set it before running.")
    return api_key


def load_summary_index(dataset_key: str, llm: OpenAI, embed_model: OpenAIEmbedding) -> DocumentSummaryIndex:
    dataset = DATASETS[dataset_key]
    storage_context = StorageContext.from_defaults(persist_dir=str(dataset.summary_index_dir))
    index = DocumentSummaryIndex.from_storage(storage_context, llm=llm, embed_model=embed_model)
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    Settings.transformations = [splitter]
    return index


def build_recursive_retriever(dataset_key: str, llm: OpenAI, embed_model: OpenAIEmbedding) -> Dict:
    dataset = DATASETS[dataset_key]
    if not dataset.summary_index_dir.exists():
        LOGGER.warning("Summary index missing for %s", dataset.title)
        return {}

    doc_summary_index = load_summary_index(dataset.key, llm, embed_model)
    doc_ids = list(doc_summary_index.ref_doc_info.keys())

    doc_query_engines: Dict[str, RetrieverQueryEngine] = {}
    index_nodes: List[IndexNode] = []

    for idx, doc_id in enumerate(doc_ids):
        doc_info = doc_summary_index.ref_doc_info[doc_id]
        doc_title = doc_info.metadata.get("file_name") or doc_info.metadata.get("file_path", f"doc_{idx}")
        doc_summary = doc_summary_index.get_document_summary(doc_id)

        doc_chunks = [
            node
            for node in doc_summary_index.docstore.docs.values()
            if getattr(node, "ref_doc_id", None) == doc_id
        ]

        if not doc_chunks:
            continue

        chunk_index = VectorStoreIndex(doc_chunks)
        chunk_index_dir = dataset.chunk_index_base_dir / f"doc_{idx:04d}"
        chunk_index_dir.mkdir(parents=True, exist_ok=True)
        chunk_index.storage_context.persist(persist_dir=str(chunk_index_dir))

        query_engine = chunk_index.as_query_engine(response_mode="compact", similarity_top_k=3)
        engine_id = f"doc_{idx:04d}"
        doc_query_engines[engine_id] = query_engine

        index_node = IndexNode(
            text=f"Document: {doc_title}\n\nSummary: {doc_summary}",
            index_id=engine_id,
            metadata={
                "doc_id": doc_id,
                "doc_title": doc_title,
                "chunk_count": len(doc_chunks),
                "dataset": dataset.key,
            },
        )
        index_nodes.append(index_node)

    if not index_nodes:
        LOGGER.warning("No index nodes produced for %s", dataset.title)
        return {}

    top_index = VectorStoreIndex(index_nodes)
    top_index.storage_context.persist(persist_dir=str(dataset.recursive_index_dir))

    recursive_retriever = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": top_index.as_retriever(similarity_top_k=2)},
        query_engine_dict=doc_query_engines,
        verbose=False,
    )
    query_engine = RetrieverQueryEngine.from_args(
        recursive_retriever,
        response_synthesizer=get_response_synthesizer(response_mode="compact"),
    )

    metadata = {
        "dataset": dataset.key,
        "title": dataset.title,
        "doc_node_count": len(index_nodes),
        "chunk_query_engines": len(doc_query_engines),
        "recursive_index_dir": str(dataset.recursive_index_dir),
        "generated_at": datetime.utcnow().isoformat(),
    }

    metadata_path = dataset.recursive_index_dir / "recursive_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Recursive retriever persisted for %s", dataset.title)

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma separated dataset keys to process (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    ensure_directories()

    args = parse_args()
    api_key = validate_api_key()

    llm = OpenAI(model=LLM_MODEL, temperature=0, api_key=api_key)
    embed_model = OpenAIEmbedding(model=SUMMARY_EMBED_MODEL, api_key=api_key)
    Settings.llm = llm
    Settings.embed_model = embed_model

    results = []
    for dataset in dataset_choices(args.datasets):
        metadata = build_recursive_retriever(dataset.key, llm, embed_model)
        if metadata:
            results.append(metadata)

    if results:
        report_path = Path("data/orion/reports/recursive_retrievers.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        LOGGER.info("Recursive retriever report written to %s", report_path)


if __name__ == "__main__":
    main()
