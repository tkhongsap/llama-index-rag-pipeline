"""Build DocumentSummaryIndex artifacts for Orion markdown corpora."""
from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Iterable

from dotenv import load_dotenv
from llama_index.core import (
    DocumentSummaryIndex,
    Settings,
    SimpleDirectoryReader,
    get_response_synthesizer,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from paths import DATASETS, REPORTS_DIR, dataset_choices, ensure_directories

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("orion.summary_index")

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 50
SUMMARY_EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"


def validate_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not configured. Please set it before running.")
    return api_key


def build_index_for_dataset(dataset_key: str, llm: OpenAI, embed_model: OpenAIEmbedding) -> Dict:
    dataset = DATASETS[dataset_key]
    markdown_dir = dataset.markdown_dir

    if not markdown_dir.exists():
        LOGGER.warning("Skipping %s (markdown directory missing)", dataset.title)
        return {}

    docs = SimpleDirectoryReader(str(markdown_dir), required_exts=[".md"]).load_data()
    if not docs:
        LOGGER.warning("Skipping %s (no markdown files)", dataset.title)
        return {}

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    response_synthesiser = get_response_synthesizer(response_mode="tree_summarize", use_async=True)

    LOGGER.info("Building DocumentSummaryIndex for %s (%s documents)", dataset.title, len(docs))
    index = DocumentSummaryIndex.from_documents(
        docs,
        llm=llm,
        embed_model=embed_model,
        transformations=[splitter],
        response_synthesizer=response_synthesiser,
        show_progress=True,
    )

    dataset.summary_index_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(dataset.summary_index_dir))

    stats = {
        "dataset": dataset.key,
        "title": dataset.title,
        "document_count": len(docs),
        "chunk_count": len(index.docstore.docs),
        "persist_dir": str(dataset.summary_index_dir),
        "generated_at": datetime.utcnow().isoformat(),
    }
    LOGGER.info("Persisted summary index for %s", dataset.title)
    return stats


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

    summary_stats = {"generated_at": datetime.utcnow().isoformat(), "datasets": []}

    for dataset in dataset_choices(args.datasets):
        stats = build_index_for_dataset(dataset.key, llm, embed_model)
        if stats:
            summary_stats["datasets"].append(stats)

    if summary_stats["datasets"]:
        report_path = REPORTS_DIR / "summary_index_report.json"
        report_path.write_text(
            json.dumps(summary_stats, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        LOGGER.info("Summary index report written to %s", report_path)


if __name__ == "__main__":
    main()
