"""Analyze Orion summary and recursive indexes for QA."""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable

from llama_index.core import DocumentSummaryIndex, StorageContext

from paths import DATASETS, REPORTS_DIR, dataset_choices, ensure_directories

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("orion.index_analysis")


def analyse_summary_index(dataset_key: str) -> Dict:
    dataset = DATASETS[dataset_key]
    storage_context = StorageContext.from_defaults(persist_dir=str(dataset.summary_index_dir))
    index = DocumentSummaryIndex.from_storage(storage_context)

    doc_ids = list(index.ref_doc_info.keys())
    chunk_lengths = [len(node.get_content(metadata_mode="all")) for node in index.docstore.docs.values()]

    metadata_counter = Counter()
    for doc_id in doc_ids:
        doc_info = index.ref_doc_info[doc_id]
        metadata_counter.update(doc_info.metadata.keys())

    return {
        "dataset": dataset.key,
        "title": dataset.title,
        "document_count": len(doc_ids),
        "chunk_count": len(chunk_lengths),
        "average_chunk_characters": sum(chunk_lengths) / max(len(chunk_lengths), 1),
        "top_metadata_fields": metadata_counter.most_common(10),
    }


def analyse_recursive_index(dataset_key: str) -> Dict:
    dataset = DATASETS[dataset_key]
    metadata_path = dataset.recursive_index_dir / "recursive_metadata.json"
    if not metadata_path.exists():
        return {}

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma separated dataset keys to analyze (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    ensure_directories()
    args = parse_args()

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "summary_indexes": [],
        "recursive_indexes": [],
    }

    for dataset in dataset_choices(args.datasets):
        if dataset.summary_index_dir.exists():
            try:
                report["summary_indexes"].append(analyse_summary_index(dataset.key))
            except Exception as exc:
                LOGGER.error("Failed to analyze summary index for %s: %s", dataset.title, exc)
        if dataset.recursive_index_dir.exists():
            recursive_info = analyse_recursive_index(dataset.key)
            if recursive_info:
                report["recursive_indexes"].append(recursive_info)

    report_path = REPORTS_DIR / "index_analysis.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Index analysis report written to %s", report_path)


if __name__ == "__main__":
    main()
