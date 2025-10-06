"""Render Orion JSONL documents into markdown narratives."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List

from paths import DATASETS, ensure_directories, dataset_choices

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("orion.markdown")


def load_documents(jsonl_path: Path) -> List[Dict]:
    documents: List[Dict] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                documents.append(json.loads(line))
    return documents


def render_metadata_section(metadata: Dict, key_fields: Iterable[str]) -> str:
    lines = ["## Key Metrics", ""]
    for key in key_fields:
        value = metadata.get(key)
        if value is None or value == "":
            continue
        pretty_key = key.replace("_", " ").title()
        lines.append(f"- **{pretty_key}:** {value}")
    return "\n".join(lines)


def render_metadata_table(metadata: Dict) -> str:
    lines = ["\n## Metadata", "", "| Field | Value |", "| --- | --- |"]
    for key, value in sorted(metadata.items()):
        if key in {"text"}:
            continue
        if value is None or value == "":
            continue
        pretty_key = key.replace("_", " ").title()
        lines.append(f"| {pretty_key} | {value} |")
    return "\n".join(lines)


def render_markdown(dataset_key: str, documents: List[Dict]) -> List[str]:
    dataset = DATASETS[dataset_key]
    markdown_blocks: List[str] = []

    for idx, doc in enumerate(documents, start=1):
        metadata = doc.get("metadata", {})
        text = doc.get("text", "").strip()
        heading = f"# {dataset.title} · Record {idx:04d}"

        lines = [heading, ""]
        lines.append(render_metadata_section(metadata, dataset.key_fields))
        if text:
            lines.extend(["", "## Narrative", "", text])
        lines.append(render_metadata_table(metadata))
        markdown_blocks.append("\n".join(lines))

    return markdown_blocks


def save_markdown_files(dataset_key: str, markdown_blocks: List[str]) -> List[Path]:
    dataset = DATASETS[dataset_key]
    saved_files: List[Path] = []
    padding = len(str(len(markdown_blocks))) or 1

    for idx, content in enumerate(markdown_blocks, start=1):
        filename = f"{dataset.markdown_prefix}_{idx:0{padding}d}.md"
        path = dataset.markdown_dir / filename
        path.write_text(content, encoding="utf-8")
        saved_files.append(path)

    return saved_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma separated dataset keys to render (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()

    for dataset in dataset_choices(args.datasets):
        if not dataset.jsonl_path.exists():
            LOGGER.warning("Skipping %s (JSONL not found)", dataset.title)
            continue

        documents = load_documents(dataset.jsonl_path)
        markdown_blocks = render_markdown(dataset.key, documents)
        saved_files = save_markdown_files(dataset.key, markdown_blocks)
        LOGGER.info("Rendered %s markdown files for %s", len(saved_files), dataset.title)


if __name__ == "__main__":
    main()
