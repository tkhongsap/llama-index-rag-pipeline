"""Output helpers for persisting iLand documents."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

try:  # pragma: no cover
    from .models import DatasetConfig, SimpleDocument
except ImportError:  # pragma: no cover
    from models import DatasetConfig, SimpleDocument


class FileOutputManager:
    """Persist documents as JSONL and markdown artefacts."""

    def __init__(self, output_directory: str, dataset_config: DatasetConfig):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.dataset_config = dataset_config

    def save_documents_as_jsonl(self, documents: Iterable[SimpleDocument], filename: str) -> str:
        """Write documents into a JSONL file and return the file path."""

        path = self.output_directory / filename
        with path.open("w", encoding="utf-8") as handle:
            for doc in documents:
                handle.write(json.dumps(doc.as_dict(), ensure_ascii=False))
                handle.write("\n")
        return str(path)

    def save_documents_as_markdown_files(
        self,
        documents: Iterable[SimpleDocument],
        prefix: str = "document",
        batch_size: int = 50,
    ) -> List[str]:
        """Persist documents as numbered markdown files."""

        paths: List[str] = []
        batch: List[SimpleDocument] = []
        for doc in documents:
            batch.append(doc)
            if len(batch) >= batch_size:
                paths.extend(self._write_batch(batch, prefix, len(paths)))
                batch = []
        if batch:
            paths.extend(self._write_batch(batch, prefix, len(paths)))
        return paths

    def _write_batch(self, batch: List[SimpleDocument], prefix: str, start_index: int) -> List[str]:
        written: List[str] = []
        for offset, doc in enumerate(batch):
            filename = f"{prefix}_{start_index + offset:05d}.md"
            path = self.output_directory / filename
            front_matter = {
                "dataset": self.dataset_config.name,
                "doc_id": doc.metadata.get("doc_id", doc.id),
                "config": self.dataset_config.description,
            }
            content = ["---", json.dumps(front_matter, ensure_ascii=False), "---", doc.text]
            path.write_text("\n".join(content), encoding="utf-8")
            written.append(str(path))
        return written


__all__ = ["FileOutputManager"]
