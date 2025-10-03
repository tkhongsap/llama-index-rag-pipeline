"""Lightweight data models used by the iLand document processing helpers."""
from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Optional


class FieldMapping:
    """Describe how a CSV column should be mapped into document metadata."""

    def __init__(
        self,
        csv_column: str,
        metadata_key: str,
        field_type: str,
        data_type: str = "string",
        required: bool = False,
        aliases: Optional[List[str]] = None,
        description: str = "",
    ) -> None:
        self.csv_column = csv_column
        self.metadata_key = metadata_key
        self.field_type = field_type
        self.data_type = data_type
        self.required = required
        self.aliases = list(aliases or [])
        self.description = description


class DatasetConfig:
    """Configuration block for converting tabular land records into documents."""

    def __init__(
        self,
        name: str,
        description: str,
        field_mappings: List[FieldMapping],
        embedding_fields: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.field_mappings = field_mappings
        self.embedding_fields = embedding_fields or ["province", "district", "deed_type"]


class SimpleDocument:
    """Minimal document container compatible with downstream tooling."""

    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata
        self.id = metadata.get("doc_id", self._generate_id())

    def _generate_id(self) -> str:
        payload = f"{self.text}\n{sorted(self.metadata.items())}"
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:16]

    def as_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation used by JSONL exporters."""

        return {"text": self.text, "metadata": self.metadata, "doc_id": self.id}


def iter_field_mappings(config: DatasetConfig) -> Iterable[FieldMapping]:
    """Convenience generator used by helper modules."""

    return iter(config.field_mappings)
