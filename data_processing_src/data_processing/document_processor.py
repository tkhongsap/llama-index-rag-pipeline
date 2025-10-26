"""Document processing utilities tailored for iLand land records."""
from __future__ import annotations

from datetime import UTC, datetime
import json
import re
from typing import Any, Dict, Iterable, List

import pandas as pd

try:  # pragma: no cover - support relative imports when bundled
    from .models import DatasetConfig, FieldMapping, SimpleDocument
except ImportError:  # pragma: no cover
    from models import DatasetConfig, FieldMapping, SimpleDocument

_PLACEHOLDERS = {"", "-", "n/a", "na", "none", "null", "ไม่ระบุ"}
_PUNCT_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\uff0c": ",",
        "\uff1a": ":",
        "\uff1b": ";",
        "\uff01": "!",
        "\uff1f": "?",
        "\uff0e": ".",
    }
)


class DocumentProcessor:
    """Convert pandas rows into SimpleDocument instances."""

    def __init__(self, dataset_config: DatasetConfig):
        self.dataset_config = dataset_config

    def clean_value(self, value: Any) -> Any:
        """Normalise placeholder values and punctuation."""

        if pd.isna(value):
            return None
        if isinstance(value, str):
            cleaned = value.strip().translate(_PUNCT_TRANSLATION)
            if cleaned.lower() in _PLACEHOLDERS or cleaned in _PLACEHOLDERS:
                return None
            cleaned = re.sub(r"\s+", " ", cleaned)
            return cleaned
        return value

    def convert_row_to_document(self, row: pd.Series, row_index: int = 0) -> SimpleDocument:
        """Convert a single CSV row into a :class:`SimpleDocument`."""

        metadata = self._build_metadata(row)
        metadata.update(
            {
                "source": "iland_csv_import",
                "doc_type": "land_deed_record",
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "config_name": self.dataset_config.name,
                "row_index": row_index,
            }
        )
        doc_text = self._generate_text(metadata)
        metadata.setdefault("doc_id", self._build_document_id(metadata, row_index))
        return SimpleDocument(text=doc_text, metadata=metadata)

    def _build_metadata(self, row: pd.Series) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        for mapping in self._iter_mappings():
            value = None
            if mapping.csv_column in row:
                value = row[mapping.csv_column]
            elif mapping.aliases:
                for alias in mapping.aliases:
                    if alias in row:
                        value = row[alias]
                        break
            cleaned = self.clean_value(value)
            if cleaned is None:
                continue
            metadata[mapping.metadata_key] = cleaned

        province = metadata.get("province")
        district = metadata.get("district")
        if province and district:
            metadata.setdefault("location_hierarchy", f"{province} > {district}")
        elif province:
            metadata.setdefault("location_hierarchy", province)

        area_val = metadata.get("area_rai")
        if area_val is not None:
            metadata.setdefault("area_formatted", f"{area_val} ไร่")

        return metadata

    def _generate_text(self, metadata: Dict[str, Any]) -> str:
        header = "# บันทึกข้อมูลโฉนดที่ดิน"
        summary_lines = ["", f"- ประเภทเอกสาร: {metadata.get('doc_type', 'ไม่ระบุ')}" ]
        if "province" in metadata:
            summary_lines.append(f"- จังหวัด: {metadata['province']}")
        if "district" in metadata:
            summary_lines.append(f"- อำเภอ/เขต: {metadata['district']}")
        if "area_formatted" in metadata:
            summary_lines.append(f"- ขนาดที่ดิน: {metadata['area_formatted']}")
        summary_lines.append("")
        summary_lines.append("```json")
        summary_lines.append(json.dumps(metadata, ensure_ascii=False, indent=2))
        summary_lines.append("```")
        return "\n".join([header, *summary_lines])

    def _iter_mappings(self) -> Iterable[FieldMapping]:
        return self.dataset_config.field_mappings

    def _build_document_id(self, metadata: Dict[str, Any], row_index: int) -> str:
        key_parts: List[str] = []
        for field in ("province", "district", "deed_type"):
            value = metadata.get(field)
            if value:
                key_parts.append(str(value))
        key_parts.append(str(row_index))
        base = "|".join(key_parts)
        hashed = hash(base) & 0xFFFFFFFF
        return f"iland_{hashed:08x}"


__all__ = ["DocumentProcessor"]
