"""Common path utilities and dataset metadata for the Orion financial pipeline."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ORION_DIR = PROJECT_ROOT / "src-orion"
AGGREGATED_DIR = SRC_ORION_DIR / "orion-financial-aggregated"
PIPELINE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PIPELINE_DIR / "configs"
DATA_ROOT = PROJECT_ROOT / "data" / "orion"
DOCUMENTS_DIR = DATA_ROOT / "documents"
MARKDOWN_DIR = DATA_ROOT / "markdown"
INDEX_DIR = DATA_ROOT / "indexes"
EMBEDDINGS_DIR = DATA_ROOT / "embeddings"
REPORTS_DIR = DATA_ROOT / "reports"


@dataclass(frozen=True)
class DatasetDefinition:
    """Metadata describing each aggregated Orion dataset."""

    key: str
    title: str
    csv_filename: str
    config_filename: str
    markdown_prefix: str
    jsonl_filename: str
    key_fields: List[str]

    @property
    def csv_path(self) -> Path:
        return AGGREGATED_DIR / self.csv_filename

    @property
    def config_path(self) -> Path:
        return CONFIG_DIR / self.config_filename

    @property
    def documents_dir(self) -> Path:
        return DOCUMENTS_DIR / self.key

    @property
    def markdown_dir(self) -> Path:
        return MARKDOWN_DIR / self.key

    @property
    def jsonl_path(self) -> Path:
        return self.documents_dir / self.jsonl_filename

    @property
    def summary_index_dir(self) -> Path:
        return INDEX_DIR / self.key / "summary_index"

    @property
    def recursive_index_dir(self) -> Path:
        return INDEX_DIR / self.key / "recursive_index"

    @property
    def chunk_index_base_dir(self) -> Path:
        return INDEX_DIR / self.key / "chunk_indexes"

    @property
    def embedding_dir(self) -> Path:
        return EMBEDDINGS_DIR / self.key


DATASETS: Dict[str, DatasetDefinition] = {
    "gl_monthly": DatasetDefinition(
        key="gl_monthly",
        title="General Ledger Monthly Summary",
        csv_filename="fact_gl_monthly.csv",
        config_filename="gl_monthly.yaml",
        markdown_prefix="gl_monthly",
        jsonl_filename="gl_monthly_documents.jsonl",
        key_fields=[
            "period",
            "gl_account",
            "cost_center",
            "order_number",
            "total_amount",
            "transaction_count",
            "document_types",
        ],
    ),
    "sales_customer": DatasetDefinition(
        key="sales_customer",
        title="Sales by Customer and Period",
        csv_filename="fact_sales_customer.csv",
        config_filename="sales_customer.yaml",
        markdown_prefix="sales_customer",
        jsonl_filename="sales_customer_documents.jsonl",
        key_fields=[
            "period",
            "customer_name",
            "material_description",
            "total_net_value",
            "total_gross_value",
            "total_tax",
            "line_item_count",
        ],
    ),
    "cost_allocation": DatasetDefinition(
        key="cost_allocation",
        title="Cost Allocation Summary",
        csv_filename="fact_cost_allocation.csv",
        config_filename="cost_allocation.yaml",
        markdown_prefix="cost_allocation",
        jsonl_filename="cost_allocation_documents.jsonl",
        key_fields=[
            "period",
            "cost_center",
            "gl_account",
            "partner_internal_order",
            "total_amount",
            "transaction_count",
            "document_types",
        ],
    ),
    "actual_sale": DatasetDefinition(
        key="actual_sale",
        title="Actual Sale Summary",
        csv_filename="fact_actual_sale.csv",
        config_filename="actual_sale.yaml",
        markdown_prefix="actual_sale",
        jsonl_filename="actual_sale_documents.jsonl",
        key_fields=[
            "period",
            "cost_element",
            "order_number",
            "material_code",
            "total_amount",
            "transaction_count",
            "document_types",
        ],
    ),
    "material_product": DatasetDefinition(
        key="material_product",
        title="Material and Product Summary",
        csv_filename="fact_material_product.csv",
        config_filename="material_product.yaml",
        markdown_prefix="material_product",
        jsonl_filename="material_product_documents.jsonl",
        key_fields=[
            "period",
            "material_code",
            "order_number",
            "total_amount",
            "transaction_count",
            "data_source",
        ],
    ),
}


def ensure_directories() -> None:
    """Create the shared output directories used by the pipeline."""

    for path in [
        DATA_ROOT,
        DOCUMENTS_DIR,
        MARKDOWN_DIR,
        INDEX_DIR,
        EMBEDDINGS_DIR,
        REPORTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS.values():
        dataset.documents_dir.mkdir(parents=True, exist_ok=True)
        dataset.markdown_dir.mkdir(parents=True, exist_ok=True)
        dataset.summary_index_dir.mkdir(parents=True, exist_ok=True)
        dataset.recursive_index_dir.mkdir(parents=True, exist_ok=True)
        dataset.chunk_index_base_dir.mkdir(parents=True, exist_ok=True)
        dataset.embedding_dir.mkdir(parents=True, exist_ok=True)


def dataset_choices(selection: str | None) -> Iterable[DatasetDefinition]:
    """Return dataset definitions based on a CLI selection string."""

    if not selection or selection.lower() == "all":
        return DATASETS.values()

    keys = [key.strip() for key in selection.split(",") if key.strip()]
    for key in keys:
        if key not in DATASETS:
            raise KeyError(f"Unknown dataset key '{key}'. Valid options: {', '.join(DATASETS)}")
        yield DATASETS[key]


def write_report(report_path: Path, payload: Dict) -> None:
    """Persist a JSON report with pretty formatting."""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = [
    "AGGREGATED_DIR",
    "CONFIG_DIR",
    "DATASETS",
    "DATA_ROOT",
    "DOCUMENTS_DIR",
    "MARKDOWN_DIR",
    "INDEX_DIR",
    "EMBEDDINGS_DIR",
    "REPORTS_DIR",
    "DatasetDefinition",
    "dataset_choices",
    "ensure_directories",
    "write_report",
]
