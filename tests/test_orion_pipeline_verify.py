"""Unit tests for the Orion financial pipeline helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = PROJECT_ROOT / "src-orion" / "pipeline"


def load_module(name: str, path: Path):
    """Load a module from ``path`` and register it under ``name``."""

    if name in sys.modules:
        del sys.modules[name]

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module '{name}' from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def pipeline_modules():
    """Provide freshly loaded pipeline modules for each test."""

    paths_module = load_module("paths", PIPELINE_DIR / "paths.py")
    verify_module = load_module("orion_verify", PIPELINE_DIR / "01_verify_aggregated_data.py")
    return paths_module, verify_module


def patch_pipeline_directories(paths_module, verify_module, tmp_path: Path) -> None:
    """Redirect pipeline paths to a temporary directory tree for testing."""

    aggregated_dir = tmp_path / "aggregated"
    data_root = tmp_path / "orion"
    documents_dir = data_root / "documents"
    markdown_dir = data_root / "markdown"
    index_dir = data_root / "indexes"
    embeddings_dir = data_root / "embeddings"
    reports_dir = data_root / "reports"

    aggregated_dir.mkdir(parents=True, exist_ok=True)

    for module in (paths_module, verify_module):
        setattr(module, "AGGREGATED_DIR", aggregated_dir)
        setattr(module, "DATA_ROOT", data_root)
        setattr(module, "DOCUMENTS_DIR", documents_dir)
        setattr(module, "MARKDOWN_DIR", markdown_dir)
        setattr(module, "INDEX_DIR", index_dir)
        setattr(module, "EMBEDDINGS_DIR", embeddings_dir)
        setattr(module, "REPORTS_DIR", reports_dir)

    # Ensure verify module references the shared dataset registry defined in paths.
    verify_module.DATASETS = paths_module.DATASETS


def test_dataset_choices_filters_selection(pipeline_modules):
    paths_module, _ = pipeline_modules

    # Explicit selection returns only the requested datasets.
    selected = list(paths_module.dataset_choices("gl_monthly, sales_customer"))
    assert [dataset.key for dataset in selected] == ["gl_monthly", "sales_customer"]

    # "all" returns the full registry.
    all_datasets = list(paths_module.dataset_choices("all"))
    assert [dataset.key for dataset in all_datasets] == list(paths_module.DATASETS.keys())

    # Unknown dataset keys raise a helpful error.
    with pytest.raises(KeyError):
        list(paths_module.dataset_choices("does_not_exist"))


def test_run_verification_creates_report(tmp_path, pipeline_modules):
    paths_module, verify_module = pipeline_modules
    patch_pipeline_directories(paths_module, verify_module, tmp_path)

    # Create minimal CSVs for every dataset to satisfy verification checks.
    for dataset in paths_module.DATASETS.values():
        df = pd.DataFrame({"period": ["2024-01"], "total_amount": [100.0]})
        df.to_csv(paths_module.AGGREGATED_DIR / dataset.csv_filename, index=False)

    report_name = "aggregated_dataset_status"
    results = verify_module.run_verification(report_name)

    assert set(results["datasets"].keys()) == set(paths_module.DATASETS.keys())
    assert all(info["exists"] for info in results["datasets"].values())

    report_path = verify_module.REPORTS_DIR / f"{report_name}.json"
    assert report_path.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["datasets"]["gl_monthly"]["summary"]["rows"] == 1


def test_run_verification_flags_missing_dataset(tmp_path, pipeline_modules):
    paths_module, verify_module = pipeline_modules
    patch_pipeline_directories(paths_module, verify_module, tmp_path)

    # Only create a CSV for the first dataset to force a failure on the rest.
    first_dataset = next(iter(paths_module.DATASETS.values()))
    df = pd.DataFrame({"period": ["2024-01"], "total_amount": [42.0]})
    df.to_csv(paths_module.AGGREGATED_DIR / first_dataset.csv_filename, index=False)

    with pytest.raises(SystemExit):
        verify_module.run_verification("aggregated_dataset_status")

    report_path = verify_module.REPORTS_DIR / "aggregated_dataset_status.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    missing = [key for key, info in payload["datasets"].items() if not info["exists"]]
    assert len(missing) == len(paths_module.DATASETS) - 1

