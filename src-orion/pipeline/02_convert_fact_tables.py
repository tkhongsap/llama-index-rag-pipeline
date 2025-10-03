"""Convert aggregated Orion financial CSVs into JSONL documents."""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Iterable, List

from paths import DATASETS, ensure_directories, dataset_choices

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("orion.convert")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_MODULE = PROJECT_ROOT / "src" / "02_prep_doc_for_embedding.py"


def load_converter_class():
    """Dynamically load FlexibleCSVConverter from the main pipeline script."""

    spec = importlib.util.spec_from_file_location("csv_converter", SRC_MODULE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load converter from {SRC_MODULE}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FlexibleCSVConverter


FlexibleCSVConverter = load_converter_class()


def process_dataset(dataset_key: str) -> Path:
    dataset = DATASETS[dataset_key]
    LOGGER.info("Processing %s", dataset.title)

    converter = FlexibleCSVConverter(
        input_csv_path=str(dataset.csv_path),
        output_dir=str(dataset.documents_dir),
        config_path=str(dataset.config_path),
    )

    converter.setup_configuration(auto_generate=False)
    documents = converter.process_csv_to_documents()
    jsonl_path = converter.save_documents_as_jsonl(documents, filename=dataset.jsonl_filename)

    LOGGER.info("Saved %s documents to %s", len(documents), jsonl_path)
    return Path(jsonl_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma separated dataset keys to convert (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()

    for dataset in dataset_choices(args.datasets):
        if not dataset.csv_path.exists():
            LOGGER.warning("Skipping %s (CSV not found)", dataset.title)
            continue
        process_dataset(dataset.key)


if __name__ == "__main__":
    main()
