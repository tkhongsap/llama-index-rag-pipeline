#!/usr/bin/env python3
"""
Standalone entry point for the sales promotion data processing pipeline.

Usage:
    python -m src_sale_promotion.data_processing.run_data_processing_standalone
    python -m src_sale_promotion.data_processing.run_data_processing_standalone --input path/to/file.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

# Ensure project root is on sys.path when executed directly
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_sale_promotion.common import INPUT_DOCS_DIR, PROCESSED_DOCS_DIR
from src_sale_promotion.data_processing.main import PromotionDataProcessor

SAMPLE_CSV = Path(__file__).parent.parent / "data" / "promotion_list.csv"


def _discover_csv_files(path: Path) -> List[Path]:
    """Return a sorted list of CSV files from a file or directory."""
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Input file must be .csv, got: {path}")
        return [path]

    if path.is_dir():
        csv_files = sorted(path.glob("*.csv"))
        if csv_files:
            return csv_files
        raise FileNotFoundError(f"No CSV files found in directory: {path}")

    raise FileNotFoundError(f"Input path does not exist: {path}")


def _select_csv(args) -> Path:
    """Resolve which CSV file should be processed."""
    if args.sample:
        if not SAMPLE_CSV.exists():
            raise FileNotFoundError(
                f"Sample CSV not found at {SAMPLE_CSV}. "
                "Please ensure the mock data file exists."
            )
        logging.info("Using packaged sample CSV for dry run/testing")
        return SAMPLE_CSV

    input_path = Path(args.input).expanduser()
    csv_files = _discover_csv_files(input_path)

    if len(csv_files) > 1:
        logging.warning(
            "Multiple CSV files detected under %s. "
            "Processing the first file: %s",
            input_path,
            csv_files[0].name,
        )

    return csv_files[0]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Convert sales promotion CSVs into structured documents",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(INPUT_DOCS_DIR),
        help="Input CSV file or directory (default: data/input_docs/sale-promotion/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROCESSED_DOCS_DIR),
        help="Output directory for processed docs (default: data/processed/sale-promotion/)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Optional path to promotion_fields YAML config",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Process the packaged sample CSV instead of local data",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for data processing."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    try:
        csv_path = _select_csv(args)
    except (FileNotFoundError, ValueError) as exc:
        logging.error("%s", exc)
        logging.info("Expected CSV files inside: %s", INPUT_DOCS_DIR)
        return 1

    output_dir = Path(args.output).expanduser()
    logging.info("📂 Input CSV: %s", csv_path)
    logging.info("🗂️  Output directory: %s", output_dir)

    try:
        processor = PromotionDataProcessor(
            config_path=args.config,
            output_dir=str(output_dir),
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.exception("Failed to initialize processor: %s", exc)
        return 1

    try:
        documents = processor.process(str(csv_path))
        logging.info("✅ Completed processing %s promotions", len(documents))
        return 0
    except Exception as exc:  # pragma: no cover - surface stack trace to console
        logging.exception("Processing failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
