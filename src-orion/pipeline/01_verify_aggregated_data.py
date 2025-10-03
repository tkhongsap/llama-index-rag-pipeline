"""Verify Orion aggregated financial datasets are ready for the RAG pipeline."""
from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from paths import (
    AGGREGATED_DIR,
    DATASETS,
    REPORTS_DIR,
    ensure_directories,
    write_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("orion.verify")


def summarise_csv(csv_path: Path) -> Dict:
    """Load the CSV and produce lightweight summary metrics."""

    df = pd.read_csv(csv_path)
    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
    }

    if "period" in df.columns:
        summary["periods"] = sorted(df["period"].dropna().unique().tolist())

    if "total_amount" in df.columns:
        summary["total_amount_sum"] = float(df["total_amount"].fillna(0).sum())

    return summary


def load_aggregation_summary() -> Dict | None:
    """Read the optional aggregation summary JSON file if it exists."""

    summary_path = AGGREGATED_DIR / "aggregation_summary.json"
    if not summary_path.exists():
        return None

    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_verification(report_name: str) -> Dict:
    """Verify that all required aggregated CSVs exist and capture metrics."""

    ensure_directories()

    results: Dict[str, Dict] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "aggregated_dir": str(AGGREGATED_DIR),
        "datasets": {},
    }

    for dataset in DATASETS.values():
        csv_path = dataset.csv_path
        dataset_result: Dict[str, Dict] = {
            "exists": csv_path.exists(),
            "path": str(csv_path),
        }

        if csv_path.exists():
            LOGGER.info("✅ %s located at %s", dataset.title, csv_path)
            dataset_result["summary"] = summarise_csv(csv_path)
        else:
            LOGGER.warning("❌ Missing dataset %s", dataset.title)

        results["datasets"][dataset.key] = dataset_result

    aggregation_summary = load_aggregation_summary()
    if aggregation_summary:
        results["aggregation_summary"] = aggregation_summary

    report_path = REPORTS_DIR / f"{report_name}.json"
    write_report(report_path, results)
    LOGGER.info("Verification report written to %s", report_path)

    missing = [key for key, info in results["datasets"].items() if not info["exists"]]
    if missing:
        LOGGER.error("Missing datasets: %s", ", ".join(missing))
        raise SystemExit(1)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-name",
        default="aggregated_dataset_status",
        help="Filename (without extension) for the verification report.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_verification(args.report_name)
