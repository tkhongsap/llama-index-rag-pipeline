"""Run the Orion markdown→index→embedding flow in a single batch."""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

from paths import ensure_directories

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("orion.batch")

PIPELINE_SCRIPTS = [
    "04_build_summary_index.py",
    "05_build_recursive_retriever.py",
    "06_analyze_index_structure.py",
    "07_extract_embeddings.py",
]


def run_script(script: str, datasets: str) -> None:
    script_path = Path(__file__).parent / script
    cmd = [sys.executable, str(script_path), "--datasets", datasets]
    LOGGER.info("Running %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma separated dataset keys to process (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    ensure_directories()
    args = parse_args()

    for script in PIPELINE_SCRIPTS:
        run_script(script, args.datasets)

    LOGGER.info("Batch processing complete")


if __name__ == "__main__":
    main()
