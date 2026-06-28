"""Aggregate per-dataset metric CSVs into a single summary directory.

Scans a base directory for per-dataset subdirectories, reads all CSV files from
each, injects a `dataset` column, and writes merged copies into a summary/ folder.

Usage:
  python scripts/aggregate_metrics.py --dir outputs
CPU-only/post-hoc.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def aggregate_metrics(output_dir: str, summary_dir_name: str = "summary"):
    base_path = Path(output_dir)

    if not base_path.exists() or not base_path.is_dir():
        logger.error(f"Directory not found: {base_path}")
        return

    out_path = base_path / summary_dir_name
    out_path.mkdir(parents=True, exist_ok=True)

    merged_data = {}

    logger.info(f"Scanning base directory: {base_path}")

    for dataset_dir in base_path.iterdir():
        if dataset_dir.is_dir() and dataset_dir.name != summary_dir_name:
            dataset_name = dataset_dir.name
            logger.info(f"Found dataset directory: {dataset_name}")

            for csv_file in dataset_dir.glob("*.csv"):
                file_name = csv_file.name
                try:
                    df = pd.read_csv(csv_file)

                    df.insert(0, "dataset", dataset_name)

                    if file_name not in merged_data:
                        merged_data[file_name] = []
                    merged_data[file_name].append(df)

                except pd.errors.EmptyDataError:
                    logger.warning(f"Skipping empty file: {csv_file}")
                except Exception as e:
                    logger.error(f"Failed to read {csv_file}: {e}")

    if not merged_data:
        logger.warning("No CSV files found to merge.")
        return

    logger.info("--- Start Merging ---")
    for file_name, df_list in merged_data.items():
        try:
            merged_df = pd.concat(df_list, ignore_index=True)

            save_path = out_path / file_name
            merged_df.to_csv(save_path, index=False)
            logger.info(
                f"Saved aggregated file: {save_path.name} (Total rows: {len(merged_df)})"
            )
        except Exception as e:
            logger.error(f"Failed to merge {file_name}: {e}")

    logger.info(f"All done! Your aggregated tables are waiting in: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate metrics CSVs across dataset folders."
    )
    parser.add_argument(
        "--dir", default="outputs", help="Path to the outputs directory"
    )
    args = parser.parse_args()

    aggregate_metrics(args.dir)
