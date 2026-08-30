"""
Create a new CSV that merges category_examples.csv with dataset metadata
and copy images into category subfolders.

Usage:
    python export_category_examples_with_metadata.py \
        --examples-csv paper_examples/category_examples.csv \
        --output-dir paper_examples/category_examples_with_metadata
"""

import argparse
import os
import shutil
from typing import Optional

import pandas as pd

from config.config import BASE_IMAGE_PATH, CSV_PATH


def sanitize_folder_name(name: str) -> str:
    cleaned = str(name).strip()
    if not cleaned:
        return "unknown"
    cleaned = cleaned.replace(os.sep, "_").replace("/", "_")
    return cleaned


def ensure_unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while True:
        candidate = f"{base}_{i}{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1


def resolve_category(row: pd.Series, preferred_column: Optional[str]) -> str:
    if preferred_column and preferred_column in row:
        return str(row[preferred_column])
    if "display_category" in row:
        return str(row["display_category"])
    if "category" in row:
        return str(row["category"])
    if "Type" in row:
        return str(row["Type"])
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge category_examples.csv with dataset metadata and copy images by category."
    )
    parser.add_argument(
        "--examples-csv",
        type=str,
        default="paper_examples/category_examples.csv",
        help="Path to category_examples.csv",
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default=CSV_PATH,
        help="Path to dataset metadata CSV (default: CSV_PATH).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="paper_examples/category_examples_with_metadata",
        help="Output directory for copied images and merged CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Output CSV path (default: <output-dir>/category_examples_with_metadata.csv).",
    )
    parser.add_argument(
        "--category-column",
        type=str,
        default=None,
        help="Column name to use for category subfolders (defaults to display_category/category/Type).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.examples_csv):
        raise FileNotFoundError(f"Examples CSV not found: {args.examples_csv}")
    if not os.path.exists(args.metadata_csv):
        raise FileNotFoundError(f"Metadata CSV not found: {args.metadata_csv}")

    examples_df = pd.read_csv(args.examples_csv)
    metadata_df = pd.read_csv(args.metadata_csv)

    if "image_path" not in examples_df.columns:
        raise ValueError("Expected 'image_path' column in examples CSV.")
    if "image_path" not in metadata_df.columns:
        raise ValueError("Expected 'image_path' column in metadata CSV.")

    merged_df = examples_df.merge(metadata_df, on="image_path", how="left", suffixes=("", "_meta"))

    os.makedirs(args.output_dir, exist_ok=True)
    output_csv = args.output_csv or os.path.join(
        args.output_dir, "category_examples_with_metadata.csv"
    )

    copied = 0
    missing = 0
    for _, row in merged_df.iterrows():
        category_name = sanitize_folder_name(resolve_category(row, args.category_column))
        category_dir = os.path.join(args.output_dir, category_name)
        os.makedirs(category_dir, exist_ok=True)

        rel_path = str(row["image_path"])
        src_path = os.path.join(BASE_IMAGE_PATH, rel_path)
        if not os.path.exists(src_path):
            missing += 1
            continue

        dest_name = os.path.basename(rel_path)
        dest_path = ensure_unique_path(os.path.join(category_dir, dest_name))
        shutil.copy2(src_path, dest_path)
        copied += 1

    merged_df.to_csv(output_csv, index=False)

    print(f"Merged CSV saved to: {output_csv}")
    print(f"Images copied: {copied}")
    if missing:
        print(f"Images missing: {missing}")


if __name__ == "__main__":
    main()
