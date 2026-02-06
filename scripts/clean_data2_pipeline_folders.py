#!/usr/bin/env python3
"""
Remove redundant files from data2 pipeline folders so one format matches pipeline_metadata.csv.

Usage:
  python scripts/clean_data2_pipeline_folders.py [data_root]

  data_root  Default: data2. Should contain TCGA-KIRP/CT/, MR/, PT/ with early_stage/ and advanced_stage/.

If pipeline_metadata.csv lists .dcm paths: removes all .png from TCGA-KIRP/<mod>/<class>/.
If pipeline_metadata.csv lists .png paths: removes all .dcm from those folders.

Keeps patient folders (TCGA-B9-*, etc.) untouched; only cleans modality/class pipeline folders.
"""

import argparse
import csv
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_root", type=str, nargs="?", default="data2", help="Data root (default: data2)")
    args = parser.parse_args()

    root = Path(args.data_root).resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory.", file=sys.stderr)
        sys.exit(1)

    meta_file = root / "pipeline_metadata.csv"
    if not meta_file.exists():
        print(f"Error: {meta_file} not found.", file=sys.stderr)
        sys.exit(1)

    # Detect format from first data row
    with open(meta_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
    if not row:
        print("No rows in pipeline_metadata.csv.", file=sys.stderr)
        sys.exit(1)
    path = row.get("image_path", "")
    if path.endswith(".png"):
        remove_ext = ".dcm"
        keep_format = "PNG"
    else:
        remove_ext = ".png"
        keep_format = "DICOM (.dcm)"

    base = root / "TCGA-KIRP"
    if not base.is_dir():
        print(f"Error: {base} not found.", file=sys.stderr)
        sys.exit(1)

    mods = ["CT", "MR", "PT"]
    classes = ["early_stage", "advanced_stage"]
    removed = 0
    for mod in mods:
        for cls in classes:
            folder = base / mod / cls
            if not folder.is_dir():
                continue
            for f in folder.iterdir():
                if f.is_file() and f.suffix.lower() == remove_ext:
                    try:
                        f.unlink()
                        removed += 1
                    except OSError as e:
                        print(f"Warning: could not remove {f}: {e}", file=sys.stderr)

    print(f"Removed {removed} {remove_ext} files from pipeline folders. Pipeline uses {keep_format}.")


if __name__ == "__main__":
    main()
