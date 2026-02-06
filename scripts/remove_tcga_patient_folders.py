#!/usr/bin/env python3
"""
Remove patient folders from data2/TCGA-KIRP so only CT, MR, PT (pipeline folders) remain.

Usage:
  python scripts/remove_tcga_patient_folders.py [data_root]

  data_root  Default: data2. Must contain TCGA-KIRP/ with CT/, MR/, PT/.

Removes every directory under TCGA-KIRP/ that is not CT, MR, or PT.
Keeps LICENSE and any files. After this you cannot rebuild pipeline from DICOMs
without re-downloading the patient data.
"""

import argparse
import shutil
import sys
from pathlib import Path

KEEP = {"CT", "MR", "PT"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_root", type=str, nargs="?", default="data2", help="Data root (default: data2)")
    args = parser.parse_args()

    root = Path(args.data_root).resolve()
    base = root / "TCGA-KIRP"
    if not base.is_dir():
        print(f"Error: {base} not found.", file=sys.stderr)
        sys.exit(1)

    removed = []
    for item in base.iterdir():
        if item.is_dir() and item.name not in KEEP:
            try:
                shutil.rmtree(item)
                removed.append(item.name)
            except OSError as e:
                print(f"Warning: could not remove {item}: {e}", file=sys.stderr)

    print(f"Removed {len(removed)} patient folders from {base}.")
    print(f"Kept: CT/, MR/, PT/. Remaining under TCGA-KIRP: {sorted(p.name for p in base.iterdir())}")


if __name__ == "__main__":
    main()
