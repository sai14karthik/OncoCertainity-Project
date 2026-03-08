#!/usr/bin/env python3
"""
Remove obsolete 4-class folders from Lung-PET-CT-Dx so only non_small_cell and small_cell remain.
Run from project root; pass the Lung-PET-CT-Dx root (e.g. "Lung-PET-CT-Dx ").

Usage:
  python scripts/clean_lung_pet_ct_dx.py "Lung-PET-CT-Dx "
"""

import argparse
import shutil
import sys
from pathlib import Path

# 2-class (NSCLC vs SCLC): keep non_small_cell, small_cell; remove adenocarcinoma, large_cell, squamous
OBSOLETE_CLASSES = ["adenocarcinoma", "large_cell", "squamous"]
KEEP_CLASSES = ["non_small_cell", "small_cell"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lung_root", type=str, help="Lung-PET-CT-Dx root (has CT/, PT/)")
    args = parser.parse_args()

    root = Path(args.lung_root).resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory.", file=sys.stderr)
        sys.exit(1)

    removed = []
    for mod in ("CT", "PT"):
        mod_dir = root / mod
        if not mod_dir.is_dir():
            continue
        for cls in OBSOLETE_CLASSES:
            cls_dir = mod_dir / cls
            if cls_dir.is_dir():
                print(f"Removing {cls_dir} ...", flush=True)
                shutil.rmtree(cls_dir)
                removed.append(str(cls_dir))

    if removed:
        print(f"\nRemoved {len(removed)} folders. CT/ and PT/ now only have: {KEEP_CLASSES}")
    else:
        print("No obsolete folders found (already clean).")


if __name__ == "__main__":
    main()
