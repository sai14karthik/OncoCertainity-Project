#!/usr/bin/env python3
"""
Fix project Lung-PET-CT-Dx from manifest.
Run after manifest-1608669183333/Lung-PET-CT-Dx is restored (re-download from NBIA).

  python scripts/fix_lung_pet_ct_dx_from_manifest.py

This will:
  1. Require manifest-1608669183333/Lung-PET-CT-Dx to exist with patient folders.
  2. Populate project Lung-PET-CT-Dx with CT/, PT/ (symlinks) and pipeline_metadata.csv.
  3. Use two-class mapping: A,G,E -> non_small_cell, B -> small_cell.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = PROJECT_ROOT / "manifest-1608669183333"
DATA_DIR = MANIFEST_DIR / "Lung-PET-CT-Dx"
OUTPUT_DIR = PROJECT_ROOT / "Lung-PET-CT-Dx"


def main():
    if not DATA_DIR.is_dir():
        print(
            f"Error: {DATA_DIR} not found.\n"
            "Restore the Lung-PET-CT-Dx folder inside the manifest (re-download from NBIA), then run this again.",
            file=sys.stderr,
        )
        sys.exit(1)
    subjects = [d for d in DATA_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not subjects:
        print(f"Error: No patient folders in {DATA_DIR}", file=sys.stderr)
        sys.exit(1)
    # Run the build script
    import subprocess
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_lung_pet_ct_dx_metadata.py"),
            str(PROJECT_ROOT),
            "--manifest-dir",
            str(MANIFEST_DIR),
            "--output-dir",
            str(OUTPUT_DIR),
            "--two-class",
        ],
        cwd=str(PROJECT_ROOT),
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
