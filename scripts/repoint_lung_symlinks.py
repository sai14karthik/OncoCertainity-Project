#!/usr/bin/env python3
"""
Repoint all symlinks in project Lung-PET-CT-Dx to a new DICOM root.

Use when you have restored the patient DICOM folder (Lung_Dx-A0001, etc.)
to a different path than manifest-1608669183333/Lung-PET-CT-Dx.

  python scripts/repoint_lung_symlinks.py /path/to/Lung-PET-CT-Dx

The path must contain patient folders (Lung_Dx-A0001, Lung_Dx-B0001, ...)
with study/series/*.dcm inside.
"""
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LUNG_DX = PROJECT_ROOT / "Lung-PET-CT-Dx"


def main():
    if len(sys.argv) < 2:
        print("Usage: python repoint_lung_symlinks.py /path/to/Lung-PET-CT-Dx", file=sys.stderr)
        sys.exit(1)
    new_root = Path(sys.argv[1]).resolve()
    if not new_root.is_dir():
        print(f"Error: {new_root} is not a directory.", file=sys.stderr)
        sys.exit(1)
    # Expect patient folders
    patients = [d for d in new_root.iterdir() if d.is_dir() and re.match(r"Lung_Dx-[ABGE]\d+", d.name)]
    if not patients:
        print(f"Error: No patient folders (Lung_Dx-A0001, ...) in {new_root}", file=sys.stderr)
        sys.exit(1)

    # Build mapping: (patient, series_normalized, dcm_basename) -> Path
    # Link names: Lung_Dx-A0002_2.000000-ThoraxRoutine__8.0.0__B70f-62919_1-01.dcm
    def norm(s):
        return s.replace(" ", "_").replace("/", "_").replace("\\", "_")
    file_map = {}  # (patient, series_slug, base) -> Path
    for patient_dir in new_root.iterdir():
        if not patient_dir.is_dir() or not re.match(r"Lung_Dx-[ABGE]\d+", patient_dir.name):
            continue
        for study_dir in patient_dir.iterdir():
            if not study_dir.is_dir():
                continue
            for series_dir in study_dir.iterdir():
                if not series_dir.is_dir():
                    continue
                series_slug = norm(series_dir.name)
                for f in series_dir.iterdir():
                    if f.suffix.lower() == ".dcm":
                        file_map[(patient_dir.name, series_slug, f.name)] = f

    n_ok = 0
    n_missing = 0
    n_repointed = 0
    for mod in ["CT", "PT"]:
        for cls in ["non_small_cell", "small_cell"]:
            d = LUNG_DX / mod / cls
            if not d.is_dir():
                continue
            for link in d.iterdir():
                if not link.is_symlink():
                    continue
                if link.exists():
                    n_ok += 1
                    continue
                # Broken: name = Lung_Dx-A0002_2.000000-ThoraxRoutine__8.0.0__B70f-62919_1-01.dcm
                name = link.name
                m = re.match(r"(Lung_Dx-[ABGE]\d+)_(.+)_(\d+-\d+\.dcm)", name)
                if not m:
                    n_missing += 1
                    continue
                patient, series_slug, base = m.groups()
                key = (patient, series_slug, base)
                if key in file_map:
                    target = file_map[key]
                    if target.exists():
                        link.unlink()
                        link.symlink_to(target)
                        n_repointed += 1
                    else:
                        n_missing += 1
                else:
                    n_missing += 1

    print(f"Valid symlinks: {n_ok}, repointed: {n_repointed}, still missing: {n_missing}")
    if n_missing > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
