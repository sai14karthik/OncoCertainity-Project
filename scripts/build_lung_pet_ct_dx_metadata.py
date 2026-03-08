#!/usr/bin/env python3
"""
Build pipeline-ready metadata and folder structure for Lung-PET-CT-Dx from NBIA manifest.

Labels are derived from subject ID prefix per NBIA documentation:
  A = Adenocarcinoma
  B = Small Cell
  E = Large Cell
  G = Squamous

Usage:
  python scripts/build_lung_pet_ct_dx_metadata.py <lung_pet_ct_dx_root> [--output-dir DIR] [--no-symlink] [--two-class]

  lung_pet_ct_dx_root   Root of Lung-PET-CT-Dx (contains manifest-* folder with Lung-PET-CT-Dx/).
  --output-dir          Where to write pipeline_metadata.csv and modality/class folders (default: lung_pet_ct_dx_root).
  --no-symlink          Copy files instead of symlinks (default: symlink).
  --two-class           Binary: A,G,E -> non_small_cell (0), B -> small_cell (1). All four histologies included.

Output:
  With --two-class: CT/non_small_cell/, CT/small_cell/, PT/non_small_cell/, PT/small_cell/
  (A,G,E -> non_small_cell; B -> small_cell)
  Without: CT/adenocarcinoma/, CT/small_cell/, CT/large_cell/, CT/squamous/, PT/...
"""

import argparse
import csv
import os
import re
import random
import sys
from pathlib import Path
from typing import Optional

LABEL_MAP = {"A": "adenocarcinoma", "B": "small_cell", "E": "large_cell", "G": "squamous"}
# Binary: A,G,E -> 0 (non_small_cell), B -> 1 (small_cell)
TWO_CLASS_MAP = {"A": "non_small_cell", "G": "non_small_cell", "E": "non_small_cell", "B": "small_cell"}


def get_label(subject_id: str, two_class: bool = False) -> str:
    """Derive label from subject ID. two_class: A,G,E -> non_small_cell (0), B -> small_cell (1)."""
    m = re.match(r"Lung_Dx-([ABEG])\d+", subject_id)
    if not m:
        return "unknown"
    letter = m.group(1)
    if two_class:
        return TWO_CLASS_MAP.get(letter, "unknown")
    return LABEL_MAP.get(letter, "unknown")


def classify_modality(study: str, series: str) -> Optional[str]:
    """Return 'CT' or 'PT' based on SERIES name (PET/CT studies contain both CT and PET series)."""
    s_upper = series.upper()
    # PET series: e.g. "PET WB Corrected", "PET WB"
    if "PET" in s_upper and ("CORRECTED" in s_upper or " WB " in s_upper or s_upper.strip().startswith("PET")):
        return "PT"
    if "PET " in s_upper and "CT" not in s_upper:
        return "PT"
    # CT series (including "CT WB" inside PET/CT studies)
    if "CT " in s_upper or "CT WB" in s_upper or "-CT-" in s_upper:
        return "CT"
    if any(x in s_upper for x in ("CHEST", "THORAX", "LUNG")):
        return "CT"
    return None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "lung_root",
        type=str,
        help="Root of Lung-PET-CT-Dx download (has manifest-* and Lung-PET-CT-Dx/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output root (default: lung_root)",
    )
    parser.add_argument(
        "--no-symlink",
        action="store_true",
        help="Copy files instead of symlinks (default: symlink)",
    )
    parser.add_argument(
        "--two-class",
        action="store_true",
        help="Binary: A,G,E -> non_small_cell (0), B -> small_cell (1). All four histologies included.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default=None,
        help="Path to folder containing manifest-* (or to manifest-* itself). Use if manifest is not under lung_root.",
    )
    args = parser.parse_args()

    lung_root = Path(args.lung_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else lung_root
    use_symlink = not args.no_symlink

    # Find manifest folder
    if args.manifest_dir:
        md = Path(args.manifest_dir).resolve()
        if (md / "Lung-PET-CT-Dx").is_dir():
            manifest_dir = md
        elif md.name.startswith("manifest-") and (md / "Lung-PET-CT-Dx").is_dir():
            manifest_dir = md
        else:
            manifest_candidates = list(md.glob("manifest-*"))
            if not manifest_candidates:
                print(f"Error: No manifest-* (with Lung-PET-CT-Dx inside) found under {md}", file=sys.stderr)
                sys.exit(1)
            manifest_dir = manifest_candidates[0]
    else:
        manifest_candidates = list(lung_root.glob("manifest-*"))
        if not manifest_candidates:
            print(f"Error: No manifest-* folder found under {lung_root}. Re-download from NBIA or pass --manifest-dir /path/to/manifest.", file=sys.stderr)
            sys.exit(1)
        manifest_dir = manifest_candidates[0]
    data_dir = manifest_dir / "Lung-PET-CT-Dx"
    if not data_dir.is_dir():
        print(f"Error: {data_dir} not found.", file=sys.stderr)
        sys.exit(1)

    # Create output structure (binary: non_small_cell=0, small_cell=1)
    if args.two_class:
        class_names = ["non_small_cell", "small_cell"]
    else:
        class_names = list(LABEL_MAP.values())
    for mod in ("CT", "PT"):
        for cls in class_names:
            (output_dir / mod / cls).mkdir(parents=True, exist_ok=True)

    pipeline_rows = []
    seen_names = {}  # (mod, cls, unique_name) -> count for dedup
    n_skipped = 0
    n_linked = 0

    subject_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith("."))

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        label = get_label(subject_id, two_class=args.two_class)
        if label == "unknown":
            n_skipped += 1
            continue

        for study_dir in subject_dir.iterdir():
            if not study_dir.is_dir():
                continue
            for series_dir in study_dir.iterdir():
                if not series_dir.is_dir():
                    continue
                mod = classify_modality(study_dir.name, series_dir.name)
                if mod is None:
                    continue

                dcm_files = list(series_dir.glob("*.dcm"))
                for dcm in dcm_files:
                    unique_name = f"{subject_id}_{series_dir.name}_{dcm.stem}.dcm".replace(" ", "_")
                    for c in "/\\:*?\"<>|":
                        unique_name = unique_name.replace(c, "_")
                    target_rel = f"{mod}/{label}/{unique_name}"
                    target_abs = output_dir / target_rel

                    if not target_abs.exists():
                        try:
                            if use_symlink:
                                target_abs.symlink_to(dcm.resolve())
                            else:
                                import shutil
                                shutil.copy2(dcm, target_abs)
                            n_linked += 1
                        except OSError as e:
                            print(f"Warning: skip {dcm}: {e}", file=sys.stderr)
                            continue

                    pipeline_rows.append({
                        "image_path": target_rel,
                        "patient_id": subject_id,
                        "label": label,
                        "modality": mod,
                        "split": None,
                    })

    if not pipeline_rows:
        print("No images found. Check paths and modality classification.", file=sys.stderr)
        sys.exit(1)

    # Assign split by patient (80/10/10)
    patients = sorted(set(r["patient_id"] for r in pipeline_rows))
    random.seed(42)
    random.shuffle(patients)
    n = len(patients)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_set = set(patients[:n_train])
    val_set = set(patients[n_train : n_train + n_val])
    for r in pipeline_rows:
        if r["patient_id"] in train_set:
            r["split"] = "train"
        elif r["patient_id"] in val_set:
            r["split"] = "val"
        else:
            r["split"] = "test"

    out_csv = output_dir / "pipeline_metadata.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "patient_id", "label", "modality", "split"]
        )
        writer.writeheader()
        writer.writerows(pipeline_rows)

    # Stats
    by_mod = {}
    by_label = {}
    for r in pipeline_rows:
        by_mod[r["modality"]] = by_mod.get(r["modality"], 0) + 1
        by_label[r["label"]] = by_label.get(r["label"], 0) + 1

    print(f"Wrote {len(pipeline_rows)} rows to {out_csv}")
    print(f"Modalities: {by_mod}")
    print(f"Classes: {by_label}")
    print(f"Patients: {len(patients)} (train {len(train_set)}, val {len(val_set)}, test {n - len(train_set) - len(val_set)})")
    print(f"Linked/copied: {n_linked} new files")


if __name__ == "__main__":
    main()
