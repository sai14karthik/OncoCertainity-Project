#!/usr/bin/env python3
"""
Build pipeline-ready metadata and folder structure for TCGA-KIRP from NBIA/TCIA manifest.

Usage:
  python scripts/build_tcga_kirp_metadata.py <manifest_dir> [--output-dir DIR] [--clinical-csv PATH] [--no-symlink] [--to-png]

  manifest_dir     Root of downloaded TCGA-KIRP (contains metadata.csv and TCGA-KIRP/).
  --output-dir     Where to write pipeline_metadata.csv and optional modality/class folders (default: manifest_dir).
  --clinical-csv   Optional CSV with Subject ID (or patient_id) and label (e.g. data2/clinical_tcga_kirp.csv).
  --no-symlink     Copy files instead of symlinks (default: symlink). Ignored if --to-png.
  --to-png         Convert DICOM to PNG so pipeline uses PNG like data/ (same format as data folder).

Output:
  - <output_dir>/pipeline_metadata.csv with: image_path, patient_id, label, modality, split
  - <output_dir>/TCGA-KIRP/CT/early_stage/, CT/advanced_stage/, MR/..., PT/... (or class names from clinical CSV)
    with DICOM files symlinked/copied, or PNG files if --to-png.

Without --clinical-csv, labels are placeholder class0/class1 (by patient_id hash) so the pipeline runs;
replace with real labels from TCGA clinical data when available.
"""

import argparse
import csv
import hashlib
import os
import random
import sys
from pathlib import Path

# Allow importing project utils (dicom_utils) when converting to PNG
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest_dir", type=str, help="Root of TCGA-KIRP download (has metadata.csv and TCGA-KIRP/)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output root (default: manifest_dir)")
    parser.add_argument("--clinical-csv", type=str, default=None, help="CSV with Subject ID (or patient_id) and label")
    parser.add_argument("--no-symlink", action="store_true", help="Copy files instead of symlinks (default: symlink)")
    parser.add_argument("--to-png", action="store_true", help="Convert DICOM to PNG so data2 uses PNG like data/")
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else manifest_dir
    use_symlink = not args.no_symlink and not args.to_png
    to_png = args.to_png

    manifest_csv = manifest_dir / "metadata.csv"
    if not manifest_csv.exists():
        print(f"Error: {manifest_csv} not found.", file=sys.stderr)
        sys.exit(1)

    # Load optional clinical labels: Subject ID -> label
    subject_to_label = {}
    if args.clinical_csv:
        clinical_path = Path(args.clinical_csv).resolve()
        if not clinical_path.exists():
            print(f"Error: Clinical CSV not found: {clinical_path}", file=sys.stderr)
            sys.exit(1)
        with open(clinical_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            id_col = "Subject ID" if "Subject ID" in reader.fieldnames else "patient_id"
            if id_col not in reader.fieldnames or "label" not in reader.fieldnames:
                print(f"Error: Clinical CSV must have '{id_col}' and 'label' columns.", file=sys.stderr)
                sys.exit(1)
            for row in reader:
                sid = row.get(id_col, "").strip()
                label = row.get("label", "").strip()
                if sid and label:
                    subject_to_label[sid] = label

    # Use class names from clinical or placeholders
    if subject_to_label:
        class_names = sorted(set(subject_to_label.values()))
        if len(class_names) != 2:
            print(f"Warning: Expected 2 classes, got {class_names}. Using as-is.", file=sys.stderr)
    else:
        class_names = ["class0", "class1"]

    # Read manifest
    rows_by_modality = {}  # modality -> list of (Subject ID, File Location)
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("Subject ID", "").strip()
            mod = row.get("Modality", "").strip()
            loc = row.get("File Location", "").strip()
            if not sid or not mod or not loc:
                continue
            if mod not in rows_by_modality:
                rows_by_modality[mod] = []
            rows_by_modality[mod].append((sid, loc))

    # Build output structure: output_dir/TCGA-KIRP/CT/class0/, etc.
    base = output_dir / "TCGA-KIRP"
    for mod in rows_by_modality:
        for cls in class_names:
            (base / mod / cls).mkdir(parents=True, exist_ok=True)

    # Collect all image paths and assign labels
    pipeline_rows = []
    seen_paths = set()
    n_converted = 0  # for --to-png progress

    for mod, entries in rows_by_modality.items():
        for sid, file_loc in entries:
            series_path = manifest_dir / file_loc.lstrip("./")
            if not series_path.exists():
                continue
            # List DICOM files in this series folder
            dcm_files = list(series_path.glob("*.dcm")) or list(series_path.rglob("*.dcm"))
            if not dcm_files:
                continue
            label = subject_to_label.get(sid)
            if label is None:
                # Placeholder: deterministic class by patient_id
                h = int(hashlib.md5(sid.encode()).hexdigest(), 16) % 2
                label = class_names[h]
            for dcm in dcm_files:
                rel_path = os.path.relpath(dcm, manifest_dir)
                if rel_path in seen_paths:
                    continue
                seen_paths.add(rel_path)
                ext = ".png" if to_png else ".dcm"
                unique_name = f"{sid}_{dcm.stem}{ext}".replace(" ", "_")
                target_rel = f"TCGA-KIRP/{mod}/{label}/{unique_name}"
                target_abs = output_dir / target_rel
                if not target_abs.exists():
                    try:
                        if to_png:
                            from src.utils.dicom_utils import load_dicom_image
                            img = load_dicom_image(str(dcm))
                            img.save(target_abs, "PNG")
                            n_converted += 1
                            if n_converted % 500 == 0:
                                print(f"Converted {n_converted} DICOMs to PNG...", flush=True)
                        elif use_symlink:
                            target_abs.symlink_to(dcm.resolve())
                        else:
                            import shutil
                            shutil.copy2(dcm, target_abs)
                    except OSError as e:
                        print(f"Warning: skip {dcm}: {e}", file=sys.stderr)
                        continue
                    except Exception as e:
                        print(f"Warning: skip {dcm}: {e}", file=sys.stderr)
                        continue
                pipeline_rows.append({
                    "image_path": target_rel,
                    "patient_id": sid,
                    "label": label,
                    "modality": mod,
                    "split": None,  # set below
                })

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

    # Write pipeline_metadata.csv (relative to output_dir)
    out_csv = output_dir / "pipeline_metadata.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "patient_id", "label", "modality", "split"])
        writer.writeheader()
        writer.writerows(pipeline_rows)

    print(f"Wrote {len(pipeline_rows)} rows to {out_csv}")
    print(f"Modalities: {list(rows_by_modality.keys())}")
    print(f"Classes: {class_names}")
    print(f"Patients: {len(patients)} (train {len(train_set)}, val {len(val_set)}, test {len(patients)-len(train_set)-len(val_set)})")
    if to_png:
        print("Output format: PNG (same as data/).")
    if not subject_to_label:
        print("Labels are placeholder (class0/class1). Add --clinical-csv for real labels.")


if __name__ == "__main__":
    main()
