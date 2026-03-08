#!/usr/bin/env python3
"""
Create a subset folder with 10 patients (5 non_small_cell, 5 small_cell) who have both CT and PT.
Structure: non_small_cell/<patient_id>/ct.dcm, pt.dcm  and  small_cell/<patient_id>/ct.dcm, pt.dcm
1 CT + 1 PET per patient = 20 images. Uses symlinks. Output: Lung-PET-CT-Dx-10patients/

Usage:
  python scripts/build_lung_10patient_subset.py [--output-dir Lung-PET-CT-Dx-10patients]
"""

import csv
import sys
from pathlib import Path

def main():
    root = Path("Lung-PET-CT-Dx ").resolve()
    if not root.exists():
        root = Path("Lung-PET-CT-Dx").resolve()
    csv_path = root / "pipeline_metadata.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.", file=sys.stderr)
        sys.exit(1)

    out_root = Path(__file__).resolve().parent.parent / "Lung-PET-CT-Dx-10patients"
    if "--output-dir" in sys.argv:
        i = sys.argv.index("--output-dir")
        if i + 1 < len(sys.argv):
            out_root = Path(sys.argv[i + 1]).resolve()

    from collections import defaultdict
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys()) if rows else []

    by_pat = defaultdict(list)
    for r in rows:
        by_pat[r["patient_id"]].append(r)

    both = [pid for pid, recs in by_pat.items() if len(set(x["modality"] for x in recs)) == 2]
    non_small = [pid for pid in both if by_pat[pid][0]["label"] == "non_small_cell"]
    small = [pid for pid in both if by_pat[pid][0]["label"] == "small_cell"]

    selected = non_small[:5] + small[:5]
    if len(selected) < 10:
        print(f"Warning: only {len(selected)} patients with both CT+PT (need 5 non_small_cell + 5 small_cell).", file=sys.stderr)

    # 1 CT + 1 PT per patient; build patient -> (ct_row, pt_row)
    patient_data = []
    for pid in selected:
        recs = by_pat[pid]
        ct_rows = [r for r in recs if r["modality"] == "CT"]
        pt_rows = [r for r in recs if r["modality"] == "PT"]
        if ct_rows and pt_rows:
            patient_data.append((pid, by_pat[pid][0]["label"], ct_rows[0], pt_rows[0]))

    import shutil
    if out_root.exists():
        shutil.rmtree(out_root)

    # Structure: non_small_cell/<patient_id>/ct.dcm, pt.dcm  and  small_cell/<patient_id>/ct.dcm, pt.dcm
    subset_rows = []
    for pid, label, ct_row, pt_row in patient_data:
        pat_dir = out_root / label / pid
        pat_dir.mkdir(parents=True, exist_ok=True)
        ct_src = root / ct_row["image_path"]
        pt_src = root / pt_row["image_path"]
        ct_dst = pat_dir / "ct.dcm"
        pt_dst = pat_dir / "pt.dcm"
        if ct_src.exists():
            try:
                ct_dst.symlink_to(ct_src.resolve())
                subset_rows.append({
                    "image_path": f"{label}/{pid}/ct.dcm",
                    "patient_id": pid,
                    "label": label,
                    "modality": "CT",
                    "split": ct_row.get("split", "train"),
                })
            except OSError as e:
                print(f"Skip CT {pid}: {e}", file=sys.stderr)
        if pt_src.exists():
            try:
                pt_dst.symlink_to(pt_src.resolve())
                subset_rows.append({
                    "image_path": f"{label}/{pid}/pt.dcm",
                    "patient_id": pid,
                    "label": label,
                    "modality": "PT",
                    "split": pt_row.get("split", "train"),
                })
            except OSError as e:
                print(f"Skip PT {pid}: {e}", file=sys.stderr)

    out_csv = out_root / "pipeline_metadata.csv"
    if subset_rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["image_path", "patient_id", "label", "modality", "split"])
            w.writeheader()
            w.writerows(subset_rows)

    n_ct = sum(1 for r in subset_rows if r["modality"] == "CT")
    n_pt = sum(1 for r in subset_rows if r["modality"] == "PT")
    print(f"Created {out_root}")
    print(f"  Patients: {len(selected)} (5 non_small_cell, 5 small_cell)")
    print(f"  Rows: {len(subset_rows)} (CT: {n_ct}, PT: {n_pt})")
    print(f"  CSV: {out_csv}")

if __name__ == "__main__":
    main()
