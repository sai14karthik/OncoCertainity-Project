#!/usr/bin/env python3
"""
Full comprehensive stats for data2/TCGA-KIRP pipeline_metadata.csv.

Shows:
- Overall stats (total images, patients, modalities, classes)
- Split breakdown (train/val/test)
- Modality combinations and patient counts
- Images per modality, class, and modality×class
- Patient-level stats (images per patient per modality)
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from itertools import combinations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=str, nargs="?", default="data2", help="Root dir with pipeline_metadata.csv")
    args = parser.parse_args()

    root = Path(args.data_root).resolve()
    csv_path = root / "pipeline_metadata.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.", file=sys.stderr)
        sys.exit(1)

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Basic stats
    n = len(rows)
    unique_image_paths = set(row["image_path"] for row in rows if row.get("image_path"))
    n_unique = len(unique_image_paths)
    patients = set(row["patient_id"] for row in rows if row.get("patient_id"))
    modalities = sorted(set(row["modality"] for row in rows))
    labels = sorted(set(row["label"] for row in rows))
    splits = Counter(row["split"] for row in rows)
    
    # Group by patient and modality
    patients_by_modality = defaultdict(set)
    images_by_patient_modality = defaultdict(lambda: defaultdict(int))
    images_by_patient = defaultdict(int)
    by_mod = Counter(row["modality"] for row in rows)
    by_label = Counter(row["label"] for row in rows)
    by_mod_label = defaultdict(lambda: defaultdict(int))
    # Count unique image paths per modality×class
    by_mod_label_unique = defaultdict(lambda: defaultdict(set))
    patients_by_split = defaultdict(set)
    
    for row in rows:
        patient_id = row.get("patient_id")
        modality = row.get("modality")
        label = row.get("label")
        split = row.get("split")
        image_path = row.get("image_path", "")
        
        if patient_id and modality:
            patients_by_modality[modality].add(patient_id)
            images_by_patient_modality[patient_id][modality] += 1
            images_by_patient[patient_id] += 1
        
        if split and patient_id:
            patients_by_split[split].add(patient_id)
        
        if modality and label:
            by_mod_label[modality][label] += 1
            if image_path:
                by_mod_label_unique[modality][label].add(image_path)

    print("=" * 80)
    print("FULL STATS FOR data2/TCGA-KIRP")
    print("=" * 80)
    print()
    
    # 1. Overall stats
    print("1. OVERALL DATASET STATS")
    print("-" * 80)
    print(f"  Data root:              {root}")
    print(f"  Total image rows:       {n:,} (includes duplicates)")
    print(f"  Unique image paths:     {n_unique:,}")
    print(f"  Duplicate entries:      {n - n_unique:,}")
    print(f"  Unique patients:        {len(patients):,}")
    print(f"  Modalities:             {modalities}  ({len(modalities)})")
    print(f"  Classes:                {labels}  ({len(labels)})")
    print()
    
    # 2. Split breakdown
    print("2. SPLIT BREAKDOWN")
    print("-" * 80)
    print("  Split (image rows):")
    for s in ["train", "val", "test"]:
        count = splits.get(s, 0)
        pct = (count / n * 100) if n > 0 else 0
        print(f"    {s:6s}: {count:>7,} ({pct:>5.1f}%)")
    print("  Split (patients):")
    for s in ["train", "val", "test"]:
        count = len(patients_by_split.get(s, set()))
        pct = (count / len(patients) * 100) if patients else 0
        print(f"    {s:6s}: {count:>7,} ({pct:>5.1f}%)")
    print()
    
    # 3. Images per modality
    print("3. IMAGES PER MODALITY")
    print("-" * 80)
    for m in modalities:
        count = by_mod[m]
        pct = (count / n * 100) if n > 0 else 0
        print(f"  {m:3s}: {count:>7,} ({pct:>5.1f}%)")
    print()
    
    # 4. Images per class
    print("4. IMAGES PER CLASS")
    print("-" * 80)
    for c in labels:
        count = by_label[c]
        pct = (count / n * 100) if n > 0 else 0
        print(f"  {c:20s}: {count:>7,} ({pct:>5.1f}%)")
    print()
    
    # 5. Images per modality × class
    print("5. IMAGES PER MODALITY × CLASS")
    print("-" * 80)
    print("  (Showing UNIQUE image paths - matches historical counts)")
    for m in modalities:
        for c in labels:
            count_total = by_mod_label[m].get(c, 0)
            count_unique = len(by_mod_label_unique[m].get(c, set()))
            if count_unique > 0:
                pct = (count_unique / len(by_mod_label_unique[m].get(c, set())) * 100) if by_mod_label_unique[m].get(c, set()) else 0
                dup_info = f" ({count_total - count_unique:,} duplicates)" if count_total > count_unique else ""
                print(f"  {m:3s} / {c:20s}: {count_unique:>7,} unique{dup_info}")
    print()
    
    # 6. Patients per modality
    print("6. PATIENTS PER MODALITY")
    print("-" * 80)
    for m in modalities:
        count = len(patients_by_modality[m])
        pct = (count / len(patients) * 100) if patients else 0
        print(f"  {m:3s}: {count:>3,} patients ({pct:>5.1f}%)")
    print()
    
    # 7. Patients per modality combination
    print("7. PATIENTS PER MODALITY COMBINATION")
    print("-" * 80)
    for r in range(1, len(modalities) + 1):
        for mod_combo in combinations(modalities, r):
            mod_combo = list(mod_combo)
            if mod_combo:
                common_patients = patients_by_modality[mod_combo[0]]
                for mod in mod_combo[1:]:
                    common_patients = common_patients & patients_by_modality[mod]
                combo_str = "+".join(mod_combo)
                count = len(common_patients)
                if count > 0:
                    pct = (count / len(patients) * 100) if patients else 0
                    print(f"  {combo_str:<15s}: {count:>3,} patients ({pct:>5.1f}%)")
    print()
    
    # 8. Images per patient (summary stats)
    print("8. IMAGES PER PATIENT (SUMMARY)")
    print("-" * 80)
    if images_by_patient:
        image_counts = list(images_by_patient.values())
        print(f"  Total patients:         {len(image_counts):,}")
        print(f"  Min images/patient:     {min(image_counts):,}")
        print(f"  Max images/patient:     {max(image_counts):,}")
        print(f"  Mean images/patient:    {sum(image_counts) / len(image_counts):.1f}")
        print(f"  Median images/patient:   {sorted(image_counts)[len(image_counts) // 2]:,}")
    print()
    
    # 9. Images per patient per modality (summary)
    print("9. IMAGES PER PATIENT PER MODALITY (SUMMARY)")
    print("-" * 80)
    for m in modalities:
        counts = [images_by_patient_modality[pid][m] for pid in images_by_patient_modality if images_by_patient_modality[pid][m] > 0]
        if counts:
            print(f"  {m:3s}:")
            print(f"    Patients with {m}:     {len(counts):,}")
            print(f"    Min images/patient:   {min(counts):,}")
            print(f"    Max images/patient:   {max(counts):,}")
            print(f"    Mean images/patient:  {sum(counts) / len(counts):.1f}")
            print(f"    Median images/patient: {sorted(counts)[len(counts) // 2]:,}")
    print()
    
    # 10. Top patients by total images
    print("10. TOP 10 PATIENTS BY TOTAL IMAGES")
    print("-" * 80)
    sorted_patients = sorted(images_by_patient.items(), key=lambda x: x[1], reverse=True)
    for i, (pid, count) in enumerate(sorted_patients[:10], 1):
        mod_counts = ", ".join(f"{m}:{images_by_patient_modality[pid][m]}" for m in modalities if images_by_patient_modality[pid][m] > 0)
        print(f"  {i:2d}. Patient {pid:20s}: {count:>4,} images ({mod_counts})")
    print()
    
    print("=" * 80)


if __name__ == "__main__":
    main()
