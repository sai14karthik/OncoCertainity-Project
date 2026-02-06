#!/usr/bin/env python3
"""
Print full-dataset stats for TCGA-KIRP pipeline_metadata.csv.

Usage:
  python scripts/dataset_stats.py [data_root]

  data_root  Root containing pipeline_metadata.csv (default: data2).
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


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

    n = len(rows)
    patients = set(row["patient_id"] for row in rows)
    modalities = sorted(set(row["modality"] for row in rows))
    labels = sorted(set(row["label"] for row in rows))
    splits = Counter(row["split"] for row in rows)
    by_mod = Counter(row["modality"] for row in rows)
    by_label = Counter(row["label"] for row in rows)
    patients_by_split = defaultdict(set)
    for row in rows:
        patients_by_split[row["split"]].add(row["patient_id"])
    by_mod_label = defaultdict(lambda: defaultdict(int))
    for row in rows:
        by_mod_label[row["modality"]][row["label"]] += 1

    print("=== TCGA-KIRP full dataset stats ===")
    print(f"  data_root: {root}")
    print()
    print(f"  Total image rows:  {n:,}")
    print(f"  Unique patients:   {len(patients)}")
    print(f"  Modalities:        {modalities}  ({len(modalities)})")
    print(f"  Classes:           {labels}")
    print()
    print("  Split (image rows):")
    for s in ["train", "val", "test"]:
        print(f"    {s}: {splits.get(s, 0):,}")
    print("  Split (patients):")
    for s in ["train", "val", "test"]:
        print(f"    {s}: {len(patients_by_split.get(s, set()))}")
    print()
    print("  Images per modality:")
    for m in modalities:
        print(f"    {m}: {by_mod[m]:,}")
    print("  Images per class:")
    for c in labels:
        print(f"    {c}: {by_label[c]:,}")
    print()
    print("  Images per modality × class:")
    for m in modalities:
        for c in labels:
            k = by_mod_label[m].get(c, 0)
            print(f"    {m} / {c}: {k:,}")


if __name__ == "__main__":
    main()
