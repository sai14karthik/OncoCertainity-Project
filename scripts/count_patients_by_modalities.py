#!/usr/bin/env python3
"""
Count patients in data2 that have specific modality combinations.

Usage:
  python scripts/count_patients_by_modalities.py [data_root]

  data_root  Root containing pipeline_metadata.csv (default: data2).
"""

import argparse
import csv
import sys
from collections import defaultdict
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

    # Group images by patient_id and modality
    patients_by_modality = defaultdict(set)
    for row in rows:
        patient_id = row.get("patient_id")
        modality = row.get("modality")
        if patient_id and modality:
            patients_by_modality[modality].add(patient_id)

    modalities = sorted(patients_by_modality.keys())
    print(f"=== Patient counts by modality combination (data2) ===\n")
    print(f"Available modalities: {modalities}\n")

    # Count patients for each modality combination
    for r in range(1, len(modalities) + 1):
        for mod_combo in combinations(modalities, r):
            mod_combo = list(mod_combo)
            # Find patients that have ALL modalities in this combination
            if mod_combo:
                common_patients = patients_by_modality[mod_combo[0]]
                for mod in mod_combo[1:]:
                    common_patients = common_patients & patients_by_modality[mod]
                combo_str = "+".join(mod_combo)
                print(f"  {combo_str:<15} {len(common_patients):>3} patients")
    
    print()


if __name__ == "__main__":
    main()
