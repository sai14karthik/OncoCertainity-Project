#!/usr/bin/env python3
"""
Inspect an NBIA digest Excel file: sheets, columns, modalities, patient/series counts.
Usage: python scripts/inspect_digest.py <path-to-digest.xlsx>
Example: python scripts/inspect_digest.py ReMIND-Manifest-Sept-2023-nbia-digest.xlsx
"""
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_digest.py <path-to-digest.xlsx>")
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        sys.exit(1)

    try:
        import pandas as pd
    except ImportError:
        print("Need pandas: pip install pandas openpyxl")
        sys.exit(1)
    try:
        xl = pd.ExcelFile(path, engine="openpyxl")
    except Exception as e:
        print(f"Open Excel failed: {e}")
        print("Try: pip install openpyxl")
        sys.exit(1)

    print("=" * 60)
    print(f"Digest: {path}")
    print("=" * 60)
    print("Sheet names:", xl.sheet_names)

    for name in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=name, engine="openpyxl")
        print(f"\n--- Sheet: {name} ---")
        print("Shape:", df.shape)
        print("Columns:", list(df.columns))

        # Find modality-like column (Modality, Series Modality, etc.)
        mod_cols = [c for c in df.columns if "mod" in c.lower() or "series" in c.lower()]
        if mod_cols:
            for c in mod_cols:
                vc = df[c].value_counts()
                print(f"  {c} value_counts:\n{vc.head(15)}")
                print(f"  Unique: {df[c].nunique()}")

        # Patient/Subject ID column
        id_cols = [c for c in df.columns if "patient" in c.lower() or "subject" in c.lower() or "case" in c.lower()]
        if id_cols:
            for c in id_cols:
                print(f"  {c} unique count: {df[c].nunique()}")

        # Class/label column if any
        label_cols = [c for c in df.columns if "class" in c.lower() or "label" in c.lower() or "diagnosis" in c.lower() or "grade" in c.lower()]
        if label_cols:
            for c in label_cols:
                print(f"  {c} value_counts:\n{df[c].value_counts().head(10)}")

        # ========== LABEL CHECK: every column that could be a label ==========
        print("\n--- LABEL CHECK (all columns with non-null values) ---")
        label_keywords = ["class", "label", "diagnosis", "grade", "type", "outcome", "finding",
                         "description", "category", "group", "status", "result", "pathology"]
        for c in df.columns:
            non_null = df[c].notna().sum()
            if non_null == 0:
                print(f"  {c}: all null (no values)")
                continue
            n_unique = df[c].nunique()
            # Show value_counts for categorical-looking columns (few unique) or label-like names
            if n_unique <= 20 or any(kw in c.lower() for kw in label_keywords):
                vc = df[c].value_counts(dropna=False)
                print(f"  {c}: {non_null} non-null, {n_unique} unique")
                print(f"    value_counts: {vc.head(15).to_dict()}")
            else:
                print(f"  {c}: {non_null} non-null, {n_unique} unique (sample: {df[c].dropna().iloc[0]})")

        print("\nFirst 2 rows (first 8 cols):")
        print(df.iloc[:2, :8].to_string())

    print("\nDone.")

if __name__ == "__main__":
    main()
