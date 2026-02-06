#!/usr/bin/env python3
"""
Fetch TCGA-KIRP clinical labels from GDC API and write a clinical CSV for use with
build_tcga_kirp_metadata.py --clinical-csv.

Uses ajcc_pathologic_stage and ajcc_pathologic_t (primary diagnosis from GDC).
Labels reflect stage, not histologic grade:
  early_stage   = T1a (tumor ≤7 cm) or Stage I with no T
  advanced_stage = T1b, T2, T3, T4 or Stage II/III/IV (larger or advanced)
  unknown = missing (excluded from CSV so build script keeps placeholder for those)

Usage:
  python scripts/fetch_tcga_kirp_clinical.py [--manifest-csv PATH] [--output PATH]

  --manifest-csv  Optional: only output rows for Subject IDs in this metadata.csv (imaging cohort).
  --output        Output CSV path (default: data/clinical_tcga_kirp.csv).
"""

import argparse
import csv
import json
import ssl
import sys
import urllib.parse
import urllib.request
from pathlib import Path


GDC_CASES_URL = (
    "https://api.gdc.cancer.gov/cases"
    "?filters={}"
    "&expand=diagnoses"
    "&size={}"
    "&from={}"
    "&format=json"
)
FILTER_TCGA_KIRP = '{"op":"=","content":{"field":"project.project_id","value":"TCGA-KIRP"}}'


def _get_primary_diagnosis(diagnoses):
    for d in diagnoses or []:
        if d.get("diagnosis_is_primary_disease") is True:
            return d
    return (diagnoses or [{}])[0] if diagnoses else None


def _stage_to_label(stage_str, t_str):
    """Map AJCC stage and T to early_stage / advanced_stage / unknown (by T/pathologic stage)."""
    stage = (stage_str or "").strip().lower()
    t = (t_str or "").strip().upper()
    # T stage takes precedence when available
    if t:
        if t == "T1A" or t == "T1a":
            return "early_stage"   # T1a = tumor ≤7 cm, confined to kidney
        if t.startswith("T1") or t.startswith("T2") or t.startswith("T3") or t.startswith("T4"):
            return "advanced_stage"  # T1b+ or T2–T4
    if stage:
        if "stage iii" in stage or "stage iv" in stage:
            return "advanced_stage"
        if "stage ii" in stage:
            return "advanced_stage"
        if "stage i" in stage:
            return "early_stage"  # Stage I with no T
    return "unknown"


def fetch_all_tcga_kirp_cases():
    size = 100
    from_ = 0
    all_hits = []
    while True:
        url = GDC_CASES_URL.format(
            urllib.parse.quote(FILTER_TCGA_KIRP),
            size,
            from_,
        )
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            ctx = ssl.create_default_context()
        except Exception:
            ctx = ssl._create_unverified_context()
        try:
            with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
                data = json.loads(resp.read().decode())
        except Exception:
            # Retry with unverified SSL (e.g. macOS Python without system certs)
            with urllib.request.urlopen(req, timeout=60, context=ssl._create_unverified_context()) as resp:
                data = json.loads(resp.read().decode())
        hits = data.get("data", {}).get("hits", [])
        if not hits:
            break
        all_hits.extend(hits)
        pag = data.get("data", {}).get("pagination", {})
        total = pag.get("total", 0)
        if from_ + len(hits) >= total:
            break
        from_ += len(hits)
    return all_hits


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest-csv", type=str, default=None, help="Only output Subject IDs present in this metadata.csv")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (default: data/clinical_tcga_kirp.csv)")
    args = parser.parse_args()

    out_path = Path(args.output).resolve() if args.output else (Path(__file__).resolve().parents[1] / "data" / "clinical_tcga_kirp.csv")
    manifest_ids = None
    if args.manifest_csv:
        manifest_path = Path(args.manifest_csv).resolve()
        if not manifest_path.exists():
            print(f"Error: {manifest_path} not found.", file=sys.stderr)
            sys.exit(1)
        with open(manifest_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            manifest_ids = set(row.get("Subject ID", "").strip() for row in reader if row.get("Subject ID", "").strip())

    print("Fetching TCGA-KIRP cases from GDC...", flush=True)
    try:
        hits = fetch_all_tcga_kirp_cases()
    except Exception as e:
        print(f"Error fetching GDC: {e}", file=sys.stderr)
        sys.exit(1)

    subject_to_label = {}
    for case in hits:
        sid = case.get("submitter_id", "").strip()
        if not sid:
            continue
        if manifest_ids is not None and sid not in manifest_ids:
            continue
        diag = _get_primary_diagnosis(case.get("diagnoses"))
        if not diag:
            subject_to_label[sid] = "unknown"
            continue
        stage = diag.get("ajcc_pathologic_stage") or diag.get("ajcc_clinical_stage")
        t = diag.get("ajcc_pathologic_t") or diag.get("ajcc_clinical_t")
        subject_to_label[sid] = _stage_to_label(stage, t)

    # Exclude "unknown" from CSV so build_tcga_kirp_metadata.py uses placeholder for those patients
    rows_written = [(sid, subject_to_label[sid]) for sid in sorted(subject_to_label.keys()) if subject_to_label[sid] != "unknown"]
    classes = sorted(set(label for _, label in rows_written))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Subject ID", "label"])
        writer.writeheader()
        for sid, label in rows_written:
            writer.writerow({"Subject ID": sid, "label": label})

    print(f"Wrote {len(rows_written)} rows to {out_path}")
    print(f"Classes: {classes}")
    unknown_count = sum(1 for label in subject_to_label.values() if label == "unknown")
    if unknown_count:
        print(f"Excluded {unknown_count} Subject ID(s) with unknown stage (build script will use placeholder).", file=sys.stderr)
    if manifest_ids:
        missing = manifest_ids - set(subject_to_label.keys())
        if missing:
            print(f"Warning: {len(missing)} manifest Subject IDs not found in GDC: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}", file=sys.stderr)


if __name__ == "__main__":
    main()
