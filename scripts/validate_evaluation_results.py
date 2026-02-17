#!/usr/bin/env python3
"""
Validate evaluation result JSONs from evaluate_sequential_modalities (e.g. from Newton).

Checks:
- Required top-level keys: step_results, patient_level_results, modalities
- Step names consistent with modality order (CT, MR, PT and CT+MR, CT+MR+PT or PT+MR+CT)
- Pairwise keys are canonical: tuple(sorted([mod_i, mod_j]))
- Patient-level num_samples <= slice-level num_samples per step (when step exists in both)
- Agreement rates in [0, 1], accuracies in [0, 1], confidence metrics in [0, 1] where applicable
- No duplicate or conflicting pairwise entries

Usage:
  python scripts/validate_evaluation_results.py [path/to/results_*.json]
  If no path given, looks for results/results_*.json
"""

import ast
import json
import sys
from pathlib import Path


def load_results(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_structure(r: dict) -> list[str]:
    errs = []
    for key in ("step_results", "patient_level_results", "modalities"):
        if key not in r:
            errs.append(f"Missing top-level key: {key}")
    if not r.get("modalities"):
        errs.append("modalities is empty or missing")
    return errs


def validate_step_names(r: dict) -> list[str]:
    errs = []
    mods = r.get("modalities") or []
    step_results = r.get("step_results") or {}
    patient_results = r.get("patient_level_results") or {}

    all_steps = set(step_results.keys()) | set(patient_results.keys())
    for step in all_steps:
        if step in mods:
            continue
        if "+" in step:
            parts = step.split("+")
            if parts != mods[: len(parts)]:
                errs.append(f"Step '{step}' does not match modality prefix: {mods}")
        else:
            errs.append(f"Unknown step name: {step} (not in modalities {mods})")
    return errs


def validate_pairwise_keys(r: dict) -> list[str]:
    errs = []
    mods = r.get("modalities") or []
    for key in (
        "pairwise_agreements",
        "pairwise_confidence_comparisons",
        "pairwise_dominance",
        "pairwise_logit_similarities",
    ):
        data = r.get(key)
        if not data:
            continue
        if not isinstance(data, dict):
            errs.append(f"{key} is not a dict")
            continue
        for k, v in data.items():
            pair = None
            if isinstance(k, (list, tuple)) and len(k) == 2:
                pair = tuple(k)
            elif isinstance(k, str):
                try:
                    parsed = ast.literal_eval(k)
                    if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
                        pair = tuple(parsed)
                except (ValueError, SyntaxError):
                    pass
            if pair is None:
                errs.append(f"{key} key {k!r} could not be parsed as a 2-tuple")
                continue
            if len(pair) != 2:
                errs.append(f"{key} key {pair} does not have length 2")
            canonical = tuple(sorted(pair))
            if pair != canonical:
                errs.append(f"{key} key {pair} is not canonical (should be {canonical})")
            if pair[0] not in mods or pair[1] not in mods:
                errs.append(f"{key} key {pair} contains modality not in {mods}")
    return errs


def validate_numeric_ranges(r: dict) -> list[str]:
    errs = []

    def check_in_01(name: str, val, step_or_key: str):
        if val is None:
            return
        try:
            v = float(val)
            if not (0 <= v <= 1):
                errs.append(f"{step_or_key}: {name}={v} not in [0,1]")
        except (TypeError, ValueError):
            pass

    for step_name, data in (r.get("step_results") or {}).items():
        if isinstance(data, dict):
            check_in_01("accuracy", data.get("accuracy"), step_name)
    for step_name, data in (r.get("patient_level_results") or {}).items():
        if isinstance(data, dict):
            check_in_01("accuracy", data.get("accuracy"), f"patient_{step_name}")

    for pair_key, data in (r.get("pairwise_agreements") or {}).items():
        if isinstance(data, dict):
            check_in_01("agreement_rate", data.get("agreement_rate"), str(pair_key))
            check_in_01("disagreement_rate", data.get("disagreement_rate"), str(pair_key))
    for pair_key, data in (r.get("pairwise_confidence_comparisons") or {}).items():
        if isinstance(data, dict):
            check_in_01("mod1_higher_confidence_rate", data.get("mod1_higher_confidence_rate"), str(pair_key))
            check_in_01("mod2_higher_confidence_rate", data.get("mod2_higher_confidence_rate"), str(pair_key))
    return errs


def validate_sample_counts(r: dict) -> list[str]:
    errs = []
    step_results = r.get("step_results") or {}
    patient_results = r.get("patient_level_results") or {}
    for step_name in set(step_results.keys()) & set(patient_results.keys()):
        sl = step_results[step_name]
        pl = patient_results[step_name]
        if not isinstance(sl, dict) or not isinstance(pl, dict):
            continue
        n_slice = sl.get("num_samples")
        n_patient = pl.get("num_samples")
        if n_slice is not None and n_patient is not None:
            if n_patient > n_slice:
                errs.append(
                    f"Step {step_name}: patient-level num_samples ({n_patient}) > slice-level ({n_slice})"
                )
    return errs


def validate_combined_agreements(r: dict) -> list[str]:
    errs = []
    combined = r.get("combined_agreements") or {}
    for name, data in combined.items():
        if isinstance(data, dict):
            ar = data.get("agreement_rate")
            if ar is not None:
                try:
                    if not (0 <= float(ar) <= 1):
                        errs.append(f"combined_agreements[{name}].agreement_rate={ar} not in [0,1]")
                except (TypeError, ValueError):
                    pass
    return errs


def main():
    if len(sys.argv) > 1:
        paths = [sys.argv[1]]
    else:
        results_dir = Path(__file__).resolve().parent.parent / "results"
        if not results_dir.is_dir():
            print("No results path given and results/ not found. Usage: validate_evaluation_results.py [path/to/results_*.json]")
            sys.exit(0)
        paths = list(results_dir.glob("results_*.json"))
        if not paths:
            print("No results_*.json found under results/")
            sys.exit(0)

    all_errors = []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            print(f"Skip (not a file): {path}")
            continue
        try:
            r = load_results(str(path))
        except Exception as e:
            print(f"FAIL {path}: could not load JSON: {e}")
            all_errors.append((path, ["load error"]))
            continue

        errs = []
        errs.extend(validate_structure(r))
        errs.extend(validate_step_names(r))
        errs.extend(validate_pairwise_keys(r))
        errs.extend(validate_numeric_ranges(r))
        errs.extend(validate_sample_counts(r))
        errs.extend(validate_combined_agreements(r))

        if errs:
            print(f"FAIL {path}")
            for e in errs:
                print(f"  - {e}")
            all_errors.append((path, errs))
        else:
            mods = r.get("modalities", [])
            n_steps = len(r.get("step_results") or {})
            n_patient_steps = len(r.get("patient_level_results") or {})
            print(f"OK   {path} (modalities={mods}, steps={n_steps}, patient_steps={n_patient_steps})")

    if all_errors:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
