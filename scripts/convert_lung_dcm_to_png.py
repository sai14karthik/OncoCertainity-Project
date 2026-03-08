#!/usr/bin/env python3
"""
Convert Lung-PET-CT-Dx DICOM to PNG. Output: same CT/PT class structure under --output-dir.

Default conversion: min-max scale to 0-255, handle MONOCHROME1. No crop or resize.

Usage:
  From organised root (CT/PT with .dcm or valid symlinks):
    python scripts/convert_lung_dcm_to_png.py Lung-PET-CT-Dx [--output-dir Lung-PET-CT-Dx-PNG] [--update-csv]
  From raw DICOM folder when symlinks are broken:
    python scripts/convert_lung_dcm_to_png.py Lung-PET-CT-Dx --dicom-root /path/to/folder/with/Lung_Dx-A0001 [--output-dir Lung-PET-CT-Dx-PNG]

  --style notebook: optional crop + resize to 256 (like reference notebook).
"""

import argparse
import csv
import os
import random
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Add project root for src.utils
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    pydicom = None

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    import png
except ImportError:
    png = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

TWO_CLASS_MAP = {"A": "non_small_cell", "G": "non_small_cell", "E": "non_small_cell", "B": "small_cell"}


def get_label(subject_id: str) -> str:
    m = re.match(r"Lung_Dx-([ABEG])\d+", subject_id)
    if not m:
        return "unknown"
    return TWO_CLASS_MAP.get(m.group(1), "unknown")


def classify_modality(study: str, series: str) -> Optional[str]:
    s_upper = series.upper()
    if "PET" in s_upper and ("CORRECTED" in s_upper or " WB " in s_upper or s_upper.strip().startswith("PET")):
        return "PT"
    if "PET " in s_upper and "CT" not in s_upper:
        return "PT"
    if "CT " in s_upper or "CT WB" in s_upper or "-CT-" in s_upper:
        return "CT"
    if any(x in s_upper for x in ("CHEST", "THORAX", "LUNG")):
        return "CT"
    return None


def read_dicom_notebook(dcm_path: str) -> np.ndarray:
    """Notebook style: load, min-max normalize, MONOCHROME1 flip. Returns float [0,1]."""
    if not PYDICOM_AVAILABLE:
        raise ImportError("pydicom is required")
    r = pydicom.dcmread(dcm_path, force=True)
    pix = np.asarray(r.pixel_array, dtype=np.float64)
    if pix.ndim == 3 and pix.shape[0] != 3:
        pix = pix[0]
    if pix.ndim != 2:
        pix = pix.squeeze()
    if pix.ndim != 2:
        raise ValueError(f"Unsupported shape {pix.shape}")
    mn, mx = pix.min(), pix.max()
    if mx <= mn:
        return np.zeros(pix.shape)
    pix = (pix - mn) / (mx - mn)
    if getattr(r, "PhotometricInterpretation", None) == "MONOCHROME1":
        pix = 1.0 - pix
    return pix


def crop_dicom_notebook(pixel_float: np.ndarray) -> np.ndarray:
    """Crop to content using connectedComponentsWithStats (notebook style)."""
    if not CV2_AVAILABLE:
        return pixel_float
    mask = (pixel_float > 0.005).astype(np.uint8)
    if mask.sum() == 0:
        return pixel_float
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    if num_labels < 2:
        return pixel_float
    second_row = stats[1:, 4].argmax() + 1
    x1, y1, w, h = stats[second_row][:4]
    x2, y2 = x1 + w, y1 + h
    if w < 2 or h < 2:
        return pixel_float
    return pixel_float[y1:y2, x1:x2]


def dcm_to_png_array(dcm_path: str) -> np.ndarray:
    """Load DICOM and return 2D uint8 array (0-255). Correct conversion: min-max scale, handle MONOCHROME1."""
    if not PYDICOM_AVAILABLE:
        raise ImportError("pydicom is required")
    ds = pydicom.dcmread(dcm_path, force=True)
    arr = np.asarray(ds.pixel_array, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[0] != 3:
        arr = arr[0]
    if arr.ndim != 2:
        arr = arr.squeeze()
    if arr.ndim != 2:
        raise ValueError(f"Unsupported shape {arr.shape}")
    mn, mx = arr.min(), arr.max()
    if mx <= mn:
        out = np.zeros(arr.shape, dtype=np.uint8)
    else:
        out = ((arr - mn) / (mx - mn) * 255.0).astype(np.uint8)
    if getattr(ds, "PhotometricInterpretation", None) == "MONOCHROME1":
        out = 255 - out
    return out


def convert_one(dcm_path: Path, png_path: Path, use_pil: bool = True, use_project_loader: bool = False,
                style: str = "simple", size: int = 256) -> bool:
    """Convert one DICOM to PNG. style: 'simple' (max-scale) or 'notebook' (norm+crop+resize)."""
    try:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        if style == "notebook" and PYDICOM_AVAILABLE:
            pix = read_dicom_notebook(str(dcm_path))
            pix = crop_dicom_notebook(pix)
            if CV2_AVAILABLE:
                pix = cv2.resize(pix, (size, size), interpolation=cv2.INTER_LINEAR)
            else:
                from PIL import Image
                img = Image.fromarray((pix * 255).astype(np.uint8), mode="L")
                img = img.resize((size, size), Image.Resampling.LANCZOS)
                pix = np.asarray(img) / 255.0
            arr = (pix * 255).astype(np.uint8)
            if use_pil and PIL_AVAILABLE:
                Image.fromarray(arr, mode="L").save(str(png_path), "PNG")
            else:
                Image.fromarray(arr, mode="L").save(str(png_path), "PNG")
            return True
        if use_project_loader:
            from src.utils.dicom_utils import load_dicom_image
            img = load_dicom_image(str(dcm_path))
            if img.mode != "L":
                img = img.convert("L")
            img.save(str(png_path), "PNG")
            return True
        arr = dcm_to_png_array(str(dcm_path))
        if use_pil and PIL_AVAILABLE:
            Image.fromarray(arr, mode="L").save(str(png_path), "PNG")
        elif png is not None:
            with open(png_path, "wb") as f:
                w = png.Writer(arr.shape[1], arr.shape[0], greyscale=True)
                w.write(f, arr.tolist())
        else:
            Image.fromarray(arr, mode="L").save(str(png_path), "PNG")
        return True
    except Exception as e:
        print(f"FAIL {dcm_path} -> {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lung_root", type=str, help="Lung-PET-CT-Dx root (for metadata/output paths)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output root (default: <lung_root>_PNG)")
    parser.add_argument("--dicom-root", type=str, default=None,
                        help="Path on your machine to folder containing Lung_Dx-A0001, Lung_Dx-A0002, ... (use when CT/PT symlinks are broken)")
    parser.add_argument("--style", choices=["simple", "notebook"], default="simple",
                        help="Conversion: simple=min-max scale + MONOCHROME1 (default); notebook=crop+resize 256")
    parser.add_argument("--size", type=int, default=256, help="Resize size for --style notebook (default: 256)")
    parser.add_argument("--update-csv", action="store_true", help="Write pipeline_metadata.csv with .png paths under output-dir")
    parser.add_argument("--max", type=int, default=None, help="Max number of DICOMs to convert (for testing)")
    parser.add_argument("--use-project-loader", action="store_true", help="Use project load_dicom_image (simple mode only)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip DICOMs that already have a PNG (resume partial run)")
    args = parser.parse_args()

    root = Path(args.lung_root).resolve()
    out_root = Path(args.output_dir).resolve() if args.output_dir else root.parent / (root.name.strip() + "_PNG")

    if not root.is_dir() and not args.dicom_root:
        print(f"Error: {root} is not a directory.", file=sys.stderr)
        sys.exit(1)

    dcm_files: List[Tuple[Path, str]] = []  # (path, "CT/non_small_cell" etc.)
    pipeline_rows: List[dict] = []

    if args.dicom_root:
        # Discover from raw DICOM folder (patient/study/series/*.dcm)
        dicom_root = Path(args.dicom_root).resolve()
        if not dicom_root.is_dir():
            print(
                f"Error: --dicom-root must be a real path on your machine to the folder that contains\n"
                f"  patient subfolders (Lung_Dx-A0001, Lung_Dx-A0002, ...).\n"
                f"  You passed: {dicom_root}\n"
                f"  If you have not downloaded/restored the DICOM data, do that first (e.g. from NBIA),\n"
                f"  then pass the path to that folder (e.g. manifest-1608669183333/Lung-PET-CT-Dx).",
                file=sys.stderr,
            )
            sys.exit(1)
        for patient_dir in sorted(dicom_root.iterdir()):
            if not patient_dir.is_dir() or not re.match(r"Lung_Dx-[ABEG]\d+", patient_dir.name):
                continue
            label = get_label(patient_dir.name)
            if label == "unknown":
                continue
            for study_dir in patient_dir.iterdir():
                if not study_dir.is_dir():
                    continue
                for series_dir in study_dir.iterdir():
                    if not series_dir.is_dir():
                        continue
                    mod = classify_modality(study_dir.name, series_dir.name)
                    if mod is None:
                        continue
                    sub = f"{mod}/{label}"
                    for dcm in series_dir.glob("*.dcm"):
                        unique_name = f"{patient_dir.name}_{series_dir.name}_{dcm.stem}.png".replace(" ", "_")
                        for c in '/\\:*?"<>|':
                            unique_name = unique_name.replace(c, "_")
                        dcm_files.append((dcm, sub))
                        pipeline_rows.append({
                            "image_path": f"{sub}/{unique_name}",
                            "patient_id": patient_dir.name,
                            "label": label,
                            "modality": mod,
                            "split": None,
                        })
        if not dcm_files:
            print(f"No DICOM files found under {dicom_root} (expected Lung_Dx-A0001/.../series/*.dcm)", file=sys.stderr)
            sys.exit(1)
        # Assign splits by patient (80/10/10)
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
    else:
        # From organised root (CT/PT class folders). Include .dcm files and symlinks whose target exists.
        subdirs = ["CT/non_small_cell", "CT/small_cell", "PT/non_small_cell", "PT/small_cell"]
        for sub in subdirs:
            src = root / sub
            if not src.is_dir():
                continue
            for f in src.iterdir():
                if f.suffix.lower() != ".dcm":
                    continue
                # Include real files or symlinks that resolve to a file
                if f.is_file():
                    dcm_files.append((f, sub))
                elif f.is_symlink() and f.exists():
                    dcm_files.append((f, sub))
        if not dcm_files:
            print(
                f"No .dcm files found under {root}.\n"
                f"  The CT/PT folders contain symlinks that point to missing files.\n"
                f"  Restore the DICOM folder (patient folders Lung_Dx-A0001, ... in\n"
                f"  manifest-1608669183333/Lung-PET-CT-Dx), then run again without --dicom-root.",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.max is not None and args.max > 0:
        dcm_files = dcm_files[: args.max]
        if pipeline_rows:
            pipeline_rows = pipeline_rows[: args.max]
        print(f"Limiting to first {len(dcm_files)} files (--max {args.max})")

    try:
        from tqdm import tqdm
        iterator = tqdm(dcm_files, desc="DICOM->PNG", unit="img")
    except ImportError:
        iterator = dcm_files

    print(f"Found {len(dcm_files)} DICOM files. Output: {out_root} (style={args.style})")
    ok = 0
    skipped = 0
    for i, (dcm_path, sub) in enumerate(iterator):
        if pipeline_rows and i < len(pipeline_rows):
            png_name = pipeline_rows[i]["image_path"].split("/")[-1]
        else:
            png_name = dcm_path.stem + ".png"
        png_path = out_root / sub / png_name
        if args.skip_existing and png_path.exists():
            skipped += 1
            continue
        if convert_one(dcm_path, png_path, use_project_loader=args.use_project_loader,
                       style=args.style, size=args.size):
            ok += 1
        if (i + 1) % 10000 == 0 and not hasattr(iterator, "set_postfix"):
            print(f"  {i + 1}/{len(dcm_files)} ... {ok} OK", flush=True)

    if skipped:
        print(f"Skipped (already exist): {skipped}", flush=True)
    print(f"Converted: {ok}/{len(dcm_files)}", flush=True)

    if args.update_csv or pipeline_rows:
        out_csv = out_root / "pipeline_metadata.csv"
        if pipeline_rows:
            out_root.mkdir(parents=True, exist_ok=True)
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["image_path", "patient_id", "label", "modality", "split"])
                w.writeheader()
                w.writerows(pipeline_rows)
            print(f"Wrote {out_csv} ({len(pipeline_rows)} rows, .png paths)")
        elif args.update_csv and ok > 0:
            orig_csv = root / "pipeline_metadata.csv"
            if orig_csv.exists():
                out_root.mkdir(parents=True, exist_ok=True)
                rows_out = []
                with open(orig_csv, newline="", encoding="utf-8") as f:
                    r = csv.DictReader(f)
                    fieldnames = r.fieldnames
                    for row in r:
                        path = row.get("image_path", "")
                        if path.endswith(".dcm"):
                            row["image_path"] = path[:-4] + ".png"
                        rows_out.append(row)
                with open(out_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    w.writerows(rows_out)
                print(f"Wrote {out_csv} ({len(rows_out)} rows, .png paths)")
            else:
                print("Original pipeline_metadata.csv not found; skipping --update-csv.", file=sys.stderr)


if __name__ == "__main__":
    main()
