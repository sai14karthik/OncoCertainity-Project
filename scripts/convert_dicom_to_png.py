#!/usr/bin/env python3
"""
Convert DICOM files to PNG images for consistency or performance.

This script:
1. Reads DICOM files from any organized dataset
2. Converts them to PNG using load_dicom_image() (VOI LUT or windowing, 0–255, MONOCHROME1 inversion, first frame for multi-frame)
3. Saves PNG files in the same structure
4. Preserves patient IDs and class organization

Note: This conversion is OPTIONAL - the codebase handles DICOM files directly.
      Use this script only if you specifically need PNG format.

Usage:
    python3 convert_dicom_to_png.py --input_dir <dataset_path> [--output_dir OUTPUT_DIR]
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
from src.utils.dicom_utils import load_dicom_image, is_dicom_file

def convert_dicom_to_png(dicom_path: str, png_path: str):
    """Convert a single DICOM file to PNG."""
    try:
        img = load_dicom_image(dicom_path)
        img.save(png_path, 'PNG')
        return True
    except Exception as e:
        print(f"Error converting {dicom_path}: {e}")
        return False

def convert_dataset(input_dir: str, output_dir: str = None, keep_original: bool = True):
    """
    Convert all DICOM files in organized dataset to PNG.
    
    Args:
        input_dir: Directory with DICOM files
        output_dir: Where to save PNG files (default: same as input with _PNG suffix)
        keep_original: If True, keep DICOM files; if False, replace with PNG
    """
    input_path = Path(input_dir)
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = Path(str(input_path) + '_PNG')
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all DICOM files
    dcm_files = list(input_path.rglob('*.dcm'))
    print(f"Found {len(dcm_files)} DICOM files to convert")
    
    converted = 0
    failed = 0
    
    for dcm_file in tqdm(dcm_files, desc="Converting DICOM to PNG"):
        # Get relative path from input_dir
        rel_path = dcm_file.relative_to(input_path)
        
        # Change extension to .png
        png_rel_path = rel_path.with_suffix('.png')
        png_file = output_path / png_rel_path
        
        # Create parent directories
        png_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert
        if convert_dicom_to_png(str(dcm_file), str(png_file)):
            converted += 1
            if not keep_original:
                # Optionally remove DICOM file (not recommended)
                pass
        else:
            failed += 1
    
    print(f"\nConversion complete!")
    print(f"  Converted: {converted} files")
    print(f"  Failed: {failed} files")
    print(f"  Output: {output_path}")
    
    if keep_original:
        print(f"\nNote: Original DICOM files preserved in {input_path}")
        print(f"To use PNG files, update dataset config to point to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert DICOM files to PNG images")
    parser.add_argument(
        '--input_dir',
        type=str,
        default=None,
        help='Input directory with DICOM files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for PNG files (default: input_dir_PNG)'
    )
    parser.add_argument(
        '--replace',
        action='store_true',
        help='Replace DICOM files with PNG (not recommended - keeps both)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return
    
    convert_dataset(
        args.input_dir,
        args.output_dir,
        keep_original=not args.replace
    )

if __name__ == '__main__':
    main()
