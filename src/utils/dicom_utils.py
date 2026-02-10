"""
DICOM file handling utilities.
Converts DICOM files to PIL Images for processing.
"""

import os
from typing import Optional
import numpy as np
from PIL import Image

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    pydicom = None


def load_dicom_image(dicom_path: str, window_center: Optional[float] = None, 
                     window_width: Optional[float] = None) -> Image.Image:
    """
    Load a DICOM file and convert it to a PIL Image.
    
    Args:
        dicom_path: Path to DICOM file
        window_center: Optional window center for windowing (if None, uses DICOM tags)
        window_width: Optional window width for windowing (if None, uses DICOM tags)
    
    Returns:
        PIL Image in RGB mode
    
    Raises:
        ImportError: If pydicom is not installed
        FileNotFoundError: If DICOM file doesn't exist
        ValueError: If DICOM file cannot be read
    """
    if not PYDICOM_AVAILABLE:
        raise ImportError(
            "pydicom is required for DICOM support. Install it with: pip install pydicom"
        )
    
    if not os.path.exists(dicom_path):
        raise FileNotFoundError(f"DICOM file not found: {dicom_path}")
    
    # Read DICOM file
    try:
        ds = pydicom.dcmread(dicom_path, force=True)
    except Exception as e:
        raise ValueError(f"Failed to read DICOM file {dicom_path}: {e}")
    
    # Get pixel array (keep original dtype for VOI LUT)
    try:
        pixel_array = np.asarray(ds.pixel_array)
    except Exception as e:
        raise ValueError(f"Failed to extract pixel array from {dicom_path}: {e}")
    
    # Multi-frame: use first frame (shape N, H, W); avoid treating 3-channel as frames
    if pixel_array.ndim == 3 and pixel_array.shape[0] != 3:
        pixel_array = np.asarray(pixel_array[0])
    
    # Apply VOI LUT (Window/Level) if available - applies one of VOI LUT Sequence or Window Center/Width
    voi_applied = False
    try:
        pixel_array = apply_voi_lut(pixel_array, ds)
        voi_applied = True
    except Exception:
        pass
    
    # Apply windowing only when VOI LUT was not applied (avoid double windowing)
    if not voi_applied:
        if window_center is not None and window_width is not None:
            window_min = window_center - window_width / 2
            window_max = window_center + window_width / 2
            pixel_array = np.clip(pixel_array, window_min, window_max)
        elif hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
            try:
                wc = float(ds.WindowCenter) if not isinstance(ds.WindowCenter, (list, tuple)) else float(ds.WindowCenter[0])
                ww = float(ds.WindowWidth) if not isinstance(ds.WindowWidth, (list, tuple)) else float(ds.WindowWidth[0])
                window_min = wc - ww / 2
                window_max = wc + ww / 2
                pixel_array = np.clip(pixel_array, window_min, window_max)
            except (ValueError, TypeError):
                pass
    
    # Normalize to 0-255 range
    pixel_min = pixel_array.min()
    pixel_max = pixel_array.max()
    
    if pixel_max > pixel_min:
        pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype(np.uint8)
    else:
        pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
    
    # MONOCHROME1: higher pixel value = darker; invert so display is correct
    if getattr(ds, 'PhotometricInterpretation', None) == 'MONOCHROME1':
        pixel_array = 255 - pixel_array
    
    # Convert to PIL Image
    if len(pixel_array.shape) == 2:
        # Grayscale image
        img = Image.fromarray(pixel_array, mode='L')
        # Convert to RGB (required by models)
        img = img.convert('RGB')
    elif len(pixel_array.shape) == 3:
        # Color image (uncommon in medical DICOM)
        if pixel_array.shape[2] == 3:
            img = Image.fromarray(pixel_array, mode='RGB')
        else:
            # Take first 3 channels
            img = Image.fromarray(pixel_array[:, :, :3], mode='RGB')
    else:
        raise ValueError(f"Unsupported pixel array shape: {pixel_array.shape}")
    
    return img


def is_dicom_file(file_path: str) -> bool:
    """
    Check if a file is a DICOM file based on extension.
    
    Args:
        file_path: Path to file
    
    Returns:
        True if file has .dcm or .dicom extension
    """
    ext = os.path.splitext(file_path)[1].lower()
    return ext in ['.dcm', '.dicom']


def load_image_smart(image_path: str) -> Image.Image:
    """
    Smart image loader that handles both regular images (PNG/JPG) and DICOM files.
    
    Args:
        image_path: Path to image file (PNG, JPG, or DICOM)
    
    Returns:
        PIL Image in RGB mode
    """
    if is_dicom_file(image_path):
        return load_dicom_image(image_path)
    else:
        # Regular image file
        return Image.open(image_path).convert('RGB')
