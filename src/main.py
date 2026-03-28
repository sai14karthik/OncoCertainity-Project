"""
Sequential modality evaluation script.
Evaluates model performance with sequential modality evaluation:
    1. First modality evaluation (baseline) - e.g., CT or PET
    2. Second modality evaluation without context
    3. Second modality evaluation with first modality context from Step 1
    Supports any medical imaging modalities (CT, PET, MRI, etc.) in any order.
"""

# Set up stderr and stdout filtering FIRST, before any imports that might trigger warnings
import sys
import os
import warnings

# Suppress stderr and stdout for cadam32bit warning - must be before any imports
_original_stderr = sys.stderr
_original_stdout = sys.stdout

def should_filter_message(s):
    """Check if a message should be filtered (cadam32bit warning)."""
    if not s:
        return False
    s_lower = s.lower()
    return (
        'cadam32bit' in s_lower or
        'text_config_dict' in s or
        'cliptextconfig' in s_lower or
        ('overriden' in s_lower and 'text_config' in s_lower) or
        ('nonetype' in s_lower and 'cadam32bit' in s_lower) or
        ("'nonetype' object has no attribute 'cadam32bit" in s_lower) or
        ("nonetype' object has no attribute 'cadam32bit" in s_lower) or
        ("object has no attribute 'cadam32bit" in s_lower) or
        ("'nonetype' object has no attribute 'cadam32bit_grad_fp32" in s_lower) or
        ("nonetype' object has no attribute 'cadam32bit_grad_fp32" in s_lower) or
        ("object has no attribute 'cadam32bit_grad_fp32" in s_lower) or
        ("has no attribute 'cadam32bit" in s_lower) or
        ("nonetype' object has no attribute" in s_lower and 'cadam32bit' in s_lower) or
        (s.strip().startswith("'") and "nonetype" in s_lower and "cadam32bit" in s_lower) or
        ("'nonetype'" in s_lower and "cadam32bit" in s_lower) or
        ("nonetype'" in s_lower and "cadam32bit" in s_lower) or
        (s.strip() == "'NoneType' object has no attribute 'cadam32bit_grad_fp32'\n") or
        (s.strip() == "'NoneType' object has no attribute 'cadam32bit_grad_fp32'")
    )

class FilteredStderr:
    def __init__(self):
        self._original = _original_stderr
        self._buffer = ""  # Buffer for multi-line warnings
    
    def write(self, s):
        if s:
            # Buffer the string to handle multi-line warnings
            self._buffer += s
            
            # Check if buffer contains the warning
            combined = self._buffer.lower()
            if should_filter_message(s) or should_filter_message(combined):
                self._buffer = ""
                return
            
            # Clear buffer if we see a newline and it's not a warning
            if '\n' in s:
                self._buffer = ""
            
            # Write if not filtered
            self._original.write(s)
    
    def flush(self):
        self._original.flush()
    
    def __getattr__(self, name):
        # Forward any other attributes to original stderr
        return getattr(self._original, name)

class FilteredStdout:
    def __init__(self):
        self._original = _original_stdout
        self._buffer = ""  # Buffer for multi-line warnings
    
    def write(self, s):
        if s:
            # Buffer the string to handle multi-line warnings
            self._buffer += s
            
            # Check if buffer contains the warning
            combined = self._buffer.lower()
            if should_filter_message(s) or should_filter_message(combined):
                self._buffer = ""
                return
            
            # Clear buffer if we see a newline and it's not a warning
            if '\n' in s:
                self._buffer = ""
            
            # Write if not filtered
            self._original.write(s)
    
    def flush(self):
        self._original.flush()
    
    def __getattr__(self, name):
        # Forward any other attributes to original stdout
        return getattr(self._original, name)

sys.stderr = FilteredStderr()
sys.stdout = FilteredStdout()

# Set environment variables early to prevent warnings
os.environ.setdefault('BITSANDBYTES_NOWELCOME', '1')
os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

# Load .env file if it exists (for HF_TOKEN and other secrets)
try:
    from dotenv import load_dotenv
    # Load .env from project root
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass
except Exception as e:
    # Silently fail if .env loading has issues
    pass

import argparse
import random
import re
import traceback

from src.utils.tf_mock import ensure_tensorflow_stub
ensure_tensorflow_stub()

# Suppress additional warnings after imports
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*bitsandbytes.*')
warnings.filterwarnings('ignore', message='.*resume_download.*')
warnings.filterwarnings('ignore', message='.*Redirects.*')
warnings.filterwarnings('ignore', message='.*trust_remote_code.*')
warnings.filterwarnings('ignore', message='.*text_config.*')
warnings.filterwarnings('ignore', message='.*cadam32bit.*')
warnings.filterwarnings('ignore', message='.*cadam32bit_grad_fp32.*')
warnings.filterwarnings('ignore', message='.*text_config_dict.*')
warnings.filterwarnings('ignore', message='.*CLIPTextConfig.*')
warnings.filterwarnings('ignore', message='.*will be overriden.*')
warnings.filterwarnings('ignore', message='.*overriden.*')
warnings.filterwarnings('ignore', message='.*id2label.*')
warnings.filterwarnings('ignore', message='.*bos_token_id.*')
warnings.filterwarnings('ignore', message='.*eos_token_id.*')

# Override warnings.showwarning to filter cadam32bit warnings
_original_showwarning = warnings.showwarning
def filtered_showwarning(message, category, filename, lineno, file=None, line=None):
    """Filter out cadam32bit warnings before they're printed."""
    msg_str = str(message).lower()
    if 'cadam32bit' in msg_str or ('nonetype' in msg_str and 'cadam32bit' in msg_str):
        return  # Suppress this warning
    # Call original showwarning for all other warnings
    _original_showwarning(message, category, filename, lineno, file, line)
warnings.showwarning = filtered_showwarning

from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from src.utils.dicom_utils import load_image_smart, is_dicom_file, is_image_usable

# Configure tqdm for better display in log files (Slurm jobs)
# Use file=sys.stderr for better compatibility with Slurm log files
is_slurm = os.environ.get('SLURM_JOB_ID') is not None
# Check if running in interactive terminal
is_interactive = sys.stdout.isatty() if hasattr(sys.stdout, 'isatty') else False

# Configure tqdm with proper progress bar display
# Use stderr for progress bars to avoid interfering with stdout (results, etc.)
tqdm_kwargs = {
    'file': sys.stderr,  # Always use stderr for progress bars (works better with redirected output)
    'ncols': 100,  # Reasonable width for progress bar
    'mininterval': 0.5,  # Update every 0.5 seconds for smooth but not too frequent updates
    'miniters': 1,  # Update every iteration
    'disable': False,  # Enable by default
    'dynamic_ncols': False,  # Fixed width for consistency
    'leave': False,  # Don't leave progress bar after completion
    'bar_format': '{desc}: {percentage:3.0f}%|{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'  # Full progress bar with percentage, bar, count, time, and rate
}

from src.data.config import load_dataset_config
from src.data.dataloader import get_all_images_by_modality
# MultimodalModelWrapper is imported only for model_arch=clip (see below). LLaVA / LLaVA-Med use
# separate runners and must not require model_wrapper.py to parse at import time.
from src.utils.evaluation import (
    evaluate_sequential_modalities,
    print_evaluation_results,
    save_results,
    aggregate_patient_predictions,
    analyze_modality_agreement,
    analyze_patient_level_agreement
)

def extract_patient_id_from_img(img_info):
    """
    Extract patient_id from img_info with consistent fallback logic.
    This ensures the same patient_id is used for the same image across all steps.
    IMPORTANT: This function should return the SAME patient_id for the same image
    that was used during initial patient matching.
    """
    # First try: Use patient_id from img_info (set by dataloader)
    patient_id = img_info.get('patient_id')
    if patient_id is not None and str(patient_id).strip():
        return str(patient_id).strip()
    
    # Fallback 1: Try other metadata fields
    if patient_id is None:
        patient_id = img_info.get('image_id') or img_info.get('series_uid')
        if patient_id is not None and str(patient_id).strip():
            return str(patient_id).strip()
    
    # Fallback 2: Extract from filename/path (same pattern as dataloader)
    if patient_id is None:
        image_path = img_info.get('image_path', '')
        if image_path:
            filename = os.path.basename(image_path)
            # Try common patterns: A0001, patient_001, etc. (same as dataloader)
            try:
                import re
                patient_match = re.search(r'([A-Z]?\d{4,})', filename)
                if patient_match:
                    return patient_match.group(1)
            except Exception:
                pass
    
    # Fallback 3: Use hash of image path (ONLY if no other option)
    # This ensures consistency but should rarely be needed if dataloader works correctly
    if patient_id is None:
        image_path = img_info.get('image_path', '')
        if image_path:
            import hashlib
            patient_id = hashlib.md5(image_path.encode()).hexdigest()[:8]
    
    return patient_id

def main():
    parser = argparse.ArgumentParser(
        description="Sequential Modality Evaluation for Multimodal Models"
    )
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Root directory containing modality-specific folders'
    )
    parser.add_argument(
        '--modalities',
        type=str,
        nargs='+',
        default=['CT', 'MRI'],
        help='List of modalities to evaluate (e.g., CT MRI mammography)'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='openai/clip-vit-large-patch14',
        help='HuggingFace model name (default: openai/clip-vit-large-patch14)'
    )
    parser.add_argument(
        '--model_arch',
        type=str,
        choices=['clip', 'llava', 'llava_med'],
        default='clip',
        help='Model architecture to use (clip, llava, or llava_med). Default: clip'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu). Auto-detected if not specified'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of images per patient per modality (e.g., 100 means 100 CT + 100 PET per patient). If None, loads ALL images from all patients'
    )
    parser.add_argument(
        '--max_patients',
        type=int,
        default=None,
        help='Maximum number of patients to evaluate (only those with all modalities). If None, use all matched patients. Useful for quick runs (e.g. --max_patients 1)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for inference (default: 1). Use larger values (e.g., 8, 16) for faster GPU processing'
    )
    parser.add_argument(
        '--no_preprocess',
        action='store_true',
        help='Disable medical image preprocessing (contrast enhancement, histogram equalization)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Temperature scaling for logits (default: 0.8, lower = more confident predictions). Recommended: 0.7-0.9 for better calibration'
    )
    parser.add_argument(
        '--aggressive_preprocess',
        action='store_true',
        help='Use aggressive preprocessing (stronger contrast/sharpness enhancement). May improve difficult cases but can introduce artifacts'
    )
    parser.add_argument(
        '--no_weighted_ensemble',
        action='store_true',
        help='Disable weighted prompt ensemble (use simple mean instead)'
    )
    parser.add_argument(
        '--no_swap_test',
        action='store_true',
        help='Disable testing both swap strategies (use original swap only)'
    )
    parser.add_argument(
        '--no_progress',
        action='store_true',
        help='Disable progress bars for cleaner output'
    )
    parser.add_argument(
        '--dataset_config',
        type=str,
        default=None,
        help='Optional path to dataset_config.yaml describing modality folders and classes'
    )
    parser.add_argument(
        '--split',
        type=str,
        default=None,
        help='Optional split name (e.g., train/val/test) to filter images using metadata'
    )
    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='Hugging Face token for accessing private models. Can also be set via HF_TOKEN or HUGGING_FACE_HUB_TOKEN environment variable'
    )
    parser.add_argument(
        '--llava_med_flip_predictions',
        action='store_true',
        help='For LLaVA-Med only: flip the final binary prediction (0 ↔ 1). '
             'Useful when the model is systematically predicting the opposite label.'
    )
    parser.add_argument(
        '--class_names',
        type=str,
        nargs='+',
        default=None,
        help='Override class names used for prompts and labels (expects exactly two names)'
    )
    parser.add_argument(
        '--allow_single_modality',
        action='store_true',
        help='Permit running with a single modality (steps 2 and 3 will be skipped)'
    )
    parser.add_argument(
        '--skip_bad_images',
        action='store_true',
        help='Skip images that are pure white, pure black, or very low contrast (unclear). Logs counts at end.'
    )
    parser.add_argument(
        '--reverse_order',
        action='store_true',
        help='Process modalities in reversed order (e.g., Mod4→Mod3→Mod2→Mod1 instead of Mod1→Mod2→Mod3→Mod4)'
    )
    parser.add_argument(
        '--run_both_orders',
        action='store_true',
        help='Run both forward and reverse modality orders in one command. Saves two result files (e.g. results_*_CT_PET.json and results_*_PET_CT.json). Overrides --reverse_order.'
    )
    
    args = parser.parse_args()
    
    # Support N modalities for cascading context evaluation
    # No longer limited to 2 modalities
    if len(args.modalities) == 0:
        if not args.allow_single_modality:
            parser.error("Please provide at least one modality (e.g., CT).")
    
    if len(set(args.modalities)) != len(args.modalities):
        parser.error("Modalities for sequential evaluation must be distinct.")
    
    # Support reversed order if requested (ignored when --run_both_orders)
    if args.run_both_orders:
        pass  # run both orders; do not reverse now
    elif args.reverse_order:
        args.modalities = list(reversed(args.modalities))
        print(f"Reversed modality order: {args.modalities}")
    
    if not os.path.isdir(args.data_root):
        parser.error(f"Data root directory not found: {args.data_root}")

    dataset_config = load_dataset_config(args.dataset_config)

    if args.class_names is not None:
        if len(args.class_names) != 2:
            parser.error("Please provide exactly two class names (e.g., Adenocarcinoma Squamous).")
        class_names = args.class_names
    else:
        class_names = None
        for mod in args.modalities:
            mod_cfg = dataset_config.get('modalities', {}).get(mod, {})
            classes_cfg = mod_cfg.get('classes')
            if classes_cfg:
                # Sort by label index when available
                sorted_classes = sorted(classes_cfg.items(), key=lambda kv: kv[1])
                class_names = [name for name, _ in sorted_classes]
                break
        if class_names is None:
            # Generic fallback: use Class0 and Class1 if no classes found in config
            class_names = ['Class0', 'Class1']
    args.class_names = class_names
    
    # Set device
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Handle Hugging Face token
    hf_token = args.hf_token or os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    
    if hf_token:
        # Set environment variable for huggingface_hub to pick up
        os.environ['HF_TOKEN'] = hf_token
        os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    if args.model_arch == 'llava':
        from src.models.llava_runner import LLaVARunner
        model = LLaVARunner(
            model_name=args.model_name,
            device=device,
            class_names=args.class_names,
            hf_token=hf_token
        )
    elif args.model_arch == 'llava_med':
        from src.models.llava_med_runner import LLaVAMedRunner
        model = LLaVAMedRunner(
            model_name=args.model_name,
            device=device,
            class_names=args.class_names,
            hf_token=hf_token,
            flip_predictions=getattr(args, 'llava_med_flip_predictions', False),
        )
    else:
        from src.models.model_wrapper import MultimodalModelWrapper

        model = MultimodalModelWrapper(
            model_name=args.model_name,
            device=device,
            class_names=args.class_names,
            hf_token=hf_token
        )
    
    # Disable progress bars if --no_progress flag is set
    if hasattr(args, 'no_progress') and args.no_progress:
        tqdm_kwargs['disable'] = True

    # Load images for all modalities
    modalities = args.modalities
    modality_images = {}

    for mod in modalities:
        mod_images = get_all_images_by_modality(
            args.data_root,
            mod,
            classes=args.class_names,
            dataset_config_path=args.dataset_config,
            split=args.split
        )
        if not mod_images:
            parser.error(f"No images found for modality '{mod}'. Check folder structure and casing.")
        modality_images[mod] = mod_images

    # Use the global extract_patient_id_from_img function for consistency
    # This ensures the same patient_id extraction logic is used throughout

    def has_valid_patient_id(img):
        """Check if image has a valid patient_id (not None, not a hash)."""
        # Use the global extract_patient_id_from_img function
        patient_id = extract_patient_id_from_img(img)
        # Consider it valid if it's not None and not a hash (hashes are 8 hex chars)
        if patient_id is None:
            return False
        # If it's a hash (8 hex characters), it's not a real patient_id
        if len(patient_id) == 8 and all(c in '0123456789abcdef' for c in patient_id.lower()):
            return False
        return True

    # Ensure all patients with ALL specified modalities are included
    # When max_samples=100: Distribute 100 images per modality across ALL patients that have all modalities
    # Each patient gets all modality scans
    # When max_samples=None: Process ALL images from ALL patients (still with patient matching)
    if len(modalities) > 1:
        # Filter: Only keep images with valid patient_id for all modalities
        modality_images_with_pid = {}
        for mod in modalities:
            modality_images_with_pid[mod] = [img for img in modality_images[mod] if has_valid_patient_id(img)]
            if args.max_samples is not None and len(modality_images_with_pid[mod]) < args.max_samples:
                print(f"  Warning: Only {len(modality_images_with_pid[mod])} {mod} images with valid patient_id available.")

        # Group images by patient_id for all modalities (only valid patient_ids)
        modality_by_patient = {}
        for mod in modalities:
            modality_by_patient[mod] = {}
            for img in modality_images_with_pid[mod]:
                patient_id = extract_patient_id_from_img(img)
                if patient_id is not None:
                    if patient_id not in modality_by_patient[mod]:
                        modality_by_patient[mod][patient_id] = []
                    modality_by_patient[mod][patient_id].append(img)

        # Find ALL patients that have ALL specified modalities
        # When max_samples=100: Try to get 100 different patients, each with all modalities
        # Result: Patient 1 (all mods), Patient 2 (all mods), ..., Patient N (all mods)
        # If fewer patients available, use all available patients
        matched_patients = []
        # Start with patients from first modality
        for patient_id in modality_by_patient[modalities[0]]:
            # Check if this patient has all other modalities
            has_all_mods = True
            for mod in modalities[1:]:
                if patient_id not in modality_by_patient[mod] or len(modality_by_patient[mod][patient_id]) == 0:
                    has_all_mods = False
                    break
            if has_all_mods:
                # Patient must have at least 1 image for each modality
                for mod in modalities:
                    if len(modality_by_patient[mod][patient_id]) < 1:
                        has_all_mods = False
                        break
            if has_all_mods:
                matched_patients.append(patient_id)

        if len(matched_patients) == 0:
            mod_list = " and ".join(modalities)
            parser.error(f"No patients found with all modalities ({mod_list}). Check patient_id matching.")

        # Shuffle for reproducibility
        random.seed(42)
        random.shuffle(matched_patients)

        # Select all available patients (optionally limit with --max_patients)
        # max_samples controls how many images per patient per modality, not how many patients
        selected_patients = matched_patients
        if args.max_patients is not None and args.max_patients > 0:
            selected_patients = matched_patients[:args.max_patients]
        mod_list = ", ".join(modalities)
        print(f"\nFound {len(matched_patients)} patients with all modalities ({mod_list})", flush=True)
        if args.max_patients is not None and args.max_patients > 0:
            print(f"  Using {len(selected_patients)} patient(s) (--max_patients {args.max_patients})", flush=True)
        if args.max_samples is not None:
            print(f"  Will take up to {args.max_samples} images per modality from each patient", flush=True)
            print(f"  Expected total: ~{len(matched_patients) * args.max_samples} images per modality", flush=True)

        # For each selected patient, take images based on max_samples
        # If max_samples is None: take ALL images from each patient
        # If max_samples is specified: take up to max_samples images per modality per patient
        selected_modality_images = {mod: [] for mod in modalities}

        # Shuffle patient images for variety
        # CRITICAL: Set seed before each shuffle to ensure deterministic ordering
        # This ensures same image selection regardless of modality processing order
        for patient_id in selected_patients:
            for mod in modalities:
                patient_mod_images = modality_by_patient[mod][patient_id].copy()
                # Set seed based on patient_id and modality for deterministic but varied shuffling
                # This ensures same images selected for same patient+modality regardless of processing order
                seed_value = hash(f"{patient_id}_{mod}") % (2**31)
                random.seed(seed_value)
                random.shuffle(patient_mod_images)

                if args.max_samples is None:
                    # Take ALL images from this patient for this modality
                    selected_modality_images[mod].extend(patient_mod_images)
                else:
                    # Take up to max_samples images from this patient for this modality
                    mod_to_take = min(args.max_samples, len(patient_mod_images))
                    selected_modality_images[mod].extend(patient_mod_images[:mod_to_take])

        # Update image lists
        for mod in modalities:
            modality_images[mod] = selected_modality_images[mod]
    else:
        # Single modality case: shuffle and limit independently
        random.seed(42)
        for mod in modalities:
            random.shuffle(modality_images[mod])
            if args.max_samples is not None and args.max_samples > 0:
                modality_images[mod] = modality_images[mod][:args.max_samples]

    # Determine which modality orders to run (both when --run_both_orders)
    orders = [args.modalities, list(reversed(args.modalities))] if args.run_both_orders else [args.modalities]
    for run_idx, current_order in enumerate(orders):
        modalities = current_order
        if args.run_both_orders:
            print("\n" + "="*60, flush=True)
            print("Order {}: {}".format(run_idx + 1, " → ".join(modalities)), flush=True)
            print("="*60 + "\n", flush=True)

        # Print step plan

        results = {}
        # Format: {patient_id: {modality: {'prediction': int, 'class_name': str, 'confidence': float}}}
        # Each patient gets aggregated predictions for each modality from all their slices
        patient_predictions = {}
        # Count skipped bad images (pure white/black, low contrast) when --skip_bad_images
        bad_image_counts = {}

        # Store predictions by image for better matching and aggregation
        # Format: {modality: {patient_id: [{'image_id': str, 'prediction': int, 'class_name': str, 'confidence': float}, ...]}}
        patient_modality_predictions_list = {mod: {} for mod in modalities}

        # STEP 1-N: Process each modality alone (no context)
        step_num = 1
        for mod_idx, current_mod in enumerate(modalities):
            mod_images = modality_images[current_mod]
        
            total_images = len(mod_images)
    
            for img_info in tqdm(mod_images, desc=f"Processing {current_mod}", total=total_images, **tqdm_kwargs):
                # Make case_id unique by including image_path (image_id may not be unique)
                # Use basename of image_path to keep it readable but unique
                image_basename = os.path.basename(img_info.get('image_path', ''))
                case_id = f"{img_info['class'].lower()}_{image_basename}_{current_mod}"
                label = img_info['label']
                patient_id = extract_patient_id_from_img(img_info)
            
                if case_id not in results:
                    results[case_id] = []
                
                try:
                    # Smart loader: handles both regular images and DICOM files
                    img = load_image_smart(img_info['image_path'])
                    usable, reason = is_image_usable(img)
                    if not usable:
                        bad_image_counts[reason] = bad_image_counts.get(reason, 0) + 1
                        if getattr(args, 'skip_bad_images', False):
                            continue
                    
                    prediction = model.predict(
                        images={current_mod: img},
                        available_modalities=[current_mod],
                        batch_size=args.batch_size,
                        preprocess=not args.no_preprocess,
                        temperature=args.temperature,
                        use_weighted_ensemble=not args.no_weighted_ensemble,
                        try_both_swaps=not args.no_swap_test,
                        aggressive_preprocess=args.aggressive_preprocess,
                        previous_predictions=None  # No context for standalone processing
                    )
    
                    pred_class_name = args.class_names[prediction['prediction']]
    
                    # Handle patient_id extraction with robust fallbacks
                    # CRITICAL: Don't skip images - use fallback identifiers if needed
                    if patient_id is None:
                        # Try multiple fallback strategies
                        patient_id = (
                        img_info.get('patient_id') or  # Try metadata again
                        img_info.get('image_id') or    # Use image_id
                        img_info.get('series_uid') or  # Use series_uid
                        case_id                        # Use case_id as last resort
                        )
    
                    # If still None, generate a unique identifier from image path
                    if patient_id is None:
                        image_path = img_info.get('image_path', '')
                        if image_path:
                            # Extract any identifier from path (filename, folder, etc.)
                            path_parts = image_path.replace('\\', '/').split('/')
                            # Try to find any numeric or alphanumeric identifier
                            for part in reversed(path_parts):
                                if part and (part.replace('.', '').replace('_', '').replace('-', '').isalnum()):
                                    patient_id = part.split('.')[0]  # Remove extension
                                    break
                        
                            if patient_id is None:
                                import hashlib
                                patient_id = hashlib.md5(image_path.encode()).hexdigest()[:8]
                
                    # Store ALL predictions with their identifiers for 1-to-1 matching
                    # Note: patient_predictions will be populated after aggregation (lines 724-737)
                    if patient_id not in patient_modality_predictions_list[current_mod]:
                        patient_modality_predictions_list[current_mod][patient_id] = []
    
                    mod_prediction_entry = {
                    'image_id': img_info.get('image_id', img_info.get('image_path', 'unknown')),
                    'prediction': prediction['prediction'],
                    'class_name': pred_class_name,
                    'confidence': prediction['confidence']
                    }
    
                    if 'slice_index' in img_info:
                        mod_prediction_entry['slice_index'] = img_info['slice_index']
    
                    if 'series_uid' in img_info:
                        mod_prediction_entry['series_uid'] = img_info['series_uid']
    
                    patient_modality_predictions_list[current_mod][patient_id].append(mod_prediction_entry)
    
                    # Store result
                    mod_result = {
                    'modalities_used': [current_mod],
                    'prediction': prediction['prediction'],
                    'confidence': prediction['confidence'],
                    'label': label,
                    'used_context': False,
                    'context_from': [],
                    'probabilities': prediction.get('probabilities', {}),
                    'probabilities_array': prediction.get('probabilities_array', []),
                    'probabilities_before_boosting': prediction.get('probabilities_before_boosting'),
                    'logits': prediction.get('logits', []),
                    'patient_id': patient_id
                    }
    
                    if 'slice_index' in img_info:
                        mod_result['slice_index'] = img_info['slice_index']
    
                    if 'image_id' in img_info:
                        mod_result['image_id'] = img_info['image_id']
    
                    results[case_id].append(mod_result)
    
                except Exception as e:
                    print(f"Error processing {case_id} with {current_mod}: {e}", file=sys.stderr)
                    traceback.print_exc()
            
            # Aggregate predictions per patient for patient-level evaluation (OUTSIDE try/except)
            unique_patients = len(patient_modality_predictions_list[current_mod])
            total_predictions = sum(len(preds) for preds in patient_modality_predictions_list[current_mod].values())
            aggregated_count = 0
            
            
            
            for patient_id, slices in patient_modality_predictions_list[current_mod].items():
                if patient_id is None:
                    print(f"Warning: Skipping aggregation for None patient_id (has {len(slices)} slices)", file=sys.stderr, flush=True)
                    continue
    
                try:
                    aggregated = aggregate_patient_predictions(slices)
                    if patient_id not in patient_predictions:
                        patient_predictions[patient_id] = {}
                    patient_predictions[patient_id][current_mod] = {
                        'prediction': aggregated['prediction'],
                        'class_name': args.class_names[aggregated['prediction']],
                        'confidence': aggregated['confidence']
                    }
                    aggregated_count += 1
                except Exception as e:
                    print(f"Warning: Failed to aggregate {current_mod} predictions for patient {patient_id}: {e}", file=sys.stderr, flush=True)
                    if slices:
                        first_slice = slices[0]
                        if patient_id not in patient_predictions:
                            patient_predictions[patient_id] = {}
                        patient_predictions[patient_id][current_mod] = {
                            'prediction': first_slice.get('prediction', 0),
                            'class_name': args.class_names[first_slice.get('prediction', 0)],
                            'confidence': first_slice.get('confidence', 0.5)
                        }
                        aggregated_count += 1
            
            
            step_num += 1
    
        # STEP N+1 to 2N-1: Process each modality with context from all previous ones
        # For modality i, use context from modalities [0..i-1]
        for mod_idx in range(1, len(modalities)):
            current_mod = modalities[mod_idx]
            context_mods = modalities[:mod_idx]  # All previous modalities
            context_str = "+".join(context_mods)
    
            mod_images = modality_images[current_mod]
            # Match images to predictions by patient_id
            # Goal: Each current modality image should get context from all previous modalities from the same patient
            filtered_mod_images = []
            patients_with_context = set(patient_predictions.keys())
    
            # Match each current modality image to its corresponding previous modality predictions by patient_id
            images_without_patient_id = 0
            images_missing_context = 0
            for img_info in mod_images:
                patient_id = extract_patient_id_from_img(img_info)
    
                # Use same fallback logic as Step 1-2
                if patient_id is None:
                    patient_id = (
                        img_info.get('patient_id') or
                        img_info.get('image_id') or
                        img_info.get('series_uid') or
                        f"{img_info.get('class', 'unknown').lower()}_{os.path.basename(img_info.get('image_path', 'unknown'))}_{current_mod}"
                    )
                    if not patient_id:
                        import hashlib
                        image_path = img_info.get('image_path', '')
                        if image_path:
                            patient_id = hashlib.md5(image_path.encode('utf-8')).hexdigest()[:8]
                        else:
                            patient_id = f"unknown_{len(filtered_mod_images)}"
                        images_without_patient_id += 1
    
                # Check if this patient has predictions for ALL context modalities
                has_all_context = True
                if patient_id not in patients_with_context:
                    has_all_context = False
                    images_missing_context += 1
                else:
                    for ctx_mod in context_mods:
                        if ctx_mod not in patient_predictions[patient_id]:
                            has_all_context = False
                            break
                    if not has_all_context:
                        images_missing_context += 1
                
                if has_all_context:
                    filtered_mod_images.append(img_info)
    
            # Initialize predictions list for aggregation (with context)
            patient_mod_predictions_list = {}
            context_used_count = 0
            total_mod_images = len(filtered_mod_images)
    
            for img_info in tqdm(filtered_mod_images, desc=f"Processing {current_mod} with {context_str} context", total=total_mod_images, **tqdm_kwargs):
                image_basename = os.path.basename(img_info.get('image_path', ''))
                case_id = f"{img_info['class'].lower()}_{image_basename}_{current_mod}"
                label = img_info['label']
    
                patient_id = extract_patient_id_from_img(img_info)
                
                if case_id not in results:
                    results[case_id] = []
                
                try:
                    # Smart loader: handles both regular images and DICOM files
                    img = load_image_smart(img_info['image_path'])
                    usable, reason = is_image_usable(img)
                    if not usable:
                        bad_image_counts[reason] = bad_image_counts.get(reason, 0) + 1
                        if getattr(args, 'skip_bad_images', False):
                            continue

                    # Build previous_predictions dict from all context modalities
                    # BEST STRATEGY: 1-to-1 matching with intelligent fallbacks
                    previous_predictions = {}
                    context_used = False
    
                    if patient_id and patient_id in patient_predictions:
                        # Check if patient has all context modalities
                        has_all = True
                        for ctx_mod in context_mods:
                            if ctx_mod not in patient_predictions[patient_id]:
                                has_all = False
                                break
                        
                        if has_all:
                            current_mod_slice_index = img_info.get('slice_index')
                            
                            # For each context modality, try to find best matching prediction
                            for ctx_mod in context_mods:
                                matched_ctx_prediction = None
                                
                                # Try to match by slice_index if available
                                if current_mod_slice_index is not None and patient_id in patient_modality_predictions_list[ctx_mod]:
                                    # Strategy 1: Try exact match by slice_index
                                    for ctx_pred in patient_modality_predictions_list[ctx_mod][patient_id]:
                                        if 'slice_index' in ctx_pred and ctx_pred['slice_index'] == current_mod_slice_index:
                                            matched_ctx_prediction = ctx_pred
                                            break
                                    
                                    # Strategy 2: If no exact match, find nearest slice_index
                                    if matched_ctx_prediction is None:
                                        min_distance = float('inf')
                                        for ctx_pred in patient_modality_predictions_list[ctx_mod][patient_id]:
                                            if 'slice_index' in ctx_pred:
                                                distance = abs(ctx_pred['slice_index'] - current_mod_slice_index)
                                                if distance < min_distance:
                                                    min_distance = distance
                                                    matched_ctx_prediction = ctx_pred
                                
                                # Strategy 3: Use matched prediction or fallback to patient-level
                                if matched_ctx_prediction is not None:
                                    previous_predictions[ctx_mod] = {
                                        'prediction': matched_ctx_prediction['prediction'],
                                        'class_name': matched_ctx_prediction['class_name'],
                                        'confidence': matched_ctx_prediction.get('confidence', 0.5)
                                    }
                                else:
                                    # Fallback: Use patient-level aggregated prediction
                                    if ctx_mod in patient_predictions[patient_id]:
                                        ctx_result = patient_predictions[patient_id][ctx_mod].copy()
                                        if 'confidence' not in ctx_result:
                                            ctx_result['confidence'] = 0.5
                                        previous_predictions[ctx_mod] = ctx_result
                            
                            if previous_predictions:
                                context_used = True
                                context_used_count += 1
    
                    # Model.predict() receives:
                    # 1. Patient i's current modality image
                    # 2. Patient i's previous modality output results from all context modalities
                    # The model uses prompts like: "Given that the {mod1} scan showed X, and the {mod2} scan showed Y, this {current_mod} scan shows..."
                    prediction = model.predict(
                        images={current_mod: img},
                        available_modalities=[current_mod],
                        batch_size=args.batch_size,
                        preprocess=not args.no_preprocess,
                        temperature=args.temperature,
                        use_weighted_ensemble=not args.no_weighted_ensemble,
                        try_both_swaps=not args.no_swap_test,
                        aggressive_preprocess=args.aggressive_preprocess,
                        previous_predictions=previous_predictions if context_used else None
                    )
    
                    # Store prediction for this patient
                    if patient_id is None:
                        continue
    
                    if patient_id not in patient_predictions:
                        patient_predictions[patient_id] = {}
                    
                    # Safeguard: Use class_names mapping with int prediction, fallback to str if KeyError
                    try:
                        pred_class_name = args.class_names[prediction['prediction']]
                    except (KeyError, IndexError, TypeError):
                        pred_class_name = str(prediction['prediction'])
    
                    # Store all predictions for aggregation
                    if patient_id not in patient_mod_predictions_list:
                        patient_mod_predictions_list[patient_id] = []
    
                    mod_prediction_entry = {
                        'image_id': img_info.get('image_id', img_info.get('image_path', 'unknown')),
                        'prediction': prediction['prediction'],
                        'class_name': pred_class_name,
                        'confidence': prediction['confidence']
                    }
                    if 'slice_index' in img_info:
                        mod_prediction_entry['slice_index'] = img_info['slice_index']
                    if 'series_uid' in img_info:
                        mod_prediction_entry['series_uid'] = img_info['series_uid']
                    patient_mod_predictions_list[patient_id].append(mod_prediction_entry)
    
                    # Store result with context information
                    mod_result_with_context = {
                        'modalities_used': [current_mod],
                        'prediction': prediction['prediction'],
                        'confidence': prediction['confidence'],
                        'label': label,
                        'used_context': context_used,
                        'context_from': context_mods if context_used else [],
                        'probabilities': prediction.get('probabilities', {}),
                        'probabilities_array': prediction.get('probabilities_array', []),
                        'probabilities_before_boosting': prediction.get('probabilities_before_boosting'),
                        'logits': prediction.get('logits', []),
                        'patient_id': patient_id
                    }
                    if 'slice_index' in img_info:
                        mod_result_with_context['slice_index'] = img_info['slice_index']
                    if 'image_id' in img_info:
                        mod_result_with_context['image_id'] = img_info['image_id']
                    if case_id not in results:
                        results[case_id] = []
                    results[case_id].append(mod_result_with_context)
                except Exception as e:
                    print(f"Error processing {case_id} with {current_mod}: {e}", file=sys.stderr)
                    traceback.print_exc()
    
            # Aggregate predictions per patient
            aggregated_count = 0
            for patient_id, slices in patient_mod_predictions_list.items():
                if patient_id is None:
                    print(f"Warning: Skipping aggregation for None patient_id in Step {step_num} (has {len(slices)} slices)", 
                        file=sys.stderr, flush=True)
                    continue
                try:
                    aggregated = aggregate_patient_predictions(slices)
                    if patient_id not in patient_predictions:
                        patient_predictions[patient_id] = {}
                    patient_predictions[patient_id][current_mod] = {
                        'prediction': aggregated['prediction'],
                        'class_name': args.class_names[aggregated['prediction']],
                        'confidence': aggregated['confidence']
                    }
                    aggregated_count += 1
                except Exception as e:
                    print(f"Warning: Failed to aggregate {current_mod} predictions for patient {patient_id}: {e}", 
                        file=sys.stderr, flush=True)
                    if slices:
                        first_slice = slices[0]
                        if patient_id not in patient_predictions:
                            patient_predictions[patient_id] = {}
                        patient_predictions[patient_id][current_mod] = {
                            'prediction': first_slice.get('prediction', 0),
                            'class_name': args.class_names[first_slice.get('prediction', 0)],
                            'confidence': first_slice.get('confidence', 0.5)
                        }
                        aggregated_count += 1
    
            # Calculate statistics before debug check
            total_mod_predictions = sum(len(slices) for slices in patient_mod_predictions_list.values())
            unique_patients_mod = len(patient_mod_predictions_list)
    
    
            step_num += 1
        
        # Evaluate results
        if not results:
            print("Warning: No results to evaluate. Check if images were processed successfully.", file=sys.stderr, flush=True)
            if args.run_both_orders and run_idx + 1 < len(orders):
                print("Skipping to next modality order.", flush=True)
                continue
            return
        
        # Count results by modality
        total_results = sum(len(preds) for preds in results.values())
    
        # Run fingerprint: verifies this run used this model (different model or order => different hash)
        if total_results > 0:
            _sample_preds = []
            for _cid, preds in list(results.items())[:20]:
                for p in preds[:1]:
                    _sample_preds.append(p.get('prediction'))
                    _sample_preds.append(round(p.get('confidence', 0), 4))
            import hashlib
            _fp = hashlib.sha256(str(_sample_preds).encode()).hexdigest()[:12]
            print(f"\n[Run fingerprint] model={args.model_name} predictions_hash={_fp}\n", flush=True)
    
        # Count by step type (for logging/debugging)
        step_counts = {}
        for case_id, preds in results.items():
            for pred in preds:
                mods = pred.get('modalities_used', [])
                used_ctx = pred.get('used_context', False)
                ctx_from = pred.get('context_from', [])
    
                # Determine step name based on modalities and context
                if len(mods) == 1:
                    mod = mods[0]
                    if used_ctx and ctx_from:
                        # Modality with context
                        step_name = f"{mod}_with_{'+'.join(ctx_from)}"
                    else:
                        # Modality alone
                        step_name = mod
                else:
                    # Multiple modalities together
                    step_name = "+".join(mods)
    
                if step_name not in step_counts:
                    step_counts[step_name] = 0
                step_counts[step_name] += 1
    
        try:
            evaluation_results = evaluate_sequential_modalities(
                results, modalities, calibration_temperature=args.temperature
            )
        except Exception as e:
            print(f"ERROR: Failed to evaluate results: {e}", file=sys.stderr, flush=True)
            traceback.print_exc()
            if args.run_both_orders and run_idx + 1 < len(orders):
                print("Continuing with next modality order after evaluation failure.", flush=True)
                continue
            return
    
        # Add patient-level analysis if we have multiple modalities
        if len(modalities) >= 2 and 'agreement_metrics' in evaluation_results:
            try:
                # Extract patient-level predictions for agreement analysis
                # For each modality, collect predictions WITHOUT context (standalone)
                patient_mod_preds = {mod: {} for mod in modalities}
                
                for case_id, case_results in results.items():
                    for result in case_results:
                        patient_id = result.get('patient_id')
                        if patient_id is None:
                            continue
                        
                        mods_used = result.get('modalities_used', [])
                        used_context = result.get('used_context', False)
                        
                        # CRITICAL: Only collect predictions WITHOUT context for agreement analysis
                        # (predictions with context are combined, not pure modality)
                        if len(mods_used) == 1 and not used_context:
                            mod = mods_used[0]
                            if mod in patient_mod_preds:
                                if patient_id not in patient_mod_preds[mod]:
                                    patient_mod_preds[mod][patient_id] = []
                                patient_mod_preds[mod][patient_id].append(result)
                
                # Aggregate to patient-level for agreement analysis
                # Find common patients across all modalities
                # For N modalities: analyze all pairs (generic, not limited to first two)
                # Store patient-level results for all modality pairs
                patient_level_results = {}
                
                # Analyze all pairs of modalities (generic for N modalities)
                for i in range(len(modalities)):
                    for j in range(i + 1, len(modalities)):
                        mod1 = modalities[i]
                        mod2 = modalities[j]
                        pair_key = (mod1, mod2)
                        pair_key_canonical = tuple(sorted([mod1, mod2]))  # Match evaluation's key
                        
                        # Find common patients for this pair
                        common_patient_ids = sorted(set(patient_mod_preds[mod1].keys()) & set(patient_mod_preds[mod2].keys()))
                        
                        if not common_patient_ids:
                            continue
                        
                        patient_level_mod1 = []
                        patient_level_mod2 = []
                        patient_ids_list = []
                        
                        for patient_id in common_patient_ids:
                            mod1_slices = patient_mod_preds[mod1][patient_id]
                            mod2_slices = patient_mod_preds[mod2][patient_id]
                            
                            # Skip if either list is empty
                            if not mod1_slices or not mod2_slices:
                                continue
                            
                            # Aggregate predictions for this patient
                            mod1_aggregated = aggregate_patient_predictions(mod1_slices)
                            mod2_aggregated = aggregate_patient_predictions(mod2_slices)
                            
                            # Add full prediction info for certainty analysis
                            # For patient-level, use pre-boosting probabilities for realistic certainty metrics
                            mod1_conf = mod1_aggregated['confidence']
                            
                            # Calculate confidence from pre-boosting probabilities if available
                            mod2_probs_before_list = []
                            for mod2_slice in mod2_slices:
                                probs_before = mod2_slice.get('probabilities_before_boosting')
                                if probs_before is not None and len(probs_before) >= 2:
                                    mod2_probs_before_list.append(np.array(probs_before))
                            
                            if mod2_probs_before_list:
                                avg_probs_before = np.mean(mod2_probs_before_list, axis=0)
                                mod2_conf = float(np.max(avg_probs_before))
                            else:
                                first_mod2_slice = mod2_slices[0]
                                probs_before = first_mod2_slice.get('probabilities_before_boosting')
                                if probs_before is not None and len(probs_before) >= 2:
                                    mod2_conf = float(np.max(np.array(probs_before)))
                                else:
                                    mod2_conf = mod2_aggregated['confidence']
                            
                            mod1_full = {
                                'prediction': mod1_aggregated['prediction'],
                                'confidence': mod1_conf,
                                'probabilities': mod1_slices[0].get('probabilities', {}),
                                'probabilities_array': mod1_slices[0].get('probabilities_array', []),
                                'logits': mod1_slices[0].get('logits', []),
                                'patient_id': patient_id
                            }
                            mod2_full = {
                                'prediction': mod2_aggregated['prediction'],
                                'confidence': mod2_conf,
                                'probabilities': mod2_slices[0].get('probabilities', {}),
                                'probabilities_array': mod2_slices[0].get('probabilities_array', []),
                                'probabilities_before_boosting': mod2_slices[0].get('probabilities_before_boosting'),
                                'logits': mod2_slices[0].get('logits', []),
                                'patient_id': patient_id
                            }
                            
                            patient_level_mod1.append(mod1_full)
                            patient_level_mod2.append(mod2_full)
                            patient_ids_list.append(patient_id)
                        
                        # Analyze patient-level vs slice-level agreement for this pair
                        if patient_level_mod1 and patient_level_mod2:
                            # Get slice-level agreement for this specific pair (use canonical key)
                            slice_level_agreement = {}
                            pairwise = evaluation_results.get('pairwise_agreements', {})
                            if pair_key_canonical in pairwise:
                                slice_level_agreement = pairwise[pair_key_canonical]
                            elif pair_key in pairwise:
                                slice_level_agreement = pairwise[pair_key]
                            elif 'agreement_metrics' in evaluation_results:
                                # Fallback to backward compatibility variable
                                slice_level_agreement = evaluation_results.get('agreement_metrics', {})
                            
                            patient_agreement_analysis = analyze_patient_level_agreement(
                                slice_level_agreement,
                                patient_level_mod1,
                                patient_level_mod2,
                                patient_ids_list
                            )
                            
                            # Store results for this pair (generic for N modalities)
                            if 'patient_level_agreements' not in evaluation_results:
                                evaluation_results['patient_level_agreements'] = {}
                            evaluation_results['patient_level_agreements'][pair_key] = patient_agreement_analysis
                            
                            # For backward compatibility: store first pair as 'patient_level_agreement'
                            if i == 0 and j == 1:
                                evaluation_results['patient_level_agreement'] = patient_agreement_analysis
            except Exception as e:
                print(f"Warning: Failed to complete patient-level analysis: {e}", file=sys.stderr, flush=True)
                traceback.print_exc()
        
        # Save results with modality order in filename
        # Format: results_MODELNAME_MOD1_MOD2_MOD3_...json
        # Example: results_openai_clip-vit-base-patch32_CT_PET_MRI.json
        model_name_safe = args.model_name.replace("/", "_")
        modality_suffix = "_" + "_".join(modalities)
        
        output_file = os.path.join(
            args.output_dir,
            f'results_{model_name_safe}{modality_suffix}.json'
        )
        save_results(evaluation_results, output_file)
    
        # Print all evaluation results
        print_evaluation_results(evaluation_results)
        
        print(f"\nResults saved to {output_file}", flush=True)
        if bad_image_counts:
            total = sum(bad_image_counts.values())
            parts = [f"{k}: {v}" for k, v in sorted(bad_image_counts.items())]
            if getattr(args, 'skip_bad_images', False):
                print(f"Skipped {total} bad image(s) ({', '.join(parts)})", flush=True)
            else:
                print(f"Detected {total} bad image(s) ({', '.join(parts)}). Use --skip_bad_images to exclude from evaluation.", flush=True)

if __name__ == '__main__':
    main()

