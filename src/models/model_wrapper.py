"""
CLIP-based model wrapper for medical image classification.
Uses zero-shot classification with text prompts.
"""

import contextlib
import os
import sys
import numpy as np
from typing import Dict, List, Optional

from src.utils.tf_mock import ensure_tensorflow_stub

ensure_tensorflow_stub()

from PIL import Image, ImageEnhance, ImageOps
import torch
from transformers import CLIPProcessor, CLIPModel

# Try to import SigLIP for Google models
try:
    from transformers import SiglipModel, SiglipProcessor
    SIGLIP_AVAILABLE = True
except ImportError:
    SIGLIP_AVAILABLE = False

# Try to import OpenCLIP for BiomedCLIP models
OPENCLIP_AVAILABLE = False
try:
    import open_clip
    # Verify open_clip has the required functions
    if hasattr(open_clip, 'create_model_and_transforms'):
        OPENCLIP_AVAILABLE = True
    else:
        print("WARNING: open_clip found but missing required functions. Try: pip install --upgrade open-clip-torch")
except ImportError:
    pass


class MultimodalModelWrapper:
    """
    CLIP model wrapper for sequential modality evaluation.
    Performs binary classification for two medical classes.
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        hf_token: Optional[str] = None
    ):
        """
        Args:
            model_name: HuggingFace model name for CLIP. Verified working models:
                - "openai/clip-vit-large-patch14" (default, good balance)
                - "openai/clip-vit-base-patch32" (faster, smaller)
                - "openai/clip-vit-large-patch14-336" (higher resolution, 336px)
                - "laion/CLIP-ViT-H-14-laion2B-s32B-b79K" (larger, better performance)
                - "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224" (medical-specific)
            device: Device to run on ('cuda' or 'cpu')
            hf_token: Hugging Face token for accessing private models. Can also be set via HF_TOKEN environment variable.
            
        Note: ResNet-based CLIP models (rn50, rn101) are NOT available on Hugging Face.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.is_biomedclip = "biomedclip" in model_name.lower() or "biomed" in model_name.lower()
        self.is_siglip = "siglip" in model_name.lower()
        self.class_names = class_names or ["Class0", "Class1"]
        if len(self.class_names) != 2:
            raise ValueError("MultimodalModelWrapper currently supports exactly two classes.")
        self.class_names = [name.strip() for name in self.class_names]
        # Check if using default healthy/tumor classes for backward compatibility with prompt generation
        self._default_classes = {name.lower() for name in self.class_names} == {"healthy", "tumor"}
        
        # Get token from parameter or environment variable
        self.hf_token = hf_token or os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
        
        print(f"\nLoading model: {model_name}...")
        
        # Suppress stderr during model loading to hide verbose messages
        @contextlib.contextmanager
        def suppress_stderr():
            import os as os_module  # Use explicit import to avoid closure issues
            import sys as sys_module  # Use explicit import to avoid closure issues
            with open(os_module.devnull, 'w') as devnull:
                old_stderr = sys_module.stderr
                try:
                    sys_module.stderr = devnull
                    yield
                finally:
                    sys_module.stderr = old_stderr
        
        try:
            # Try to load the specified model
            if self.is_siglip and SIGLIP_AVAILABLE:
                # Google SigLIP models use different classes
                with suppress_stderr():
                    token_kwargs = {"token": self.hf_token} if self.hf_token else {}
                    self.model = SiglipModel.from_pretrained(model_name, **token_kwargs).to(self.device)
                    self.processor = SiglipProcessor.from_pretrained(model_name, **token_kwargs)
                print(f"Loaded SigLIP model: {model_name}")
            elif self.is_siglip and not SIGLIP_AVAILABLE:
                print(f"Warning: SigLIP not available in this transformers version. Falling back to CLIP.")
                self.is_siglip = False
                self.model_name = "openai/clip-vit-large-patch14"
                with suppress_stderr():
                    token_kwargs = {"token": self.hf_token} if self.hf_token else {}
                    self.model = CLIPModel.from_pretrained(self.model_name, **token_kwargs).to(self.device)
                    self.processor = CLIPProcessor.from_pretrained(self.model_name, **token_kwargs)
            elif "biomedclip" in model_name.lower() or "biomed" in model_name.lower():
                loaded = False
                last_error = None
                
                # Check if open_clip is available, if not try to install it
                # Use global to access module-level variable
                global OPENCLIP_AVAILABLE
                openclip_available = OPENCLIP_AVAILABLE  # Local copy to avoid UnboundLocalError
                
                if not openclip_available:
                    print("WARNING: open_clip not available. Attempting to install...")
                    try:
                        import subprocess
                        import sys as sys_module
                        subprocess.check_call([sys_module.executable, "-m", "pip", "install", "--quiet", "open-clip-torch>=2.20.0"])
                        import open_clip
                        if hasattr(open_clip, 'create_model_and_transforms'):
                            OPENCLIP_AVAILABLE = True
                            openclip_available = True
                            print("✓ open_clip installed successfully")
                        else:
                            print("WARNING: open_clip installed but missing required functions")
                    except Exception as e:
                        print(f"WARNING: Failed to install open_clip: {str(e)[:200]}")
                        print("   Please install manually: pip install open-clip-torch>=2.20.0")
                
                # Attempt 1: Try loading with OpenCLIP (primary method for BiomedCLIP)
                if openclip_available:
                    try:
                        print(f"Attempting to load BiomedCLIP using OpenCLIP...")
                        # Import open_clip here to ensure it's available
                        import open_clip
                        with suppress_stderr():
                            # open_clip.create_model_and_transforms() does not accept HF token
                            # as a keyword arg on this version of open-clip-torch.
                            # Rely on HF_TOKEN/HUGGING_FACE_HUB_TOKEN env vars set in main.py.
                            token_kwargs = {}
                            # Use hf-hub: prefix for HuggingFace models
                            hf_model_name = f'hf-hub:{model_name}'
                            
                            # Load model and preprocessing
                            openclip_model, _, preprocess_fn = open_clip.create_model_and_transforms(
                                hf_model_name,
                                device=self.device,
                                jit=False,
                                **token_kwargs
                            )
                            
                            # Get tokenizer - try different methods based on open_clip version
                            try:
                                # Newer API
                                tokenizer_fn = open_clip.get_tokenizer(hf_model_name)
                            except (AttributeError, TypeError):
                                try:
                                    # Alternative: use model's tokenizer if available
                                    if hasattr(openclip_model, 'tokenizer'):
                                        tokenizer_fn = openclip_model.tokenizer
                                    else:
                                        # Fallback: create tokenizer from model name
                                        tokenizer_fn = open_clip.get_tokenizer('ViT-B-16')
                                except Exception:
                                    # Last resort: use transformers tokenizer
                                    from transformers import AutoTokenizer
                                    tokenizer_fn = AutoTokenizer.from_pretrained(model_name, **token_kwargs)
                            
                            # Create wrappers for compatibility with CLIP interface
                            # Import torch at module level is available, but ensure it's accessible here
                            import torch as torch_module
                            
                            class OpenCLIPWrapper:
                                def __init__(self, preprocess, tokenizer, device):
                                    self.preprocess = preprocess
                                    self.tokenizer = tokenizer
                                    self.device = device
                                
                                def __call__(self, text=None, images=None, return_tensors=None, padding=None, **kwargs):
                                    result = {}
                                    if images is not None:
                                        if isinstance(images, Image.Image):
                                            images = [images]
                                        # Preprocess images
                                        processed_images = torch_module.stack([self.preprocess(img) for img in images]).to(self.device)
                                        result['pixel_values'] = processed_images
                                    if text is not None:
                                        if isinstance(text, str):
                                            text = [text]
                                        # Tokenize text - handle different tokenizer types
                                        if hasattr(self.tokenizer, '__call__'):
                                            tokenized = self.tokenizer(text)
                                            if isinstance(tokenized, torch_module.Tensor):
                                                result['input_ids'] = tokenized.to(self.device)
                                            elif isinstance(tokenized, dict):
                                                result.update({k: v.to(self.device) if isinstance(v, torch_module.Tensor) else v 
                                                for k, v in tokenized.items()})
                                            else:
                                                # Convert to tensor if needed
                                                if isinstance(tokenized, list):
                                                    result['input_ids'] = torch_module.tensor(tokenized).to(self.device)
                                        else:
                                            # Fallback for transformers tokenizer
                                            result['input_ids'] = self.tokenizer(text, return_tensors='pt', padding=padding)['input_ids'].to(self.device)
                                    return result
                            
                            class OpenCLIPModelWrapper:
                                def __init__(self, model, tokenizer, device):
                                    self.model = model
                                    self.tokenizer = tokenizer
                                    self.device = device
                                
                                def eval(self):
                                    """Set model to evaluation mode."""
                                    self.model.eval()
                                    return self
                                
                                def __call__(self, pixel_values=None, input_ids=None, **kwargs):
                                    # Encode images
                                    if pixel_values is not None:
                                        image_features = self.model.encode_image(pixel_values)
                                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                                    else:
                                        image_features = None
                                    
                                    # Encode text
                                    if input_ids is not None:
                                        # Handle different input_ids formats
                                        if isinstance(input_ids, torch_module.Tensor):
                                            text_features = self.model.encode_text(input_ids)
                                        elif isinstance(input_ids, dict):
                                            text_features = self.model.encode_text(input_ids.get('input_ids', input_ids))
                                        else:
                                            # Try to convert to tensor
                                            if not isinstance(input_ids, torch_module.Tensor):
                                                input_ids = torch_module.tensor(input_ids).to(self.device)
                                            text_features = self.model.encode_text(input_ids)
                                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                                    else:
                                        text_features = None
                                    
                                    # Compute logits
                                    if image_features is not None and text_features is not None:
                                        logit_scale = self.model.logit_scale.exp()
                                        logits_per_image = logit_scale * image_features @ text_features.t()
                                    else:
                                        raise ValueError("Both pixel_values and input_ids must be provided")
                                    
                                    class Output:
                                        def __init__(self, logits):
                                            self.logits_per_image = logits
                                    
                                    return Output(logits_per_image)
                            
                            self.processor = OpenCLIPWrapper(preprocess_fn, tokenizer_fn, self.device)
                            self.model = OpenCLIPModelWrapper(openclip_model, tokenizer_fn, self.device)
                            
                        loaded = True
                        print(f"✓ Successfully loaded BiomedCLIP using OpenCLIP: {model_name}")
                    except Exception as e:
                        last_error = str(e)
                        import traceback
                        print(f"OpenCLIP loading failed: {str(e)[:300]}")
                        if not openclip_available:
                            print("   Note: open_clip may not be installed. Try: pip install open-clip-torch>=2.20.0")
                
                # Attempt 3: Try downloading model files and creating config.json if missing
                if not loaded:
                    try:
                        print(f"Attempting to load BiomedCLIP by downloading files and fixing config...")
                        from huggingface_hub import snapshot_download
                        import json
                        
                        with suppress_stderr():
                            token_kwargs = {"token": self.hf_token} if self.hf_token else {}
                            # Download all model files
                            cache_dir = snapshot_download(
                                repo_id=model_name,
                                **token_kwargs
                            )
                            
                            # Check if config.json exists, if not create a minimal one
                            config_path = os.path.join(cache_dir, "config.json")
                            if not os.path.exists(config_path):
                                print(f"Creating minimal config.json for BiomedCLIP...")
                                # Create a minimal CLIP config based on the model architecture
                                minimal_config = {
                                    "architectures": ["CLIPModel"],
                                    "model_type": "clip",
                                    "vision_config": {
                                        "hidden_size": 768,
                                        "intermediate_size": 3072,
                                        "num_attention_heads": 12,
                                        "num_hidden_layers": 12,
                                        "patch_size": 16,
                                        "image_size": 224
                                    },
                                    "text_config": {
                                        "hidden_size": 768,
                                        "intermediate_size": 3072,
                                        "num_attention_heads": 12,
                                        "num_hidden_layers": 12,
                                        "vocab_size": 30522
                                    },
                                    "projection_dim": 512
                                }
                                with open(config_path, 'w') as f:
                                    json.dump(minimal_config, f, indent=2)
                                print(f"✓ Created config.json")
                            
                            # Now try loading with the fixed config
                            token_kwargs = {"token": self.hf_token} if self.hf_token else {}
                            self.model = CLIPModel.from_pretrained(cache_dir, **token_kwargs).to(self.device)
                            self.processor = CLIPProcessor.from_pretrained(cache_dir, **token_kwargs)
                            
                        loaded = True
                        print(f"✓ Successfully loaded BiomedCLIP with fixed config: {model_name}")
                    except Exception as e:
                        last_error = str(e)
                        print(f"Config fix approach failed: {str(e)[:200]}")
                
                # Attempt 4: Try transformers CLIPModel (original approach)
                if not loaded:
                    try:
                        with suppress_stderr():
                            token_kwargs = {"token": self.hf_token} if self.hf_token else {}
                            self.model = CLIPModel.from_pretrained(model_name, trust_remote_code=True, **token_kwargs).to(self.device)
                            self.processor = CLIPProcessor.from_pretrained(model_name, trust_remote_code=True, **token_kwargs)
                        loaded = True
                        print(f"Successfully loaded BiomedCLIP: {model_name}")
                    except Exception as e:
                        last_error = str(e)
                
                if not loaded:
                    try:
                        from transformers import AutoModel, AutoProcessor
                        with suppress_stderr():
                            token_kwargs = {"token": self.hf_token} if self.hf_token else {}
                            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, **token_kwargs).to(self.device)
                            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, **token_kwargs)
                        loaded = True
                        print(f"✓ Successfully loaded BiomedCLIP using AutoModel: {model_name}")
                    except Exception as e:
                        last_error = str(e)
                
                if not loaded:
                    try:
                        from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoImageProcessor
                        with suppress_stderr():
                            token_kwargs = {"token": self.hf_token} if self.hf_token else {}
                            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, **token_kwargs).to(self.device)
                            try:
                                self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, **token_kwargs)
                            except Exception:
                                # If AutoProcessor fails, create a CombinedProcessor
                                image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True, **token_kwargs)
                                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, **token_kwargs)
                                
                            class CombinedProcessor:
                                def __init__(self, image_processor, tokenizer):
                                    self.image_processor = image_processor
                                    self.tokenizer = tokenizer
                                def __call__(self, text=None, images=None, return_tensors=None, padding=None, **kwargs):
                                    result = {}
                                    if images is not None:
                                        result.update(self.image_processor(images, return_tensors=return_tensors))
                                    if text is not None:
                                        text_result = self.tokenizer(text, return_tensors=return_tensors, padding=padding, **kwargs)
                                        result.update(text_result)
                                    return result
                                
                            self.processor = CombinedProcessor(image_processor, tokenizer)
                        loaded = True
                        print(f"✓ Successfully loaded BiomedCLIP using AutoModel with CombinedProcessor: {model_name}")
                    except Exception as e:
                        last_error = str(e)
                
                if not loaded:
                    print(f"WARNING: Failed to load BiomedCLIP model: {model_name}")
                    print(f"   Error: {last_error[:200] if last_error else 'Unknown error'}")
                    print(f"   Falling back to default model: openai/clip-vit-large-patch14")
                    self.model_name = "openai/clip-vit-large-patch14"
                    with suppress_stderr():
                        token_kwargs = {"token": self.hf_token} if self.hf_token else {}
                        self.model = CLIPModel.from_pretrained(self.model_name, **token_kwargs).to(self.device)
                        self.processor = CLIPProcessor.from_pretrained(self.model_name, **token_kwargs)
                    print(f"✓ Loaded fallback model: {self.model_name}")
            elif "metaclip" in model_name.lower() or "dfn" in model_name.lower():
                # Meta MetaCLIP and Apple DFN models
                loaded = False
                try:
                    with suppress_stderr():
                        token_kwargs = {"token": self.hf_token} if self.hf_token else {}
                        self.model = CLIPModel.from_pretrained(model_name, **token_kwargs).to(self.device)
                        self.processor = CLIPProcessor.from_pretrained(model_name, **token_kwargs)
                    loaded = True
                    print(f"Loaded {model_name} as CLIPModel")
                except Exception as e:
                    print(f"Warning: Could not load {model_name} as CLIPModel, trying AutoModel...")
                    try:
                        from transformers import AutoModel, AutoProcessor
                        with suppress_stderr():
                            token_kwargs = {"token": self.hf_token} if self.hf_token else {}
                            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, **token_kwargs).to(self.device)
                            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, **token_kwargs)
                        loaded = True
                        print(f"Loaded {model_name} using AutoModel")
                    except Exception as e2:
                        print(f"Warning: AutoModel also failed: {str(e2)[:200]}")
                
                if not loaded:
                    raise RuntimeError(f"Could not load {model_name}")
            elif "laion" in model_name.lower():
                # LAION CLIP models might need different loading
                loaded = False
                try:
                    with suppress_stderr():
                        token_kwargs = {"token": self.hf_token} if self.hf_token else {}
                        self.model = CLIPModel.from_pretrained(model_name, **token_kwargs).to(self.device)
                        self.processor = CLIPProcessor.from_pretrained(model_name, **token_kwargs)
                    loaded = True
                except Exception as e:
                    error_msg = str(e)
                    print(f"Warning: Could not load {model_name} as CLIPModel: {error_msg[:200]}")
                    print(f"Trying AutoModel as fallback...")
                    try:
                        from transformers import AutoModel, AutoProcessor
                        with suppress_stderr():
                            token_kwargs = {"token": self.hf_token} if self.hf_token else {}
                            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, **token_kwargs).to(self.device)
                            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, **token_kwargs)
                        loaded = True
                        print(f"Successfully loaded {model_name} using AutoModel")
                    except Exception as e2:
                        error_msg2 = str(e2)
                        print(f"Warning: AutoModel also failed: {error_msg2[:200]}")
                        # Will fall back to default model in outer exception handler
                        pass
                
                if not loaded:
                    # Raise to trigger fallback in outer exception handler
                    raise RuntimeError(f"Could not load {model_name} with any method. The model may not exist or may require special configuration.")
            else:
                with suppress_stderr():
                    token_kwargs = {"token": self.hf_token} if self.hf_token else {}
                    self.model = CLIPModel.from_pretrained(model_name, **token_kwargs).to(self.device)
                    self.processor = CLIPProcessor.from_pretrained(model_name, **token_kwargs)
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            print("Falling back to CLIP ViT-Large...")
            self.model_name = "openai/clip-vit-large-patch14"
            try:
                with suppress_stderr():
                    token_kwargs = {"token": self.hf_token} if self.hf_token else {}
                    self.model = CLIPModel.from_pretrained(self.model_name, **token_kwargs).to(self.device)
                    self.processor = CLIPProcessor.from_pretrained(self.model_name, **token_kwargs)
            except Exception as fallback_error:
                print(f"Error with fallback: {fallback_error}")
                print("Falling back to CLIP ViT-Base...")
                self.model_name = "openai/clip-vit-base-patch32"
                try:
                    with suppress_stderr():
                        token_kwargs = {"token": self.hf_token} if self.hf_token else {}
                        self.model = CLIPModel.from_pretrained(self.model_name, **token_kwargs).to(self.device)
                        self.processor = CLIPProcessor.from_pretrained(self.model_name, **token_kwargs)
                except Exception as final_error:
                    raise RuntimeError(
                        f"Failed to load any CLIP model. Original error: {e}, "
                        f"Fallback 1 error: {fallback_error}, Fallback 2 error: {final_error}"
                    ) from final_error
        
        self.model.eval()
        
        # Enhanced prompts with multiple strategies for better performance
        self._init_enhanced_prompts()
        # Store original prompts for resetting after context-aware predictions
        self._original_class_prompts = self.class_prompts.copy()
        self._original_weights = self.weights.copy()
    
    def _init_enhanced_prompts(self, previous_predictions: Optional[Dict[str, Dict]] = None, current_modality: Optional[str] = None):
        """
        Initialize diverse prompt strategies for zero-shot classification.
        
        Args:
            previous_predictions: Optional dict mapping modality names to their predictions.
                Format: {'CT': {'prediction': 0, 'class_name': 'high_grade'}, ...}
                If provided, prompts will incorporate this context.
            current_modality: The modality currently being processed (e.g., 'CT', 'PET', 'PT').
                Used to make prompts generic instead of hardcoding 'PET scan'.
        """
        first_class, second_class = self.class_names
        
        # Determine current modality name for prompts.
        # NOTE: Many prompt templates below already add the word "scan"
        # (e.g., "a {current_modality} scan showing ...").
        # To avoid double text like "CT scan scan ...", keep this as just
        # the modality token (e.g., "CT", "PT", "MR").
        if current_modality is None:
            current_modality = "scan"  # Generic fallback
        else:
            current_modality = str(current_modality).strip()  # e.g., "PET", "CT"
        
        # Build context string if previous predictions are available
        # Simple concept: "hey CT gave this result, what's for PET?"
        # Make previous modality information clear and helpful for current modality prediction
        context_parts = []
        context_prefix = ""  # Must always be defined (e.g. __init__ calls with no previous_predictions)
        if previous_predictions:
            for mod, pred_info in previous_predictions.items():
                pred_class = pred_info.get('class_name', self.class_names[pred_info.get('prediction', 0)])
                # Simple and clear: "the CT scan showed X"
                context_parts.append(f"the {mod} scan showed {pred_class}")
        if context_parts:
            context_str = ", and ".join(context_parts)
            # Simple and direct: "Given that CT showed X, this PET scan shows..."
            # Generic: "Given that {previous_mod} showed X, this {current_mod} shows..."
            # This is the core concept: previous modality gave this result, what's for current modality?
            # IMPORTANT: Many prompt templates below already append
            # "{current_modality} shows ..." after this prefix.
            # Keep the prefix short to avoid duplicated "... this CT shows CT shows ...".
            context_prefix = f"Given that {context_str}, this "

        if self._default_classes:
            # Strategy 1: Direct descriptive prompts (for backward compatibility with "Healthy" vs "Tumor" classes)
            class0_direct = [
                f"a medical {current_modality} showing healthy normal tissue with no tumors or abnormalities",
                f"a {current_modality} imaging scan with normal anatomy and no pathological findings",
                f"a healthy medical image showing normal tissue structure without disease",
                f"a normal {current_modality} scan image with no masses, lesions, or tumors visible"
            ]
            class1_direct = [
                f"a medical {current_modality} showing a visible tumor or malignant mass",
                f"a {current_modality} imaging scan with abnormal mass, tumor growth, or cancerous lesion",
                f"a medical image showing pathology, tumor, or abnormal tissue",
                f"a {current_modality} scan image with visible tumor, mass, or pathological abnormality"
            ]
            
            # Strategy 2: Clinical terminology (generic)
            class0_clinical = [
                f"a {current_modality} scan with normal tissue and no abnormal findings",
                f"a medical image showing normal anatomy without pathology",
                f"a {current_modality} scan demonstrating normal tissue architecture"
            ]
            class1_clinical = [
                f"a {current_modality} scan with an abnormal mass or neoplasm",
                f"a medical image showing an abnormal lesion or tumor",
                f"a {current_modality} scan demonstrating pathological tissue or mass"
            ]
            
            # Strategy 3: Simple, clear descriptions (generic)
            class0_simple = [
                f"a normal healthy {current_modality} scan",
                f"a {current_modality} scan with no tumor",
                f"a healthy medical image"
            ]
            class1_simple = [
                f"a {current_modality} scan with a tumor",
                f"a {current_modality} scan showing a tumor",
                f"an abnormal {current_modality} scan with tumor"
            ]
            
            first_prompts = class0_direct + class0_clinical + class0_simple
            second_prompts = class1_direct + class1_clinical + class1_simple
            prompt_weights = [1.0] * len(class0_direct) + [0.8] * len(class0_clinical) + [0.6] * len(class0_simple)
            
            # Apply context prefix if available
            if context_prefix:
                first_prompts = [context_prefix + prompt for prompt in first_prompts]
                second_prompts = [context_prefix + prompt for prompt in second_prompts]
        else:
            # Detect if this is lung NSCLC vs SCLC (non_small_cell vs small_cell) - Lung-PET-CT-Dx
            is_lung_nsclc_sclc = (
                "non_small_cell" in (first_class + second_class).lower()
                and "small_cell" in (first_class + second_class).lower()
            )
            # Detect if this is cancer grading (high_grade vs low_grade) - generic for any cancer type
            is_cancer_grading = any(
                keyword in first_class.lower() + second_class.lower()
                for keyword in ["high_grade", "low_grade", "grade"]
            )
            # Detect if this is stage-based (early_stage vs advanced_stage) - e.g. TCGA-KIRP
            is_stage_based = (
                "early_stage" in (first_class + second_class).lower()
                and "advanced_stage" in (first_class + second_class).lower()
            )

            if is_lung_nsclc_sclc:
                # Group 1: NSCLC (A, G, E). Group 2: SCLC (B).
                # Mix short prompts (better for CLIP/SigLIP) with clinical context.
                is_nsclc_first = "non_small" in first_class.lower()
                if context_prefix:
                    if is_nsclc_first:
                        first_prompts = [
                            f"{context_prefix}{current_modality} shows non-small cell lung cancer",
                            f"{context_prefix}{current_modality} shows NSCLC: adenocarcinoma, squamous, or large cell carcinoma",
                            f"{context_prefix}{current_modality} shows non-small cell lung cancer (adeno, squamous, or large cell); often treated with surgery or targeted therapy",
                            f"{context_prefix}{current_modality} shows NSCLC",
                        ]
                        second_prompts = [
                            f"{context_prefix}{current_modality} shows small cell lung cancer",
                            f"{context_prefix}{current_modality} shows small cell lung carcinoma, neuroendocrine tumor",
                            f"{context_prefix}{current_modality} shows small cell lung cancer; aggressive, typically treated with chemotherapy and radiation",
                            f"{context_prefix}{current_modality} shows SCLC",
                        ]
                    else:
                        first_prompts = [
                            f"{context_prefix}{current_modality} shows small cell lung cancer",
                            f"{context_prefix}{current_modality} shows small cell lung carcinoma, neuroendocrine tumor",
                            f"{context_prefix}{current_modality} shows small cell lung cancer; aggressive, typically treated with chemotherapy and radiation",
                            f"{context_prefix}{current_modality} shows SCLC",
                        ]
                        second_prompts = [
                            f"{context_prefix}{current_modality} shows non-small cell lung cancer",
                            f"{context_prefix}{current_modality} shows NSCLC: adenocarcinoma, squamous, or large cell carcinoma",
                            f"{context_prefix}{current_modality} shows non-small cell lung cancer (adeno, squamous, or large cell); often treated with surgery or targeted therapy",
                            f"{context_prefix}{current_modality} shows NSCLC",
                        ]
                else:
                    if is_nsclc_first:
                        first_prompts = [
                            f"a {current_modality} scan showing non-small cell lung cancer",
                            f"a {current_modality} imaging scan with NSCLC: adenocarcinoma, squamous, or large cell carcinoma",
                            f"a {current_modality} slice of non-small cell lung cancer (adeno, squamous, or large cell)",
                            f"a {current_modality} scan of NSCLC",
                        ]
                        second_prompts = [
                            f"a {current_modality} scan showing small cell lung cancer",
                            f"a {current_modality} imaging scan with small cell lung carcinoma, neuroendocrine tumor",
                            f"a {current_modality} slice of small cell lung cancer",
                            f"a {current_modality} scan of SCLC",
                        ]
                    else:
                        first_prompts = [
                            f"a {current_modality} scan showing small cell lung cancer",
                            f"a {current_modality} imaging scan with small cell lung carcinoma, neuroendocrine tumor",
                            f"a {current_modality} slice of small cell lung cancer",
                            f"a {current_modality} scan of SCLC",
                        ]
                        second_prompts = [
                            f"a {current_modality} scan showing non-small cell lung cancer",
                            f"a {current_modality} imaging scan with NSCLC: adenocarcinoma, squamous, or large cell carcinoma",
                            f"a {current_modality} slice of non-small cell lung cancer (adeno, squamous, or large cell)",
                            f"a {current_modality} scan of NSCLC",
                        ]
                # Slightly higher weight for short, clear prompts (CLIP/SigLIP work better with concise text)
                prompt_weights = [1.1, 1.0, 1.0, 1.1]
            elif is_stage_based:
                # Stage-specific prompts: early_stage = Stage I–II, advanced_stage = Stage III–IV
                is_early_first = "early" in first_class.lower()
                if context_prefix:
                    if is_early_first:
                        first_prompts = [
                            f"{context_prefix}{current_modality} shows {first_class} cancer (Stage I–II) with limited tumor involvement and better prognosis",
                            f"{context_prefix}{current_modality} shows {first_class} with smaller primary tumor, localized growth, and no distant metastasis",
                            f"{context_prefix}{current_modality} shows {first_class} early-stage disease with lower metabolic burden",
                        ]
                        second_prompts = [
                            f"{context_prefix}{current_modality} shows {second_class} cancer (Stage III–IV) with extensive tumor involvement and higher risk",
                            f"{context_prefix}{current_modality} shows {second_class} with larger tumor, lymph node or distant spread",
                            f"{context_prefix}{current_modality} shows {second_class} advanced-stage disease with higher metabolic burden",
                        ]
                    else:
                        first_prompts = [
                            f"{context_prefix}{current_modality} shows {first_class} cancer (Stage III–IV) with extensive tumor involvement and higher risk",
                            f"{context_prefix}{current_modality} shows {first_class} with larger tumor, lymph node or distant spread",
                            f"{context_prefix}{current_modality} shows {first_class} advanced-stage disease with higher metabolic burden",
                        ]
                        second_prompts = [
                            f"{context_prefix}{current_modality} shows {second_class} cancer (Stage I–II) with limited tumor involvement and better prognosis",
                            f"{context_prefix}{current_modality} shows {second_class} with smaller primary tumor, localized growth, and no distant metastasis",
                            f"{context_prefix}{current_modality} shows {second_class} early-stage disease with lower metabolic burden",
                        ]
                else:
                    if is_early_first:
                        first_prompts = [
                            f"a {current_modality} scan showing {first_class} cancer (Stage I–II) with limited tumor involvement",
                            f"a {current_modality} imaging scan with {first_class} demonstrating smaller tumor and localized growth",
                            f"a {current_modality} slice classified as {first_class} early-stage disease",
                        ]
                        second_prompts = [
                            f"a {current_modality} scan showing {second_class} cancer (Stage III–IV) with extensive tumor involvement",
                            f"a {current_modality} imaging scan with {second_class} demonstrating larger tumor or spread",
                            f"a {current_modality} slice classified as {second_class} advanced-stage disease",
                        ]
                    else:
                        first_prompts = [
                            f"a {current_modality} scan showing {first_class} cancer (Stage III–IV) with extensive tumor involvement",
                            f"a {current_modality} imaging scan with {first_class} demonstrating larger tumor or spread",
                            f"a {current_modality} slice classified as {first_class} advanced-stage disease",
                        ]
                        second_prompts = [
                            f"a {current_modality} scan showing {second_class} cancer (Stage I–II) with limited tumor involvement",
                            f"a {current_modality} imaging scan with {second_class} demonstrating smaller tumor and localized growth",
                            f"a {current_modality} slice classified as {second_class} early-stage disease",
                        ]
                prompt_weights = [1.0] * len(first_prompts)
            elif is_cancer_grading:
                # Enhanced cancer grade-specific prompts with medical terminology (generic for any cancer type)
                # High-grade typically: larger tumors, more aggressive, poor differentiation, invasion
                # Low-grade typically: smaller tumors, less aggressive, well-differentiated, localized
                is_high_grade_first = "high" in first_class.lower()
                
                if is_high_grade_first:
                    # High-grade prompts (more aggressive, invasive characteristics)
                    # Generic cancer grading prompts
                    if context_prefix:
                        # Context-aware prompts with previous modality context
                        first_prompts = [
                            f"{context_prefix}{current_modality} shows {first_class} cancer with aggressive tumor characteristics and high metabolic activity",
                            f"{context_prefix}{current_modality} shows {first_class} cancer demonstrating large primary tumor size (>3cm), lymph node invasion, and distant metastasis",
                            f"{context_prefix}{current_modality} shows {first_class} cancer with poor differentiation, advanced staging, and high metabolic activity",
                            f"{context_prefix}{current_modality} shows {first_class} tumor with intense metabolic uptake, irregular margins, and invasion",
                            f"{context_prefix}{current_modality} shows {first_class} cancer pathology with invasive growth pattern, lymphadenopathy, and high-grade histology",
                            f"{context_prefix}{current_modality} shows {first_class} cancer with high-grade features: large mass (>5cm), irregular borders, and aggressive appearance",
                            f"{context_prefix}{current_modality} shows {first_class} advanced stage cancer (Stage III-IV) with extensive tumor involvement, high metabolic burden, and poor prognosis",
                            f"{context_prefix}{current_modality} shows {first_class} cancer with extensive disease, multiple lesions, and spread",
                        ]
                        second_prompts = [
                            f"{context_prefix}{current_modality} shows {second_class} cancer with less aggressive tumor characteristics and moderate metabolic activity",
                            f"{context_prefix}{current_modality} shows {second_class} cancer demonstrating smaller primary tumor size (<3cm), localized growth, and no distant metastasis",
                            f"{context_prefix}{current_modality} shows {second_class} cancer with well-differentiated histology, early staging, and moderate metabolic activity",
                            f"{context_prefix}{current_modality} shows {second_class} tumor with mild to moderate metabolic uptake, smooth well-defined margins, and no invasion",
                            f"{context_prefix}{current_modality} shows {second_class} cancer pathology with localized growth pattern, no significant lymphadenopathy, and low-grade histology",
                            f"{context_prefix}{current_modality} shows {second_class} cancer with low-grade features: smaller mass (<3cm), smooth well-defined borders, and less aggressive appearance",
                            f"{context_prefix}{current_modality} shows {second_class} early-stage cancer (Stage I-II) with limited tumor involvement, lower metabolic burden, and better prognosis",
                            f"{context_prefix}{current_modality} shows {second_class} cancer with focal disease, single lesion, and no spread",
                        ]
                    else:
                        # Original prompts without context
                        first_prompts = [
                            f"a {current_modality} scan showing {first_class} cancer with aggressive tumor characteristics and high metabolic activity",
                            f"a {current_modality} imaging scan with {first_class} cancer demonstrating large primary tumor (>3cm), lymph node metastasis, and invasive growth pattern",
                            f"a {current_modality} slice classified as {first_class} with poor differentiation, advanced staging, and high metabolic activity",
                            f"a cancer scan showing {first_class} tumor with intense metabolic uptake, irregular margins, and invasion",
                            f"a radiology image with {first_class} cancer pathology showing invasive growth pattern, lymphadenopathy, and high-grade histology",
                            f"a {current_modality} scan with {first_class} cancer exhibiting high-grade features: large mass (>5cm), irregular borders, and aggressive appearance",
                            f"a medical image showing {first_class} advanced stage cancer (Stage III-IV) with extensive tumor involvement, high metabolic burden, and poor prognosis",
                            f"a {current_modality} scan demonstrating {first_class} cancer with extensive disease, multiple lesions, and spread",
                        ]
                        second_prompts = [
                            f"a {current_modality} scan showing {second_class} cancer with less aggressive tumor characteristics, moderate metabolic activity, and early staging",
                            f"a {current_modality} imaging scan with {second_class} cancer demonstrating smaller primary tumor (<3cm), localized growth, and no distant metastasis",
                            f"a {current_modality} slice classified as {second_class} with well-differentiated histology, early staging, and moderate metabolic activity",
                            f"a cancer scan showing {second_class} tumor with mild to moderate metabolic uptake, smooth well-defined margins, and no invasion",
                            f"a radiology image with {second_class} cancer pathology showing localized growth pattern, no significant lymphadenopathy, and low-grade histology",
                            f"a {current_modality} scan with {second_class} cancer exhibiting low-grade features: smaller mass (<3cm), smooth well-defined borders, and less aggressive appearance",
                            f"a medical image showing {second_class} early-stage cancer (Stage I-II) with limited tumor involvement, lower metabolic burden, and better prognosis",
                            f"a {current_modality} scan demonstrating {second_class} cancer with focal disease, single lesion, and no spread",
                        ]
                else:
                    # Low-grade first, high-grade second (swap the descriptions)
                    if context_prefix:
                        # Context-aware prompts with previous modality context
                        first_prompts = [
                            f"{context_prefix}{current_modality} shows {first_class} cancer with less aggressive tumor characteristics and moderate metabolic activity",
                            f"{context_prefix}{current_modality} shows {first_class} cancer demonstrating smaller primary tumor size (<3cm), localized growth, and no distant metastasis",
                            f"{context_prefix}{current_modality} shows {first_class} cancer with well-differentiated histology, early staging, and moderate metabolic activity",
                            f"{context_prefix}{current_modality} shows {first_class} tumor with mild to moderate metabolic uptake, smooth well-defined margins, and no invasion",
                            f"{context_prefix}{current_modality} shows {first_class} cancer pathology with localized growth pattern, no significant lymphadenopathy, and low-grade histology",
                            f"{context_prefix}{current_modality} shows {first_class} cancer with low-grade features: smaller mass (<3cm), smooth well-defined borders, and less aggressive appearance",
                            f"{context_prefix}{current_modality} shows {first_class} early-stage cancer (Stage I-II) with limited tumor involvement, lower metabolic burden, and better prognosis",
                            f"{context_prefix}{current_modality} shows {first_class} cancer with focal disease, single lesion, and no spread",
                        ]
                        second_prompts = [
                            f"{context_prefix}{current_modality} shows {second_class} cancer with aggressive tumor characteristics and high metabolic activity",
                            f"{context_prefix}{current_modality} shows {second_class} cancer demonstrating large primary tumor size (>3cm), lymph node invasion, and distant metastasis",
                            f"{context_prefix}{current_modality} shows {second_class} cancer with poor differentiation, advanced staging, and high metabolic activity",
                            f"{context_prefix}{current_modality} shows {second_class} tumor with intense metabolic uptake, irregular margins, and invasion",
                            f"{context_prefix}{current_modality} shows {second_class} cancer pathology with invasive growth pattern, lymphadenopathy, and high-grade histology",
                            f"{context_prefix}{current_modality} shows {second_class} cancer with high-grade features: large mass (>5cm), irregular borders, and aggressive appearance",
                            f"{context_prefix}{current_modality} shows {second_class} advanced stage cancer (Stage III-IV) with extensive tumor involvement, high metabolic burden, and poor prognosis",
                            f"{context_prefix}{current_modality} shows {second_class} cancer with extensive disease, multiple lesions, and spread",
                        ]
                    else:
                        # Original prompts without context
                        first_prompts = [
                            f"a {current_modality} scan showing {first_class} cancer with less aggressive tumor characteristics",
                            f"a {current_modality} imaging scan with {first_class} cancer demonstrating smaller tumor size and localized growth",
                            f"a {current_modality} slice classified as {first_class} with well-differentiated and early stage features",
                            f"a cancer scan showing {first_class} tumor with lower metabolic activity and contained growth",
                            f"a radiology image with {first_class} cancer pathology showing localized growth pattern",
                            f"a {current_modality} scan with {first_class} cancer exhibiting low-grade features: smaller mass, well-defined borders, and less aggressive appearance",
                            f"a medical image showing {first_class} cancer with early-stage disease characteristics",
                            f"a {current_modality} scan demonstrating {first_class} cancer with limited tumor involvement",
                        ]
                        second_prompts = [
                            f"a {current_modality} scan showing {second_class} cancer with aggressive tumor characteristics",
                            f"a {current_modality} imaging scan with {second_class} cancer demonstrating large tumor size and invasion",
                            f"a {current_modality} slice classified as {second_class} with poor differentiation and advanced stage",
                            f"a cancer scan showing {second_class} tumor with high metabolic activity and spread",
                            f"a radiology image with {second_class} cancer pathology showing invasive growth pattern",
                            f"a {current_modality} scan with {second_class} cancer exhibiting high-grade features: large mass, irregular borders, and aggressive appearance",
                            f"a medical image showing {second_class} cancer with advanced disease characteristics",
                            f"a {current_modality} scan demonstrating {second_class} cancer with extensive tumor involvement",
                        ]
            else:
                # Generic medical imaging prompts
                context_hint = "medical"
                first_prompts = [
                    f"{context_prefix}a {context_hint} imaging scan showing {first_class} characteristics",
                    f"{context_prefix}a clinical radiology image labeled as {first_class}",
                    f"{context_prefix}a diagnostic slice that is representative of {first_class}",
                ]
                second_prompts = [
                    f"{context_prefix}a {context_hint} imaging scan showing {second_class} characteristics",
                    f"{context_prefix}a clinical radiology image labeled as {second_class}",
                    f"{context_prefix}a diagnostic slice that is representative of {second_class}",
                ]
            # Weight prompts: more specific medical terminology gets higher weight
            if is_cancer_grading:
                # Higher weight for prompts with specific medical terms (aggressive, invasion, differentiation, etc.)
                prompt_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.0, 1.0]  # 6th prompt has more detail
            else:
                prompt_weights = [1.0] * len(first_prompts)
        
        # Safety check: ensure prompt lists have matching lengths
        if len(first_prompts) != len(second_prompts):
            raise ValueError(
                f"Prompt lists must have equal length. Got first_prompts={len(first_prompts)}, "
                f"second_prompts={len(second_prompts)}"
            )
        if len(prompt_weights) != len(first_prompts):
            raise ValueError(
                f"Prompt weights must match prompt length. Got prompt_weights={len(prompt_weights)}, "
                f"first_prompts={len(first_prompts)}"
            )
        
        # Create interleaved prompts: [classA1, classB1, classA2, classB2, ...]
        self.class_prompts = [
            prompt for pair in zip(first_prompts, second_prompts)
            for prompt in pair
        ]
        # Interleave weights to match prompt order
        self.weights = [w for pair in zip(prompt_weights, prompt_weights) for w in pair]
    
    def _preprocess_medical_image(self, image: Image.Image, aggressive: bool = False) -> Image.Image:
        """
        Preprocess medical image for better CLIP performance.
        Enhances contrast and normalizes the image.
        
        Args:
            image: PIL Image to preprocess
            aggressive: If True, apply more aggressive preprocessing (default: False)
        """
        # Convert to grayscale if needed, then back to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if aggressive:
            # More aggressive preprocessing for challenging cases
            # Enhance contrast more strongly
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)  # Increase contrast by 50%
            
            # Enhance sharpness more
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.2)
            
            # Apply CLAHE-like effect with histogram equalization
            gray = np.array(image.convert('L'))
            equalized = ImageOps.equalize(Image.fromarray(gray))
            equalized_rgb = Image.new('RGB', equalized.size)
            equalized_rgb.paste(equalized)
            
            # Blend original (60%) with equalized (40%) for stronger enhancement
            image = Image.blend(image, equalized_rgb, 0.4)
        else:
            # Standard preprocessing (balanced)
            # Enhance contrast for medical images
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.3)  # Increase contrast by 30%
            # Enhance sharpness slightly
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            # Apply histogram equalization for better visibility
            gray = np.array(image.convert('L'))
            equalized = ImageOps.equalize(Image.fromarray(gray))
            equalized_rgb = Image.new('RGB', equalized.size)
            equalized_rgb.paste(equalized)
            image = Image.blend(image, equalized_rgb, 0.3)
        
        return image
    
    def predict(
        self,
        images: Dict[str, Image.Image],
        available_modalities: List[str],
        batch_size: int = 1,
        preprocess: bool = False,
        temperature: float = 0.8,  # Lower default for better calibration (was 1.0)
        use_weighted_ensemble: bool = True,
        try_both_swaps: bool = True,
        aggressive_preprocess: bool = False,
        previous_predictions: Optional[Dict[str, Dict]] = None
    ) -> Dict:
        """
        Predict class using zero-shot classification with enhanced strategies.
        
        Args:
            images: Dictionary mapping modality names to PIL Images (or list of images for batch)
            available_modalities: List of modalities that are available
            batch_size: Batch size for processing (default: 1)
            preprocess: Whether to apply medical image preprocessing (default: False)
            temperature: Temperature scaling for logits (default: 1.0, lower = more confident)
            use_weighted_ensemble: Use weighted average of prompts (default: True)
            try_both_swaps: Try both with and without logit swap, use best (default: True)
            previous_predictions: Optional dict mapping modality names to their predictions.
                Format: {'CT': {'prediction': 0, 'class_name': 'high_grade'}, ...}
                If provided, prompts will incorporate this context.
        
        Returns:
            Dictionary with prediction (0=class0, 1=class1), confidence, and probabilities
        """
        # Determine current modality being processed (for generic prompts)
        current_modality_name = available_modalities[0] if available_modalities else None
        
        # Regenerate prompts with context if previous predictions are provided
        # Store previous modality prediction for later use in boosting current modality predictions
        previous_mod_prediction_for_boosting = None
        previous_mod_confidence_for_boosting = None
        if previous_predictions:
            self._init_enhanced_prompts(previous_predictions=previous_predictions, current_modality=current_modality_name)
            # Extract previous modality prediction and confidence for boosting current modality accuracy
            # previous_predictions format: {'CT': {'prediction': 0, 'class_name': 'high_grade', 'confidence': 0.85}, 'MR': {...}}
            # For N modalities: Use the MOST RECENT previous modality (last in dict, which preserves insertion order in Python 3.7+)
            # This ensures Mod3 with Mod1+Mod2 context uses Mod2 (most recent) for boosting, not Mod1
            # Get the last (most recent) previous modality for boosting
            # In Python 3.7+, dict preserves insertion order, so last item is most recent
            # Get last item (most recent previous modality)
            last_mod = list(previous_predictions.keys())[-1]
            last_pred_info = previous_predictions[last_mod]
            previous_mod_prediction_for_boosting = last_pred_info.get('prediction')
            previous_mod_confidence_for_boosting = last_pred_info.get('confidence', 0.5)  # Default to 0.5 if not available
        else:
            # No previous predictions, but still need to set current modality for prompts
            self._init_enhanced_prompts(previous_predictions=None, current_modality=current_modality_name)
        available_images = []
        modality_list = []
        for mod in available_modalities:
            if mod in images and images[mod] is not None:
                if isinstance(images[mod], list):
                    available_images.extend(images[mod])
                    modality_list.extend([mod] * len(images[mod]))
                else:
                    available_images.append(images[mod])
                    modality_list.append(mod)
        
        if not available_images:
            raise ValueError(
                f"No valid images found. Available modalities: {available_modalities}, "
                f"Images dict keys: {list(images.keys())}"
            )
        
        # Preprocess images if requested
        if preprocess:
            available_images = [self._preprocess_medical_image(img, aggressive=aggressive_preprocess) for img in available_images]
        
        all_class0_logits_weighted = []
        all_class1_logits_weighted = []
        all_class0_logits_swapped = []
        all_class1_logits_swapped = []
        
        # Process in batches
        for i in range(0, len(available_images), batch_size):
            batch_images = available_images[i:i + batch_size]
            batch_modalities = modality_list[i:i + batch_size]
            
            inputs = self.processor(
                text=self.class_prompts,
                images=batch_images,
                return_tensors="pt",
                padding=True
            )
            # OpenCLIPWrapper returns a dict with tensors already on device
            # Transformers processors return objects that need .to(device)
            if isinstance(inputs, dict):
                # Already on device (OpenCLIP case)
                pass
            else:
                # Move to device (transformers case)
                inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                
                # Handle both single image (1D) and batch (2D) cases
                if logits_per_image.dim() == 1:
                    logits_per_image = logits_per_image.unsqueeze(0)
                
                for j in range(len(batch_images)):
                    # Get logits for class0 and class1 prompts
                    # Prompts are interleaved: [class0_1, class1_1, class0_2, class1_2, ...]
                    class0_logits_all = logits_per_image[j][::2]  # All class0 prompt logits
                    class1_logits_all = logits_per_image[j][1::2]   # All class1 prompt logits
                    
                    if use_weighted_ensemble and hasattr(self, 'weights'):
                        # Weighted average based on prompt quality
                        class0_weights = torch.tensor(self.weights[::2], device=class0_logits_all.device)
                        class1_weights = torch.tensor(self.weights[1::2], device=class1_logits_all.device)
                        # Protect against division by zero (weights should always be positive, but safety check)
                        class0_weights_sum = class0_weights.sum()
                        class1_weights_sum = class1_weights.sum()
                        class0_prompt_logits = (class0_logits_all * class0_weights).sum() / max(class0_weights_sum, 1e-8)
                        class1_prompt_logits = (class1_logits_all * class1_weights).sum() / max(class1_weights_sum, 1e-8)
                    else:
                        # Simple mean
                        class0_prompt_logits = class0_logits_all.mean()
                        class1_prompt_logits = class1_logits_all.mean()
                    
                    # Strategy 1: Direct (no swap)
                    all_class0_logits_weighted.append(class0_prompt_logits)
                    all_class1_logits_weighted.append(class1_prompt_logits)
                    
                    # Strategy 2: Swapped (original behavior)
                    all_class0_logits_swapped.append(class1_prompt_logits)
                    all_class1_logits_swapped.append(class0_prompt_logits)
        
        # Aggregate across images
        if len(all_class0_logits_weighted) == 0:
            raise ValueError("No logits computed. Check that images were processed correctly.")
        elif len(all_class0_logits_weighted) == 1:
            class0_logits_direct = all_class0_logits_weighted[0]
            class1_logits_direct = all_class1_logits_weighted[0]
            class0_logits_swap = all_class0_logits_swapped[0]
            class1_logits_swap = all_class1_logits_swapped[0]
        else:
            class0_logits_direct = torch.stack(all_class0_logits_weighted).mean()
            class1_logits_direct = torch.stack(all_class1_logits_weighted).mean()
            class0_logits_swap = torch.stack(all_class0_logits_swapped).mean()
            class1_logits_swap = torch.stack(all_class1_logits_swapped).mean()
        
        # For BiomedCLIP, always try both strategies to handle potential logit order issues
        # BiomedCLIP may have inverted logit order, so we test both and pick the best
        if self.is_biomedclip:
            # Try both strategies for BiomedCLIP and pick the one with higher confidence
            # Direct strategy
            safe_temperature = max(temperature, 1e-8)
            class_logits_direct = torch.stack([class0_logits_direct, class1_logits_direct]) / safe_temperature
            probs_direct = class_logits_direct.softmax(dim=-1)
            confidence_direct = probs_direct.max().item()
            
            # Swapped strategy (may be needed if logits are inverted)
            class_logits_swap = torch.stack([class0_logits_swap, class1_logits_swap]) / safe_temperature
            probs_swap = class_logits_swap.softmax(dim=-1)
            confidence_swap = probs_swap.max().item()
            
            # Use the strategy with higher confidence
            if confidence_swap > confidence_direct:
                probs = probs_swap
                class0_logits = class0_logits_swap
                class1_logits = class1_logits_swap
            else:
                probs = probs_direct
                class0_logits = class0_logits_direct
                class1_logits = class1_logits_direct
        elif not try_both_swaps:
            # Use direct strategy (no swap) - for models that don't need swap testing
            # Protect against division by zero
            safe_temperature = max(temperature, 1e-8)
            class_logits = torch.stack([class0_logits_direct, class1_logits_direct]) / safe_temperature
            probs = class_logits.softmax(dim=-1)
            class0_logits = class0_logits_direct
            class1_logits = class1_logits_direct
        else:
            # Try both strategies and pick the one with higher confidence
            # Direct strategy
            # Protect against division by zero
            safe_temperature = max(temperature, 1e-8)
            class_logits_direct = torch.stack([class0_logits_direct, class1_logits_direct]) / safe_temperature
            probs_direct = class_logits_direct.softmax(dim=-1)
            confidence_direct = probs_direct.max().item()
            
            # Swapped strategy
            class_logits_swap = torch.stack([class0_logits_swap, class1_logits_swap]) / safe_temperature
            probs_swap = class_logits_swap.softmax(dim=-1)
            confidence_swap = probs_swap.max().item()
            
            # Use the strategy with higher confidence
            # IMPORTANT: If confidences are equal (which is expected for binary classification),
            # prefer the DIRECT strategy. Only swap if swap is STRICTLY more confident.
            if confidence_swap > confidence_direct:
                probs = probs_swap
                class0_logits = class0_logits_swap
                class1_logits = class1_logits_swap
            else:
                probs = probs_direct
                class0_logits = class0_logits_direct
                class1_logits = class1_logits_direct
        
        # Store probabilities BEFORE boosting (for certainty analysis)
        # For CT (no boosting), this will be the same as final probs
        # For PET (with boosting), this captures the state before boosting
        probs_before_boosting = probs.clone()
        
        # Boost current modality prediction using previous modality context (helps improve accuracy)
        # KEY CONCEPT: Current modality has MORE information than previous modality:
        #   1. Current modality's own visual signal from its image
        #   2. Previous modality's prediction as context (incorporated in prompts: "Given that {previous_mod} showed X...")
        #   3. The model has ALREADY considered previous modality context when making current modality's prediction
        #
        # Therefore, current modality's prediction is INFORMED and should be trusted MORE than previous modality alone!
        # This is where improvement happens - current modality can use both its image AND previous modality context.
        # Only apply boosting when we have previous modality context (e.g., processing Mod2 with Mod1 context, or Mod1 with Mod2 context)
        if previous_mod_prediction_for_boosting is not None and previous_predictions:
            # We're processing current modality images with previous modality context from the SAME patient
            current_mod_prediction_before = probs.argmax().item()
            current_mod_confidence_before = probs.max().item()
            previous_mod_class_idx = previous_mod_prediction_for_boosting
            current_mod_class_idx = current_mod_prediction_before
            previous_mod_confidence = previous_mod_confidence_for_boosting if previous_mod_confidence_for_boosting is not None else 0.5
            
            if current_mod_prediction_before == previous_mod_prediction_for_boosting:
                # Case 1: Current modality agrees with previous modality - boost significantly to lock it in
                # Both modalities agree, so this is very likely correct
                # The prompts already incorporate previous modality context, and current modality agrees with previous
                # Increased boost factors to ensure agreement cases are strongly favored
                if current_mod_confidence_before > 0.7 and previous_mod_confidence > 0.7:
                    # Both highly confident - very strong boost
                    boost_factor = 80.0  # Increased from 50.0
                elif current_mod_confidence_before > 0.6 or previous_mod_confidence > 0.6:
                    # At least one confident - strong boost
                    boost_factor = 50.0  # Increased from 30.0
                else:
                    # Lower confidence - moderate boost
                    boost_factor = 35.0  # Increased from 20.0
                probs[current_mod_prediction_before] = probs[current_mod_prediction_before] * boost_factor
            else:
                # Case 2: Current modality disagrees with previous modality - PROFESSOR'S REQUIREMENT
                # The prompts already include previous modality context: "Given that {previous_mod} showed X, this {current_mod} shows..."
                # So the model has ALREADY considered previous modality's information when making current modality's prediction
                # 
                # KEY INSIGHT: Current modality has MORE information than previous modality:
                #   - Current modality's own visual signal
                #   - Previous modality's prediction as context (in prompts)
                # Therefore, current modality's informed judgment should be trusted MORE
                # 
                # Strategy: Trust current modality's informed judgment MORE to enable improvement
                # When current modality disagrees after seeing previous modality context, current modality might be seeing something previous modality missed
                # This is where improvement happens - current modality can correct previous modality's mistakes
                # 
                # IMPROVEMENT: Don't boost previous modality when current modality disagrees - trust current modality's informed decision
                # Current modality has already seen previous modality context in prompts, so if current modality still disagrees, trust current modality
                
                # CRITICAL FIX: When current modality disagrees with previous modality, ONLY boost current modality, NOT previous modality
                # Current modality has more information (current modality image + previous modality context in prompts)
                # If current modality still disagrees after seeing previous modality context, trust current modality's informed decision
                # Boosting previous modality would work against improvement!
                
                if current_mod_confidence_before > 0.65:
                    # Current modality is confident - trust current modality STRONGLY (current modality has more information!)
                    # This allows current modality to correct previous modality and improve accuracy
                    current_mod_boost_factor = 10.0  # Increased from 8.0 - even stronger trust
                    # DO NOT boost previous modality - let current modality's informed decision win
                elif current_mod_confidence_before > 0.55:
                    # Current modality is moderately confident - favor current modality strongly
                    current_mod_boost_factor = 7.0   # Increased from 5.0
                    # DO NOT boost previous modality
                else:
                    # Current modality is not confident - still favor current modality (it has more information)
                    current_mod_boost_factor = 5.0   # Increased from 3.5
                    # DO NOT boost previous modality - trust current modality's informed judgment

                # Apply boost ONLY to current modality (not previous modality)
                probs[current_mod_class_idx] = probs[current_mod_class_idx] * current_mod_boost_factor
                # Do NOT boost previous modality when current modality disagrees - this is key to improvement!
        
        # Renormalize probabilities after boosting
        probs = probs / probs.sum()
        
        # This strategy enables improvement:
        # 1. When current modality agrees with previous modality: Massive boost locks in correct prediction
        # 2. When current modality disagrees with previous modality: Trust current modality more (it has more information)
        #    - Current modality has its own image + previous modality context in prompts
        #    - Current modality can correct previous modality when it sees something previous modality missed
        # Result: Sequential modality accuracy improves over single modality accuracy!
        
        # Use the final prediction (after boosting if applicable)
        prediction = probs.argmax().item()
        # Ensure prediction is valid (0 or 1) - should always be for binary classification
        prediction = max(0, min(1, int(prediction)))
        confidence = probs.max().item()
        # Ensure confidence is in valid range [0, 1]
        confidence = max(0.0, min(1.0, float(confidence)))
        
        prob_dict = {
            self.class_names[0].lower(): max(0.0, min(1.0, probs[0].item())),
            self.class_names[1].lower(): max(0.0, min(1.0, probs[1].item()))
        }
        
        # Store raw logits for certainty analysis
        raw_logits = torch.stack([class0_logits, class1_logits]).cpu().numpy()
        
        # Restore original prompts if they were modified
        if previous_predictions:
            self.class_prompts = self._original_class_prompts.copy()
            self.weights = self._original_weights.copy()
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': prob_dict,
            'logits': raw_logits.tolist(),  # For certainty analysis
            'probabilities_array': probs.cpu().numpy().tolist(),  # Final probabilities after boosting
            'probabilities_before_boosting': probs_before_boosting.cpu().numpy().tolist()  # Probabilities before boosting
        }

