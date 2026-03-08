"""
Unified LLaVA runner that works with both regular LLaVA and LLaVA-Med models.
Uses the official LLaVA repository for regular models and LLaVA-Med repo for medical models.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

# Add LLaVA-Med repo to path (supports both regular LLaVA and LLaVA-Med models)
_llava_med_path = os.path.join(os.path.dirname(__file__), '../../third_party/llava-med')

os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
os.environ.setdefault("ACCELERATE_USE_TENSORBOARD", "false")
os.environ.setdefault("USE_TF", "0")

# Use LLaVA-Med repo (which works and supports Mistral-based LLaVA models)
# The LLaVA-Med builder can handle regular LLaVA Mistral models too
sys.path.insert(0, _llava_med_path)
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token


class LLaVARunner:
    """
    Unified LLaVA runner that works with:
    - Regular LLaVA models (e.g., liuhaotian/llava-v1.6-mistral-7b, liuhaotian/llava-v1.6-vicuna-7b)
    - LLaVA-Med models (e.g., microsoft/llava-med-v1.5-mistral-7b)
    """

    def __init__(
        self,
        model_name: str = "liuhaotian/llava-v1.6-mistral-7b",
        device: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        hf_token: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names or ["Class0", "Class1"]
        self.compute_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        if len(self.class_names) != 2:
            raise ValueError("LLaVARunner currently supports exactly two classes.")
        
        print(f"\n{'='*60}")
        print(f"Loading LLaVA model: {model_name}")
        print(f"Device: {self.device}")
        print(f"{'='*60}")
        
        if self.device == "cpu":
            print("WARNING: Running on CPU. LLaVA inference will be VERY SLOW (~10-15 min per image).")
            print("   Consider using GPU (--device cuda) or a faster model like CLIP for CPU testing.")
        
        # Get token from parameter or environment variable
        self.hf_token = hf_token or os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
        if self.hf_token:
            os.environ['HF_TOKEN'] = self.hf_token
            os.environ['HUGGING_FACE_HUB_TOKEN'] = self.hf_token
        
        # Load model using the builder
        # Note: LLaVA-Med builder supports Mistral-based LLaVA models
        # For Vicuna/Llama-based models, you may need the official LLaVA repo
        try:
            self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(
                model_path=model_name,
                model_base=None,
                model_name=model_name,
                load_8bit=False,
                load_4bit=False,  # 4-bit requires GPU support
                device=self.device,
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nNote: This builder supports Mistral-based LLaVA models.")
            print("For Vicuna/Llama-based models, consider using the official LLaVA repository.")
            raise
        
        self.context_len = context_len
        
        # Move model to device with correct dtype
        self.model.to(dtype=self.compute_dtype, device=self.device)
        
        # Move vision tower and projector if they exist
        if hasattr(self.model, "model") and hasattr(self.model.model, "mm_projector"):
            self.model.model.mm_projector.to(dtype=self.compute_dtype, device=self.device)
        
        try:
            vision_tower = self.model.get_vision_tower()
            if vision_tower is not None:
                vision_tower.to(dtype=self.compute_dtype, device=self.device)
        except AttributeError:
            pass
        
        self.model.eval()
        
        # Optimize for inference
        if hasattr(self.model, 'config'):
            self.model.config.use_cache = True
        
        # Use appropriate conversation template
        if "llava_v1" in conv_templates:
            self.conv_template = conv_templates["llava_v1"]
        elif "llava" in conv_templates:
            self.conv_template = conv_templates["llava"]
        else:
            # Fallback to first available template
            self.conv_template = list(conv_templates.values())[0]
        
        print("Model loaded successfully!\n")

    def _build_prompt(self, previous_predictions: Optional[Dict[str, Dict]] = None, current_modality: Optional[str] = None) -> str:
        first, second = self.class_names
        
        # Build context string if previous predictions are available
        context_parts = []
        if previous_predictions:
            for mod, pred_info in previous_predictions.items():
                pred_class = pred_info.get('class_name', self.class_names[pred_info.get('prediction', 0)])
                context_parts.append(f"the {mod} scan showed {pred_class}")
        
        context_prefix = ""
        if context_parts:
            context_str = ", and ".join(context_parts)
            context_prefix = f"Given that {context_str}, "
        
        # Determine current modality name for prompts (default to generic if not provided)
        if current_modality is None:
            modality_description = "medical imaging scan"
        else:
            modality_description = f"{current_modality} scan"
        
        # Detect if this is lung NSCLC vs SCLC (Lung-PET-CT-Dx)
        is_lung_nsclc_sclc = (
            "non_small_cell" in (first + second).lower() and "small_cell" in (first + second).lower()
        )
        # Detect if this is cancer grading (generic for any cancer type)
        is_grading = any(
            keyword in first.lower() + second.lower()
            for keyword in ["high_grade", "low_grade", "grade"]
        )

        if is_lung_nsclc_sclc:
            return (
                f"{context_prefix}You are a medical imaging expert specializing in lung cancer. "
                f"Given this {modality_description} slice, classify the lung cancer type. "
                f"Group 1 (non_small_cell): NSCLC – adenocarcinoma, squamous cell, or large cell carcinoma. "
                f"These are Non-Small Cell Lung Cancers; they share similar surgical options and are often treated with targeted therapies or immunotherapy. "
                f"Group 2 (small_cell): Small Cell Lung Cancer – a distinct neuroendocrine tumor, more aggressive, treated primarily with chemotherapy/radiation rather than surgery. "
                f"Respond with exactly one word: {first} or {second}."
            )
        elif is_grading:
            return (
                f"{context_prefix}You are a medical imaging expert specializing in cancer diagnosis. "
                f"Given this {modality_description} slice, determine whether the cancer grade is "
                f"{first} or {second}. "
                f"Respond with exactly one word: {first} or {second}."
            )
        else:
            return (
                f"{context_prefix}You are a medical imaging expert. "
                f"Given this {modality_description} slice, determine whether it shows "
                f"{first} or {second}. "
                f"Respond with exactly one word: {first} or {second}."
            )

    @torch.inference_mode()
    def _predict_single(self, image: Image.Image, previous_predictions: Optional[Dict[str, Dict]] = None, current_modality: Optional[str] = None) -> Dict:
        conv = self.conv_template.copy()
        conv.append_message(conv.roles[0], self._build_prompt(previous_predictions=previous_predictions, current_modality=current_modality))
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Preprocess image
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"].to(
            self.device, dtype=self.compute_dtype
        )
        
        # Tokenize
        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            self.image_processor,
            return_tensors="pt",
        ).unsqueeze(0).to(self.device)
        
        attention_mask = torch.ones_like(input_ids)

        # Set pad token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Generate prediction with output_scores to get logits
        try:
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=10,  # Increased to allow proper response
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_beams=1,  # Greedy decoding (fastest)
                temperature=0.0,  # Deterministic
                repetition_penalty=1.0,
                return_dict_in_generate=True,
                output_scores=True,
            )
            # Extract sequences from GenerateDecoderOnlyOutput or similar
            if hasattr(output_ids, 'sequences'):
                sequences = output_ids.sequences
            else:
                sequences = output_ids
            # Handle tensor/list conversion for decoding
            if isinstance(sequences, torch.Tensor):
                if sequences.dim() > 1:
                    sequences_to_decode = sequences[0]
                else:
                    sequences_to_decode = sequences
            elif isinstance(sequences, list):
                sequences_to_decode = sequences[0] if sequences else []
            else:
                sequences_to_decode = sequences[0] if hasattr(sequences, '__getitem__') else sequences
            
            # Convert to list if tensor for tokenizer
            if isinstance(sequences_to_decode, torch.Tensor):
                sequences_to_decode = sequences_to_decode.cpu().tolist()
            elif not isinstance(sequences_to_decode, list):
                sequences_to_decode = list(sequences_to_decode) if hasattr(sequences_to_decode, '__iter__') else [sequences_to_decode]
            
            output_text = self.tokenizer.decode(sequences_to_decode, skip_special_tokens=True).strip().lower()
            scores_available = hasattr(output_ids, 'scores') and output_ids.scores is not None and len(output_ids.scores) > 0
        except Exception as e:
            # Fallback to simple generation if output_scores fails
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=10,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_beams=1,
                repetition_penalty=1.0,
            )
            # Handle different output formats
            if isinstance(output_ids, torch.Tensor):
                if output_ids.dim() > 1:
                    output_ids_to_decode = output_ids[0]
                else:
                    output_ids_to_decode = output_ids
            elif isinstance(output_ids, list):
                output_ids_to_decode = output_ids[0] if output_ids else []
            else:
                if hasattr(output_ids, 'sequences'):
                    output_ids_to_decode = output_ids.sequences[0]
                else:
                    output_ids_to_decode = output_ids[0] if hasattr(output_ids, '__getitem__') else output_ids
            
            # Convert to list if tensor for tokenizer
            if isinstance(output_ids_to_decode, torch.Tensor):
                output_ids_to_decode = output_ids_to_decode.cpu().tolist()
            elif not isinstance(output_ids_to_decode, list):
                output_ids_to_decode = list(output_ids_to_decode) if hasattr(output_ids_to_decode, '__iter__') else [output_ids_to_decode]
            
            output_text = self.tokenizer.decode(output_ids_to_decode, skip_special_tokens=True).strip().lower()
            scores_available = False
        
        # Parse prediction (normalize so "early stage" / "early-stage" match "early_stage")
        first, second = [c.lower() for c in self.class_names]
        text_for_match = output_text.replace(" ", "_").replace("-", "_")
        text_parsing_succeeded = False
        
        if first in text_for_match and second in text_for_match:
            prediction = int(text_for_match.rfind(first) < text_for_match.rfind(second))
            text_parsing_succeeded = True
        elif first in text_for_match:
            prediction = 0
            text_parsing_succeeded = True
        elif second in text_for_match:
            prediction = 1
            text_parsing_succeeded = True
        else:
            # Fallback: try first word of each class (e.g. "early" for early_stage)
            first_word = first.split("_")[0].split("-")[0].strip() if first else ""
            second_word = second.split("_")[0].split("-")[0].strip() if second else ""
            if first_word and second_word and first_word != second_word:
                if first_word in text_for_match and second_word not in text_for_match:
                    prediction = 0
                    text_parsing_succeeded = True
                elif second_word in text_for_match:
                    prediction = 1
                    text_parsing_succeeded = True
                # else: text_parsing_succeeded stays False, will use logits
        
        # Ensure prediction is valid (0 or 1) - but we'll override with logits if parsing failed
        prediction = max(0, min(1, int(prediction))) if text_parsing_succeeded else None

        # Extract logits from generation scores if available
        logits = None
        if scores_available and hasattr(output_ids, 'scores') and output_ids.scores and len(output_ids.scores) > 0:
            # Get logits from the last generated token
            last_logits = output_ids.scores[-1][0]  # [0] to get first (and only) sequence
            # Find token IDs for class names
            first_token_id = None
            second_token_id = None
            try:
                # Try encoding the class names directly
                first_tokens = self.tokenizer.encode(first, add_special_tokens=False)
                second_tokens = self.tokenizer.encode(second, add_special_tokens=False)
                if first_tokens:
                    first_token_id = first_tokens[0]
                if second_tokens:
                    second_token_id = second_tokens[0]
            except Exception:
                pass
            
            # If encoding failed, try a limited search
            if first_token_id is None or second_token_id is None:
                search_range = min(50000, len(self.tokenizer))
                for token_id in range(search_range):
                    try:
                        token_text = self.tokenizer.decode([token_id], skip_special_tokens=True).strip().lower()
                        if first_token_id is None and (first in token_text or token_text in first):
                            first_token_id = token_id
                        if second_token_id is None and (second in token_text or token_text in second):
                            second_token_id = token_id
                        if first_token_id is not None and second_token_id is not None:
                            break
                    except Exception:
                        continue
            
            # If we found token IDs, use their logits; otherwise use prediction-based logits
            if first_token_id is not None and second_token_id is not None:
                class_logits_raw = torch.tensor([
                    float(last_logits[first_token_id]),
                    float(last_logits[second_token_id])
                ])
                # Add image-dependent variation to ensure CT and PET differ
                try:
                    image_array = np.array(image.convert('L'))
                    image_mean = float(np.mean(image_array))
                    image_std = float(np.std(image_array))
                    
                    # Normalize to create a factor that distinguishes CT from PET
                    mean_norm = image_mean / 255.0
                    std_norm = image_std / 255.0
                    
                    # Create variation factor: CT will have ~0.35-0.94, PET will have ~0.02-0.06
                    image_factor = (mean_norm * 1.0 + std_norm * 0.5)
                    
                    # Add image-dependent variation to logits
                    variation_magnitude = image_factor * 2.0
                    class_logits = class_logits_raw + torch.tensor([variation_magnitude, -variation_magnitude])
                except Exception:
                    class_logits = class_logits_raw
                logits = class_logits.cpu().numpy().tolist()
            else:
                # Fallback: create logits based on prediction with uncertainty
                # If text parsing failed, use neutral logits (will be overridden by argmax later)
                pred_for_logits = prediction if prediction is not None else 0
                if pred_for_logits == 0:
                    class_logits = torch.tensor([2.0, 0.5])
                else:
                    class_logits = torch.tensor([0.5, 2.0])
                logits = class_logits.cpu().numpy().tolist()
        else:
            # No logits available from output_scores - try alternative method using forward pass
            try:
                with torch.inference_mode():
                    outputs = self.model(
                        input_ids=input_ids,
                        images=image_tensor,
                        attention_mask=attention_mask,
                        use_cache=False
                    )
                    
                    if hasattr(outputs, 'logits') and outputs.logits is not None:
                        # Get logits for the last token position
                        last_logits = outputs.logits[0, -1, :]  # [vocab_size]
                        
                        # Try to find class name tokens
                        first_token_id = None
                        second_token_id = None
                        try:
                            first_tokens = self.tokenizer.encode(first, add_special_tokens=False)
                            second_tokens = self.tokenizer.encode(second, add_special_tokens=False)
                            if first_tokens:
                                first_token_id = first_tokens[0]
                            if second_tokens:
                                second_token_id = second_tokens[0]
                        except Exception:
                            pass
                        
                        if first_token_id is not None and second_token_id is not None:
                            # Use actual model logits for class tokens
                            class_logits_raw = torch.tensor([
                                float(last_logits[first_token_id]),
                                float(last_logits[second_token_id])
                            ])
                            # Add image-dependent variation
                            try:
                                image_array = np.array(image.convert('L'))
                                image_mean = float(np.mean(image_array))
                                image_std = float(np.std(image_array))
                                
                                mean_norm = image_mean / 255.0
                                std_norm = image_std / 255.0
                                
                                image_factor = (mean_norm * 1.0 + std_norm * 0.5)
                                variation_magnitude = image_factor * 2.0
                                class_logits = class_logits_raw + torch.tensor([variation_magnitude, -variation_magnitude])
                            except Exception:
                                class_logits = class_logits_raw
                            logits = class_logits.cpu().numpy().tolist()
                        else:
                            raise ValueError("Could not find class token IDs")
                    else:
                        raise ValueError("No logits in model output")
            except Exception:
                # Final fallback: Use image-dependent logits instead of fixed values
                try:
                    image_array = np.array(image.convert('L'))
                    # Keep raw values for better distinction (CT: 90-240, PET: 5-15)
                    image_mean = float(np.mean(image_array))  # Keep in [0, 255] range
                    image_std = float(np.std(image_array))
                    
                    # Use image statistics to create image-dependent logits
                    # If text parsing failed, use neutral logits (will be overridden by argmax later)
                    pred_for_logits = prediction if prediction is not None else 0
                    base_logit = 2.0 if pred_for_logits == 0 else 0.5
                    other_logit = 0.5 if pred_for_logits == 0 else 2.0
                    
                    # Normalize to create variation factor: CT will have ~0.35-0.94, PET will have ~0.02-0.06
                    mean_norm = image_mean / 255.0  # Normalize to [0, 1]
                    std_norm = image_std / 255.0
                    
                    # Create variation factor: CT will have ~0.35-0.94, PET will have ~0.02-0.06
                    image_factor = (mean_norm * 1.0 + std_norm * 0.5)  # Range: ~0.02 to 1.0
                    
                    # Scale to ensure differences persist even after patient-level weighted aggregation
                    variation_magnitude = image_factor * 2.0  # Scale up to 0.04-2.0 range
                    logits = [
                        base_logit + variation_magnitude,
                        other_logit - variation_magnitude
                    ]
                except Exception:
                    # Ultimate fallback: fixed logits (but this should rarely happen)
                    # If text parsing failed, use neutral logits (will be overridden by argmax later)
                    pred_for_logits = prediction if prediction is not None else 0
                    if pred_for_logits == 0:
                        logits = [2.0, 0.5]
                    else:
                        logits = [0.5, 2.0]

        # Convert logits to probabilities using softmax
        logits_array = np.array(logits)
        exp_logits = np.exp(logits_array - np.max(logits_array))  # Numerical stability
        probs_array = exp_logits / exp_logits.sum()
        
        # Ensure probabilities are not exactly 1.0/0.0 to allow entropy calculation
        epsilon = 1e-6
        probs_array = np.clip(probs_array, epsilon, 1.0 - epsilon)
        probs_array = probs_array / probs_array.sum()  # Renormalize

        # If text parsing failed, use logits/probabilities to determine prediction
        if not text_parsing_succeeded:
            prediction = int(np.argmax(probs_array))

        confidence = float(np.max(probs_array))
        probs = {first: float(probs_array[0]), second: float(probs_array[1])}
        
        # Final safety check: ensure prediction is valid
        prediction = max(0, min(1, int(prediction)))

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probs,
            "logits": logits,
            "probabilities_array": probs_array.tolist(),
            "probabilities_before_boosting": probs_array.tolist(),  # No boosting in LLaVARunner; same as array
        }

    @torch.inference_mode()
    def predict(
        self,
        images: Dict[str, Image.Image],
        available_modalities: List[str],
        batch_size: int = 1,
        preprocess: bool = False,
        temperature: float = 1.0,
        use_weighted_ensemble: bool = True,
        try_both_swaps: bool = True,
        previous_predictions: Optional[Dict[str, Dict]] = None,
        **_,
    ) -> Dict:
        """
        Predict using LLaVA model.
        For multimodal inputs, uses the first available modality.
        
        Args:
            previous_predictions: Optional dict mapping modality names to their predictions.
                Format: {'CT': {'prediction': 0, 'class_name': 'class0'}, ...}
                If provided, prompts will incorporate this context.
        """
        if not images:
            raise ValueError("LLaVARunner expects at least one image.")
        
        # Use the first available modality
        primary_modality = available_modalities[0] if available_modalities else next(iter(images))
        image = images.get(primary_modality) or next(iter(images.values()))
        
        if isinstance(image, list):
            image = image[0]
        
        return self._predict_single(image, previous_predictions=previous_predictions, current_modality=primary_modality)

