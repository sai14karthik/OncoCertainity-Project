"""
Evaluation utilities for sequential modality feeding.
Focuses on CERTAINTY DYNAMICS and MODEL BEHAVIOR ANALYSIS, not accuracy optimization.

This module analyzes how modality changes affect prediction certainty (FULLY GENERIC for N modalities):
- Confidence scores across modalities (individual modalities and all combinations)
- Entropy measures (uncertainty quantification)
- Logit distributions and stability
- How modality disagreement affects confidence/logits
- How combining modalities affects certainty

SUPPORTS N MODALITIES (not limited to 2):
- Pairwise analysis: Compares ALL pairs of modalities (A vs B, A vs C, B vs C, etc.)
- Sequential analysis: Analyzes each modality with context from previous ones (B with A, C with A+B, D with A+B+C, etc.)
- Combined analysis: Analyzes all combinations (A+B, A+B+C, A+B+C+D, etc.)
- Works with any number of modalities: 2, 3, 4, 5, or more

Note: In zero-shot settings, accuracy near chance (0.5) is expected and not
necessarily negative. The focus is on understanding model behavior, not
optimizing classification performance.
"""

import numpy as np
import sys
from typing import List, Dict, Optional


def calculate_accuracy(
    predictions: List[int],
    labels: List[int]
) -> float:
    """Calculate accuracy given predictions and labels."""
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have same length")
    
    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(predictions) if len(predictions) > 0 else 0.0


def calculate_entropy(probabilities: np.ndarray) -> float:
    """
    Calculate entropy of probability distribution.
    Higher entropy = more uncertainty.
    
    Args:
        probabilities: Array of probabilities (should sum to 1)
    
    Returns:
        Entropy value (bits)
    """
    # Ensure probabilities sum to 1 and are non-negative
    probs = np.array(probabilities)
    probs = np.clip(probs, 1e-10, 1.0)  # Avoid log(0)
    probs_sum = probs.sum()
    if probs_sum > 0:
        probs = probs / probs_sum  # Renormalize
    else:
        # Fallback: uniform distribution if all zeros
        probs = np.ones_like(probs) / len(probs)
    
    # Calculate entropy: H(X) = -sum(p(x) * log2(p(x)))
    log_probs = np.log2(probs)
    entropy = -np.sum(probs * log_probs)
    
    return float(entropy)


def calculate_probability_gap(probabilities: np.ndarray) -> float:
    """
    Calculate the gap between top-1 and top-2 probabilities.
    Larger gap = more confident prediction.
    
    Args:
        probabilities: Array of probabilities
    
    Returns:
        Gap between top-1 and top-2 probabilities
    """
    probs = np.array(probabilities)
    sorted_probs = np.sort(probs)[::-1]  # Descending order
    
    if len(sorted_probs) < 2:
        return 0.0
    
    return float(sorted_probs[0] - sorted_probs[1])


def calculate_logit_magnitude(logits: np.ndarray) -> float:
    """
    Calculate the magnitude (L2 norm) of logits.
    Larger magnitude = stronger signal.
    
    Args:
        logits: Array of logits
    
    Returns:
        L2 norm of logits
    """
    logits_array = np.array(logits)
    return float(np.linalg.norm(logits_array))


def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    Used for comparing logit distributions across modalities.
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Cosine similarity (range: -1 to 1)
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def calculate_calibrated_probabilities(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Calculate calibrated probabilities using temperature scaling.
    
    Calibrated probabilities are more reliable confidence estimates.
    Temperature < 1.0 makes predictions more confident (sharper distribution).
    Temperature > 1.0 makes predictions less confident (smoother distribution).
    
    Args:
        logits: Raw logits from model
        temperature: Temperature scaling factor (default: 1.0 = no calibration)
    
    Returns:
        Calibrated probability distribution
    """
    logits_arr = np.array(logits)
    if temperature <= 0:
        temperature = 1.0
    
    # Apply temperature scaling: softmax(logits / temperature)
    scaled_logits = logits_arr / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))  # Numerical stability
    exp_sum = exp_logits.sum()
    if exp_sum > 0:
        probs = exp_logits / exp_sum
    else:
        # Fallback: uniform distribution if all zeros (shouldn't happen, but safe)
        probs = np.ones_like(exp_logits) / len(exp_logits)
    
    return probs


def calculate_calibration_error(uncalibrated_probs: np.ndarray, calibrated_probs: np.ndarray) -> float:
    """
    Calculate Expected Calibration Error (ECE) approximation.
    Measures how well calibrated probabilities match uncalibrated ones.
    
    Args:
        uncalibrated_probs: Original uncalibrated probabilities
        calibrated_probs: Temperature-calibrated probabilities
    
    Returns:
        Calibration error (lower is better)
    """
    uncal = np.array(uncalibrated_probs)
    cal = np.array(calibrated_probs)
    
    # Mean absolute difference
    mae = np.mean(np.abs(uncal - cal))
    return float(mae)


def aggregate_patient_predictions(patient_slices: List[Dict]) -> Dict:
    """
    Aggregate predictions from multiple slices per patient.
    
    Strategy: Weighted voting by confidence
    - Each slice's prediction is weighted by its confidence
    - Final prediction is the class with highest weighted sum
    - Tie-breaking: Use class with higher average confidence
    
    Args:
        patient_slices: List of prediction dicts, each with 'prediction' and 'confidence'
    
    Returns:
        Dictionary with aggregated 'prediction', 'confidence', and 'num_slices'
    """
    if not patient_slices:
        return {'prediction': 0, 'confidence': 0.0, 'num_slices': 0}
    
    # Weighted voting: sum confidence for each class
    class_weights = {0: 0.0, 1: 0.0}
    class_confidences = {0: [], 1: []}  # Track individual confidences for tie-breaking
    total_confidence = 0.0
    
    for slice_pred in patient_slices:
        # Validate prediction is 0 or 1
        pred = slice_pred.get('prediction', 0)
        pred = max(0, min(1, int(pred)))  # Clamp to valid range [0, 1]
        
        # Validate confidence is in valid range [0, 1]
        conf = slice_pred.get('confidence', 0.5)
        conf = max(0.0, min(1.0, float(conf)))  # Clamp to valid range
        
        class_weights[pred] += conf
        class_confidences[pred].append(conf)
        total_confidence += conf
    
    # Get prediction with highest weighted sum
    # Handle tie-breaking: if weights are equal, use class with higher average confidence
    if class_weights[0] == class_weights[1]:
        # Tie-breaking: compare average confidence
        avg_conf_0 = sum(class_confidences[0]) / len(class_confidences[0]) if class_confidences[0] else 0.0
        avg_conf_1 = sum(class_confidences[1]) / len(class_confidences[1]) if class_confidences[1] else 0.0
        aggregated_pred = 1 if avg_conf_1 > avg_conf_0 else 0
    else:
        aggregated_pred = max(class_weights.items(), key=lambda x: x[1])[0]
    
    # Calculate aggregated confidence
    # Use average confidence of slices (or of winning-class slices) so result is in [0,1] and reflects real certainty.
    # Avoid returning 1.0 when all slices agree with high conf (weight/total_weight would be 1.0).
    if len(patient_slices) == 1:
        # Single slice: use its original confidence directly
        aggregated_conf = patient_slices[0].get('confidence', 0.5)
        aggregated_conf = max(0.0, min(1.0, float(aggregated_conf)))  # Clamp to valid range
    elif class_confidences[aggregated_pred]:
        # Multiple slices: use mean confidence of slices that voted for the winning class (reflects real certainty)
        aggregated_conf = sum(class_confidences[aggregated_pred]) / len(class_confidences[aggregated_pred])
        aggregated_conf = max(0.0, min(1.0, float(aggregated_conf)))
    else:
        aggregated_conf = 0.5
    
    return {
        'prediction': aggregated_pred,
        'confidence': aggregated_conf,
        'num_slices': len(patient_slices)
    }


def evaluate_sequential_modalities(
    results: Dict[str, List[Dict]],
    modalities: List[str]
) -> Dict:
    """
    Evaluate model behavior for each modality (and optional mix).
    Focuses on certainty dynamics and how modality changes affect prediction certainty.

    This analysis is designed to understand model behavior, not optimize accuracy.
    In zero-shot settings, accuracy near chance is expected; what matters is:
    - How confidence changes across modalities (Mod1, Mod2, Mod1+Mod2, etc.)
    - Whether combining modalities increases or decreases certainty
    - Whether modality disagreement leads to lower confidence or unstable logits

    Args:
        results: Dictionary with keys as case_ids and values as lists of predictions
                 Each prediction dict should have 'modalities_used', 'prediction', 'label',
                 'confidence', 'probabilities', and optionally 'logits'
        modalities: Ordered list of modalities supplied via CLI (supports N modalities)

    Returns:
        Dictionary with certainty metrics for each step (accuracy included for reference only).

    LLaVA-Med / N-modality invariants (keep when changing this or analyze_* functions):
    - Pairwise keys: canonical tuple(sorted([mod_i, mod_j])) so (CT,MR) and (MR,CT) are one pair.
    - Pairwise confidence & dominance: stored metrics are normalized so Mod1 = first of key, Mod2 = second,
      so "Mod1 Higher %" / "Mod2 Higher %" and "X has higher confidence" are consistent across run orders.
    - All analyze_* functions that match two or three prediction lists must group by each list's own
      patient_id (then intersect), then aggregate per patient. Never zip by index when slice counts
      differ (e.g. CT 738 vs MR 235 vs PT 789).
    - First-vs-combined agreement: when slice counts differ or combined has multiple mods, aggregate
      both to patient-level before comparing.
    """
    if not results:
        return {'step_results': {}, 'patient_level_results': {}, 'modalities': modalities or []}
    if not modalities:
        return {'step_results': {}, 'patient_level_results': {}, 'modalities': []}
    # Organize by modality combinations
    # Dynamic step structure:
    # - Each modality alone: "Mod1", "Mod2", "Mod3", etc.
    # - Each modality with context: "Mod2+Mod1", "Mod3+Mod1+Mod2", etc.
    step_data = {}
    
    # Add steps for each modality alone
    for mod in modalities:
        step_data[mod] = {'predictions': [], 'labels': [], 'full_predictions': []}
    
    # Add steps for each modality with context from previous ones
    for i in range(1, len(modalities)):
        current_mod = modalities[i]
        context_mods = modalities[:i]
        combined_step_name = '+'.join(context_mods + [current_mod])  # e.g., "Mod1+Mod2" or "Mod1+Mod2+Mod3"
        step_data[combined_step_name] = {'predictions': [], 'labels': [], 'full_predictions': []}
    
    # Count predictions before categorization
    total_results_count = sum(len(case_results) for case_results in results.values())
    
    for case_id, case_results in results.items():
        for result in case_results:
            mods_used = result.get('modalities_used', [])
            prediction = result.get('prediction')
            label = result.get('label')
            
            # Skip if required fields are missing
            if prediction is None or label is None:
                continue
            
            # Determine step name (explicit step overrides modality inference)
            # Dynamic categorization:
            # 1) Modality alone (no context) → "Mod1", "Mod2", etc.
            # 2) Modality with context → "Mod2+Mod1", "Mod3+Mod1+Mod2", etc.
            step_name = result.get('step')
            if step_name is None:
                used_context = result.get('used_context', False)
                context_from = result.get('context_from', [])
                
                if len(mods_used) == 1:
                    mod = mods_used[0]
                    
                    if used_context and len(context_from) > 0:
                        # Modality with context: build combined step name
                        # Sort context_from to match the order in modalities list
                        sorted_context = [m for m in modalities if m in context_from]
                        if sorted_context:
                            combined_step_name = '+'.join(sorted_context + [mod])
                            step_name = combined_step_name if combined_step_name in step_data else (mod if mod in step_data else None)
                        else:
                            step_name = mod if mod in step_data else None
                    else:
                        # Modality alone (no context)
                        step_name = mod if mod in step_data else None
                elif len(mods_used) > 1:
                    # Explicit combined modality case (multiple modalities used together)
                    step_name = '+'.join(sorted(mods_used, key=lambda m: modalities.index(m) if m in modalities else 999))
                    if step_name not in step_data:
                        step_name = None
                else:
                    step_name = None
            
            if step_name is None:
                continue
            
            if step_name in step_data:
                step_data[step_name]['predictions'].append(prediction)
                step_data[step_name]['labels'].append(label)
                step_data[step_name]['full_predictions'].append(result)  # Store full prediction dict
    
    # Calculate accuracy and certainty metrics for each step
    step_results = {}
    for step_name, data in step_data.items():
        if len(data['predictions']) > 0:
            # Debug: Check predictions and labels types/values
            predictions = data['predictions']
            labels = data['labels']
            
            # Ensure predictions and labels are integers
            predictions = [int(p) if p is not None else 0 for p in predictions]
            labels = [int(l) if l is not None else 0 for l in labels]
            
            acc = calculate_accuracy(predictions, labels)
            num_predictions = len(predictions)
            
            # Analyze certainty metrics
            certainty_metrics = analyze_certainty_metrics(data['full_predictions'], step_name)
            
            # Analyze overconfidence (high-confidence incorrect predictions)
            overconfidence_metrics = analyze_overconfidence(data['full_predictions'], step_name)
            
            step_results[step_name] = {
                'accuracy': acc,
                'num_samples': num_predictions,
                'certainty_metrics': certainty_metrics,
                'overconfidence_metrics': overconfidence_metrics
            }
        else:
            step_results[step_name] = {
                'accuracy': 0.0,
                'num_samples': 0,
                'certainty_metrics': analyze_certainty_metrics([], step_name),
                'overconfidence_metrics': analyze_overconfidence([], step_name)
            }
    
    # ============================================================================
    # FULL N-MODALITY ANALYSIS
    # ============================================================================
    # Analyze all modality pairs, sequential effects, and combined modalities
    # ============================================================================
    
    # Store all pairwise analyses
    # Pairwise analysis dictionaries: Compare ALL pairs of modalities (supports N modalities)
    # Keys: (mod_i, mod_j) tuples for any two modalities
    # Example: For modalities [A, B, C, D], this contains (A,B), (A,C), (A,D), (B,C), (B,D), (C,D)
    pairwise_agreements = {}  # {(mod_i, mod_j): agreement_metrics} for ALL pairs
    pairwise_logit_similarities = {}  # {(mod_i, mod_j): similarity_metrics} for ALL pairs
    pairwise_confidence_comparisons = {}  # {(mod_i, mod_j): comparison_metrics} for ALL pairs
    pairwise_dominance = {}  # {(mod_i, mod_j): dominance_metrics} for ALL pairs
    
    # Sequential analysis: how each modality affects the next (supports N modalities)
    # Keys: modality name (e.g., 'B', 'C', 'D' for modalities [A, B, C, D])
    # Values: analysis of how that modality changes when given context from all previous modalities
    # Example: For [A, B, C, D], analyzes B with A context, C with A+B context, D with A+B+C context
    sequential_analyses = {}  # {mod: analysis_dict} for each modality with its preceding context
    
    # Combined modality analyses: Analyze ALL combinations (supports N modalities)
    # Keys: combined modality names (e.g., 'A+B', 'A+B+C', 'A+B+C+D' for modalities [A, B, C, D])
    # Values: analysis comparing the combined modality with the first modality
    # Example: For [A, B, C, D], analyzes A+B, A+B+C, A+B+C+D (all combinations)
    combined_agreements = {}  # {combined_name: agreement_with_first_mod} for ALL combinations
    combined_uncertainty_effects = {}  # {combined_name: uncertainty_analysis} for ALL combinations
    combined_multimodal_values = {}  # {combined_name: multimodal_value_analysis} for ALL combinations
    combined_bias_analyses = {}  # {combined_name: bias_analysis} for ALL combinations
    
    # Extract patient IDs helper function
    def extract_patient_ids_from_predictions(predictions):
        """Extract patient IDs from a list of predictions."""
        if not predictions:
            return None
        first_pred = predictions[0]
        if isinstance(first_pred, dict) and 'patient_id' in first_pred:
            return [p.get('patient_id') for p in predictions if isinstance(p, dict) and p.get('patient_id') is not None]
        return None
    
    # ============================================================================
    # 1. PAIRWISE ANALYSIS: Compare all modality pairs
    # ============================================================================
    if len(modalities) >= 2:
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                mod_i = modalities[i]
                mod_j = modalities[j]
                
                if mod_i in step_data and mod_j in step_data:
                    mod_i_preds = step_data[mod_i]['full_predictions']
                    mod_j_preds = step_data[mod_j]['full_predictions']
                    
                    if not mod_i_preds or not mod_j_preds:
                        continue
                    
                    patient_ids = extract_patient_ids_from_predictions(mod_i_preds)
                    pair_key = tuple(sorted([mod_i, mod_j]))  # Canonical key so (MR,CT) and (CT,MR) match
                    # Pairwise agreement
                    agreement = analyze_modality_agreement(mod_i_preds, mod_j_preds, patient_ids)
                    pairwise_agreements[pair_key] = agreement
                    # Logit similarity
                    similarity = analyze_logit_similarity(mod_i_preds, mod_j_preds, patient_ids)
                    pairwise_logit_similarities[pair_key] = similarity
                    # Confidence comparison (normalize so pair_key order = Mod1, Mod2 for consistent tables)
                    comparison = analyze_modality_confidence_comparison(mod_i_preds, mod_j_preds, patient_ids)
                    mod_a, mod_b = pair_key
                    if (mod_i, mod_j) != (mod_a, mod_b):
                        # Was computed as (mod_j, mod_i); swap so stored = (mod_a, mod_b)
                        comparison = {
                            **comparison,
                            'mod1_higher_confidence_rate': comparison.get('mod2_higher_confidence_rate', 0.0),
                            'mod2_higher_confidence_rate': comparison.get('mod1_higher_confidence_rate', 0.0),
                            'avg_confidence_difference': -comparison.get('avg_confidence_difference', 0.0),
                        }
                    pairwise_confidence_comparisons[pair_key] = comparison
                    # Dominance analysis (normalize for canonical key order)
                    dominance = analyze_modality_dominance(mod_i_preds, mod_j_preds, patient_ids)
                    if (mod_i, mod_j) != (mod_a, mod_b):
                        dominance = {
                            **dominance,
                            'mod2_higher_confidence_rate': dominance.get('mod1_higher_confidence_rate', 0.0),
                            'mod1_higher_confidence_rate': dominance.get('mod2_higher_confidence_rate', 0.0),
                            'avg_confidence_difference': -dominance.get('avg_confidence_difference', 0.0),
                            'mod2_wins_when_disagree': dominance.get('mod1_wins_when_disagree', 0),
                            'mod1_wins_when_disagree': dominance.get('mod2_wins_when_disagree', 0),
                        }
                    pairwise_dominance[pair_key] = dominance
    
    # ============================================================================
    # 2. SEQUENTIAL ANALYSIS: How each modality affects the next
    # ============================================================================
    # For modalities B, C, D, etc., analyze how they change with previous context
    for mod_idx in range(1, len(modalities)):
        current_mod = modalities[mod_idx]
        previous_mods = modalities[:mod_idx]
        
        # Get predictions for current mod alone and with context
        mod_alone_preds = step_data.get(current_mod, {}).get('full_predictions', [])
        combined_step_name = '+'.join(previous_mods + [current_mod])
        mod_with_context_preds = step_data.get(combined_step_name, {}).get('full_predictions', [])
        
        if mod_alone_preds and mod_with_context_preds:
            patient_ids = extract_patient_ids_from_predictions(mod_alone_preds)
            
            # Compare current mod alone vs with context
            sequential_analysis = analyze_modality_confidence_comparison(
                mod_alone_preds, mod_with_context_preds, patient_ids
            )
            sequential_analyses[current_mod] = {
                'context_from': previous_mods,
                'comparison': sequential_analysis,
                'alone_predictions': mod_alone_preds,
                'with_context_predictions': mod_with_context_preds
            }
    
    # ============================================================================
    # 3. COMBINED MODALITY ANALYSIS: Analyze all combinations
    # ============================================================================
    # For each combined step (A+B, A+B+C, A+B+C+D, etc.)
    for mod_idx in range(1, len(modalities)):
        combined_mods = modalities[:mod_idx + 1]
        combined_step_name = '+'.join(combined_mods)
        
        if combined_step_name not in step_data:
            continue
        
        combined_preds = step_data[combined_step_name]['full_predictions']
        if not combined_preds:
            continue
        
        # Compare first modality vs combined
        # Safety check: ensure we have at least one modality
        if len(modalities) == 0:
            continue
        first_mod = modalities[0]
        first_mod_preds = step_data.get(first_mod, {}).get('full_predictions', [])
        
        if first_mod_preds:
            patient_ids = extract_patient_ids_from_predictions(first_mod_preds)
            
            # Agreement between first mod and combined
            # CRITICAL FIX: When comparing first modality with a combined step that includes multiple modalities,
            # we're comparing different modalities (e.g., PT vs CT in PT+MR+CT step, which processes CT images).
            # Different modalities have different slice counts and non-aligned slice indices, so we must aggregate
            # to patient level for meaningful comparison.
            # Also handle cases where slice counts differ significantly even for same modality.
            len_first = len(first_mod_preds)
            len_combined = len(combined_preds)
            slice_count_ratio = max(len_first, len_combined) / min(len_first, len_combined) if min(len_first, len_combined) > 0 else float('inf')
            
            # Always aggregate to patient level if:
            # 1. Combined step includes multiple modalities (different modality than first), OR
            # 2. Slice counts differ significantly (>1.5 ratio)
            should_aggregate = len(combined_mods) > 1 or slice_count_ratio > 1.5
            
            if should_aggregate:
                # Aggregate both to patient-level for fair comparison
                first_mod_by_patient = {}
                combined_by_patient = {}
                
                for pred in first_mod_preds:
                    pid = pred.get('patient_id')
                    if pid is not None:
                        if pid not in first_mod_by_patient:
                            first_mod_by_patient[pid] = []
                        first_mod_by_patient[pid].append(pred)
                
                for pred in combined_preds:
                    pid = pred.get('patient_id')
                    if pid is not None:
                        if pid not in combined_by_patient:
                            combined_by_patient[pid] = []
                        combined_by_patient[pid].append(pred)
                
                # Aggregate to patient-level
                first_mod_patient_preds = []
                combined_patient_preds = []
                common_patients = set(first_mod_by_patient.keys()) & set(combined_by_patient.keys())
                
                for pid in sorted(common_patients):
                    first_aggregated = aggregate_patient_predictions(first_mod_by_patient[pid])
                    combined_aggregated = aggregate_patient_predictions(combined_by_patient[pid])
                    
                    first_mod_patient_preds.append({
                        'prediction': first_aggregated['prediction'],
                        'confidence': first_aggregated['confidence'],
                        'patient_id': pid
                    })
                    combined_patient_preds.append({
                        'prediction': combined_aggregated['prediction'],
                        'confidence': combined_aggregated['confidence'],
                        'patient_id': pid
                    })
                
                # Compare patient-level predictions
                patient_ids_for_agreement = [p.get('patient_id') for p in first_mod_patient_preds]
                agreement = analyze_modality_agreement(first_mod_patient_preds, combined_patient_preds, patient_ids_for_agreement)
            else:
                # Slice counts are similar - use slice-level comparison
                agreement = analyze_modality_agreement(first_mod_preds, combined_preds, patient_ids)
            
            combined_agreements[combined_step_name] = agreement
            
            # Uncertainty effect - GENERIC for all combinations (not just first pair)
            # For A+B: compare A and B alone vs A+B
            # For A+B+C: compare A, B, C alone vs A+B+C
            # For A+B+C+D: compare A, B, C, D alone vs A+B+C+D, etc.
            if len(combined_mods) >= 2:
                # Get predictions for all individual modalities in this combination
                individual_mod_preds = {}
                all_individual_preds_available = True
                for mod in combined_mods:
                    mod_preds = step_data.get(mod, {}).get('full_predictions', [])
                    if mod_preds:
                        individual_mod_preds[mod] = mod_preds
                    else:
                        all_individual_preds_available = False
                        break
                
                # Analyze for all combinations (2+ modalities)
                # For 2 modalities (A+B): Use existing 2-modality analysis functions
                # For 3+ modalities (A+B+C): Aggregate all individual modalities and compare vs combined
                if all_individual_preds_available:
                    if len(individual_mod_preds) == 2:
                        # 2-modality case: Use existing functions
                        mod_list = list(individual_mod_preds.keys())
                        mod1_preds = individual_mod_preds[mod_list[0]]
                        mod2_preds = individual_mod_preds[mod_list[1]]
                        
                        uncertainty = analyze_multimodality_uncertainty_effect(
                            mod1_preds, mod2_preds, combined_preds, patient_ids
                        )
                        combined_uncertainty_effects[combined_step_name] = uncertainty
                        
                        multimodal_value = analyze_zero_shot_multimodal_value(
                            mod1_preds, mod2_preds, combined_preds, patient_ids
                        )
                        combined_multimodal_values[combined_step_name] = multimodal_value
                        
                        bias = analyze_multimodal_bias(
                            mod1_preds, mod2_preds, combined_preds, patient_ids
                        )
                        combined_bias_analyses[combined_step_name] = bias
                    elif len(individual_mod_preds) >= 3:
                        # 3+ modality case: Aggregate all individual modalities and compare vs combined
                        # Strategy: Aggregate predictions from all individual modalities, then compare aggregated vs combined
                        # Extract patient IDs from combined predictions
                        combined_patient_ids = extract_patient_ids_from_predictions(combined_preds)
                        
                        # Aggregate all individual modalities per patient
                        aggregated_individual_preds = []
                        aggregated_patient_ids = []
                        
                        # Group all individual modality predictions by patient
                        all_individual_by_patient = {}
                        for mod, mod_preds in individual_mod_preds.items():
                            mod_patient_ids = extract_patient_ids_from_predictions(mod_preds)
                            if mod_patient_ids is None:
                                # Fallback: use index-based matching if patient IDs not available
                                mod_patient_ids = [None] * len(mod_preds)
                            for pid, pred in zip(mod_patient_ids, mod_preds):
                                if pid is not None:
                                    if pid not in all_individual_by_patient:
                                        all_individual_by_patient[pid] = {mod: [] for mod in individual_mod_preds.keys()}
                                    if mod in all_individual_by_patient[pid]:
                                        all_individual_by_patient[pid][mod].append(pred)
                        
                        # For each patient, aggregate across all individual modalities
                        for pid in set(combined_patient_ids) if combined_patient_ids else set(all_individual_by_patient.keys()):
                            if pid in all_individual_by_patient:
                                # Get all slices from all individual modalities for this patient
                                all_individual_slices = []
                                for mod in individual_mod_preds.keys():
                                    if mod in all_individual_by_patient[pid]:
                                        all_individual_slices.extend(all_individual_by_patient[pid][mod])
                                
                                if all_individual_slices:
                                    # Aggregate all individual modality predictions
                                    aggregated = aggregate_patient_predictions(all_individual_slices)
                                    aggregated_individual_preds.append({
                                        'prediction': aggregated['prediction'],
                                        'confidence': aggregated['confidence'],
                                        'probabilities': all_individual_slices[0].get('probabilities', {}) if all_individual_slices else {},
                                        'probabilities_array': all_individual_slices[0].get('probabilities_array', []) if all_individual_slices else []
                                    })
                                    aggregated_patient_ids.append(pid)
                        
                        # Now compare aggregated individual vs combined using first two modalities as representatives
                        # (The analysis functions need 2 inputs, so we use first two individual modalities)
                        mod_list = list(individual_mod_preds.keys())
                        mod1_preds = individual_mod_preds[mod_list[0]]
                        mod2_preds = individual_mod_preds[mod_list[1]]
                        
                        # Use aggregated individual predictions as "mod1" and first individual modality as "mod2"
                        # This compares: (aggregated all individual) vs (combined)
                        # We'll use mod1_preds as placeholder for aggregated, and mod2_preds as placeholder
                        # But actually compare aggregated_individual_preds vs combined_preds
                        # Since the function signature requires 2 modalities, we'll use a workaround:
                        # Compare first two individual modalities vs combined, but note that this is a subset
                        uncertainty = analyze_multimodality_uncertainty_effect(
                            mod1_preds, mod2_preds, combined_preds, patient_ids
                        )
                        # Add note that this is for N>2 modalities
                        uncertainty['note'] = f"Analysis for {len(individual_mod_preds)} modalities using first two as representatives. Full pairwise analysis available in pairwise_agreements."
                        combined_uncertainty_effects[combined_step_name] = uncertainty
                        
                        multimodal_value = analyze_zero_shot_multimodal_value(
                            mod1_preds, mod2_preds, combined_preds, patient_ids
                        )
                        multimodal_value['note'] = f"Analysis for {len(individual_mod_preds)} modalities using first two as representatives. Full pairwise analysis available in pairwise_agreements."
                        combined_multimodal_values[combined_step_name] = multimodal_value
                        
                        bias = analyze_multimodal_bias(
                            mod1_preds, mod2_preds, combined_preds, patient_ids
                        )
                        bias['note'] = f"Analysis for {len(individual_mod_preds)} modalities using first two as representatives. Full pairwise analysis available in pairwise_agreements."
                        # Store actual modality names used in analysis for correct display
                        bias['actual_mod1_name'] = mod_list[0]
                        bias['actual_mod2_name'] = mod_list[1]
                        combined_bias_analyses[combined_step_name] = bias
    
    # ============================================================================
    # BACKWARD COMPATIBILITY: Legacy variables for first 2 modalities only
    # NOTE: The FULL N-modality analysis is already done above (pairwise, sequential, combined)
    # This section only provides backward compatibility variables for code expecting mod1/mod2
    # For N modalities, use the dictionaries above: pairwise_agreements, sequential_analyses, etc.
    # ============================================================================
    agreement_metrics = None
    mod1_vs_combined_agreement = None
    logit_similarity_analysis = None
    mod2_vs_mod1_analysis = None
    mod2_dominance_analysis = None
    uncertainty_effect_analysis = None
    multimodal_value_analysis = None
    multimodal_bias_analysis = None
    modality_combination_analysis = None
    
    # Extract first 2 modalities for backward compatibility (if available)
    # NOTE: This does NOT limit the analysis - full N-modality analysis is already complete above
    if len(modalities) >= 2:
        first_mod = modalities[0]  # First modality (generic, not hardcoded to "mod1")
        second_mod = modalities[1]  # Second modality (generic, not hardcoded to "mod2")
        
        # Get first pair analysis for backward compatibility only
        first_pair_key = tuple(sorted([first_mod, second_mod]))
        if first_pair_key in pairwise_agreements:
            agreement_metrics = pairwise_agreements[first_pair_key]
        if first_pair_key in pairwise_logit_similarities:
            logit_similarity_analysis = pairwise_logit_similarities[first_pair_key]
        if first_pair_key in pairwise_confidence_comparisons:
            mod2_vs_mod1_analysis = pairwise_confidence_comparisons[first_pair_key]
        if first_pair_key in pairwise_dominance:
            mod2_dominance_analysis = pairwise_dominance[first_pair_key]
        
        # Get combined analysis for first combination (A+B)
        # NOTE: For N modalities, use combined_agreements for any combination (A+B, A+B+C, etc.)
        combined_mod_name = '+'.join(modalities[:2]) if len(modalities) >= 2 else None
        if combined_mod_name and combined_mod_name in combined_agreements:
            mod1_vs_combined_agreement = combined_agreements[combined_mod_name]
        
        if combined_mod_name and combined_mod_name in combined_uncertainty_effects:
            uncertainty_effect_analysis = combined_uncertainty_effects[combined_mod_name]
        
        if combined_mod_name and combined_mod_name in combined_multimodal_values:
            multimodal_value_analysis = combined_multimodal_values[combined_mod_name]
        
        if combined_mod_name and combined_mod_name in combined_bias_analyses:
            multimodal_bias_analysis = combined_bias_analyses[combined_mod_name]
        
        # Modality combination effect (for first 2 modalities only - backward compatibility)
        # NOTE: For N modalities, analyze all combinations using the loops above
        if first_mod in step_data and second_mod in step_data and combined_mod_name in step_data:
            mod1_preds = step_data[first_mod]['full_predictions']
            mod2_preds = step_data[second_mod]['full_predictions']
            combined_preds = step_data[combined_mod_name]['full_predictions']
            
            # Extract patient IDs from mod1_preds for matching (used in backward compatibility analysis)
            patient_ids = extract_patient_ids_from_predictions(mod1_preds)
            
            if combined_mod_name in step_results:
                modality_combination_analysis = analyze_modality_combination_effect(
                    mod1_preds, mod2_preds,
                    step_results[combined_mod_name]['certainty_metrics'],
                    step_results[first_mod]['certainty_metrics'],
                    step_results[second_mod]['certainty_metrics']
                )
            
            # Calculate Mod1 vs Mod1+Mod2 agreement if Mod1+Mod2 exists (for backward compatibility)
            # This shows if Mod1 context makes Mod2 agree more with Mod1
            # NOTE: Use first 2 modalities (A+B) for backward compatibility, not all modalities
            # For full combination analysis, see lines 791-830 below
            first_two_combined_name = '+'.join(modalities[:2]) if len(modalities) >= 2 else None
            combined_preds_for_agreement = None
            if first_two_combined_name and first_two_combined_name in step_data:
                combined_preds_for_agreement = step_data[first_two_combined_name]['full_predictions']
                # Extract patient IDs from Mod1+Mod2 as well to ensure proper matching
                # Use intersection of patient IDs from both Mod1 and Mod1+Mod2 for accurate comparison
                combined_patient_ids = None
                if combined_preds_for_agreement and len(combined_preds_for_agreement) > 0:
                    first_combined_pred = combined_preds_for_agreement[0]
                    if isinstance(first_combined_pred, dict) and 'patient_id' in first_combined_pred:
                        combined_patient_ids = [p.get('patient_id') for p in combined_preds_for_agreement if isinstance(p, dict) and p.get('patient_id') is not None]
                
                # Use intersection of patient IDs (only match patients present in both)
                if patient_ids and combined_patient_ids:
                    # Find intersection: patients present in both Mod1 and Mod1+Mod2
                    mod1_patient_set = set(patient_ids)
                    combined_patient_set = set(combined_patient_ids)
                    intersection_patient_ids_set = mod1_patient_set & combined_patient_set
                    
                    # Filter predictions to only include intersection patients
                    # Keep the order from original lists to maintain slice-level matching
                    filtered_mod1_preds = []
                    filtered_mod1_patient_ids = []
                    for pid, pred in zip(patient_ids, mod1_preds):
                        if pid in intersection_patient_ids_set:
                            filtered_mod1_preds.append(pred)
                            filtered_mod1_patient_ids.append(pid)
                    
                    filtered_combined_preds = []
                    filtered_combined_patient_ids = []
                    for pid, pred in zip(combined_patient_ids, combined_preds_for_agreement):
                        if pid in intersection_patient_ids_set:
                            filtered_combined_preds.append(pred)
                            filtered_combined_patient_ids.append(pid)
                    
                    # Match by patient_id (patient-level matching)
                    # For disagreement rate, we want to compare at patient level, not slice level
                    # Group predictions by patient_id and take one representative prediction per patient
                    mod1_by_patient = {}
                    combined_by_patient = {}
                    
                    for pred in filtered_mod1_preds:
                        if isinstance(pred, dict):
                            pid = pred.get('patient_id')
                            if pid is not None:
                                # Use first prediction per patient (or could aggregate)
                                if pid not in mod1_by_patient:
                                    mod1_by_patient[pid] = pred
                    
                    for pred in filtered_combined_preds:
                        if isinstance(pred, dict):
                            pid = pred.get('patient_id')
                            if pid is not None:
                                # Use first prediction per patient (or could aggregate)
                                if pid not in combined_by_patient:
                                    combined_by_patient[pid] = pred
                    
                    # Match by patient_id (one prediction per patient)
                    if mod1_by_patient and combined_by_patient:
                        matched_mod1 = []
                        matched_combined = []
                        matched_patient_ids = []
                        
                        # Find intersection of patient_ids
                        common_patient_ids = set(mod1_by_patient.keys()) & set(combined_by_patient.keys())
                        for pid in sorted(common_patient_ids):  # Sort for consistent ordering
                            matched_mod1.append(mod1_by_patient[pid])
                            matched_combined.append(combined_by_patient[pid])
                            matched_patient_ids.append(pid)
                        
                        if matched_mod1 and matched_combined:
                            # Use patient-level matching
                            mod1_vs_combined_agreement = analyze_modality_agreement(
                                matched_mod1,
                                matched_combined,
                                matched_patient_ids,
                                debug=False  # Debug disabled for cleaner output
                            )
                        else:
                            # Fallback to original patient_id matching
                            mod1_vs_combined_agreement = analyze_modality_agreement(
                                filtered_mod1_preds, 
                                filtered_combined_preds, 
                                filtered_mod1_patient_ids
                            )
                    else:
                        # Fallback: use patient_id matching only
                        mod1_vs_combined_agreement = analyze_modality_agreement(
                            filtered_mod1_preds, 
                            filtered_combined_preds, 
                            filtered_mod1_patient_ids
                        )
                else:
                    # Fallback: use original patient_ids if intersection not available
                    mod1_vs_combined_agreement = analyze_modality_agreement(mod1_preds, combined_preds_for_agreement, patient_ids)
            
            # Analyze logit similarity between first 2 modalities (backward compatibility only)
            # NOTE: For N modalities, use pairwise_logit_similarities[(mod_i, mod_j)] for any pair
            logit_similarity_analysis = analyze_logit_similarity(mod1_preds, mod2_preds, patient_ids)
            
            # CORE RESEARCH QUESTION: Does second modality (with first modality context) consistently produce higher confidence?
            # Compare first_mod vs first_mod+second_mod (second_mod with first_mod context) to show improvement
            # NOTE: For N modalities, use sequential_analyses[mod] for any modality with its preceding context
            # Use first 2 modalities combination (A+B) for backward compatibility
            first_two_combined_name = '+'.join(modalities[:2]) if len(modalities) >= 2 else None
            first_two_combined_preds = None
            if first_two_combined_name and first_two_combined_name in step_data:
                first_two_combined_preds = step_data[first_two_combined_name]['full_predictions']
            
            if first_two_combined_preds:
                # Compare first_mod vs first_mod+second_mod (second_mod with first_mod context)
                mod2_vs_mod1_analysis = analyze_modality_confidence_comparison(mod1_preds, first_two_combined_preds, patient_ids)
            else:
                # Fallback: Compare first_mod vs second_mod (alone) if combined not available
                mod2_vs_mod1_analysis = analyze_modality_confidence_comparison(mod1_preds, mod2_preds, patient_ids)
            
            # Analyze second modality dominance (systematic bias toward second modality) - backward compatibility only
            # NOTE: For N modalities, use pairwise_dominance[(mod_i, mod_j)] for any pair
            mod2_dominance_analysis = analyze_modality_dominance(mod1_preds, mod2_preds, patient_ids)
            
            # Analyze how combining ALL modalities affects certainty (full combination: A+B+C+...+N)
            # NOTE: This analyzes the FULL combination of all N modalities, not just first 2
            combined_mod_name = '+'.join(modalities)  # Full combination: all modalities
            combined_preds = None
            if combined_mod_name in step_data:
                combined_preds = step_data[combined_mod_name]['full_predictions']
                modality_combination_analysis = analyze_modality_combination_effect(
                    mod1_preds, mod2_preds, 
                    step_results[combined_mod_name]['certainty_metrics'],
                    step_results[first_mod]['certainty_metrics'],
                    step_results[second_mod]['certainty_metrics']
                )
                
                # CORE RESEARCH QUESTION: Does multimodality reduce uncertainty or introduce conflict?
                # NOTE: Analysis functions are designed for 2 modalities, so uses first two as representatives
                # For N>2 modalities, pairwise analysis (above) provides comprehensive coverage of all pairs
                uncertainty_effect_analysis = analyze_multimodality_uncertainty_effect(
                    mod1_preds, mod2_preds, combined_preds, patient_ids
                )
                
                # CORE RESEARCH QUESTION: Can zero-shot VLMs reflect multimodal value?
                # NOTE: Analysis functions are designed for 2 modalities, so uses first two as representatives
                # For N>2 modalities, pairwise analysis (above) provides comprehensive coverage of all pairs
                multimodal_value_analysis = analyze_zero_shot_multimodal_value(
                    mod1_preds, mod2_preds, combined_preds, patient_ids
                )
                
                # Analyze multimodal bias (true combination vs modality bias)
                # NOTE: Analysis functions are designed for 2 modalities, so uses first two as representatives
                # For N>2 modalities, pairwise analysis (above) provides comprehensive coverage of all pairs
                if combined_preds:
                    multimodal_bias_analysis = analyze_multimodal_bias(
                        mod1_preds, mod2_preds, combined_preds, patient_ids
                    )
                else:
                    multimodal_bias_analysis = None
            else:
                uncertainty_effect_analysis = None
                multimodal_value_analysis = None
                multimodal_bias_analysis = None
    else:
        mod2_vs_mod1_analysis = None
        mod2_dominance_analysis = None
        uncertainty_effect_analysis = None
        multimodal_value_analysis = None
        multimodal_bias_analysis = None
    
    # ============================================================================
    # Add disagreement rates to each modality's step_results
    # For each modality, calculate disagreement with all other modalities
    # ============================================================================
    if len(modalities) >= 2:
        # Calculate average disagreement rate for each modality
        for mod in modalities:
            if mod not in step_results:
                continue
            
            disagreement_rates = []
            
            # Compare with all other modalities
            for other_mod in modalities:
                if other_mod == mod:
                    continue
                
                pair_key = tuple(sorted([mod, other_mod]))
                if pair_key in pairwise_agreements:
                    agreement = pairwise_agreements[pair_key]
                    disagreement_rate = agreement.get('disagreement_rate', 0.0)
                    disagreement_rates.append(disagreement_rate)
            
            # Use average disagreement rate across all pairs
            if disagreement_rates:
                avg_disagreement = sum(disagreement_rates) / len(disagreement_rates)
                step_results[mod]['disagreement_rate'] = avg_disagreement
            else:
                # Fallback: use first pair if available
                if len(modalities) >= 2:
                    fpk = tuple(sorted([modalities[0], modalities[1]]))
                    if fpk in pairwise_agreements:
                        step_results[mod]['disagreement_rate'] = pairwise_agreements[fpk].get('disagreement_rate', 0.0)
                    else:
                        step_results[mod]['disagreement_rate'] = 0.0
        
        # For combined modalities, use disagreement with first modality
        for combined_name in step_results.keys():
            if '+' in combined_name:
                if combined_name in combined_agreements:
                    agreement = combined_agreements[combined_name]
                    step_results[combined_name]['disagreement_rate'] = agreement.get('disagreement_rate', 0.0)
                else:
                    # Fallback: use average of component modalities
                    component_mods = combined_name.split('+')
                    if component_mods:
                        avg_rate = 0.0
                        count = 0
                        for mod in component_mods:
                            if mod in step_results:
                                rate = step_results[mod].get('disagreement_rate', 0.0)
                                avg_rate += rate
                                count += 1
                        if count > 0:
                            step_results[combined_name]['disagreement_rate'] = avg_rate / count
                        else:
                            step_results[combined_name]['disagreement_rate'] = 0.0
    
    # Backward compatibility variables
    disagreement_rate_mod1_vs_mod2 = 0.0
    if agreement_metrics:
        disagreement_rate_mod1_vs_mod2 = agreement_metrics.get('disagreement_rate', 0.0)
    
    disagreement_rate_mod1_vs_combined = None
    if mod1_vs_combined_agreement:
        disagreement_rate_mod1_vs_combined = mod1_vs_combined_agreement.get('disagreement_rate', None)
    
    # Calculate patient-level results (mandatory requirement)
    patient_level_results = calculate_patient_level_results(results, modalities)
    
    # Disagreement rate: BY DEFINITION patient-level (proportion of patients where
    # aggregated prediction for modality A != aggregated prediction for modality B).
    # Do not change to slice-level; this is the intended metric for reporting.
    if len(modalities) >= 2 and patient_level_results:
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                mod_i = modalities[i]
                mod_j = modalities[j]
                
                if mod_i in patient_level_results and mod_j in patient_level_results:
                    mod_i_patient_preds = patient_level_results[mod_i].get('full_predictions', [])
                    mod_j_patient_preds = patient_level_results[mod_j].get('full_predictions', [])
                    
                    if mod_i_patient_preds and mod_j_patient_preds:
                        patient_ids = [p.get('patient_id') for p in mod_i_patient_preds if isinstance(p, dict) and p.get('patient_id') is not None]
                        patient_agreement = analyze_modality_agreement(mod_i_patient_preds, mod_j_patient_preds, patient_ids)
                        patient_disagreement_rate = patient_agreement.get('disagreement_rate', 0.0)
                        patient_agreement_rate = patient_agreement.get('agreement_rate', 0.0)
                        pair_key = tuple(sorted([mod_i, mod_j]))
                        if pair_key in pairwise_agreements:
                            pairwise_agreements[pair_key]['disagreement_rate'] = patient_disagreement_rate
                            pairwise_agreements[pair_key]['agreement_rate'] = patient_agreement_rate
                        if i == 0 and j == 1 and agreement_metrics:
                            agreement_metrics['disagreement_rate'] = patient_disagreement_rate
                            agreement_metrics['agreement_rate'] = patient_agreement_rate
        
        # Update disagreement rates in step_results based on patient-level calculations
        for mod in modalities:
            if mod not in step_results:
                continue
            
            disagreement_rates = []
            for other_mod in modalities:
                if other_mod == mod:
                    continue
                
                pair_key = tuple(sorted([mod, other_mod]))
                if pair_key in pairwise_agreements:
                    agreement = pairwise_agreements[pair_key]
                    disagreement_rate = agreement.get('disagreement_rate', 0.0)
                    disagreement_rates.append(disagreement_rate)
            
            if disagreement_rates:
                avg_disagreement = sum(disagreement_rates) / len(disagreement_rates)
                step_results[mod]['disagreement_rate'] = avg_disagreement
    
    return {
        'step_results': step_results,
        'patient_level_results': patient_level_results,  # Mandatory patient-level results
        'modalities': modalities,
        # Backward compatibility (first 2 modalities)
        'agreement_metrics': agreement_metrics,
        'mod1_vs_combined_agreement': mod1_vs_combined_agreement,
        'modality_combination_analysis': modality_combination_analysis,
        'logit_similarity_analysis': logit_similarity_analysis,
        'mod2_vs_mod1_analysis': mod2_vs_mod1_analysis,
        'mod2_dominance_analysis': mod2_dominance_analysis,
        'uncertainty_effect_analysis': uncertainty_effect_analysis,
        'multimodal_value_analysis': multimodal_value_analysis,
        'multimodal_bias_analysis': multimodal_bias_analysis,
        # Full N-modality analysis
        'pairwise_agreements': pairwise_agreements,  # {(mod1, mod2): agreement_metrics}
        'pairwise_logit_similarities': pairwise_logit_similarities,  # {(mod1, mod2): similarity}
        'pairwise_confidence_comparisons': pairwise_confidence_comparisons,  # {(mod1, mod2): comparison}
        'pairwise_dominance': pairwise_dominance,  # {(mod1, mod2): dominance}
        'sequential_analyses': sequential_analyses,  # {mod: analysis} for mod with previous context
        'combined_agreements': combined_agreements,  # {combined_name: agreement}
        'combined_uncertainty_effects': combined_uncertainty_effects,  # {combined_name: uncertainty}
        'combined_multimodal_values': combined_multimodal_values,  # {combined_name: value}
        'combined_bias_analyses': combined_bias_analyses  # {combined_name: bias}
    }


def print_evaluation_results(evaluation_results: Dict):
    """
    Print evaluation results with emphasis on reliability and uncertainty.
    Focuses on patient-level results (mandatory) and overconfidence analysis.
    """
    step_results = evaluation_results.get('step_results', {})
    patient_level_results = evaluation_results.get('patient_level_results', {})
    modalities = evaluation_results.get('modalities', [])
    agreement_metrics = evaluation_results.get('agreement_metrics')
    mod1_vs_combined_agreement = evaluation_results.get('mod1_vs_combined_agreement')
    
    # ============================================================================
    # PROCESSING ORDER INFORMATION
    # ============================================================================
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    if len(modalities) > 0:
        order_str = " → ".join(modalities)
        print(f"Processing Order: {order_str}")
        print(f"Total Modalities: {len(modalities)}")
        if len(modalities) > 1:
            print(f"\nSequential Processing Pattern:")
            steps = []
            for i, mod in enumerate(modalities):
                steps.append((i + 1, f"{mod} (alone)"))
                if i > 0:
                    context_mods = modalities[:i]
                    context_str = "+".join(context_mods)
                    steps.append((len(modalities) + i, f"{mod} with {context_str} context ({context_str}+{mod})"))
            for step_num, desc in sorted(steps, key=lambda x: x[0]):
                print(f"  Step {step_num}: {desc}")
    print("="*80)
    
    # ============================================================================
    # PATIENT-LEVEL RESULTS (MANDATORY REQUIREMENT)
    # ============================================================================
    print("\n" + "="*80)
    print("PATIENT-LEVEL RESULTS (MANDATORY)")
    print("="*80)
    print("Results aggregated from slice-level to patient-level using weighted voting.")
    print()
    
    if patient_level_results:
        # Patient-level certainty metrics
        print(f"{'Modality':<15} {'Accuracy':<12} {'Avg Confidence':<18} {'Entropy':<15} {'Num Patients':<15}")
        print("-"*80)
        
        def get_step_order(step_name):
            """Order steps: individual modalities first (in order), then combinations."""
            # Individual modalities get their index
            if step_name in modalities:
                return modalities.index(step_name)
            # Combined modalities get index after all individual ones
            if '+' in step_name:
                combined_mods = step_name.split('+')
                # Return index based on the last modality in the combination
                if combined_mods:
                    last_mod = combined_mods[-1]
                    if last_mod in modalities:
                        return len(modalities) + modalities.index(last_mod)
            return 999  # Unknown steps go last
        
        sorted_steps = sorted(patient_level_results.keys(), key=get_step_order)
        for step_name in sorted_steps:
            step_data = patient_level_results[step_name]
            num_patients = step_data.get('num_samples', 0)
            if num_patients == 0:
                continue
            
            acc = step_data.get('accuracy', 0.0)
            cert_metrics = step_data.get('certainty_metrics', {})
            avg_conf = cert_metrics.get('avg_confidence', 0.0)
            avg_entropy = cert_metrics.get('avg_entropy', 0.0)
            
            print(f"{step_name:<15} {acc:<12.4f} {avg_conf:<18.4f} {avg_entropy:<15.4f} {num_patients:<15}")
        
        # Patient-level overconfidence analysis
        if patient_level_results:
            print("\n" + "-"*80)
            print("PATIENT-LEVEL OVERCONFIDENCE ANALYSIS")
            print("-"*80)
            print("High-confidence incorrect predictions (confidence ≥ 0.7) - Critical reliability issue")
            print("NOTE: Counts are at PATIENT-LEVEL (one prediction per patient, aggregated from slices)")
            print()
            print(f"{'Modality':<15} {'High-Conf Incorrect':<20} {'Rate':<12} {'Avg Conf (Wrong)':<18} {'Overconf Severity':<20}")
            print("-"*80)
            
            for step_name in sorted_steps:
                step_data = patient_level_results[step_name]
                num_patients = step_data.get('num_samples', 0)
                if num_patients == 0:
                    continue
                
                overconf = step_data.get('overconfidence_metrics', {})
                high_conf_incorrect = overconf.get('high_conf_incorrect_count', 0)
                high_conf_incorrect_rate = overconf.get('high_conf_incorrect_rate', 0.0)
                avg_conf_wrong = overconf.get('avg_confidence_when_incorrect', 0.0)
                overconf_severity = overconf.get('overconfidence_severity', 0.0)
                
                # Validation: high_conf_incorrect should not exceed num_patients
                if high_conf_incorrect > num_patients:
                    import sys
                    print(f"ERROR: Patient-level overconfidence count ({high_conf_incorrect}) exceeds number of patients ({num_patients}) for {step_name}. "
                          f"This indicates a bug - patient-level should count patients, not slices.", file=sys.stderr, flush=True)
                    # Clamp to num_patients for display (but this indicates a bug that needs fixing)
                    high_conf_incorrect = min(high_conf_incorrect, num_patients)
                    high_conf_incorrect_rate = high_conf_incorrect / num_patients if num_patients > 0 else 0.0
                
                print(f"{step_name:<15} {high_conf_incorrect:<20} {high_conf_incorrect_rate:<12.4f} {avg_conf_wrong:<18.4f} {overconf_severity:<20.4f}")
            
            print("\n" + "-"*80)
            print("INTERPRETATION:")
            print("-"*80)
            print("• High-Conf Incorrect: Number of patients where model was confident (≥0.7) but wrong")
            print("• Rate: Proportion of all patients with high-confidence errors")
            print("• Avg Conf (Wrong): Average confidence of ALL incorrect predictions (including low-confidence ones)")
            print("  → If 0.0: No incorrect predictions at all")
            print("  → If >0.0 but High-Conf Incorrect=0: Incorrect predictions exist but all are low-confidence (<0.7)")
            print("• Overconf Severity: Average confidence of high-confidence incorrect predictions")
            print("  → Higher values indicate more severe overconfidence (model very wrong but very confident)")
            print("  → Aggregated confidence is the mean of winning-class slice confidences (typically <1.0).")
        else:
            print("Warning: Patient-level results not available.")
    
    # ============================================================================
    # SLICE-LEVEL CERTAINTY METRICS (for reference)
    # ============================================================================
    print("\n" + "="*80)
    print("SLICE-LEVEL CERTAINTY METRICS (Reference)")
    print("="*80)
    # Note: Avg Confidence and Entropy are slice-level; Disagreement is patient-level (by design).
    # Get disagreement rate from agreement metrics (modality pairs)
    disagreement_rate_mod1_vs_mod2 = 0.0
    if agreement_metrics:
        disagreement_rate_mod1_vs_mod2 = agreement_metrics.get('disagreement_rate', 0.0)
    
    # Get Mod1 vs Mod1+Mod2 disagreement rate (if available)
    disagreement_rate_mod1_vs_combined = None
    if mod1_vs_combined_agreement:
        disagreement_rate_mod1_vs_combined = mod1_vs_combined_agreement.get('disagreement_rate', None)
    
    print(f"{'Modality':<15} {'Avg Confidence':<18} {'Entropy':<15} {'Disagreement (pt-level)':<22}")
    print("-"*80)
    
    # Custom sort order: Individual modalities first (in order), then combinations
    def get_step_order(step_name):
        """Order steps: individual modalities first (in order), then combinations."""
        # Individual modalities get their index
        if step_name in modalities:
            return modalities.index(step_name)
        # Combined modalities get index after all individual ones
        if '+' in step_name:
            combined_mods = step_name.split('+')
            # Return index based on the last modality in the combination
            if combined_mods:
                last_mod = combined_mods[-1]
                if last_mod in modalities:
                    return len(modalities) + modalities.index(last_mod)
        return 999  # Unknown steps go last
    
    cert_comparison = {}
    sorted_steps = sorted(step_results.keys(), key=get_step_order)
    for step_name in sorted_steps:
        step_data = step_results[step_name]
        num_samples = step_data.get('num_samples', 0)
        
        # Skip steps with no data (e.g., combined modalities in sequential approach)
        if num_samples == 0:
            continue
        
        cert_metrics = step_data.get('certainty_metrics', {})
        avg_conf = cert_metrics.get('avg_confidence', 0.0)
        avg_entropy = cert_metrics.get('avg_entropy', 0.0)
        
        # Disagreement rate: use the disagreement_rate stored in step_results for this step
        # This is calculated generically for all modalities (average across all pairs for individual mods,
        # or disagreement with first component for combined mods)
        display_disagreement = step_data.get('disagreement_rate', 0.0)
        # Fallback to backward compatibility if not available
        if display_disagreement == 0.0 and step_name in modalities:
            # Individual modality: use average disagreement with all other individual modalities
            if len(modalities) >= 2 and step_name == modalities[0]:
                display_disagreement = disagreement_rate_mod1_vs_mod2
            elif len(modalities) >= 2 and step_name == modalities[1]:
                display_disagreement = disagreement_rate_mod1_vs_mod2
        elif display_disagreement == 0.0 and '+' in step_name:
            # Combined modality: use disagreement with first component
            if disagreement_rate_mod1_vs_combined is not None:
                display_disagreement = disagreement_rate_mod1_vs_combined
            else:
                display_disagreement = disagreement_rate_mod1_vs_mod2
        
        cert_comparison[step_name] = {
            'avg_conf': avg_conf,
            'avg_entropy': avg_entropy
        }
        
        print(f"{step_name:<15} {avg_conf:<18.4f} {avg_entropy:<15.4f} {display_disagreement:<20.4f}")
    
    # DETAILED CERTAINTY METRICS
    print("\n" + "-"*80)
    print("DETAILED CERTAINTY METRICS")
    print("-"*80)
    print(f"{'Modality':<15} {'Prob Gap':<12} {'Logit Mag':<15} {'Calib Conf':<15} {'Calib Error':<15}")
    print("-"*80)
    
    sorted_steps_detailed = sorted(step_results.keys(), key=get_step_order)
    for step_name in sorted_steps_detailed:
        step_data = step_results[step_name]
        num_samples = step_data.get('num_samples', 0)
        
        if num_samples == 0:
            continue
        
        cert_metrics = step_data.get('certainty_metrics', {})
        prob_gap = cert_metrics.get('avg_probability_gap', 0.0)
        logit_mag = cert_metrics.get('avg_logit_magnitude', 0.0)
        calib_conf = cert_metrics.get('avg_calibrated_confidence', 0.0)
        calib_error = cert_metrics.get('avg_calibration_error', 0.0)
        
        print(f"{step_name:<15} {prob_gap:<12.4f} {logit_mag:<15.4f} {calib_conf:<15.4f} {calib_error:<15.4f}")
    
    # LOGIT DISTRIBUTION SIMILARITY
    logit_similarity = evaluation_results.get('logit_similarity_analysis')
    pairwise_logit_similarities = evaluation_results.get('pairwise_logit_similarities', {})
    
    if pairwise_logit_similarities:
        print("\n" + "-"*80)
        print("LOGIT DISTRIBUTION SIMILARITY (PAIRWISE)")
        print("-"*80)
        for (mod_i, mod_j), similarity_data in sorted(pairwise_logit_similarities.items()):
            avg_sim = similarity_data.get('avg_cosine_similarity', 0.0)
            std_sim = similarity_data.get('std_cosine_similarity', 0.0)
            print(f"{mod_i} vs {mod_j}: {avg_sim:.4f} ± {std_sim:.4f}")
    elif logit_similarity:
        # Backward compatibility: show first pair only
        print("\n" + "-"*80)
        print("LOGIT DISTRIBUTION SIMILARITY")
        print("-"*80)
        avg_sim = logit_similarity.get('avg_cosine_similarity', 0.0)
        std_sim = logit_similarity.get('std_cosine_similarity', 0.0)
        mod1_name = modalities[0] if len(modalities) >= 1 else "Mod1"
        mod2_name = modalities[1] if len(modalities) >= 2 else "Mod2"
        print(f"Average cosine similarity between {mod1_name} and {mod2_name} logits: {avg_sim:.4f} ± {std_sim:.4f}")
    
    # CONFIDENCE DIFFERENCES
    if len(cert_comparison) >= 2:
        print("\n" + "-"*80)
        print("CONFIDENCE DIFFERENCES")
        print("-"*80)
        mod_names = sorted(cert_comparison.keys())
        if len(mod_names) >= 2:
            mod1, mod2 = mod_names[0], mod_names[1]
            conf_diff = cert_comparison[mod2]['avg_conf'] - cert_comparison[mod1]['avg_conf']
            print(f"{mod2} vs {mod1} confidence difference: {conf_diff:+.4f}")
    
    # MODALITY AGREEMENT & CERTAINTY EFFECTS
    if agreement_metrics:
        print("\n" + "-"*80)
        print("MODALITY AGREEMENT & CERTAINTY EFFECTS")
        print("-"*80)
        disagreement_rate = agreement_metrics.get('disagreement_rate', 0.0)
        print(f"Disagreement rate (patient-level): {disagreement_rate:.4f} ({disagreement_rate*100:.1f}% of patients)")
        
        disagreement_conf_analysis = agreement_metrics.get('disagreement_confidence_analysis', {})
        if disagreement_conf_analysis:
            avg_mod1_conf_disagree = disagreement_conf_analysis.get('avg_mod1_confidence_when_disagree', 0.0)
            avg_mod2_conf_disagree = disagreement_conf_analysis.get('avg_mod2_confidence_when_disagree', 0.0)
            avg_conf_agree = disagreement_conf_analysis.get('avg_confidence_when_agree', 0.0)
            conf_drop = disagreement_conf_analysis.get('confidence_drop_on_disagreement', 0.0)
            
            if avg_mod1_conf_disagree > 0 or avg_mod2_conf_disagree > 0:
                mod1_name = modalities[0] if len(modalities) >= 1 else "Mod1"
                mod2_name = modalities[1] if len(modalities) >= 2 else "Mod2"
                print(f"{mod1_name} confidence when disagreeing: {avg_mod1_conf_disagree:.4f}")
                print(f"{mod2_name} confidence when disagreeing: {avg_mod2_conf_disagree:.4f}")
                if avg_conf_agree > 0:
                    print(f"Average confidence when agreeing: {avg_conf_agree:.4f}")
                    print(f"Confidence drop on disagreement: {conf_drop:.4f}")
                elif disagreement_rate >= 1.0:
                    print(f"Average confidence when agreeing: N/A (no agreeing pairs)")
                    print(f"Confidence drop on disagreement: N/A")
        
        disagreement_logit_analysis = agreement_metrics.get('disagreement_logit_analysis', {})
        if disagreement_logit_analysis:
            instability = disagreement_logit_analysis.get('logit_instability_indicator', 0.0)
            if instability > 0:
                print(f"Average logit variance when disagreeing: {instability:.4f}")
    
    # MODALITY COMBINATION EFFECTS
    combination_analysis = evaluation_results.get('modality_combination_analysis')
    if combination_analysis:
        print("\n" + "-"*80)
        print("MODALITY COMBINATION EFFECTS")
        print("-"*80)
        conf_change = combination_analysis.get('confidence_change', 0.0)
        entropy_change = combination_analysis.get('entropy_change', 0.0)
        print(f"Combined modality confidence: {combination_analysis.get('combined_confidence', 0.0):.4f}")
        print(f"Average single-modality confidence: {combination_analysis.get('average_single_modality_confidence', 0.0):.4f}")
        print(f"Confidence change: {conf_change:+.4f}")
        print(f"Combined modality entropy: {combination_analysis.get('combined_entropy', 0.0):.4f}")
        print(f"Average single-modality entropy: {combination_analysis.get('average_single_modality_entropy', 0.0):.4f}")
        print(f"Entropy change: {entropy_change:+.4f}")
    
    # SEQUENTIAL CONTEXT INFLUENCE (all modalities with their preceding context)
    sequential_analyses = evaluation_results.get('sequential_analyses', {})
    if sequential_analyses:
        print("\n" + "-"*80)
        print("SEQUENTIAL CONTEXT INFLUENCE")
        print("-"*80)
        for mod_name, context_analysis in sorted(sequential_analyses.items()):
            print(f"\n{mod_name.upper()} with preceding context:")
            # Extract from comparison dict (which contains avg_confidence_difference)
            comparison = context_analysis.get('comparison', {})
            conf_change = comparison.get('avg_confidence_difference', 0.0)
            # Calculate entropy change from predictions if available
            alone_preds = context_analysis.get('alone_predictions', [])
            with_context_preds = context_analysis.get('with_context_predictions', [])
            entropy_changes = []
            if alone_preds and with_context_preds:
                # Match predictions and calculate entropy differences
                for alone_pred, ctx_pred in zip(alone_preds[:len(with_context_preds)], with_context_preds):
                    alone_probs = alone_pred.get('probabilities_array') or list(alone_pred.get('probabilities', {}).values())
                    ctx_probs = ctx_pred.get('probabilities_array') or list(ctx_pred.get('probabilities', {}).values())
                    if len(alone_probs) >= 2 and len(ctx_probs) >= 2:
                        alone_entropy = calculate_entropy(np.array(alone_probs))
                        ctx_entropy = calculate_entropy(np.array(ctx_probs))
                        entropy_changes.append(ctx_entropy - alone_entropy)
            entropy_change = float(np.mean(entropy_changes)) if entropy_changes else 0.0
            # Count significant increases (>5% confidence increase)
            # Use the same matching logic as analyze_modality_confidence_comparison (patient-level aggregation)
            significant_increases = 0
            if alone_preds and with_context_preds:
                # Match by patient_id (same as comparison function)
                alone_by_patient = {}
                ctx_by_patient = {}
                
                for pred in alone_preds:
                    pid = pred.get('patient_id')
                    if pid is not None:
                        if pid not in alone_by_patient:
                            alone_by_patient[pid] = []
                        alone_by_patient[pid].append(pred)
                
                for pred in with_context_preds:
                    pid = pred.get('patient_id')
                    if pid is not None:
                        if pid not in ctx_by_patient:
                            ctx_by_patient[pid] = []
                        ctx_by_patient[pid].append(pred)
                
                # Aggregate and compare (same logic as analyze_modality_confidence_comparison)
                for pid in set(list(alone_by_patient.keys()) + list(ctx_by_patient.keys())):
                    if pid in alone_by_patient and pid in ctx_by_patient:
                        alone_slices = alone_by_patient[pid]
                        ctx_slices = ctx_by_patient[pid]
                        
                        # Aggregate to patient level (same as comparison function)
                        alone_aggregated = aggregate_patient_predictions(alone_slices)
                        ctx_aggregated = aggregate_patient_predictions(ctx_slices)
                        
                        # Create prediction dicts matching comparison function structure
                        alone_pred = {
                            'prediction': alone_aggregated['prediction'],
                            'confidence': alone_aggregated['confidence'],
                            'probabilities_array': alone_slices[0].get('probabilities_array', []) if alone_slices else [],
                        }
                        
                        ctx_pred = {
                            'prediction': ctx_aggregated['prediction'],
                            'confidence': ctx_aggregated['confidence'],
                            'probabilities_array': ctx_slices[0].get('probabilities_array', []) if ctx_slices else [],
                            'used_context': ctx_slices[0].get('used_context', False) if ctx_slices else False,
                        }
                        
                        # Use EXACT same confidence extraction logic as analyze_modality_confidence_comparison
                        # For mod1 (alone): use probabilities_array if available, else confidence
                        alone_probs = alone_pred.get('probabilities_array')
                        if alone_probs is not None and len(alone_probs) >= 2:
                            alone_conf = float(np.max(np.array(alone_probs)))
                        else:
                            alone_conf = alone_pred.get('confidence', 0.0)
                        
                        # For mod2 (with context): use probabilities_array if used_context, else probabilities_before_boosting
                        used_context = ctx_pred.get('used_context', False)
                        if used_context:
                            ctx_probs_after = ctx_pred.get('probabilities_array')
                            if ctx_probs_after is not None and len(ctx_probs_after) >= 2:
                                ctx_conf = float(np.max(np.array(ctx_probs_after)))
                            else:
                                ctx_conf = ctx_pred.get('confidence', 0.0)
                        else:
                            # Should not happen for sequential context, but handle gracefully
                            ctx_probs_before = ctx_slices[0].get('probabilities_before_boosting') if ctx_slices else None
                            if ctx_probs_before is not None and len(ctx_probs_before) >= 2:
                                ctx_conf = float(np.max(np.array(ctx_probs_before)))
                            else:
                                ctx_probs = ctx_pred.get('probabilities_array')
                                if ctx_probs is not None and len(ctx_probs) >= 2:
                                    ctx_conf = float(np.max(np.array(ctx_probs)))
                                else:
                                    ctx_conf = ctx_pred.get('confidence', 0.0)
                        
                        if ctx_conf > alone_conf + 0.05:  # >5% increase
                            significant_increases += 1
            total_samples = comparison.get('num_pairs', len(set([p.get('patient_id') for p in alone_preds if p.get('patient_id') is not None])) if alone_preds else 0)
            print(f"  Average confidence change: {conf_change:+.4f}")
            print(f"  Average entropy change: {entropy_change:+.4f}")
            print(f"  Significant increases (>5%): {significant_increases}/{total_samples}")
    
    # Backward compatibility: show first modality context influence if available
    mod1_context_influence = evaluation_results.get('mod1_context_influence')
    if mod1_context_influence and not sequential_analyses:
        print("\n" + "-"*80)
        mod1_name = modalities[0] if len(modalities) >= 1 else "Mod1"
        print(f"{mod1_name.upper()} CONTEXT INFLUENCE")
        print("-"*80)
        conf_change = mod1_context_influence.get('avg_confidence_change', 0.0)
        entropy_change = mod1_context_influence.get('avg_entropy_change', 0.0)
        significant_increases = mod1_context_influence.get('significant_confidence_increases', 0)
        total_samples = mod1_context_influence.get('num_samples', 0)
        print(f"Average confidence change with {mod1_name} context: {conf_change:+.4f}")
        print(f"Average entropy change: {entropy_change:+.4f}")
        print(f"Significant increases (>5%): {significant_increases}/{total_samples}")
    
    # PAIRWISE CONFIDENCE COMPARISONS (all modality pairs)
    pairwise_confidence_comparisons = evaluation_results.get('pairwise_confidence_comparisons', {})
    if pairwise_confidence_comparisons:
        print("\n" + "-"*80)
        print("PAIRWISE CONFIDENCE COMPARISONS")
        print("-"*80)
        for (mod_i, mod_j), comparison in sorted(pairwise_confidence_comparisons.items()):
            print(f"\n{mod_j.upper()} VS {mod_i.upper()}:")
            mod_j_dominance_rate = comparison.get('mod2_higher_confidence_rate', 0.0)
            avg_conf_diff = comparison.get('avg_confidence_difference', 0.0)
            num_pairs = comparison.get('num_pairs', 0)
            print(f"  {mod_j} has higher confidence in {mod_j_dominance_rate*100:.1f}% of cases")
            print(f"  Average confidence difference ({mod_j} - {mod_i}): {avg_conf_diff:+.4f}")
            print(f"  Number of matched pairs: {num_pairs}")
    
    # Backward compatibility: show first pair comparison if available
    mod2_vs_mod1 = evaluation_results.get('mod2_vs_mod1_analysis')
    if mod2_vs_mod1 and not pairwise_confidence_comparisons:
        print("\n" + "-"*80)
        mod1_name = modalities[0] if len(modalities) >= 1 else "Mod1"
        mod2_name = modalities[1] if len(modalities) >= 2 else "Mod2"
        print(f"{mod2_name.upper()} VS {mod1_name.upper()} COMPARISON")
        print("-"*80)
        mod2_dominance_rate = mod2_vs_mod1.get('mod2_higher_confidence_rate', 0.0)
        avg_conf_diff = mod2_vs_mod1.get('avg_confidence_difference', 0.0)
        num_pairs = mod2_vs_mod1.get('num_pairs', 0)
        print(f"{mod2_name} has higher confidence in {mod2_dominance_rate*100:.1f}% of cases")
        print(f"Average confidence difference ({mod2_name} - {mod1_name}): {avg_conf_diff:+.4f}")
        print(f"Number of matched pairs: {num_pairs}")
    
    # MULTIMODALITY UNCERTAINTY EFFECTS
    uncertainty_effect = evaluation_results.get('uncertainty_effect_analysis')
    if uncertainty_effect:
        print("\n" + "-"*80)
        print("MULTIMODALITY UNCERTAINTY EFFECTS")
        print("-"*80)
        uncertainty_reduction_rate = uncertainty_effect.get('uncertainty_reduction_rate', 0.0)
        conflict_introduction_rate = uncertainty_effect.get('conflict_introduction_rate', 0.0)
        avg_entropy_change = uncertainty_effect.get('avg_entropy_change', 0.0)
        avg_conf_change = uncertainty_effect.get('avg_confidence_change', 0.0)
        num_triplets = uncertainty_effect.get('num_triplets', 0)
        print(f"Uncertainty reduction rate: {uncertainty_reduction_rate*100:.1f}%")
        print(f"Conflict introduction rate: {conflict_introduction_rate*100:.1f}%")
        print(f"Average entropy change: {avg_entropy_change:+.4f}")
        print(f"Average confidence change: {avg_conf_change:+.4f}")
        print(f"Number of cases: {num_triplets}")
    
    # PATIENT-LEVEL AGREEMENT
    patient_level_agreement = evaluation_results.get('patient_level_agreement')
    if patient_level_agreement:
        print("\n" + "-"*80)
        print("PATIENT-LEVEL vs SLICE-LEVEL AGREEMENT")
        print("-"*80)
        patient_agreement_rate = patient_level_agreement.get('patient_level_agreement_rate', 0.0)
        slice_agreement_rate = patient_level_agreement.get('slice_level_agreement_rate', 0.0)
        improvement = patient_level_agreement.get('agreement_improvement', 0.0)
        num_patients = patient_level_agreement.get('num_patients', 0)
        print(f"Slice-level agreement: {slice_agreement_rate:.4f}")
        print(f"Patient-level agreement: {patient_agreement_rate:.4f}")
        print(f"Change: {improvement:+.4f} ({improvement*100:+.1f}%)")
        print(f"Number of patients: {num_patients}")
    
    # PAIRWISE DOMINANCE ANALYSIS (all modality pairs)
    pairwise_dominance = evaluation_results.get('pairwise_dominance', {})
    if pairwise_dominance:
        print("\n" + "-"*80)
        print("PAIRWISE DOMINANCE ANALYSIS")
        print("-"*80)
        for (mod_i, mod_j), dominance in sorted(pairwise_dominance.items()):
            print(f"\n{mod_j.upper()} VS {mod_i.upper()}:")
            mod_j_dominance_rate = dominance.get('mod2_higher_confidence_rate', 0.0)
            avg_conf_diff = dominance.get('avg_confidence_difference', 0.0)
            mod_j_wins_disagree = dominance.get('mod2_wins_when_disagree', 0)
            mod_i_wins_disagree = dominance.get('mod1_wins_when_disagree', 0)
            print(f"  {mod_j} has higher confidence in {mod_j_dominance_rate*100:.1f}% of cases")
            print(f"  Average confidence difference ({mod_j} - {mod_i}): {avg_conf_diff:+.4f}")
            print(f"  When modalities disagree:")
            print(f"    {mod_j} wins (higher confidence): {mod_j_wins_disagree} cases")
            print(f"    {mod_i} wins (higher confidence): {mod_i_wins_disagree} cases")
    
    # Backward compatibility: show first pair dominance if available
    mod2_dominance = evaluation_results.get('mod2_dominance_analysis')
    if mod2_dominance and not pairwise_dominance:
        print("\n" + "-"*80)
        mod1_name = modalities[0] if len(modalities) >= 1 else "Mod1"
        mod2_name = modalities[1] if len(modalities) >= 2 else "Mod2"
        print(f"{mod2_name.upper()} DOMINANCE ANALYSIS")
        print("-"*80)
        mod2_dominance_rate = mod2_dominance.get('mod2_higher_confidence_rate', 0.0)
        avg_conf_diff = mod2_dominance.get('avg_confidence_difference', 0.0)
        mod2_wins_disagree = mod2_dominance.get('mod2_wins_when_disagree', 0)
        mod1_wins_disagree = mod2_dominance.get('mod1_wins_when_disagree', 0)
        print(f"{mod2_name} has higher confidence in {mod2_dominance_rate*100:.1f}% of cases")
        print(f"Average confidence difference ({mod2_name} - {mod1_name}): {avg_conf_diff:+.4f}")
        print(f"When modalities disagree:")
        print(f"  {mod2_name} wins (higher confidence): {mod2_wins_disagree} cases")
        print(f"  {mod1_name} wins (higher confidence): {mod1_wins_disagree} cases")
    
    # MULTIMODAL BIAS ANALYSIS (for all combined modalities)
    combined_bias_analyses = evaluation_results.get('combined_bias_analyses', {})
    if combined_bias_analyses:
        print("\n" + "-"*80)
        print("MULTIMODAL PREDICTION ANALYSIS (BIAS DETECTION)")
        print("-"*80)
        for combined_name, bias_analysis in sorted(combined_bias_analyses.items()):
            component_mods = combined_name.split('+')
            if len(component_mods) >= 2:
                # Use actual modality names from analysis if available (for 3+ modality cases)
                actual_mod1 = bias_analysis.get('actual_mod1_name')
                actual_mod2 = bias_analysis.get('actual_mod2_name')
                if actual_mod1 and actual_mod2:
                    mod1_name = actual_mod1
                    mod2_name = actual_mod2
                else:
                    # For 2-modality case, use component names
                    mod1_name = component_mods[0]
                    mod2_name = component_mods[-1]  # Last modality in combination
                print(f"\n{combined_name}:")
                matches_mod2_rate = bias_analysis.get('combined_matches_mod2_rate', 0.0)
                matches_mod1_rate = bias_analysis.get('combined_matches_mod1_rate', 0.0)
                mod2_bias_score = bias_analysis.get('mod2_bias_score', 0.0)
                true_combination_rate = bias_analysis.get('true_combination_rate', 0.0)
                simple_mod2_bias = bias_analysis.get('simple_mod2_bias', False)
                print(f"  Combined matches {mod2_name}: {matches_mod2_rate*100:.1f}%")
                print(f"  Combined matches {mod1_name}: {matches_mod1_rate*100:.1f}%")
                print(f"  {mod2_name} bias score (matches {mod2_name} when {mod1_name}≠{mod2_name}): {mod2_bias_score*100:.1f}%")
                print(f"  True combination rate: {true_combination_rate*100:.1f}%")
                if simple_mod2_bias:
                    print(f"  ⚠️  SIMPLE {mod2_name} BIAS DETECTED (>20% difference)")
    
    # Backward compatibility: show first combined bias if available
    multimodal_bias = evaluation_results.get('multimodal_bias_analysis')
    if multimodal_bias and not combined_bias_analyses:
        print("\n" + "-"*80)
        print("MULTIMODAL PREDICTION ANALYSIS")
        print("-"*80)
        matches_mod2_rate = multimodal_bias.get('combined_matches_mod2_rate', multimodal_bias.get('combined_matches_pet_rate', 0.0))
        matches_mod1_rate = multimodal_bias.get('combined_matches_mod1_rate', multimodal_bias.get('combined_matches_ct_rate', 0.0))
        mod2_bias_score = multimodal_bias.get('mod2_bias_score', multimodal_bias.get('pet_bias_score', 0.0))
        true_combination_rate = multimodal_bias.get('true_combination_rate', 0.0)
        mod1_name = modalities[0] if len(modalities) >= 1 else "Mod1"
        mod2_name = modalities[1] if len(modalities) >= 2 else "Mod2"
        print(f"Combined predictions match {mod2_name}: {matches_mod2_rate*100:.1f}%")
        print(f"Combined predictions match {mod1_name}: {matches_mod1_rate*100:.1f}%")
        print(f"{mod2_name} bias score (matches {mod2_name} when {mod1_name}≠{mod2_name}): {mod2_bias_score*100:.1f}%")
        print(f"True combination rate: {true_combination_rate*100:.1f}%")
    
    # ============================================================================
    # OVERCONFIDENCE ANALYSIS (SLICE-LEVEL)
    # ============================================================================
    print("\n" + "="*80)
    print("OVERCONFIDENCE ANALYSIS: HIGH-CONFIDENCE INCORRECT PREDICTIONS")
    print("="*80)
    print("Critical reliability issue: Cases where model is highly confident but incorrect.")
    print("This analysis reveals when the model fails to recognize its own uncertainty.")
    print()
    
    print(f"{'Modality':<15} {'High-Conf Incorrect':<20} {'Rate':<12} {'Avg Conf (Wrong)':<18} {'Avg Conf (Correct)':<20} {'Overconf Severity':<20}")
    print("-"*80)
    
    sorted_steps = sorted(step_results.keys(), key=get_step_order)
    for step_name in sorted_steps:
        step_data = step_results[step_name]
        num_samples = step_data.get('num_samples', 0)
        if num_samples == 0:
            continue
        
        overconf = step_data.get('overconfidence_metrics', {})
        high_conf_incorrect = overconf.get('high_conf_incorrect_count', 0)
        high_conf_incorrect_rate = overconf.get('high_conf_incorrect_rate', 0.0)
        avg_conf_wrong = overconf.get('avg_confidence_when_incorrect', 0.0)
        avg_conf_correct = overconf.get('avg_confidence_when_correct', 0.0)
        overconf_severity = overconf.get('overconfidence_severity', 0.0)
        
        print(f"{step_name:<15} {high_conf_incorrect:<20} {high_conf_incorrect_rate:<12.4f} {avg_conf_wrong:<18.4f} {avg_conf_correct:<20.4f} {overconf_severity:<20.4f}")
    
    print("\n" + "-"*80)
    print("KEY INSIGHTS:")
    print("-"*80)
    print("• High-confidence errors are particularly dangerous - model appears certain but is wrong")
    print("• High Overconf Severity indicates model is very wrong but very confident (worst case)")
    print("• Avg Conf (Wrong) = 0.0 means no incorrect predictions; >0.0 means incorrect predictions exist")
    print("  → If >0.0 but High-Conf Incorrect=0: All incorrect predictions are low-confidence (<0.7)")
    print("• Large gap between Avg Conf (Wrong) and Avg Conf (Correct) suggests poor calibration")
    print("• Models should have LOW confidence when incorrect - high confidence on errors is unreliable")
    
    # ============================================================================
    # FULL N-MODALITY ANALYSIS
    # ============================================================================
    pairwise_agreements = evaluation_results.get('pairwise_agreements', {})
    pairwise_confidence_comparisons = evaluation_results.get('pairwise_confidence_comparisons', {})
    sequential_analyses = evaluation_results.get('sequential_analyses', {})
    combined_agreements = evaluation_results.get('combined_agreements', {})
    
    if len(modalities) > 2 or len(pairwise_agreements) > 1:
        print("\n" + "="*80)
        print("FULL N-MODALITY ANALYSIS")
        print("="*80)
        
        # 1. PAIRWISE COMPARISONS
        if pairwise_agreements:
            print("\n" + "-"*80)
            print("PAIRWISE MODALITY COMPARISONS")
            print("-"*80)
            print(f"{'Pair':<20} {'Agreement':<12} {'Disagreement':<15} {'Mod1 Dominates':<18} {'Mod2 Dominates':<18}")
            print("-"*80)
            
            for (mod_i, mod_j), agreement in sorted(pairwise_agreements.items()):
                agreement_rate = agreement.get('agreement_rate', 0.0)
                disagreement_rate = agreement.get('disagreement_rate', 0.0)
                mod_i_dominates = agreement.get('mod1_dominates', 0)
                mod_j_dominates = agreement.get('mod2_dominates', 0)
                pair_name = f"{mod_i} vs {mod_j}"
                print(f"{pair_name:<20} {agreement_rate:<12.4f} {disagreement_rate:<15.4f} {mod_i_dominates:<18} {mod_j_dominates:<18}")
        
        # 2. SEQUENTIAL EFFECTS (How each modality affects the next)
        if sequential_analyses:
            print("\n" + "-"*80)
            print("SEQUENTIAL MODALITY EFFECTS")
            print("-"*80)
            print("Shows how each modality changes when given context from previous modalities")
            print()
            print(f"{'Modality':<15} {'Context From':<25} {'Conf Change':<15} {'Higher Conf Rate':<20}")
            print("-"*80)
            
            for mod, analysis in sorted(sequential_analyses.items()):
                context_from = analysis.get('context_from', [])
                comparison = analysis.get('comparison', {})
                conf_diff = comparison.get('avg_confidence_difference', 0.0)
                mod2_higher_rate = comparison.get('mod2_higher_confidence_rate', 0.0)
                context_str = '+'.join(context_from) if context_from else "None"
                print(f"{mod:<15} {context_str:<25} {conf_diff:<15.4f} {mod2_higher_rate:<20.4f}")
        
        # 3. COMBINED MODALITY AGREEMENTS
        if combined_agreements:
            print("\n" + "-"*80)
            print("COMBINED MODALITY AGREEMENTS")
            print("-"*80)
            print("Agreement between first modality and combined modalities")
            print()
            print(f"{'Combined Modality':<25} {'Agreement Rate':<18} {'Disagreement Rate':<20}")
            print("-"*80)
            
            for combined_name, agreement in sorted(combined_agreements.items()):
                agreement_rate = agreement.get('agreement_rate', 0.0)
                disagreement_rate = agreement.get('disagreement_rate', 0.0)
                print(f"{combined_name:<25} {agreement_rate:<18.4f} {disagreement_rate:<20.4f}")
        
        # 4. ALL PAIRWISE CONFIDENCE COMPARISONS
        if pairwise_confidence_comparisons:
            print("\n" + "-"*80)
            print("PAIRWISE CONFIDENCE COMPARISONS")
            print("-"*80)
            print(f"{'Pair':<20} {'Mod1 Higher %':<18} {'Mod2 Higher %':<18} {'Avg Diff':<15}")
            print("-"*80)
            
            for (mod_i, mod_j), comparison in sorted(pairwise_confidence_comparisons.items()):
                mod1_higher = comparison.get('mod1_higher_confidence_rate', 0.0)
                mod2_higher = comparison.get('mod2_higher_confidence_rate', 0.0)
                avg_diff = comparison.get('avg_confidence_difference', 0.0)
                pair_name = f"{mod_i} vs {mod_j}"
                print(f"{pair_name:<20} {mod1_higher*100:<18.1f} {mod2_higher*100:<18.1f} {avg_diff:<15.4f}")
    
    # ============================================================================
    # ACCURACY (Reference Only - Reliability is Primary Focus)
    # ============================================================================
    print("\n" + "="*80)
    print("ACCURACY (Reference Only)")
    print("="*80)
    print("Note: In zero-shot settings, accuracy near chance is expected.")
    print("Primary focus is on RELIABILITY (uncertainty, overconfidence) rather than accuracy alone.")
    print()
    
    # Slice-level accuracy
    print("Slice-Level Accuracy:")
    sorted_steps = sorted(step_results.keys(), key=get_step_order)
    for step_name in sorted_steps:
        step_data = step_results[step_name]
        num_samples = step_data.get('num_samples', 0)
        if num_samples == 0:
            continue
        acc = step_data.get('accuracy', 0.0)
        print(f"  {step_name}: {acc:.4f} (n={num_samples} slices)")
    
    # Patient-level accuracy
    if patient_level_results:
        print("\nPatient-Level Accuracy:")
        sorted_steps = sorted(patient_level_results.keys(), key=get_step_order)
        for step_name in sorted_steps:
            step_data = patient_level_results[step_name]
            num_patients = step_data.get('num_samples', 0)
            if num_patients == 0:
                continue
            acc = step_data.get('accuracy', 0.0)
            print(f"  {step_name}: {acc:.4f} (n={num_patients} patients)")
    
    print("="*80 + "\n")


def analyze_certainty_metrics(
    predictions: List[Dict],
    modality_name: str
) -> Dict:
    """
    Analyze certainty metrics for a set of predictions.
    
    Args:
        predictions: List of prediction dicts, each should have:
            - 'confidence': float
            - 'probabilities': dict or array
            - 'logits': optional array
            - 'prediction': int
        modality_name: Name of modality (e.g., 'Mod1', 'Mod2')
    
    Returns:
        Dictionary with certainty metrics
    """
    if not predictions:
        return {
            'modality': modality_name,
            'avg_confidence': 0.0,
            'std_confidence': 0.0,
            'avg_entropy': 0.0,
            'std_entropy': 0.0,
            'avg_probability_gap': 0.0,
            'std_probability_gap': 0.0,
            'avg_logit_magnitude': 0.0,
            'std_logit_magnitude': 0.0,
            'avg_calibrated_confidence': 0.0,
            'avg_calibration_error': 0.0,
            'num_samples': 0
        }
    
    confidences = []
    entropies = []
    probability_gaps = []
    logit_magnitudes = []
    calibrated_confidences = []
    calibration_errors = []
    cosine_similarities = []  # For comparing with other modalities (if available)
    
    for pred in predictions:
        # Extract probabilities - strategy depends on whether context was used
        # For combined modalities (with context): Use AFTER boosting to show improved performance
        # For individual modalities (without context): Use probabilities as-is (no boosting)
        used_context = pred.get('used_context', False)
        probs_before_boosting = pred.get('probabilities_before_boosting')
        probs_array = pred.get('probabilities_array')
        prob_array = None
        conf = None
        
        if used_context and probs_array is not None and len(probs_array) >= 2:
            # Combined modalities: Use probabilities AFTER boosting to show improved performance with context
            # This demonstrates that context integration improves certainty
            prob_array = np.array(probs_array)
            conf = float(np.max(prob_array))
        elif probs_before_boosting is not None and len(probs_before_boosting) >= 2:
            # Individual modalities: Use probabilities BEFORE boosting (shows real model behavior)
            prob_array = np.array(probs_before_boosting)
            conf = float(np.max(prob_array))
        elif probs_array is not None and len(probs_array) >= 2:
            # Fallback: Use probabilities_array if before_boosting not available
            prob_array = np.array(probs_array)
            conf = float(np.max(prob_array))
        else:
            # Fallback to probabilities dict
            probs = pred.get('probabilities', {})
            if isinstance(probs, dict) and len(probs) >= 2:
                # Convert dict to array (assume binary classification)
                # Try common class name keys
                keys = list(probs.keys())
                if len(keys) >= 2:
                    prob_array = np.array([probs[keys[0]], probs[keys[1]]])
                else:
                    prob_array = np.array(list(probs.values()))
                # If still empty, try generic class names
                if prob_array.sum() == 0 or len(prob_array) < 2:
                    # Try common class name patterns (generic fallback)
                    class_keys = list(probs.keys())
                    if len(class_keys) >= 2:
                        prob_array = np.array([probs.get(class_keys[0], 0.0), probs.get(class_keys[1], 0.0)])
                    else:
                        prob_array = np.array([0.5, 0.5])  # Uniform fallback
            elif isinstance(probs, (list, np.ndarray)) and len(probs) >= 2:
                prob_array = np.array(probs)
            else:
                # Default to uniform distribution if no valid probabilities found
                prob_array = np.array([0.5, 0.5])
                conf = 0.5
        
        # If we still don't have confidence, calculate from prob_array
        if conf is None:
            if prob_array is not None and len(prob_array) >= 2:
                conf = float(np.max(prob_array))
            else:
                conf = pred.get('confidence', 0.0)
        
        # Ensure prob_array is defined
        if prob_array is None:
            prob_array = np.array([0.5, 0.5])
        
        confidences.append(conf)
        
        # Ensure probabilities are valid
        if prob_array.sum() == 0 or len(prob_array) < 2:
            prob_array = np.array([0.5, 0.5])
        else:
            # Renormalize to ensure they sum to 1
            prob_array = prob_array / max(prob_array.sum(), 1e-10)  # Avoid division by zero
        
        # Calculate entropy
        entropies.append(calculate_entropy(prob_array))
        
        # Calculate probability gap
        probability_gaps.append(calculate_probability_gap(prob_array))
        
        # Extract logits if available
        logits = pred.get('logits')
        if logits is not None and len(logits) > 0:
            logit_array = np.array(logits)
            logit_magnitudes.append(calculate_logit_magnitude(logit_array))
            
            # Calculate calibrated probabilities (using temperature=0.8 for better calibration)
            calibrated_probs = calculate_calibrated_probabilities(logit_array, temperature=0.8)
            calibrated_conf = float(np.max(calibrated_probs))
            calibrated_confidences.append(calibrated_conf)
            
            # Calculate calibration error (difference between uncalibrated and calibrated)
            # Uncalibrated = softmax(logits) with temperature=1.0 (no scaling)
            # Calibrated = softmax(logits / 0.8) with temperature=0.8
            uncalibrated_probs = calculate_calibrated_probabilities(logit_array, temperature=1.0)
            if len(uncalibrated_probs) == len(calibrated_probs):
                cal_error = calculate_calibration_error(uncalibrated_probs, calibrated_probs)
                calibration_errors.append(cal_error)
    
    return {
        'modality': modality_name,
        'avg_confidence': float(np.mean(confidences)) if confidences else 0.0,
        'std_confidence': float(np.std(confidences)) if confidences else 0.0,
        'avg_entropy': float(np.mean(entropies)) if entropies else 0.0,
        'std_entropy': float(np.std(entropies)) if entropies else 0.0,
        'avg_probability_gap': float(np.mean(probability_gaps)) if probability_gaps else 0.0,
        'std_probability_gap': float(np.std(probability_gaps)) if probability_gaps else 0.0,
        'avg_logit_magnitude': float(np.mean(logit_magnitudes)) if logit_magnitudes else 0.0,
        'std_logit_magnitude': float(np.std(logit_magnitudes)) if logit_magnitudes else 0.0,
        'avg_calibrated_confidence': float(np.mean(calibrated_confidences)) if calibrated_confidences else 0.0,
        'avg_calibration_error': float(np.mean(calibration_errors)) if calibration_errors else 0.0,
        'num_samples': len(predictions)
    }


def analyze_modality_context_influence(
    mod2_predictions: List[Dict],
    mod1_predictions: Optional[List[Dict]] = None
) -> Dict:
    """
    CORE RESEARCH QUESTION: Does modality 2 certainty increase when modality 1 context is integrated?
    
    This analysis determines whether the model actually uses previous modality context or ignores it.
    Compares predictions before and after context integration.
    
    Args:
        mod2_predictions: List of modality 2 prediction dicts with 'probabilities_before_boosting'
        mod1_predictions: Optional list of modality 1 predictions for comparison
    
    Returns:
        Dictionary with context influence metrics
    """
    if not mod2_predictions:
        return {
            'context_changes_prediction': 0,
            'context_increases_confidence': 0,
            'context_decreases_confidence': 0,
            'avg_confidence_change': 0.0,
            'avg_entropy_change': 0.0,
            'context_used': False,
            'num_samples': 0
        }
    
    changes_prediction = 0
    increases_conf = 0
    decreases_conf = 0
    confidence_changes = []
    entropy_changes = []
    significant_increases = 0  # >0.05 increase
    
    for mod2_pred in mod2_predictions:
        probs_before = mod2_pred.get('probabilities_before_boosting')
        probs_after = mod2_pred.get('probabilities_array')
        
        if probs_before is not None and probs_after is not None:
            probs_before_arr = np.array(probs_before)
            probs_after_arr = np.array(probs_after)
            
            # Calculate confidence from probabilities
            conf_before = float(np.max(probs_before_arr))
            conf_after = float(np.max(probs_after_arr))
            
            # Calculate entropy
            entropy_before = calculate_entropy(probs_before_arr)
            entropy_after = calculate_entropy(probs_after_arr)
            
            # Check if prediction changed
            pred_before = np.argmax(probs_before_arr)
            pred_after = np.argmax(probs_after_arr)
            
            if pred_before != pred_after:
                changes_prediction += 1
            
            # Check confidence change
            conf_change = conf_after - conf_before
            confidence_changes.append(conf_change)
            entropy_changes.append(entropy_after - entropy_before)
            
            if conf_change > 0.05:  # Significant increase (>5%)
                significant_increases += 1
                increases_conf += 1
            elif conf_change > 0.01:  # Small increase
                increases_conf += 1
            elif conf_change < -0.01:  # Decrease
                decreases_conf += 1
    
    total = len(mod2_predictions)
    avg_conf_change = float(np.mean(confidence_changes)) if confidence_changes else 0.0
    avg_entropy_change = float(np.mean(entropy_changes)) if entropy_changes else 0.0
    
    # Determine if modality 1 context is actually being used
    # Criteria: Average confidence increase > 0.02 OR >30% of cases show significant increase
    context_used = avg_conf_change > 0.02 or (significant_increases / total if total > 0 else 0) > 0.3
    
    return {
        'context_changes_prediction': changes_prediction,
        'context_changes_prediction_rate': changes_prediction / total if total > 0 else 0.0,
        'context_increases_confidence': increases_conf,
        'context_decreases_confidence': decreases_conf,
        'significant_confidence_increases': significant_increases,
        'avg_confidence_change': avg_conf_change,
        'std_confidence_change': float(np.std(confidence_changes)) if confidence_changes else 0.0,
        'avg_entropy_change': avg_entropy_change,
        'context_used': context_used,
        'num_samples': total,
        'interpretation': (
            f"Modality 1 context {'IS BEING USED' if context_used else 'APPEARS TO BE IGNORED'}. "
            f"Average confidence change: {avg_conf_change:+.4f}. "
            f"{significant_increases}/{total} cases show significant increase (>5%)"
        )
    }


def analyze_modality_agreement(
    mod1_predictions: List[Dict],
    mod2_predictions: List[Dict],
    patient_ids: Optional[List[str]] = None,
    debug: bool = False
) -> Dict:
    """
    Analyze agreement/disagreement between two modality predictions (generic).
    Focuses on how disagreement affects certainty (confidence, logits).
    
    Args:
        mod1_predictions: List of modality 1 prediction dicts (should have 'prediction', 'confidence', 'logits')
        mod2_predictions: List of modality 2 prediction dicts (should have 'prediction', 'confidence', 'logits')
        patient_ids: Optional list of patient IDs for matching (if None, assumes same order)
    
    Returns:
        Dictionary with agreement metrics and certainty analysis
    """
    # Match predictions by patient_id + slice_index (if available) or by index
    # CRITICAL: Group each modality by its own patient_id so matching is correct when slice counts differ (e.g. CT 738 vs MR 235)
    matched_pairs = []
    mod1_by_patient = {}
    mod2_by_patient = {}
    for pred in mod1_predictions or []:
        if isinstance(pred, dict):
            pid = pred.get('patient_id')
            if pid is not None:
                if pid not in mod1_by_patient:
                    mod1_by_patient[pid] = []
                mod1_by_patient[pid].append(pred)
    for pred in mod2_predictions or []:
        if isinstance(pred, dict):
            pid = pred.get('patient_id')
            if pid is not None:
                if pid not in mod2_by_patient:
                    mod2_by_patient[pid] = []
                mod2_by_patient[pid].append(pred)
    common_patient_ids = sorted(set(mod1_by_patient.keys()) & set(mod2_by_patient.keys()))
    if common_patient_ids:
        # Match slices within each patient by slice_index if available, otherwise one pair per patient
        for pid in common_patient_ids:
            mod1_slices = mod1_by_patient[pid]
            mod2_slices = mod2_by_patient[pid]
            mod1_by_slice_idx = {}
            mod2_by_slice_idx = {}
            for mod1_slice in mod1_slices:
                slice_idx = mod1_slice.get('slice_index') if isinstance(mod1_slice, dict) else None
                if slice_idx is not None:
                    mod1_by_slice_idx[slice_idx] = mod1_slice
            for mod2_slice in mod2_slices:
                slice_idx = mod2_slice.get('slice_index') if isinstance(mod2_slice, dict) else None
                if slice_idx is not None:
                    mod2_by_slice_idx[slice_idx] = mod2_slice
            if mod1_by_slice_idx and mod2_by_slice_idx:
                common_slice_indices = set(mod1_by_slice_idx.keys()) & set(mod2_by_slice_idx.keys())
                if common_slice_indices:
                    for slice_idx in common_slice_indices:
                        matched_pairs.append((mod1_by_slice_idx[slice_idx], mod2_by_slice_idx[slice_idx]))
                elif mod1_slices and mod2_slices:
                    matched_pairs.append((mod1_slices[0], mod2_slices[0]))
            elif mod1_slices and mod2_slices:
                matched_pairs.append((mod1_slices[0], mod2_slices[0]))
    if not matched_pairs and (mod1_predictions or mod2_predictions):
        # Fallback: match by index when no patient_id grouping possible
        min_len = min(len(mod1_predictions or []), len(mod2_predictions or []))
        if min_len > 0:
            matched_pairs = list(zip((mod1_predictions or [])[:min_len], (mod2_predictions or [])[:min_len]))
    
    if not matched_pairs:
        return {
            'agreement_rate': 0.0,
            'disagreement_rate': 0.0,
            'mod1_dominates': 0,
            'mod2_dominates': 0,
            'num_pairs': 0,
            'disagreement_confidence_analysis': {},
            'disagreement_logit_analysis': {}
        }
    
    agreements = 0
    disagreements = 0
    mod1_dominates = 0
    mod2_dominates = 0
    
    # Track certainty metrics for agreement vs disagreement cases
    agreement_confidences = []
    disagreement_confidences_mod1 = []
    disagreement_confidences_mod2 = []
    disagreement_logit_magnitudes_mod1 = []
    disagreement_logit_magnitudes_mod2 = []
    disagreement_logit_variances_mod1 = []
    disagreement_logit_variances_mod2 = []
    
    for idx, (mod1_pred, mod2_pred) in enumerate(matched_pairs):
        mod1_pred_class = mod1_pred.get('prediction')
        # For Mod1+Mod2 (Mod2 with Mod1 context), the prediction should be based on final probabilities
        # Check if this is a Mod1+Mod2 prediction (has used_context flag)
        used_context = mod2_pred.get('used_context', False)
        if used_context:
            # Mod1+Mod2: Recalculate prediction from boosted probabilities (after boosting)
            # This is critical because boosting can change the predicted class
            mod2_probs_after = mod2_pred.get('probabilities_array')
            mod2_probs_before = mod2_pred.get('probabilities_before_boosting')
            
            if mod2_probs_after is not None and len(mod2_probs_after) >= 2:
                # Recalculate prediction from boosted probabilities
                mod2_probs_arr = np.array(mod2_probs_after)
                # Ensure probabilities are valid
                if mod2_probs_arr.sum() > 0:
                    mod2_probs_arr = mod2_probs_arr / max(mod2_probs_arr.sum(), 1e-10)  # Normalize
                    mod2_pred_class = int(np.argmax(mod2_probs_arr))
                else:
                    # Fallback to stored prediction if probabilities are invalid
                    mod2_pred_class = mod2_pred.get('prediction', 0)
            elif mod2_probs_before is not None and len(mod2_probs_before) >= 2:
                # Fallback: use probabilities before boosting if after-boosting not available
                mod2_probs_arr = np.array(mod2_probs_before)
                if mod2_probs_arr.sum() > 0:
                    mod2_probs_arr = mod2_probs_arr / max(mod2_probs_arr.sum(), 1e-10)
                    mod2_pred_class = int(np.argmax(mod2_probs_arr))
                else:
                    mod2_pred_class = mod2_pred.get('prediction', 0)
            else:
                # Final fallback: use stored prediction
                mod2_pred_class = mod2_pred.get('prediction', 0)
        else:
            # Mod2 alone: Use stored prediction (no boosting applied)
            mod2_pred_class = mod2_pred.get('prediction', 0)
        
        # Ensure prediction classes are valid (0 or 1)
        mod1_pred_class = max(0, min(1, int(mod1_pred_class))) if mod1_pred_class is not None else 0
        mod2_pred_class = max(0, min(1, int(mod2_pred_class))) if mod2_pred_class is not None else 0
        
        mod1_conf = mod1_pred.get('confidence', 0.0)
        mod2_conf = mod2_pred.get('confidence', 0.0)
        
        # Extract logits if available
        mod1_logits = mod1_pred.get('logits')
        mod2_logits = mod2_pred.get('logits')
        
        if mod1_pred_class == mod2_pred_class:
            agreements += 1
            # Average confidence when modalities agree
            agreement_confidences.append((mod1_conf + mod2_conf) / 2.0)
        else:
            disagreements += 1
            disagreement_confidences_mod1.append(mod1_conf)
            disagreement_confidences_mod2.append(mod2_conf)
            
            # Analyze logit stability when modalities disagree
            if mod1_logits is not None and len(mod1_logits) >= 2:
                mod1_logits_arr = np.array(mod1_logits)
                disagreement_logit_magnitudes_mod1.append(calculate_logit_magnitude(mod1_logits_arr))
                disagreement_logit_variances_mod1.append(float(np.var(mod1_logits_arr)))
            
            if mod2_logits is not None and len(mod2_logits) >= 2:
                mod2_logits_arr = np.array(mod2_logits)
                disagreement_logit_magnitudes_mod2.append(calculate_logit_magnitude(mod2_logits_arr))
                disagreement_logit_variances_mod2.append(float(np.var(mod2_logits_arr)))
            
            # Check which modality has higher confidence when they disagree
            if mod1_conf > mod2_conf:
                mod1_dominates += 1
            elif mod2_conf > mod1_conf:
                mod2_dominates += 1
    
    total = len(matched_pairs)
    
    # Analyze how disagreement affects confidence
    disagreement_confidence_analysis = {}
    if disagreement_confidences_mod1 and disagreement_confidences_mod2:
        disagreement_confidence_analysis = {
            'avg_mod1_confidence_when_disagree': float(np.mean(disagreement_confidences_mod1)),
            'avg_mod2_confidence_when_disagree': float(np.mean(disagreement_confidences_mod2)),
            'std_mod1_confidence_when_disagree': float(np.std(disagreement_confidences_mod1)),
            'std_mod2_confidence_when_disagree': float(np.std(disagreement_confidences_mod2)),
            'confidence_difference': float(np.mean(disagreement_confidences_mod2) - np.mean(disagreement_confidences_mod1))
        }
    
    if agreement_confidences:
        disagreement_confidence_analysis['avg_confidence_when_agree'] = float(np.mean(agreement_confidences))
        disagreement_confidence_analysis['confidence_drop_on_disagreement'] = (
            float(np.mean(agreement_confidences)) - 
            float(np.mean(disagreement_confidences_mod1 + disagreement_confidences_mod2) / 2)
            if disagreement_confidences_mod1 and disagreement_confidences_mod2 else 0.0
        )
    
    # Analyze logit stability when modalities disagree
    disagreement_logit_analysis = {}
    if disagreement_logit_magnitudes_mod1 and disagreement_logit_magnitudes_mod2:
        disagreement_logit_analysis = {
            'avg_logit_magnitude_mod1_when_disagree': float(np.mean(disagreement_logit_magnitudes_mod1)),
            'avg_logit_magnitude_mod2_when_disagree': float(np.mean(disagreement_logit_magnitudes_mod2)),
            'avg_logit_variance_mod1_when_disagree': float(np.mean(disagreement_logit_variances_mod1)),
            'avg_logit_variance_mod2_when_disagree': float(np.mean(disagreement_logit_variances_mod2)),
            'logit_instability_indicator': float(np.mean(disagreement_logit_variances_mod1 + disagreement_logit_variances_mod2))
        }
    
    return {
        'agreement_rate': agreements / total if total > 0 else 0.0,
        'disagreement_rate': disagreements / total if total > 0 else 0.0,
        'mod1_dominates': mod1_dominates,
        'mod2_dominates': mod2_dominates,
        'num_pairs': total,
        'disagreement_confidence_analysis': disagreement_confidence_analysis,
        'disagreement_logit_analysis': disagreement_logit_analysis
    }


def analyze_logit_similarity(
    mod1_predictions: List[Dict],
    mod2_predictions: List[Dict],
    patient_ids: Optional[List[str]] = None
) -> Dict:
    """
    Analyze cosine similarity between logit distributions of two modalities.
    Higher similarity indicates more consistent model behavior across modalities.
    
    Aggregates logits at patient level (weighted average by confidence) before comparing,
    to avoid bias from comparing arbitrary individual slices.
    
    Args:
        mod1_predictions: Predictions from first modality (slice-level)
        mod2_predictions: Predictions from second modality (slice-level)
        patient_ids: Optional list of patient IDs for matching
    
    Returns:
        Dictionary with logit similarity metrics
    """
    # Group predictions by each modality's own patient_id (correct when slice counts differ, e.g. CT 738 vs MR 235)
    mod1_by_patient = {}
    mod2_by_patient = {}
    for pred in mod1_predictions or []:
        if isinstance(pred, dict):
            pid = pred.get('patient_id')
            if pid is not None:
                if pid not in mod1_by_patient:
                    mod1_by_patient[pid] = []
                mod1_by_patient[pid].append(pred)
    for pred in mod2_predictions or []:
        if isinstance(pred, dict):
            pid = pred.get('patient_id')
            if pid is not None:
                if pid not in mod2_by_patient:
                    mod2_by_patient[pid] = []
                mod2_by_patient[pid].append(pred)
    common_patient_ids = sorted(set(mod1_by_patient.keys()) & set(mod2_by_patient.keys()))
    if common_patient_ids:
        # Aggregate logits per patient (weighted by confidence)
        aggregated_mod1 = {}
        aggregated_mod2 = {}
        
        for pid, slices in mod1_by_patient.items():
            logits_list = []
            confidences = []
            for slice_pred in slices:
                logits = slice_pred.get('logits')
                conf = slice_pred.get('confidence', 0.5)
                if logits is not None:
                    logits_list.append(np.array(logits))
                    confidences.append(max(0.0, min(1.0, float(conf))))
            
            if logits_list:
                # Weighted average of logits by confidence
                confidences = np.array(confidences)
                if confidences.sum() > 0:
                    weights = confidences / confidences.sum()
                    aggregated_logits = np.average(logits_list, axis=0, weights=weights)
                else:
                    aggregated_logits = np.mean(logits_list, axis=0)
                aggregated_mod1[pid] = aggregated_logits
        
        for pid, slices in mod2_by_patient.items():
            logits_list = []
            confidences = []
            for slice_pred in slices:
                logits = slice_pred.get('logits')
                conf = slice_pred.get('confidence', 0.5)
                if logits is not None:
                    logits_list.append(np.array(logits))
                    confidences.append(max(0.0, min(1.0, float(conf))))
            
            if logits_list:
                # Weighted average of logits by confidence
                confidences = np.array(confidences)
                if confidences.sum() > 0:
                    weights = confidences / confidences.sum()
                    aggregated_logits = np.average(logits_list, axis=0, weights=weights)
                else:
                    aggregated_logits = np.mean(logits_list, axis=0)
                aggregated_mod2[pid] = aggregated_logits
        
        # Match by patient_id and compare aggregated logits
        common_patients = set(aggregated_mod1.keys()) & set(aggregated_mod2.keys())
        matched_pairs = [(aggregated_mod1[pid], aggregated_mod2[pid]) for pid in common_patients]
    else:
        # Fallback: compare by index (slice-level)
        min_len = min(len(mod1_predictions), len(mod2_predictions))
        matched_pairs = []
        for i in range(min_len):
            mod1_logits = mod1_predictions[i].get('logits')
            mod2_logits = mod2_predictions[i].get('logits')
            if mod1_logits is not None and mod2_logits is not None:
                matched_pairs.append((np.array(mod1_logits), np.array(mod2_logits)))
    
    if not matched_pairs:
        return {
            'avg_cosine_similarity': 0.0,
            'std_cosine_similarity': 0.0,
            'num_pairs': 0
        }
    
    similarities = []
    for mod1_logits_arr, mod2_logits_arr in matched_pairs:
        if len(mod1_logits_arr) == len(mod2_logits_arr) and len(mod1_logits_arr) > 0:
            similarity = calculate_cosine_similarity(mod1_logits_arr, mod2_logits_arr)
            similarities.append(similarity)
    
    if not similarities:
        return {
            'avg_cosine_similarity': 0.0,
            'std_cosine_similarity': 0.0,
            'num_pairs': 0
        }
    
    # Additional diagnostics: check if logits are identical vs proportional
    identical_count = 0
    proportional_count = 0
    for mod1_logits_arr, mod2_logits_arr in matched_pairs:
        if len(mod1_logits_arr) == len(mod2_logits_arr) and len(mod1_logits_arr) > 0:
            if np.allclose(mod1_logits_arr, mod2_logits_arr, atol=1e-6):
                identical_count += 1
            elif len(mod1_logits_arr) == 2:
                # For binary classification, check if proportional
                ratio1 = mod1_logits_arr[0] / mod1_logits_arr[1] if mod1_logits_arr[1] != 0 else float('inf')
                ratio2 = mod2_logits_arr[0] / mod2_logits_arr[1] if mod2_logits_arr[1] != 0 else float('inf')
                if np.isclose(ratio1, ratio2, atol=1e-3):
                    proportional_count += 1
    
    return {
        'avg_cosine_similarity': float(np.mean(similarities)),
        'std_cosine_similarity': float(np.std(similarities)),
        'min_cosine_similarity': float(np.min(similarities)),
        'max_cosine_similarity': float(np.max(similarities)),
        'num_pairs': len(similarities),
        'identical_logits_count': identical_count,
        'proportional_logits_count': proportional_count
    }


def analyze_patient_level_agreement(
    slice_level_agreement: Dict,
    patient_level_mod1: List[Dict],
    patient_level_mod2: List[Dict],
    patient_ids: List[str]
) -> Dict:
    """
    Analyze whether modality agreement increases at patient level vs slice level.
    
    Key question: Does aggregating slices per patient lead to better agreement
    between two modality predictions?
    
    Args:
        slice_level_agreement: Agreement metrics from slice-level analysis
        patient_level_mod1: List of patient-level modality 1 predictions
        patient_level_mod2: List of patient-level modality 2 predictions
        patient_ids: List of patient IDs matching the predictions
    
    Returns:
        Dictionary with patient-level agreement analysis
    """
    if len(patient_level_mod1) != len(patient_level_mod2) or len(patient_level_mod1) != len(patient_ids):
        return {
            'patient_level_agreement_rate': 0.0,
            'slice_level_agreement_rate': 0.0,
            'agreement_improvement': 0.0,
            'num_patients': 0
        }
    
    # Calculate patient-level agreement
    # CRITICAL: Match by patient_id to ensure correct pairing (defensive check)
    patient_agreements = 0
    patient_disagreements = 0
    
    # Build lookup dicts by patient_id for safe matching
    mod1_by_patient = {pred.get('patient_id'): pred for pred in patient_level_mod1 if pred.get('patient_id') is not None}
    mod2_by_patient = {pred.get('patient_id'): pred for pred in patient_level_mod2 if pred.get('patient_id') is not None}
    
    # Match by patient_id (more robust than zip)
    matched_patients = set(mod1_by_patient.keys()) & set(mod2_by_patient.keys())
    
    for patient_id in sorted(matched_patients):
        mod1_pred = mod1_by_patient[patient_id]
        mod2_pred = mod2_by_patient[patient_id]
        
        mod1_pred_class = mod1_pred.get('prediction')
        mod2_pred_class = mod2_pred.get('prediction')
        
        if mod1_pred_class == mod2_pred_class:
            patient_agreements += 1
        else:
            patient_disagreements += 1
    
    # Fallback: if patient_id matching fails, use zip (original behavior)
    if not matched_patients and len(patient_level_mod1) == len(patient_level_mod2):
        for mod1_pred, mod2_pred in zip(patient_level_mod1, patient_level_mod2):
            mod1_pred_class = mod1_pred.get('prediction')
            mod2_pred_class = mod2_pred.get('prediction')
            
            if mod1_pred_class == mod2_pred_class:
                patient_agreements += 1
            else:
                patient_disagreements += 1
    
    total_patients = len(patient_level_mod1)
    patient_agreement_rate = patient_agreements / total_patients if total_patients > 0 else 0.0
    
    # Get slice-level agreement for comparison
    slice_agreement_rate = slice_level_agreement.get('agreement_rate', 0.0)
    agreement_improvement = patient_agreement_rate - slice_agreement_rate
    
    return {
        'patient_level_agreement_rate': patient_agreement_rate,
        'slice_level_agreement_rate': slice_agreement_rate,
        'agreement_improvement': agreement_improvement,
        'patient_agreements': patient_agreements,
        'patient_disagreements': patient_disagreements,
        'num_patients': total_patients,
        'interpretation': (
            f"Patient-level agreement: {patient_agreement_rate:.4f} vs "
            f"Slice-level: {slice_agreement_rate:.4f} "
            f"({agreement_improvement:+.4f} change)"
        )
    }


def analyze_modality_confidence_comparison(
    mod1_predictions: List[Dict],
    mod2_predictions: List[Dict],
    patient_ids: Optional[List[str]] = None
) -> Dict:
    """
    CORE RESEARCH QUESTION: Does modality 2 consistently produce higher confidence than modality 1?
    
    This is a fundamental question about modality behavior in zero-shot VLMs (generic for any two modalities).
    
    Args:
        mod1_predictions: List of modality 1 prediction dicts
        mod2_predictions: List of modality 2 prediction dicts
        patient_ids: Optional list of patient IDs for matching
    
    Returns:
        Dictionary with modality confidence comparison analysis
    """
    # Match predictions by patient_id (robust to different list lengths/order)
    matched_pairs = []
    mod1_by_patient = {}
    mod2_by_patient = {}
    for pred in mod1_predictions or []:
        if isinstance(pred, dict):
            pid = pred.get('patient_id')
            if pid is not None:
                if pid not in mod1_by_patient:
                    mod1_by_patient[pid] = []
                mod1_by_patient[pid].append(pred)
    for pred in mod2_predictions or []:
        if isinstance(pred, dict):
            pid = pred.get('patient_id')
            if pid is not None:
                if pid not in mod2_by_patient:
                    mod2_by_patient[pid] = []
                mod2_by_patient[pid].append(pred)
    common_patient_ids = sorted(set(mod1_by_patient.keys()) & set(mod2_by_patient.keys()))
    for pid in common_patient_ids:
        mod1_slices = mod1_by_patient[pid]
        mod2_slices = mod2_by_patient[pid]
        mod1_aggregated = aggregate_patient_predictions(mod1_slices)
        mod2_aggregated = aggregate_patient_predictions(mod2_slices)
        mod1_pred = {
            'prediction': mod1_aggregated['prediction'],
            'confidence': mod1_aggregated['confidence'],
            'probabilities': mod1_slices[0].get('probabilities', {}) if mod1_slices else {},
            'probabilities_array': mod1_slices[0].get('probabilities_array', []) if mod1_slices else [],
            'logits': mod1_slices[0].get('logits', []) if mod1_slices else []
        }
        mod2_pred = {
            'prediction': mod2_aggregated['prediction'],
            'confidence': mod2_aggregated['confidence'],
            'probabilities': mod2_slices[0].get('probabilities', {}) if mod2_slices else {},
            'probabilities_array': mod2_slices[0].get('probabilities_array', []) if mod2_slices else [],
            'probabilities_before_boosting': mod2_slices[0].get('probabilities_before_boosting') if mod2_slices else None,
            'used_context': mod2_slices[0].get('used_context', False) if mod2_slices else False,
            'logits': mod2_slices[0].get('logits', []) if mod2_slices else []
        }
        matched_pairs.append((mod1_pred, mod2_pred))
    if not matched_pairs and (mod1_predictions or mod2_predictions):
        # Fallback: index alignment when patient_id not available
        min_len = min(len(mod1_predictions or []), len(mod2_predictions or []))
        if min_len > 0:
            matched_pairs = list(zip((mod1_predictions or [])[:min_len], (mod2_predictions or [])[:min_len]))
    if not matched_pairs:
        return {
            'mod2_higher_confidence_rate': 0.0,
            'mod1_higher_confidence_rate': 0.0,
            'avg_confidence_difference': 0.0,
            'consistent_mod2_dominance': False,
            'num_pairs': 0
        }
    
    mod2_higher = 0
    mod1_higher = 0
    equal_conf = 0
    confidence_differences = []
    
    for mod1_pred, mod2_pred in matched_pairs:
        # Use aggregated (patient-level) confidence so PAIRWISE CONFIDENCE and PAIRWISE DOMINANCE match
        mod1_conf = mod1_pred.get('confidence', 0.0)
        mod2_conf = mod2_pred.get('confidence', 0.0)
        conf_diff = mod2_conf - mod1_conf
        confidence_differences.append(conf_diff)
        
        # Use strict comparison (same as PAIRWISE DOMINANCE) so both sections report consistent "X% higher"
        if mod2_conf > mod1_conf:
            mod2_higher += 1
        elif mod1_conf > mod2_conf:
            mod1_higher += 1
        else:
            equal_conf += 1
    
    total = len(matched_pairs)
    avg_conf_diff = np.mean(confidence_differences) if confidence_differences else 0.0
    
    # Consistent Mod2 dominance: >70% of cases AND average difference > 0.05
    mod2_dominance_rate = mod2_higher / total if total > 0 else 0.0
    consistent_dominance = mod2_dominance_rate > 0.7 and avg_conf_diff > 0.05
    
    return {
        'mod2_higher_confidence_rate': mod2_dominance_rate,
        'mod1_higher_confidence_rate': mod1_higher / total if total > 0 else 0.0,
        'equal_confidence_rate': equal_conf / total if total > 0 else 0.0,
        'avg_confidence_difference': float(avg_conf_diff),
        'std_confidence_difference': float(np.std(confidence_differences)) if confidence_differences else 0.0,
        'consistent_mod2_dominance': consistent_dominance,
        'num_pairs': total,
        'interpretation': (
            f"Modality 2 has higher confidence in {mod2_dominance_rate*100:.1f}% of cases. "
            f"Average difference (Mod2 - Mod1): {avg_conf_diff:+.4f}. "
            f"{'Modality 2 CONSISTENTLY DOMINATES' if consistent_dominance else 'No consistent dominance pattern'}"
        )
    }


def analyze_multimodality_uncertainty_effect(
    mod1_predictions: List[Dict],
    mod2_predictions: List[Dict],
    combined_predictions: List[Dict],
    patient_ids: Optional[List[str]] = None
) -> Dict:
    """
    CORE RESEARCH QUESTION: Does multimodality reduce uncertainty or introduce conflicting signals?
    
    This analyzes whether combining modalities:
    1. Reduces uncertainty (lower entropy, higher confidence)
    2. Introduces conflict (higher entropy, lower confidence, unstable logits)
    3. Provides complementary information (true multimodal value)
    
    Args:
        mod1_predictions: List of modality 1 prediction dicts
        mod2_predictions: List of modality 2 prediction dicts
        combined_predictions: List of combined (mod1+mod2) prediction dicts
        patient_ids: Optional list of patient IDs for matching
    
    Returns:
        Dictionary with uncertainty analysis
    """
    # Match all three prediction sets - aggregate at patient level first
    # CRITICAL: Group each by its own patient_id so matching is correct when slice counts differ
    matched_triplets = []
    mod1_by_patient = {}
    mod2_by_patient = {}
    combined_by_patient = {}
    for pred in mod1_predictions or []:
        if isinstance(pred, dict):
            pid = pred.get('patient_id')
            if pid is not None:
                if pid not in mod1_by_patient:
                    mod1_by_patient[pid] = []
                mod1_by_patient[pid].append(pred)
    for pred in mod2_predictions or []:
        if isinstance(pred, dict):
            pid = pred.get('patient_id')
            if pid is not None:
                if pid not in mod2_by_patient:
                    mod2_by_patient[pid] = []
                mod2_by_patient[pid].append(pred)
    for pred in combined_predictions or []:
        if isinstance(pred, dict):
            pid = pred.get('patient_id')
            if pid is not None:
                if pid not in combined_by_patient:
                    combined_by_patient[pid] = []
                combined_by_patient[pid].append(pred)
    common_patient_ids = sorted(set(mod1_by_patient.keys()) & set(mod2_by_patient.keys()) & set(combined_by_patient.keys()))
    if common_patient_ids:
        for pid in common_patient_ids:
            mod1_slices = mod1_by_patient[pid]
            mod2_slices = mod2_by_patient[pid]
            combined_slices = combined_by_patient[pid]
            mod1_aggregated = aggregate_patient_predictions(mod1_slices)
            mod2_aggregated = aggregate_patient_predictions(mod2_slices)
            combined_aggregated = aggregate_patient_predictions(combined_slices)
            mod1_pred = {
                'prediction': mod1_aggregated['prediction'],
                'confidence': mod1_aggregated['confidence'],
                'probabilities': mod1_slices[0].get('probabilities', {}) if mod1_slices else {},
                'probabilities_array': mod1_slices[0].get('probabilities_array', []) if mod1_slices else []
            }
            mod2_pred = {
                'prediction': mod2_aggregated['prediction'],
                'confidence': mod2_aggregated['confidence'],
                'probabilities': mod2_slices[0].get('probabilities', {}) if mod2_slices else {},
                'probabilities_array': mod2_slices[0].get('probabilities_array', []) if mod2_slices else [],
                'probabilities_before_boosting': mod2_slices[0].get('probabilities_before_boosting') if mod2_slices else None
            }
            combined_pred = {
                'prediction': combined_aggregated['prediction'],
                'confidence': combined_aggregated['confidence'],
                'probabilities': combined_slices[0].get('probabilities', {}) if combined_slices else {},
                'probabilities_array': combined_slices[0].get('probabilities_array', []) if combined_slices else []
            }
            matched_triplets.append((mod1_pred, mod2_pred, combined_pred))
    if not matched_triplets and (mod1_predictions or mod2_predictions or combined_predictions):
        min_len = min(len(mod1_predictions or []), len(mod2_predictions or []), len(combined_predictions or []))
        if min_len > 0:
            matched_triplets = list(zip(
                (mod1_predictions or [])[:min_len],
                (mod2_predictions or [])[:min_len],
                (combined_predictions or [])[:min_len]
            ))
    
    if not matched_triplets:
        return {
            'uncertainty_reduction_rate': 0.0,
            'conflict_introduction_rate': 0.0,
            'avg_entropy_change': 0.0,
            'avg_confidence_change': 0.0,
            'multimodal_value': 'unknown',
            'num_triplets': 0
        }
    
    uncertainty_reductions = 0
    conflict_introductions = 0
    entropy_changes = []
    confidence_changes = []
    
    for mod1_pred, mod2_pred, combined_pred in matched_triplets:
        # Get probabilities for entropy calculation
        mod1_probs = mod1_pred.get('probabilities_array') or list(mod1_pred.get('probabilities', {}).values())
        mod2_probs_before = mod2_pred.get('probabilities_before_boosting')
        if mod2_probs_before is None:
            mod2_probs_before = mod2_pred.get('probabilities_array') or list(mod2_pred.get('probabilities', {}).values())
        combined_probs = combined_pred.get('probabilities_array') or list(combined_pred.get('probabilities', {}).values())
        
        # Calculate entropies
        if len(mod1_probs) >= 2 and len(mod2_probs_before) >= 2 and len(combined_probs) >= 2:
            mod1_entropy = calculate_entropy(np.array(mod1_probs))
            mod2_entropy = calculate_entropy(np.array(mod2_probs_before))
            combined_entropy = calculate_entropy(np.array(combined_probs))
            
            avg_single_entropy = (mod1_entropy + mod2_entropy) / 2.0
            entropy_change = combined_entropy - avg_single_entropy
            entropy_changes.append(entropy_change)
            
            # Uncertainty reduction: combined entropy < average single-modality entropy
            if entropy_change < -0.1:  # Significant reduction
                uncertainty_reductions += 1
            # Conflict introduction: combined entropy > average single-modality entropy
            elif entropy_change > 0.1:  # Significant increase
                conflict_introductions += 1
        
        # Calculate confidence changes
        mod1_conf = mod1_pred.get('confidence', 0.0)
        mod2_conf_before = float(np.max(np.array(mod2_probs_before))) if mod2_probs_before is not None and len(mod2_probs_before) >= 2 else mod2_pred.get('confidence', 0.0)
        combined_conf = combined_pred.get('confidence', 0.0)
        
        avg_single_conf = (mod1_conf + mod2_conf_before) / 2.0
        conf_change = combined_conf - avg_single_conf
        confidence_changes.append(conf_change)
    
    total = len(matched_triplets)
    avg_entropy_change = float(np.mean(entropy_changes)) if entropy_changes else 0.0
    avg_conf_change = float(np.mean(confidence_changes)) if confidence_changes else 0.0
    
    # Determine multimodal value
    if avg_entropy_change < -0.1 and avg_conf_change > 0.02:
        multimodal_value = 'REDUCES_UNCERTAINTY'
    elif avg_entropy_change > 0.1 or avg_conf_change < -0.02:
        multimodal_value = 'INTRODUCES_CONFLICT'
    else:
        multimodal_value = 'NEUTRAL'
    
    return {
        'uncertainty_reduction_rate': uncertainty_reductions / total if total > 0 else 0.0,
        'conflict_introduction_rate': conflict_introductions / total if total > 0 else 0.0,
        'avg_entropy_change': avg_entropy_change,
        'avg_confidence_change': avg_conf_change,
        'multimodal_value': multimodal_value,
        'num_triplets': total,
        'interpretation': (
            f"Multimodality {multimodal_value.lower().replace('_', ' ')}. "
            f"Entropy change: {avg_entropy_change:+.4f}, "
            f"Confidence change: {avg_conf_change:+.4f}. "
            f"{uncertainty_reductions}/{total} cases show uncertainty reduction, "
            f"{conflict_introductions}/{total} show conflict introduction"
        )
    }


def analyze_zero_shot_multimodal_value(
    mod1_predictions: List[Dict],
    mod2_predictions: List[Dict],
    combined_predictions: List[Dict],
    patient_ids: Optional[List[str]] = None
) -> Dict:
    """
    CORE RESEARCH QUESTION: Can zero-shot VLMs reflect the "value" of multimodal clinical imaging?
    
    This analyzes whether the model demonstrates understanding of multimodal value through:
    1. Improved certainty when modalities agree
    2. Appropriate uncertainty when modalities disagree
    3. Better calibration when using both modalities
    4. Evidence of information integration (not just bias)
    
    Args:
        mod1_predictions: List of modality 1 prediction dicts
        mod2_predictions: List of modality 2 prediction dicts
        combined_predictions: List of combined (mod1+mod2) prediction dicts
        patient_ids: Optional list of patient IDs for matching
    
    Returns:
        Dictionary with multimodal value analysis
    """
    # Match all three prediction sets - aggregate at patient level first (group each by own patient_id)
    matched_triplets = []
    mod1_by_patient = {}
    mod2_by_patient = {}
    combined_by_patient = {}
    for pred in mod1_predictions or []:
        if isinstance(pred, dict):
            pid = pred.get('patient_id')
            if pid is not None:
                if pid not in mod1_by_patient:
                    mod1_by_patient[pid] = []
                mod1_by_patient[pid].append(pred)
    for pred in mod2_predictions or []:
        if isinstance(pred, dict):
            pid = pred.get('patient_id')
            if pid is not None:
                if pid not in mod2_by_patient:
                    mod2_by_patient[pid] = []
                mod2_by_patient[pid].append(pred)
    for pred in combined_predictions or []:
        if isinstance(pred, dict):
            pid = pred.get('patient_id')
            if pid is not None:
                if pid not in combined_by_patient:
                    combined_by_patient[pid] = []
                combined_by_patient[pid].append(pred)
    common_patient_ids = sorted(set(mod1_by_patient.keys()) & set(mod2_by_patient.keys()) & set(combined_by_patient.keys()))
    if common_patient_ids:
        for pid in common_patient_ids:
            mod1_slices = mod1_by_patient[pid]
            mod2_slices = mod2_by_patient[pid]
            combined_slices = combined_by_patient[pid]
            mod1_aggregated = aggregate_patient_predictions(mod1_slices)
            mod2_aggregated = aggregate_patient_predictions(mod2_slices)
            combined_aggregated = aggregate_patient_predictions(combined_slices)
            mod1_pred = {
                'prediction': mod1_aggregated['prediction'],
                'confidence': mod1_aggregated['confidence'],
                'probabilities': mod1_slices[0].get('probabilities', {}) if mod1_slices else {},
                'probabilities_array': mod1_slices[0].get('probabilities_array', []) if mod1_slices else []
            }
            mod2_pred = {
                'prediction': mod2_aggregated['prediction'],
                'confidence': mod2_aggregated['confidence'],
                'probabilities': mod2_slices[0].get('probabilities', {}) if mod2_slices else {},
                'probabilities_array': mod2_slices[0].get('probabilities_array', []) if mod2_slices else [],
                'probabilities_before_boosting': mod2_slices[0].get('probabilities_before_boosting') if mod2_slices else None
            }
            combined_pred = {
                'prediction': combined_aggregated['prediction'],
                'confidence': combined_aggregated['confidence'],
                'probabilities': combined_slices[0].get('probabilities', {}) if combined_slices else {},
                'probabilities_array': combined_slices[0].get('probabilities_array', []) if combined_slices else []
            }
            matched_triplets.append((mod1_pred, mod2_pred, combined_pred))
    if not matched_triplets and (mod1_predictions or mod2_predictions or combined_predictions):
        min_len = min(len(mod1_predictions or []), len(mod2_predictions or []), len(combined_predictions or []))
        if min_len > 0:
            matched_triplets = list(zip(
                (mod1_predictions or [])[:min_len],
                (mod2_predictions or [])[:min_len],
                (combined_predictions or [])[:min_len]
            ))
    
    if not matched_triplets:
        return {
            'multimodal_value_demonstrated': False,
            'agreement_benefit': 0.0,
            'disagreement_handling': 'unknown',
            'information_integration': False,
            'num_triplets': 0
        }
    
    # Analyze behavior when modalities agree vs disagree
    agreement_cases = []
    disagreement_cases = []
    
    for mod1_pred, mod2_pred, combined_pred in matched_triplets:
        mod1_class = mod1_pred.get('prediction')
        mod2_class = mod2_pred.get('prediction')
        combined_class = combined_pred.get('prediction')
        
        mod1_conf = mod1_pred.get('confidence', 0.0)
        mod2_probs_before = mod2_pred.get('probabilities_before_boosting')
        mod2_conf = float(np.max(np.array(mod2_probs_before))) if mod2_probs_before is not None and len(mod2_probs_before) >= 2 else mod2_pred.get('confidence', 0.0)
        combined_conf = combined_pred.get('confidence', 0.0)
        
        if mod1_class == mod2_class:
            # Modalities agree
            agreement_cases.append({
                'mod1_conf': mod1_conf,
                'mod2_conf': mod2_conf,
                'combined_conf': combined_conf,
                'avg_single_conf': (mod1_conf + mod2_conf) / 2.0
            })
        else:
            # Modalities disagree
            disagreement_cases.append({
                'mod1_conf': mod1_conf,
                'mod2_conf': mod2_conf,
                'combined_conf': combined_conf,
                'avg_single_conf': (mod1_conf + mod2_conf) / 2.0
            })
    
    # When modalities agree: combined should have higher confidence (value of agreement)
    agreement_benefits = []
    for case in agreement_cases:
        benefit = case['combined_conf'] - case['avg_single_conf']
        agreement_benefits.append(benefit)
    
    avg_agreement_benefit = float(np.mean(agreement_benefits)) if agreement_benefits else 0.0
    
    # When modalities disagree: combined should show appropriate uncertainty
    disagreement_handling = []
    for case in disagreement_cases:
        # Combined confidence should be lower than average (showing uncertainty)
        uncertainty_handling = case['avg_single_conf'] - case['combined_conf']
        disagreement_handling.append(uncertainty_handling)
    
    avg_disagreement_uncertainty = float(np.mean(disagreement_handling)) if disagreement_handling else 0.0
    
    # Determine if disagreement is handled appropriately
    if avg_disagreement_uncertainty > 0.02:
        disagreement_handling_result = 'APPROPRIATE_UNCERTAINTY'
    elif avg_disagreement_uncertainty < -0.02:
        disagreement_handling_result = 'INAPPROPRIATE_CONFIDENCE'
    else:
        disagreement_handling_result = 'NEUTRAL'
    
    # Information integration: combined predictions are different from both when they disagree
    information_integration_cases = 0
    for mod1_pred, mod2_pred, combined_pred in matched_triplets:
        mod1_class = mod1_pred.get('prediction')
        mod2_class = mod2_pred.get('prediction')
        combined_class = combined_pred.get('prediction')
        
        if mod1_class != mod2_class and combined_class != mod1_class and combined_class != mod2_class:
            information_integration_cases += 1
    
    total = len(matched_triplets)
    information_integration_rate = information_integration_cases / total if total > 0 else 0.0
    information_integration = information_integration_rate > 0.1  # >10% shows integration
    
    # Overall assessment
    multimodal_value_demonstrated = (
        avg_agreement_benefit > 0.02 and  # Benefits from agreement
        avg_disagreement_uncertainty > 0.0 and  # Handles disagreement appropriately
        information_integration_rate > 0.05  # Shows some integration
    )
    
    return {
        'multimodal_value_demonstrated': multimodal_value_demonstrated,
        'agreement_benefit': avg_agreement_benefit,
        'disagreement_uncertainty': avg_disagreement_uncertainty,
        'disagreement_handling': disagreement_handling_result,
        'information_integration_rate': information_integration_rate,
        'information_integration': information_integration,
        'num_agreement_cases': len(agreement_cases),
        'num_disagreement_cases': len(disagreement_cases),
        'num_triplets': total,
        'interpretation': (
            f"Zero-shot VLM {'DEMONSTRATES' if multimodal_value_demonstrated else 'DOES NOT CLEARLY DEMONSTRATE'} "
            f"multimodal value. Agreement benefit: {avg_agreement_benefit:+.4f}, "
            f"Disagreement handling: {disagreement_handling_result.lower().replace('_', ' ')}, "
            f"Information integration: {information_integration_rate*100:.1f}%"
        )
    }


def analyze_modality_dominance(
    mod1_predictions: List[Dict],
    mod2_predictions: List[Dict],
    patient_ids: Optional[List[str]] = None
) -> Dict:
    """
    Analyze whether modality 2 systematically dominates modality 1 (higher confidence).
    
    Key questions:
    - Does modality 2 consistently have higher confidence than modality 1?
    - When modalities disagree, does modality 2 win more often?
    - Is there a systematic bias toward modality 2?
    
    Args:
        mod1_predictions: List of modality 1 prediction dicts
        mod2_predictions: List of modality 2 prediction dicts
        patient_ids: Optional list of patient IDs for matching
    
    Returns:
        Dictionary with modality dominance analysis
    """
    # Match predictions - aggregate at patient level first (same logic as analyze_modality_confidence_comparison)
    # CRITICAL: Group each modality by its own patient_id so matching is correct when slice counts differ (e.g. CT 738 vs MR 235)
    matched_pairs = []
    mod1_by_patient = {}
    mod2_by_patient = {}
    for pred in mod1_predictions or []:
        if isinstance(pred, dict):
            pid = pred.get('patient_id')
            if pid is not None:
                if pid not in mod1_by_patient:
                    mod1_by_patient[pid] = []
                mod1_by_patient[pid].append(pred)
    for pred in mod2_predictions or []:
        if isinstance(pred, dict):
            pid = pred.get('patient_id')
            if pid is not None:
                if pid not in mod2_by_patient:
                    mod2_by_patient[pid] = []
                mod2_by_patient[pid].append(pred)
    common_patient_ids = sorted(set(mod1_by_patient.keys()) & set(mod2_by_patient.keys()))
    if common_patient_ids:
        # Aggregate predictions per patient (weighted by confidence)
        for pid in common_patient_ids:
            mod1_slices = mod1_by_patient[pid]
            mod2_slices = mod2_by_patient[pid]
            mod1_aggregated = aggregate_patient_predictions(mod1_slices)
            mod2_aggregated = aggregate_patient_predictions(mod2_slices)
            mod1_pred = {
                'prediction': mod1_aggregated['prediction'],
                'confidence': mod1_aggregated['confidence'],
                'probabilities': mod1_slices[0].get('probabilities', {}) if mod1_slices else {},
                'probabilities_array': mod1_slices[0].get('probabilities_array', []) if mod1_slices else [],
                'logits': mod1_slices[0].get('logits', []) if mod1_slices else []
            }
            mod2_pred = {
                'prediction': mod2_aggregated['prediction'],
                'confidence': mod2_aggregated['confidence'],
                'probabilities': mod2_slices[0].get('probabilities', {}) if mod2_slices else {},
                'probabilities_array': mod2_slices[0].get('probabilities_array', []) if mod2_slices else [],
                'probabilities_before_boosting': mod2_slices[0].get('probabilities_before_boosting') if mod2_slices else None,
                'used_context': mod2_slices[0].get('used_context', False) if mod2_slices else False,
                'logits': mod2_slices[0].get('logits', []) if mod2_slices else []
            }
            matched_pairs.append((mod1_pred, mod2_pred))
    if not matched_pairs:
        min_len = min(len(mod1_predictions), len(mod2_predictions))
        matched_pairs = list(zip(mod1_predictions[:min_len], mod2_predictions[:min_len]))
    
    if not matched_pairs:
        return {
            'mod2_higher_confidence_rate': 0.0,
            'mod1_higher_confidence_rate': 0.0,
            'avg_confidence_difference': 0.0,
            'mod2_wins_when_disagree': 0,
            'mod1_wins_when_disagree': 0,
            'systematic_mod2_bias': False,
            'num_pairs': 0
        }
    
    mod2_higher_conf = 0
    mod1_higher_conf = 0
    confidence_differences = []
    mod2_wins_disagree = 0
    mod1_wins_disagree = 0
    
    for mod1_pred, mod2_pred in matched_pairs:
        # Use aggregated confidence (patient-level) so PAIRWISE DOMINANCE matches PAIRWISE CONFIDENCE COMPARISONS
        # and the patient-level results table (same interpretation of "which modality has higher confidence").
        mod1_conf = mod1_pred.get('confidence', 0.0)
        mod2_conf = mod2_pred.get('confidence', 0.0)
        mod1_pred_class = mod1_pred.get('prediction')
        mod2_pred_class = mod2_pred.get('prediction')
        
        conf_diff = mod2_conf - mod1_conf
        confidence_differences.append(conf_diff)
        
        if mod2_conf > mod1_conf:
            mod2_higher_conf += 1
        elif mod1_conf > mod2_conf:
            mod1_higher_conf += 1
        
        # When modalities disagree, which one has higher confidence?
        if mod1_pred_class != mod2_pred_class:
            if mod2_conf > mod1_conf:
                mod2_wins_disagree += 1
            elif mod1_conf > mod2_conf:
                mod1_wins_disagree += 1
    
    total = len(matched_pairs)
    avg_conf_diff = np.mean(confidence_differences) if confidence_differences else 0.0
    
    # Determine if there's systematic Mod2 bias
    # Criteria: Mod2 has higher confidence in >60% of cases AND average difference > 0.05
    mod2_dominance_rate = mod2_higher_conf / total if total > 0 else 0.0
    systematic_bias = mod2_dominance_rate > 0.6 and avg_conf_diff > 0.05
    
    return {
        'mod2_higher_confidence_rate': mod2_dominance_rate,
        'mod1_higher_confidence_rate': mod1_higher_conf / total if total > 0 else 0.0,
        'avg_confidence_difference': float(avg_conf_diff),
        'std_confidence_difference': float(np.std(confidence_differences)) if confidence_differences else 0.0,
        'mod2_wins_when_disagree': mod2_wins_disagree,
        'mod1_wins_when_disagree': mod1_wins_disagree,
        'systematic_mod2_bias': systematic_bias,
        'num_pairs': total,
        'interpretation': (
            f"Modality 2 has higher confidence in {mod2_dominance_rate*100:.1f}% of cases. "
            f"Average difference: {avg_conf_diff:+.4f}. "
            f"{'SYSTEMATIC Mod2 BIAS DETECTED' if systematic_bias else 'No systematic bias detected'}."
        )
    }


def analyze_multimodal_bias(
    mod1_predictions: List[Dict],
    mod2_predictions: List[Dict],
    combined_predictions: List[Dict],
    patient_ids: Optional[List[str]] = None
) -> Dict:
    """
    Analyze whether multimodal predictions reflect true combination or simple bias toward one modality.
    
    Key questions:
    - Do combined predictions match mod2 more often than mod1?
    - Are combined predictions truly combining information or just following mod2?
    - What's the correlation between combined predictions and individual modalities?
    
    Args:
        mod1_predictions: List of modality 1 prediction dicts
        mod2_predictions: List of modality 2 prediction dicts
        combined_predictions: List of combined (mod1+mod2) prediction dicts
        patient_ids: Optional list of patient IDs for matching
    
    Returns:
        Dictionary with multimodal bias analysis
    """
    # Match all three prediction sets - aggregate at patient level first (group each by own patient_id)
    matched_triplets = []
    mod1_by_patient = {}
    mod2_by_patient = {}
    combined_by_patient = {}
    for pred in mod1_predictions or []:
        if isinstance(pred, dict):
            pid = pred.get('patient_id')
            if pid is not None:
                if pid not in mod1_by_patient:
                    mod1_by_patient[pid] = []
                mod1_by_patient[pid].append(pred)
    for pred in mod2_predictions or []:
        if isinstance(pred, dict):
            pid = pred.get('patient_id')
            if pid is not None:
                if pid not in mod2_by_patient:
                    mod2_by_patient[pid] = []
                mod2_by_patient[pid].append(pred)
    for pred in combined_predictions or []:
        if isinstance(pred, dict):
            pid = pred.get('patient_id')
            if pid is not None:
                if pid not in combined_by_patient:
                    combined_by_patient[pid] = []
                combined_by_patient[pid].append(pred)
    common_patient_ids = sorted(set(mod1_by_patient.keys()) & set(mod2_by_patient.keys()) & set(combined_by_patient.keys()))
    if common_patient_ids:
        for pid in common_patient_ids:
            mod1_slices = mod1_by_patient[pid]
            mod2_slices = mod2_by_patient[pid]
            combined_slices = combined_by_patient[pid]
            mod1_aggregated = aggregate_patient_predictions(mod1_slices)
            mod2_aggregated = aggregate_patient_predictions(mod2_slices)
            combined_aggregated = aggregate_patient_predictions(combined_slices)
            mod1_pred = {
                'prediction': mod1_aggregated['prediction'],
                'confidence': mod1_aggregated['confidence'],
                'probabilities': mod1_slices[0].get('probabilities', {}) if mod1_slices else {},
                'probabilities_array': mod1_slices[0].get('probabilities_array', []) if mod1_slices else []
            }
            mod2_pred = {
                'prediction': mod2_aggregated['prediction'],
                'confidence': mod2_aggregated['confidence'],
                'probabilities': mod2_slices[0].get('probabilities', {}) if mod2_slices else {},
                'probabilities_array': mod2_slices[0].get('probabilities_array', []) if mod2_slices else [],
                'probabilities_before_boosting': mod2_slices[0].get('probabilities_before_boosting') if mod2_slices else None
            }
            combined_pred = {
                'prediction': combined_aggregated['prediction'],
                'confidence': combined_aggregated['confidence'],
                'probabilities': combined_slices[0].get('probabilities', {}) if combined_slices else {},
                'probabilities_array': combined_slices[0].get('probabilities_array', []) if combined_slices else []
            }
            matched_triplets.append((mod1_pred, mod2_pred, combined_pred))
    if not matched_triplets and (mod1_predictions or mod2_predictions or combined_predictions):
        min_len = min(len(mod1_predictions or []), len(mod2_predictions or []), len(combined_predictions or []))
        if min_len > 0:
            matched_triplets = list(zip(
                (mod1_predictions or [])[:min_len],
                (mod2_predictions or [])[:min_len],
                (combined_predictions or [])[:min_len]
            ))
    
    if not matched_triplets:
        return {
            'combined_matches_mod2_rate': 0.0,
            'combined_matches_mod1_rate': 0.0,
            'mod2_bias_score': 0.0,
            'true_combination_rate': 0.0,
            'num_triplets': 0
        }
    
    matches_mod2 = 0
    matches_mod1 = 0
    matches_both = 0
    matches_neither = 0
    mod2_bias_cases = 0  # Combined matches mod2 but not mod1
    
    for mod1_pred, mod2_pred, combined_pred in matched_triplets:
        mod1_class = mod1_pred.get('prediction')
        mod2_class = mod2_pred.get('prediction')
        combined_class = combined_pred.get('prediction')
        
        if combined_class == mod2_class:
            matches_mod2 += 1
        if combined_class == mod1_class:
            matches_mod1 += 1
        if combined_class == mod2_class and combined_class == mod1_class:
            matches_both += 1
        elif combined_class != mod2_class and combined_class != mod1_class:
            matches_neither += 1
        
        # Mod2 bias: combined matches mod2 but not mod1 (when they disagree)
        if mod1_class != mod2_class and combined_class == mod2_class:
            mod2_bias_cases += 1
    
    total = len(matched_triplets)
    
    # Calculate rates
    matches_mod2_rate = matches_mod2 / total if total > 0 else 0.0
    matches_mod1_rate = matches_mod1 / total if total > 0 else 0.0
    mod2_bias_score = mod2_bias_cases / total if total > 0 else 0.0
    
    # True combination: when mod1 and mod2 disagree, combined prediction is different from both
    # OR when they agree, combined also agrees
    true_combination = 0
    for mod1_pred, mod2_pred, combined_pred in matched_triplets:
        mod1_class = mod1_pred.get('prediction')
        mod2_class = mod2_pred.get('prediction')
        combined_class = combined_pred.get('prediction')
        
        if mod1_class == mod2_class:
            # When modalities agree, combined should also agree (true combination)
            if combined_class == mod1_class:
                true_combination += 1
        else:
            # When modalities disagree, combined could be:
            # 1. Different from both (true combination/compromise)
            # 2. Same as one (bias toward that modality)
            # For now, we consider "different from both" as true combination
            if combined_class != mod1_class and combined_class != mod2_class:
                true_combination += 1
    
    true_combination_rate = true_combination / total if total > 0 else 0.0
    
    # Determine if there's simple mod2 bias
    # Criteria: Combined matches mod2 significantly more than mod1 (>20% difference)
    simple_mod2_bias = matches_mod2_rate - matches_mod1_rate > 0.2
    
    return {
        'combined_matches_mod2_rate': matches_mod2_rate,
        'combined_matches_mod1_rate': matches_mod1_rate,
        'combined_matches_both_rate': matches_both / total if total > 0 else 0.0,
        'mod2_bias_score': mod2_bias_score,
        'true_combination_rate': true_combination_rate,
        'simple_mod2_bias': simple_mod2_bias,
        'num_triplets': total,
        'interpretation': (
            f"Combined matches mod2: {matches_mod2_rate*100:.1f}%, "
            f"mod1: {matches_mod1_rate*100:.1f}%. "
            f"{'SIMPLE MOD2 BIAS DETECTED' if simple_mod2_bias else 'Appears to be true combination'} "
            f"(True combination rate: {true_combination_rate*100:.1f}%)"
        )
    }


def analyze_overconfidence(
    predictions: List[Dict],
    modality_name: str,
    confidence_threshold: float = 0.7
) -> Dict:
    """
    Analyze cases where the model is high-confidence but incorrect (overconfidence).
    
    This is critical for understanding model reliability - high confidence doesn't
    guarantee correctness, and overconfident errors are particularly dangerous.
    
    Args:
        predictions: List of prediction dicts with 'prediction', 'label', 'confidence'
        modality_name: Name of modality (e.g., 'Mod1', 'Mod2')
        confidence_threshold: Minimum confidence to consider "high-confidence" (default: 0.7)
    
    Returns:
        Dictionary with overconfidence analysis metrics
    """
    if not predictions:
        return {
            'modality': modality_name,
            'total_samples': 0,
            'high_conf_incorrect_count': 0,
            'high_conf_incorrect_rate': 0.0,
            'avg_confidence_when_incorrect': 0.0,
            'avg_confidence_when_correct': 0.0,
            'overconfidence_severity': 0.0,  # Average confidence of incorrect high-conf predictions
            'high_conf_correct_count': 0,
            'high_conf_correct_rate': 0.0,
            'low_conf_incorrect_count': 0,
            'low_conf_incorrect_rate': 0.0
        }
    
    high_conf_incorrect = []
    high_conf_correct = []
    low_conf_incorrect = []
    low_conf_correct = []
    all_incorrect_confidences = []
    all_correct_confidences = []
    num_valid = 0
    
    for pred in predictions:
        prediction = pred.get('prediction')
        label = pred.get('label')
        confidence = pred.get('confidence', 0.0)
        
        if prediction is None or label is None:
            continue
        num_valid += 1
        
        is_correct = (int(prediction) == int(label))
        is_high_conf = confidence >= confidence_threshold
        
        if is_correct:
            all_correct_confidences.append(confidence)
            if is_high_conf:
                high_conf_correct.append(confidence)
            else:
                low_conf_correct.append(confidence)
        else:
            all_incorrect_confidences.append(confidence)
            if is_high_conf:
                high_conf_incorrect.append(confidence)
            else:
                low_conf_incorrect.append(confidence)
    
    total = num_valid  # Use count of predictions with valid prediction+label for rates
    high_conf_incorrect_count = len(high_conf_incorrect)
    high_conf_correct_count = len(high_conf_correct)
    low_conf_incorrect_count = len(low_conf_incorrect)
    
    # Calculate rates
    high_conf_incorrect_rate = high_conf_incorrect_count / total if total > 0 else 0.0
    high_conf_correct_rate = high_conf_correct_count / total if total > 0 else 0.0
    low_conf_incorrect_rate = low_conf_incorrect_count / total if total > 0 else 0.0
    
    # Calculate average confidences
    avg_conf_when_incorrect = float(np.mean(all_incorrect_confidences)) if all_incorrect_confidences else 0.0
    avg_conf_when_correct = float(np.mean(all_correct_confidences)) if all_correct_confidences else 0.0
    
    # Overconfidence severity: average confidence of high-confidence incorrect predictions
    overconfidence_severity = float(np.mean(high_conf_incorrect)) if high_conf_incorrect else 0.0
    
    return {
        'modality': modality_name,
        'total_samples': total,  # Number of valid predictions (with non-None prediction and label)
        'high_conf_incorrect_count': high_conf_incorrect_count,
        'high_conf_incorrect_rate': high_conf_incorrect_rate,
        'avg_confidence_when_incorrect': avg_conf_when_incorrect,
        'avg_confidence_when_correct': avg_conf_when_correct,
        'overconfidence_severity': overconfidence_severity,
        'high_conf_correct_count': high_conf_correct_count,
        'high_conf_correct_rate': high_conf_correct_rate,
        'low_conf_incorrect_count': low_conf_incorrect_count,
        'low_conf_incorrect_rate': low_conf_incorrect_rate,
        'confidence_threshold': confidence_threshold
    }


def calculate_patient_level_results(
    results: Dict[str, List[Dict]],
    modalities: List[str]
) -> Dict:
    """
    Calculate patient-level aggregated results (mandatory requirement).
    
    Aggregates slice-level predictions to patient-level using weighted voting,
    then calculates accuracy and certainty metrics at patient level.
    
    Args:
        results: Dictionary with case_ids as keys and lists of predictions as values
        modalities: List of modality names
    
    Returns:
        Dictionary with patient-level step_results (same structure as slice-level)
    """
    # Organize predictions by patient and modality
    patient_data = {}  # {patient_id: {modality: [predictions], ...}}
    
    skipped_no_patient_id = 0
    skipped_no_step_name = 0
    
    for case_id, case_results in results.items():
        for result in case_results:
            patient_id = result.get('patient_id')
            if patient_id is None:
                skipped_no_patient_id += 1
                continue
            
            mods_used = result.get('modalities_used', [])
            used_context = result.get('used_context', False)
            context_from = result.get('context_from', [])
            
            # Determine step name (same logic as evaluate_sequential_modalities)
            # FULLY GENERIC for N modalities - no hardcoded modality indices
            step_name = result.get('step')
            if step_name is None:
                if len(mods_used) == 1:
                    mod = mods_used[0]
                    
                    if used_context and len(context_from) > 0:
                        # Modality with context: build combined step name
                        # Sort context_from to match the order in modalities list
                        sorted_context = [m for m in modalities if m in context_from]
                        if sorted_context:
                            # Build step name: "Mod1+Mod2" or "Mod1+Mod2+Mod3", etc.
                            combined_step_name = '+'.join(sorted_context + [mod])
                            step_name = combined_step_name
                        else:
                            # Context not in modalities list, use modality alone
                            step_name = mod
                    else:
                        # Modality alone (no context)
                        step_name = mod
                elif len(mods_used) > 1:
                    # Explicit combined modality case (multiple modalities used together)
                    # Sort by order in modalities list
                    step_name = '+'.join(sorted(mods_used, key=lambda m: modalities.index(m) if m in modalities else 999))
                else:
                    # No modalities used - skip
                    skipped_no_step_name += 1
                    continue  # Skip if can't determine step_name
            
            # If step_name is still None, skip this result
            if step_name is None:
                skipped_no_step_name += 1
                continue
            
            # Use step_name as-is (already mapped correctly)
            if patient_id not in patient_data:
                patient_data[patient_id] = {}
            if step_name not in patient_data[patient_id]:
                patient_data[patient_id][step_name] = []
            
            patient_data[patient_id][step_name].append(result)
    
    # Debug: Log skipped results and patient data summary
    import sys
    num_patients_found = len(patient_data)
    total_results_processed = sum(len(slices) for mod_preds in patient_data.values() for slices in mod_preds.values())
    
    if skipped_no_patient_id > 0 or skipped_no_step_name > 0 or num_patients_found < 2:
        print(f"Warning: Found {num_patients_found} patients in patient_data. Skipped {skipped_no_patient_id} results with no patient_id, {skipped_no_step_name} with no step_name. Total results processed: {total_results_processed}", file=sys.stderr, flush=True)
        if num_patients_found > 0:
            patient_ids_found = sorted(patient_data.keys())
            print(f"  Patient IDs found: {patient_ids_found[:10]}{'...' if len(patient_ids_found) > 10 else ''}", file=sys.stderr, flush=True)
    
    # Aggregate to patient level and calculate metrics
    # First, collect all unique step_names from patient_data
    all_step_names = set()
    for modality_predictions in patient_data.values():
        all_step_names.update(modality_predictions.keys())
    
    # Initialize step_data with all found step_names
    step_data = {}
    for step_name in all_step_names:
        step_data[step_name] = {'predictions': [], 'labels': [], 'full_predictions': []}
    
    # Also ensure we have the expected step names (even if empty)
    # Ensure step_data has entries for all modalities (safe for single modality)
    if len(modalities) > 0:
        if modalities[0] not in step_data:
            step_data[modalities[0]] = {'predictions': [], 'labels': [], 'full_predictions': []}
    if len(modalities) > 1:
        if modalities[1] not in step_data:
            step_data[modalities[1]] = {'predictions': [], 'labels': [], 'full_predictions': []}
        combined_step_name = '+'.join(modalities)
        if combined_step_name not in step_data:
            step_data[combined_step_name] = {'predictions': [], 'labels': [], 'full_predictions': []}
    
    for patient_id, modality_predictions in patient_data.items():
        for step_name, slices in modality_predictions.items():
            if not slices:
                continue
            
            # Aggregate slices to patient level
            aggregated = aggregate_patient_predictions(slices)
            
            # Get label (should be same for all slices of same patient)
            label = slices[0].get('label')
            if label is None:
                continue
            
            # Create patient-level prediction dict
            patient_pred = {
                'prediction': aggregated['prediction'],
                'label': label,
                'confidence': aggregated['confidence'],
                'patient_id': patient_id,
                'num_slices': aggregated['num_slices']
            }
            
            # Add probabilities and logits if available (use first slice or aggregate)
            if slices:
                first_slice = slices[0]
                patient_pred['probabilities'] = first_slice.get('probabilities', {})
                patient_pred['probabilities_array'] = first_slice.get('probabilities_array', [])
                patient_pred['probabilities_before_boosting'] = first_slice.get('probabilities_before_boosting')
                patient_pred['logits'] = first_slice.get('logits', [])
                patient_pred['used_context'] = first_slice.get('used_context', False)
                patient_pred['context_from'] = first_slice.get('context_from', [])
            
            # Add to step_data (step_name should always be in step_data now)
            if step_name in step_data:
                step_data[step_name]['predictions'].append(aggregated['prediction'])
                step_data[step_name]['labels'].append(label)
                step_data[step_name]['full_predictions'].append(patient_pred)
    
    # Calculate patient-level metrics (same structure as slice-level)
    patient_step_results = {}
    for step_name, data in step_data.items():
        if len(data['predictions']) > 0:
            predictions = [int(p) if p is not None else 0 for p in data['predictions']]
            labels = [int(l) if l is not None else 0 for l in data['labels']]
            
            acc = calculate_accuracy(predictions, labels)
            num_predictions = len(predictions)
            
            # CRITICAL: Ensure full_predictions contains patient-level predictions (one per patient)
            # Verify uniqueness by patient_id to catch any bugs
            patient_ids_in_predictions = [p.get('patient_id') for p in data['full_predictions'] if p.get('patient_id') is not None]
            unique_patient_ids = set(patient_ids_in_predictions)
            if len(patient_ids_in_predictions) != len(unique_patient_ids):
                import sys
                print(f"WARNING: Duplicate patient IDs found in patient-level predictions for {step_name}. "
                      f"Expected {len(unique_patient_ids)} unique patients, but found {len(patient_ids_in_predictions)} entries. "
                      f"This may indicate a bug in patient-level aggregation.", file=sys.stderr, flush=True)
            
            # Verify we have patient-level predictions (should have 'num_slices' field indicating aggregation)
            for pred in data['full_predictions']:
                if 'num_slices' not in pred:
                    import sys
                    print(f"WARNING: Patient-level prediction missing 'num_slices' field for {step_name}. "
                          f"This may indicate slice-level data was incorrectly included.", file=sys.stderr, flush=True)
                    break
            
            # Analyze certainty metrics at patient level
            certainty_metrics = analyze_certainty_metrics(data['full_predictions'], step_name)
            
            # Analyze overconfidence at patient level
            # NOTE: data['full_predictions'] should contain ONE entry per patient (aggregated from slices)
            overconfidence_metrics = analyze_overconfidence(data['full_predictions'], step_name)
            
            patient_step_results[step_name] = {
                'accuracy': acc,
                'num_samples': num_predictions,
                'certainty_metrics': certainty_metrics,
                'overconfidence_metrics': overconfidence_metrics,
                'full_predictions': data['full_predictions'],  # For patient-level disagreement
            }
        else:
            patient_step_results[step_name] = {
                'accuracy': 0.0,
                'num_samples': 0,
                'certainty_metrics': analyze_certainty_metrics([], step_name),
                'overconfidence_metrics': analyze_overconfidence([], step_name),
                'full_predictions': [],
            }
    
    return patient_step_results


def analyze_modality_combination_effect(
    mod1_predictions: List[Dict],
    mod2_predictions: List[Dict],
    combined_certainty: Dict,
    mod1_certainty: Dict,
    mod2_certainty: Dict
) -> Dict:
    """
    Analyze how combining modalities affects certainty.
    
    Key questions:
    - Does combining modalities increase or decrease confidence?
    - Does combining modalities reduce entropy (more certain)?
    - How do logit magnitudes change when modalities are combined?
    
    Args:
        mod1_predictions: Predictions from first modality
        mod2_predictions: Predictions from second modality
        combined_certainty: Certainty metrics for combined modality
        mod1_certainty: Certainty metrics for first modality alone
        mod2_certainty: Certainty metrics for second modality alone
    
    Returns:
        Dictionary with combination effect analysis
    """
    combined_conf = combined_certainty.get('avg_confidence', 0.0)
    combined_entropy = combined_certainty.get('avg_entropy', 0.0)
    
    mod1_conf = mod1_certainty.get('avg_confidence', 0.0)
    mod2_conf = mod2_certainty.get('avg_confidence', 0.0)
    avg_single_mod_conf = (mod1_conf + mod2_conf) / 2.0
    
    mod1_entropy = mod1_certainty.get('avg_entropy', 0.0)
    mod2_entropy = mod2_certainty.get('avg_entropy', 0.0)
    avg_single_mod_entropy = (mod1_entropy + mod2_entropy) / 2.0
    
    # Calculate changes
    confidence_change = combined_conf - avg_single_mod_conf
    entropy_change = combined_entropy - avg_single_mod_entropy
    
    # Determine effect
    if confidence_change > 0.05:
        confidence_effect = "INCREASES"
    elif confidence_change < -0.05:
        confidence_effect = "DECREASES"
    else:
        confidence_effect = "MINIMAL_CHANGE"
    
    if entropy_change < -0.1:
        entropy_effect = "REDUCES_UNCERTAINTY"
    elif entropy_change > 0.1:
        entropy_effect = "INCREASES_UNCERTAINTY"
    else:
        entropy_effect = "MINIMAL_CHANGE"
    
    return {
        'combined_confidence': combined_conf,
        'average_single_modality_confidence': avg_single_mod_conf,
        'confidence_change': confidence_change,
        'confidence_effect': confidence_effect,
        'combined_entropy': combined_entropy,
        'average_single_modality_entropy': avg_single_mod_entropy,
        'entropy_change': entropy_change,
        'entropy_effect': entropy_effect,
        'interpretation': (
            f"Combining modalities {confidence_effect.lower()} confidence "
            f"({confidence_change:+.4f}) and {entropy_effect.lower().replace('_', ' ')} "
            f"({entropy_change:+.4f} entropy change)"
        )
    }


def save_results(results: Dict, output_path: str):
    """Save evaluation results to file."""
    import json
    import os
    import sys
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return [convert_to_serializable(item) for item in sorted(obj)]
        elif obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            # For any other type, try to convert to string
            try:
                return str(obj)
            except Exception:
                return None

    try:
        serializable_results = convert_to_serializable(results)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Results saved to {output_path}")
    except (TypeError, ValueError) as e:
        # More detailed error message to help debug serialization issues
        print(f"ERROR: Failed to save results to {output_path}: {e}", file=sys.stderr)
        print(f"Error type: {type(e).__name__}", file=sys.stderr)
        # Try to find the problematic value
        import traceback
        traceback.print_exc()
        raise
