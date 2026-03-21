#!/bin/bash
# Submit 10 vision-language models for sequential evaluation (forward + reverse via submit_single_model.sh).
#
# data1 = Lung-PET-CT-Dx-PNG (fast PNG I/O). Raw DICOM tree is Lung-PET-CT-Dx (not used here).
#
# Usage:
#   ./submit_all10_models_data1.sh              # DEFAULT: Lung-PET-CT-Dx-PNG, CT+PT, non_small_cell vs small_cell
#   ./submit_all10_models_data1.sh kirp         # TCGA-KIRP (data2): CT MR PT, early_stage vs advanced_stage
#
# Override paths (optional):
#   DATA_ROOT=... DATASET_CONFIG=... ./submit_all10_models_data1.sh
#
# Requires: submit_single_model.sh, SLURM (sbatch) on Newton.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

PRESET="${1:-lung}"

if [[ "$PRESET" == "kirp" || "$PRESET" == "data2" || "$PRESET" == "TCGA-KIRP" ]]; then
  DATA_ROOT="${DATA_ROOT:-data2}"
  DATASET_CONFIG="${DATASET_CONFIG:-data2/tcga_kirp_config.yaml}"
  CLASS1="early_stage"
  CLASS2="advanced_stage"
  MODALITIES=("CT" "MR" "PT")
  DATASET_LABEL="data2/TCGA-KIRP (early_stage vs advanced_stage)"
else
  # lung | Lung-PET-CT-Dx | png — PNG dataset root
  DATA_ROOT="${DATA_ROOT:-Lung-PET-CT-Dx-PNG}"
  DATASET_CONFIG="${DATASET_CONFIG:-configs/lung_pet_ct_dx_png.yaml}"
  CLASS1="non_small_cell"
  CLASS2="small_cell"
  MODALITIES=("CT" "PT")
  DATASET_LABEL="Lung-PET-CT-Dx-PNG (non_small_cell vs small_cell)"
fi

if [[ ! -f "$DATASET_CONFIG" ]]; then
  echo "Error: dataset config not found: $DATASET_CONFIG" >&2
  echo "  Set DATASET_CONFIG=... or run from repo root." >&2
  exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Warning: data root not found: $DATA_ROOT (jobs may fail on cluster if path differs)." >&2
fi

echo "=========================================="
echo "Submitting 10 models — $DATASET_LABEL"
echo "=========================================="
echo "DATA_ROOT=$DATA_ROOT"
echo "DATASET_CONFIG=$DATASET_CONFIG"
echo "Classes: $CLASS1, $CLASS2"
echo "Modalities: ${MODALITIES[*]}"
echo ""

# 10 models: "HF_model_id:arch"  (arch = clip | llava | llava_med)
# Edit this list to match your experiments.
declare -a MODEL_ENTRIES=(
  "openai/clip-vit-base-patch32:clip"
  "openai/clip-vit-base-patch16:clip"
  "openai/clip-vit-large-patch14:clip"
  "openai/clip-vit-large-patch14-336:clip"
  "openai/clip-vit-huge-patch14:clip"
  "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224:clip"
  "google/siglip-base-patch16-224:clip"
  "google/siglip-large-patch16-256:clip"
  "google/siglip-so400m-patch14-384:clip"
  "google/siglip-large-patch16-384:clip"
)

i=0
for entry in "${MODEL_ENTRIES[@]}"; do
  i=$((i + 1))
  MODEL_NAME="${entry%%:*}"
  MODEL_ARCH="${entry##*:}"
  echo "------------------------------------------"
  echo "[$i/10] $MODEL_NAME ($MODEL_ARCH)"
  echo "------------------------------------------"
  ./submit_single_model.sh \
    "$MODEL_NAME" \
    "$MODEL_ARCH" \
    "$DATA_ROOT" \
    "$DATASET_CONFIG" \
    "$CLASS1" \
    "$CLASS2" \
    "${MODALITIES[@]}"
  echo ""
done

echo "=========================================="
echo "All 10 submit_single_model jobs queued."
echo "Monitor: squeue -u \$USER"
echo "Logs:    output_*.log  error_*.log"
echo "=========================================="
