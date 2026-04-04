
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1


export SLURM_TIME="${SLURM_TIME:-34:00:00}"

PRESET="${1:-lung}"

if [[ "$PRESET" == "kirp" || "$PRESET" == "data2" || "$PRESET" == "TCGA-KIRP" ]]; then
  DATA_ROOT="${DATA_ROOT:-data2}"
  DATASET_CONFIG="${DATASET_CONFIG:-data2/tcga_kirp_config.yaml}"
  CLASS1="early_stage"
  CLASS2="advanced_stage"
  MODALITIES=("CT" "MR" "PT")
  DATASET_LABEL="data2/TCGA-KIRP (early_stage vs advanced_stage)"
else
  DATA_ROOT="${DATA_ROOT:-Lung-PET-CT-Dx-PNG}"
  DATASET_CONFIG="${DATASET_CONFIG:-configs/lung_pet_ct_dx.yaml}"
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
echo "SLURM_TIME per job: $SLURM_TIME"
echo "=========================================="
echo "DATA_ROOT=$DATA_ROOT"
echo "DATASET_CONFIG=$DATASET_CONFIG"
echo "Classes: $CLASS1, $CLASS2"
echo "Modalities: ${MODALITIES[*]}"
echo ""
echo "Models (display names):"
echo "  1. BioMedCLIP   2. LAION-CLIP   3. OpenAI CLIP (ViT-B/32)   4. OpenAI CLIP (ViT-L/14)   5. OpenAI CLIP (ViT-B/16)"
echo "  6. PubMedCLIP   7. MetaCLIP   8. SigLIP (Google)   9. DFN (Apple)   10. LLaVA-Med"
echo ""

declare -a MODEL_DISPLAY_NAMES=(
  "BioMedCLIP"
  "LAION-CLIP"
  "OpenAI CLIP (ViT-B/32)"
  "OpenAI CLIP (ViT-L/14)"
  "OpenAI CLIP (ViT-B/16)"
  "PubMedCLIP"
  "MetaCLIP"
  "SigLIP (Google)"
  "DFN (Apple)"
  "LLaVA-Med"
)

# HuggingFace:arch (clip | llava_med)
declare -a MODEL_ENTRIES=(
  "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224:clip"
  "laion/CLIP-ViT-B-32-laion2B-s34B-b79K:clip"
  "openai/clip-vit-base-patch32:clip"
  "openai/clip-vit-large-patch14:clip"
  "openai/clip-vit-base-patch16:clip"
  "sarahESL/PubMedCLIP:clip"
  "facebook/metaclip-b32-fullcc2.5b:clip"
  "google/siglip-base-patch16-224:clip"
  "apple/DFN2B-CLIP-ViT-B-16:clip"
  "microsoft/llava-med-v1.5-mistral-7b:llava_med"
)

declare -a SLURM_JOB_TAGS=(
  "d1-BioMedCLIP"
  "d1-LAION-CLIP"
  "d1-OpenAI-CLIP-ViT-B-32"
  "d1-OpenAI-CLIP-ViT-L-14"
  "d1-OpenAI-CLIP-ViT-B-16"
  "d1-PubMedCLIP"
  "d1-MetaCLIP"
  "d1-SigLIP-Google"
  "d1-DFN-Apple"
  "d1-LLaVA-Med"
)

i=0
for idx in "${!MODEL_ENTRIES[@]}"; do
  i=$((idx + 1))
  entry="${MODEL_ENTRIES[$idx]}"
  MODEL_NAME="${entry%%:*}"
  MODEL_ARCH="${entry##*:}"
  export SLURM_JOB_TAG="${SLURM_JOB_TAGS[$idx]}"
  echo "------------------------------------------"
  echo "[$i/10] ${MODEL_DISPLAY_NAMES[$idx]}"
  echo "HuggingFace: $MODEL_NAME  |  arch=$MODEL_ARCH  |  SLURM job name: $SLURM_JOB_TAG"
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
unset SLURM_JOB_TAG

echo "=========================================="
echo "All 10 submit_single_model jobs queued."
echo "Monitor: squeue -u \$USER"
echo "Logs:    output_*.log  error_*.log"
echo "=========================================="
