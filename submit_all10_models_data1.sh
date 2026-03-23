
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
  # lung | Lung-PET-CT-Dx | png — PNG dataset root (absolute path under this repo)
  DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/Lung-PET-CT-Dx-PNG}"
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
  echo "Error: data root not found: $DATA_ROOT" >&2
  echo "  Place Lung-PET-CT-Dx-PNG next to this script or set DATA_ROOT=/path/to/Lung-PET-CT-Dx-PNG" >&2
  exit 1
fi

echo "=========================================="
echo "Submitting 10 models — $DATASET_LABEL"
echo "=========================================="
echo "DATA_ROOT=$DATA_ROOT"
echo "DATASET_CONFIG=$DATASET_CONFIG"
echo "Classes: $CLASS1, $CLASS2"
echo "Modalities: ${MODALITIES[*]}"
echo ""

# 10 models: "HF_model_id:arch:slurm_label"  (slurm_label = squeue NAME / log prefix)
#   SLURM_JOB_LABEL is set so jobs are not all named clip-vit / CLIP-ViT.
declare -a MODEL_ENTRIES=(
  "apple/DFN5B-CLIP-ViT-H-14:clip:apple-dfn"
  "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224:clip:biomedclip"
  "google/siglip-base-patch16-224:clip:google-siglip"
  "laion/CLIP-ViT-H-14-laion2B-s32B-b79K:clip:laion-huge"
  "microsoft/llava-med-v1.5-mistral-7b:llava_med:llava-med"
  "facebook/metaclip-2-worldwide-b16-384:clip:meta-clip"
  "openai/clip-vit-large-patch14-336:clip:openai-336"
  "openai/clip-vit-base-patch32:clip:openai-base"
  "openai/clip-vit-large-patch14:clip:openai-large"
  "flaviagiammarino/pubmed-clip-vit-base-patch32:clip:pubmed-clip"
)

i=0
for entry in "${MODEL_ENTRIES[@]}"; do
  i=$((i + 1))
  MODEL_NAME="${entry%%:*}"
  REST="${entry#"${MODEL_NAME}":}"
  MODEL_ARCH="${REST%%:*}"
  SLURM_LABEL="${REST#*:}"
  echo "------------------------------------------"
  echo "[$i/10] $SLURM_LABEL — $MODEL_NAME ($MODEL_ARCH)"
  echo "------------------------------------------"
  SLURM_JOB_LABEL="$SLURM_LABEL" ./submit_single_model.sh \
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
