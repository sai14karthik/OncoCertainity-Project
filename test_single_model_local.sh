#!/usr/bin/env bash
# Test a single model locally on sample data (both forward and reverse orders)
#
# --- Lung PNG (Lung-PET-CT-Dx-PNG) shortcut ---
#   ./test_single_model_local.sh lung <model_name> <model_arch> [--max_samples N] [CT PT]
#   Optional env overrides (same as submit_all10_models_data1.sh):
#     DATA_ROOT=/path/to/Lung-PET-CT-Dx-PNG DATASET_CONFIG=configs/lung_pet_ct_dx.yaml ...
#
# --- Full form (any dataset) ---
#   ./test_single_model_local.sh <model_name> <model_arch> <data_root> <dataset_config> <class1> <class2> [--max_samples N] [modality1] [modality2] ...
#
# Examples:
#   ./test_single_model_local.sh lung openai/clip-vit-base-patch32 clip
#   ./test_single_model_local.sh lung openai/clip-vit-base-patch32 clip --max_samples 20 CT PT
#   ./test_single_model_local.sh openai/clip-vit-base-patch32 clip data data/dataset_config.yaml high_grade low_grade --max_samples 10 CT PET
#   ./test_single_model_local.sh openai/clip-vit-base-patch32 clip data2 data2/tcga_kirp_config.yaml early_stage advanced_stage CT MR

set -euo pipefail

# Always run from repo root so relative paths (data_root, dataset_config, src.main) work
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage_full() {
    echo "Usage: $0 <model_name> <model_arch> <data_root> <dataset_config> <class1> <class2> [--max_samples N] [modality1] [modality2] ..."
    echo ""
    echo "Lung-PET-CT-Dx-PNG shortcut:"
    echo "  $0 lung <model_name> <model_arch> [--max_samples N] [CT PT]"
    echo "  Uses DATA_ROOT=\${DATA_ROOT:-Lung-PET-CT-Dx-PNG}, configs/lung_pet_ct_dx.yaml,"
    echo "  classes non_small_cell vs small_cell (override DATA_ROOT / DATASET_CONFIG if needed)."
    echo ""
    echo "Arguments (full form):"
    echo "  model_name:     HuggingFace model name (e.g., openai/clip-vit-base-patch32)"
    echo "  model_arch:     Model architecture (clip, llava, llava_med)"
    echo "  data_root:      Root directory for data (relative to this repo or absolute)"
    echo "  dataset_config: Path to dataset config YAML"
    echo "  class1, class2: Class folder names in the config"
    echo "  --max_samples:  (Optional) Max images per patient per modality (default: 10)"
    echo "  modalities:     Two or more modality names (config must define them; lung uses CT PT)"
}

# Optional preset: lung | png | Lung-PET-CT-Dx-PNG
LUNG_PRESET=0
if [[ "${1:-}" == "lung" || "${1:-}" == "png" || "${1:-}" == "Lung-PET-CT-Dx-PNG" ]]; then
    LUNG_PRESET=1
    shift
    if [ $# -lt 2 ]; then
        usage_full
        exit 1
    fi
    MODEL_NAME="$1"
    MODEL_ARCH="$2"
    shift 2
    DATA_ROOT="${DATA_ROOT:-Lung-PET-CT-Dx-PNG}"
    DATASET_CONFIG="${DATASET_CONFIG:-configs/lung_pet_ct_dx.yaml}"
    CLASS1="non_small_cell"
    CLASS2="small_cell"
else
    if [ $# -lt 6 ]; then
        usage_full
        echo ""
        echo "Examples:"
        echo "  $0 lung openai/clip-vit-base-patch32 clip"
        echo "  $0 openai/clip-vit-base-patch32 clip data data/dataset_config.yaml high_grade low_grade --max_samples 10 CT PET"
        exit 1
    fi
    MODEL_NAME="$1"
    MODEL_ARCH="$2"
    DATA_ROOT="$3"
    DATASET_CONFIG="$4"
    CLASS1="$5"
    CLASS2="$6"
    shift 6
fi

# Parse remaining args: optional --max_samples anywhere; rest are modalities
MAX_SAMPLES="10"
MODALITIES=()
while [ $# -gt 0 ]; do
    if [ "$1" = "--max_samples" ]; then
        if [ -z "${2:-}" ] || ! [[ "$2" =~ ^[0-9]+$ ]]; then
            echo "Error: --max_samples requires a positive integer"
            exit 1
        fi
        MAX_SAMPLES="$2"
        shift 2
    else
        MODALITIES+=("$1")
        shift
    fi
done

# Lung preset: default modalities CT + PT (folder names in configs/lung_pet_ct_dx.yaml)
if [[ "$LUNG_PRESET" -eq 1 ]] && [ ${#MODALITIES[@]} -eq 0 ]; then
    MODALITIES=("CT" "PT")
fi

if [ ${#MODALITIES[@]} -lt 2 ]; then
    echo "Error: Please provide at least 2 modalities (or use the lung preset without extra modality args to default to CT PT)"
    exit 1
fi

MOD_SUFFIX_FORWARD=$(IFS='_'; echo "${MODALITIES[*]}")

# Build reversed modalities array for display and execution
REVERSED_MODALITIES=()
for ((i=${#MODALITIES[@]}-1; i>=0; i--)); do
    REVERSED_MODALITIES+=("${MODALITIES[i]}")
done
MOD_SUFFIX_REVERSE=$(IFS='_'; echo "${REVERSED_MODALITIES[*]}")

echo "=========================================="
echo "LOCAL TEST - Single Model"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Architecture: $MODEL_ARCH"
echo "Data root: $DATA_ROOT"
echo "Dataset config: $DATASET_CONFIG"
echo "Classes: $CLASS1, $CLASS2"
echo "Max samples per patient per modality: $MAX_SAMPLES"
echo "Modalities: ${MODALITIES[*]}"
echo "Forward order: ${MOD_SUFFIX_FORWARD}"
echo "Reverse order: ${MOD_SUFFIX_REVERSE}"
echo ""

# Check if data directory exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Data root directory not found: $DATA_ROOT"
    echo "  (Resolved from repo root: $SCRIPT_DIR)"
    echo ""
    echo "  Restore or recreate Lung-PET-CT-Dx-PNG (PNG dataset), or set DATA_ROOT to where it lives, e.g.:"
    echo "    DATA_ROOT=/path/to/Lung-PET-CT-Dx-PNG $0 lung <model> <arch>"
    echo ""
    echo "  For TCGA-KIRP you typically need:"
    echo "    $SCRIPT_DIR/data2/TCGA-KIRP/   (large; not in git) — see data2/README.md"
    echo ""
    echo "  Full form with any dataset:"
    echo "    $0 <model> <arch> /path/to/your_dataset your_config.yaml class1 class2 CT MR"
    exit 1
fi

# Check if dataset config exists
if [ ! -f "$DATASET_CONFIG" ]; then
    echo "Warning: Dataset config not found: $DATASET_CONFIG"
    echo "  Python may error if a config is required for your dataset layout."
fi

MAX_SAMPLES_ARG=(--max_samples "${MAX_SAMPLES}")

echo "=========================================="
echo "Running BOTH orders in one command: ${MOD_SUFFIX_FORWARD} and ${MOD_SUFFIX_REVERSE}"
echo "=========================================="
python3 -u -m src.main \
    --data_root "${DATA_ROOT}" \
    --modalities "${MODALITIES[@]}" \
    --run_both_orders \
    --model_arch "${MODEL_ARCH}" \
    --model_name "${MODEL_NAME}" \
    --output_dir results \
    --batch_size 4 \
    --dataset_config "${DATASET_CONFIG}" \
    --class_names "${CLASS1}" "${CLASS2}" \
    --temperature 0.8 \
    --no_progress \
    "${MAX_SAMPLES_ARG[@]}"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
ORDER_FWD=$(IFS='→'; echo "${MODALITIES[*]}")
ORDER_REV=$(IFS='→'; echo "${REVERSED_MODALITIES[*]}")
echo "Both orders (${ORDER_FWD} and ${ORDER_REV}): Exit code $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Both orders completed successfully!"
    echo ""
    echo "Results saved in: results/"
    MODEL_NAME_SAFE=$(echo "$MODEL_NAME" | sed 's/\//_/g')
    echo "Check: ls -lh results/*${MODEL_NAME_SAFE}*${MOD_SUFFIX_FORWARD}*.json"
    echo "Check: ls -lh results/*${MODEL_NAME_SAFE}*${MOD_SUFFIX_REVERSE}*.json"
    exit 0
else
    echo "One or both orders failed"
    exit 1
fi
