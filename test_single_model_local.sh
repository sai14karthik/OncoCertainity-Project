#!/bin/bash
# Test a single model locally on sample data (both forward and reverse orders)
# Usage: ./test_single_model_local.sh <model_name> <model_arch> <data_root> <dataset_config> <class1> <class2> [--max_samples N] [modality1] [modality2] ...
# Example: ./test_single_model_local.sh openai/clip-vit-base-patch32 clip data data/dataset_config.yaml high_grade low_grade --max_samples 10 CT PET
#          ./test_single_model_local.sh openai/clip-vit-base-patch32 clip data2 data2/tcga_kirp_config.yaml early_stage advanced_stage --max_samples 10 CT MR

# Parse arguments
if [ $# -lt 6 ]; then
    echo "Usage: $0 <model_name> <model_arch> <data_root> <dataset_config> <class1> <class2> [--max_samples N] [modality1] [modality2] ..."
    echo ""
    echo "Arguments:"
    echo "  model_name:     HuggingFace model name (e.g., openai/clip-vit-base-patch32)"
    echo "  model_arch:     Model architecture (clip, llava, llava_med)"
    echo "  data_root:      Root directory for data (e.g., data or .)"
    echo "  dataset_config: Path to dataset config YAML (e.g., data/dataset_config.yaml)"
    echo "  class1:         First class name (e.g., high_grade or class0)"
    echo "  class2:         Second class name (e.g., low_grade or class1)"
    echo "  --max_samples:  (Optional) Maximum number of images per patient per modality (default: 10 for local testing)"
    echo "  modalities:     Two or more modality names (e.g., CT PET or CT XA MR)"
    echo ""
    echo "Examples:"
    echo "  $0 openai/clip-vit-base-patch32 clip data data/dataset_config.yaml high_grade low_grade --max_samples 10 CT PET"
    echo "  $0 microsoft/llava-med-v1.5-mistral-7b llava_med data data/dataset_config.yaml class0 class1 --max_samples 5 CT PET"
    exit 1
fi

MODEL_NAME="$1"
MODEL_ARCH="$2"
DATA_ROOT="$3"
DATASET_CONFIG="$4"
CLASS1="$5"
CLASS2="$6"
shift 6

# Parse optional --max_samples flag
MAX_SAMPLES="10"  # Default to 10 for local testing
if [ "$1" == "--max_samples" ]; then
    if [ -z "$2" ] || ! [[ "$2" =~ ^[0-9]+$ ]]; then
        echo "Error: --max_samples requires a number"
        exit 1
    fi
    MAX_SAMPLES="$2"
    shift 2
fi

# Get modalities from remaining arguments
if [ $# -lt 2 ]; then
    echo "Error: Please provide at least 2 modalities"
    exit 1
fi

MODALITIES=("$@")

MOD1="${MODALITIES[0]}"
MOD2="${MODALITIES[1]}"
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
    echo "   Please create the directory or specify a different path"
    exit 1
fi

# Check if dataset config exists
if [ ! -f "$DATASET_CONFIG" ]; then
    echo "Warning: Dataset config not found: $DATASET_CONFIG"
    echo "   Will use default config"
fi

# Build max_samples argument
MAX_SAMPLES_ARG="--max_samples ${MAX_SAMPLES}"

echo "=========================================="
echo "Running BOTH orders in one command: ${MOD_SUFFIX_FORWARD} and ${MOD_SUFFIX_REVERSE}"
echo "=========================================="
python3 -u -m src.main \
    --data_root "${DATA_ROOT}" \
    --modalities ${MODALITIES[*]} \
    --run_both_orders \
    --model_arch "${MODEL_ARCH}" \
    --model_name "${MODEL_NAME}" \
    --output_dir results \
    --batch_size 4 \
    --dataset_config "${DATASET_CONFIG}" \
    --class_names "${CLASS1}" "${CLASS2}" \
    --temperature 0.8 \
    --no_progress \
    ${MAX_SAMPLES_ARG}

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
