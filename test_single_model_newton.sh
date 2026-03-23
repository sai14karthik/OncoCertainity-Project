
if [ $# -lt 3 ]; then
    echo "Usage: $0 <model_name> <model_arch> [modality1] [modality2] ..."
    echo ""
    echo "Arguments:"
    echo "  model_name:     HuggingFace model name (e.g., microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)"
    echo "  model_arch:     Model architecture (clip, llava, llava_med)"
    echo "  modalities:     Two or more modality names (e.g., CT MR or CT MR PT)"
    echo ""
    echo "Examples:"
    echo "  $0 microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 clip CT MR"
    echo "  $0 openai/clip-vit-base-patch32 clip CT MR PT"
    echo "  $0 microsoft/llava-med-v1.5-mistral-7b llava_med CT MR"
    echo ""
    echo "This will run on FULL dataset (no --max_samples limit)"
    echo "Data: data2/TCGA-KIRP"
    echo "Config: data2/tcga_kirp_config.yaml"
    echo "Classes: early_stage, advanced_stage"
    exit 1
fi

MODEL_NAME="$1"
MODEL_ARCH="$2"
shift 2

# Get modalities from remaining arguments
if [ $# -lt 2 ]; then
    echo "Error: Please provide at least 2 modalities (e.g., CT MR)"
    exit 1
fi

MODALITIES=("$@")

# Fixed parameters for data2/TCGA-KIRP
DATA_ROOT="data2"
DATASET_CONFIG="data2/tcga_kirp_config.yaml"
CLASS1="early_stage"
CLASS2="advanced_stage"

echo "=========================================="
echo "NEWTON TEST - Single Model (Full Dataset)"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Architecture: $MODEL_ARCH"
echo "Data root: $DATA_ROOT"
echo "Dataset config: $DATASET_CONFIG"
echo "Classes: $CLASS1, $CLASS2"
echo "Modalities: ${MODALITIES[*]}"
echo "Full dataset (no --max_samples limit)"
echo ""

# Call submit_single_model.sh with full dataset (no --max_samples)
./submit_single_model.sh \
    "$MODEL_NAME" \
    "$MODEL_ARCH" \
    "$DATA_ROOT" \
    "$DATASET_CONFIG" \
    "$CLASS1" \
    "$CLASS2" \
    "${MODALITIES[@]}"
