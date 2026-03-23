#!/bin/bash
# Submit a single model for medical imaging evaluation (both forward and reverse orders)
# Usage: ./submit_single_model.sh <model_name> <model_arch> <data_root> <dataset_config> <class1> <class2> [--max_samples N] [modality1] [modality2] ...
# Optional env: SLURM_JOB_LABEL="openai-base"  -> short job name in squeue and output_openai-base_%j.log
#
# Example: ./submit_single_model.sh microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 clip data data/dataset_config.yaml high_grade low_grade CT PET
#          ./submit_single_model.sh openai/clip-vit-base-patch32 clip . data/cmb_aml_config.yaml class0 class1 --max_samples 100 CT XA MR

# Parse arguments
if [ $# -lt 6 ]; then
    echo "Usage: $0 <model_name> <model_arch> <data_root> <dataset_config> <class1> <class2> [--max_samples N] [modality1] [modality2] ..."
    echo ""
    echo "Arguments:"
    echo "  model_name:     HuggingFace model name (e.g., microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)"
    echo "  model_arch:     Model architecture (clip, llava, llava_med)"
    echo "  data_root:      Root directory for data (e.g., data or .)"
    echo "  dataset_config: Path to dataset config YAML (e.g., data/dataset_config.yaml)"
    echo "  class1:         First class name (e.g., high_grade or class0)"
    echo "  class2:         Second class name (e.g., low_grade or class1)"
    echo "  --max_samples:  (Optional) Maximum number of images per patient per modality"
    echo "  modalities:     Two or more modality names (e.g., CT PET or CT XA MR)"
    echo ""
    echo "Examples:"
    echo "  $0 microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 clip data data/dataset_config.yaml high_grade low_grade CT PET"
    echo "  $0 openai/clip-vit-base-patch32 clip . data/cmb_aml_config.yaml class0 class1 --max_samples 100 CT XA"
    echo "  $0 microsoft/llava-med-v1.5-mistral-7b llava_med data data/dataset_config.yaml high_grade low_grade --max_samples 50 CT PET"
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
MAX_SAMPLES=""
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

# SLURM job name (squeue / log file prefix). Prefer a short unique label when set.
# Usage: SLURM_JOB_LABEL=openai-base ./submit_single_model.sh ...
# Allowed: letters, digits, hyphens, underscores (max 60 chars for SBATCH).
if [[ -n "${SLURM_JOB_LABEL:-}" ]]; then
  JOB_NAME=$(echo "$SLURM_JOB_LABEL" | tr ' ' '_' | sed 's/[^a-zA-Z0-9_-]/_/g' | sed 's/__*/_/g' | sed 's/^_\|_$//g' | cut -c1-60)
else
  JOB_NAME=$(echo "$MODEL_NAME" | sed 's/.*\///' | sed 's/[^a-zA-Z0-9]/-/g' | sed 's/--*/-/g' | sed 's/^-\|-$//g' | cut -c1-40)
fi
if [[ -z "$JOB_NAME" ]]; then
  JOB_NAME="model_job"
fi

MOD1="${MODALITIES[0]}"
MOD2="${MODALITIES[1]}"
MOD_SUFFIX_FORWARD=$(IFS='_'; echo "${MODALITIES[*]}")

# Build reversed modalities array for display and execution
REVERSED_MODALITIES=()
for ((i=${#MODALITIES[@]}-1; i>=0; i--)); do
    REVERSED_MODALITIES+=("${MODALITIES[i]}")
done
MOD_SUFFIX_REVERSE=$(IFS='_'; echo "${REVERSED_MODALITIES[*]}")

echo "Submitting single model: $MODEL_NAME"
echo "Architecture: $MODEL_ARCH"
if [[ -n "${SLURM_JOB_LABEL:-}" ]]; then
  echo "SLURM job label: $SLURM_JOB_LABEL -> job name: $JOB_NAME"
fi
echo "Data root: $DATA_ROOT"
echo "Dataset config: $DATASET_CONFIG"
echo "Classes: $CLASS1, $CLASS2"
if [ -n "$MAX_SAMPLES" ]; then
    echo "Max samples per patient per modality: $MAX_SAMPLES"
fi
echo "Modalities: ${MODALITIES[*]}"
echo "Forward order: ${MOD_SUFFIX_FORWARD}"
echo "Reverse order: ${MOD_SUFFIX_REVERSE}"
echo ""

# Build max_samples argument if provided
MAX_SAMPLES_ARG=""
if [ -n "$MAX_SAMPLES" ]; then
    MAX_SAMPLES_ARG="--max_samples ${MAX_SAMPLES}"
fi

# REVERSED_MODALITIES array is already built above

# Submit single job with both orders
echo "Submitting job with both orders (forward and reverse)..."
JOB_SCRIPT="submit_${JOB_NAME}_both_orders_tmp.sh"

# Check if this is LLaVA-Med and needs dependency installation
if [ "${MODEL_ARCH}" == "llava_med" ]; then
    cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=output_${JOB_NAME}_%j.log
#SBATCH --error=error_${JOB_NAME}_%j.log

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd ~/Multi-Modal-AI
source venv/bin/activate

# Install missing dependencies for LLaVA-Med
echo "Installing LLaVA-Med dependencies..."
pip install "transformers>=4.30.0,<4.38.0" accelerate>=0.20.0 "tiktoken>=0.5.0,<0.8.0" protobuf>=4.21.0 sentencepiece>=0.1.99
echo "Dependencies installed successfully!"

# Clear corrupted tokenizer cache
echo "Clearing corrupted tokenizer cache..."
rm -rf ~/.cache/huggingface/hub/models--microsoft--llava-med-v1.5-mistral-7b/snapshots/*/tokenizer.model 2>/dev/null || true

# Verify installation
python3 -c "import tiktoken; import google.protobuf; import sentencepiece; print('✓ Dependencies verified')" || {
    echo "✗ Dependency verification failed!"
    exit 1
}

echo "=========================================="
echo "Running BOTH orders in one command: ${MOD_SUFFIX_FORWARD} and ${MOD_SUFFIX_REVERSE}"
echo "=========================================="
python3 -u -m src.main --data_root "${DATA_ROOT}" --modalities ${MODALITIES[*]} --run_both_orders --model_arch ${MODEL_ARCH} --model_name ${MODEL_NAME} --output_dir results --batch_size 8 --dataset_config "${DATASET_CONFIG}" --class_names "${CLASS1}" "${CLASS2}" --temperature 0.8 --aggressive_preprocess ${MAX_SAMPLES_ARG}

EXIT_CODE=\$?

echo ""
echo "=========================================="
echo "Job Summary"
echo "=========================================="
echo "Both orders (${MOD1}→${MOD2} and ${MOD2}→${MOD1}): Exit code \$EXIT_CODE"

if [ \$EXIT_CODE -eq 0 ]; then
    echo "Both orders completed successfully!"
    exit 0
else
    echo "Run failed"
    exit 1
fi
EOF
else
        cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=output_${JOB_NAME}_%j.log
#SBATCH --error=error_${JOB_NAME}_%j.log

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd ~/Multi-Modal-AI
source venv/bin/activate

echo "=========================================="
echo "Running BOTH orders in one command: ${MOD_SUFFIX_FORWARD} and ${MOD_SUFFIX_REVERSE}"
echo "=========================================="
python3 -u -m src.main --data_root "${DATA_ROOT}" --modalities ${MODALITIES[*]} --run_both_orders --model_arch ${MODEL_ARCH} --model_name ${MODEL_NAME} --output_dir results --batch_size 8 --dataset_config "${DATASET_CONFIG}" --class_names "${CLASS1}" "${CLASS2}" --temperature 0.8 --aggressive_preprocess ${MAX_SAMPLES_ARG}

EXIT_CODE=\$?

echo ""
echo "=========================================="
echo "Job Summary"
echo "=========================================="
echo "Both orders (${MOD1}→${MOD2} and ${MOD2}→${MOD1}): Exit code \$EXIT_CODE"

if [ \$EXIT_CODE -eq 0 ]; then
    echo "Both orders completed successfully!"
    exit 0
else
    echo "Run failed"
    exit 1
fi
EOF
fi

chmod +x "$JOB_SCRIPT"
JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
rm -f "$JOB_SCRIPT"
echo "    Job $JOB_ID submitted (both orders)"
echo ""

echo "Job submitted with both orders:"
echo "   Job ID: $JOB_ID"
echo "   Forward (${MOD1}→${MOD2}) and Reverse (${MOD2}→${MOD1})"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs: tail -f output_${JOB_NAME}_${JOB_ID}.log"
