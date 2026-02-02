#!/bin/bash
# Submit top 5 best models for medical imaging evaluation
# Usage: ./submit_top5_models.sh [DATA_ROOT] [DATASET_CONFIG] [CLASS1] [CLASS2] [MODALITY1] [MODALITY2] ...
# Example: ./submit_top5_models.sh data data/dataset_config.yaml high_grade low_grade CT PET
#          ./submit_top5_models.sh . data/cmb_aml_config.yaml class0 class1 CT XA MR

# Parse arguments
if [ $# -lt 5 ]; then
    echo "Usage: $0 <data_root> <dataset_config> <class1> <class2> [modality1] [modality2] ..."
    echo "Example: $0 data data/dataset_config.yaml high_grade low_grade CT PET"
    echo "Example: $0 . data/cmb_aml_config.yaml class0 class1 CT XA"
    exit 1
fi

DATA_ROOT="$1"
DATASET_CONFIG="$2"
CLASS1="$3"
CLASS2="$4"
shift 4

# Get modalities from remaining arguments
if [ $# -ge 2 ]; then
    MODALITIES=("$@")
else
    echo "Error: Please provide at least 2 modalities"
    exit 1
fi

MOD1="${MODALITIES[0]}"
MOD2="${MODALITIES[1]}"
MOD_SUFFIX_FORWARD=$(IFS='_'; echo "${MODALITIES[*]}")

# Build reversed modalities array for display
REVERSED_MODALITIES_DISPLAY=()
for ((i=${#MODALITIES[@]}-1; i>=0; i--)); do
    REVERSED_MODALITIES_DISPLAY+=("${MODALITIES[i]}")
done
MOD_SUFFIX_REVERSE=$(IFS='_'; echo "${REVERSED_MODALITIES_DISPLAY[*]}")

echo "Submitting top 5 models..."
echo "Modalities: ${MODALITIES[*]}"
echo "Forward order: ${MOD_SUFFIX_FORWARD}"
echo "Reverse order: ${MOD_SUFFIX_REVERSE}"
echo ""

# Top 5 models (best to good)
MODELS=(
    "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224:clip:biomedclip"
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K:clip:laion-huge"
    "openai/clip-vit-large-patch14:clip:openai-large"
    "openai/clip-vit-large-patch14-336:clip:openai-336"
    "microsoft/llava-med-v1.5-mistral-7b:llava_med:llava-med"
)

# Build reversed modalities array
REVERSED_MODALITIES=()
for ((i=${#MODALITIES[@]}-1; i>=0; i--)); do
    REVERSED_MODALITIES+=("${MODALITIES[i]}")
done

JOB_IDS=()

for model_config in "${MODELS[@]}"; do
    IFS=':' read -r model_name model_arch job_name <<< "$model_config"
    
    echo "Submitting: $job_name"
    
    # Submit single job with both orders
    JOB_SCRIPT="submit_${job_name}_both_orders_tmp.sh"
    
    # Check if this is LLaVA-Med and needs dependency installation
    if [ "${model_arch}" == "llava_med" ]; then
        cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=output_${job_name}_%j.log
#SBATCH --error=error_${job_name}_%j.log

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
python3 -u -m src.main --data_root "${DATA_ROOT}" --modalities ${MODALITIES[*]} --run_both_orders --model_arch ${model_arch} --model_name ${model_name} --output_dir results --batch_size 8 --dataset_config "${DATASET_CONFIG}" --class_names "${CLASS1}" "${CLASS2}" --temperature 0.8 --aggressive_preprocess

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
    echo "One or both orders failed"
    exit 1
fi
EOF
    else
        cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=output_${job_name}_%j.log
#SBATCH --error=error_${job_name}_%j.log

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd ~/Multi-Modal-AI
source venv/bin/activate

echo "=========================================="
echo "Running BOTH orders in one command: ${MOD_SUFFIX_FORWARD} and ${MOD_SUFFIX_REVERSE}"
echo "=========================================="
python3 -u -m src.main --data_root "${DATA_ROOT}" --modalities ${MODALITIES[*]} --run_both_orders --model_arch ${model_arch} --model_name ${model_name} --output_dir results --batch_size 8 --dataset_config "${DATASET_CONFIG}" --class_names "${CLASS1}" "${CLASS2}" --temperature 0.8 --aggressive_preprocess

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
    echo "One or both orders failed"
    exit 1
fi
EOF
    fi
    
    chmod +x "$JOB_SCRIPT"
    JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
    rm -f "$JOB_SCRIPT"
    JOB_IDS+=("$JOB_ID")
    echo "    Job $JOB_ID submitted"
    echo ""
done

echo "Submitted ${#JOB_IDS[@]} jobs (${#MODELS[@]} models)"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs: tail -f output_*_*.log"
