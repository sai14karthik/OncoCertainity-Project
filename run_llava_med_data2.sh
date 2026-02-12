#!/bin/bash
# Run LLaVA-Med on full data2/TCGA-KIRP dataset with CT, MR, and PT modalities
# Usage: ./run_llava_med_data2.sh [local|newton]
#   local: Run locally (for testing)
#   newton: Submit to SLURM on Newton cluster (default)

# Default to newton if not specified
MODE="${1:-newton}"

# Fixed parameters for data2/TCGA-KIRP
MODEL_NAME="microsoft/llava-med-v1.5-mistral-7b"
MODEL_ARCH="llava_med"
DATA_ROOT="data2"
DATASET_CONFIG="data2/tcga_kirp_config.yaml"
CLASS1="early_stage"
CLASS2="advanced_stage"
# Use all three available modalities for sequential evaluation
MODALITIES=("CT" "MR" "PT")

MOD1="${MODALITIES[0]}"
MOD2="${MODALITIES[1]}"
MOD_SUFFIX_FORWARD=$(IFS='_'; echo "${MODALITIES[*]}")

# Build reversed modalities array
REVERSED_MODALITIES=()
for ((i=${#MODALITIES[@]}-1; i>=0; i--)); do
    REVERSED_MODALITIES+=("${MODALITIES[i]}")
done
MOD_SUFFIX_REVERSE=$(IFS='_'; echo "${REVERSED_MODALITIES[*]}")

echo "=========================================="
echo "LLaVA-Med on data2/TCGA-KIRP (Full Dataset)"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Architecture: $MODEL_ARCH"
echo "Data root: $DATA_ROOT"
echo "Dataset config: $DATASET_CONFIG"
echo "Classes: $CLASS1, $CLASS2"
echo "Modalities: ${MODALITIES[*]}"
echo "Forward order: ${MOD_SUFFIX_FORWARD}"
echo "Reverse order: ${MOD_SUFFIX_REVERSE}"
echo "Mode: $MODE"
echo "Full dataset (no --max_samples limit)"
echo ""

if [ "$MODE" == "local" ]; then
    # Run locally
    echo "=========================================="
    echo "Running LOCALLY (both orders)"
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
        --llava_med_flip_predictions \
        --aggressive_preprocess \
        --no_progress
    
    EXIT_CODE=$?
    
    echo ""
    echo "=========================================="
    echo "Run Summary"
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
        echo "Run failed"
        exit 1
    fi

elif [ "$MODE" == "newton" ]; then
    # Submit to SLURM on Newton
    echo "=========================================="
    echo "Submitting to SLURM (Newton cluster)"
    echo "=========================================="
    
    JOB_NAME="llava-med-data2"
    JOB_SCRIPT="submit_llava_med_data2_tmp.sh"
    
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
echo "Running LLaVA-Med on data2/TCGA-KIRP"
echo "Running BOTH orders: ${MOD_SUFFIX_FORWARD} and ${MOD_SUFFIX_REVERSE}"
echo "=========================================="
python3 -u -m src.main --data_root "${DATA_ROOT}" --modalities ${MODALITIES[*]} --run_both_orders --model_arch ${MODEL_ARCH} --model_name ${MODEL_NAME} --output_dir results --batch_size 8 --dataset_config "${DATASET_CONFIG}" --class_names "${CLASS1}" "${CLASS2}" --temperature 0.8 --llava_med_flip_predictions --aggressive_preprocess

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

    chmod +x "$JOB_SCRIPT"
    JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
    rm -f "$JOB_SCRIPT"
    
    echo "Job submitted successfully!"
    echo "   Job ID: $JOB_ID"
    echo "   Job name: $JOB_NAME"
    echo "   Forward order: ${MOD1}→${MOD2} (${MOD_SUFFIX_FORWARD})"
    echo "   Reverse order: ${MOD2}→${MOD1} (${MOD_SUFFIX_REVERSE})"
    echo ""
    echo "Monitor: squeue -u \$USER"
    echo "Logs: tail -f output_${JOB_NAME}_${JOB_ID}.log"
    echo "Check logs: ls -lh output_${JOB_NAME}_*.log error_${JOB_NAME}_*.log"
    exit 0

else
    echo "Error: Invalid mode '$MODE'"
    echo "Usage: $0 [local|newton]"
    echo "  local:  Run locally (for testing)"
    echo "  newton: Submit to SLURM on Newton cluster (default)"
    exit 1
fi
