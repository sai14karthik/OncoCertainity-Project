#!/bin/bash
# Submit all 10 models (top 5 + next 5) for data2/TCGA-KIRP evaluation with ALL modalities (CT MR PT)
# Usage: ./submit_all10_models_data2.sh

echo "=========================================="
echo "Submitting ALL 10 models for data2/TCGA-KIRP"
echo "=========================================="
echo "Dataset: data2/TCGA-KIRP"
echo "Config: data2/tcga_kirp_config.yaml"
echo "Classes: early_stage, advanced_stage"
echo "Modalities: CT MR PT (all modalities)"
echo ""

# Fixed parameters for data2/TCGA-KIRP
DATA_ROOT="data2"
DATASET_CONFIG="data2/tcga_kirp_config.yaml"
CLASS1="early_stage"
CLASS2="advanced_stage"
MODALITIES=("CT" "MR" "PT")

echo "Step 1: Submitting top 5 models..."
echo ""
./submit_top5_models.sh "$DATA_ROOT" "$DATASET_CONFIG" "$CLASS1" "$CLASS2" "${MODALITIES[@]}"

echo ""
echo "Step 2: Submitting next 5 models..."
echo ""
./submit_5_new_models.sh "$DATA_ROOT" "$DATASET_CONFIG" "$CLASS1" "$CLASS2" "${MODALITIES[@]}"

echo ""
echo "=========================================="
echo "All 10 models submitted!"
echo "=========================================="
echo "Monitor: squeue -u \$USER"
echo "Check logs: ls -lh output_*.log error_*.log"
