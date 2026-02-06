#!/bin/bash
# Sync code changes to Newton cluster
# Usage: ./sync_to_newton.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Syncing code to Newton..."
echo ""

# Sync source code and config files (excluding large data and results)
rsync -avz --progress \
  --exclude 'data/Lung-PET-CT-Dx' \
  --exclude 'data2/TCGA-KIRP' \
  --exclude '__pycache__' \
  --exclude '.git' \
  --exclude '*.pyc' \
  --exclude 'venv' \
  --exclude '.DS_Store' \
  --exclude 'results/' \
  --exclude 'result/' \
  --exclude 'test_results/' \
  --exclude '*.log' \
  --exclude '.pytest_cache/' \
  --exclude '.cache/' \
  --exclude '.env' \
  --exclude '.env.local' \
  src \
  scripts \
  requirements.txt \
  README.md \
  data/dataset_config.yaml \
  .env.example \
  submit_top5_models.sh \
  submit_5_new_models.sh \
  submit_single_model.sh \
  submit_all10_models_data2.sh \
  test_single_model_local.sh \
  test_single_model_newton.sh \
  run_llava_med_data2.sh \
  cleanup.sh \
  sync_to_newton.sh \
  sync_data2_to_newton.sh \
  third_party \
  sa808371@newton.ist.ucf.edu:~/Multi-Modal-AI/



echo ""
echo "Code sync complete!"
echo ""
