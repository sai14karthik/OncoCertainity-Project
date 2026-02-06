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
  test_single_model_local.sh \
  test_single_model_newton.sh \
  sync_to_newton.sh \
  sync_data2_to_newton.sh \
  third_party \
  sa808371@newton.ist.ucf.edu:~/Multi-Modal-AI/

# Sync optional verification script if it exists
if [ -f verify_data2_sync.sh ]; then
  rsync -avz --progress verify_data2_sync.sh sa808371@newton.ist.ucf.edu:~/Multi-Modal-AI/
fi

# Sync data2 config and metadata (exclude TCGA-KIRP images)
if [ -d data2 ]; then
  rsync -avz --progress \
    --exclude 'TCGA-KIRP' \
    data2/ \
    sa808371@newton.ist.ucf.edu:~/Multi-Modal-AI/data2/
fi

echo ""
echo "Code sync complete!"
echo ""
echo "To sync TCGA-KIRP images, run:"
echo "  ./sync_data2_to_newton.sh --all-at-once   # RECOMMENDED: sync entire folder (like data folder)"
echo "  ./sync_data2_to_newton.sh                 # chunked: all chunks"
echo "  ./sync_data2_to_newton.sh CT              # chunked: CT only"
echo "  ./sync_data2_to_newton.sh CT early_stage  # chunked: single chunk"
