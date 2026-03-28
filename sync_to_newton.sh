set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REMOTE="${REMOTE:-sa808371@newton.ist.ucf.edu:~/Multi-Modal-AI/}"

echo "Syncing code to Newton..."
echo "Remote: $REMOTE"
echo ""

rsync -avz --progress \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'venv/' \
  --exclude '.DS_Store' \
  --exclude '.env' \
  --exclude '.env.local' \
  --exclude 'results/' \
  --exclude 'result/' \
  --exclude 'test_results/' \
  --exclude '*.log' \
  --exclude '.pytest_cache/' \
  --exclude '.cache/' \
  --exclude 'pipeline_metadata.csv' \
  --exclude 'Lung-PET-CT-Dx/' \
  --exclude 'Lung-PET-CT-Dx_PNG/' \
  --exclude 'Lung-PET-CT-Dx-PNG/' \
  --exclude 'Lung-PET-CT-Dx-PNG-run/' \
  --exclude 'Lung-PET-CT-Dx-PNG-test/' \
  --exclude 'manifest-*/' \
  --exclude 'data/Lung-PET-CT-Dx/' \
  --exclude 'data2/TCGA-KIRP/' \
  ./ \
  "$REMOTE"

echo ""
echo "Code sync complete!"
echo ""
