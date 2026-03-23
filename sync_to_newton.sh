

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Syncing code to Newton..."
echo ""

rsync -avz --progress \
  --exclude 'Lung-PET-CT-Dx-PNG/' \
  --exclude 'Lung-PET-CT-Dx-PNG-run/' \
  --exclude 'Lung-PET-CT-Dx-PNG-test/' \
  --exclude 'manifest-*/' \
  --exclude 'data/Lung-PET-CT-Dx/' \
  --exclude 'data2/TCGA-KIRP/' \
  --exclude '__pycache__/' \
  --exclude '.git/' \
  --exclude '*.pyc' \
  --exclude 'venv/' \
  --exclude '.DS_Store' \
  --exclude 'results/' \
  --exclude 'result/' \
  --exclude 'test_results/' \
  --exclude '*.log' \
  --exclude '.pytest_cache/' \
  --exclude '.cache/' \
  --exclude '.env' \
  --exclude '.env.local' \
  ./ sa808371@newton.ist.ucf.edu:~/Multi-Modal-AI/

echo ""
echo "Code sync complete!"
echo ""
