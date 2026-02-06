#!/bin/bash
# Sync data2/TCGA-KIRP to Newton.
# Usage: 
#   ./sync_data2_to_newton.sh --all-at-once   # Sync entire folder at once (like data folder) - RECOMMENDED
#   ./sync_data2_to_newton.sh [CT|MR|PT] [early_stage|advanced_stage]  # Chunked sync
# Use SINGLE_SSH=1 to open one SSH connection (password once): SINGLE_SSH=1 ./sync_data2_to_newton.sh --all-at-once
# Tip: Use SSH keys to avoid password. Uses -z compression (like data folder sync). Re-run to resume.

set -e
# Run from project root (directory containing data2/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

REMOTE="sa808371@newton.ist.ucf.edu:~/Multi-Modal-AI/data2"
REMOTE_HOST="${REMOTE%%:*}"
SSH_SOCKET="${TMPDIR:-/tmp}/ssh_newton_$$"
MODALITIES=("CT" "MR" "PT")
CLASSES=("early_stage" "advanced_stage")

# Optional: one persistent SSH connection (password once for all chunks)
if [ -n "$SINGLE_SSH" ] && [ "$SINGLE_SSH" = "1" ]; then
  echo "Opening single SSH connection (you will enter password once)..."
  # Create master connection with ControlMaster
  ssh -M -S "$SSH_SOCKET" -o ControlPersist=600 -f -N "$REMOTE_HOST" || {
    echo "Warning: Failed to establish master SSH connection. Continuing with regular SSH."
    RSYNC_SSH="ssh"
  }
  if [ -S "$SSH_SOCKET" ]; then
    RSYNC_SSH="ssh -S $SSH_SOCKET -o ControlPath=$SSH_SOCKET"
    trap 'ssh -S "$SSH_SOCKET" -O exit "$REMOTE_HOST" 2>/dev/null || true' EXIT
  else
    RSYNC_SSH="ssh"
  fi
else
  RSYNC_SSH="ssh"
fi

# Optional: sync entire folder at once (like data folder) instead of chunking
if [ "$1" = "--all-at-once" ] || [ "$1" = "-a" ]; then
  echo "=== Syncing entire TCGA-KIRP folder at once (like data folder) ==="
  if ! rsync -avz --progress \
    --partial \
    --timeout=300 \
    --contimeout=60 \
    -e "$RSYNC_SSH" \
    "data2/TCGA-KIRP/" "$REMOTE/TCGA-KIRP/"; then
    echo "Sync failed. Re-run to retry (rsync will skip already-transferred files)."
    exit 1
  fi
  echo "TCGA-KIRP sync complete."
  exit 0
fi

# Optional: filter by modality and/or class
MOD_FILTER=""
CLASS_FILTER=""
if [ -n "$1" ] && [[ " ${MODALITIES[*]} " =~ " $1 " ]]; then
  MOD_FILTER="$1"
fi
if [ -n "$2" ] && [[ " ${CLASSES[*]} " =~ " $2 " ]]; then
  CLASS_FILTER="$2"
fi
if [ -n "$1" ] && [ -z "$MOD_FILTER" ]; then
  echo "Usage: $0 [--all-at-once|-a] | [CT|MR|PT] [early_stage|advanced_stage]"
  echo "  --all-at-once or -a: sync entire TCGA-KIRP folder at once (like data folder)"
  echo "  No args: sync all chunks. One arg: that modality. Two args: one chunk (~2–3k files)."
  exit 1
fi

# Ensure remote directory structure exists
if [ -n "$SINGLE_SSH" ] && [ "$SINGLE_SSH" = "1" ] && [ -S "$SSH_SOCKET" ]; then
  echo "Ensuring remote directory structure exists..."
  ssh -S "$SSH_SOCKET" "$REMOTE_HOST" "mkdir -p ~/Multi-Modal-AI/data2/TCGA-KIRP" 2>/dev/null || true
fi

for mod in "${MODALITIES[@]}"; do
  [ -n "$MOD_FILTER" ] && [ "$mod" != "$MOD_FILTER" ] && continue
  [ ! -d "data2/TCGA-KIRP/$mod" ] && continue
  for cls in "${CLASSES[@]}"; do
    [ -n "$CLASS_FILTER" ] && [ "$cls" != "$CLASS_FILTER" ] && continue
    [ ! -d "data2/TCGA-KIRP/$mod/$cls" ] && continue
    echo "=== Syncing TCGA-KIRP/$mod/$cls ==="
    # Use -z (compression) like sync_to_newton.sh does for data folder
    # -a flag automatically creates directories on the remote
    if ! rsync -avz --progress \
      --partial \
      --timeout=300 \
      --contimeout=60 \
      -e "$RSYNC_SSH" \
      "data2/TCGA-KIRP/$mod/$cls/" "$REMOTE/TCGA-KIRP/$mod/$cls/"; then
      echo "  -> Chunk failed; re-run script to retry (completed chunks are skipped)."
    fi
    echo ""
  done
done

echo "TCGA-KIRP sync complete."
