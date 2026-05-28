#!/bin/bash
# Continuous sync script - Run this in a separate terminal/tmux pane
# Automatically downloads checkpoints while training is running

# Configuration - UPDATE THESE!
VM_IP="YOUR_VM_IP"
VM_USER="YOUR_USERNAME"
LOCAL_BACKUP_DIR="./live_backup"
SYNC_INTERVAL=600  # Sync every 10 minutes

mkdir -p "$LOCAL_BACKUP_DIR"

echo "=========================================="
echo "Live Backup Monitor"
echo "Syncing every $SYNC_INTERVAL seconds"
echo "=========================================="

while true; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] Syncing from VM..."
    
    # Sync latest checkpoint (only newest file)
    rsync -avz --progress \
        "$VM_USER@$VM_IP:~/gpt-mini-training/checkpoints/" \
        "$LOCAL_BACKUP_DIR/checkpoints/" \
        --include="step_*.pth" \
        --exclude="*.tmp" \
        || echo "Warning: rsync failed, will retry..."
    
    # Sync progress report
    rsync -avz \
        "$VM_USER@$VM_IP:~/gpt-mini-training/training_progress.json" \
        "$LOCAL_BACKUP_DIR/" \
        2>/dev/null || true
    
    # Show current progress
    if [ -f "$LOCAL_BACKUP_DIR/training_progress.json" ]; then
        echo "Progress:"
        cat "$LOCAL_BACKUP_DIR/training_progress.json" | grep -E "last_step|progress_percentage|cost_estimate"
    fi
    
    echo "[$timestamp] Sync complete. Next sync in $SYNC_INTERVAL seconds..."
    echo "----------------------------------------"
    
    sleep $SYNC_INTERVAL
done
