#!/bin/bash
# Download trained model from AMD GPU Droplet to local machine
# Run this on your LOCAL machine (Windows with WSL or PowerShell)

# Configuration - UPDATE THESE!
VM_IP="YOUR_VM_IP"
VM_USER="YOUR_USERNAME"
LOCAL_DIR="./downloaded_models"

echo "=========================================="
echo "Model Download Script"
echo "=========================================="
echo "VM: $VM_USER@$VM_IP"
echo "Local directory: $LOCAL_DIR"
echo "=========================================="

# Create local directory
mkdir -p "$LOCAL_DIR"

# Download final model
echo "[1/4] Downloading final model..."
scp "$VM_USER@$VM_IP:~/gpt-mini-training/model.pth" "$LOCAL_DIR/"

# Download latest checkpoint
echo "[2/4] Downloading latest checkpoint..."
scp "$VM_USER@$VM_IP:~/gpt-mini-training/checkpoints/step_*.pth" "$LOCAL_DIR/"

# Download tokenizer
echo "[3/4] Downloading tokenizer..."
scp "$VM_USER@$VM_IP:~/gpt-mini-training/tokenizer/spm.model" "$LOCAL_DIR/"

# Download training progress report
echo "[4/4] Downloading progress report..."
scp "$VM_USER@$VM_IP:~/gpt-mini-training/training_progress.json" "$LOCAL_DIR/"

echo "=========================================="
echo "✅ Download complete!"
echo "Files saved to: $LOCAL_DIR"
echo "=========================================="

# List downloaded files
ls -lh "$LOCAL_DIR"
