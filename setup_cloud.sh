#!/bin/bash
# AMD GPU Droplet Setup Script
# Run this immediately after connecting to your VM via SSH

set -e

echo "=========================================="
echo "GPT-mini Cloud Training Setup"
echo "=========================================="

# Update system
echo "[1/6] Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install system dependencies
echo "[2/6] Installing system dependencies..."
sudo apt-get install -y git wget curl vim htop tmux unzip

# Verify ROCm installation
echo "[3/6] Verifying ROCm installation..."
rocm-smi || echo "Warning: ROCm not detected, but PyTorch should still work with ROCm backend"

# Create project directory
echo "[4/6] Setting up project directory..."
mkdir -p ~/gpt-mini-training
cd ~/gpt-mini-training

# Install Python dependencies
echo "[5/6] Installing Python dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
pip install numpy sentencepiece datasets huggingface_hub tqdm

# Verify PyTorch with ROCm
echo "[6/6] Verifying PyTorch installation..."
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
EOF

echo "=========================================="
echo "Setup complete! Next steps:"
echo "1. Upload your code: scp -r /path/to/GPT-mini/* user@vm-ip:~/gpt-mini-training/"
echo "2. Start training in tmux: tmux new -s training"
echo "3. Run: python train.py"
echo "=========================================="
