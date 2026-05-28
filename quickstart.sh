#!/bin/bash
# Quick deployment script - Run this on the AMD GPU Droplet VM

set -e

echo "=========================================="
echo "GPT-mini Quick Start Deployment"
echo "=========================================="

# Update and install dependencies
echo "[1/5] Installing dependencies..."
pip install --upgrade pip
pip install torch numpy sentencepiece datasets huggingface_hub tqdm

# Verify PyTorch + ROCm
echo "[2/5] Verifying GPU..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')
"

# Create tmux session
echo "[3/5] Setting up tmux session..."
tmux new-session -d -s training

# Prepare training
echo "[4/5] Starting training in tmux..."
tmux send-keys -t training "cd ~/gpt-mini-training" C-m
tmux send-keys -t training "GPT_BATCH_SIZE=512 GPT_BLOCK_SIZE=512 GPT_GRAD_ACCUM_STEPS=1 GPT_MAX_ITERS=20000 GPT_ACTIVATION_CHECKPOINTING=0 python train_cloud.py" C-m

echo "[5/5] Setup complete!"
echo "=========================================="
echo "✅ Training is running in tmux session 'training'"
echo ""
echo "Commands:"
echo "  View training:  tmux attach -t training"
echo "  Detach:         Ctrl+B, then D"
echo "  Monitor GPU:    watch -n 1 rocm-smi"
echo "  Check progress: cat training_progress.json"
echo "=========================================="
