# Updated config for cloud training with larger batch size
# Optimized for AMD MI300X with 1.5TB VRAM

import os
import torch

# Kaggle auto-detection: uses /kaggle/working for writable cache
# Override paths with GPT_DATA_PATH, GPT_SPM_MODEL, or GPT_CACHE_DIR if needed
_on_kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
_on_amd_cloud = os.path.exists("/opt/rocm")  # Detect AMD GPU Droplet

data_path = os.environ.get("GPT_DATA_PATH", "data/dataset.txt")
spm_model_path = os.environ.get("GPT_SPM_MODEL", "tokenizer/spm.model")
_cache_dir = os.environ.get(
    "GPT_CACHE_DIR",
    "/kaggle/working" if _on_kaggle else "data",
)
token_cache_path = os.path.join(_cache_dir, "dataset_tokens.pt")
cache_meta_path = os.path.join(_cache_dir, "dataset_tokens.meta.pt")

# ------------------------
# Training Settings
# ------------------------

# OPTIMIZED FOR AMD MI300X (1.5TB VRAM!)
# We can be VERY aggressive with batch size
if _on_amd_cloud:
    batch_size = 64           # 8x larger than T4!
    block_size = 1024         # 2x larger context
    print("🚀 Detected AMD Cloud GPU - Using optimized settings!")
else:
    batch_size = 8
    block_size = 512

max_iters = 80000
eval_interval = 200
learning_rate = 3e-4

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Model Architecture
# ------------------------

n_embd = 512
n_head = 8
n_layer = 12
dropout = 0.1

# ------------------------
# Extra Settings
# ------------------------

save_interval = 1000         # Save more frequently on cloud (every 1000 steps)
keep_last_checkpoints = 3    # Keep last 3 checkpoints
generate_tokens = 300
