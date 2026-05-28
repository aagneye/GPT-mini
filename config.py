# config.py
import os
import torch

# Environment detection
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

# Auto-detect environment and optimize settings
if _on_amd_cloud:
    # The larger 1024/16/16 model with 1024 context OOMs at batch_size=64.
    # Use a small micro-batch and recover throughput with gradient accumulation.
    batch_size = 4
    block_size = 1024
    save_interval = 1000
    print("Detected AMD Cloud GPU - Using optimized settings!")
elif _on_kaggle:
    # Kaggle T4: 15GB VRAM
    batch_size = 8
    block_size = 1024
    save_interval = 5000
else:
    # Local development
    batch_size = 8
    block_size = 1024
    save_interval = 5000

max_iters = 60000
eval_interval = 200
learning_rate = 3e-4
gradient_accumulation_steps = 16 if _on_amd_cloud else 1

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Model Architecture
# ------------------------

n_embd = 1024
n_head = 16
n_layer = 16
dropout = 0.1

# ------------------------
# Extra Settings
# ------------------------

keep_last_checkpoints = 3 if _on_amd_cloud else 2
generate_tokens = 300
