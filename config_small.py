# config.py - SMALLER MODEL (trains faster, better for limited compute)

import os
import torch

# Kaggle auto-detection: uses /kaggle/working for writable cache
_on_kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

data_path = os.environ.get("GPT_DATA_PATH", "data/dataset.txt")
spm_model_path = os.environ.get("GPT_SPM_MODEL", "tokenizer/spm.model")
_cache_dir = os.environ.get(
    "GPT_CACHE_DIR",
    "/kaggle/working" if _on_kaggle else "data",
)
token_cache_path = os.path.join(_cache_dir, "dataset_tokens.pt")
cache_meta_path = os.path.join(_cache_dir, "dataset_tokens.meta.pt")

# ------------------------
# Training Settings (OPTIMIZED FOR GTX 1650)
# ------------------------

batch_size = 32          # Good for smaller model
block_size = 256         # Balance between context and speed
max_iters = 50000        # Enough for good convergence
eval_interval = 200
learning_rate = 3e-4

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Model Architecture (SMALLER - ~25M params)
# ------------------------

n_embd = 384             # Sweet spot
n_head = 6               # Divides evenly into 384
n_layer = 8              # Moderate depth
dropout = 0.1

# ------------------------
# Extra Settings
# ------------------------

save_interval = 1000
generate_tokens = 300
