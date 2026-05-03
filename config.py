# config.py

import os
import torch

# Kaggle auto-detection: uses /kaggle/working for writable cache
# Override paths with GPT_DATA_PATH, GPT_SPM_MODEL, or GPT_CACHE_DIR if needed
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
# Training Settings
# ------------------------

# T4 ~15GB: batch 32 × block 512 × ~48M params OOMs on backward; use 8 (+ AMP in train.py).
batch_size = 8           # try 12 if stable; drop to 4 if still OOM
block_size = 512
max_iters = 10000
eval_interval = 200
learning_rate = 3e-4

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Model Architecture
# ------------------------

n_embd = 512
n_head = 8
n_layer = 10
dropout = 0.1            # better for larger dataset

# ------------------------
# Extra Settings
# ------------------------

save_interval = 1000     # save checkpoints
generate_tokens = 300
