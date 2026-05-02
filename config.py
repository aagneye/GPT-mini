# config.py

import os
import torch

# Kaggle kernels set KAGGLE_KERNEL_RUN_TYPE; use input datasets and /kaggle/working for cache.
# Override any path with GPT_DATA_PATH, GPT_SPM_MODEL, or GPT_CACHE_DIR if needed.
_on_kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
_kaggle_data = "/kaggle/input/datasets/aagneyesyam/gpt-dataset/dataset.txt"
_kaggle_spm = "/kaggle/input/datasets/aagneyesyam/gpt-spm/spm.model"
_kaggle_cache = "/kaggle/working"

data_path = os.environ.get(
    "GPT_DATA_PATH",
    _kaggle_data if _on_kaggle else "data/dataset.txt",
)
spm_model_path = os.environ.get(
    "GPT_SPM_MODEL",
    _kaggle_spm if _on_kaggle else "tokenizer/spm.model",
)
_cache_dir = os.environ.get(
    "GPT_CACHE_DIR",
    _kaggle_cache if _on_kaggle else "data",
)
token_cache_path = os.path.join(_cache_dir, "dataset_tokens.pt")
cache_meta_path = os.path.join(_cache_dir, "dataset_tokens.meta.pt")

# ------------------------
# Training Settings
# ------------------------

batch_size = 32          # safe for T4
block_size = 256         # better context than 128
max_iters = 1000         # TEST FIRST → later change to 50000
eval_interval = 200
learning_rate = 3e-4

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Model Architecture
# ------------------------

n_embd = 384             # stronger than 256
n_head = 8
n_layer = 8
dropout = 0.1            # better for larger dataset

# ------------------------
# Extra Settings
# ------------------------

save_interval = 1000     # save checkpoints
generate_tokens = 300
