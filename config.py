# config.py
import os
import torch

# Environment detection
_on_kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
_on_amd_cloud = os.path.exists("/opt/rocm")


def _env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _env_float(name, default):
    value = os.environ.get(name)
    return float(value) if value is not None else default


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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

# Chinchilla scaling law: optimal training uses ~20 tokens per parameter
# Model config (n_embd=768, n_head=12, n_layer=12) ≈ 110M parameters
# Optimal tokens: 110M × 20 = 2.2B tokens
# With batch_size=128, block_size=512: 65,536 tokens/step
# Required steps: 2.2B ÷ 65,536 ≈ 33,554 iterations
# Using 30,500 for practical training time (~1.8B tokens, 16.4 tokens/param)

if _on_amd_cloud:
    # Favor throughput on large-memory cloud GPUs while keeping the total token
    # budget sane enough to stay within a typical single-GPU spend cap.
    batch_size = _env_int("GPT_BATCH_SIZE", 128)
    block_size = _env_int("GPT_BLOCK_SIZE", 512)
    save_interval = _env_int("GPT_SAVE_INTERVAL", 1000)
    print("Detected AMD Cloud GPU - Using optimized settings!")
elif _on_kaggle:
    batch_size = _env_int("GPT_BATCH_SIZE", 8)
    block_size = _env_int("GPT_BLOCK_SIZE", 1024)
    save_interval = _env_int("GPT_SAVE_INTERVAL", 5000)
else:
    batch_size = _env_int("GPT_BATCH_SIZE", 8)
    block_size = _env_int("GPT_BLOCK_SIZE", 1024)
    save_interval = _env_int("GPT_SAVE_INTERVAL", 5000)

max_iters = _env_int("GPT_MAX_ITERS", 30500 if _on_amd_cloud else 40000)
eval_interval = _env_int("GPT_EVAL_INTERVAL", 200)
learning_rate = _env_float("GPT_LEARNING_RATE", 3e-4)
gradient_accumulation_steps = _env_int(
    "GPT_GRAD_ACCUM_STEPS",
    1,
)
optimizer_foreach = os.environ.get(
    "GPT_OPTIMIZER_FOREACH",
    "0" if _on_amd_cloud else "1",
) == "1"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Model Architecture
# ------------------------

if _on_amd_cloud:
    # ~110M parameter model optimized for AMD MI300X GPU
    # Architecture: 768 embedding dim, 12 heads, 12 layers
    _default_n_embd = 768
    _default_n_head = 12
    _default_n_layer = 12
else:
    _default_n_embd = 1088
    _default_n_head = 17
    _default_n_layer = 15

n_embd = _env_int("GPT_N_EMBD", _default_n_embd)
n_head = _env_int("GPT_N_HEAD", _default_n_head)
n_layer = _env_int("GPT_N_LAYER", _default_n_layer)
dropout = _env_float("GPT_DROPOUT", 0.1)
activation_checkpointing = os.environ.get(
    "GPT_ACTIVATION_CHECKPOINTING",
    "1",
) == "1"

# ------------------------
# Memory Optimization
# ------------------------

# ROCm doesn't support expandable_segments, so disable dataset-on-device
# to avoid OOM errors
if _on_amd_cloud and device == "cuda":
    dataset_on_device = False
else:
    dataset_on_device = _env_bool("GPT_DATASET_ON_DEVICE", _on_amd_cloud)

# ------------------------
# Extra Settings
# ------------------------

keep_last_checkpoints = _env_int(
    "GPT_KEEP_LAST_CHECKPOINTS",
    3 if _on_amd_cloud else 2,
)
generate_tokens = _env_int("GPT_GENERATE_TOKENS", 300)
