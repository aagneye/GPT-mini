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

if _on_amd_cloud:
    # Favor throughput on large-memory cloud GPUs while keeping the total token
    # budget sane enough to stay within a typical single-GPU spend cap.
    batch_size = _env_int("GPT_BATCH_SIZE", 512)
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

max_iters = _env_int("GPT_MAX_ITERS", 20000 if _on_amd_cloud else 40000)
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
dataset_on_device = _env_bool("GPT_DATASET_ON_DEVICE", _on_amd_cloud)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Model Architecture
# ------------------------

if _on_amd_cloud:
    # Default to a ~250M-class model while staying closer to the working
    # memory envelope on the AMD VM.
    _default_n_embd = 1088
    _default_n_head = 17
    _default_n_layer = 15
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
    "0",
) == "1"

# ------------------------
# Extra Settings
# ------------------------

keep_last_checkpoints = _env_int(
    "GPT_KEEP_LAST_CHECKPOINTS",
    3 if _on_amd_cloud else 2,
)
generate_tokens = _env_int("GPT_GENERATE_TOKENS", 300)
