#!/usr/bin/env bash
# Full 1B pretrain pipeline on 4x T4. Detach-safe (tmux/nohup).
set -euo pipefail

ROOT="/data/.gpt-mini-1b"
MNT="/mnt/gpt-mini-1b"
LOG_DIR="${MNT}/logs"
DATA_DIR="${MNT}/data/fineweb_edu"
SAMPLE_TXT="${MNT}/data/fineweb_sample.txt"
SPM_MODEL="${ROOT}/tokenizer/spm32k.model"
HF_HOME="${MNT}/hf-cache"
export HF_HOME
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export GPT_PROFILE=t4_1b
export GPT_SPM_MODEL="${SPM_MODEL}"
export GPT_DATA_DIR="${DATA_DIR}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mkdir -p "${LOG_DIR}" "${DATA_DIR}" "$(dirname "${SAMPLE_TXT}")" "${ROOT}/checkpoints_fsdp"
cd "${ROOT}"
# shellcheck disable=SC1091
source .venv/bin/activate

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${LOG_DIR}/pipeline.log"; }

log "=== 1B pipeline start | GPUs=$(python -c 'import torch; print(torch.cuda.device_count())') ==="

# Symlink heavy dirs onto /mnt so the repo stays light on /
ln -sfn "${MNT}/checkpoints_fsdp" "${ROOT}/checkpoints_fsdp" || true
ln -sfn "${DATA_DIR}" "${ROOT}/data/fineweb_edu" || true
mkdir -p "${ROOT}/data"

# ---------- 1) Tokenizer sample (~2GB text) ----------
if [[ ! -f "${SPM_MODEL}" ]]; then
  if [[ ! -f "${SAMPLE_TXT}" ]]; then
    log "Building FineWeb-Edu tokenizer sample -> ${SAMPLE_TXT}"
    export SAMPLE_TXT
    python - <<'PY'
from datasets import load_dataset
import os
out = os.environ["SAMPLE_TXT"]
ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
target = 2_000_000_000  # ~2GB chars
n = 0
with open(out, "w", encoding="utf-8") as f:
    for ex in ds:
        t = (ex.get("text") or "").strip()
        if not t:
            continue
        f.write(t.replace("\n", " ") + "\n")
        n += len(t) + 1
        if n >= target:
            break
        if n % 200_000_000 < 100_000:
            print(f"  sample chars={n:,}", flush=True)
print(f"Wrote {n:,} chars to {out}", flush=True)
PY
  else
    log "Tokenizer sample already exists"
  fi

  log "Training 32k BPE tokenizer with byte_fallback"
  rm -f "${SPM_MODEL}" "${ROOT}/tokenizer/spm32k.vocab"
  GPT_SPM_MODEL="${SPM_MODEL}" \
  GPT_TOKENIZER_TRAIN_INPUT="${SAMPLE_TXT}" \
  GPT_TOKENIZER_VOCAB_SIZE=32000 \
  GPT_TOKENIZER_INPUT_SENTENCE_SIZE=2000000 \
  python -m tokenizer 2>&1 | tee -a "${LOG_DIR}/tokenizer.log"
  log "Tokenizer ready: ${SPM_MODEL}"
else
  log "Tokenizer already present: ${SPM_MODEL}"
fi

# ---------- 2) Tokenize FineWeb into uint16 shards ----------
if [[ ! -f "${DATA_DIR}/index.json" ]]; then
  log "Tokenizing FineWeb-Edu sample-10BT -> ${DATA_DIR}"
  python prepare_fineweb.py \
    --spm-model "${SPM_MODEL}" \
    --out "${DATA_DIR}" \
    --workers 48 \
    --shard-size 100000000 \
    2>&1 | tee -a "${LOG_DIR}/prepare_fineweb.log"
  log "Shards ready"
else
  log "Shards already present"
fi

# ---------- 3) Smoke test (200 steps) ----------
SMOKE_MARKER="${LOG_DIR}/smoke.ok"
if [[ ! -f "${SMOKE_MARKER}" ]]; then
  log "Smoke test: 200 steps on 4 GPUs"
  mkdir -p "${MNT}/checkpoints_fsdp_smoke"
  GPT_PROFILE=t4_1b \
  GPT_MAX_ITERS=200 \
  GPT_SAVE_INTERVAL=100 \
  GPT_EVAL_INTERVAL=200 \
  GPT_CHECKPOINT_DIR="${MNT}/checkpoints_fsdp_smoke" \
  GPT_PROGRESS_FILE="${LOG_DIR}/smoke_progress.json" \
  torchrun --standalone --nproc_per_node=4 train_fsdp.py \
    2>&1 | tee "${LOG_DIR}/smoke.log"
  touch "${SMOKE_MARKER}"
  log "Smoke test OK"
else
  log "Smoke test already passed"
fi

# ---------- 4) Full pretrain (20k steps) ----------
log "Starting FULL pretrain (20,000 steps). Detach and monitor logs."
mkdir -p "${MNT}/checkpoints_fsdp"
ln -sfn "${MNT}/checkpoints_fsdp" "${ROOT}/checkpoints_fsdp"

GPT_PROFILE=t4_1b \
GPT_CHECKPOINT_DIR="${MNT}/checkpoints_fsdp" \
GPT_PROGRESS_FILE="${LOG_DIR}/training_progress.json" \
torchrun --standalone --nproc_per_node=4 train_fsdp.py \
  2>&1 | tee -a "${LOG_DIR}/pretrain.log"

log "=== Pretrain finished ==="
