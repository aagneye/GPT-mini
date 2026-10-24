#!/usr/bin/env bash
set -euo pipefail
STORE="${GPT_STORAGE_ROOT:-/data/gpt-mini-1b}"
cd /data/.gpt-mini-1b
git fetch origin
git checkout feat/scale-1b-t4-fsdp
git pull --ff-only origin feat/scale-1b-t4-fsdp
chmod +x scripts/run_1b_pipeline.sh
mkdir -p data "${STORE}/logs"
if tmux has-session -t gpt1b 2>/dev/null; then
  tmux kill-session -t gpt1b
fi
tmux_env=(-e "GPT_STORAGE_ROOT=${STORE}")
# Forward GPU selection only when set: an empty CUDA_VISIBLE_DEVICES hides every GPU.
for var in CUDA_VISIBLE_DEVICES GPT_NPROC; do
  if [[ -n "${!var:-}" ]]; then
    tmux_env+=(-e "${var}=${!var}")
  fi
done
tmux new-session -d -s gpt1b -c /data/.gpt-mini-1b "${tmux_env[@]}" 'bash scripts/run_1b_pipeline.sh'
sleep 4
tmux ls
echo '--- log ---'
if [[ -f "${STORE}/logs/pipeline.log" ]]; then
  tail -n 40 "${STORE}/logs/pipeline.log"
else
  echo 'log not yet'
  tmux capture-pane -t gpt1b -p | tail -n 40
fi
ps -ef | grep -E 'run_1b_pipeline|prepare_fineweb|torchrun|train_fsdp|SentencePiece' | grep -v grep | head -20
