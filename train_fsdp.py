"""
FSDP pretraining for the 1B GPT on 4x Tesla T4 (SM 75).

Launch:
    GPT_PROFILE=t4_1b torchrun --standalone --nproc_per_node=4 train_fsdp.py

Design notes tied to the T4 constraints:
- T4 is Turing (SM 75): no bf16. We use fp16 params with reduce in fp32
  (MixedPrecision) and a ShardedGradScaler.
- 16 GB/GPU cannot hold 1B with fp32 AdamW, so ShardingStrategy.FULL_SHARD.
- PCIe only (no NVLink): amortize FSDP all-gather/reduce-scatter comm with
  gradient_accumulation_steps=16.
- Activation checkpointing is applied via apply_activation_checkpointing on
  Block (NOT inside GPT.forward), which is FSDP-compatible.
- Sharded distributed checkpointing (SHARDED_STATE_DICT) writes ~13 GB across
  ranks in parallel; resume reads a small manifest instead of torch.load-ing
  candidate files.
"""

from __future__ import annotations

import functools
import json
import math
import os
import signal
import sys
import time
from datetime import datetime

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, BackwardPrefetch
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import config as cfg
from data_loader import DataLoader
from model.gpt import GPT, GPTConfig, Block
from tokenizer.tokenizer import SPTokenizer

CHECKPOINT_DIR = "checkpoints_fsdp"
PROGRESS_FILE = "training_progress.json"
HEARTBEAT_INTERVAL = 300


# =====================
# Distributed setup
# =====================

def setup_dist():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def is_main(rank):
    return rank == 0


def log(rank, *args):
    if is_main(rank):
        print(*args, flush=True)


# =====================
# LR schedule
# =====================

def get_lr(step, peak_lr, warmup_steps, max_iters, min_lr_ratio):
    min_lr = peak_lr * min_lr_ratio
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_iters:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_iters - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (peak_lr - min_lr)


# =====================
# Weight decay grouping
# =====================

def build_optimizer(model, peak_lr, weight_decay, beta1, beta2):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 2D params (matmuls, embeddings) get weight decay; biases and
        # LayerNorm (1D) do not.
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=peak_lr, betas=(beta1, beta2))


# =====================
# Sharded checkpointing
# =====================

def manifest_path():
    return os.path.join(CHECKPOINT_DIR, "latest.json")


def write_manifest(step, ckpt_subdir, gpt_config, vocab_size):
    payload = {
        "step": step,
        "dir": ckpt_subdir,
        "gpt_config": gpt_config.to_dict(),
        "vocab_size": vocab_size,
        "saved_at": datetime.now().isoformat(),
    }
    tmp = manifest_path() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, manifest_path())


def read_manifest():
    if not os.path.exists(manifest_path()):
        return None
    with open(manifest_path(), "r", encoding="utf-8") as f:
        return json.load(f)


def save_sharded_checkpoint(model, optimizer, step):
    ckpt_subdir = f"step_{step}"
    ckpt_dir = os.path.join(CHECKPOINT_DIR, ckpt_subdir)
    model_sd, optim_sd = get_state_dict(model, optimizer)
    state = {"model": model_sd, "optim": optim_sd, "step": step}
    dcp.save(state, checkpoint_id=ckpt_dir)
    return ckpt_subdir


def load_sharded_checkpoint(model, optimizer, ckpt_subdir):
    ckpt_dir = os.path.join(CHECKPOINT_DIR, ckpt_subdir)
    model_sd, optim_sd = get_state_dict(model, optimizer)
    state = {"model": model_sd, "optim": optim_sd}
    dcp.load(state, checkpoint_id=ckpt_dir)
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optim"],
    )


# =====================
# Main
# =====================

def main():
    rank, local_rank, world_size = setup_dist()
    device = f"cuda:{local_rank}"

    if os.environ.get("GPT_PROFILE", "").lower() != "t4_1b":
        log(rank, "WARNING: GPT_PROFILE is not t4_1b; using whatever config resolved.")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    tokenizer = SPTokenizer(model_file=cfg.spm_model_path)
    vocab_size = tokenizer.vocab_size

    gpt_config = GPTConfig(
        vocab_size=vocab_size,
        n_embd=cfg.n_embd,
        n_head=cfg.n_head,
        n_layer=cfg.n_layer,
        block_size=cfg.block_size,
        dropout=cfg.dropout,
    )

    log(rank, "=" * 60)
    log(rank, f"FSDP pretrain | world_size={world_size} | device={device}")
    log(rank, f"Arch: n_embd={gpt_config.n_embd} n_layer={gpt_config.n_layer} "
              f"n_head={gpt_config.n_head} block={gpt_config.block_size} vocab={vocab_size}")
    log(rank, "=" * 60)

    # Build model on CPU/meta then let FSDP shard it across GPUs.
    model = GPT(gpt_config)
    if is_main(rank):
        print(f"Model params (unsharded): {model.num_params():,}", flush=True)

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls={Block}
    )
    mp_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        device_id=local_rank,
        use_orig_params=True,
    )

    # Activation checkpointing on each Block (non-reentrant), applied AFTER FSDP
    # wrapping so it composes with the FSDP units.
    if cfg.activation_checkpointing:
        non_reentrant = functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant,
            check_fn=lambda m: isinstance(m, Block),
        )

    optimizer = build_optimizer(
        model, cfg.learning_rate, cfg.weight_decay, cfg.beta1, cfg.beta2
    )
    scaler = ShardedGradScaler()

    train_loader = DataLoader(
        cfg.data_dir, "train", cfg.batch_size, cfg.block_size,
        device=device, rank=rank, world_size=world_size,
    )
    try:
        val_loader = DataLoader(
            cfg.data_dir, "val", cfg.batch_size, cfg.block_size,
            device=device, rank=rank, world_size=world_size,
        )
    except (FileNotFoundError, ValueError):
        val_loader = None

    grad_accum = cfg.gradient_accumulation_steps
    tokens_per_step = cfg.batch_size * cfg.block_size * grad_accum * world_size

    # ---- Resume from manifest ----
    start_step = 0
    manifest = read_manifest()
    if manifest is not None:
        try:
            load_sharded_checkpoint(model, optimizer, manifest["dir"])
            start_step = int(manifest["step"]) + 1
            log(rank, f"Resumed from {manifest['dir']} at step {start_step}")
        except Exception as e:  # noqa: BLE001 - report and start fresh
            log(rank, f"Could not resume from manifest ({e}); starting fresh.")

    training_state = {
        "start_time": time.time(),
        "last_heartbeat": time.time(),
        "step_durations": [],
    }

    def save_progress(step):
        if not is_main(rank):
            return
        elapsed = time.time() - training_state["start_time"]
        durations = training_state["step_durations"]
        avg = sum(durations) / len(durations) if durations else None
        progress = {
            "last_step": step,
            "max_iters": cfg.max_iters,
            "progress_percentage": (step / cfg.max_iters) * 100,
            "elapsed_hours": elapsed / 3600,
            "tokens_processed": (step - start_step) * tokens_per_step,
            "tokens_per_second": (tokens_per_step / avg) if avg else None,
            "world_size": world_size,
            "last_update": datetime.now().isoformat(),
        }
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)

    # ---- Graceful shutdown: save a sharded emergency checkpoint ----
    current_step = {"idx": start_step}

    def signal_handler(sig, frame):
        log(rank, "\nSHUTDOWN SIGNAL: saving emergency sharded checkpoint...")
        subdir = save_sharded_checkpoint(model, optimizer, current_step["idx"])
        dist.barrier()
        if is_main(rank):
            write_manifest(subdir_step(subdir), subdir, gpt_config, vocab_size)
            save_progress(current_step["idx"])
        log(rank, "Emergency checkpoint saved. Exiting.")
        dist.destroy_process_group()
        sys.exit(0)

    def subdir_step(subdir):
        return int(subdir.split("_")[-1])

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    log(rank, f"Tokens/optimizer step: {tokens_per_step:,} | steps: {cfg.max_iters:,}")
    log(rank, f"Grad accum: {grad_accum} | grad clip: {cfg.grad_clip} | wd: {cfg.weight_decay}")

    model.train()
    for step in range(start_step, cfg.max_iters):
        current_step["idx"] = step
        step_start = time.time()

        lr = get_lr(step, cfg.learning_rate, cfg.warmup_steps, cfg.max_iters, cfg.min_lr_ratio)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro in range(grad_accum):
            xb, yb = train_loader.get_batch()
            with torch.autocast("cuda", dtype=torch.float16):
                _, loss = model(xb, yb)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            loss_accum += loss.item()

        scaler.unscale_(optimizer)
        model.clip_grad_norm_(cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        dur = time.time() - step_start
        training_state["step_durations"].append(dur)
        if len(training_state["step_durations"]) > 50:
            training_state["step_durations"].pop(0)

        if step % 100 == 0:
            log(rank, f"step {step:,} | loss {loss_accum:.4f} | lr {lr:.2e} | {dur:.2f}s")

        # Eval + sample (rank 0), infrequent to keep cost low at 1B.
        if val_loader is not None and step % cfg.eval_interval == 0 and step > 0:
            model.eval()
            with torch.no_grad():
                vlosses = []
                for _ in range(10):
                    vx, vy = val_loader.get_batch()
                    with torch.autocast("cuda", dtype=torch.float16):
                        _, vloss = model(vx, vy)
                    vlosses.append(vloss.item())
            log(rank, f"[EVAL] step {step:,} | val loss {sum(vlosses)/len(vlosses):.4f}")
            model.train()

        if is_main(rank) and step % 2000 == 0 and step > 0:
            _sample(model, tokenizer, device, rank)

        if time.time() - training_state["last_heartbeat"] > HEARTBEAT_INTERVAL:
            _heartbeat(rank, step, training_state, tokens_per_step, device)
            training_state["last_heartbeat"] = time.time()

        # Sharded checkpoint every save_interval steps.
        if (step + 1) % cfg.save_interval == 0:
            subdir = save_sharded_checkpoint(model, optimizer, step)
            dist.barrier()
            if is_main(rank):
                write_manifest(step, subdir, gpt_config, vocab_size)
                save_progress(step)
                print(f"Saved sharded checkpoint: {subdir}", flush=True)

    # Final checkpoint.
    subdir = save_sharded_checkpoint(model, optimizer, cfg.max_iters - 1)
    dist.barrier()
    if is_main(rank):
        write_manifest(cfg.max_iters - 1, subdir, gpt_config, vocab_size)
        save_progress(cfg.max_iters - 1)
        print("Training complete.", flush=True)

    dist.destroy_process_group()


def _sample(model, tokenizer, device, rank):
    try:
        bos = tokenizer.sp.bos_id()
        ctx = torch.tensor([[bos]], dtype=torch.long, device=device)
        with FSDP.summon_full_params(model, writeback=False):
            out = model.generate(ctx, max_new_tokens=50, temperature=0.8, top_k=40)
        text = tokenizer.decode(out[0].tolist())
        print(f"[SAMPLE] {text[:200]}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[SAMPLE] skipped ({e})", flush=True)


def _heartbeat(rank, step, state, tokens_per_step, device):
    if rank != 0:
        return
    elapsed = time.time() - state["start_time"]
    durations = state["step_durations"]
    avg = sum(durations) / len(durations) if durations else None
    tps = (tokens_per_step / avg) if avg else None
    mem = torch.cuda.memory_allocated(device) / 1e9
    total = torch.cuda.get_device_properties(device).total_memory / 1e9
    ts = datetime.now().strftime("%H:%M:%S")
    if tps:
        msg = (
            f"[HEARTBEAT] {ts} step {step:,} | runtime {elapsed/3600:.2f}h | "
            f"mem {mem:.2f}/{total:.2f}GB | throughput {tps:,.0f} tok/s"
        )
    else:
        msg = f"[HEARTBEAT] {ts} step {step:,} | warming up"
    print(msg, flush=True)


if __name__ == "__main__":
    main()
