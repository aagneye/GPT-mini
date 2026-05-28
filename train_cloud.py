import os
import glob
import hashlib
import time
import sys
import signal
import json
from datetime import datetime

if "PYTORCH_HIP_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

import torch

from model.gpt import GPT
from tokenizer.tokenizer import SPTokenizer
from config import *

"""
Cloud-optimized training script with automatic checkpoint saving and progress tracking
Designed for AMD GPU Droplets with proper cleanup and syncing
"""

# =====================
# Cloud-specific config
# =====================
CHECKPOINT_DIR = "checkpoints"
SYNC_DIR = "/mnt/persistent"  # Change this to your persistent storage mount
PROGRESS_FILE = "training_progress.json"
SYNC_INTERVAL = 10  # Sync every N checkpoints
HEARTBEAT_INTERVAL = 300  # Log status every 5 minutes
tokens_per_optimizer_step = batch_size * block_size * gradient_accumulation_steps

# Track training state
training_state = {
    "start_time": time.time(),
    "last_sync": time.time(),
    "last_heartbeat": time.time(),
    "total_steps": 0,
    "total_tokens_processed": 0,
    "step_durations": [],
}

# Handle graceful shutdown
def signal_handler(sig, frame):
    print("\n" + "="*50)
    print("🛑 SHUTDOWN SIGNAL RECEIVED")
    print("="*50)
    print("Saving emergency checkpoint...")
    emergency_path = os.path.join(CHECKPOINT_DIR, f"emergency_step_{step_idx}.pth")
    save_checkpoint(emergency_path, step_idx)
    print(f"✅ Emergency checkpoint saved: {emergency_path}")
    
    sync_to_persistent_storage()
    save_progress_report()
    
    print("="*50)
    print("Safe to terminate now. Your progress is saved.")
    print("="*50)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# =====================
# Helper Functions
# =====================

def save_progress_report():
    """Save detailed training progress to JSON"""
    elapsed = time.time() - training_state["start_time"]
    avg_step_time = (
        sum(training_state["step_durations"]) / len(training_state["step_durations"])
        if training_state["step_durations"]
        else None
    )
    estimated_total_hours = (
        (avg_step_time * max_iters) / 3600 if avg_step_time is not None else None
    )
    progress = {
        "last_step": step_idx,
        "total_steps_completed": step_idx - start_step,
        "max_iters": max_iters,
        "progress_percentage": (step_idx / max_iters) * 100,
        "elapsed_hours": elapsed / 3600,
        "estimated_total_hours": estimated_total_hours,
        "tokens_processed": (step_idx - start_step) * tokens_per_optimizer_step,
        "last_update": datetime.now().isoformat(),
        "cost_estimate_usd": (elapsed / 3600) * 1.99,  # $1.99/hr for single GPU
    }
    
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)
    
    return progress

def sync_to_persistent_storage():
    """Sync checkpoints to persistent storage (if mounted)"""
    if os.path.exists(SYNC_DIR):
        os.makedirs(os.path.join(SYNC_DIR, "checkpoints"), exist_ok=True)
        
        # Copy latest checkpoint
        latest = find_latest_checkpoint()
        if latest:
            import shutil
            dest = os.path.join(SYNC_DIR, "checkpoints", os.path.basename(latest))
            shutil.copy2(latest, dest)
            print(f"📦 Synced checkpoint to persistent storage: {dest}")
        
        # Copy progress report
        if os.path.exists(PROGRESS_FILE):
            import shutil
            shutil.copy2(PROGRESS_FILE, SYNC_DIR)
        
        training_state["last_sync"] = time.time()
    else:
        print(f"⚠️  Persistent storage not mounted at {SYNC_DIR}, skipping sync")

def print_heartbeat():
    """Print training status heartbeat"""
    elapsed = time.time() - training_state["start_time"]
    progress = (step_idx / max_iters) * 100
    cost = (elapsed / 3600) * 1.99
    avg_step_time = (
        sum(training_state["step_durations"]) / len(training_state["step_durations"])
        if training_state["step_durations"]
        else None
    )
    tokens_per_second = (
        tokens_per_optimizer_step / avg_step_time if avg_step_time else None
    )
    
    print("\n" + "="*60)
    print(f"💓 HEARTBEAT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Step: {step_idx:,}/{max_iters:,} ({progress:.2f}%)")
    print(f"   Runtime: {elapsed/3600:.2f} hours | Cost: ${cost:.2f}")
    if torch.cuda.is_available():
        print(
            f"   GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / "
            f"{torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB"
        )
    if avg_step_time is not None:
        print(
            f"   Avg step: {avg_step_time:.2f}s | "
            f"Throughput: {tokens_per_second:,.0f} tokens/s"
        )
    print("="*60 + "\n")
    
    training_state["last_heartbeat"] = time.time()

# =====================
# Initialize Tokenizer
# =====================

print("="*60)
print("🚀 CLOUD TRAINING INITIALIZATION")
print("="*60)
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("="*60 + "\n")

tokenizer = SPTokenizer(model_file=spm_model_path, data_path=data_path)

dataset_stat = os.stat(data_path)
tokenizer_stat = os.stat(tokenizer.model_file)
expected_cache_meta = {
    "dataset_size": dataset_stat.st_size,
    "dataset_mtime": dataset_stat.st_mtime,
    "tokenizer_size": tokenizer_stat.st_size,
    "tokenizer_mtime": tokenizer_stat.st_mtime,
    "vocab_size": tokenizer.vocab_size,
}

use_cache = False
if os.path.exists(token_cache_path) and os.path.exists(cache_meta_path):
    cached_meta = torch.load(cache_meta_path, map_location="cpu", weights_only=True)
    use_cache = cached_meta == expected_cache_meta

if use_cache:
    print(f"Loading tokenized dataset cache from {token_cache_path} ...")
    data = torch.load(token_cache_path, map_location="cpu", weights_only=True)
else:
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Dataset loaded: {len(text):,} characters")
    print("Tokenizing dataset (first run, this can take a while)...")
    chunk_size = 2_000_000
    total_chars = len(text)
    all_token_ids = []
    start_time = time.time()

    for chunk_idx, start in enumerate(range(0, total_chars, chunk_size), start=1):
        end = min(start + chunk_size, total_chars)
        chunk_text = text[start:end]
        chunk_token_ids = tokenizer.encode(chunk_text)
        all_token_ids.extend(chunk_token_ids)

        elapsed = time.time() - start_time
        progress = (end / total_chars) * 100 if total_chars else 100.0
        print(
            f"[tokenize] chunk={chunk_idx} chars={end:,}/{total_chars:,} "
            f"({progress:.1f}%) tokens={len(all_token_ids):,} elapsed={elapsed:.1f}s"
        )

    data = torch.tensor(all_token_ids, dtype=torch.long)
    torch.save(data, token_cache_path)
    torch.save(expected_cache_meta, cache_meta_path)
    print(f"Saved tokenized cache to {token_cache_path}")

print(f"Total tokens: {len(data):,}")

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
data_on_device = device == "cuda" and dataset_on_device

if data_on_device:
    print("Moving tokenized dataset to GPU memory for faster batch sampling...")
    train_data = train_data.to(device, non_blocking=True)
    val_data = val_data.to(device, non_blocking=True)

batch_index_device = device if data_on_device else "cpu"
batch_offsets = torch.arange(block_size + 1, device=batch_index_device).unsqueeze(0)

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,), device=batch_index_device)
    batch = data[ix.unsqueeze(1) + batch_offsets]
    x = batch[:, :-1].contiguous()
    y = batch[:, 1:].contiguous()
    if data_on_device:
        return x, y
    if device == "cuda":
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return x, y

# =====================
# Model Setup
# =====================

model = GPT(tokenizer.vocab_size).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    foreach=optimizer_foreach,
)
use_amp = device == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}  |  AMP (fp16): {use_amp}")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def save_checkpoint(path, step):
    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_model_sha256": file_sha256(tokenizer.model_file),
        "training_time_hours": (time.time() - training_state["start_time"]) / 3600,
    }
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)

def checkpoint_step_from_path(path):
    name = os.path.basename(path)
    if name.startswith("step_") and name.endswith(".pth"):
        try:
            return int(name[5:-4])
        except ValueError:
            return -1
    return -1

def prune_old_checkpoints(keep_last=None):
    keep_last = keep_last if keep_last is not None else keep_last_checkpoints
    files = glob.glob(os.path.join(CHECKPOINT_DIR, "step_*.pth"))
    files.sort(key=checkpoint_step_from_path)
    for old_path in files[:-keep_last]:
        try:
            os.remove(old_path)
            print(f"Removed old checkpoint: {old_path}")
        except OSError as e:
            print(f"Could not remove {old_path}: {e}")

def find_latest_checkpoint():
    files = glob.glob(os.path.join(CHECKPOINT_DIR, "step_*.pth"))
    files = [f for f in files if not f.endswith(".tmp")]
    files.sort(key=checkpoint_step_from_path, reverse=True)
    for path in files:
        try:
            torch.load(path, map_location="cpu")
            return path
        except (RuntimeError, OSError) as e:
            print(f"Skipping corrupt checkpoint {path}: {e}")
            try:
                os.remove(path)
            except OSError:
                pass
    return None

# =====================
# Resume Training
# =====================

start_step = 0
resumed = False

def try_resume(path, include_optimizer=True):
    global start_step
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("vocab_size") != tokenizer.vocab_size:
        raise ValueError(
            f"vocab mismatch (checkpoint={checkpoint.get('vocab_size')}, current={tokenizer.vocab_size})"
        )
    expected_tokenizer_hash = checkpoint.get("tokenizer_model_sha256")
    if expected_tokenizer_hash is not None:
        current_tokenizer_hash = file_sha256(tokenizer.model_file)
        if expected_tokenizer_hash != current_tokenizer_hash:
            raise ValueError(
                "tokenizer mismatch (checkpoint tokenizer file hash does not match current tokenizer/spm.model)"
            )
    model.load_state_dict(checkpoint["model_state_dict"])
    if include_optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_step = checkpoint.get("step", -1) + 1

latest_checkpoint = find_latest_checkpoint()
if latest_checkpoint:
    try:
        try_resume(latest_checkpoint, include_optimizer=True)
        resumed = True
        print(f"✅ Resumed from {latest_checkpoint} at step {start_step}")
    except (RuntimeError, KeyError, ValueError) as e:
        print(f"⚠️  Skipping incompatible checkpoint {latest_checkpoint}: {e}")

if (not resumed) and os.path.exists("model.pth"):
    try:
        try_resume("model.pth", include_optimizer=True)
        resumed = True
        print(f"✅ Resumed from model.pth at step {start_step}")
    except (RuntimeError, KeyError, ValueError) as e:
        print(f"⚠️  Skipping incompatible model.pth: {e}")

if not resumed:
    print("🎯 Starting training from scratch")

print(f"\nTraining Configuration:")
print(f"  Total iterations: {max_iters:,}")
print(f"  Dataset tokens: {len(data):,}")
print(f"  Micro-batch size: {batch_size:,}")
print(f"  Gradient accumulation steps: {gradient_accumulation_steps:,}")
print(f"  Tokens per optimizer step: {tokens_per_optimizer_step:,}")
print(f"  Total tokens to process: {max_iters * tokens_per_optimizer_step:,}")
print(f"  Activation checkpointing: {activation_checkpointing}")
print(f"  Dataset on device: {data_on_device}")
print("  Estimated training time: benchmarking after first few steps")
print("  Estimated cost: benchmarking after first few steps")
print()

# =====================
# Training Loop
# =====================

step_idx = start_step

for step_idx in range(start_step, max_iters):
    step_start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    loss = None
    for micro_step in range(gradient_accumulation_steps):
        xb, yb = get_batch("train")
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, loss = model(xb, yb)
            loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()

    scaler.step(optimizer)
    scaler.update()
    step_duration = time.time() - step_start_time
    training_state["step_durations"].append(step_duration)
    if len(training_state["step_durations"]) > 50:
        training_state["step_durations"].pop(0)

    if step_idx % 100 == 0:
        print(f"step {step_idx:,}, loss {(loss.item() * gradient_accumulation_steps):.4f}")
    
    if step_idx % eval_interval == 0 and step_idx > 0:
        model.eval()
        with torch.no_grad():
            val_losses = []
            for _ in range(10):
                xb_val, yb_val = get_batch("val")
                with torch.amp.autocast("cuda", enabled=use_amp):
                    _, val_loss = model(xb_val, yb_val)
                val_losses.append(val_loss.item())
            avg_val_loss = sum(val_losses) / len(val_losses)
            train_loss = loss.item() * gradient_accumulation_steps if loss is not None else float("nan")
            print(f"[EVAL] step {step_idx:,} | train loss: {train_loss:.4f} | val loss: {avg_val_loss:.4f}")
            
            # Generate sample text
            context = torch.tensor([tokenizer.sp.bos_id()], dtype=torch.long, device=device).unsqueeze(0)
            generated_ids = model.generate(context, max_new_tokens=100, temperature=0.8, top_k=40)
            sample_text = tokenizer.decode(generated_ids[0].tolist())
            print(f"[SAMPLE] {sample_text[:200]}")
            print()
        model.train()
    
    # Heartbeat logging
    if time.time() - training_state["last_heartbeat"] > HEARTBEAT_INTERVAL:
        print_heartbeat()

    # Save checkpoint
    if (step_idx + 1) % save_interval == 0:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"step_{step_idx + 1}.pth")
        save_checkpoint(checkpoint_path, step_idx)
        prune_old_checkpoints()
        print(f"💾 Saved checkpoint: {checkpoint_path}")
        
        # Save progress report
        progress = save_progress_report()
        eta_hours = (
            progress["estimated_total_hours"] - progress["elapsed_hours"]
            if progress["estimated_total_hours"] is not None
            else None
        )
        eta_text = f"{eta_hours:.1f}h" if eta_hours is not None else "warming up"
        print(f"📊 Progress: {progress['progress_percentage']:.1f}% | "
              f"Cost: ${progress['cost_estimate_usd']:.2f} | "
              f"ETA: {eta_text}")
        
        # Sync to persistent storage periodically
        if (step_idx + 1) % (save_interval * SYNC_INTERVAL) == 0:
            sync_to_persistent_storage()

# =====================
# Final Save
# =====================

print("\n" + "="*60)
print("🎉 TRAINING COMPLETE!")
print("="*60)

save_checkpoint("model.pth", max_iters - 1)
save_checkpoint(os.path.join(CHECKPOINT_DIR, f"final_step_{max_iters}.pth"), max_iters - 1)

progress = save_progress_report()
sync_to_persistent_storage()

print(f"✅ Final model saved as model.pth")
print(f"📊 Total training time: {progress['elapsed_hours']:.2f} hours")
print(f"💰 Total cost: ${progress['cost_estimate_usd']:.2f}")
print(f"🎯 Tokens processed: {progress['tokens_processed']:,}")
print("="*60)
