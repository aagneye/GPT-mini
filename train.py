import torch
from torch.cuda.amp import GradScaler, autocast

from model.gpt import GPT
from tokenizer.tokenizer import SPTokenizer
from config import *
import os
import glob
import hashlib
import time

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
    cached_meta = torch.load(cache_meta_path, map_location="cpu")
    use_cache = cached_meta == expected_cache_meta

if use_cache:
    print(f"Loading tokenized dataset cache from {token_cache_path} ...")
    data = torch.load(token_cache_path, map_location="cpu")
else:
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Dataset loaded: {len(text):,} characters")
    print("Tokenizing dataset (first run, this can take a while)...")
    chunk_size = 2_000_000  # chars per chunk
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

# split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


model = GPT(tokenizer.vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
use_amp = device == "cuda"
scaler = GradScaler(enabled=use_amp)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}  |  AMP (fp16): {use_amp}")
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)


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
    files = glob.glob(os.path.join(checkpoint_dir, "step_*.pth"))
    files.sort(key=checkpoint_step_from_path)
    for old_path in files[:-keep_last]:
        try:
            os.remove(old_path)
            print(f"Removed old checkpoint: {old_path}")
        except OSError as e:
            print(f"Could not remove {old_path}: {e}")


def find_latest_checkpoint():
    files = glob.glob(os.path.join(checkpoint_dir, "step_*.pth"))
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
        print(f"Resumed from {latest_checkpoint} at step {start_step}")
    except (RuntimeError, KeyError, ValueError) as e:
        print(f"Skipping incompatible checkpoint {latest_checkpoint}: {e}")

if (not resumed) and os.path.exists("model.pth"):
    try:
        try_resume("model.pth", include_optimizer=True)
        resumed = True
        print(f"Resumed from model.pth at step {start_step}")
    except (RuntimeError, KeyError, ValueError) as e:
        print(f"Skipping incompatible model.pth: {e}")

if not resumed:
    print("Starting training from scratch")

print(f"\nTraining for {max_iters:,} iterations")
print(f"Dataset tokens: {len(data):,}")
print(f"Tokens per batch: {batch_size * block_size:,}")
print(f"Total tokens to process: {max_iters * batch_size * block_size:,}")
print()

for step_idx in range(start_step, max_iters):
    xb, yb = get_batch("train")

    optimizer.zero_grad(set_to_none=True)
    with autocast(enabled=use_amp):
        logits, loss = model(xb, yb)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if step_idx % 100 == 0:
        print(f"step {step_idx}, loss {loss.item():.4f}")
    
    if step_idx % eval_interval == 0 and step_idx > 0:
        model.eval()
        with torch.no_grad():
            val_losses = []
            for _ in range(10):
                xb_val, yb_val = get_batch("val")
                with autocast(enabled=use_amp):
                    _, val_loss = model(xb_val, yb_val)
                val_losses.append(val_loss.item())
            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f"[EVAL] step {step_idx} | train loss: {loss.item():.4f} | val loss: {avg_val_loss:.4f}")
            
            # Generate sample text
            context = torch.tensor([tokenizer.sp.bos_id()], dtype=torch.long, device=device).unsqueeze(0)
            generated_ids = model.generate(context, max_new_tokens=100, temperature=0.8, top_k=40)
            sample_text = tokenizer.decode(generated_ids[0].tolist())
            print(f"[SAMPLE] {sample_text[:200]}")
            print()
        model.train()

    if (step_idx + 1) % save_interval == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"step_{step_idx + 1}.pth")
        save_checkpoint(checkpoint_path, step_idx)
        prune_old_checkpoints()
        print(f"Saved checkpoint: {checkpoint_path}")

save_checkpoint("model.pth", max_iters - 1)

print("✅ Model saved as model.pth")