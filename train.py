import torch
from model.gpt import GPT
from tokenizer.tokenizer import SPTokenizer
from config import *
import random
import os
import glob
import hashlib

# load data
with open("data/dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Clean WikiText section separators.
text = text.replace("= = =", "")

tokenizer = SPTokenizer()
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

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
checkpoint_dir = "checkpoints"
checkpoint_every = 500
os.makedirs(checkpoint_dir, exist_ok=True)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(path, step):
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab_size": tokenizer.vocab_size,
            "tokenizer_model_sha256": file_sha256(tokenizer.model_file),
        },
        path,
    )


start_step = 0
checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "step_*.pth"))
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


if checkpoint_files:
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
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

for iter in range(start_step, max_iters):
    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % 100 == 0:
        print(f"step {iter}, loss {loss.item()}")

    if (iter + 1) % checkpoint_every == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"step_{iter + 1}.pth")
        save_checkpoint(checkpoint_path, iter)
        print(f"Saved checkpoint: {checkpoint_path}")

save_checkpoint("model.pth", max_iters - 1)

print("✅ Model saved as model.pth")