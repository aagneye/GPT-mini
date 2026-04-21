import torch
from model.gpt import GPT
from tokenizer.tokenizer import SPTokenizer
from config import *
import random
import os
import glob

# load data
with open("data/dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()

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


def save_checkpoint(path, step):
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab_size": tokenizer.vocab_size,
        },
        path,
    )


start_step = 0
checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "step_*.pth"))
if checkpoint_files:
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_step = checkpoint["step"] + 1
    print(f"Resumed from {latest_checkpoint} at step {start_step}")
elif os.path.exists("model.pth"):
    checkpoint = torch.load("model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    start_step = checkpoint.get("step", 0) + 1
    print(f"Resumed from model.pth at step {start_step}")
else:
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