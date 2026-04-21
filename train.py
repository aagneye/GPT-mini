import torch
from model.gpt import GPT
from tokenizer.tokenizer import SPTokenizer
from config import *
import random

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

for iter in range(max_iters):
    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % 100 == 0:
        print(f"step {iter}, loss {loss.item()}")

torch.save({
    "model_state_dict": model.state_dict(),
    "vocab_size": tokenizer.vocab_size,
}, "model.pth")

print("✅ Model saved as model.pth")