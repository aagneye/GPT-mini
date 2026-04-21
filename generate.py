import torch
import torch.nn.functional as F
from model.gpt import GPT
from config import *


checkpoint = torch.load("model.pth", map_location=device)
vocab_size = checkpoint["vocab_size"]
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]


def encode(s):
    return [stoi[c] for c in s]


def decode(tokens):
    return "".join([itos[i] for i in tokens])


def generate(model, idx, max_new_tokens, temperature=1.0):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


model = GPT(vocab_size).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("Model loaded")

prompt = "Hello"
context = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
out = generate(model, context, max_new_tokens=200)

print("\nGenerated text:\n")
print(decode(out[0].tolist()))