import torch
import torch.nn.functional as F
from model.gpt import GPT
from config import *
from tokenizer.tokenizer import SPTokenizer

checkpoint = torch.load("model.pth", map_location=device)
vocab_size = checkpoint["vocab_size"]
tokenizer = SPTokenizer()


def generate(model, idx, max_new_tokens, temperature=0.8, top_k=40):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)

        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)

        # top-k filtering
        values, indices = torch.topk(probs, top_k)
        probs = torch.zeros_like(probs).scatter_(1, indices, values)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


model = GPT(vocab_size).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("Model loaded")

prompt = "Hello"
context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
out = generate(model, context, max_new_tokens=200)

print("\nGenerated text:\n")
print(tokenizer.decode(out[0].tolist()))