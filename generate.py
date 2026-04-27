import torch
import torch.nn.functional as F
from model.gpt import GPT
from config import *
from tokenizer.tokenizer import SPTokenizer
import hashlib

checkpoint = torch.load("model.pth", map_location=device, weights_only=True)
vocab_size = checkpoint["vocab_size"]
tokenizer = SPTokenizer()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if vocab_size != tokenizer.vocab_size:
    raise ValueError(
        f"Tokenizer/model vocab mismatch: checkpoint={vocab_size}, tokenizer={tokenizer.vocab_size}"
    )

expected_tokenizer_hash = checkpoint.get("tokenizer_model_sha256")
if expected_tokenizer_hash is not None:
    current_tokenizer_hash = file_sha256(tokenizer.model_file)
    if expected_tokenizer_hash != current_tokenizer_hash:
        raise ValueError(
            "Tokenizer mismatch: checkpoint was trained with a different tokenizer/spm.model file."
        )


def generate(model, idx, max_new_tokens, temperature=0.8, top_k=40):
    banned_ids = [tokenizer.unk_id]
    # Avoid generating sentence boundary tokens as plain text output.
    if tokenizer.sp.bos_id() >= 0:
        banned_ids.append(tokenizer.sp.bos_id())
    if tokenizer.sp.pad_id() >= 0:
        banned_ids.append(tokenizer.sp.pad_id())

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)

        logits = logits[:, -1, :] / temperature
        for token_id in banned_ids:
            logits[:, token_id] = float("-inf")
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
# Debug: Check for invalid token IDs
tokens = out[0].tolist()
unk_count = sum(1 for t in tokens if t == tokenizer.unk_id)
invalid_tokens = [t for t in tokens if t >= vocab_size or t < 0]

print(f"Vocab size: {vocab_size}")
print(f"Unknown token ID: {tokenizer.unk_id}")
print(f"Unknown tokens in output: {unk_count}/{len(tokens)} ({100*unk_count/len(tokens):.1f}%)")
if invalid_tokens:
    print(f"WARNING: Found {len(invalid_tokens)} invalid token IDs: {invalid_tokens[:10]}")
print(f"Max token ID generated: {max(tokens)}")
print(f"\nFirst 20 token IDs: {tokens[:20]}")
print()
print(tokenizer.decode(tokens))