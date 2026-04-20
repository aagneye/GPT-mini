import torch
from model.gpt import GPT
from tokenizer.tokenizer import CharTokenizer
from config import *

with open("data/dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = CharTokenizer(text)

model = GPT(tokenizer.vocab_size).to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()

context = torch.zeros((1, 1), dtype=torch.long).to(device)
out = model.generate(context, max_new_tokens=300)

print(tokenizer.decode(out[0].tolist()))