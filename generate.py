"""
Interactive chat assistant using trained GPT model.
Usage: python generate.py [--mode chat|single]
"""

import torch
import torch.nn.functional as F
from model.gpt import GPT
from config import *
from tokenizer.tokenizer import SPTokenizer
import hashlib
import sys

checkpoint = torch.load("model.pth", map_location=device, weights_only=True)
vocab_size = checkpoint["vocab_size"]
tokenizer = SPTokenizer(model_file=spm_model_path, data_path=data_path)


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
    """Generate text autoregressively from context."""
    banned_ids = [tokenizer.unk_id]
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


def format_instruction_prompt(instruction, context=""):
    """Format user instruction in clean ### format (matches training data)."""
    prompt = f"### Instruction:\n{instruction}\n\n"
    
    if context.strip():
        prompt += f"### Context:\n{context}\n\n"
    
    prompt += "### Response:\n"
    
    return prompt


def chat_mode(model):
    """Interactive chat loop."""
    print("\n" + "="*60)
    print("🤖 GPT Chat Assistant (Dolly-15K + Alpaca trained)")
    print("="*60)
    print("Type your instruction/question and press Enter.")
    print("Commands:")
    print("  /help    - Show help")
    print("  /temp N  - Set temperature (0.1-2.0, default 0.8)")
    print("  /tokens N- Set max tokens (default from config)")
    print("  /quit    - Exit")
    print("="*60 + "\n")

    temperature = 0.8
    max_tokens = generate_tokens

    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Commands
            if user_input.startswith("/"):
                cmd = user_input.lower().split()
                
                if cmd[0] == "/quit":
                    print("Goodbye!")
                    break
                
                elif cmd[0] == "/help":
                    print("\nChat with the model by typing instructions like:")
                    print("  - Write a poem about AI")
                    print("  - Explain quantum computing in simple terms")
                    print("  - What are 5 ways to reduce stress?")
                    print()
                    continue
                
                elif cmd[0] == "/temp" and len(cmd) > 1:
                    try:
                        temperature = float(cmd[1])
                        temperature = max(0.1, min(2.0, temperature))
                        print(f"Temperature set to {temperature}")
                    except ValueError:
                        print("Invalid temperature. Use: /temp 0.8")
                    continue
                
                elif cmd[0] == "/tokens" and len(cmd) > 1:
                    try:
                        max_tokens = int(cmd[1])
                        max_tokens = max(10, min(1000, max_tokens))
                        print(f"Max tokens set to {max_tokens}")
                    except ValueError:
                        print("Invalid token count. Use: /tokens 300")
                    continue
                
                else:
                    print(f"Unknown command: {cmd[0]}")
                    continue

            # Generate response
            prompt = format_instruction_prompt(user_input)
            context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
            
            print("\nAssistant: ", end="", flush=True)
            
            with torch.no_grad():
                output = generate(model, context, max_new_tokens=max_tokens, temperature=temperature)
            
            # Decode and extract response after "### Response:\n"
            full_text = tokenizer.decode(output[0].tolist())
            
            # Extract only the generated response (after the prompt)
            if "### Response:\n" in full_text:
                response = full_text.split("### Response:\n", 1)[1].strip()
            else:
                response = full_text[len(prompt):].strip()
            
            print(response)
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


def single_mode(model, prompt="Hello"):
    """Single-shot generation with debug info."""
    print(f"\n{'='*60}")
    print("Single-shot generation mode")
    print(f"{'='*60}")
    print(f"Prompt: {prompt}\n")
    
    formatted_prompt = format_instruction_prompt(prompt)
    context = torch.tensor([tokenizer.encode(formatted_prompt)], dtype=torch.long).to(device)
    
    with torch.no_grad():
        out = generate(model, context, max_new_tokens=generate_tokens)

    tokens = out[0].tolist()
    unk_count = sum(1 for t in tokens if t == tokenizer.unk_id)
    invalid_tokens = [t for t in tokens if t >= vocab_size or t < 0]

    print(f"Vocab size: {vocab_size}")
    print(f"Unknown token ID: {tokenizer.unk_id}")
    print(f"Unknown tokens: {unk_count}/{len(tokens)} ({100*unk_count/len(tokens):.1f}%)")
    if invalid_tokens:
        print(f"WARNING: Invalid token IDs: {invalid_tokens[:10]}")
    print(f"Max token ID: {max(tokens)}")
    print(f"First 20 tokens: {tokens[:20]}\n")
    
    full_text = tokenizer.decode(tokens)
    print("Generated text:\n")
    print(full_text)
    print(f"\n{'='*60}")


def main():
    model = GPT(vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"✅ Model loaded (vocab={vocab_size}, device={device})")

    # Parse args
    mode = "chat"
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--mode", "-m"] and len(sys.argv) > 2:
            mode = sys.argv[2]
        elif sys.argv[1] in ["single", "chat"]:
            mode = sys.argv[1]

    if mode == "chat":
        chat_mode(model)
    else:
        prompt = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Write a short poem about artificial intelligence"
        single_mode(model, prompt)


if __name__ == "__main__":
    main()
