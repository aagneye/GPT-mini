"""
GPT-mini Chatbot using step_20000.pth model checkpoint.

Automatically configures the model architecture to match the trained checkpoint.

Usage:
  python chatbot.py                           # Interactive chat mode
  python chatbot.py -i "Your question here"   # Single query mode
  python chatbot.py --temp 0.7                # Adjust temperature
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Set environment variables BEFORE importing config
# These match the step_20000.pth training configuration
os.environ["GPT_N_EMBD"] = "768"
os.environ["GPT_N_HEAD"] = "12"
os.environ["GPT_N_LAYER"] = "12"
os.environ["GPT_BLOCK_SIZE"] = "512"

from config import (
    block_size,
    data_path,
    device,
    generate_tokens,
    spm_model_path,
)
from model.gpt import GPT, GPTConfig
from tokenizer.tokenizer import SPTokenizer


def load_checkpoint_model(checkpoint_path):
    """Load a specific checkpoint model."""
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"[*] Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    vocab_sz = checkpoint["vocab_size"]
    tok = SPTokenizer(model_file=spm_model_path, data_path=data_path)

    if vocab_sz != tok.vocab_size:
        raise ValueError(
            f"Tokenizer/model vocab mismatch: checkpoint={vocab_sz}, tokenizer={tok.vocab_size}"
        )

    # New checkpoints carry their own GPTConfig so the model knows its own
    # shape. Legacy checkpoints (e.g. step_20000.pth) predate both the config
    # and the fused architecture, so load them with the legacy GPT; the GPT_N_*
    # env vars above still pin the legacy 768/12/12 shape for those.
    gpt_config_dict = checkpoint.get("gpt_config")
    if gpt_config_dict is not None:
        cfg = GPTConfig.from_dict(gpt_config_dict)
        mdl = GPT(cfg).to(device)
    else:
        from model.gpt_legacy import GPT as LegacyGPT

        mdl = LegacyGPT(vocab_sz).to(device)
    mdl.load_state_dict(checkpoint["model_state_dict"])
    mdl.eval()
    
    return mdl, tok, vocab_sz


def generate_response(
    model,
    tokenizer,
    prompt_text,
    max_tokens=150,
    temperature=0.8,
    greedy=False,
):
    """Generate a response from the model."""
    
    # Format prompt
    prompt = f"### Instruction:\n{prompt_text.strip()}\n\n### Response:\n"
    token_ids = tokenizer.encode(prompt)
    context = torch.tensor([token_ids], dtype=torch.long).to(device)
    
    # Generate
    with torch.no_grad():
        for gen_step in range(max_tokens):
            context_cond = context[:, -block_size:]
            logits, _ = model(context_cond)
            logits_last = logits[:, -1, :].clone()
            
            if greedy:
                idx_next = logits_last.argmax(dim=-1, keepdim=True)
            else:
                logits_last = logits_last / temperature
                probs = torch.softmax(logits_last, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            
            context = torch.cat((context, idx_next), dim=1)
            
            # Early stopping if we generate another instruction
            if gen_step > 10:
                decoded = tokenizer.decode(context[0].tolist())
                if decoded.count("### Instruction:") > 1:
                    break
    
    # Decode and extract response
    output_tokens = context[0].tolist()
    output_text = tokenizer.decode(output_tokens)
    
    if "### Response:\n" in output_text:
        response = output_text.split("### Response:\n")[1]
        if "### Instruction:" in response:
            response = response.split("### Instruction:")[0]
        response = response.strip()
    else:
        response = output_text[len(prompt):].strip()
    
    return response


def interactive_chat(model, tokenizer, vocab_size, temperature=0.8, greedy=False):
    """Run interactive chat mode."""
    print("\n" + "="*70)
    print("GPT-Mini Chatbot (Powered by step_20000.pth)")
    print("="*70)
    print("Type your question and press Enter to get a response.")
    print("Commands: /quit, /temp N (set temperature), /greedy (toggle greedy mode)")
    print("="*70 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower().split()
                if cmd[0] == "/quit":
                    print("Goodbye!")
                    break
                elif cmd[0] == "/temp" and len(cmd) > 1:
                    temperature = max(0.1, min(2.0, float(cmd[1])))
                    print(f"[*] Temperature set to {temperature}")
                    continue
                elif cmd[0] == "/greedy":
                    greedy = not greedy
                    mode = "greedy (deterministic)" if greedy else "sampling"
                    print(f"[*] Decoding mode: {mode}")
                    continue
                else:
                    print("[!] Unknown command. Try /help, /quit, /temp N, or /greedy")
                    continue
            
            # Generate response
            print("\nAssistant: ", end="", flush=True)
            response = generate_response(
                model,
                tokenizer,
                user_input,
                max_tokens=200,
                temperature=temperature,
                greedy=greedy,
            )
            print(response)
            print()
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"[!] Error: {e}")
            continue


def single_query(model, tokenizer, prompt_text, temperature=0.8, greedy=False):
    """Run a single query."""
    response = generate_response(
        model,
        tokenizer,
        prompt_text,
        max_tokens=200,
        temperature=temperature,
        greedy=greedy,
    )
    print(response)


def main():
    parser = argparse.ArgumentParser(
        description="GPT-Mini Chatbot using step_20000.pth checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chatbot.py                              # Interactive chat
  python chatbot.py -i "What is machine learning?"
  python chatbot.py -i "Explain AI" --temp 0.5
  python chatbot.py --checkpoint models/step_20000.pth
  python chatbot.py -i "Hello" --greedy         # Deterministic mode
""",
    )
    parser.add_argument(
        "-i", "--instruction",
        type=str,
        help="Single query mode: ask a question",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/step_20000.pth",
        help="Path to model checkpoint (default: models/step_20000.pth)",
    )
    parser.add_argument(
        "--temp", "--temperature",
        type=float,
        default=0.8,
        dest="temperature",
        help="Temperature for sampling (default: 0.8)",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding (deterministic)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate (default: 200)",
    )
    
    args = parser.parse_args()
    
    # Load model
    try:
        model, tokenizer, vocab_size = load_checkpoint_model(args.checkpoint)
        print(f"[+] Model loaded successfully!")
        print(f"[+] Vocab size: {vocab_size}")
        print(f"[+] Device: {device}")
        print(f"[+] Architecture: 768d, 12 heads, 12 layers\n")
    except Exception as e:
        print(f"[!] Failed to load model: {e}")
        sys.exit(1)
    
    # Run chat or single query
    if args.instruction:
        single_query(
            model,
            tokenizer,
            args.instruction,
            temperature=args.temperature,
            greedy=args.greedy,
        )
    else:
        interactive_chat(
            model,
            tokenizer,
            vocab_size,
            temperature=args.temperature,
            greedy=args.greedy,
        )


if __name__ == "__main__":
    main()
