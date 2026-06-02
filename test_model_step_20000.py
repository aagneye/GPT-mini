"""
Test script to load and run inference on the step_20000.pth model.
This model was trained with the cloud config: n_embd=768, n_head=12, n_layer=12
"""

import torch
from pathlib import Path

# Import everything from config but override the architecture to match the checkpoint
import sys
import os

# Set environment to match checkpoint config
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
    n_embd,
    n_head,
    n_layer,
)
from model.gpt import GPT
from tokenizer.tokenizer import SPTokenizer


def load_checkpoint_model(checkpoint_path):
    """Load a specific checkpoint model."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    vocab_sz = checkpoint["vocab_size"]
    tok = SPTokenizer(model_file=spm_model_path, data_path=data_path)

    if vocab_sz != tok.vocab_size:
        raise ValueError(
            f"Tokenizer/model vocab mismatch: checkpoint={vocab_sz}, tokenizer={tok.vocab_size}"
        )

    print(f"Vocab size: {vocab_sz}")
    print(f"Device: {device}")
    print(f"Model architecture: n_embd={n_embd}, n_head={n_head}, n_layer={n_layer}")
    
    mdl = GPT(vocab_sz).to(device)
    mdl.load_state_dict(checkpoint["model_state_dict"])
    mdl.eval()
    
    return mdl, tok, vocab_sz


def test_inference(model, tokenizer, vocab_size, prompt_text, max_tokens=100):
    """Run a simple inference test."""
    
    # Format the prompt
    prompt = f"### Instruction:\n{prompt_text.strip()}\n\n### Response:\n"
    
    print(f"\n{'='*60}")
    print("INFERENCE TEST")
    print(f"{'='*60}")
    print(f"\nPrompt:\n{prompt}")
    
    # Encode
    token_ids = tokenizer.encode(prompt)
    context = torch.tensor([token_ids], dtype=torch.long).to(device)
    
    print(f"Prompt tokens: {len(token_ids)}")
    print(f"Context shape: {context.shape}")
    
    # Generate
    print("\nGenerating...")
    with torch.no_grad():
        for i in range(max_tokens):
            context_cond = context[:, -block_size:]
            logits, _ = model(context_cond)
            logits_last = logits[:, -1, :].clone()
            
            # Apply softmax and sample
            probs = torch.softmax(logits_last / 0.8, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_token), dim=1)
            
            if i % 30 == 0:
                print(f"  Generated {i+1} tokens...")
    
    # Decode
    output_tokens = context[0].tolist()
    output_text = tokenizer.decode(output_tokens)
    
    # Extract just the response part
    if "### Response:\n" in output_text:
        response = output_text.split("### Response:\n")[1]
        if "### Instruction:" in response:
            response = response.split("### Instruction:")[0]
        response = response.strip()
    else:
        response = output_text[len(prompt):].strip()
    
    print(f"\nModel Response:\n{response}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # Test prompts
    test_prompts = [
        "Explain machine learning in simple terms",
        "Write a short poem about AI",
        "What is Python used for?"
    ]
    
    # Load the checkpoint model
    checkpoint_path = Path("models/step_20000.pth")
    model, tokenizer, vocab_size = load_checkpoint_model(checkpoint_path)
    
    print(f"[SUCCESS] Model loaded!")
    print(f"Model parameters: ~{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Run inference tests
    for i, prompt in enumerate(test_prompts[:1], 1):  # Test with first prompt
        print(f"\n[Test {i}/{len(test_prompts[:1])}]")
        test_inference(model, tokenizer, vocab_size, prompt, max_tokens=100)
