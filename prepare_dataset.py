"""
Download and prepare combined Alpaca + Dolly-15K dataset for GPT training.

Why this combination:
- Alpaca: 52K examples, GPT-3.5 generated, diverse instructions
- Dolly-15K: 15K examples, human-written by Databricks employees, high quality
- Total: 67K instruction-following examples
- Both are open-source and commercially usable

Usage: python prepare_dataset.py
"""

from datasets import load_dataset
import os


def format_dolly_example(example):
    """Format Dolly-15K example to match Alpaca's instruction format."""
    instruction = example["instruction"]
    context = example.get("context", "").strip()
    response = example["response"]
    
    # Build prompt similar to Alpaca format
    if context:
        # Has context (like Wikipedia passage for summarization/QA)
        prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n{response}"
    else:
        # No context (like creative writing, brainstorming)
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}"
    
    return prompt


def main():
    print("=" * 60)
    print("Loading datasets from Hugging Face...")
    print("=" * 60)
    
    # Load Alpaca
    print("\n[1/2] Loading Alpaca dataset...")
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    print(f"  ✓ Alpaca: {len(alpaca):,} examples")
    
    # Load Dolly-15K
    print("\n[2/2] Loading Dolly-15K dataset...")
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    print(f"  ✓ Dolly-15K: {len(dolly):,} examples")
    
    print(f"\n{'='*60}")
    print(f"Total: {len(alpaca) + len(dolly):,} examples")
    print(f"{'='*60}")
    
    # Extract formatted texts
    print("\nFormatting examples...")
    
    # Alpaca already has 'text' field with proper formatting
    alpaca_texts = [example["text"] for example in alpaca]
    print(f"  ✓ Formatted {len(alpaca_texts):,} Alpaca examples")
    
    # Format Dolly examples to match Alpaca structure
    dolly_texts = [format_dolly_example(example) for example in dolly]
    print(f"  ✓ Formatted {len(dolly_texts):,} Dolly examples")
    
    # Combine datasets
    all_texts = alpaca_texts + dolly_texts
    print(f"\n  → Combined: {len(all_texts):,} examples")
    
    # Join with double newline separator
    full_text = "\n\n".join(all_texts)
    
    # Save to data/dataset.txt
    os.makedirs("data", exist_ok=True)
    output_path = "data/dataset.txt"
    
    print(f"\nWriting to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    # Statistics
    total_chars = len(full_text)
    avg_chars = total_chars // len(all_texts)
    size_mb = total_chars / (1024 * 1024)
    
    print(f"\n{'='*60}")
    print("✅ Dataset prepared successfully!")
    print(f"{'='*60}")
    print(f"  File: {output_path}")
    print(f"  Examples: {len(all_texts):,}")
    print(f"    - Alpaca: {len(alpaca_texts):,}")
    print(f"    - Dolly: {len(dolly_texts):,}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Characters: {total_chars:,}")
    print(f"  Avg per example: {avg_chars:,} chars")
    print(f"{'='*60}")
    
    print("\n📋 Next steps:")
    print("  1. Train tokenizer: python -m tokenizer")
    print("  2. Train model: python train.py")
    print("  3. Generate text: python generate.py")


if __name__ == "__main__":
    main()
