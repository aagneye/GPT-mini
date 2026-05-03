"""
Download and prepare Dolly-15K + Alpaca datasets in clean instruction format.

Format used (no verbose preamble):
    ### Instruction:
    <task description>
    
    ### Context:  (optional, only if context/input exists)
    <additional context>
    
    ### Response:
    <answer>

Why this combination:
- Dolly-15K: 15K human-written, high-quality, 8 task categories
- Alpaca: 52K GPT-3.5 generated, diverse instructions
- Total: 67K examples
- Clean format improves instruction-following

Usage: python prepare_dataset.py
"""

from datasets import load_dataset
import os


def format_dolly_example(example):
    """Format Dolly-15K with clean ### structure."""
    instruction = example["instruction"].strip()
    context = example.get("context", "").strip()
    response = example["response"].strip()
    
    prompt = f"### Instruction:\n{instruction}\n\n"
    
    if context:
        prompt += f"### Context:\n{context}\n\n"
    
    prompt += f"### Response:\n{response}"
    
    return prompt


def format_alpaca_example(example):
    """Format Alpaca with clean ### structure (extract from verbose format)."""
    instruction = example["instruction"].strip()
    input_text = example.get("input", "").strip()
    output = example["output"].strip()
    
    prompt = f"### Instruction:\n{instruction}\n\n"
    
    if input_text:
        prompt += f"### Context:\n{input_text}\n\n"
    
    prompt += f"### Response:\n{output}"
    
    return prompt


def main():
    print("=" * 70)
    print("Loading datasets from Hugging Face...")
    print("=" * 70)
    
    # Load Dolly-15K (prioritize: human-written, cleaner)
    print("\n[1/2] Loading Dolly-15K dataset...")
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    print(f"  ✓ Dolly-15K: {len(dolly):,} examples (human-written)")
    
    # Load Alpaca
    print("\n[2/2] Loading Alpaca dataset...")
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    print(f"  ✓ Alpaca: {len(alpaca):,} examples (GPT-3.5 generated)")
    
    print(f"\n{'='*70}")
    print(f"Total: {len(dolly) + len(alpaca):,} examples")
    print(f"{'='*70}")
    
    # Format examples with clean ### structure
    print("\nFormatting examples (clean ### Instruction/Response format)...")
    
    dolly_texts = [format_dolly_example(ex) for ex in dolly]
    print(f"  ✓ Formatted {len(dolly_texts):,} Dolly examples")
    
    alpaca_texts = [format_alpaca_example(ex) for ex in alpaca]
    print(f"  ✓ Formatted {len(alpaca_texts):,} Alpaca examples")
    
    # Strategy: Dolly first (higher quality), then Alpaca (diversity)
    all_texts = dolly_texts + alpaca_texts
    print(f"\n  → Combined: {len(all_texts):,} examples (Dolly + Alpaca)")
    
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
    
    print(f"\n{'='*70}")
    print("✅ Dataset prepared successfully!")
    print(f"{'='*70}")
    print(f"  Format: ### Instruction / ### Context / ### Response")
    print(f"  File: {output_path}")
    print(f"  Examples: {len(all_texts):,}")
    print(f"    - Dolly-15K: {len(dolly_texts):,} (human, high-quality)")
    print(f"    - Alpaca: {len(alpaca_texts):,} (synthetic, diverse)")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Characters: {total_chars:,}")
    print(f"  Avg per example: {avg_chars:,} chars")
    print(f"{'='*70}")
    
    # Show sample
    print("\n📄 Sample (first Dolly example):")
    print("-" * 70)
    print(dolly_texts[0][:500] + "..." if len(dolly_texts[0]) > 500 else dolly_texts[0])
    print("-" * 70)
    
    print("\n📋 Next steps:")
    print("  1. Train tokenizer: python -m tokenizer")
    print("  2. Train model: python train.py")
    print("  3. Chat: python generate.py")


if __name__ == "__main__":
    main()
