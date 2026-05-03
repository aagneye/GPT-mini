"""
GPT-mini inference: chat loop or single-shot generation.

Notebook-friendly (no fragile multiline shell strings):
  python generate.py -i "Explain constellations in simple words"
  python generate.py --prompt-file prompt.txt
  python generate.py --stdin < prompt.txt

Legacy:
  python generate.py single words joined as instruction
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

from config import (
    block_size,
    data_path,
    device,
    generate_tokens,
    spm_model_path,
)
from model.gpt import GPT
from tokenizer.tokenizer import SPTokenizer


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def format_instruction_prompt(instruction: str, context: str = "") -> str:
    """Training-aligned prompt (### Instruction / optional ### Context / ### Response)."""
    prompt = f"### Instruction:\n{instruction.strip()}\n\n"
    if context.strip():
        prompt += f"### Context:\n{context.strip()}\n\n"
    prompt += "### Response:\n"
    return prompt


def normalize_inference_prompt(text: str) -> str:
    """
    If text already contains ### Instruction (full template), ensure it ends with
    ### Response:\\n so the model continues the answer. Otherwise wrap as instruction-only.
    """
    text = text.strip()
    if "### Instruction:" in text:
        if "### Response:" not in text:
            text = text.rstrip() + "\n\n### Response:\n"
        else:
            tail = text.rstrip()
            if tail.endswith("### Response:"):
                text = tail + "\n"
        return text
    return format_instruction_prompt(text)


def decode_reply(full_text: str, prompt: str) -> str:
    """Extract only the first response, stop at next instruction."""
    if "### Response:\n" in full_text:
        after_response = full_text.split("### Response:\n", 1)[1]
        # If another instruction started, take only text before it
        if "### Instruction:" in after_response:
            after_response = after_response.split("### Instruction:")[0]
        return after_response.strip()
    return full_text[len(prompt):].strip()


def generate_tokens_autoreg(
    model,
    idx,
    max_new_tokens,
    temperature=0.8,
    top_k=40,
    stop_on_next_instruction=True,
    greedy=False,
):
    """
    Generate tokens autoregressively.
    
    Args:
        stop_on_next_instruction: If True, stop when '### Instruction:' appears in decoded output
                                  (prevents generating multiple instruction/response pairs).
    """
    banned_ids = [tokenizer.unk_id]
    if tokenizer.sp.bos_id() >= 0:
        banned_ids.append(tokenizer.sp.bos_id())
    if tokenizer.sp.pad_id() >= 0:
        banned_ids.append(tokenizer.sp.pad_id())

    generated_count = 0
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)

        logits_last = logits[:, -1, :].clone()
        for token_id in banned_ids:
            logits_last[:, token_id] = float("-inf")

        if greedy:
            idx_next = logits_last.argmax(dim=-1, keepdim=True)
        else:
            logits_last = logits_last / temperature
            probs = torch.softmax(logits_last, dim=-1)
            values, indices = torch.topk(probs, top_k)
            probs = torch.zeros_like(probs).scatter_(1, indices, values)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        generated_count += 1

        # Early stopping: check if we generated "### Instruction:" after the initial prompt
        if stop_on_next_instruction and generated_count > 5:  # Wait a few tokens before checking
            decoded = tokenizer.decode(idx[0].tolist())
            # Count occurrences of "### Instruction:"
            instruction_count = decoded.count("### Instruction:")
            if instruction_count > 1:  # More than the original prompt
                # Truncate back to just before the second instruction
                parts = decoded.split("### Instruction:")
                # Keep original instruction + first response, discard rest
                truncated = "### Instruction:".join(parts[:2])
                # Re-encode to get proper token count
                truncated_tokens = tokenizer.encode(truncated)
                idx = torch.tensor([truncated_tokens], dtype=torch.long, device=idx.device)
                break

    return idx


def load_model_and_tokenizer():
    checkpoint = torch.load("model.pth", map_location=device, weights_only=True)
    vocab_sz = checkpoint["vocab_size"]
    tok = SPTokenizer(model_file=spm_model_path, data_path=data_path)

    if vocab_sz != tok.vocab_size:
        raise ValueError(
            f"Tokenizer/model vocab mismatch: checkpoint={vocab_sz}, tokenizer={tok.vocab_size}"
        )

    expected_hash = checkpoint.get("tokenizer_model_sha256")
    if expected_hash is not None:
        current_hash = file_sha256(tok.model_file)
        if expected_hash != current_hash:
            raise ValueError(
                "Tokenizer mismatch: checkpoint tokenizer hash does not match current spm.model."
            )

    mdl = GPT(vocab_sz).to(device)
    mdl.load_state_dict(checkpoint["model_state_dict"])
    mdl.eval()
    return mdl, tok, vocab_sz


tokenizer = None


def chat_mode(model, vocab_size):
    """Interactive chat loop."""
    print("\n" + "=" * 60)
    print("GPT Chat Assistant (Dolly-15K + Alpaca trained)")
    print("=" * 60)
    print("Type your instruction/question and press Enter.")
    print("Commands: /help  /temp N  /tokens N  /quit")
    print("=" * 60 + "\n")

    temperature = 0.8
    max_tokens = generate_tokens

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            if user_input.startswith("/"):
                cmd = user_input.lower().split()
                if cmd[0] == "/quit":
                    print("Goodbye!")
                    break
                if cmd[0] == "/help":
                    print("\nExamples: Write a poem about AI | Explain quantum physics simply\n")
                    continue
                if cmd[0] == "/temp" and len(cmd) > 1:
                    temperature = max(0.1, min(2.0, float(cmd[1])))
                    print(f"Temperature set to {temperature}")
                    continue
                if cmd[0] == "/tokens" and len(cmd) > 1:
                    max_tokens = max(10, min(1000, int(cmd[1])))
                    print(f"Max tokens set to {max_tokens}")
                    continue
                print(f"Unknown command: {cmd[0]}")
                continue

            prompt = format_instruction_prompt(user_input)
            context_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)

            print("\nAssistant: ", end="", flush=True)
            with torch.no_grad():
                output = generate_tokens_autoreg(
                    model,
                    context_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    greedy=False,
                )

            full_text = tokenizer.decode(output[0].tolist())
            print(decode_reply(full_text, prompt))
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


def run_single_inference(
    model,
    vocab_size,
    prompt_text: str,
    *,
    temperature: float,
    max_new_tokens: int,
    reply_only: bool,
    quiet: bool,
    greedy: bool,
):
    prompt_text = normalize_inference_prompt(prompt_text)
    context_ids = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long).to(device)

    with torch.no_grad():
        out = generate_tokens_autoreg(
            model,
            context_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            greedy=greedy,
        )

    tokens = out[0].tolist()
    full_text = tokenizer.decode(tokens)
    reply = decode_reply(full_text, prompt_text)

    if reply_only:
        print(reply)
        return

    if not quiet:
        unk_count = sum(1 for t in tokens if t == tokenizer.unk_id)
        invalid_tokens = [t for t in tokens if t >= vocab_size or t < 0]
        print(f"\n{'=' * 60}")
        print("Single-shot inference")
        print(f"{'=' * 60}")
        print(f"Prompt (normalized):\n{prompt_text}\n")
        print(f"Vocab size: {vocab_size}")
        print(f"Unknown tokens: {unk_count}/{len(tokens)} ({100 * unk_count / len(tokens):.1f}%)")
        if invalid_tokens:
            print(f"WARNING: Invalid token IDs: {invalid_tokens[:10]}")
        print(f"Max token ID: {max(tokens)}")
        print(f"First 20 tokens: {tokens[:20]}\n")

    print("Assistant reply:\n")
    print(reply)
    if not quiet:
        print(f"\n{'=' * 60}")


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="GPT-mini inference (chat or single-shot).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate.py
  python generate.py -i "Explain constellations in simple words"
  python generate.py --prompt-file prompt.txt --reply-only
  python generate.py single Explain stars in simple words
""",
    )
    p.add_argument(
        "mode",
        nargs="?",
        default="chat",
        choices=["chat", "single"],
        help='chat (default) or single (legacy: remaining words = instruction)',
    )
    p.add_argument(
        "-i",
        "--instruction",
        help="Instruction text (wrapped as ### Instruction / ### Response). Best for notebooks.",
    )
    p.add_argument("-c", "--context", default="", help="Optional ### Context block.")
    p.add_argument(
        "-f",
        "--prompt-file",
        type=Path,
        metavar="PATH",
        help="UTF-8 file with instruction only or full ### template.",
    )
    p.add_argument(
        "--stdin",
        action="store_true",
        help="Read prompt from stdin (multiline OK).",
    )
    p.add_argument(
        "--reply-only",
        action="store_true",
        help="Print only the assistant reply (no debug stats).",
    )
    p.add_argument("-q", "--quiet", action="store_true", help="Less debug output for single-shot.")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument(
        "--greedy",
        action="store_true",
        help="Deterministic argmax decoding (often steadier on small / under-trained models).",
    )
    p.add_argument("--max-new-tokens", type=int, default=None, metavar="N")
    p.add_argument(
        "legacy_words",
        nargs="*",
        help="After 'single': words joined as instruction (avoid multiline).",
    )
    return p.parse_args(argv)


def resolve_prompt(args) -> str | None:
    if args.instruction is not None:
        return format_instruction_prompt(args.instruction, args.context)
    if args.prompt_file is not None:
        path = args.prompt_file
        if not path.is_file():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8")
    if args.stdin:
        return sys.stdin.read()
    if args.mode == "single" and args.legacy_words:
        return format_instruction_prompt(" ".join(args.legacy_words))
    if args.mode == "single":
        return format_instruction_prompt(
            "Write a short poem about artificial intelligence."
        )
    return None


def main(argv=None):
    global tokenizer

    argv = argv if argv is not None else sys.argv[1:]
    args = parse_args(argv)

    max_nt = args.max_new_tokens if args.max_new_tokens is not None else generate_tokens

    model, tok, vocab_size = load_model_and_tokenizer()
    tokenizer = tok
    print(f"Model loaded (vocab={vocab_size}, device={device})")

    prompt = resolve_prompt(args)

    if prompt is None:
        chat_mode(model, vocab_size)
        return

    run_single_inference(
        model,
        vocab_size,
        prompt,
        temperature=args.temperature,
        max_new_tokens=max_nt,
        reply_only=args.reply_only,
        quiet=args.quiet or args.reply_only,
        greedy=args.greedy,
    )


if __name__ == "__main__":
    main()
