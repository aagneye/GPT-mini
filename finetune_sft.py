"""
Supervised fine-tuning (SFT) on Dolly-15K + Alpaca.

Takes a pretrained checkpoint (from train_fsdp.py, carrying its GPTConfig) and
fine-tunes it to follow instructions, using the same prompt template as
generate.py / chatbot.py:

    ### Instruction:
    {instruction}

    ### Context:
    {optional context}

    ### Response:
    {response}<eos>

Loss is masked to the ### Response: span only, so the model is trained to
produce answers rather than to memorize the prompts. Runs 2 epochs at LR 2e-5.

Usage (single GPU is sufficient for a few hours of SFT):
    GPT_PROFILE=t4_1b python finetune_sft.py \\
        --init-checkpoint checkpoints_fsdp/step_19999 \\
        --spm-model tokenizer/spm32k.model \\
        --out checkpoints_sft

The pretrain uses FSDP sharded checkpoints; for SFT we consolidate to a single
full state dict first (see load_pretrained). If you already have a plain .pth
with model_state_dict + gpt_config, pass it directly and it will be used as-is.
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F

import config as cfg
from model.gpt import GPT, GPTConfig
from tokenizer.tokenizer import SPTokenizer

RESPONSE_MARKER = "### Response:\n"


def format_prompt(instruction: str, context: str = "") -> str:
    prompt = f"### Instruction:\n{instruction.strip()}\n\n"
    if context and context.strip():
        prompt += f"### Context:\n{context.strip()}\n\n"
    prompt += RESPONSE_MARKER
    return prompt


def load_sft_examples(limit=None):
    """Load Dolly-15K + Alpaca, normalized to (instruction, context, response)."""
    from datasets import load_dataset

    examples = []

    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    for ex in dolly:
        examples.append(
            (ex.get("instruction", ""), ex.get("context", ""), ex.get("response", ""))
        )

    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    for ex in alpaca:
        examples.append(
            (ex.get("instruction", ""), ex.get("input", ""), ex.get("output", ""))
        )

    if limit is not None:
        examples = examples[:limit]
    return examples


def build_masked_example(tokenizer, instruction, context, response, block_size):
    """
    Return (input_ids, target_ids) where target positions inside the prompt are
    set to -100 (ignored by cross_entropy) so loss is computed only on the
    response span (plus the trailing EOS).
    """
    prompt = format_prompt(instruction, context)
    prompt_ids = tokenizer.encode(prompt)
    response_ids = tokenizer.encode(response.strip())
    eos = tokenizer.eos_id()

    input_ids = prompt_ids + response_ids + [eos]
    # Labels: ignore the prompt tokens, learn the response tokens + EOS.
    labels = [-100] * len(prompt_ids) + response_ids + [eos]

    # Truncate to block_size (keep the start so the instruction survives).
    input_ids = input_ids[:block_size]
    labels = labels[:block_size]
    return input_ids, labels


def collate(batch, pad_id, block_size):
    maxlen = min(block_size, max(len(x[0]) for x in batch))
    input_batch, label_batch = [], []
    for input_ids, labels in batch:
        input_ids = input_ids[:maxlen]
        labels = labels[:maxlen]
        pad = maxlen - len(input_ids)
        input_batch.append(input_ids + [pad_id] * pad)
        label_batch.append(labels + [-100] * pad)
    x = torch.tensor(input_batch, dtype=torch.long)
    y = torch.tensor(label_batch, dtype=torch.long)
    return x, y


def load_pretrained(init_checkpoint, device):
    """
    Load a pretrained model + GPTConfig.

    Supports either a plain .pth (dict with 'model_state_dict' and 'gpt_config')
    or an FSDP sharded checkpoint directory, which is consolidated to a full
    state dict on CPU first.
    """
    if os.path.isdir(init_checkpoint):
        import torch.distributed.checkpoint as dcp

        # A sharded dir must also carry the config; we expect a manifest sibling.
        manifest = os.path.join(os.path.dirname(init_checkpoint), "latest.json")
        if not os.path.exists(manifest):
            raise FileNotFoundError(
                f"Expected manifest {manifest} next to sharded checkpoint {init_checkpoint}"
            )
        import json

        with open(manifest, "r", encoding="utf-8") as f:
            gpt_config = GPTConfig.from_dict(json.load(f)["gpt_config"])
        model = GPT(gpt_config)
        state = {"model": model.state_dict()}
        dcp.load(state, checkpoint_id=init_checkpoint)
        model.load_state_dict(state["model"])
    else:
        ckpt = torch.load(init_checkpoint, map_location="cpu", weights_only=True)
        gpt_config = GPTConfig.from_dict(ckpt["gpt_config"])
        model = GPT(gpt_config)
        model.load_state_dict(ckpt["model_state_dict"])

    return model.to(device), gpt_config


def build_optimizer(model, lr, weight_decay, beta1, beta2):
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(beta1, beta2))


def main():
    parser = argparse.ArgumentParser(description="SFT on Dolly + Alpaca")
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--spm-model", default=os.environ.get("GPT_SPM_MODEL", "tokenizer/spm32k.model"))
    parser.add_argument("--out", default="checkpoints_sft")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="cap examples (smoke test)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    tokenizer = SPTokenizer(model_file=args.spm_model)
    pad_id = tokenizer.sp.pad_id()
    block_size = cfg.block_size

    print("Loading SFT dataset (Dolly + Alpaca)...")
    raw = load_sft_examples(limit=args.limit)
    examples = [
        build_masked_example(tokenizer, ins, ctx, resp, block_size)
        for (ins, ctx, resp) in raw
        if resp and resp.strip()
    ]
    print(f"Prepared {len(examples):,} SFT examples.")

    model, gpt_config = load_pretrained(args.init_checkpoint, device)
    model.train()
    optimizer = build_optimizer(model, args.lr, cfg.weight_decay, cfg.beta1, cfg.beta2)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    steps_per_epoch = math.ceil(len(examples) / (args.batch_size * args.grad_accum))
    print(f"Fine-tuning {args.epochs} epochs, ~{steps_per_epoch} optimizer steps/epoch, lr={args.lr}")

    rng = torch.Generator().manual_seed(1337)
    global_step = 0
    for epoch in range(args.epochs):
        perm = torch.randperm(len(examples), generator=rng).tolist()
        i = 0
        while i < len(perm):
            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            for _ in range(args.grad_accum):
                batch_idx = perm[i : i + args.batch_size]
                if not batch_idx:
                    break
                i += args.batch_size
                batch = [examples[j] for j in batch_idx]
                x, y = collate(batch, pad_id, block_size)
                x, y = x.to(device), y.to(device)
                with torch.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                    logits, _ = model(x, targets=None)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100
                    )
                    loss = loss / args.grad_accum
                scaler.scale(loss).backward()
                loss_accum += loss.item()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            if global_step % 50 == 0:
                print(f"epoch {epoch} step {global_step} loss {loss_accum:.4f}")

        # Save after each epoch as a plain checkpoint carrying its GPTConfig.
        out_path = os.path.join(args.out, f"sft_epoch_{epoch + 1}.pth")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "gpt_config": gpt_config.to_dict(),
                "vocab_size": gpt_config.vocab_size,
                "epoch": epoch + 1,
            },
            out_path,
        )
        print(f"Saved {out_path}")

    print("SFT complete.")


if __name__ == "__main__":
    main()
