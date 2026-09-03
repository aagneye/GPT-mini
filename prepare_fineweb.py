"""
Tokenize FineWeb-Edu (sample-10BT) into uint16 .bin shards for pretraining.

Streams ``HuggingFaceFW/fineweb-edu`` (name="sample-10BT"), tokenizes documents
in parallel with a process pool, appends an EOS token after each document, and
writes fixed-size shards of ``uint16`` tokens to ``data/fineweb_edu/``:

    data/fineweb_edu/train_000000.bin
    data/fineweb_edu/train_000001.bin
    ...
    data/fineweb_edu/val_000000.bin      (last shard held out for validation)
    data/fineweb_edu/index.json          (manifest: shard files + token counts)

~10.4B tokens at 32k vocab is ~20.8 GB on disk. uint16 requires vocab_size
<= 65536, which the 32k tokenizer satisfies.

Usage:
    GPT_SPM_MODEL=tokenizer/spm32k.model python prepare_fineweb.py \\
        --workers 60 --shard-size 100000000 --out data/fineweb_edu

The single-threaded loop in train_cloud.py would take 15+ hours for this corpus;
a 60-worker pool should finish in ~1-2 hours.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

# Set by each worker process in _init_worker so encoding avoids re-loading the
# SentencePiece model per document.
_worker_tokenizer = None
_worker_eos_id = None


def _init_worker(spm_model_path: str):
    global _worker_tokenizer, _worker_eos_id
    from tokenizer.tokenizer import SPTokenizer

    _worker_tokenizer = SPTokenizer(model_file=spm_model_path)
    _worker_eos_id = _worker_tokenizer.eos_id()


def _tokenize_doc(text: str) -> np.ndarray:
    """Encode one document and append EOS. Returns uint16 array."""
    ids = _worker_tokenizer.encode(text)
    ids.append(_worker_eos_id)
    arr = np.array(ids, dtype=np.uint16)
    return arr


class ShardWriter:
    """Accumulates tokens and flushes fixed-size uint16 shards to disk."""

    def __init__(self, out_dir: str, shard_size: int, split: str):
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.split = split
        self.buffer = np.empty(shard_size, dtype=np.uint16)
        self.fill = 0
        self.shard_index = 0
        self.manifest = []  # list of (filename, token_count)

    def _flush(self, count: int):
        filename = f"{self.split}_{self.shard_index:06d}.bin"
        path = os.path.join(self.out_dir, filename)
        self.buffer[:count].tofile(path)
        self.manifest.append({"file": filename, "tokens": int(count), "split": self.split})
        print(f"  wrote {path} ({count:,} tokens)", flush=True)
        self.shard_index += 1

    def add(self, arr: np.ndarray):
        pos = 0
        n = len(arr)
        while pos < n:
            space = self.shard_size - self.fill
            take = min(space, n - pos)
            self.buffer[self.fill : self.fill + take] = arr[pos : pos + take]
            self.fill += take
            pos += take
            if self.fill == self.shard_size:
                self._flush(self.shard_size)
                self.fill = 0

    def close(self):
        if self.fill > 0:
            self._flush(self.fill)
            self.fill = 0


def main():
    parser = argparse.ArgumentParser(description="Tokenize FineWeb-Edu into uint16 shards")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--name", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default="data/fineweb_edu")
    parser.add_argument("--shard-size", type=int, default=100_000_000,
                        help="tokens per shard (default 100M)")
    parser.add_argument("--workers", type=int, default=60)
    parser.add_argument("--spm-model", default=os.environ.get("GPT_SPM_MODEL", "tokenizer/spm32k.model"))
    parser.add_argument("--limit-docs", type=int, default=None,
                        help="optional cap on documents (for smoke tests)")
    parser.add_argument("--chunksize", type=int, default=64,
                        help="documents dispatched per worker task")
    args = parser.parse_args()

    if not os.path.exists(args.spm_model):
        raise FileNotFoundError(
            f"Tokenizer model not found: {args.spm_model}. Train the 32k tokenizer first "
            "(see tokenizer/tokenizer.py main docstring)."
        )

    os.makedirs(args.out, exist_ok=True)

    from datasets import load_dataset

    print(f"Streaming {args.dataset} ({args.name}) split={args.split} ...")
    ds = load_dataset(args.dataset, name=args.name, split=args.split, streaming=True)

    def doc_iter():
        for i, ex in enumerate(ds):
            if args.limit_docs is not None and i >= args.limit_docs:
                return
            text = ex.get("text")
            if text:
                yield text

    # Reserve the final shard as validation by writing train shards first, then
    # relabeling the last flushed shard as val at the end via the manifest.
    writer = ShardWriter(args.out, args.shard_size, split="train")

    import multiprocessing as mp
    import sys

    # "fork" after HuggingFace/datasets has spun up threads is unstable and can
    # silently kill the parent mid-stream. spawn is slower to start but safe.
    ctx = mp.get_context("spawn")

    total_tokens = 0
    total_docs = 0
    with ctx.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(args.spm_model,),
    ) as pool:
        for arr in pool.imap(_tokenize_doc, doc_iter(), chunksize=args.chunksize):
            writer.add(arr)
            total_tokens += len(arr)
            total_docs += 1
            if total_docs % 10_000 == 0:
                print(
                    f"  docs={total_docs:,} tokens={total_tokens:,}",
                    flush=True,
                )
                sys.stdout.flush()

    writer.close()

    manifest = writer.manifest
    # Hold out the last shard as validation.
    if len(manifest) > 1:
        last = manifest[-1]
        old_path = os.path.join(args.out, last["file"])
        new_file = last["file"].replace("train_", "val_")
        os.replace(old_path, os.path.join(args.out, new_file))
        last["file"] = new_file
        last["split"] = "val"

    index = {
        "dataset": args.dataset,
        "name": args.name,
        "vocab_dtype": "uint16",
        "shard_size": args.shard_size,
        "total_tokens": total_tokens,
        "total_docs": total_docs,
        "train_shards": [m for m in manifest if m["split"] == "train"],
        "val_shards": [m for m in manifest if m["split"] == "val"],
    }
    with open(os.path.join(args.out, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"\nDone. {total_docs:,} docs, {total_tokens:,} tokens across "
          f"{len(manifest)} shards -> {args.out}")
    print(f"Approx {total_tokens * 2 / 1e9:.1f} GB on disk (uint16).")


if __name__ == "__main__":
    main()
