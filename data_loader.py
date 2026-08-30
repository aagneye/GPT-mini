"""
Memory-mapped batch loader for uint16 token shards produced by prepare_fineweb.py.

Each shard is a flat ``uint16`` array on disk. We ``np.memmap`` it (mode="r") so
the OS page cache serves reads without loading the whole corpus into RAM. A batch
is sampled by picking random start offsets, slicing ``block_size + 1`` tokens, and
casting only that small window to int64 for the model.

Per-rank behavior: each FSDP rank seeds its RNG with ``base_seed + rank`` so the
four GPUs draw disjoint-in-expectation batches, and shard selection is spread
across ranks. This is an IID sampler (not an epoch-exact iterator), which is the
standard approach for large pretraining corpora.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch


class ShardedTokenDataset:
    def __init__(self, data_dir: str, split: str = "train"):
        self.data_dir = data_dir
        self.split = split
        index_path = os.path.join(data_dir, "index.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Missing manifest {index_path}. Run prepare_fineweb.py first."
            )
        with open(index_path, "r", encoding="utf-8") as f:
            self.index = json.load(f)

        key = "val_shards" if split == "val" else "train_shards"
        shard_entries = self.index.get(key, [])
        if not shard_entries:
            raise ValueError(f"No shards for split={split!r} in {index_path}")

        self.shard_paths = [os.path.join(data_dir, e["file"]) for e in shard_entries]
        self.shard_tokens = [int(e["tokens"]) for e in shard_entries]
        self.total_tokens = sum(self.shard_tokens)
        # Lazily-opened memmaps, one per shard.
        self._memmaps: list[np.memmap | None] = [None] * len(self.shard_paths)

    def _get_memmap(self, shard_idx: int) -> np.memmap:
        mm = self._memmaps[shard_idx]
        if mm is None:
            mm = np.memmap(self.shard_paths[shard_idx], dtype=np.uint16, mode="r")
            self._memmaps[shard_idx] = mm
        return mm

    def num_shards(self) -> int:
        return len(self.shard_paths)


class DataLoader:
    def __init__(
        self,
        data_dir: str,
        split: str,
        batch_size: int,
        block_size: int,
        device: str = "cuda",
        rank: int = 0,
        world_size: int = 1,
        seed: int = 1337,
    ):
        self.dataset = ShardedTokenDataset(data_dir, split)
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.rank = rank
        self.world_size = world_size
        # Per-rank RNG so ranks sample different windows.
        self.rng = np.random.default_rng(seed + rank)

    def get_batch(self):
        bs, bl = self.batch_size, self.block_size
        n_shards = self.dataset.num_shards()
        x = np.empty((bs, bl), dtype=np.int64)
        y = np.empty((bs, bl), dtype=np.int64)

        for i in range(bs):
            shard_idx = int(self.rng.integers(0, n_shards))
            mm = self.dataset._get_memmap(shard_idx)
            hi = len(mm) - (bl + 1)
            if hi <= 0:
                # Degenerate tiny shard: fall back to shard 0.
                mm = self.dataset._get_memmap(0)
                hi = len(mm) - (bl + 1)
            start = int(self.rng.integers(0, hi + 1))
            window = np.asarray(mm[start : start + bl + 1], dtype=np.int64)
            x[i] = window[:-1]
            y[i] = window[1:]

        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y)
        if self.device.startswith("cuda"):
            xt = xt.pin_memory().to(self.device, non_blocking=True)
            yt = yt.pin_memory().to(self.device, non_blocking=True)
        else:
            xt = xt.to(self.device)
            yt = yt.to(self.device)
        return xt, yt
