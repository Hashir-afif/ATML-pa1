"""The PACS protocol shared by Tasks 2 and 3: splits, domain-balanced sampling, preprocessing.

* Split: stratified 80/20 train/validation **within each source domain**, seed 6304, saved to
  ``shared/splits/pacs_sketch_seed6304.json`` and reused unchanged by Task 3.
* Sampling: every update draws ``n_per_source`` (8) images from each source domain; adaptation
  methods additionally draw ``n_target`` (24) unlabelled target images. Each domain is an endless
  stream of seeded permutations ("cycle a loader when necessary"). Source streams, target stream and
  the two augmentation generators are **independent**, so the source images *and* their crops/flips
  are identical for every method, whether or not it draws target batches.
* One epoch = ceil(largest source-train domain / n_per_source) updates (user decision 2026-09-22).
* Preprocessing: images are cached at 256x256; training = random 224x224 crop + horizontal flip,
  evaluation = centre 224x224 crop; ImageNet normalisation of ResNet18_Weights.IMAGENET1K_V1.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision.models import ResNet18_Weights

from shared import pacs

SEED = 6304
SPLIT_PATH = pacs.REPO / "shared" / "splits" / "pacs_sketch_seed6304.json"
CROP = 224
_W = ResNet18_Weights.IMAGENET1K_V1.transforms()
MEAN = torch.tensor(_W.mean).view(1, 3, 1, 1)
STD = torch.tensor(_W.std).view(1, 3, 1, 1)


# ------------------------------------------------------------------ splits
def make_source_splits(val_frac: float = 0.2, seed: int = SEED, overwrite: bool = False) -> dict:
    """Stratified per-domain 80/20 split of the three source domains. Never touches the target."""
    if SPLIT_PATH.exists() and not overwrite:
        return load_splits()
    splits = {"seed": seed, "val_frac": val_frac, "target": pacs.TARGET_DOMAIN, "domains": {}}
    for dom in pacs.SOURCE_DOMAINS:
        _, y = pacs.load_source_domain(dom)
        idx = np.arange(len(y))
        tr, va = train_test_split(idx, test_size=val_frac, stratify=y.numpy(), random_state=seed)
        splits["domains"][dom] = {"train_idx": sorted(int(i) for i in tr), "val_idx": sorted(int(i) for i in va),
                                  "n_train": int(len(tr)), "n_val": int(len(va))}
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(json.dumps(splits, indent=1))
    return splits


def load_splits() -> dict:
    return json.loads(SPLIT_PATH.read_text())


def load_source_data() -> dict:
    """{'train': {dom: (x_u8, y)}, 'val': {dom: (x_u8, y)}} for the three source domains."""
    splits = load_splits()
    out = {"train": {}, "val": {}}
    for dom in pacs.SOURCE_DOMAINS:
        x, y = pacs.load_source_domain(dom)
        for part in ("train", "val"):
            idx = torch.as_tensor(splits["domains"][dom][f"{part}_idx"])
            out[part][dom] = (x[idx], y[idx])
    return out


def iters_per_epoch(train_sizes: dict, n_per_source: int) -> int:
    return math.ceil(max(train_sizes.values()) / n_per_source)


# ------------------------------------------------------------------ sampling
class IndexStream:
    """Endless stream of indices 0..n-1: successive seeded permutations, concatenated."""

    def __init__(self, n: int, seed: int):
        self.n, self.g = n, np.random.default_rng(seed)
        self.buf, self.pos = self.g.permutation(n), 0

    def next(self, k: int) -> np.ndarray:
        out = []
        while k > 0:
            if self.pos == self.n:
                self.buf, self.pos = self.g.permutation(self.n), 0
            take = min(k, self.n - self.pos)
            out.append(self.buf[self.pos:self.pos + take])
            self.pos += take
            k -= take
        return np.concatenate(out)


class BatchSampler:
    """Domain-balanced source batches (+ optional target batches) with independent seeded streams."""

    def __init__(self, source_sizes: dict, n_per_source: int, target_size: int | None = None,
                 n_target: int = 0, seed: int = SEED):
        self.domains = list(source_sizes)
        self.n_per_source, self.n_target = n_per_source, n_target
        self.src = {d: IndexStream(n, seed + 1 + i) for i, (d, n) in enumerate(source_sizes.items())}
        self.tgt = IndexStream(target_size, seed + 100) if target_size else None
        self.src_aug = torch.Generator().manual_seed(seed + 200)
        self.tgt_aug = torch.Generator().manual_seed(seed + 300)

    def source_indices(self) -> dict:
        return {d: self.src[d].next(self.n_per_source) for d in self.domains}

    def target_indices(self) -> np.ndarray:
        if self.tgt is None:
            raise RuntimeError("this sampler was built without a target stream")
        return self.tgt.next(self.n_target)


# ------------------------------------------------------------------ preprocessing
def random_crop_flip(x_u8: torch.Tensor, g: torch.Generator, crop: int = CROP) -> torch.Tensor:
    """Per-image random crop + horizontal flip of a uint8 batch. Randomness comes only from ``g``."""
    n, _, h, w = x_u8.shape
    top = torch.randint(0, h - crop + 1, (n,), generator=g)
    left = torch.randint(0, w - crop + 1, (n,), generator=g)
    flip = torch.rand(n, generator=g) < 0.5
    out = torch.stack([x_u8[i, :, top[i]:top[i] + crop, left[i]:left[i] + crop] for i in range(n)])
    return torch.where(flip.view(n, 1, 1, 1), out.flip(-1), out)


def center_crop(x_u8: torch.Tensor, crop: int = CROP) -> torch.Tensor:
    h, w = x_u8.shape[-2:]
    t, l = (h - crop) // 2, (w - crop) // 2
    return x_u8[..., t:t + crop, l:l + crop]


def normalize(x_u8: torch.Tensor) -> torch.Tensor:
    """uint8 -> float, ImageNet mean/std (on x's device)."""
    x = x_u8.float().div_(255)
    return (x - MEAN.to(x.device)) / STD.to(x.device)
