"""One training loop for every PACS method (Tasks 2 and 3).

A method is an ``nn.Module`` subclass of :class:`Method` holding any extra trainable modules (e.g. a
domain discriminator). It defines ``loss(model, batch, progress)``; methods that need a different
update (SAM in Task 3) override ``step``. The loop itself fixes, for every method:
initialisation, source sampling and augmentation, optimiser (AdamW over model + method parameters),
epoch budget, frozen BatchNorm statistics, per-epoch source-validation evaluation, checkpoint selection
by mean source-validation macro-F1, and early stopping.

Target isolation: the loop only receives target *images*, and only when ``method.uses_target`` is True.
A method with ``uses_target = False`` (Source-only / Task 3 ERM, DAN-DG, SAM) never loads the target.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from shared import pacs_protocol as proto
from shared.evaluation import evaluate_sources
from shared.models import set_train_mode


@dataclass
class Batch:
    xs: torch.Tensor                  # normalised source images [n_src,3,224,224]
    ys: torch.Tensor                  # source labels
    ds: torch.Tensor                  # source-domain index (0..n_sources-1) of each source image
    xt: torch.Tensor | None = None    # normalised target images (adaptation methods only)


class Method(nn.Module):
    name = "base"
    uses_target = False

    def loss(self, model, batch: Batch, progress: float):
        raise NotImplementedError

    def step(self, model, optimizer, batch: Batch, progress: float) -> dict:
        loss, logs = self.loss(model, batch, progress)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return logs


def _to_py(o):
    if isinstance(o, dict):
        return {k: _to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_py(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def train(model: nn.Module, method: Method, source: dict, cfg: dict, run_dir: Path, device: str,
          target_images: torch.Tensor | None = None, iters_override: int | None = None,
          overwrite: bool = False) -> dict:
    """Train ``model`` with ``method``; save ``best.pt``, ``history.json``, ``summary.json``, ``config.yaml``.

    ``source`` is ``pacs_protocol.load_source_data()``. ``cfg`` is the resolved run config.
    """
    tc, seed = cfg["train"], cfg["seed"]
    run_dir = Path(run_dir)
    if (run_dir / "best.pt").exists() and not overwrite:
        raise FileExistsError(f"{run_dir}/best.pt exists; pass overwrite=True to replace it deliberately")
    if method.uses_target and target_images is None:
        raise ValueError(f"{method.name} needs target images")
    if not method.uses_target and target_images is not None:
        raise ValueError(f"{method.name} must not receive target images")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    domains = list(source["train"])
    train_x = {d: source["train"][d][0] for d in domains}
    train_y = {d: source["train"][d][1] for d in domains}
    sampler = proto.BatchSampler({d: len(train_y[d]) for d in domains}, tc["n_per_source"],
                                 len(target_images) if method.uses_target else None,
                                 tc["n_target"] if method.uses_target else 0, seed)
    ipe = iters_override or proto.iters_per_epoch({d: len(train_y[d]) for d in domains}, tc["n_per_source"])
    total_steps = tc["max_epochs"] * ipe

    model.to(device)
    method.to(device)
    # Model (backbone + head) always uses tc["lr"]. Extra method modules (e.g. a domain discriminator) use
    # tc["lr"] * method.lr_multiplier; the multiplier is 1.0 unless a config sets it (Task 2 DANN/CDAN
    # stabilisation, decision 2026-09-22). With a multiplier of 1 this is identical to a single AdamW group.
    groups = [{"params": list(model.parameters()), "lr": tc["lr"]}]
    extra = list(method.parameters())
    if extra:
        groups.append({"params": extra, "lr": tc["lr"] * float(getattr(method, "lr_multiplier", 1.0))})
    optimizer = torch.optim.AdamW(groups, lr=tc["lr"], weight_decay=tc["weight_decay"])

    history, best_f1, best_epoch, bad, step = [], -1.0, 0, 0, 0
    t_start = time.time()
    for epoch in range(1, tc["max_epochs"] + 1):
        t0 = time.time()
        set_train_mode(model, method)
        sums: dict = {}
        for _ in range(ipe):
            idx = sampler.source_indices()
            xs = torch.cat([train_x[d][idx[d]] for d in domains])
            ys = torch.cat([train_y[d][idx[d]] for d in domains])
            ds = torch.cat([torch.full((len(idx[d]),), k, dtype=torch.long) for k, d in enumerate(domains)])
            xs = proto.normalize(proto.random_crop_flip(xs, sampler.src_aug).to(device, non_blocking=True))
            xt = None
            if method.uses_target:
                xt = target_images[sampler.target_indices()]
                xt = proto.normalize(proto.random_crop_flip(xt, sampler.tgt_aug).to(device, non_blocking=True))
            batch = Batch(xs, ys.to(device), ds.to(device), xt)
            logs = method.step(model, optimizer, batch, step / total_steps)
            for k, v in logs.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            step += 1
        train_logs = {k: v / ipe for k, v in sums.items()}

        val = evaluate_sources(model, source["val"], device, tc["eval_batch_size"])
        rec = {"epoch": epoch, "iters": ipe, "progress_end": step / total_steps, "train": train_logs, "val": val,
               "epoch_seconds": time.time() - t0}
        history.append(rec)
        improved = val["mean_macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, bad = val["mean_macro_f1"], epoch, 0
            torch.save({"model": model.state_dict(), "method": method.state_dict(), "epoch": epoch,
                        "val": val, "config": cfg}, run_dir / "best.pt")
        else:
            bad += 1
        (run_dir / "history.json").write_text(json.dumps(_to_py(history), indent=1))
        print(f"[{cfg['run_name']}] epoch {epoch:2d}  " + "  ".join(f"{k}={v:.4f}" for k, v in train_logs.items())
              + f"  val_mean_f1={val['mean_macro_f1']:.4f}{'  *' if improved else ''}  ({rec['epoch_seconds']:.0f}s)")
        if bad >= tc["patience"]:
            break

    summary = {"run_name": cfg["run_name"], "method": method.name, "best_epoch": best_epoch,
               "best_mean_val_macro_f1": best_f1, "epochs_run": len(history), "iters_per_epoch": ipe,
               "total_planned_steps": total_steps, "steps_run": step,
               "stopped_early": len(history) < tc["max_epochs"], "train_seconds": time.time() - t_start,
               "best_val": history[best_epoch - 1]["val"], "checkpoint": str(run_dir / "best.pt"),
               "config_file": str(run_dir / "config.yaml"), "uses_target_images": method.uses_target,
               "extra_module_lr": tc["lr"] * float(getattr(method, "lr_multiplier", 1.0)) if list(method.parameters()) else None}
    (run_dir / "summary.json").write_text(json.dumps(_to_py(summary), indent=1))
    return summary
