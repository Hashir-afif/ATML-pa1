"""Task 4 training loop (manual section 4).

SGD lr 0.1, momentum 0.9, weight decay 5e-4, cosine decay, batch size 128, 100 epochs, seed 6304. The checkpoint
with the highest CIFAR-10 **validation accuracy** is kept (PROSER uses 50 epochs and lr 1e-3 via its config).

A method supplies ``loss(model, x_u8, y, generator, device)`` and receives raw uint8 images, so it controls its own
augmentation (GCSC adds RandAugment) and batch handling (PROSER splits the batch in half).
No CIFAR-100 image is ever visible here.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from task4.data import cifar10


@torch.no_grad()
def evaluate(model, x_u8, y, device, batch_size: int = 512, num_known: int = 10) -> float:
    """Closed-set accuracy using only the known-class logits."""
    model.eval()
    correct = 0
    for s in range(0, len(x_u8), batch_size):
        _, z = model(cifar10.normalize(x_u8[s:s + batch_size].to(device)))
        correct += (z[:, :num_known].argmax(1).cpu() == y[s:s + batch_size]).sum().item()
    return correct / len(y)


def _archive_interrupted_run(run_dir: Path) -> None:
    """A ``best.pt`` with no ``summary.json`` next to it was never a completed, reported run (summary.json is
    written only on success) — it is the debris of a crash or a killed process. Move it aside (never delete) and
    let the caller start clean. Because the seed is fixed, a from-scratch restart reproduces the crashed attempt
    exactly, so no partial-resume logic is needed."""
    import shutil
    import time as _time
    stamp = _time.strftime("%Y%m%dT%H%M%S")
    dest = run_dir.parent / f"{run_dir.name}_interrupted_{stamp}"
    shutil.move(str(run_dir), str(dest))
    print(f"[recovery] {run_dir} looked interrupted (best.pt without summary.json); archived to {dest} and restarting from scratch")


def train(model, method, data: dict, cfg: dict, run_dir: Path, device: str, overwrite: bool = False) -> dict:
    tc = cfg["train"]
    run_dir = Path(run_dir)
    if (run_dir / "summary.json").exists() and not overwrite:
        raise FileExistsError(f"{run_dir}/summary.json exists; pass overwrite=True to replace it deliberately")
    if run_dir.exists() and not (run_dir / "summary.json").exists():
        _archive_interrupted_run(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    xtr, ytr = data["train"]
    xva, yva = data["val"]
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=tc["lr"], momentum=tc["momentum"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"])
    g = torch.Generator().manual_seed(cfg["seed"])
    n_batches = int(np.ceil(len(ytr) / tc["batch_size"]))

    history, best_acc, best_epoch, t_start = [], -1.0, 0, time.time()
    for epoch in range(1, tc["epochs"] + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(ytr), generator=g)
        sums, seen = {}, 0
        for b in range(n_batches):
            idx = perm[b * tc["batch_size"]:(b + 1) * tc["batch_size"]]
            loss, logs = method.loss(model, xtr[idx], ytr[idx].to(device), g, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            for k, v in logs.items():
                sums[k] = sums.get(k, 0.0) + float(v) * len(idx)
            seen += len(idx)
        sched.step()
        train_logs = {k: v / seen for k, v in sums.items()}
        val_acc = evaluate(model, xva, yva, device, num_known=cfg["model"]["num_classes"])
        rec = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train": train_logs, "val_acc": val_acc,
               "epoch_seconds": time.time() - t0}
        history.append(rec)
        improved = val_acc > best_acc
        if improved:
            best_acc, best_epoch = val_acc, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc, "config": cfg},
                       run_dir / "best.pt")
        (run_dir / "history.json").write_text(json.dumps(history, indent=1))
        if epoch <= 3 or improved or epoch % max(1, tc["epochs"] // 20) == 0:
            print(f"[{cfg['run_name']}] epoch {epoch:3d}/{tc['epochs']}  "
                  + "  ".join(f"{k}={v:.4f}" for k, v in train_logs.items())
                  + f"  val_acc={val_acc:.4f}{'  *' if improved else ''}  ({rec['epoch_seconds']:.0f}s)", flush=True)

    summary = {"run_name": cfg["run_name"], "method": cfg["method"], "best_epoch": best_epoch,
               "best_val_acc": best_acc, "epochs_run": len(history), "train_seconds": time.time() - t_start,
               "checkpoint": str(run_dir / "best.pt"), "config_file": str(run_dir / "config.yaml"),
               "uses_cifar100": False}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary
