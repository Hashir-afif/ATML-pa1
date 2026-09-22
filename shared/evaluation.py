"""Evaluation utilities shared by Tasks 2 and 3: predictions, accuracy/macro-F1, domain separability."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from shared import pacs_protocol as proto

NUM_CLASSES = 7


@torch.no_grad()
def predict(model, x_u8: torch.Tensor, device: str, batch_size: int = 256):
    """Centre-crop evaluation in eval mode. Returns (features [N,512], logits [N,C]) on CPU."""
    model.eval()
    feats, logits = [], []
    for s in range(0, len(x_u8), batch_size):
        x = proto.normalize(proto.center_crop(x_u8[s:s + batch_size]).to(device))
        f, z = model(x)
        feats.append(f.float().cpu()); logits.append(z.float().cpu())
    return torch.cat(feats), torch.cat(logits)


def cls_metrics(y, yhat, num_classes: int = NUM_CLASSES) -> dict:
    y, yhat = np.asarray(y), np.asarray(yhat)
    return {"acc": float((y == yhat).mean()),
            "macro_f1": float(f1_score(y, yhat, average="macro", labels=list(range(num_classes)), zero_division=0)),
            "n": int(len(y))}


def per_class_accuracy(y, yhat, class_names: list[str]) -> dict:
    y, yhat = np.asarray(y), np.asarray(yhat)
    return {c: (float((yhat[y == k] == k).mean()) if (y == k).any() else None) for k, c in enumerate(class_names)}


def confusion(y, yhat, num_classes: int = NUM_CLASSES) -> np.ndarray:
    return confusion_matrix(y, yhat, labels=list(range(num_classes)))


def evaluate_sources(model, val: dict, device: str, batch_size: int = 256) -> dict:
    """Per-source-domain accuracy/macro-F1 plus their mean and worst over domains."""
    out = {"domains": {}}
    for dom, (x, y) in val.items():
        _, z = predict(model, x, device, batch_size)
        out["domains"][dom] = cls_metrics(y.numpy(), z.argmax(1).numpy())
    accs = [m["acc"] for m in out["domains"].values()]
    f1s = [m["macro_f1"] for m in out["domains"].values()]
    out.update(mean_acc=float(np.mean(accs)), mean_macro_f1=float(np.mean(f1s)),
               worst_acc=float(np.min(accs)), worst_macro_f1=float(np.min(f1s)))
    return out


def domain_separability(groups: dict, seed: int = 6304, test_size: float = 0.3, C: float = 1.0) -> dict:
    """Held-out accuracy of a balanced logistic regression predicting the group (domain) of a feature.

    Every group is subsampled (seed) to the size of the smallest group, then a stratified
    ``1-test_size / test_size`` split is made (seed). Chance = 1 / number of groups.
    """
    names = list(groups)
    n = min(len(g) for g in groups.values())
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k, name in enumerate(names):
        F = np.asarray(groups[name], dtype=np.float64)
        idx = rng.choice(len(F), n, replace=False) if len(F) > n else np.arange(len(F))
        X.append(F[idx]); y.append(np.full(n, k))
    X, y = np.concatenate(X), np.concatenate(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    return {"separability_acc": float((pred == yte).mean()), "chance": 1.0 / len(names), "groups": names,
            "n_per_group": int(n), "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "test_confusion": confusion_matrix(yte, pred, labels=list(range(len(names)))).tolist(),
            "lbfgs_iterations": int(np.max(clf.n_iter_)), "converged": bool(np.max(clf.n_iter_) < 5000),
            "seed": seed, "C": C, "test_size": test_size}


def sharpness_proxy(model, x_u8: torch.Tensor, y: torch.Tensor, device: str, radius: float = 0.05) -> dict:
    """Task 3 local sharpness proxy on a fixed batch, in eval mode (manual section 3, step 4).

    Delta = L(theta + eps) - L(theta), eps = radius * grad L / ||grad L||_2, with the gradient taken over every
    trainable parameter (backbone and head) at the centre-cropped batch. Parameters are restored afterwards.
    """
    model.eval()
    x = proto.normalize(proto.center_crop(x_u8).to(device))
    y = y.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    loss = F.cross_entropy(model(x)[1], y)
    grads = torch.autograd.grad(loss, params)
    gnorm = torch.sqrt(sum((g.float() ** 2).sum() for g in grads))
    eps = [radius * g / gnorm for g in grads]
    with torch.no_grad():
        for p, e in zip(params, eps):
            p.add_(e)
        loss_pert = F.cross_entropy(model(x)[1], y)
        for p, e in zip(params, eps):
            p.sub_(e)
    return {"loss": float(loss.item()), "loss_perturbed": float(loss_pert.item()),
            "delta_sharp": float(loss_pert.item() - loss.item()), "grad_norm": float(gnorm.item()),
            "radius": radius, "n": int(len(y))}
