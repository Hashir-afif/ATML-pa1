"""Manifold mixup for PROSER's data placeholders (after layer2, before layer3).

Each example is paired with one of a *different* class, their layer2 activations are mixed with
lambda ~ Beta(2, 2), and the mixture is passed through the rest of the network. The result is a proxy unknown: a
point between two known-class regions, built only from CIFAR-10 data (no CIFAR-100 image is involved).
"""
from __future__ import annotations

import torch


def different_class_partner(y: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """For each i, an index j with y[j] != y[i] (drawn from a seeded permutation; falls back to a scan)."""
    n = len(y)
    perm = torch.randperm(n, generator=g).to(y.device)
    partner = perm.clone()
    bad = y[partner] == y
    for _ in range(10):
        if not bad.any():
            break
        partner[bad] = torch.randperm(n, generator=g).to(y.device)[bad]
        bad = y[partner] == y
    if bad.any():                                    # deterministic fallback for the few remaining
        order = torch.argsort(y)
        for i in torch.nonzero(bad).flatten().tolist():
            cand = order[y[order] != y[i]]
            if len(cand):
                partner[i] = cand[i % len(cand)]
    return partner


def mix_after_layer2(model, x_norm: torch.Tensor, y: torch.Tensor, g: torch.Generator, alpha: float = 2.0):
    """Returns (features, logits) of the mixed activations and the mask of pairs with different classes."""
    h = model.stem_to_layer2(x_norm)
    partner = different_class_partner(y, g)
    lam = torch.distributions.Beta(alpha, alpha).sample((len(y),)).to(h.device).view(-1, 1, 1, 1)
    h_mixed = lam * h + (1 - lam) * h[partner]
    f, z = model.after_layer2(h_mixed)
    return f, z, (y[partner] != y)
