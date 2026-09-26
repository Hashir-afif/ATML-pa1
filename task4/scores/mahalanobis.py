"""Mahalanobis distance to the nearest known-class mean, with one shared diagonal covariance.

u(x) = min_c (f(x) - mu_c)^T Sigma^-1 (f(x) - mu_c)

mu_c and Sigma are estimated from **unaugmented CIFAR-10 training** features of the frozen model (manual section 4,
step 2). Sigma is diagonal and shared across classes; 1e-6 is added to every diagonal entry. No validation, test or
CIFAR-100 data is used for fitting.
"""
from __future__ import annotations

import torch


class MahalanobisScore:
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.means: torch.Tensor | None = None
        self.inv_var: torch.Tensor | None = None

    def fit(self, train_features: torch.Tensor, train_labels: torch.Tensor, num_classes: int = 10) -> "MahalanobisScore":
        f = train_features.double()
        self.means = torch.stack([f[train_labels == c].mean(0) for c in range(num_classes)])
        centred = f - self.means[train_labels]                      # shared within-class covariance
        var = centred.pow(2).mean(0) + self.eps                     # diagonal only
        self.inv_var = 1.0 / var
        return self

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        assert self.means is not None, "fit() on unaugmented CIFAR-10 training features first"
        f = features.double()
        d = torch.stack([((f - mu).pow(2) * self.inv_var).sum(1) for mu in self.means], dim=1)
        return d.min(1).values.float()

    def state(self) -> dict:
        return {"eps": self.eps, "means": self.means, "inv_var": self.inv_var}
