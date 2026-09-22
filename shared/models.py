"""Models shared by Tasks 2 and 3.

* :class:`PACSNet` — torchvision ResNet-18 (``IMAGENET1K_V1``) backbone producing the 512-d
  pre-classifier feature, plus a new 7-class linear head. The whole network is fine-tuned.
* BatchNorm policy (manual, Tasks 2 and 3): running mean/variance stay at their ImageNet values.
  :func:`set_train_mode` calls ``model.train()`` and then puts **only** the BatchNorm modules in
  eval mode; BN scale/bias (gamma/beta) remain trainable.
* :class:`DomainDiscriminator` — 256-unit hidden layer, ReLU, dropout 0.5, 2-class output (DANN/CDAN).
* :func:`grad_reverse` — gradient-reversal layer (identity forward, ``-alpha * grad`` backward),
  written from the description in Ganin et al. (2016).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

FEAT_DIM = 512


class PACSNet(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        assert net.fc.in_features == FEAT_DIM
        net.fc = nn.Identity()
        self.backbone = net
        self.head = nn.Linear(FEAT_DIM, num_classes)

    def forward(self, x):
        f = self.backbone(x)
        return f, self.head(f)


def build_model(num_classes: int = 7, seed: int = 6304) -> PACSNet:
    """Identical initialisation for every method: pretrained backbone + head drawn from ``seed``."""
    torch.manual_seed(seed)
    return PACSNet(num_classes)


def freeze_bn_stats(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def set_train_mode(model: nn.Module, *extra: nn.Module) -> None:
    """``model.train()`` then freeze the pretrained network's BatchNorm running statistics (gamma/beta stay
    trainable). ``extra`` modules (e.g. a domain discriminator) are simply put in train mode: the manual's BN
    policy concerns the pretrained ImageNet statistics, which newly added modules do not have."""
    model.train()
    freeze_bn_stats(model)
    for m in extra:
        m.train()


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


def grad_reverse(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return _GradReverse.apply(x, alpha)


def dann_alpha(progress: float, gamma: float = 10.0, max_strength: float = 1.0) -> float:
    """Standard DANN schedule alpha(p) = 2/(1+exp(-gamma p)) - 1, scaled by ``max_strength``; p in [0,1]."""
    return max_strength * (2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0)


class DomainDiscriminator(nn.Module):
    """Linear(in, 256) -> [BatchNorm1d] -> ReLU -> Dropout(0.5) -> Linear(256, 2).

    ``batchnorm=False`` is the manual's architecture. ``batchnorm=True`` (Task 2 post-evaluation fix, see the
    Task 2 README) normalises the hidden pre-activations per batch, so the backbone cannot win the adversarial game
    by inflating feature scale and the ReLU units cannot all die. This mirrors common public DANN/CDAN code."""

    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.5, batchnorm: bool = False):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden)]
        if batchnorm:
            layers.append(nn.BatchNorm1d(hidden))
        layers += [nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
