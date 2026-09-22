"""DAN: L = CE(source) + lambda * MK-MMD^2(F(x_s), F(x_t)) on the 512-d pre-classifier feature.

The 24 source and 24 target images go through the network in one forward pass (valid because
BatchNorm statistics are frozen, so batch composition does not change any output).
"""
import torch
import torch.nn.functional as F

from shared.losses import mk_mmd
from shared.training import Method


class DAN(Method):
    name = "dan"
    uses_target = True

    def __init__(self, lambda_mmd: float = 1.0, bandwidth_multipliers=(0.5, 1.0, 2.0)):
        super().__init__()
        self.lambda_mmd, self.mults = float(lambda_mmd), tuple(bandwidth_multipliers)

    def loss(self, model, batch, progress):
        ns = len(batch.xs)
        f, logits = model(torch.cat([batch.xs, batch.xt]))
        cls = F.cross_entropy(logits[:ns], batch.ys)
        mmd = mk_mmd(f[:ns], f[ns:], self.mults)
        total = cls + self.lambda_mmd * mmd
        return total, {"cls_loss": cls.item(), "mmd": mmd.item(), "total_loss": total.item(),
                       "train_acc": (logits[:ns].argmax(1) == batch.ys).float().mean().item()}
