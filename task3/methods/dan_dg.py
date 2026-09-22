"""DAN-DG: L = L_ERM + (lambda_DG / 3) * sum_{e<e'} MMD^2(F(X_e), F(X_e')) over the three source domains.

The MMD is the same multi-kernel estimator as Task 2's DAN (``shared.losses.mk_mmd``), applied to every pair of
observed source domains within the domain-balanced batch (8 + 8 features per pair; bandwidths 0.5/1/2 x the
median pairwise squared distance of that pair's combined batch). The target domain is never used.
"""
import torch.nn.functional as F

from shared.losses import pairwise_mk_mmd
from shared.pacs import SOURCE_DOMAINS
from shared.training import Method


class DANDG(Method):
    name = "dan_dg"
    uses_target = False

    def __init__(self, lambda_dg: float = 1.0, bandwidth_multipliers=(0.5, 1.0, 2.0)):
        super().__init__()
        self.lambda_dg, self.mults = float(lambda_dg), tuple(bandwidth_multipliers)

    def loss(self, model, batch, progress):
        f, logits = model(batch.xs)
        cls = F.cross_entropy(logits, batch.ys)
        mmd, pair_vals, pairs = pairwise_mk_mmd(f, batch.ds, self.mults)
        total = cls + self.lambda_dg * mmd
        logs = {"cls_loss": cls.item(), "mmd": mmd.item(), "total_loss": total.item(),
                "train_acc": (logits.argmax(1) == batch.ys).float().mean().item()}
        for (a, b), v in zip(pairs, pair_vals):
            logs[f"mmd_{SOURCE_DOMAINS[a]}_{SOURCE_DOMAINS[b]}"] = v
        return total, logs
