"""DANN: a domain discriminator on the 512-d feature behind a gradient-reversal layer.

L = CE(source) + w * CE_domain(D(GRL_alpha(F(x))), {source: 0, target: 1}), alpha(p) from the standard
schedule 2/(1+exp(-10p)) - 1 (optionally scaled by ``max_grl``). Only source images enter the class
loss; source and target images both enter the domain loss. ``disc_lr_multiplier`` (default 1 = manual) scales only the discriminator's learning rate;
see the Task 2 README. ``p`` = global step / planned steps
(max_epochs x iterations per epoch), so the schedule is the same whether or not training stops early.
"""
import torch
import torch.nn.functional as F

from shared.models import FEAT_DIM, DomainDiscriminator, dann_alpha, grad_reverse
from shared.training import Method


def domain_loss(disc, g, n_src, alpha):
    d_logits = disc(grad_reverse(g, alpha))
    d_true = torch.cat([torch.zeros(n_src, dtype=torch.long), torch.ones(len(g) - n_src, dtype=torch.long)]).to(g.device)
    return F.cross_entropy(d_logits, d_true), (d_logits.argmax(1) == d_true).float().mean()


class DANN(Method):
    name = "dann"
    uses_target = True

    def __init__(self, hidden=256, dropout=0.5, max_grl=1.0, gamma=10.0, domain_loss_weight=1.0, seed=6304,
                 disc_lr_multiplier=1.0, disc_batchnorm=False):
        super().__init__()
        torch.manual_seed(seed + 1)                     # discriminator init independent of the backbone/head init
        self.disc = DomainDiscriminator(FEAT_DIM, hidden, dropout, disc_batchnorm)
        self.max_grl, self.gamma, self.w = float(max_grl), float(gamma), float(domain_loss_weight)
        self.lr_multiplier = float(disc_lr_multiplier)   # discriminator lr = base lr x this (1.0 = manual)

    def features_for_discriminator(self, f, logits):
        return f

    def loss(self, model, batch, progress):
        ns = len(batch.xs)
        f, logits = model(torch.cat([batch.xs, batch.xt]))
        cls = F.cross_entropy(logits[:ns], batch.ys)
        alpha = dann_alpha(progress, self.gamma, self.max_grl)
        dom, dacc = domain_loss(self.disc, self.features_for_discriminator(f, logits), ns, alpha)
        total = cls + self.w * dom
        return total, {"cls_loss": cls.item(), "domain_loss": dom.item(), "disc_acc": dacc.item(), "alpha": alpha,
                       "total_loss": total.item(), "train_acc": (logits[:ns].argmax(1) == batch.ys).float().mean().item()}
