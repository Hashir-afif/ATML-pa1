"""CDAN: the DANN discriminator conditioned on predictions through g(x) = vec(f (outer) p).

f is the 512-d feature, p = softmax(C(f)) the 7-class probability vector, so g has 512*7 = 3584 dims.
Same hidden width, ReLU, dropout, gradient-reversal schedule and loss weight as DANN.
No entropy conditioning; neither f nor p is detached, so the reversed gradient reaches both the
backbone and the classifier head.
"""
import torch

from shared.models import FEAT_DIM, DomainDiscriminator
from task2.methods.dann import DANN


class CDAN(DANN):
    name = "cdan"

    def __init__(self, num_classes=7, hidden=256, dropout=0.5, max_grl=1.0, gamma=10.0, domain_loss_weight=1.0, seed=6304,
                 disc_lr_multiplier=1.0, disc_batchnorm=False):
        super().__init__(hidden, dropout, max_grl, gamma, domain_loss_weight, seed, disc_lr_multiplier, disc_batchnorm)
        torch.manual_seed(seed + 1)
        self.disc = DomainDiscriminator(FEAT_DIM * num_classes, hidden, dropout, disc_batchnorm)

    def features_for_discriminator(self, f, logits):
        p = logits.softmax(1)
        return torch.bmm(f.unsqueeze(2), p.unsqueeze(1)).flatten(1)
