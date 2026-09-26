"""PROSER (Zhou et al., 2021): classifier placeholders + data placeholders.

Written for this repository from the paper and the manual; no code was copied. The network keeps its 10 known-class
outputs and gains 5 dummy outputs; ``dummy = max_d z_d`` is the strongest dummy response.

Each mini-batch is split into two equal halves (manual): the first half trains the **classifier placeholders**, the
second half trains the **data placeholders**.

* Classifier placeholders (weight beta = 1), on half 1:
  - ``L1``: with [known logits ; dummy] the true class must stay the largest.
  - ``L2``: with the true class masked out, the dummy must become the largest.
* Data placeholders (weight gamma = 0.1), on half 2: manifold mixup after layer2 between two *different* known
  classes; the mixed example is trained towards the dummy output, i.e. [known logits ; dummy] with target = dummy.

Detection scores at test time:
* ``MLS`` on the ten known logits only (comparable with Vanilla and GCSC), and CSA likewise.
* ``placeholder``: dummy minus the largest known logit (larger = more unknown), the reference implementation's
  combination of the strongest dummy response with the known-class responses.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from task4.data import cifar10
from task4.methods.manifold_mixup import mix_after_layer2


class PROSER:
    name = "proser"

    def __init__(self, num_known: int = 10, dummy: int = 5, beta: float = 1.0, gamma: float = 0.1,
                 mixup_alpha: float = 2.0):
        self.K, self.D = num_known, dummy
        self.beta, self.gamma, self.mixup_alpha = beta, gamma, mixup_alpha

    def _known_and_dummy(self, z):
        return z[:, :self.K], z[:, self.K:].max(1, keepdim=True).values

    def loss(self, model, x_u8, y, g, device):
        x = cifar10.normalize(cifar10.random_crop_flip(x_u8, g).to(device))
        half = len(y) // 2
        x1, y1, x2, y2 = x[:half], y[:half], x[half:], y[half:]

        # --- classifier placeholders (half 1) ---
        _, z1 = model(x1)
        known1, dummy1 = self._known_and_dummy(z1)
        aug1 = torch.cat([known1, dummy1], 1)                       # [K+1]: known classes + strongest dummy
        l1 = F.cross_entropy(aug1, y1)                              # true class stays largest
        masked = known1.clone()
        masked.scatter_(1, y1.view(-1, 1), float("-inf"))           # exclude the true class
        l2 = F.cross_entropy(torch.cat([masked, dummy1], 1),
                             torch.full_like(y1, self.K))           # dummy becomes the largest of the rest
        loss_clf = l1 + self.beta * l2

        # --- data placeholders (half 2): manifold mixup between different known classes ---
        _, z_mix, valid = mix_after_layer2(model, x2, y2, g, self.mixup_alpha)
        known_m, dummy_m = self._known_and_dummy(z_mix)
        aug_m = torch.cat([known_m, dummy_m], 1)
        tgt_m = torch.full((len(y2),), self.K, device=device, dtype=torch.long)
        loss_data = F.cross_entropy(aug_m[valid], tgt_m[valid]) if valid.any() else torch.zeros((), device=device)

        total = loss_clf + self.gamma * loss_data
        with torch.no_grad():
            acc = (known1.argmax(1) == y1).float().mean().item()
            dummy_wins = (dummy_m.squeeze(1) > known_m.max(1).values).float().mean().item()
        return total, {"total_loss": total.item(), "clf_placeholder_loss": loss_clf.item(), "l1": l1.item(),
                       "l2": l2.item(), "data_placeholder_loss": float(loss_data.item()),
                       "train_acc": acc, "mixed_dummy_wins": dummy_wins}
