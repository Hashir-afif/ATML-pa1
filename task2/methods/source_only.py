"""Source-only ERM: cross-entropy on domain-balanced source batches (8 per source domain).

Never receives target images, so its checkpoint is a valid Task 3 ERM baseline.
"""
import torch.nn.functional as F

from shared.training import Method


class SourceOnly(Method):
    name = "source_only"
    uses_target = False

    def loss(self, model, batch, progress):
        _, logits = model(batch.xs)
        cls = F.cross_entropy(logits, batch.ys)
        return cls, {"cls_loss": cls.item(), "train_acc": (logits.argmax(1) == batch.ys).float().mean().item(),
                     "total_loss": cls.item()}
