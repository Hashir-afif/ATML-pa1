"""Vanilla closed-set baseline: cross-entropy on CIFAR-10 with random crop (4-pixel padding) + horizontal flip."""
import torch.nn.functional as F

from task4.data import cifar10


class Vanilla:
    name = "vanilla"
    use_randaugment = False        # GCSC overrides this

    def augment(self, x_u8, g):
        x = cifar10.random_crop_flip(x_u8, g)
        if self.use_randaugment:   # GCSC: after the crop and flip, before conversion/normalisation
            x = cifar10.rand_augment(x)
        return x

    def loss(self, model, x_u8, y, g, device):
        _, z = model(cifar10.normalize(self.augment(x_u8, g).to(device)))
        loss = F.cross_entropy(z, y)
        return loss, {"cls_loss": loss.item(), "train_acc": (z.argmax(1) == y).float().mean().item()}
