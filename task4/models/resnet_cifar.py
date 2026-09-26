"""CIFAR-appropriate ResNet-18 for Task 4.

torchvision ResNet-18 with the ImageNet stem replaced as the manual requires: a 3x3 stride-1 first convolution and
no initial max-pooling, so 32x32 inputs keep their resolution through the stem. The classifier can carry extra
``dummy`` outputs for PROSER's placeholders; known-class logits are always the first ``num_classes`` columns.

``forward`` returns (penultimate feature f(x) [N,512], logits z(x)). ``stem_to_layer2`` / ``after_layer2`` expose
the split used for manifold mixup (after layer2, before layer3).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18

FEAT_DIM = 512


class ResNetCIFAR(nn.Module):
    def __init__(self, num_classes: int = 10, dummy: int = 0):
        super().__init__()
        net = resnet18(weights=None, num_classes=num_classes + dummy)
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)   # was 7x7 stride 2
        net.maxpool = nn.Identity()                                                     # remove initial max-pool
        self.net = net
        self.num_classes, self.dummy = num_classes, dummy

    def stem_to_layer2(self, x):
        n = self.net
        return n.layer2(n.layer1(n.maxpool(n.relu(n.bn1(n.conv1(x))))))

    def after_layer2(self, h):
        n = self.net
        f = torch.flatten(n.avgpool(n.layer4(n.layer3(h))), 1)
        return f, n.fc(f)

    def forward(self, x):
        return self.after_layer2(self.stem_to_layer2(x))


def build_model(num_classes: int = 10, dummy: int = 0, seed: int = 6304) -> ResNetCIFAR:
    torch.manual_seed(seed)
    return ResNetCIFAR(num_classes, dummy)
