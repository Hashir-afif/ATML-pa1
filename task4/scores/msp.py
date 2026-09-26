"""MSP: u(x) = 1 - max_k p_k(x), with p = softmax(z) over the known-class logits."""
import torch


def msp(logits: torch.Tensor) -> torch.Tensor:
    return 1.0 - logits.softmax(1).max(1).values
