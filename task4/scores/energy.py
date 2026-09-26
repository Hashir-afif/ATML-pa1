"""Energy: u(x) = -log sum_k exp(z_k) (negative free energy; larger = more unknown)."""
import torch


def energy(logits: torch.Tensor) -> torch.Tensor:
    return -torch.logsumexp(logits, dim=1)
