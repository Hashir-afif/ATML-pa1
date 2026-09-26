"""MLS: u(x) = -max_k z_k (maximum logit score, negated so that larger = more unknown)."""
import torch


def mls(logits: torch.Tensor) -> torch.Tensor:
    return -logits.max(1).values
