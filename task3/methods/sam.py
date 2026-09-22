"""SAM (Foret et al., 2021), standard non-adaptive version, on the ERM objective with AdamW as base optimiser.

Each update: (1) loss and gradient at theta; (2) eps = rho * g / ||g||_2 over all parameters with a gradient;
(3) loss and gradient at theta + eps; (4) restore theta and take the AdamW step with the gradient from (3).
Two forward/backward passes per batch. BatchNorm running statistics stay frozen in both passes (the training loop
keeps the backbone BN modules in eval mode). Written for this repository from the paper.
"""
import torch
import torch.nn.functional as F

from shared.training import Method


class SAM(Method):
    name = "sam"
    uses_target = False

    def __init__(self, rho: float = 0.05):
        super().__init__()
        self.rho = float(rho)

    def step(self, model, optimizer, batch, progress):
        params = [p for g in optimizer.param_groups for p in g["params"]]
        # pass 1: gradient at theta
        _, logits = model(batch.xs)
        loss = F.cross_entropy(logits, batch.ys)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        active = [p for p in params if p.grad is not None]
        gnorm = torch.norm(torch.stack([p.grad.detach().norm(2) for p in active]), 2)
        eps = []
        with torch.no_grad():
            for p in active:
                e = p.grad * (self.rho / (gnorm + 1e-12))
                p.add_(e)
                eps.append(e)
        # pass 2: gradient at theta + eps
        optimizer.zero_grad(set_to_none=True)
        _, logits_p = model(batch.xs)
        loss_p = F.cross_entropy(logits_p, batch.ys)
        loss_p.backward()
        with torch.no_grad():
            for p, e in zip(active, eps):
                p.sub_(e)
        optimizer.step()
        return {"cls_loss": loss.item(), "perturbed_loss": loss_p.item(),
                "sam_gap": loss_p.item() - loss.item(), "grad_norm": gnorm.item(), "total_loss": loss_p.item(),
                "train_acc": (logits.argmax(1) == batch.ys).float().mean().item()}
