"""Multi-kernel MMD shared by Task 2 (DAN, source vs target) and Task 3 (DAN-DG, source pairs).

MMD^2 is estimated with the biased (V-statistic) estimator

    MMD^2(X, Y) = mean k(x, x') + mean k(y, y') - 2 mean k(x, y)

with k = sum of three RBF kernels k_m(a, b) = exp(-||a - b||^2 / (m * med)), m in {0.5, 1, 2}, where
``med`` is the median pairwise squared distance over the current *combined* batch [X; Y] (off-diagonal
pairs, treated as a constant: no gradient flows through the bandwidth). This follows the manual's
"sum of three RBF kernels whose bandwidths are 0.5, 1, and 2 times the median pairwise squared feature
distance in the current combined batch"; the formulation follows Long et al. (2015).
"""
from __future__ import annotations

import torch


def pairwise_sq_dists(z: torch.Tensor) -> torch.Tensor:
    """||z_i - z_j||^2 without sqrt (so gradients stay finite at zero distance)."""
    sq = (z * z).sum(1, keepdim=True)
    return (sq + sq.T - 2.0 * z @ z.T).clamp_min(0.0)


def mk_mmd(x: torch.Tensor, y: torch.Tensor, bandwidth_multipliers=(0.5, 1.0, 2.0)) -> torch.Tensor:
    n = x.shape[0]
    z = torch.cat([x, y])
    d2 = pairwise_sq_dists(z)
    iu = torch.triu_indices(len(z), len(z), offset=1, device=z.device)
    med = d2.detach()[iu[0], iu[1]].median().clamp_min(1e-12)
    k = sum(torch.exp(-d2 / (m * med)) for m in bandwidth_multipliers)
    return k[:n, :n].mean() + k[n:, n:].mean() - 2.0 * k[:n, n:].mean()


def pairwise_mk_mmd(f: torch.Tensor, domain_ids: torch.Tensor, bandwidth_multipliers=(0.5, 1.0, 2.0)):
    """DAN-DG penalty (Task 3): mean MK-MMD^2 over all unordered pairs of source domains in the batch.

    Uses exactly :func:`mk_mmd` above, so each pair's bandwidths come from that pair's own combined batch
    (8 + 8 features in Task 3). With three domains the mean equals (1/3) * sum over the three pairs, as in the
    manual. Returns (mean penalty tensor, list of per-pair values as floats, list of pair index tuples).
    """
    doms = [int(d) for d in torch.unique(domain_ids).tolist()]
    pairs = [(a, b) for i, a in enumerate(doms) for b in doms[i + 1:]]
    vals = [mk_mmd(f[domain_ids == a], f[domain_ids == b], bandwidth_multipliers) for a, b in pairs]
    return torch.stack(vals).mean(), [v.item() for v in vals], pairs
