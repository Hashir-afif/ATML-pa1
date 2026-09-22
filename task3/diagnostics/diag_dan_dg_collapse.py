"""Source-side diagnostic (no Sketch): where and how does DAN-DG collapse?

Replays the first updates of a DAN-DG run with the exact Task 3 protocol and logs, every 10 updates: the
classification loss, the pairwise MMD penalty (biased estimator, as used for training) and its unbiased
counterpart on the same batch, the 512-d feature norm, the mean per-dimension standard deviation, the fraction of
feature dimensions that are exactly zero for every image in the batch (dead units), and the mean max softmax.

Usage: python task3/diagnostics/diag_dan_dg_collapse.py <lambda_dg> <n_updates> <out.json>
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from shared import pacs_protocol as proto
from shared.losses import pairwise_mk_mmd, pairwise_sq_dists
from shared.models import build_model, set_train_mode
from shared.training import Batch
from task3.methods import DANDG

lam, n_updates, out = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3]


def unbiased_mk_mmd(x, y, mults=(0.5, 1.0, 2.0)):
    """U-statistic MK-MMD^2 (diagonal kernel terms excluded), same kernels and median bandwidth."""
    n, m = len(x), len(y)
    z = torch.cat([x, y]); d2 = pairwise_sq_dists(z)
    iu = torch.triu_indices(len(z), len(z), offset=1, device=z.device)
    med = d2[iu[0], iu[1]].median().clamp_min(1e-12)
    k = sum(torch.exp(-d2 / (mu * med)) for mu in mults)
    kxx = (k[:n, :n].sum() - k[:n, :n].diagonal().sum()) / (n * (n - 1))
    kyy = (k[n:, n:].sum() - k[n:, n:].diagonal().sum()) / (m * (m - 1))
    return kxx + kyy - 2 * k[:n, n:].mean()


src = proto.load_source_data()
dom = list(src["train"])
dev = "cuda"
model, method = build_model(7, 6304).to(dev), DANDG(lam)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
s = proto.BatchSampler({d: len(src["train"][d][1]) for d in dom}, 8)
set_train_mode(model, method)
rows = []
for step in range(n_updates):
    idx = s.source_indices()
    xs = proto.normalize(proto.random_crop_flip(torch.cat([src["train"][d][0][idx[d]] for d in dom]), s.src_aug).to(dev))
    ys = torch.cat([src["train"][d][1][idx[d]] for d in dom]).to(dev)
    ds = torch.cat([torch.full((8,), k, dtype=torch.long) for k in range(3)]).to(dev)
    if step % 10 == 0:
        with torch.no_grad():
            f, z = model(xs)
            mmd_b, pv, pairs = pairwise_mk_mmd(f, ds)
            mmd_u = torch.stack([unbiased_mk_mmd(f[ds == a], f[ds == b]) for a, b in pairs]).mean()
            rows.append({"step": step, "cls_loss": F.cross_entropy(z, ys).item(), "mmd_biased": mmd_b.item(),
                         "mmd_unbiased": mmd_u.item(), "feat_norm": f.norm(dim=1).mean().item(),
                         "feat_dim_std": f.std(0).mean().item(), "dead_dims_frac": (f.abs().sum(0) == 0).float().mean().item(),
                         "mean_max_softmax": z.softmax(1).max(1).values.mean().item(),
                         "logit_std_across_images": z.std(0).mean().item()})
            r = rows[-1]
            print(f"step {step:4d} cls {r['cls_loss']:.3f} mmdB {r['mmd_biased']:.3f} mmdU {r['mmd_unbiased']:+.3f} "
                  f"|f| {r['feat_norm']:.2f} dimstd {r['feat_dim_std']:.4f} dead {r['dead_dims_frac']:.2f} "
                  f"maxp {r['mean_max_softmax']:.3f} logit-std {r['logit_std_across_images']:.4f}", flush=True)
    method.step(model, opt, Batch(xs, ys, ds), 0.0)
json.dump(rows, open(out, "w"), indent=1)
