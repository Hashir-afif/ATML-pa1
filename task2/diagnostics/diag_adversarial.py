"""Instrumented DANN/CDAN training (source + UNLABELLED target only) to locate the divergence mechanism.

Usage: python diag_long.py <config> <max_steps> <disc_bn:0|1> <out.json>
Logs every 25 steps: source/target feature norms, discriminator hidden pre-activation scale, logit magnitude,
discriminator/head/backbone-last-block weight norms, losses, alpha, discriminator accuracy.
Stops early once the domain loss exceeds 50 (divergence reached).
"""
import json
import sys

import torch
import torch.nn as nn

sys.path.insert(0, r"C:\Users\afifh\Desktop\ATML\PA1")
from shared import pacs, pacs_protocol as proto
from shared.config import load_config
from shared.models import build_model, freeze_bn_stats
from shared.training import Batch
from task2.methods import build_method

cfg_name, max_steps, disc_bn, out = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1", sys.argv[4]
cfg = load_config(r"C:\Users\afifh\Desktop\ATML\PA1\task2\configs", cfg_name)
if disc_bn:
    cfg["adversarial"]["disc_batchnorm"] = True
src, tgt = proto.load_source_data(), pacs.load_target_images()
dom = list(src["train"])
model, method = build_model(), build_method(cfg)
dev = "cuda"
model.to(dev); method.to(dev)
mult = float(getattr(method, "lr_multiplier", 1.0))
opt = torch.optim.AdamW([{"params": list(model.parameters()), "lr": 1e-4},
                         {"params": list(method.parameters()), "lr": 1e-4 * mult}], lr=1e-4, weight_decay=1e-4)
s = proto.BatchSampler({d: len(src["train"][d][1]) for d in dom}, 8, len(tgt), 24)
total = 30 * 235
model.train(); freeze_bn_stats(model); method.train()          # backbone BN frozen; discriminator BN (if any) trains
first = method.disc.net[0]
hidden_in = {}
first.register_forward_hook(lambda m, i, o: hidden_in.__setitem__("pre", o.detach()))
rows = []
for step in range(max_steps):
    idx = s.source_indices()
    xs = proto.normalize(proto.random_crop_flip(torch.cat([src["train"][d][0][idx[d]] for d in dom]), s.src_aug).to(dev))
    ys = torch.cat([src["train"][d][1][idx[d]] for d in dom]).to(dev)
    xt = proto.normalize(proto.random_crop_flip(tgt[s.target_indices()], s.tgt_aug).to(dev))
    logs = method.step(model, opt, Batch(xs, ys, None, xt), step / total)
    if step % 25 == 0 or logs["domain_loss"] > 50:
        with torch.no_grad():
            f, z = model(torch.cat([xs, xt]))
            g = method.features_for_discriminator(f, z)
            dl = method.disc(g)
        rows.append({"step": step, "epoch": step / 235, "alpha": logs["alpha"], "cls_loss": logs["cls_loss"],
                     "domain_loss": logs["domain_loss"], "disc_acc": logs["disc_acc"],
                     "feat_norm_src": f[:24].norm(dim=1).mean().item(), "feat_norm_tgt": f[24:].norm(dim=1).mean().item(),
                     "disc_hidden_pre_absmean": hidden_in["pre"].abs().mean().item(), "disc_logit_absmax": dl.abs().max().item(),
                     "disc_w1_norm": first.weight.norm().item(), "head_w_norm": model.head.weight.norm().item(),
                     "layer4_w_norm": sum(p.norm() ** 2 for p in model.backbone.layer4.parameters()).sqrt().item()})
        r = rows[-1]
        print(f"step {step:5d} ep {r['epoch']:.2f} a {r['alpha']:.3f} cls {r['cls_loss']:.3f} dom {r['domain_loss']:.3f} "
              f"dacc {r['disc_acc']:.2f} |f|s {r['feat_norm_src']:.1f} |f|t {r['feat_norm_tgt']:.1f} "
              f"hid {r['disc_hidden_pre_absmean']:.2f} logit {r['disc_logit_absmax']:.1f} |W1| {r['disc_w1_norm']:.1f}", flush=True)
        if logs["domain_loss"] > 50:
            print("DIVERGED at step", step, flush=True)
            break
json.dump(rows, open(out, "w"), indent=1)
