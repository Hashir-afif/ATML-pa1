# Produced task2_diag_dann.json (config dann, 120 steps). task2_diag_dann_disclr10.json came from the same script with
# config dann_disclr10, 300 steps, and a two-group AdamW (discriminator lr x10).
"""Diagnostic (source + unlabelled target only): what diverges in the first DANN updates?
Logs per-step feature norm, discriminator logit magnitude, losses and backbone grad norm."""
import sys, json, torch
sys.path.insert(0, r"C:\Users\afifh\Desktop\ATML\PA1")
from shared import pacs, pacs_protocol as proto
from shared.config import load_config
from shared.models import build_model, set_train_mode
from shared.training import Batch
from task2.methods import build_method

cfg = load_config(r"C:\Users\afifh\Desktop\ATML\PA1\task2\configs", "dann")
src = proto.load_source_data(); tgt = pacs.load_target_images()
dom = list(src["train"])
model, method = build_model(), build_method(cfg)
dev = "cuda"; model.to(dev); method.to(dev)
opt = torch.optim.AdamW(list(model.parameters()) + list(method.parameters()), lr=1e-4, weight_decay=1e-4)
s = proto.BatchSampler({d: len(src["train"][d][1]) for d in dom}, 8, len(tgt), 24)
total = 30 * 235
set_train_mode(model, method)
rows = []
for step in range(120):
    idx = s.source_indices()
    xs = proto.normalize(proto.random_crop_flip(torch.cat([src["train"][d][0][idx[d]] for d in dom]), s.src_aug).to(dev))
    ys = torch.cat([src["train"][d][1][idx[d]] for d in dom]).to(dev)
    xt = proto.normalize(proto.random_crop_flip(tgt[s.target_indices()], s.tgt_aug).to(dev))
    with torch.no_grad():
        f, _ = model(torch.cat([xs, xt])); dl = method.disc(f)
    logs = method.step(model, opt, Batch(xs, ys, None, xt), step / total)
    gn = sum(p.grad.norm() ** 2 for p in model.backbone.parameters() if p.grad is not None).sqrt().item()
    rows.append({"step": step, "feat_norm": f.norm(dim=1).mean().item(), "disc_logit_absmax": dl.abs().max().item(),
                 "cls_loss": logs["cls_loss"], "domain_loss": logs["domain_loss"], "alpha": logs["alpha"], "backbone_grad_norm": gn})
    if step % 10 == 0 or step < 5:
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in rows[-1].items()}, flush=True)
json.dump(rows, open(r"C:\Users\afifh\AppData\Local\Temp\claude\c--Users-afifh-Desktop-ATML-PA1\fecfbb18-6f4f-4e53-986f-c7e9cbd64314\scratchpad\diag_dann.json", "w"), indent=1)
