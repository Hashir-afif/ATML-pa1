"""Task 2 methods. ``build_method(cfg)`` maps a resolved config to a Method instance."""
from task2.methods.cdan import CDAN
from task2.methods.dan import DAN
from task2.methods.dann import DANN
from task2.methods.source_only import SourceOnly


def build_method(cfg: dict):
    m, seed = cfg["method"], cfg["seed"]
    if m == "source_only":
        return SourceOnly()
    if m == "dan":
        return DAN(cfg["dan"]["lambda_mmd"], cfg["dan"]["bandwidth_multipliers"])
    if m == "dann":
        d = cfg["adversarial"]
        return DANN(d["hidden"], d["dropout"], d["max_grl"], d["gamma"], d["domain_loss_weight"], seed,
                    d.get("disc_lr_multiplier", 1.0), d.get("disc_batchnorm", False))
    if m == "cdan":
        d = cfg["adversarial"]
        return CDAN(cfg["model"]["num_classes"], d["hidden"], d["dropout"], d["max_grl"], d["gamma"], d["domain_loss_weight"], seed,
                    d.get("disc_lr_multiplier", 1.0), d.get("disc_batchnorm", False))
    raise ValueError(f"unknown method {m!r}")
