"""Task 3 methods. ERM is not trained here (it is the Task 2 Source-only checkpoint)."""
from task3.methods.dan_dg import DANDG
from task3.methods.sam import SAM


def build_method(cfg: dict):
    m = cfg["method"]
    if m == "dan_dg":
        return DANDG(cfg["dan_dg"]["lambda_dg"], cfg["dan_dg"]["bandwidth_multipliers"])
    if m == "sam":
        return SAM(cfg["sam"]["rho"])
    if m == "erm":
        raise ValueError("ERM is loaded from the Task 2 Source-only checkpoint, never trained in Task 3")
    raise ValueError(f"unknown method {m!r}")
