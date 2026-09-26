"""Task 4 evaluation: validation-calibrated thresholds, AUROC, acceptance/rejection rates, FPR@95TPR.

Convention: every score is an **unknownness** u(x); an example is *accepted* (called known) when u(x) <= tau.
The threshold tau is the 95th percentile of u on the CIFAR-10 **validation** set, so it aims to accept 95 % of known
examples. No unknown example takes part in choosing tau.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def validation_threshold(u_val: np.ndarray, percentile: float = 95.0) -> float:
    """tau = the given percentile of unknownness on CIFAR-10 validation data (known data only)."""
    return float(np.percentile(np.asarray(u_val, dtype=np.float64), percentile))


def auroc(u_known: np.ndarray, u_unknown: np.ndarray) -> float:
    """AUROC for separating unknown (positive) from known (negative) by unknownness."""
    y = np.concatenate([np.zeros(len(u_known)), np.ones(len(u_unknown))])
    s = np.concatenate([np.asarray(u_known, dtype=np.float64), np.asarray(u_unknown, dtype=np.float64)])
    return float(roc_auc_score(y, s))


def fpr_at_95tpr(u_known: np.ndarray, u_unknown: np.ndarray) -> float:
    """Fraction of unknowns accepted when the threshold accepts 95 % of knowns (the same convention as tau)."""
    tau = validation_threshold(u_known, 95.0)
    return float((np.asarray(u_unknown) <= tau).mean())


def rates(u_test_known: np.ndarray, u_unknown_groups: dict, tau: float) -> dict:
    """Acceptance rate on CIFAR-10 test and rejection rate per unknown group at the frozen threshold."""
    out = {"threshold": float(tau),
           "test_acceptance_rate": float((np.asarray(u_test_known) <= tau).mean())}
    for g, u in u_unknown_groups.items():
        out[f"{g}_rejection_rate"] = float((np.asarray(u) > tau).mean())
        out[f"{g}_acceptance_rate"] = float((np.asarray(u) <= tau).mean())
    return out


def osr_row(u_val: np.ndarray, u_test: np.ndarray, u_near: np.ndarray, u_far: np.ndarray,
            percentile: float = 95.0) -> dict:
    """All required numbers for one (model, score): AUROCs, validation-calibrated rates and FPR@95TPR."""
    tau = validation_threshold(u_val, percentile)
    u_all = np.concatenate([u_near, u_far])
    row = {"auroc_known_vs_near": auroc(u_test, u_near), "auroc_known_vs_far": auroc(u_test, u_far),
           "auroc_known_vs_all": auroc(u_test, u_all),
           "fpr_at_95tpr_near": fpr_at_95tpr(u_test, u_near), "fpr_at_95tpr_far": fpr_at_95tpr(u_test, u_far),
           "fpr_at_95tpr_all": fpr_at_95tpr(u_test, u_all)}
    row.update(rates(u_test, {"near": u_near, "far": u_far, "all": u_all}, tau))
    return row
