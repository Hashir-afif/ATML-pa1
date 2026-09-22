# Task 3: Domain Generalization (PACS; Sketch unseen)

## Purpose

Can a model trained only on Photo, Art Painting and Cartoon generalise to Sketch, a domain it never
sees during training or model selection? Three hypotheses are compared:

- **ERM:** diverse labelled sources are enough.
- **DAN-DG:** removing differences between the observed source domains helps.
- **SAM:** a locally flat source solution transfers better.

The controlled study is **DAN-DG λ_DG ∈ {0.1, 1, 10}**. It mirrors Task 2's DAN λ study, which
supports the comparison of target-aware DAN (Task 2) with target-free DAN-DG (Task 3). This choice
was made for that structural reason; Task 2 results played no part (see §Integrity).

## Protocol: identical to Task 2

`task3/configs/base.yaml` is a byte copy of `task2/configs/base.yaml`, and notebook 00 asserts the two are
equal:

- **Data and model:** same source splits (`shared/splits/pacs_sketch_seed6304.json`), ResNet-18
  `IMAGENET1K_V1` initialisation and 7-class head.
- **Training:** same preprocessing and augmentation, domain-balanced batches (8 per source), AdamW
  (lr 1e-4, wd 1e-4), frozen BatchNorm running statistics.
- **Budget and selection:** ≤ 30 epochs × 235 updates, early stopping and checkpoint selection on mean
  source-validation macro-F1 (patience 5), seed 6304.

All methods run through the same loop, `shared/training.py`.

## Methods

| Method | Implementation |
|---|---|
| ERM | **Not retrained.** The Task 2 Source-only checkpoint `task2/checkpoints/source_only/best.pt` is loaded. Its SHA-256 and config are checked in notebook 00 (`task3_erm_reference.json`). |
| DAN-DG (`methods/dan_dg.py`) | CE + (λ_DG/3)·Σ_{e<e'} MMD²(F(X_e), F(X_e')) on the 512-d feature, λ_DG = 1. The MMD is **the same function as Task 2's DAN** (`shared.losses.mk_mmd`, via `pairwise_mk_mmd`). Each domain pair's bandwidths come from that pair's combined 8 + 8 batch (0.5/1/2 × median squared distance). |
| SAM (`methods/sam.py`) | Standard, non-adaptive SAM, ρ = 0.05, with AdamW as the base optimiser. Pass 1 computes the gradient at θ; ε = ρ·g/‖g‖; pass 2 computes the gradient at θ+ε; then θ is restored and AdamW steps. Two forward/backward passes per batch. BN statistics stay frozen in both. Verified exactly (difference 0.0) against a hand-written reference step on the deterministic CPU path. |

## Diagnostics (source side, notebook 02)

- **Per-domain source validation:** accuracy and macro-F1 on each source validation domain, plus mean
  and worst.
- **Source-domain separability:** 512-d features of the three source validation sets, balanced to 334
  each (seed 6304), stratified 70/30 split, multinomial logistic regression (C = 1). Held-out accuracy;
  chance = 1/3.
- **Sharpness proxy:** Δ = L(θ+ε) − L(θ), where ε = 0.05·∇L/‖∇L‖ over all parameters, in eval mode.
  It is measured on a fixed batch of 32 validation images per source domain (seed 6304), chosen in
  notebook 00 before any training (`task3_sharpness_batch.json`).
- **Training curves:** classification loss, source-validation F1 (mean and worst), the DAN-DG MMD
  penalty (mean and per pair), and SAM's perturbed-minus-clean loss.

## Sketch isolation (stricter than Task 2)

- No Sketch image or label is loaded by notebooks 00–02, `task3/methods/` or `shared/training.py`.
  Notebook 00 enforces this with a static scan (`task3_no_sketch_access_check.json`). The training
  loop also refuses target images for these methods (`uses_target = False`).
- Notebook 02 writes `task3/cache/freeze_manifest.json`, which holds the SHA-256 of every checkpoint
  (including the reused ERM), config, run summary, the split and the sharpness batch.
- Notebook 03 is the **only** place Sketch is touched. It calls `pacs.load_target_labels(freeze_manifest)`,
  which re-hashes every frozen file, refuses on any change and logs the access. Only then does it load
  and predict the Sketch images.
- In smoke mode (`TASK3_SMOKE=1`), notebook 03 uses random stand-in images and labels.

## Pipeline

| # | Notebook | Touches Sketch? | Main outputs |
|---|---|---|---|
| 00 | `00_verify_protocol.ipynb` | no | protocol check, `task3_erm_reference.json`, `task3_sharpness_batch.json`, `task3_no_sketch_access_check.json` |
| 01 | `01_train_methods.ipynb` | no | `task3/checkpoints/<run>/…`, `task3_training_summary.csv` |
| 02 | `02_source_diagnostics.ipynb` | no | `task3_source_validation.csv`, `task3_domain_separability.csv`, `task3_sharpness.csv`, curves, freeze manifest |
| 03 | `03_evaluate_sketch.ipynb` | **yes (only here)** | `task3_method_comparison.csv`, class analysis, confusions, transfer cases, flips, failure examples, controlled study, **Task 2 comparison** (`task3_task2_comparison.csv`, `task3_task2_class_comparison.csv`) |
| 04 | `04_export_results.ipynb` | no | `results/task3/` (+ `results.json`), `figures/task3/` |

```bash
cd task3
for nb in 00_verify_protocol 01_train_methods 02_source_diagnostics 03_evaluate_sketch 04_export_results; do
  ../.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.kernel_name=pa1 --ExecutePreprocessor.timeout=36000 $nb.ipynb || break
done
```

- `TASK3_RUNS=dan_dg,sam` trains a subset. Finished runs are skipped and never overwritten.
- Smoke test: `TASK3_SMOKE=1` (2 epochs × 5 updates, outputs under `_smoke/`, no Sketch).
- Expected cost: 4 trained runs. DAN-DG costs about the same as ERM (~12 s per epoch). SAM costs
  about twice that, because it does two passes. With early stopping, the total is roughly 20–40 min
  on an RTX 4060 Laptop.

## Manual structure mapping

| Manual suggestion | Here |
|---|---|
| `task3/configs/{erm,dan_dg,sam}.yaml` | same (+ `base.yaml`, the two λ-study configs, `evaluation.yaml`) |
| `task3/models/backbone.py`, `classifier_head.py` | `shared/models.py` (shared with Task 2) |
| `task3/methods/{erm,dan_dg,sam}.py` | same |
| `task3/selection/source_validation.py` | `shared/training.py` (selection on mean source-val macro-F1) + notebook 02 |
| `task3/evaluation/domain_metrics.py`, `source_domain_separability.py`, `sharpness.py` | `shared/evaluation.py` (`evaluate_sources`, `domain_separability`, `sharpness_proxy`) |
| `task3/train.py` | `shared/training.py` + `01_train_methods.ipynb` |
| `task3/evaluate_sketch.py` | `03_evaluate_sketch.ipynb` (kept separate from source-side selection, as the manual asks) |

## Integrity

- **Settings are the manual's:** λ_DG = 1 and ρ = 0.05 for the main comparison. They are never
  replaced by a post-hoc study winner.
- **Task 2 results influence nothing:** no Task 3 setting depends on Task 2's Sketch results. The
  study choice (DAN-DG λ) rests on the pairing with Task 2's DAN λ study, which was fixed before any
  Sketch label existed. Task 2's result files are read only in notebook 03, for the required
  comparison.

## External code

SAM, DAN-DG and the sharpness proxy are written for this repository from Foret et al. (2021), Long et al.
(2015) and the manual. No code was copied.
