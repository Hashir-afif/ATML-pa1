# Task 4: Open-Set Recognition (CIFAR-10 known, CIFAR-100 unknown)

## Purpose

A closed-set classifier must label every input as one of its ten known classes. This task adds the ability to
**reject** inputs that belong to no known class, and asks how rejection relates to closed-set accuracy. It compares
four post-hoc novelty scores on one frozen model, then three models (Vanilla, GCSC, PROSER) with a common score,
and separates **near** unknowns (semantically close to CIFAR-10) from **far** unknowns.

## Data

| Item | Setting |
|---|---|
| Known | CIFAR-10, all 10 classes (HF mirror `uoft-cs/cifar10`, checksums in `task4_data_summary.json`) |
| Split | stratified 90/10 of the official training partition, seed 6304 → 45 000 train / 5 000 validation (`task4_cifar10_split.json`) |
| Closed-set test | the full official CIFAR-10 test set (10 000) |
| Unknowns | CIFAR-100 **test** split only: near = bus, pickup_truck, motorcycle, tractor, wolf, fox, leopard, camel; far = bottle, bowl, chair, clock, keyboard, mushroom, sunflower, wardrobe (800 images per group) |
| Not downloaded | the CIFAR-100 **training** parquet is never fetched, so its images cannot be used even by accident |

Validation data does double duty, as the manual requires: it selects checkpoints **and** calibrates the rejection
thresholds. No unknown example takes part in either.

## Model and training

- **CIFAR ResNet-18** (`models/resnet_cifar.py`): torchvision ResNet-18 with a 3×3 stride-1 first convolution and
  the initial max-pool removed; 32×32 inputs; 512-d penultimate feature. PROSER appends 5 dummy outputs, and the
  known-class logits are always the first ten columns.
- **Recipe (all runs):** random 32×32 crop with 4-pixel padding + horizontal flip, SGD lr 0.1, momentum 0.9, weight
  decay 5e-4, cosine decay, batch 128, 100 epochs, seed 6304; the checkpoint with the highest CIFAR-10 validation
  accuracy is kept.
- **GCSC:** the same, with `RandAugment(num_ops=2, magnitude=9)` inserted after the crop and flip and before
  normalisation. Everything else is unchanged.
- **PROSER:** initialised from the selected Vanilla checkpoint (all weights copied; the classifier keeps its ten
  rows and gains five randomly initialised dummy rows), then fine-tuned for 50 epochs with SGD lr 1e-3, β = 1,
  γ = 0.1, manifold mixup after `layer2` with λ ~ Beta(2,2). Each mini-batch is split in half: the first half trains
  classifier placeholders, the second half trains data placeholders.

## Scores (all unknownness: larger = more novel)

| Score | Definition | Used on |
|---|---|---|
| MSP | 1 − max_k softmax(z)_k | Vanilla (comparison 1) |
| MLS | −max_k z_k | Vanilla, GCSC, PROSER (comparison 2) |
| Energy | −log Σ_k exp(z_k) | Vanilla |
| Mahalanobis | min_c (f − μ_c)ᵀ Σ⁻¹ (f − μ_c) | Vanilla; μ_c and one shared **diagonal** Σ (+1e-6) fitted on **unaugmented CIFAR-10 training** features |
| PROSER placeholder | max dummy logit − max known logit | PROSER (extra row) |

All four post-hoc scores read the *same* saved logits and features of the frozen Vanilla model.

**Threshold:** for every (model, score), τ = the 95th percentile of unknownness on the CIFAR-10 **validation** set;
an example is accepted when u(x) ≤ τ. **Metrics:** AUROC (known vs near, far, all), CIFAR-10 test acceptance rate,
near/far rejection rates, FPR@95TPR. PROSER's CSA uses only the ten known logits.

## Unknown-data protection

1. Notebooks 00–02 never decode a CIFAR-100 image; notebook 00 only records the fixed class lists and the test
   parquet's checksum.
2. Notebook 02 writes `task4/cache/freeze_manifest.json` with the SHA-256 of every checkpoint, config, run summary,
   saved output, the Mahalanobis state and the threshold table.
3. `task4.data.cifar100_unknowns.load_unknowns(manifest)` is the only way to obtain unknown images. It re-hashes
   every frozen file, refuses on any change, and appends each access to `data/cifar/unknown_access_log.jsonl`.
4. Notebook 03 is the only caller, and nothing it computes may revise a model, a score or a threshold.
5. In smoke mode (`TASK4_SMOKE=1`), notebook 03 uses random stand-in images instead.

## Pipeline

| # | Notebook | Touches CIFAR-100? | Main outputs |
|---|---|---|---|
| 00 | `00_prepare_data.ipynb` | no | CIFAR-10 cache, 90/10 split, `task4_data_summary.json`, `task4_cifar10_examples.png` |
| 01 | `01_train.ipynb` | no | `task4/checkpoints/<run>/…`, `task4_training_summary.csv` |
| 02 | `02_extract_outputs.ipynb` | no | frozen features/logits, CSA, Mahalanobis fit, `task4_thresholds.csv`, freeze manifest |
| 03 | `03_evaluate_osr.ipynb` | **yes (only here)** | `task4_score_comparison.csv`, `task4_osr_metrics.csv`, `task4_unknown_class_analysis.csv`, `task4_failure_cases.csv` + figures |
| 04 | `04_export_results.ipynb` | no | `results/task4/` (+ `results.json`), `figures/task4/` |

```bash
cd task4
for nb in 00_prepare_data 01_train 02_extract_outputs 03_evaluate_osr 04_export_results; do
  ../.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.kernel_name=pa1 --ExecutePreprocessor.timeout=86400 $nb.ipynb || break
done
```

- `TASK4_RUNS=vanilla` trains a subset (PROSER needs the Vanilla checkpoint first). Finished runs are skipped.
- `TASK4_SMOKE=1`: 2 epochs on 2 000 images, outputs under `_smoke/`, random stand-in unknowns.
- **Cost (measured on an RTX 4060 Laptop):** 110 ms per training iteration, 352 iterations per epoch.
  Vanilla ≈ 64 min, GCSC ≈ 110 min (RandAugment adds ~74 ms per batch on the CPU), PROSER ≈ 40 min.
  **Total ≈ 3.5 h**, plus a few minutes for notebooks 02–04. Disk: ~340 MB data cache, ~45 MB per checkpoint,
  ~1 GB of saved features/logits, all git-ignored.

## Manual structure mapping

| Manual suggestion | Here |
|---|---|
| `configs/{vanilla,gcsc,proser,rpl}.yaml` | `configs/{base,vanilla,gcsc,proser}.yaml` (RPL not implemented, see below) |
| `methods/{vanilla,gcsc,proser,manifold_mixup}.py` | same files |
| `data/{cifar10,cifar100_unknowns,make_splits}.py` | `data/cifar10.py` (includes the split), `data/cifar100_unknowns.py` |
| `models/resnet_cifar.py` | same |
| `scores/{msp,mls,energy,mahalanobis}.py` | same |
| `evaluation/{metrics,thresholds,failure_analysis}.py` | `evaluation/metrics.py` (thresholds + metrics) + notebook 03 (failure analysis) |
| `train.py`, `extract_outputs.py`, `evaluate_osr.py` | `training.py` + notebooks 01, 02, 03 |
| `cache/` | `task4/cache/` (git-ignored saved features/logits) |

## Optional extension

**RPL is not implemented.** The manual marks it optional; the required methods (Vanilla, GCSC, PROSER) and all four
scores are complete.

## External code

The CIFAR ResNet-18 stem change, PROSER's two placeholder losses, manifold mixup, all four scores and the metric
code are written for this repository from the manual and the papers (Zhou et al. 2021; Vaze et al. 2022; Hendrycks
& Gimpel 2017; Liu et al. 2020). No code was copied. Datasets come from the Hugging Face mirrors `uoft-cs/cifar10`
and `uoft-cs/cifar100` (Krizhevsky, 2009).
