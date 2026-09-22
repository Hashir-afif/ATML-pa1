# Task 2: Unsupervised Domain Adaptation (PACS, target = Sketch)

## Purpose

This task tests whether unlabelled Sketch images help a ResNet-18 trained on Photo, Art Painting and
Cartoon. It compares Source-only ERM with marginal alignment (DAN: MMD; DANN: adversarial) and
class-conditional alignment (CDAN). It then checks whether lower domain separability goes with
better target recognition, and runs one controlled alignment-strength study: **DAN λ ∈ {0.1, 1, 10}**.

## Data and protocol (manual §2; shared with Task 3)

| Item | Setting |
|---|---|
| Dataset | PACS, Hugging Face mirror [`flwrlabs/pacs`](https://huggingface.co/datasets/flwrlabs/pacs) (one parquet file, SHA-256 `4fc041ee…5ff5`; domain counts match the PACS paper: 1670 / 2048 / 2344 / 3929) |
| Sources / target | Photo, Art Painting, Cartoon (labelled) → Sketch (all 3929 images; unlabelled during training) |
| Split | Stratified 80/20 train/validation **within each source domain**, seed 6304 → `shared/splits/pacs_sketch_seed6304.json` |
| Preprocessing | Resize to 256×256 once (cached); training: random 224 crop + horizontal flip; evaluation: centre 224 crop; ImageNet normalisation |
| Model | torchvision ResNet-18 `IMAGENET1K_V1`, new 7-class linear head, full fine-tuning |
| BatchNorm | Running mean/variance frozen at ImageNet values (`model.train()`, then only the BN modules set to `eval()`); γ/β trainable |
| Optimiser | AdamW, lr 1e-4, weight decay 1e-4 (model and discriminator parameters) |
| Budget | ≤ 30 epochs; 1 epoch = ⌈largest source-train domain / 8⌉ = ⌈1875/8⌉ = 235 updates (decision recorded 2026-09-22) |
| Early stopping / selection | Best mean source-validation macro-F1; stop after 5 epochs without improvement |
| Batches | 8 images per source domain (24) + 24 target images for adaptation methods; each domain is an endless seeded permutation stream (cycled) |
| Seed | 6304 for the split, the initialisation, the sampling streams, the augmentation streams, separability and example picks |

Source sampling and source augmentation use their own random generators. The Source-only, DAN,
DANN and CDAN runs therefore see **exactly the same source images with the same crops and flips**;
only the loss differs.

## Methods (`task2/methods/`)

| Method | Loss |
|---|---|
| Source-only (`source_only.py`) | CE on the 24 source images. **Never receives Sketch images**, so its checkpoint is also the Task 3 ERM baseline. |
| DAN (`dan.py`) | CE + λ · MK-MMD² between the 24 source and 24 target 512-d features. Biased estimator; sum of RBF kernels `exp(-d²/(m·median))`, m ∈ {0.5, 1, 2}; median pairwise squared distance of the combined batch, treated as a constant (`shared/losses.py`). |
| DANN (`dann.py`) | CE + 1 · domain CE. Discriminator 512→256 (ReLU, dropout 0.5)→2 behind a gradient-reversal layer, α(p) = 2/(1+e^(−10p)) − 1, p = step / (30 × 235). |
| CDAN (`cdan.py`) | Same as DANN, but the discriminator input is vec(f ⊗ p) (512×7 = 3584-d), with p = softmax(logits). No entropy conditioning; neither f nor p is detached. |

## Deviation: DANN/CDAN stabilisation (full record in [RUN_LOG.md](RUN_LOG.md))

With the manual's exact settings (AdamW lr 1e-4 for every parameter, frozen backbone BatchNorm), the adversarial
methods were unstable. **DANN diverged** in both attempts, with domain loss up to ≈10¹¹. **CDAN collapsed
repeatedly** (source-val F1 swinging between 0.05 and 0.90). Source-side diagnostics (`diagnostics/`,
`results/diagnostics/`) located the cause: the 512-d feature scale is unbounded, so the backbone beats an
unnormalised ReLU discriminator. It either inflates features until the domain loss explodes, or drives every
discriminator ReLU unit dead. A dead discriminator outputs exactly 0, reports "accuracy 0.5" and passes no
gradient.

Two stabilisations were tried. Both were chosen from source-side evidence only and checked against criteria
fixed in advance: every epoch must have a mean domain loss below 2.0 and a source-val macro-F1 of at least 0.5.

| Config | Discriminator | Discriminator lr | Stable? | Role |
|---|---|---|---|---|
| `dann`, `cdan` | manual (no BN) | 1e-4 (manual) | no / no | reported "as specified" rows |
| `dann_disclr10`, `cdan_disclr10` | manual (no BN) | 1e-3 | no (diverged at epoch 10) / yes | first attempt, decided before target evaluation; reported |
| `dann_discbn`, `cdan_discbn` | **BatchNorm1d after the first layer** | 1e-4 (manual) | yes / yes | **main DANN/CDAN rows**; post-evaluation rerun after the root-cause diagnosis |

The BatchNorm variant became the main row under a rule written before its runs finished. It turned out **worse on
Sketch** than the lr ×10 variant. The main row was not switched back, because doing so would mean selecting by
target labels. Options: `adversarial.disc_lr_multiplier` (default 1.0) and `adversarial.disc_batchnorm` (default
false). With the defaults, the code reproduces the manual's settings exactly.

## Target-label protection

1. `shared/pacs.py` writes Sketch labels to a separate file. The only function that returns them is
   `load_target_labels(freeze_manifest)`.
2. Notebook 02 computes every target prediction (no labels are needed), then writes
   `task2/cache/freeze_manifest.json` with the SHA-256 of every checkpoint, config, run summary,
   prediction file and the split file.
3. `load_target_labels` re-hashes all of them and refuses if anything changed. Every access is
   appended to `data/pacs/target_label_access_log.jsonl`.
4. Notebook 03 is the only caller. Nothing it computes feeds back into any setting.

## Pipeline

| # | Notebook | Uses Sketch labels? | Main outputs |
|---|---|---|---|
| 00 | `00_prepare_pacs.ipynb` | no | `data/pacs/*` cache, split file, `task2_data_summary.json`, `task2_example_images.png` |
| 01 | `01_train_methods.ipynb` | no | `task2/checkpoints/<run>/{best.pt, history.json, summary.json, config.yaml}`, `task2_training_summary.csv` |
| 02 | `02_source_diagnostics.ipynb` | no | `task2_source_validation.csv`, `task2_domain_separability.csv`, training/alignment curves, frozen target predictions and freeze manifest |
| 03 | `03_evaluate_target.ipynb` | **yes (only here; 3 logged reads, see RUN_LOG.md)** | `task2_method_comparison.csv`, `task2_class_analysis.csv`, `task2_confusions.csv`, `task2_transfer_cases.csv`, `task2_prediction_flips.csv`, `task2_failure_examples.csv`, `task2_controlled_study.csv` + figures |
| 04 | `04_export_results.ipynb` | no | `results/task2/` (+ `results.json`), `figures/task2/` |

### Commands (from the repository root)

```bash
cd task2
for nb in 00_prepare_pacs 01_train_methods 02_source_diagnostics 03_evaluate_target 04_export_results; do
  ../.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.kernel_name=pa1 --ExecutePreprocessor.timeout=36000 $nb.ipynb || break
done
```

- Train a subset: `TASK2_RUNS=source_only,dan` before notebook 01. Finished runs are skipped and never overwritten.
- Smoke test: `TASK2_SMOKE=1` gives 2 epochs × 5 updates per run. Outputs go under `_smoke/`, and
  notebook 03 uses **random** labels instead of the real Sketch labels.
- Expected cost: 10 runs, roughly 2–6 min each (early stopping) on an RTX 4060 Laptop GPU. Disk: ~2 GB image cache
  plus ~45–50 MB per checkpoint, all git-ignored.

## Configuration

`configs/base.yaml` holds the shared protocol, and each method file (`source_only.yaml`, `dan.yaml`,
`dann.yaml`, `cdan.yaml`, `dann_disclr10.yaml`, `cdan_disclr10.yaml`, `dann_discbn.yaml`, `cdan_discbn.yaml`,
`dan_lambda0.1.yaml`, `dan_lambda10.yaml`) adds only its own settings.
The resolved config of every run is saved as `task2/checkpoints/<run>/config.yaml`.

## Code layout and mapping to the manual's suggested structure

The manual asks for one PACS protocol and training loop shared by Tasks 2 and 3. They therefore
live in importable modules (`shared/`), and the notebooks drive them.

| Manual suggestion | Here |
|---|---|
| `shared/pacs.py`, `shared/pacs_protocol.py`, `shared/splits/pacs_sketch_seed6304.json` | same files |
| `task2/models/backbone.py`, `classifier_head.py`, `domain_discriminator.py` | `shared/models.py` (`PACSNet`, `DomainDiscriminator`, `grad_reverse`) |
| `task2/methods/{source_only,dan,dann,cdan}.py` | same files |
| `task2/train.py` | `shared/training.py` (common loop) + `01_train_methods.ipynb` |
| `task2/evaluation/metrics.py`, `domain_separability.py` | `shared/evaluation.py` |
| `task2/evaluation/class_analysis.py`, `evaluate_final.py` | `03_evaluate_target.ipynb` |
| `task2/configs/*.yaml` | same files |

## External code and data attribution

- PACS: Li et al., *Deeper, Broader and Artier Domain Generalization* (ICCV 2017), via the
  `flwrlabs/pacs` Hugging Face mirror.
- The gradient-reversal layer, DANN schedule, CDAN multilinear conditioning and MK-MMD are written
  for this repository from the papers (Ganin et al. 2016; Long et al. 2018; Long et al. 2015). No
  code was copied.
- ResNet-18 weights: torchvision `ResNet18_Weights.IMAGENET1K_V1`.

## Reproducibility notes

- GPU kernels are not bit-deterministic, so reruns may differ slightly in the last digits.
- Separability uses unscaled 512-d features with `LogisticRegression(C=1, class_weight="balanced",
  max_iter=5000)`. The number of lbfgs iterations is recorded.
