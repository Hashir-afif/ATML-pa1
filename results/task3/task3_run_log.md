# Task 3 run log: every run, failure, diagnostic and rerun

Every execution for Task 3 is recorded here, including failures and superseded runs (same convention as
`task2/RUN_LOG.md`).

| # | Date / time | What was run | Outcome | Evidence | Used in results? |
|---|---|---|---|---|---|
| 0 | 2026-09-22 | Unit checks: DAN-DG pairwise MMD = (1/3)·Σ pairs; SAM step vs hand-written reference (deterministic CPU: max diff 0.000); BN statistics frozen through both SAM passes; sharpness proxy restores parameters | all passed | this log | no |
| 1 | 2026-09-22 | Notebook 00 (real): protocol identical to Task 2; ERM checkpoint verified; sharpness batch fixed; static no-Sketch scan | passed; 0 Sketch-loader calls in 8 files | `results/tables/task3_{erm_reference,sharpness_batch,no_sketch_access_check}.json` | yes |
| 2 | 2026-09-22 | Smoke test of 01–04 (2 epochs × 5 updates; random stand-in Sketch data) | all passed; target-label access log unchanged (3 lines, all from Task 2) | `task3/results/_smoke/` (git-ignored) | no |
| 3 | 2026-09-22 18:35–18:45 | Notebook 01: train `dan_dg` (λ=1), `sam`, `dan_dg_lambda0.1`, `dan_dg_lambda10` | `dan_dg` (λ=1): **COLLAPSED in epoch 1** (classification loss ≈ ln 7, source-val F1 0.051) and stayed collapsed; best epoch 1/6. `sam`: best 5/10, F1 0.958. `dan_dg_lambda0.1`: best 5/10, F1 0.954. `dan_dg_lambda10`: **COLLAPSED** like λ=1, best 1/6. No run was re-trained or altered; the main comparison keeps λ_DG = 1 as the manual requires | `task3/checkpoints/*/` (git-ignored), `results/task3/results.json` | yes |
| 4 | 18:45 | Notebook 02: source validation, curves, source-domain separability, sharpness, freeze manifest | done; no Sketch access | `results/task3/task3_{source_validation,domain_separability,sharpness}.csv` | yes |
| 5 | 18:45:54 | Notebook 03: **first and only Task 3 Sketch access** (label read #4 in the shared log; manifest `task3/cache/freeze_manifest.json`) | ERM Sketch accuracy identical to Task 2 Source-only (0.67473), confirming the same checkpoint | `results/task3/task3_erm_consistency.json` | yes |
| 6 | 18:46 | Notebook 04: export | 18 result files, 18 figure files | `results/task3/`, `figures/task3/` | yes |
| 7 | after 6 | Source-side diagnostic of the DAN-DG collapse (240 updates, λ=1; no Sketch) | within ≈20 updates the 512-d features shrink to one point (per-dimension std 0.73 → 0.007) and 58% of dimensions die (95% by update 100); logits become identical for all images → uniform predictions. The biased MMD stays ≈0.35–1.0 (unbiased ≈ −0.1–0.6) because the median bandwidth is scale-invariant, so the penalty never stops pushing | `results/task3/diagnostics/task3_diag_dan_dg_lambda1.json`, `task3/diagnostics/diag_dan_dg_collapse.py` | evidence only |

## Failure record

| Run | What happened | Why (source-side evidence) | Action |
|---|---|---|---|
| DAN-DG λ=1 (main row) | collapsed in epoch 1; predicts *person* for every image | pairwise MMD over 8 + 8 features per pair with the biased (V-statistic) estimator required to be identical to Task 2: self-similarity terms add ≈3/8 per domain, so the penalty has a positive floor that only full collapse removes, and the median bandwidth makes the kernel scale-invariant, so shrinking features does not satisfy it; dimensions die (no gradient back) | **kept as specified** (manual: main comparison uses λ_DG = 1; same MMD as Task 2). Not "fixed": an unbiased estimator would break the "same MMD implementation as Task 2" requirement and would require re-training Task 2's DAN after its target labels were read |
| DAN-DG λ=10 | same collapse | same, stronger penalty | kept (study outcome) |

