# Task 4 run log: every run, failure, diagnostic and rerun

Same convention as `task2/RUN_LOG.md` and `task3/RUN_LOG.md`: everything executed for Task 4, including failures.

| # | Date / time | What was run | Outcome | Evidence | Used in results? |
|---|---|---|---|---|---|
| 0 | 2026-09-23 | Downloads: CIFAR-10 train/test + CIFAR-100 **test** parquet (HF mirrors) | CIFAR-10 train parquet first arrived **truncated** (84.7 MB, unreadable: "Parquet magic bytes not found"); resumed to 119.7 MB, verified 50 000 rows. CIFAR-100 *train* deliberately never downloaded | `task4_data_summary.json` (checksums) | yes (data) |
| 1 | 2026-09-23 | Unit checks: model stem (3x3 stride-1, no max-pool), PROSER losses, manifold-mixup partners all different-class, Mahalanobis, threshold/metric conventions | all passed (threshold accepts exactly 95 % of validation) | this log | no |
| 2 | 2026-09-23 | Notebook 00 (real): CIFAR-10 decode, 90/10 split, class counts, example figure | passed; 45 000 / 5 000 / 10 000 | `results/tables/task4_{data_summary.json,cifar10_class_counts.csv}` | yes |
| 3 | 2026-09-23 | Smoke test of 01–04 (2 epochs, 2 000 images, random stand-in unknowns) | all passed; CIFAR-100 access log still absent | `task4/results/_smoke/` (git-ignored) | no |
| 4 | 2026-09-23 | Speed measurement (RTX 4060): 110 ms/iteration; `cudnn.benchmark` no gain; `channels_last` 2x slower; RandAugment 74 ms/batch CPU (128 ms on GPU) | estimates: Vanilla ~64 min, GCSC ~110 min, PROSER ~40 min | this log | no |
| 5 | 2026-09-23 → 2026-09-24 01:18:50 | Full run of notebooks 01–04 (overnight supervisor, 0 retries, 2 h 55 min total) | SUCCESS. Vanilla 47.3 min, best epoch 95, val 95.26 %, test CSA 94.61 %; GCSC 99.7 min, best epoch 99, val 95.68 %, CSA 95.43 %; PROSER 23.2 min, **best epoch 1 of 50** (val 94.86 %; val acc fell to ≈91 % by epoch 5 as the placeholders started to win on mixed samples), CSA 94.40 %. Freeze manifest 01:15:54; single CIFAR-100 access 01:16:59 (after the freeze) | `results/task4/` (results.json + CSVs), `figures/task4/` | yes |
