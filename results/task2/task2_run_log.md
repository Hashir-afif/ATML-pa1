# Task 2 run log: every run, failure, diagnostic and rerun (2026-09-22)

This log lists **everything** that was executed for Task 2, including failed and superseded runs, so each
reported number can be placed in context. Times are local (PKT). Checkpoints are in `task2/checkpoints/`
(git-ignored); histories, summaries and configs are exported in `results/task2/results.json`. Decisions and their
timing are in `results/task2/task2_decision_log.json`.

## 1. Chronology

| # | Time | What was run | Outcome | Evidence | Used in results? |
|---|---|---|---|---|---|
| 0 | before 15:38 | Smoke test (2 epochs × 5 updates per run, random stand-in labels) | all 5 notebooks passed | `task2/results/_smoke/` (git-ignored) | no |
| 1 | 15:38 | Full pipeline start (notebooks 01 → 04) | `source_only` finished: best epoch 6/11, source-val F1 0.932 | `checkpoints/source_only/` | **yes** (baseline, also the Task 3 ERM) |
| 2 | 15:41–15:45 | `dan` (λ=1) | finished: best 5/10, F1 0.943 | `checkpoints/dan_lambda1/` | **yes** |
| 3 | 15:45–15:48 | `dann`, manual settings, **attempt 1** | **DIVERGED** in epoch 1: domain loss 1459, source-val F1 0.07. Pipeline **stopped by hand** in epoch 2, *before notebook 03 could read target labels* | `checkpoints/_failed_attempts/dann_attempt1_diverged/` | no (failure record) |
| 4 | 15:56 | Diagnostic, 120 updates, manual DANN | feature norm ≈24 → 64; domain loss rises above ln 2 from α ≈ 0.06 | `results/diagnostics/task2_diag_dann.json`, `diagnostics/diag_dann_short.py` | evidence only |
| 5 | 15:57–16:06 | Batch 2: `cdan` (manual), `dan_lambda0.1`, `dan_lambda10` | `cdan`: **UNSTABLE**, source-val F1 0.88 → 0.90 → 0.15 → 0.83 → 0.36 → 0.05 → 0.32 (best epoch 2). `dan_lambda0.1`: finished, best 4/9, F1 0.940. `dan_lambda10`: **COLLAPSED** in epoch 1 (uniform predictions, F1 0.051) and stayed collapsed; this is the controlled-study outcome and was not "fixed" | `checkpoints/{cdan,dan_lambda0.1,dan_lambda10}/` | yes (as-specified CDAN row; λ study) |
| 6 | 16:01 | Diagnostic, 300 updates, DANN with discriminator lr ×10 | stable over 300 updates (domain loss 0.3–0.7) | `results/diagnostics/task2_diag_dann_disclr10.json` | evidence only |
| 7 | 16:06–16:20 | Batch 3: `dann` manual (**attempt 2**, complete), `dann_disclr10`, `cdan_disclr10` | `dann` attempt 2: **DIVERGED** again (epoch-1 F1 0.44, then losses up to ≈10¹¹); best epoch 1. `dann_disclr10`: best 8/13, F1 0.948, then **DIVERGED in epoch 10** (fails the stability criteria). `cdan_disclr10`: stable, best 6/11, F1 0.943 | `checkpoints/{dann,dann_disclr10,cdan_disclr10}/` | yes (as-specified DANN row; ×10 rows) |
| 8 | 16:22 | Notebooks 00–04; **target label read #1** | first evaluation (main rows then = disc lr ×10 variants) | access log line 1 | superseded by #12 |
| 9 | 16:26 | Notebooks 00–04 rerun to fix two figure layouts; **target label read #2** | method-comparison table byte-identical to #8 | access log line 2 | superseded by #12 |
| 10 | 17:08–17:16 | Post-evaluation diagnostic, 3000 updates, DANN ×10 (user asked to find the cause) | no divergence within 3000 updates this time, but the **discriminator died**: from ≈update 1700 its output is exactly 0, domain loss exactly ln 2, first-layer weight norm frozen at 13.7 (no gradient). Feature norms grew to 100–300 | `results/diagnostics/task2_diag_dann10_long.json`, `diagnostics/diag_adversarial.py` | evidence only |
| 11 | 17:17–17:23 | Diagnostics, 1500 updates, discriminator with BatchNorm (manual lr) for DANN and CDAN | no explosion or dead units; DANN discriminator often *below* chance in epochs 1–4 | `results/diagnostics/task2_diag_{dann,cdan}_discbn.json` | evidence only |
| 12 | 17:24–17:32 | Batch 4: `dann_discbn`, `cdan_discbn` (post-evaluation rerun) | both **pass** the pre-set stability criteria. `dann_discbn`: best 6/11, F1 0.944; `cdan_discbn`: best 4/9, F1 0.939 | `checkpoints/{dann_discbn,cdan_discbn}/` | **yes: main DANN/CDAN rows** (rule fixed before these runs finished) |
| 13 | 17:34 | Notebooks 00–04 with the final run set; **target label read #3** | final results | access log line 3; `results/task2/` | **yes: final** |

## 2. Failure analysis summary

| Failure | Mechanism (source-side evidence) | Response |
|---|---|---|
| DANN (manual) diverges | With AdamW lr 1e-4 everywhere and frozen backbone BatchNorm, nothing bounds the 512-d feature scale. The reversed gradient lets the backbone win against an unnormalised ReLU discriminator by inflating features, so domain cross-entropy explodes (≈10³–10¹¹) | kept as a reported failure (2 attempts) |
| CDAN (manual) collapses repeatedly | Same unnormalised discriminator; the reversed gradient also reaches the classifier through the undetached p in vec(f⊗p), as the manual requires | kept as a reported failure |
| DANN, disc lr ×10, diverges at epoch 10 | Faster discriminator delays the backbone's win, but the ReLU units eventually **all die** (output 0 ⇒ "accuracy 0.5" is a dead discriminator, not confusion); reactivation at α ≈ 0.92 with large features ⇒ divergence | reported; fails pre-set criteria |
| DAN λ=10 collapses | MMD term dominates: features become uninformative (predicts *person* for every Sketch image) | not a bug; controlled-study outcome |

**Fix applied (post-evaluation):** `BatchNorm1d` after the discriminator's first linear layer (as in common public
DANN/CDAN code), discriminator learning rate back to the manual 1e-4. Hidden pre-activations are normalised per
batch, so feature scale cannot be exploited and ReLU units cannot all die. The backbone's BatchNorm stays frozen,
as the manual requires. `shared.models.set_train_mode` freezes BN statistics only in the pretrained network, and
the discriminator's BN has no pretrained statistics.

## 3. Integrity notes

- Every stabilisation decision is based on source data and unlabelled target images only: training losses,
  source-validation F1, feature norms and discriminator state.
- Runs 1–7 and the first two fixes were decided **before** any target label was read.
- The BatchNorm fix (#10–#12) came **after** target labels had already been read (reads #1–#2). It was triggered by the user's request to investigate the instability. Its selection rule was written before
  its runs finished (decision log). **It was not chosen for Sketch accuracy, and it turned out worse on Sketch
  than the ×10 variants** (DANN 65.2% vs 71.1%; CDAN 51.9% vs 76.6%). Per the pre-set rule it is still the main
  row, and all variants are reported.
- Target labels were read 3 times in total (16:22:36, 16:26:06, 17:34:33), each after a freeze manifest was
  written and verified (`results.json → target_label_access_log`).
