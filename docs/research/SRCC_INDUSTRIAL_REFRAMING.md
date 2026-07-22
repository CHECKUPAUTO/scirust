# Phase-728 Re-framing — acting on the diagnostic

The [phase-728 diagnostic](SRCC_INDUSTRIAL_DIAGNOSTIC.md) showed the central
contamination nulls were **framing** results, not robustness-power deficits,
and named three levers. This document records each lever as it lands. Every
run reuses the frozen phase-728 split seeds, fractions and missing-value
policy (`configs/phase728.json`) so a lever's effect is isolated; each is
exploratory re-framing, not a new preregistered test, deterministic (run
twice, byte-identical).

## Lever 1 — piecewise RUL + reduced decimation

**Change.** Two isolated modifications to the C-MAPSS regression framing:

- the run-to-failure target is capped at the canonical piecewise-linear knee
  `RUL_pw = min(RUL, 125)` (Heimes 2008) — early-life cycles carry no
  degradation signal, so the raw-linear target imposes an unlearnable burden
  the diagnostic measured as a low task ceiling;
- decimation is reduced from stride 20 to stride 5, recovering ~4× the
  training rows the diagnostic flagged as a sacrificed statistical power.

**Binary.** `scirust-srcc-bench/src/bin/industrial-reframed-rul.rs`, plus the
pure `clip_rul_targets` library primitive (tested). A 2×2 grid of {raw,
piecewise} × {stride 20, stride 5} reports the predict-the-mean RMSE (task
ceiling), each method's clean-fit test RMSE, and the ceiling ratio
`clean_fit / predict_mean` — lower means the features explain more of the
target.

```
SHA-256 (scientific stdout, nightly-2026-07-02, x86_64):
9662938a30296ac6224333f500f2d88b60ff5c1ce5ed781efe2ebe9dcbaf99d9
records: results/reframed_rul.jsonl
```

**Result.** Best-method ceiling ratio (lower is better):

| Subset | raw, stride 20 (728 baseline) | piecewise, stride 20 | raw, stride 5 | piecewise, stride 5 | train rows (stride 5) |
|--------|------------------------------:|---------------------:|--------------:|--------------------:|----------------------:|
| FD001 | 0.652 | 0.545 | 0.638 | **0.521** | 2488 |
| FD003 | 0.607 | **0.458** | 0.603 | 0.493 | 3093 |

- **Piecewise RUL is the strong lever.** At fixed decimation it drops the
  ceiling ratio 0.652 → 0.545 (FD001) and 0.607 → 0.458 (FD003): the features
  now explain ~46–55 % of the target spread, up from ~35–39 %. This is the
  diagnostic's "low task ceiling" finding directly addressed — early-life
  clipping removes the unlearnable burden.
- **Reduced decimation restores power, not point accuracy.** Stride 5 gives
  ~4× the training rows (639 → 2488, 797 → 3093), which tightens the paired
  bootstrap CIs a robustness comparison needs. Its effect on the *point*
  ratio is small and mixed — it improves FD001 (0.545 → 0.521) but slightly
  worsens FD003 (0.458 → 0.493, as more noisy early-life rows re-enter). The
  value here is statistical power for Lever 2, not a better ceiling by
  itself; stated rather than oversold.

**Honest limit.** Even re-framed, a ratio of ~0.5 means the pooled **linear**
model's RMSE is still half of predicting the mean — a modest fit. Piecewise
RUL raises the ceiling materially but does not make C-MAPSS RUL a strong
linear-regression task; a monotone or nonlinear per-unit model is the next
framing lever, out of scope here. What Lever 1 establishes is a task with
enough ceiling and enough rows for Lever 2 (a high-leverage contamination
model) to be a *fair* test of whether the current robustness has anything to
demonstrate.
