# TDI-5.5 — The Baseline Challenge: Confirmatory Results

## Status

This document reports the single, real, preregistered TDI-5.5 run. The
design was frozen before execution
(`docs/TDI-5.5-OVERLAP-BASELINE-CHALLENGE-PREREGISTRATION.md`) and the run
was a deliberate one-time human action under the exact confirmation token.
No classification below may be rewritten (preregistration Section 20).

**Headline.** Within the exact scope of the design, the early overlaps
`{O_1, O_2}` carry **substantial, robust predictive signal beyond both** an
exact contraction descriptor (the Dobrushin coefficient and mean pairwise
total variation) **and** a naive temporal-persistence competitor, at
**every** horizon `U_3 … U_8`. TDI's overlap signal is therefore **not**, in
this scope, a repackaging of exact-computable contraction structure nor a
by-product of trivial trajectory continuation.

## 1. Provenance and integrity

| Item | Value |
|---|---|
| Run git commit | `64eb44d85c4b290c38a80f8caae09b5667e445c0` |
| Evaluator (v55) SHA-256 | `10df698d10f010b9f6c18e2a4d78042eb399d3812b8d69c2b4bb799de828b835` |
| Preregistration SHA-256 | `37260b3349107659487e42e66c269ecad44efaf6131f8206bb28dfbcf83f9da1` |
| Scientific-manifest SHA-256 | `b5dc36e280bf2b8b70581b123aab9e1fe2423870b098aa18ada7f1fd2c6e8666` |
| Result log SHA-256 | `48bceae78f2545b1af040777dabd6793c3ccf06793ce3bcee00f59ef68c2165b` |
| Toolchain | rustc 1.97.0, cargo 1.97.0 |
| Host | Linux `tarek`, aarch64 (Jetson, ARM64) |
| Start / end (UTC) | 2026-07-22T18:09:47Z / 2026-07-22T18:28:35Z (~19 min) |

The run's recorded evaluator hash equals the committed frozen v55 evaluator
and the frozen `EVALUATOR.sha256` manifest; the preregistration and
scientific-manifest hashes likewise match the committed frozen files. The
full frozen chain TDI-5.1 → TDI-5.5 verifies, and the frozen TDI-5.3/5.4
evaluator and TDI-5.4 preregistration hashes recorded in the run metadata
match their committed values. The run therefore used the exact reviewed,
frozen scientific code; the log hash has been reverified independently.

## 2. Populations

Single generator (base width-3 + width-4 in-distribution composition),
three fresh seed blocks G/H/I, 40,000 accepted records per block, **120,000
total**, no OOD populations (preregistration Section 8). Models were fitted
on each block's combined width-3 + width-4 training population and every
criterion evaluated on that block's combined holdout. Standardized-U space
is the primary confirmatory domain.

The two exact contraction descriptors — the Dobrushin coefficient
`delta = max_{i<j} TV(P_i, P_j)` and the mean pairwise total variation
`delta_bar` of the one-step Noop kernel — were computed per candidate system
as exact rationals (`(denominator - numerator)/denominator`) and converted
to `f64` in a single rounding step, exactly like the early overlaps.

## 3. Criterion TDI-5.5A — signal beyond contraction

**CKT** (baseline + δ + δ̄ + O₁ + O₂) versus **CK** (baseline + δ + δ̄) on
combined holdout at the focal horizons, four-way classification with the
symmetric 2% relative-MSE margin:

| Horizon | Classification | Aggregate rel-MSE reduction | Aggregate 95% CI | Blocks confirming |
|---|---|---:|---|---:|
| **U₃** | **Beneficial** | 47.09% | [0.4613, 0.4802] | 3 / 3 |
| **U₆** | **Beneficial** | 22.22% | [0.2116, 0.2327] | 3 / 3 |

At both focal horizons all three blocks individually confirm the benefit,
the aggregate relative improvement exceeds the 2% margin by more than an
order of magnitude, and the aggregate bootstrap lower bound is strictly
positive. **Adding `{O_1, O_2}` on top of the exact contraction descriptors
reduces error substantially**; the overlaps are not redundant with those
descriptors.

## 4. Criterion TDI-5.5B — signal beyond persistence

**CKT** versus the **naive persistence competitor** (a fixed, zero-parameter
linear extrapolation of the recent deficit trajectory in U space,
`U_hat_h = U_2 + (h-2)(U_2 - U_1)`) at the focal horizons:

| Horizon | Classification | Aggregate rel-MSE reduction | Aggregate 95% CI | Blocks confirming |
|---|---|---:|---|---:|
| **U₃** | **Beneficial** | 42.12% | [0.4073, 0.4345] | 3 / 3 |
| **U₆** | **Beneficial** | 68.45% | [0.6767, 0.6920] | 3 / 3 |

The fitted model beats naive continuation decisively at both horizons. The
gap **widens** with horizon: the persistence competitor degrades badly at
U₆ (aggregate standardized-U MSE 0.7178, R² 0.286; in reconstructed-O space
its aggregate R² is **negative**, −0.118 — worse than predicting the mean),
because a linear extrapolation of the deficit overshoots at longer horizons,
while the model holds up (U₆ CKT MSE 0.2266). Even at the near horizon U₃,
where persistence is strongest, the model wins by ~42%. **The overlap-based
prediction captures dynamical structure well beyond trajectory
continuation.**

## 5. Criterion TDI-5.5C — decay law and redundancy horizon

CKT-vs-CK across the dense grid `U_3 … U_8` (base generator):

| Horizon | Aggregate rel-MSE reduction | Aggregate 95% CI | Classification |
|---|---:|---|---|
| U₃ | 47.09% | [0.4613, 0.4802] | Beneficial |
| U₄ | 35.29% | [0.3423, 0.3634] | Beneficial |
| U₅ | 27.37% | [0.2629, 0.2843] | Beneficial |
| U₆ | 22.22% | [0.2116, 0.2327] | Beneficial |
| U₇ | 18.40% | [0.1739, 0.1942] | Beneficial |
| U₈ | 15.78% | [0.1480, 0.1678] | Beneficial |

- **`monotone_non_increasing` = true** — the marginal value of `{O_1, O_2}`
  beyond contraction decreases monotonically with horizon.
- **redundancy horizon `h★` = none** — the classification is **Beneficial at
  every horizon** in U₃…U₈; the marginal value never enters the ±2%
  Equivalent band in the tested range.
- successive ratios `r_{h+1}/r_h` = [0.749, 0.776, 0.812, 0.828, 0.858] —
  increasing toward 1: a **decelerating** decay that plateaus far above the
  2% margin within this range, rather than collapsing to negligibility.

## 6. Relationship to TDI-5.4

TDI-5.4 studied **O₁ alone** under a nonlinear basis and found its marginal
value **decayed into Equivalence by U₅–U₆** (a genuine redundancy horizon).
TDI-5.5 studies the **joint `{O_1, O_2}` signal beyond an exact contraction
baseline** and finds it **never becomes negligible** across U₃…U₈. These are
consistent, not contradictory: O₁'s *incremental* nonlinear contribution
over O₂ fades quickly, whereas the *joint* overlap pair carries persistent
information that the exact contraction descriptors and naive persistence do
not supply.

An exploratory diagnostic supports reading the contraction baseline as
**modest**: in the fitted CK models the standardized coefficient of δ̄ is
non-trivial (≈ −0.32 in block G at U₈) while δ is near zero (≈ −0.02). The
exact Dobrushin descriptors do carry some signal (mostly through the mean
pairwise total variation), but `{O_1, O_2}` dominate them by a wide margin.

## 7. Interpretation and boundaries

Within the exact scope, this is the strongest evidence in the series that
TDI's overlap signal is a **candidate independent informational dimension**
rather than a compact estimator of already-known structure: it survives the
best *exact-computable* contraction descriptor and a naive persistence
control, decisively and at every horizon, replicated across three seed
blocks.

The result establishes exactly this and no more (preregistration Section
20). It does **not** establish:

- **control against non-exact contraction descriptors.** The stronger
  classical descriptors — spectral gap / second eigenvalue, ε-threshold
  mixing time — are transcendental or iterative and were deliberately
  deferred to the non-exact **TDI-6** track (preregistration Section 21). A
  skeptic cannot yet rule out that a spectral-gap feature would explain the
  signal; that is precisely the decisive test TDI-6 is meant to run. The
  claim here is "beyond the exact Dobrushin-based descriptors," and it is
  strengthened by, but also qualified by, the observation (Section 6) that
  those exact descriptors are modest predictors in this system;
- **control against arbitrary persistence models** beyond the single
  preregistered U-space linear extrapolator;
- sufficiency under nonlinear or non-parametric model families; robustness
  to generator changes (single generator only); causal effects; universal
  validity across dynamical systems; or external empirical validity.

The TDI-5.5A / TDI-5.5B / TDI-5.5C summaries are frozen as reported.

## 8. Reproduction

    TDI55_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI55_FREEZE_RULE \
      bash scripts/reproduce-tdi5.5.sh

The script refuses without the exact token, refuses a dirty repository,
verifies the full frozen hash chain (TDI-5.1 → TDI-5.5) before any
generation, executes the evaluator once with `--full`, verifies the final
criterion lines, and writes read-only result, metadata, hash and completion
artifacts. Determinism is exact: the log SHA-256
`48bceae78f2545b1af040777dabd6793c3ccf06793ce3bcee00f59ef68c2165b` is
reproduced by any faithful re-run of commit
`64eb44d85c4b290c38a80f8caae09b5667e445c0`.
