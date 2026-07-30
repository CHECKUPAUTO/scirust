# TDI-6.3 — Information Decomposition: Confirmatory Results

## Status

This document reports the single, real, preregistered TDI-6.3 run. The design
was frozen before execution
(`docs/TDI-6.3-INFORMATION-DECOMPOSITION-PREREGISTRATION.md`, SHA-256
`bd220fe6621d35099e729d54fa7befa18c7bf9287fe4bb1e63a449de0b96097e`) and the run
was a deliberate one-time human action under the exact confirmation token.
TDI-6.3A / B / C are **descriptive** criteria — the preregistration states
explicitly that no outcome is a success or a failure — and their summaries are
frozen as produced.

**Headline, stated in the order that avoids the trap.** The most eye-catching
line in the output is `Unique(O₁) = 0.000000000 bits` in every block, at every
horizon. **That is a definition, not a measurement.** Under the MMI redundancy
measure this experiment preregistered, `Red = min(I(T;O₁), I(T;O₂))`, so
whenever `I(T;O₁) ≤ I(T;O₂)` — which held in all 24 reported cells —
`Unique(O₁) = I(T;O₁) − Red` is **identically zero by construction**. It carries
no empirical content beyond the ordering `I(T;O₁) ≤ I(T;O₂)`.

The real finding is what the remaining components do with horizon. Total
information about the target falls (0.958 → 0.549 bits from U₃ to U₈) and
redundancy falls with it (0.268 → 0.090 bits, 28.0 % → 16.4 % of the total),
but **synergy is the one component that grows in absolute terms**: 0.000048 →
0.036 bits, i.e. 0.005 % → 6.59 % of the joint information. `Unique(O₂)`
dominates at every horizon in every block, and its *share* is almost flat
(72.0 % → 77.0 %).

## 1. Provenance and integrity

| Item | Value |
|---|---|
| Run git commit | `6f0c7c44a75bddc1eb3272a09763810b0fb48edd` |
| Evaluator | `tdi-bench/src/bin/tdi-independent-overlap-ablation-v63.rs` |
| Evaluator SHA-256 | `7e61baa85ae9d4b48cb1d6f527497cbb3980ae655f7f5aca40745d9bcb69e893` |
| Preregistration SHA-256 | `bd220fe6621d35099e729d54fa7befa18c7bf9287fe4bb1e63a449de0b96097e` |
| Scientific-manifest SHA-256 | `02ad87a708369b1bd2c0e868bba5b1d094185e5cc5248bc4e1aa2592f6a809c6` |
| Result log SHA-256 | `9554c4b8c5e5756b953123e7e200c76f383878167f8aaba87f0a68b3335572c5` |
| Toolchain | rustc 1.97.1 (8bab26f4f 2026-07-14), cargo 1.97.1 (c980f4866 2026-06-30) |
| Host | Linux `tarek`, aarch64 (Jetson, ARM64) |
| Start / end (UTC) | 2026-07-29T22:00:31Z / 2026-07-29T22:21:32Z (21 min 01 s) |

The committed log has been independently rehashed in this working tree to
`9554c4b8c5e5756b953123e7e200c76f383878167f8aaba87f0a68b3335572c5`, matching the
value the run recorded for itself. The frozen chain TDI-5.1 → 5.8 and
TDI-6.1 / 6.2 / 6.5 (evaluator + preregistration hashes) was verified before any
generation. Reproduction is **tolerance-based**: covariances, log-determinants
and the Cholesky factorization are `f64` (single-threaded, fixed operation
order), byte-exact on the reference toolchain and architecture.

## 2. Design recap

- **Quantities.** `T = U_h`; sources `S₁ = O₁`, `S₂ = O₂` — the same two early
  overlaps the ablation series has used since TDI-5.2.
- **Model.** Gaussian (second-moment-only) working model; every mutual
  information is the closed-form MI of a multivariate normal fitted to the
  sample covariance. This is a **stated modeling choice**, not a claim that
  `(O₁, O₂, U_h)` are jointly Gaussian.
- **Decomposition.** MMI redundancy (Barrett 2015), which guarantees all four
  components are non-negative under the Gaussian model:
  `Red = min(I(T;O₁), I(T;O₂))`, `Un₁ = I(T;O₁) − Red`,
  `Un₂ = I(T;O₂) − Red`, `Syn = I(T;{O₁,O₂}) − I(T;O₁) − I(T;O₂) + Red`.
- **Scope.** **Unconditional** — the decomposition does not condition on the
  exact contraction/spectral baseline descriptors. It asks how `{O₁,O₂}`'s own
  joint information about `U_h` is structured, not whether that information is
  redundant with the baselines (that is the 5.5 → 6.5 ablation question).
- **Populations.** Three fresh blocks S/T/U, 40,000 accepted records each,
  **120,000 total**; all four populations of a block pooled, no train/holdout
  split (there is no model being fitted). 120,264 attempted, so **264
  preregistered exclusions** — all `observation-fully-recovered`, 263 of them at
  width 3 and 1 at width 4.

> Naming note: the seed blocks are labelled **S**, **T**, **U** and the horizons
> **U₃ … U₈**. Block **U** and horizon **U₆** are unrelated; the log
> disambiguates by position (`bloc U — U_6`).

## 3. TDI-6.3A — the decomposition at the focal horizons

Aggregate (all three blocks pooled), bits:

| Horizon | `I(T;O₁)` | `I(T;O₂)` | `I(T;{O₁,O₂})` | Redundancy | Unique(O₁) | Unique(O₂) | Synergy | Dominant |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| **U₃** | 0.268381 | 0.957932 | 0.957980 | 0.268381 (28.02 %) | **0** | 0.689551 (71.98 %) | 0.000048 (0.005 %) | `unique(O₂)` |
| **U₆** | 0.118966 | 0.602853 | 0.630023 | 0.118966 (18.88 %) | **0** | 0.483886 (76.80 %) | 0.027171 (4.31 %) | `unique(O₂)` |

Per block, with 95 % bootstrap intervals (4,000 replicates, frozen seeds):

| Horizon | Block | Redundancy | Unique(O₂) | Synergy |
|---|:--:|---|---|---|
| U₃ | S | 0.260959 [0.252727, 0.269116] | 0.688663 [0.676837, 0.700740] | 0.00000033 [0.00000002, 0.000125] |
| U₃ | T | 0.273369 [0.265121, 0.281854] | 0.690378 [0.678590, 0.702501] | 0.000086 [0.00000054, 0.000367] |
| U₃ | U | 0.270786 [0.262412, 0.278848] | 0.689719 [0.678356, 0.701649] | 0.000116 [0.0000022, 0.000396] |
| U₆ | S | 0.113124 [0.107484, 0.118910] | 0.479671 [0.469749, 0.489515] | 0.029082 [0.026034, 0.032197] |
| U₆ | T | 0.123193 [0.117299, 0.129156] | 0.488306 [0.478448, 0.498356] | 0.026529 [0.023634, 0.029649] |
| U₆ | U | 0.120585 [0.114666, 0.126656] | 0.483642 [0.473567, 0.493589] | 0.026005 [0.023132, 0.029069] |

`Unique(O₁)`'s bootstrap interval is `[0.000000000, 0.000000000]` in every cell.
This is not a tight estimate — it is the resampled image of a quantity that is
identically zero (see §4).

**Cross-method agreement.** The preregistration declares a 1e-9 bit tolerance
between method 1 (Cholesky log-determinant) and method 2 (multiple-correlation
identity). The observed maximum absolute gap is **0.000000000000 — exactly zero
to the reported twelve decimals — in all 24 cells**, on `I(T;O₁)`, `I(T;O₂)` and
`I(T;{O₁,O₂})` alike. Two genuinely different arithmetic paths agree bit-for-bit;
the Cholesky degeneracy floor (pivot ≤ 1e-12) never triggered.

## 4. `Unique(O₁) = 0` is definitional — the one number that must not be misread

The correlations tell the whole story:

| Horizon | `ρ(T,O₁)` | `ρ(T,O₂)` | `ρ(O₁,O₂)` |
|---|---:|---:|---:|
| U₃ | 0.557390 | 0.857314 | 0.646425 |
| U₆ | 0.389923 | 0.752624 | 0.646425 |

`|ρ(T,O₂)| > |ρ(T,O₁)|` at every horizon, so `I(T;O₂) > I(T;O₁)` at every
horizon, so `Red = I(T;O₁)` and therefore `Un₁ = I(T;O₁) − Red ≡ 0`. **No
arithmetic on the data could have produced anything else.** Reporting
"O₁ contributes zero unique information" as an empirical finding would be a
category error: MMI assigns *all* of the weaker source's information to the
redundant atom, by definition. The only empirical content in that zero is the
ordering itself — that `O₂` is the more informative of the two overlaps about
`U_h`, at every horizon, in every block.

There is a corollary that makes the decomposition more useful than it first
looks. When `Un₁ = 0`, the synergy term collapses algebraically:

    Syn = I(T;{O₁,O₂}) − I(T;O₁) − I(T;O₂) + I(T;O₁) = I(T;{O₁,O₂}) − I(T;O₂)
        = I(T; O₁ | O₂)

So the "synergy" column is *exactly* the conditional information `O₁` carries
about the target given `O₂`, under the Gaussian model. This identity is verified
to machine precision in the reported values (max |Syn − (I_joint − I(T;O₂))|
≈ 1e-16 across all six aggregate horizons, at the printed precision). The
synergy row is therefore readable directly as **what `O₁` adds on top of `O₂`**.

## 5. TDI-6.3B — the dense grid: synergy is the only component that grows

Aggregate, all six horizons:

| Horizon | `I(T;{O₁,O₂})` | Redundancy | share | Unique(O₂) | share | Synergy | share |
|---|---:|---:|---:|---:|---:|---:|---:|
| U₃ | 0.957980 | 0.268381 | 28.02 % | 0.689551 | 71.98 % | 0.000048 | 0.005 % |
| U₄ | 0.794718 | 0.187341 | 23.57 % | 0.598705 | 75.34 % | 0.008671 | 1.09 % |
| U₅ | 0.694354 | 0.145592 | 20.97 % | 0.529998 | 76.33 % | 0.018763 | 2.70 % |
| U₆ | 0.630023 | 0.118966 | 18.88 % | 0.483886 | 76.80 % | 0.027171 | 4.31 % |
| U₇ | 0.582262 | 0.102283 | 17.57 % | 0.447932 | 76.93 % | 0.032047 | 5.50 % |
| U₈ | 0.549307 | 0.090215 | 16.42 % | 0.422872 | 76.98 % | 0.036220 | 6.59 % |

`Unique(O₁)` is 0 at all six. **Dominant component: `unique(O₂)` at every
horizon — stable, no shift.**

Three monotone trends, all clean:

1. **Total information decays** with horizon (0.958 → 0.549 bits), consistent
   with every prior TDI experiment's decay law.
2. **Redundancy decays faster** than the total — in bits (0.268 → 0.090, −66 %)
   and in share (28.0 % → 16.4 %). At short horizons the two overlaps largely
   tell the same story about the target; at long horizons they overlap less.
3. **Synergy — that is, `I(T;O₁|O₂)` — grows monotonically**, and is the only
   component to grow in absolute bits: 0.000048 → 0.036220, a factor of ~756.
   At U₃ it is essentially zero (4.8e-5 bits); by U₈ it is 6.6 % of everything
   the pair knows.

`Unique(O₂)`'s share, by contrast, is nearly constant (72.0 → 77.0 %) — it
absorbs almost exactly what redundancy gives up.

## 6. Reconciling trend 3 with TDI-5.4 (which found the opposite direction)

TDI-5.4 reported a **monotone decay** of `O₁`'s marginal contribution with
horizon — *Beneficial* at short horizons, *Equivalent* (≈1.2 %, inside the ±2 %
margin) at U₅/U₆/U₈. TDI-6.3 reports `I(T;O₁|O₂)` **growing** with horizon. Both
are correct; they are different quantities, and the difference is instructive
rather than contradictory:

- **5.4 conditions on the 13-variable baseline; 6.3 does not** (preregistration
  §4.3, unconditional by design). Much of what `O₁` knows may already be carried
  by the baseline structural/entropic variables, and increasingly so at long
  horizons. 6.3 measures `O₁` against `O₂` alone.
- **Different scales.** 6.3's growth is in absolute bits against a *shrinking*
  total. Converting the aggregate MIs to the residual-variance scale that 5.4's
  criterion uses (`R² = 1 − 2^(−2I)`, exact under the same Gaussian model) gives
  the relative MSE reduction `O₁` would buy on top of `O₂` **with no baseline
  present**: 0.007 % (U₃), 1.19 % (U₄), 2.57 % (U₅), 3.70 % (U₆), 4.35 % (U₇),
  4.90 % (U₈).

That derived column — a plain algebraic transform of the reported quantities,
**not** a preregistered output of this experiment — is the honest bridge: `O₁`'s
value beyond `O₂` alone is small everywhere (under 5 %) and grows with horizon,
while its value beyond `O₂` *and the 13 baseline variables* is small everywhere
and shrinks with horizon. Nothing here overturns TDI-5.4; it locates 5.4's
"redundancy of `O₁`" as redundancy with the **baseline**, not with `O₂`.

## 7. TDI-6.3C — cross-block consistency

| Horizon | Block S | Block T | Block U | `cross_block_dominant_component_consistent` |
|---|:--:|:--:|:--:|:--:|
| U₃ | `unique(O₂)` | `unique(O₂)` | `unique(O₂)` | **true** |
| U₆ | `unique(O₂)` | `unique(O₂)` | `unique(O₂)` | **true** |

3/3 at both focal horizons, with per-block point estimates tight enough that the
bootstrap intervals of the *same* component overlap heavily across blocks (e.g.
U₆ redundancy: S 0.1131, T 0.1232, U 0.1206). This is the series' usual
replication check, adapted to a descriptive decomposition.

## 8. Interpretation and boundaries

TDI-6.3 characterizes the *structure* of the information `{O₁, O₂}` carry about
`U_h`. Its usable content is: `O₂` is the informative source and dominates at
every horizon; the two overlaps are substantially redundant with each other at
short horizons and progressively less so; and `O₁`'s conditional contribution
grows with horizon while everything else shrinks.

What this does **not** establish:

- **`Unique(O₁) = 0` is not a finding.** It is what MMI does with the weaker
  source. Another PID definition (`I_BROJA`, `I_ccs`, discretized `I_min`) could
  assign `O₁` non-zero unique information from exactly the same data;
- **the Gaussian working model is a choice, not a measurement.** If the true
  joint distribution of `(O₁, O₂, U_h)` has skew, heavy tails or nonlinear
  dependence, the numbers above characterize its covariance structure, not its
  true information content;
- **no baseline conditioning** (§6) — this is not the ablation question;
- single base generator, widths 3–4, one-step Noop kernel; no causal claim
  (TDI-6.4); no cross-family (TDI-6.5) or cross-width (TDI-5.8) robustness; and
  nothing outside small synthetic finite-state families.

The TDI-6.3A / TDI-6.3B / TDI-6.3C summaries are frozen as reported.

## 9. Reproduction

    TDI63_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI63_FREEZE_RULE \
      bash scripts/reproduce-tdi6.3.sh

The script refuses without the exact token, refuses a dirty repository, verifies
the frozen hash chain before any generation, executes the evaluator once with
`--full`, verifies the final summary lines, and writes read-only artifacts under
`results/tdi6.3-information-decomposition/`. Budget roughly 25 minutes on the
reference host.
