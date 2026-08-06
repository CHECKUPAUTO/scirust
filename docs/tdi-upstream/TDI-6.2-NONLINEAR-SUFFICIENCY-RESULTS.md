# TDI-6.2 — Nonlinear Sufficiency: Confirmatory Results

## Status

This document reports the single, real, preregistered TDI-6.2 run. The design
was frozen before execution
(`docs/TDI-6.2-NONLINEAR-SUFFICIENCY-PREREGISTRATION.md`, SHA-256
`a5263642ee79fb946bc9a7aa6fea4b57c22945a91b7ffa6f2220c7e4d4a55869`) and the run
was a deliberate one-time human action under the exact confirmation token. No
classification below may be rewritten.

**Headline.** TDI-6.2 gave the spectral baseline a model that can actually use
its features nonlinearly — a degree-2 interaction ridge, so the baseline can
represent `g²`, `g·τ_ε`, saturation in the mixing time, and every pairwise
product of its 19 features. The point was that if the overlap signal were a
linear-modeling artifact, this is where it should evaporate. **It did not
evaporate; it grew.** Criterion **TDI-6.2A is _Beneficial_ at both focal
horizons**, and at every horizon of the dense grid the overlaps' relative MSE
reduction is *larger* under the nonlinear model than the linear model of
TDI-6.1 recorded — 54.18 % vs 46.98 % at U₃, 28.02 % vs 22.03 % at U₈. The
nonlinear capacity helped every layout (each one's absolute MSE is lower than
its 6.1 counterpart), but it helped the layout *with* the overlaps more. That
6.1-vs-6.2 comparison is descriptive and unpaired — §6 says exactly how far it
can be pushed.

## 1. Provenance and integrity

| Item | Value |
|---|---|
| Run git commit | `6e39328b952c59496787310b8c1a5f01e39ecdd3` |
| Evaluator | `tdi-bench/src/bin/tdi-independent-overlap-ablation-v62.rs` |
| Evaluator SHA-256 | `793fc42d0567283c0f6c773e74597a6ff38d7278cf6e14fcdca7d60e33758a37` |
| Preregistration SHA-256 | `a5263642ee79fb946bc9a7aa6fea4b57c22945a91b7ffa6f2220c7e4d4a55869` |
| Scientific-manifest SHA-256 | `a58b9f907dcaac7d7ad1aeac5231de668be59b25d8cff5252475dc53185535b6` |
| Result log SHA-256 | `c949add164899b3c63665632b3c10c74dab3506909e7b87ecc92529d9b4dbd56` |
| Toolchain | rustc 1.97.1 (8bab26f4f 2026-07-14), cargo 1.97.1 (c980f4866 2026-06-30) |
| Host | Linux `tarek`, aarch64 (Jetson, ARM64) |
| Start / end (UTC) | 2026-07-29T21:24:21Z / 2026-07-29T21:45:56Z (21 min 35 s) |

The committed log has been independently rehashed in this working tree to
`c949add164899b3c63665632b3c10c74dab3506909e7b87ecc92529d9b4dbd56`, matching the
value the run recorded for itself. The full frozen ancestor chain TDI-5.1 → 5.7
and TDI-6.1 (evaluator + preregistration + scientific manifest hashes) was
verified before any generation.

**Reproduction is tolerance-based, not bit-exact.** TDI-6.2 inherits TDI-6.1's
two non-exact descriptors (`g`, `τ_ε`) and adds no further non-exactness: the
degree-2 expansion and the ridge solve run in the same IEEE-754 binary64,
single-threaded, fixed-operation-order regime. On the reference
toolchain/architecture the log reproduces byte-for-byte; across architectures
the last f64 digits may drift, but the ±2 % relative-MSE classifier margin is
many orders of magnitude larger than the eigensolver tolerance (η = 1e-12), so
the 6.2A / 6.2B / 6.2C classifications reproduce exactly.

## 2. What changed, and what did not

The **single changed factor** is the model family. Everything else — generator,
descriptors, populations contract, horizons, bootstrap discipline, criterion
machinery — is inherited from TDI-6.1.

| | TDI-6.1 | TDI-6.2 |
|---|---|---|
| Model | linear ridge, λ = 1 | degree-2 interaction ridge, λ = 1 |
| Design matrix | `[1, x₁ … x_d]` | `[1, x₁ … x_d, {xᵢ·xⱼ : i ≤ j}]` |
| `GK` columns | 19 + intercept | 19 + **209** interaction + intercept |
| `GKT` columns | 21 + intercept | 21 + **252** interaction + intercept |
| Seed blocks | M / N / O | **P / Q / R** (fresh) |

Both layouts in a comparison are expanded **identically**, so `GKT − GK` still
isolates exactly the overlaps and their interactions (`O₁²`, `O₂²`, `O₁·O₂`, and
each overlap crossed with every baseline and descriptor term), and `GK − SK`
still isolates exactly the literal spectral descriptors and theirs.

## 3. Populations

Three fresh blocks P/Q/R, 40,000 accepted records each, **120,000 total**;
120,248 candidates attempted, so **248 preregistered exclusions** (247
`observation-fully-recovered`, 1 `target-fully-recovered-h3`). 246 of the 248
fall at width 3 and only 2 at width 4 — the same monotone rarefaction of full
recovery with state-space size that TDI-5.8 and TDI-6.4 report independently.

## 4. Criterion TDI-6.2A — the overlaps survive a nonlinear spectral baseline

`GKT` against `GK`, both degree-2, on combined holdout:

| Focal horizon | `GK` MSE (std-U) | `GKT` MSE (std-U) | Aggregate relative MSE reduction (95 % CI, point estimate) | Blocks confirming | Classification |
|---|---:|---:|---|:--:|:--:|
| **U₃** | 0.290634 | 0.133182 | **[53.29 %, 55.01 %], 54.18 %** | 3 / 3 | **Beneficial** |
| **U₆** | 0.141424 | 0.094675 | **[31.82 %, 34.16 %], 33.06 %** | 3 / 3 | **Beneficial** |

Standardized-U `R²` rises 0.7125 → 0.8683 at U₃ and 0.8576 → 0.9046 at U₆;
Spearman 0.8188 → 0.9198 (U₃) and 0.9103 → 0.9383 (U₆). In reconstructed-O
space the same ordering holds (`R²` 0.7555 → 0.8878 at U₃, 0.7543 → 0.8068 at
U₆). Per-block reductions are unanimous and tight:

| Block | U₃ (95 % CI, median) | U₆ (95 % CI, median) |
|---|---|---|
| P | [53.86 %, 56.73 %], 55.34 % | [33.79 %, 37.26 %], 35.53 % |
| Q | [52.16 %, 55.13 %], 53.66 % | [30.43 %, 33.98 %], 32.26 % |
| R | [51.87 %, 54.92 %], 53.40 % | [28.60 %, 33.46 %], 31.15 % |

All three classifier conditions hold at both horizons (3/3 blocks confirming
benefit, aggregate improvement above the +2 % margin, aggregate bootstrap lower
bound strictly positive), so the four-way classifier returns *Beneficial* under
its full conjunctive rule.

**Reading.** The overlap signal is **not** an artifact of restricting the
spectral baseline to linear terms. Given a model that can represent curvature
and every pairwise interaction among contraction descriptors, exact spectral
moments, the literal spectral gap and the mixing time, that baseline still does
not reach the model that also sees `O₁` and `O₂`.

## 5. Criterion TDI-6.2B — the control is more demanding, not less

`GK` against `SK` — do the literal spectral descriptors carry *nonlinear*
marginal value beyond the exact moments? If they did not, 6.2A would be testing
against near-inert padding.

| Focal horizon | `SK` MSE (std-U) | `GK` MSE (std-U) | Aggregate relative MSE reduction (95 % CI, median) | Classification |
|---|---:|---:|---|:--:|
| **U₃** | 0.311915 | 0.290634 | [6.09 %, 7.50 %], **6.82 %** | **Beneficial** |
| **U₆** | 0.220405 | 0.141424 | [34.55 %, 37.05 %], **35.83 %** | **Beneficial** |

Both *Beneficial*, and both larger than TDI-6.1's linear counterparts (5.13 %
and 34.82 %). At U₆ the literal gap and mixing time alone cut error by more
than a third — and the overlaps then cut a further 33 % on top of *that*. 6.2A
is therefore a genuine strengthening: its hardest case is the horizon where the
baseline it must beat is strongest.

## 6. Criterion TDI-6.2C — the decay law, and the comparison to linear

`GKT` vs `GK` across the dense grid, beside TDI-6.1's linear values:

| Horizon | TDI-6.2 (degree-2) | TDI-6.1 (linear) | Difference | 6.2 classification |
|---|---:|---:|---:|:--:|
| U₃ | **54.18 %** | 46.98 % | +7.20 pp | Beneficial |
| U₄ | **43.84 %** | 37.25 % | +6.59 pp | Beneficial |
| U₅ | **37.15 %** | 31.12 % | +6.03 pp | Beneficial |
| U₆ | **33.06 %** | 27.46 % | +5.60 pp | Beneficial |
| U₇ | **30.05 %** | 24.17 % | +5.88 pp | Beneficial |
| U₈ | **28.02 %** | 22.03 % | +5.99 pp | Beneficial |

- **Monotone non-increasing:** yes.
- **Redundancy horizon `h★` (first *Equivalent*):** none — *Beneficial* at every
  horizon through U₈.
- **Successive ratios `r_(h+1)/r_h`:** 0.809, 0.847, 0.890, 0.909, 0.933 — a
  decelerating decay, as in 6.1, but from a higher starting point and to a
  higher floor.

**How to read the 6.1 comparison — and how not to.** The right-hand columns are
a **descriptive cross-experiment observation, not a preregistered result.**
TDI-6.2's preregistration contains no criterion comparing itself to TDI-6.1, and
the two runs use *different, fresh seed blocks* (M/N/O vs P/Q/R), so this is a
comparison across independent samples of the same population contract — not a
paired comparison of two models on identical data. What supports taking the
direction seriously is that the gap is present at all six horizons, always in
the same direction, and its size (+5.6 to +7.2 pp) is roughly twice the
block-to-block spread within either experiment. What it does *not* support is
quoting "+7.2 pp" as a measured effect with that precision.

## 7. The non-exact discipline held — and one number that invites misreading

The Section-19 three-method spectral cross-validation table is reproduced in
full in the log. Two numbers from it are easy to confuse, and only one of them
is a correctness witness:

| Quantity | Value | What it is |
|---|---:|---|
| Max trace residual, method 1 (`Σλᵏ = trace(Pᵏ)`) | **2.97e-13** | **the rigorous witness** — machine-level; the frozen path's eigenvalues satisfy the trace identities |
| Max disagreement, method 1 ↔ method 2 | **7.48e-2** | a *diagnostic*, expected to be large when `λ₂` is complex |

**The 7.48e-2 is not solver error.** Method 2 is deflated power iteration, which
is a reliable witness of `|λ₂|` only for real-spectrum kernels (symmetric /
reversible birth-death); the known-spectrum battery in the test suite verifies
1 ↔ 2 agreement to 1e-9 on exactly those, with method 3 (reference crate,
dev-dependency) as the third leg. On real non-symmetric candidates a complex
`λ₂` makes power iteration converge to something else, and the disagreement is
the *expected* consequence. Of the twelve sampled candidates, eight agree to
between 5e-13 and 4e-11; four disagree (4.08e-5, 5.59e-3, 5.46e-2, 7.48e-2) —
and on every one of those twelve, including the four, method 1's trace residual
stays at machine level (≤ 2.97e-13). Correctness of the frozen path is
established by the trace identity, not by 1 ↔ 2 agreement.

## 8. Interpretation and boundaries

TDI-6.2 closes the "linear-modeling artifact" objection to TDI-6.1 within its
scope, and does so in the strongest available direction: the nonlinear baseline
got better, and the overlaps' advantage got bigger rather than smaller.

What this does **not** establish:

- **degree-2 is not "nonlinear" in general.** Polynomial interaction ridge is
  the minimal nonlinear step. Kernel methods, tree/forest ensembles and deeper
  polynomial degrees are untested; a sufficiently expressive learner might yet
  extract from `{g, τ_ε, s₂, s₃, δ, δ̄}` what the overlaps supply here;
- widths 3–4, a single base generator (cross-family robustness is TDI-6.5,
  cross-width is TDI-5.8), the one-step Noop kernel, `ε = 1/4`;
- no causal claim (TDI-6.4), no information decomposition (TDI-6.3);
- nothing here was tested outside small synthetic finite-state families, and no
  universal law is claimed.

The TDI-6.2A / TDI-6.2B / TDI-6.2C summaries are frozen as reported.

## 9. Reproduction

    TDI62_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI62_FREEZE_RULE \
      bash scripts/reproduce-tdi6.2.sh

The script refuses without the exact token, refuses a dirty repository, verifies
the full frozen hash chain before any generation, executes the evaluator once
with `--full`, verifies the final criterion lines, and writes read-only
artifacts under `results/tdi6.2-nonlinear-sufficiency/`. Budget roughly 25
minutes on the reference host.
