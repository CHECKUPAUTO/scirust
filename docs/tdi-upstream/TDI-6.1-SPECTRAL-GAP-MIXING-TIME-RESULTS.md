# TDI-6.1 — The Literal Spectral Gap and Mixing Time: Confirmatory Result

## Status

This is the frozen confirmatory result of **TDI-6.1**, the first TDI-6
experiment and the first to relax bit-exactness (localized to the two non-exact
spectral descriptors only; Section 12 of the preregistration). The real
120,000-record run was executed once, as the deliberate human action of
Section 18, on the Jetson reference host. The design was frozen before the run
(preregistration + evaluator + reproduction script + CI + SHA-256 manifests,
merged in PR #26); the classifications reported below are frozen as produced.

**Headline.** The early overlaps `{O₁, O₂}` **survive the literal spectral gap
`g = 1 − |λ₂|` and the ε-mixing time `τ_ε` of the one-step Noop kernel**:
criterion **TDI-6.1A is _Beneficial_ at both focal horizons U₃ and U₆**. This
is the decisive skeptic control the series has pointed at since TDI-5.5, and the
strongest evidence to date that the overlap signal is not a finite-sample proxy
for the classical mixing / second-eigenvalue structure of the one-step kernel —
within this experiment's scope (single base generator, linear ridge, one-step
kernel, ε = 1/4).

## Provenance and integrity

| Field | Value |
|---|---|
| Run commit | `5015f3a9365407c6c31e042fc5c951eda65db85b` (merge of PR #26 into `main`) |
| Evaluator | `tdi-bench/src/bin/tdi-independent-overlap-ablation-v61.rs` |
| Evaluator SHA-256 | `bb9d155021117b70d1483a9abbc51f45f994caddb8a17365d7fb14f02201f278` |
| Preregistration SHA-256 | `4d754f334c95b113078c28a24069ffd8fb3e93e2ba89055001aab3bf3ee1a159` |
| Result log | `results/tdi6.1-spectral-gap-mixing-time/tdi-independent-overlap-ablation-v61.log` |
| Result log SHA-256 | `7c8673fc348af2bf2cce06038603543cac0e71b542f4222a29529f1f41751fa1` |
| Completed (UTC) | `2026-07-23T11:51:02Z` |
| FP regime (Section 12) | IEEE-754 binary64, single-threaded, fixed operation order |
| Frozen ancestor chain | TDI-5.1 → 5.7 verified at run start (evaluator + preregistration + scientific manifest hashes) |

The result log SHA-256 is byte-exact on the reference toolchain/architecture.
Across architectures the raw f64 metrics may differ in their last digits, but —
because the four-way classifier's margin is ±2 % relative MSE, many orders of
magnitude larger than the eigensolver tolerance (η = 1e-12) — the 6.1A / 6.1B /
6.1C **classifications reproduce exactly** (Section 4.2). The reproduction
script's completion check verifies both the byte-exact log hash and the presence
of the classification lines. The exact toolchain versions and start timestamp
are recorded in the run's read-only `…​.metadata.txt`.

## Design recap

- **Kernel.** For each candidate system of width `w`, `P` is the one-step `Noop`
  kernel on `n = 2^w` states (`w ∈ {3, 4}`, so `n ∈ {8, 16}`), row-stochastic.
- **Non-exact descriptors (the only relaxation).** The literal spectral gap
  `g = 1 − |λ₂|` (`|λ₂|` = second-largest eigenvalue modulus of `P`, from a
  dependency-free complex shifted-QR eigensolver) and the normalized ε-mixing
  time `τ_ε / T_max` (direct `P^t` iteration to the stationary distribution,
  ε = 1/4, `T_max = 4096`).
- **Layouts.** `CK`(15) ⊂ `SK`(17) ⊂ `GK`(19) ⊂ `GKT`(21), where
  `SK = baseline + δ + δ̄ + s₂ + s₃` (exact), `GK = SK + g + τ_ε`,
  `GKT = GK + O₁ + O₂`.
- **Criteria.** **6.1A** = `GKT` vs `GK` at focal U₃, U₆ (primary); **6.1B** =
  `GK` vs `SK` at U₃, U₆ (marginal value of the literal descriptors); **6.1C** =
  `GKT` vs `GK` decay across U₃…U₈.
- **Populations.** Single base generator (5.7's F0Base under fresh seeds); three
  independent blocks M/N/O; 40,000 accepted records per block, 120,000 total;
  no OOD.

## TDI-6.1A — signal beyond the literal spectral gap (primary): **Beneficial** at U₃ and U₆

Adding the two early overlaps to a baseline that already contains the contraction
descriptors, the **exact** spectral moments, **and** the **literal** spectral gap
+ ε-mixing time still reduces error, at both focal horizons, on all three blocks.

| Focal horizon | `GK` MSE (std-U) | `GKT` MSE (std-U) | Aggregate relative MSE reduction (95 % CI, median) | Blocks confirming | Classification |
|---|---:|---:|---|:--:|:--:|
| **U₃** | 0.317372 | 0.168277 | **[46.04 %, 47.88 %], 46.98 %** | 3 / 3 | **Beneficial** |
| **U₆** | 0.161412 | 0.117088 | **[26.50 %, 28.41 %], 27.45 %** | 3 / 3 | **Beneficial** |

Standardized-U `R²` rises from 0.684 → 0.832 at U₃ and 0.838 → 0.882 at U₆;
Spearman from 0.804 → 0.906 (U₃) and 0.902 → 0.928 (U₆). The paired
per-block reductions are tight and unanimous (e.g. U₃: M [46.25 %, 49.32 %],
N [44.39 %, 47.44 %], O [45.54 %, 48.71 %]). Every aggregate bootstrap lower
bound is positive and every aggregate improvement exceeds the +2 % margin, so
the classifier returns *Beneficial* under its full three-condition rule.

**Reading.** The "TDI is just the spectral gap" hypothesis is refuted **within
this scope**: to f64 precision, the literal `|λ₂|` and the ε-mixing time of the
one-step kernel do not subsume the early-overlap signal.

## TDI-6.1B — the literal descriptors are themselves informative (the control is demanding): **Beneficial** at U₃ and U₆

This is the control that makes 6.1A meaningful: it checks that `g` and `τ_ε`
are not inert padding but carry genuine marginal value beyond the exact moments
`s₂, s₃`.

| Focal horizon | `SK` MSE (std-U) | `GK` MSE (std-U) | Aggregate relative MSE reduction (95 % CI, median) | Classification |
|---|---:|---:|---|:--:|
| **U₃** | 0.334555 | 0.317372 | [4.46 %, 5.83 %], **5.13 %** | **Beneficial** |
| **U₆** | 0.247635 | 0.161412 | [33.50 %, 36.12 %], **34.82 %** | **Beneficial** |

The literal descriptors add only ~5 % at the short horizon U₃ but a large
**~35 %** at U₆ — as expected, since the spectral gap and mixing time govern
longer-horizon convergence. In the fitted `GK` model the standardized
coefficient on the literal gap `g` is the single largest term at U₈
(≈ 0.715), confirming it is a strong, non-inert predictor.

**Consequence.** 6.1A is therefore a *genuine strengthening*, not a test against
near-inert features — and its most demanding case is U₆, where the literal
descriptors alone already cut error by ~35 % and the overlaps **still** cut a
further 27 % on top of them.

## TDI-6.1C — decay law and redundancy horizon (descriptive)

`GKT` vs `GK` across the dense grid U₃…U₈:

| Horizon | Aggregate relative MSE reduction | Classification |
|---|---:|:--:|
| U₃ | 46.98 % | Beneficial |
| U₄ | 37.25 % | Beneficial |
| U₅ | 31.12 % | Beneficial |
| U₆ | 27.46 % | Beneficial |
| U₇ | 24.17 % | Beneficial |
| U₈ | 22.03 % | Beneficial |

- **Monotone non-increasing:** yes.
- **Redundancy horizon `h★` (first *Equivalent*):** none — the overlaps remain
  *Beneficial* at every horizon through U₈.
- **Successive ratios `r_(h+1)/r_h`:** 0.793, 0.835, 0.882, 0.880, 0.911 — a
  decelerating decay: the marginal value shrinks with horizon but is still a
  22 % relative reduction at U₈.

## The non-exact discipline held

The Section-19 three-method spectral cross-validation table confirms the
canonical eigensolver (method 1, the frozen feature path) was correct to machine
precision on the real candidate kernels:

- **Trace-consistency residual `max_k |Σλᵢᵏ − trace(Pᵏ)|`: 2.16e-13** across the
  sampled n = 8 and n = 16 kernels — the rigorous correctness witness.
- The method-1↔2 disagreement reaches ~9.8e-2 on some non-symmetric candidates;
  this is *expected* and reflects complex `λ₂` (where a scalar deflated power
  iteration is not a modulus witness), **not** an error in method 1. On
  real-spectrum kernels the 1↔2 agreement is verified to 1e-9 by the known-
  spectra battery in the test suite.

Because the ±2 % classifier margin dwarfs these tolerances, the confirmatory
classifications are robust to last-digit f64 variation, exactly as Section 4.2
predicted.

## Interpretation

The exact series (5.5 → 5.7) had already shown the overlaps carry signal beyond
the exact contraction descriptors, the exact spectral moments (power sums of the
spectrum), and a family of exact generators. TDI-6.1 closes the remaining
skeptic escape: it puts the **literal** second-eigenvalue modulus and the
**ε-threshold mixing time** — the transcendental/iterative quantities the exact
arithmetic could never express — directly into the baseline, and the overlaps
still help.

The horizon structure is coherent:

- **Short horizon (U₃).** The literal spectral descriptors add little (~5 %),
  while the overlaps dominate (~47 %): near-term recovery is governed by the
  early overlap geometry, not by the asymptotic mixing rate.
- **Longer horizon (U₆).** The literal descriptors add a lot (~35 %) — mixing
  structure now matters — yet the overlaps contribute a further ~27 % of
  independent signal on top of them.

Either way the overlaps encode multi-step recovery information that the one-step
kernel's literal spectral gap and mixing time do not capture.

## Limitations

Consistent with preregistration Section 21, this result establishes the above
**only** within its scope. It does **not** establish: robustness of the literal-
spectral control across the TDI-5.7 generator families (a single base generator
is used here — 6.1 isolates the *new* factor, the generator question being
settled by 5.7); sufficiency under nonlinear / non-parametric model families
(TDI-6.2); an information-decomposition (PID) account (TDI-6.3); a causal claim
(TDI-6.4); higher-order or multi-step spectral structure; cross-width invariance
(TDI-5.8); or external empirical validity. The mixing time uses a single frozen
threshold ε = 1/4, and `|λ₂|`/`τ_ε` are f64 quantities (tolerance-based
reproduction, Section 12) rather than bit-exact.

## Reproduction

```
TDI61_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI61_FREEZE_RULE bash scripts/reproduce-tdi6.1.sh
```

The script refuses without the exact token and on a dirty repository, verifies
the full frozen chain TDI-5.1 → 5.7 plus the three TDI-6.1 manifests before any
generation, runs `--full` once, and writes read-only result / metadata / hash /
completion artifacts. The completion check verifies the result-log SHA-256
(byte-exact on the reference toolchain/architecture) and the presence of the
6.1A / 6.1B / 6.1C classification lines (the tolerance-robust invariant).
