# Algorithms audit — SciRust

> Date: 2026-07-11 · Branch: `claude/scirust-algorithm-audit-a619qo`
> Scope: the ~120 algorithmic crates of the monorepo (linear algebra,
> ODE/optimization/quadrature, tensor & SIMD kernels, autodiff/NN/optimizers,
> statistics & special functions, signal & audio, time series & finance,
> classic ML/clustering/RL, evolutionary & symbolic, reasoning/graphs/NLP,
> GPU & tensor compilation, certifiable industrial verticals), plus a
> meta-audit of the test infrastructure and a coverage-gap analysis versus
> SciPy/LAPACK/GSL/statsmodels/scikit-learn.
> **Method**: 14 independent domain auditors read the source code (not just
> the signatures) and produced 157 findings; each P0/P1-severity finding
> (46 total) was then submitted to **two independent adversarial verifiers**
> tasked with refuting it from the actual code — 44 confirmed, 1 disputed
> (nuanced below), 1 rejected (nuanced below).

---

## 1. Executive summary

**Project owner's goal: make SciRust the world reference for scientific
algorithms — implementation, improvement, testing.** This document measures,
without complacency, the gap between that ambition and the actual state of
the code.

**Overall verdict: a serious and often honest engineering foundation, but not
yet, algorithmically, a world reference.**

What goes well, and beyond the average of a project of this size:
- The implementations are overwhelmingly **faithful to their published
  sources** (every module cites its references — Golub & Van Loan, Hairer,
  Hansen, Mihalcea & Tarau…) and **deterministic by construction**, a rare
  architectural choice consistent with the stated ambition.
- The best of the repository (`scirust-sparse`, `scirust-gp`, `scirust-stiff`,
  `scirust-tolerance`, the reproducibility path of `scirust-core`) is
  **validated against external references** (independent dense oracles,
  NIST/AIAG tables, PdM/reliability literature) — this is the practice that
  distinguishes a serious numerical library, and it already exists, but
  unevenly.
- The CI infrastructure (Miri, cross-arch qemu, GPU oracles, rounding
  certification campaign over 30 billion f32 inputs) is **at the state of
  the art** for determinism verification.

What currently blocks reference status:
- **5 P0 defects**: mathematically wrong results in common use cases, not
  exotic ones (incomplete gamma wrong for every χ² test with a large number
  of degrees of freedom; the RL PPO and actor-critic do not learn or learn
  backwards; dual differentiation fails on any power of a negative number; a
  fused GPU GEMM returns a wrong result and is publicly exposed with no test
  at all).
- **41 confirmed P1 defects**: classic numerical instabilities (absolute
  thresholds not dimensionless, catastrophic cancellation, non-Joseph Kalman
  forms), robustness bugs (CG/BiCGSTAB that fail on `b=0`), and entire
  families of algorithms missing despite a doc or API that suggests them
  (IIR/FIR filtering, ARIMA, non-symmetric eigenvalues).
- **A massive coverage gap** versus SciPy/LAPACK/GSL/statsmodels/
  scikit-learn: each module covers roughly 10 to 40 % of its reference
  equivalent. This is, in the long run, the heaviest workstream.
- **A test methodology that caps out at self-consistency**: of 157
  findings, the overwhelming majority of test gaps share the same signature —
  very few external reference values (SciPy/R/published tables), no property
  tests (proptest/quickcheck absent from the ~120 crates), a single fuzzing
  target. This is precisely the hole through which the 5 P0s went unnoticed.

### Score table by domain

| Domain | Score | What blocks reference status |
|---|:---:|---|
| Dense and sparse linear algebra | B− | No non-symmetric eigen, no rcond, absolute thresholds, CG/BiCGSTAB broken on `b=0` |
| ODE, optimization, quadrature, roots | B− | DOPRI5 rejection counter wrong, fatal NaN, no BDF/L-BFGS/Levenberg-Marquardt |
| Tensor/SIMD kernels, reproducibility | B+ | Double-rounded subnormal violates the "correctly rounded" contract; GEMM very far from a BLAS |
| Autodiff, NN, optimizers | B | DP accountant not *sound*, Lottery Ticket stagnates, dummy GQA |
| Statistics, special, multivariate, estimation | B− | Incomplete gamma wrong (P0), normal quantile destroyed in the tail, PCA wrong at small scale |
| Signal and audio | C+ | No IIR/FIR filter design, O(N²) DFT throughout scirust-audio, power-of-two-only FFT |
| Time series, seasonality, finance | C+ | Circular backtest, wrong cointegration threshold, no ARIMA |
| Classic ML, clustering, AutoML, GP, RL | C+ | PPO and actor-critic broken (P0), numerically unstable GMM/EM, unweighted CART |
| Evolutionary, symbolic, synthesis | C+ | 3 mathematically wrong symbolic results (P0/P1), a CMA-ES that is not one |
| Reasoning, SAT/CSP, graphs, NLP | C+ | TextRank and UMass wrong, neither CDCL/AC-3/Dijkstra/Tarjan/HNSW |
| GPU, tensors, einsum, TN | C+ | Fused GEMM wrong and untested (P0), broken Max/Norm reduction |
| Certifiable industrial verticals | B | One-sided Page-Hinkley, non-Joseph covariance, fabricated ISO 10816 RMS |
| Test infrastructure (meta-audit) | B+ | Zero property tests, a single fuzz target, no perf tracking |
| Coverage gap vs state of the art | C+ | 10-40 % coverage per module versus SciPy/LAPACK |

---

## 2. The 5 P0 defects — to fix first

All confirmed by two independent verifiers re-reading the source code.

1. **`scirust-special/src/lib.rs:328`** — `regularized_gamma_p` is silently
   wrong for large `a`: `regularized_gamma_p(1e4, 1e4) = 0.49994` against the
   true value `0.50133` (error raised by execution). Cause: the series
   converges in O(√a) iterations near `x≈a` but `MAX_ITERS=300` is fixed.
   Direct impact: every χ² test with a large number of degrees of freedom is
   wrong. Reference: Temme, *The asymptotic expansion of the incomplete
   gamma functions* (1979/1987) — this is the switch that `cephes igam.c`
   makes in SciPy.

2. **`scirust-learning/src/rl/ppo.rs:58`** — the clipped term of the PPO
   surrogate is built outside the differentiation graph, and the
   clipped/unclipped `min` uses a strict comparison that, in the nominal zone
   `ratio ∈ [1-ε, 1+ε]`, systematically selects the constant: **the policy
   gradient is zero on the first pass for all samples**. Reference: Schulman
   et al., *PPO*, 2017, eq. 7.

3. **`scirust-rl-algo/src/lib.rs:1309`** — `ActorCriticAgent::update`
   computes `grad_scale = td_error * log_prob` (`log_prob ≤ 0` always): for an
   action better than expected, the update **decreases** its probability —
   inverted sign. The exact fix already exists, in a comment, in
   `ReinforceAgent` of the same file (line 1148) but was not applied here.

4. **`scirust-symbolic/src/lib.rs:1002`** — `Dual::powf` uses the
   log-derivative formula `v·(other.tangent·ln(self.primal) + …)`, valid only
   for a positive base. `Dual::var(-3.0).powf(Dual::primal(2.0))` returns a
   `NaN` tangent instead of `-6` — every derivative of an integer power of a
   negative variable is wrong, a very common autodiff case.

5. **`scirust-gpu/src/kernels.rs:82`** — the shared-memory indexing of
   `TILED_GEMM_WGSL` and `FUSED_GEMM_WGSL` mistakenly transposes the
   roles of the row/summation index. Exact simulation of the WGSL semantics:
   maximum error of 15 to 33 on random 16×16×16 and 32×32×32 GEMMs, for
   **all** sizes. `FUSED_GEMM_WGSL` is dispatched by the public API
   `FusedLayer::execute`, and the only two tests of the fusion path
   **deliberately short-circuit** (`eprintln!("skipped")` + `return`) — that
   is exactly why the bug survived.

---

## 3. Confirmed P1 defects — by theme

### 3.1 Numerical robustness and poorly dimensioned thresholds (the most repeated pattern)

This pattern recurs in **8 different domains** — it is the systemic problem
no. 1 of the repository:

- `scirust-solvers/src/linalg/iterative.rs:104` / `bicgstab.rs:102` — CG and
  BiCGSTAB return `Err(Singular)` on `b=0` or when `x0` is already a solution
  (no initial-convergence test), and the `1e-13` breakdown threshold is
  applied to a **quadratic** quantity: any system with `‖b‖ ≲ 3e-7` fails
  wrongly. GMRES and the sparse CG, in the same repository, do this test
  correctly — it is a local oversight, not a choice.
- `scirust-sparse/src/lib.rs:53`, `lu.rs:13`, `cholesky.rs:10`, `qr.rs:221` —
  **absolute** pivot thresholds (`1e-12`, `1e-14`): a matrix at a natural
  physical scale (farads, micro-units) is declared singular although it is
  perfectly invertible.
- `scirust-multivariate/src/lib.rs:344` — the same defect in `jacobi_eigen`:
  PCA of data correlated at scale `1e-7` returns wrong eigenvectors
  (`[1,0]` instead of `[0.707, 0.707]`) because the covariance `~2e-14`
  falls below the absolute convergence threshold.
- `scirust-estimation/src/linalg.rs:165` — same absolute pivot in the
  inverse used by the Kalman filters.
- `scirust-nav/src/fusion.rs:115`, `scirust-estimation/src/kalman.rs:84` —
  covariance update in the short form `P←(I−KH)P` instead of the
  **Joseph form**, which guarantees symmetry and positive-definiteness in
  finite precision (Bucy & Joseph 1968) — an unacceptable defect in crates
  that claim to be *certifiable*.
- `scirust-reasoning/src/lib.rs:77`, `scirust-symbolic` — the classic
  quadratic formula `(-b±√Δ)/(2a)` without the stable form of Numerical
  Recipes §5.6: catastrophic cancellation when `b² ≫ 4ac`.

**Cross-cutting recommendation**: a systematic audit of all absolute
thresholds in the repository (`grep` on `1e-1[0-9]` in the numerical
computation crates) and their replacement by criteria relative to the
norm/scale of the input data, LAPACK-style.

### 3.2 Mathematically wrong formulas (outside P0)

- **TextRank** (`scirust-nlp-advanced/src/keyword.rs:180`) normalizes by the
  degree of the receiving node instead of the emitting one — contrary to
  Mihalcea & Tarau 2004.
- **UMass coherence** (`topic.rs:334`) computes a ratio of logarithms
  instead of the logarithm of a ratio — wrong sign and magnitude versus
  Mimno et al. 2011.
- **`prove_equal`** (`scirust-symbolic/src/lib.rs:661`) declares `ln(x)`
  non-equivalent to itself as soon as the first sampling point falls outside
  the domain of `ln`.
- **`solve_quadratic`** (two independent occurrences,
  `scirust-symbolic/src/lib.rs:535` and `scirust-reasoning/src/lib.rs:77`) —
  one extracts coefficients by sampling without checking the actual degree
  (`x³-2` returns a "root" `2.0`), the other suffers from the classic
  catastrophic cancellation.
- **DP-SGD** (`scirust-core/src/dp.rs:136`) — the moments accountant
  underestimates the subsampled Gaussian log-moment by a factor
  `~2(λ+1)` relative to Lemma 3 of Abadi et al. 2016: **ε is under-reported,
  the displayed privacy guarantee is wrong** (not *sound*), and the test
  locks in the erroneous value instead of the correct one.
- **Lottery Ticket** (`scirust-core/src/pruning.rs:208`) — `prune_magnitude`
  retrains all weights (including the already-pruned zeros), so sparsity
  plateaus at `p` instead of converging to `1-(1-p)^k`.
- **RL, more broadly**: DQN without ever synchronizing its target network
  (`scirust-learning/src/rl/deep.rs:114` — `grep target_model` surfaces only
  one file, no `update_target` method), GMM/EM destroyed by an `exp`/`ln`
  round-trip (`scirust-unsupervised/src/lib.rs:786`), CART not weighted by
  counts (`scirust-automl/src/lib.rs:1225`), negative Cholesky pivots masked
  as `NaN.max(1e-12)` (`lib.rs:913`).

### 3.3 Algorithms absent despite a doc/API that suggests them

- **No classic filter design** anywhere in `scirust-signal`/`scirust-audio`
  (no Butterworth/Chebyshev/elliptic, no bilinear, no biquad/SOS, no
  `filtfilt`) — *nuance*: the `denoise` module already contains real
  frequency-domain filters (`fft_lowpass`, `notch_filter`), so the initial
  finding "no filtering function" was too absolute; the real, confirmed gap
  is the absence of **pole/zero filter synthesis** (classic IIR/FIR), which
  remains the no. 1 DSP hole.
- `scirust-audio` computes **all** its spectral features (MFCC, centroid,
  bandwidth, rolloff, flatness, entropy) via a **naive O(N²) DFT** while
  `scirust-signal`, a direct dependency, exposes an O(N log N) FFT that is
  never called — for 10 s at 44.1 kHz, ~2×10¹¹ operations.
- **CmaEs is not CMA-ES** (`scirust-evo/src/lib.rs:143`): no covariance
  matrix, no evolution paths, constant `σ` — an isotropic ES under a
  misleading name, incapable of converging on Rosenbrock.
- **Financial cointegration/backtest**: ADF threshold hardcoded to `-2.5`
  (real MacKinnon critical value ≈ `-3.34`) without lag augmentation
  (`scirust-trader/src/pairs.rs:196`); the risk engine's "backtest" uses the
  model's prediction as realized PnL — circular by construction
  (`scirust-trader/src/risk.rs:245`).
- **Incomplete GPU reduction** (`scirust-gpu/src/kernels.rs:131`) —
  `DETERMINISTIC_REDUCE_WGSL` implements neither `Max` nor `Norm` (returns
  `sqrt(sum)` for both), although the public enum declares them.
- **`einsum` panics on zero dimension**, **`fixed_point_gemm_q16` overflows
  silently in i32** — two confirmed robustness defects on the GPU/tensor
  side.
- **One-sided Page-Hinkley** (`scirust-pdm/src/change_detection.rs:167`) —
  only detects upward drifts; a typical predictive-maintenance degradation
  (drop of a health index) is **never** detected, despite an API that
  exposes `direction: -1`.
- **Fabricated ISO 10816 severity** (`scirust-pdm/src/detectors.rs:184`) — a
  hardcoded unitless `0.5` factor converts an FFT amplitude into a fake RMS
  velocity in mm/s, while a correct `iso10816.rs` module exists and is not
  used.

*Two initial findings were revised after adversarial verification:* the
alleged lack of "any constrained/global optimizer" was too broad —
`scirust-solvers::spg` (Spectral Projected Gradient, box constraints) and
`scirust-evo::{CmaEs, GeneticAlgorithm}` (global optimization) already exist;
the real confirmed gap remains the absence of **Levenberg-Marquardt** for
nonlinear least squares (no `curve_fit` anywhere in the repository).

---

## 4. Coverage gap vs SciPy / LAPACK / GSL / statsmodels / scikit-learn

Each module covers roughly **10 to 40 %** of its reference equivalent. The
most structural gaps, in order of estimated impact:

| SciPy/LAPACK domain | Critical gap | Reference |
|---|---|---|
| `scipy.signal` | Full IIR/FIR filtering, arbitrary-size FFT (Bluestein), Welch/STFT | Oppenheim & Schafer; Bluestein 1970; Welch 1967 |
| `scipy.linalg` | Non-symmetric eigen (Francis QR), `expm`/`logm`, `rcond`, rank-deficient `lstsq` | Golub & Van Loan ch. 7; LAPACK dgeev/dgelsd |
| `scipy.optimize` | Levenberg-Marquardt/`curve_fit`, L-BFGS(-B), scipy-style global optimization | Moré 1978 (MINPACK); Byrd et al. 1995 |
| `scipy.special` | Bessel J/Y/I/K, Airy, Carlson elliptic integrals, Lambert W, expint | Amos 1986; Abramowitz & Stegun |
| `scipy.stats` | Discrete distributions, non-parametric tests (Mann-Whitney, Wilcoxon, Shapiro-Wilk), KDE | Royston 1995; Silverman 1986 |
| `scipy.integrate` | Gauss-Kronrod/QUADPACK, variable-order BDF, Radau IIA | Piessens et al. 1983; Hairer & Wanner |
| `scipy.sparse` | Eigensolvers (Lanczos/ARPACK), sparse Cholesky + AMD, spgemm | Lehoucq, Sorensen & Yang 1998 |
| `scipy.spatial` | KD-tree, cdist/pdist, convex hull, Delaunay (absent → DBSCAN/LOF in O(n²)) | Bentley 1975; Barber et al. 1996 |
| `scipy.interpolate` | B-splines, smoothing splines, 2D/ND interpolation, RBF | de Boor 1978; Dierckx 1993 |
| statsmodels | ARIMA/SARIMA, GARCH, calibrated ADF/KPSS/Johansen tests | Box & Jenkins; Bollerslev 1986 |
| scikit-learn | Hierarchical/spectral/HDBSCAN clustering, full-covariance GMM, permutation importance | Ng, Jordan & Weiss 2002 |
| networkx/MiniSat/Z3 | Weighted Dijkstra/A*/Tarjan, CDCL, AC-3, SMT Simplex | Marques-Silva & Sakallah 1996 |

This is the complete list, with every bibliographic reference, that appears
in the per-domain detailed reports (see `missing_algorithms` of the raw
audit, kept in the branch history for traceability).

---

## 5. Test infrastructure — the real structural lever

**~4,147 `#[test]`** in total, a solid base in volume. The quality of the top
of the basket (`scirust-sparse`, `scirust-gp`, `scirust-stiff`,
`scirust-tolerance`, `portable_f32`) is excellent: independent dense oracles,
external tabulated values (NIST/AIAG), exhaustive campaigns (30 billion f32
inputs for the transcendentals).

But three structural gaps alone explain why the 5 P0s and most of the P1s
went unnoticed:

1. **No property tests** (`proptest`/`quickcheck`) in the ~120 crates. This
   is the standard practice of SciPy (`hypothesis`, since 2021) and of LAPACK
   (residual ratios on matrices generated with prescribed conditioning,
   *LAPACK Users' Guide* ch. 7). Without it, only the points one thought to
   test get tested — exactly the `b=0`, negative-base, zero-dimension,
   degree-≠2 cases that produced the P0/P1s.
2. **A single fuzzing target** (`qsr1_from_bytes`) over the whole
   repository; the untrusted ONNX parser and no numerical kernel are fuzzed;
   no SIMD/scalar/GPU differential fuzzing.
3. **Very few external reference values.** Most tests are self-consistency
   (reconstruction, internal properties) rather than comparisons to
   SciPy/R/LAPACK/published tables — and when an external value exists, it
   is sometimes tested with a tolerance so loose that it masks the error
   (`p < 1e-5` while the exact value `2.1e-6` is known and documented in a
   comment, never asserted).

**This is the most cost-effective workstream**: introducing `proptest` in
`scirust-solvers`, `scirust-core` (linalg), `scirust-stats` and
`scirust-special` would probably capture, on its own, a good share of the
remaining P1s before they are found in production.

---

## 6. Prioritized work plan

### Workstream 0 — Fix the 5 P0s (blocking, this week)
1. `regularized_gamma_p`/`_q`: Temme asymptotics for large `a`.
2. PPO: make the clipped/unclipped `min` properly differentiable.
3. `ActorCriticAgent`: `grad_scale = td_error` (remove the `* log_prob`).
4. `Dual::powf`: special-case constant exponent (`n·x^(n-1)·x'`).
5. Tiled GPU GEMM: fix the shared-memory indexing + finally enable the
   fusion tests (remove the `eprintln!("skipped")`).

### Workstream 1 — Absolute → relative thresholds (cross-cutting pattern, 1-2 weeks)
Systematic audit of all hardcoded `1e-1[0-9]` thresholds in
`scirust-solvers`, `scirust-sparse`, `scirust-multivariate`,
`scirust-estimation`; replacement by criteria relative to the input norm
(LAPACK-style); add the missing initial-convergence test in
CG/BiCGSTAB.

### Workstream 2 — Numerically safe forms in the "certifiable" verticals
Joseph form for all Kalman filters (`nav`, `estimation`), stable quadratic
formula everywhere, DARE via Schur rather than fixed point.
These crates claim certifiability: the proof bar must be maximal here
first.

### Workstream 3 — Test foundations (pays for everything else)
`proptest` in the 4 core numerical crates; a second fuzzing target
(ONNX parser); SIMD/scalar/GPU differential fuzzing; lock in the external
reference values already known but not asserted (e.g. the `p ≈ 2.1e-6`
documented in a comment).

### Workstream 4 — Fill the gaps that make up 80 % of usage
In order of estimated impact on usability:
1. IIR/FIR filtering (`scirust-signal`) — Butterworth/Chebyshev + bilinear + SOS.
2. Levenberg-Marquardt / `curve_fit` (`scirust-solvers`).
3. Non-symmetric eigensolver + `expm` (`scirust-solvers/linalg`).
4. ARIMA/SARIMA with Kalman MLE (`scirust-forecast`).
5. Bessel J/Y/I/K (`scirust-special`) — also unlocks the Kaiser window.
6. `scirust-spatial` (KD-tree, cdist/pdist) — unlocks DBSCAN/LOF in O(n log n).
7. Non-parametric tests + discrete distributions (`scirust-stats`).
8. CDCL/AC-3/weighted Dijkstra/Tarjan (`scirust-neuro-symbolic`, `scirust-graph`).

### Workstream 5 — True CMA-ES, e-graphs, GAE for PPO
Once workstream 0 is closed: implement the full Hansen CMA-ES (the name is
currently misleading), e-graphs for symbolic simplification (explicitly
within the targeted scope), and the GAE computation that PPO currently
receives from outside without ever producing it.

---

## 7. Method and transparency note

Of the 46 P0/P1 findings submitted to adversarial verification (two
independent verifiers per finding, explicitly tasked with refuting), **44
were confirmed as-is**. Two were nuanced rather than simply validated —
treated accordingly in this report (§3.3):
- The finding "no filtering function" was judged **too absolute**: frequency
  filters exist in `scirust-signal::denoise`, but no classic filter synthesis
  (pole/zero IIR/FIR) — the precise reformulation replaces the initial
  finding.
- The finding about the absence of constrained/global optimization was
  **rejected as stated**: `spg` (box constraints) and `CmaEs`/
  `GeneticAlgorithm` (global) already exist in the repository; only the lack
  of Levenberg-Marquardt for nonlinear least squares is retained.

The 111 P2/P3 findings were not submitted to adversarial verification
(budget constraint); they remain single-domain-auditor observations and
should be re-verified before correction if any doubt exists.

The complete raw data (157 findings, 14 domain reports, detailed
verification verdicts) is kept in the session history associated with this
branch.
