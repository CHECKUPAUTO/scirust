# Adaptive Transformed Representation Algorithms (ATRA) — Scientific Evaluation

**Date:** 2026-07-16
**Status:** Research investigation (phase 2, after the TSA falsification). No production implementation.
**Companion document:** `docs/research/TSA_TRANSFORMED_SCALAR_ALGORITHMS_2026-07-16.md` (cited below
as **[TSA]**; its Propositions 1–5 and its 24-row literature table are reused, not repeated).
**Method:** falsificationist; proofs and counterexamples first; reproducible experiments
(X1–X5, scripts in `docs/research/atra_experiments/`, pure stdlib, fixed seeds, multiple seeds
with dispersion where stochastic); load-bearing citations verified online; search-term log in
Appendix C as required by the mission's §10.

---

## 0. Verdict (read this first)

**ATRA as posed — the full class `x̂ = D(F(E(x)))` with adaptive `(E, F, D)` — is not a
hypothesis; it is the space of all computational pipelines** (Lemma 1: take `E = D = id`,
`F = A`). It therefore cannot be validated or falsified as stated, and it cannot be a research
field. Scientific content appears only after restricting the representation family, the operator
family, and the selection rule — and each useful restriction investigated here lands in a named,
mature field:

- **operator redesigned for the representation** → transform coding and its learned successor
  *nonlinear transform coding* (Ballé et al. 2021), wavelet shrinkage, dictionary learning,
  Bregman/mirror methods, invariant filtering;
- **non-invertible / lossy encoder with a designed decoder** → quantization theory, dithering,
  stochastic rounding, sketching, compressed sensing, product quantization, tropical limits;
- **context-adaptive selection of the representation** → rate–distortion-optimized codec design,
  best-basis selection, Box–Cox/link selection in statistics, precision autotuning
  (Precimonious/FPTuner/Herbie), mixed-precision and block-scaled formats (AMP, MXFP), learned
  indexes.

The mission's §5 optimization formulation **already exists under several names**: the
rate–distortion Lagrangian in coding, error-constrained cost minimization in precision
autotuning, and hardware-aware multi-objective search in quantized ML (§2.4).

**What survives, concretely, is engineering plus two narrow research-flavored items:**

1. A **finite-precision design law** for scalar transforms (Proposition 4): round-trip relative
   error ≈ ε·κ_rt(x) with κ_rt = |φ(x)| / |x·φ′(x)|. It predicts every anomaly measured in
   experiment X1 — including a genuinely useful, little-documented finding: **the Box–Cox
   −1/λ shift is numerically toxic** (inherent 4×10⁴-ulp round-trip loss near 0 that *no*
   rewriting can fix), while μ-law's matching defect is a pure implementation bug fixable by one
   `expm1` (3×10⁸ ulps → 3 ulps, X1). Elementary analysis, but it packages into a certified
   transform library that does not currently exist in Rust.
2. A **representation-selection autotuner** for numerical kernels: X3 shows dev-set selection of
   a Box–Cox exponent reaches within 0.2 dB of the Lloyd–Max optimum for 8-bit quantization of
   lognormal data (44.9 vs 45.1 dB SQNR; fixed transforms 34.8–42.7 dB). The gain is real; the
   theory is classical (Bennett 1948, Panter–Dite 1951, Lloyd 1957); the *tool* — a
   dictionary-plus-protocol autotuner at algorithm level in a scientific Rust stack — is a
   defensible engineering/benchmark contribution, not new mathematics.

**Recommended decision (deliverable 10): an engineering module plus a benchmark paper. Not a
research program.** Detail in §11.

---

## 1. Corrected theoretical framework (deliverable 1)

### 1.1 The framework, stated so it can be falsified

**Lemma 1 (universality ⇒ vacuity).** Let 𝒫 = {D∘F∘E} over unconstrained maps. Every algorithm
A equals id∘A∘id ∈ 𝒫. Hence membership in 𝒫 has no empirical content; "ATRA" cannot be a
property of an algorithm. ∎

The corrected object of study is therefore a **quadruple**:

```
ATRA instance = ( ℛ,  ℱ,  S,  J )
  ℛ : a declared representation family (e.g., componentwise monotone maps with κ_rt ≤ B;
      block-scaled integers; k-bit quantizers; unit-quaternion charts)
  ℱ : a declared operator family in representation space
  S : a selection rule  θ = S(c)  from context c (data statistics, noise model, format, hardware)
  J : an objective (error, runtime, memory, energy, distortion) with a stated evaluation protocol
```

A claim of the form "this ATRA instance beats baseline B on task T under protocol P" is
falsifiable. Nothing weaker is.

### 1.2 The three escape routes from conjugation triviality

[TSA, Prop. 1–2] showed `E⁻¹∘A∘E` with fixed bijective E is sterile. A pipeline escapes that
result in exactly three ways, and each is a known research territory:

| Route | Mechanism | Existing fields |
|---|---|---|
| R1 — operator redesign | F ≠ E∘A∘D; F is matched to the representation's statistics/geometry | transform coding, wavelet shrinkage, sparse coding, mirror descent/Bregman, GLM, invariant Kalman filtering |
| R2 — information selection | E many-to-one/quantized/stochastic; D is an *estimator*, not an inverse | quantization + dithering, stochastic rounding, sketching/streaming, compressed sensing, product quantization, rank/sign statistics, tropical limits |
| R3 — context adaptation | θ = S(c) varies with data/noise/hardware | R–D-optimized coding, best-basis selection, Box–Cox/link selection, VST choice, precision autotuning, mixed precision/MX formats, learned indexes, input-warped GPs |

### 1.3 The eight notions that must not be conflated (mission §3)

| Notion | Definition | Canonical anchor |
|---|---|---|
| value warping | pointwise map on sample values | companding, VST [TSA §4 rows 2–3] |
| domain warping | resampling the independent variable | mel/constant-Q, NUFFT, DTW [TSA §4 row 17] |
| operator transformation | act on the operator's spectrum/structure, `f(A)` | functional calculus, shift-invert [TSA §4 row 16] |
| structure transport | redefine arithmetic through a bijection | quasi-arithmetic means [TSA §2.3] |
| representation change | change the *number format* (bits, scale, base) | FP/LNS/SLI/posit/block-FP/MX (§2.3) |
| numerical preconditioning | change coordinates to improve conditioning of a solver | linear & nonlinear preconditioning (ASPIN) |
| learned normalization | data-driven standardization inside a model | batch/layer norm, GDN |
| nonlinear filtering | nonlinear estimator on the raw representation | median/bilateral/NLM, particle filters |

Every claim below names which notion it uses. The mission text itself mixes them (e.g., "μ-law"
is value warping *for the purpose of* representation change; "rank transforms" are information
selection); the taxonomy in §3 keeps them separated.

---

## 2. Literature map (deliverable 2)

[TSA §4] holds the 24-row map for fixed-transform conjugation. The rows below are the *adaptive
and lossy* additions that ATRA specifically needs. Verified-this-session citations are marked ✓.

### 2.1 Route R1 — operators designed for representations

| Field | Representation | Redesigned operator | Key references |
|---|---|---|---|
| Transform coding (classical) | KLT/DCT coefficients | scalar quantizer + entropy coder per band | Kramer–Mathews 1956; Huang–Schultheiss 1963; Goyal, *IEEE SPM* 2001 |
| Nonlinear transform coding ✓ | learned analysis/synthesis nets + GDN | quantizer + learned entropy model, jointly R–D-optimized | Ballé–Laparra–Simoncelli, ICLR 2017; Ballé et al., *IEEE JSTSP* 15(2):339–353, 2021 |
| Wavelet shrinkage | orthonormal wavelet coefficients | soft/hard thresholding, SURE-optimal | Donoho–Johnstone, *Biometrika* 1994; Blu–Luisier (SURE-LET) 2007 |
| Sparse coding / dictionary learning | learned overcomplete dictionary | pursuit + learned shrinkage (LISTA) | Olshausen–Field 1996; Aharon–Elad–Bruckstein (K-SVD) 2006; Gregor–LeCun, ICML 2010 |
| Plug-and-play / unrolling | proximal splitting variables | learned denoiser as proximal operator | Venkatakrishnan et al. 2013; Romano–Elad–Milanfar (RED) 2017 |
| Invariant/geometric filtering | Lie-group (quaternion/SE(3)) charts | error-state and invariant EKF; chordal averaging ✓ | Markley–Cheng–Crassidis–Oshman, *JGCD* 30(4):1193–1197, 2007; Barrau–Bonnabel (IEKF) 2017 |

### 2.2 Route R2 — lossy encoders with designed decoders

| Field | Encoder (lossy) | Decoder/estimator | Key references |
|---|---|---|---|
| Quantization theory | scalar quantizer | centroid decoding; companding | Bennett 1948; Panter–Dite 1951; Lloyd 1957/1982; Max 1960; Gersho 1979; Gray–Neuhoff 1998 |
| Dithered quantization | quantizer + random dither | linear estimator; moment-exact | Roberts 1962; Gray–Stockham 1993; Σ-Δ theory: Daubechies–DeVore 2003 |
| Stochastic rounding ✓ | probabilistic rounding to p bits | plain accumulation (unbiased) | Croci–Fasi–Higham–Mary–Mikaitis, *R. Soc. Open Sci.* 9:211631, 2022; Gupta et al., ICML 2015 |
| Sketching / streaming | random projections, hashes | tailored estimators (AMS, CountSketch) | Johnson–Lindenstrauss 1984; Alon–Matias–Szegedy 1996; Charikar 2002 |
| Compressed sensing | non-adaptive underdetermined linear E | optimization decoder (ℓ₁) | Candès–Romberg–Tao 2006; Donoho 2006; 1-bit CS: Boufounos–Baraniuk 2008 |
| Product quantization ✓ | per-subspace codebooks | asymmetric distance computation (ADC) — an operator designed for codes | Jégou–Douze–Schmid, *IEEE TPAMI* 33(1):117–128, 2011 |
| Error-free transformations | pair/expansion representation of FP results | compensated summation/dot | Dekker 1971; Knuth; Ogita–Rump–Oishi 2005; Shewchuk 1997; Neumaier 1974; Kahan 1965 |

### 2.3 Route R3 — context-adaptive representation selection

| Field | Context c | Selection S | Key references |
|---|---|---|---|
| R–D-optimized coding | source statistics, bitrate | Lagrangian sweep over codec params | Shoham–Gersho 1988; Sullivan–Wiegand 1998 |
| Best-basis selection | signal at hand | entropy-minimizing basis from a wavelet-packet library | Coifman–Wickerhauser, *IEEE TIT* 1992; basis pursuit: Chen–Donoho–Saunders 1998 |
| Statistical transform selection | data distribution | ML/profile-likelihood over λ | Box–Cox 1964; Yeo–Johnson 2000; GLM link choice: Nelder–Wedderburn 1972 |
| Precision autotuning ✓ | program, accuracy bound | search over per-variable formats | Rubio-González et al. (Precimonious), SC 2013; Chiang et al. (FPTuner), POPL 2017; Darulova–Kuncak (Rosa/Daisy) 2014–2018; Panchekha et al. (Herbie), PLDI 2015 |
| Mixed precision / block scaling ✓ | tensor statistics, hardware | per-op format choice; per-block shared scale | Micikevicius et al. (AMP), ICLR 2018; Darvish Rouhani et al. (MSFP), NeurIPS 2020; OCP Microscaling (MX) v1.0 spec, Sept 2023 (MXFP8/6/4, MXINT8: 32-element blocks, shared E8M0 scale) |
| Hardware-aware quantized ML | latency/energy budget | RL/Hessian-guided bit allocation | Wang et al. (HAQ), CVPR 2019; Dong et al. (HAWQ), ICCV 2019 |
| Learned data-structure representations ✓ | key distribution | learned monotone CDF model replaces index | Kraska–Beutel–Chi–Dean–Polyzotis, SIGMOD 2018, pp. 489–504 (RMI) |
| Learned monotone transforms for models | output/input marginals | warped GP, input warping, Gaussianization, flows | [TSA §4 rows 12–13]; Durkan et al. (neural spline flows) 2019 |

**Conclusion of the map:** all three routes are populated by decades of work; the mission's §5
objective is the standard multi-objective autotuning/R–D formulation. No empty quadrant was
found at the level of *frameworks*. The open space is at the level of *specific tools and
measured combinations* (§9).

---

## 3. Taxonomy of representations (deliverable 3)

| Class | Examples | Invariant preserved | Information lost | Matched operator family | Closest field |
|---|---|---|---|---|---|
| fixed monotone closed-form | log, log1p, asinh, sqrt, logit, μ-law, tanh | order; topology | none (in ℝ); low bits in FP (κ_rt, §5.P4) | linear/threshold ops on warped values | companding, VST |
| parametric monotone family | Box–Cox(λ), Yeo–Johnson(λ), μ-law(μ) | order | as above + boundary conditioning | same, with λ selected on dev data | statistical transform selection |
| noise-model-derived | Anscombe, GAT, Fisher z, arcsine | order; approx. homoscedasticity | tail exactness (low counts) | Gaussian estimators + unbiased inverse | variance stabilization |
| distribution-derived nonparametric | quantile/rank maps, copulas, Gaussianization | order (per coordinate) | inter-coordinate scale | Gaussian-model operators | nonparanormal, copula methods |
| learned constrained-monotone | RQ splines, monotone nets, integrated-positive nets | order (by construction) | depends on regularization | downstream model; needs certified inverse | normalizing flows, warped GP |
| adapted linear bases | PCA/KLT, dictionaries, wavelet packets | linearity | none (orthogonal) / redundancy (frames) | diagonal scalar ops (threshold, quantize) | transform coding, best basis |
| lossy deterministic | quantizer, clip, rank, sign, top-k | coarse order / sign | fine amplitude | centroid decoding, ADC, rank statistics | quantization, PQ, robust stats |
| lossy stochastic | dither, SR, sketches, random projection | expectations; distances w.h.p. | per-sample exactness | unbiased estimators | dithering, SR, sketching |
| number formats | binary64/32/16, bfloat16, LNS, SLI, posit, block-FP/MX | field ops up to rounding | dynamic range/precision trade | hardware arithmetic | computer arithmetic |
| manifold/group charts | polar ρ·q/|q|, quaternion log, SE(3) error state | group structure | chart-dependent (branch cuts) | intrinsic means, invariant filters | Lie-group estimation |

---

## 4. Branch analyses with experiments (Branches A–F)

### Branch A — adaptive scalar transforms (Experiment X1, Stage 1)

Round-trip accuracy, derivative range, and saturation for the mission's §6.A list, binary64,
worst case over the sampled domain (script `atra_experiments.py`, X1):

| transform | domain | max round-trip err (ulps) | min φ′ | max φ′ |
|---|---|---:|---:|---:|
| log/exp | x>0 | 254.1 | 1e−300 | 1e+300 |
| log1p/expm1 | x>−1 | 253.5 | 1e−300 | 1e+15 |
| asinh (signed log) | ℝ | 474.9 | ~0 | 1.0 |
| sqrt/square | x≥0 | **1.0** | 5e−151 | 5e+149 |
| Anscombe | x≥0 | 191 782 | 1e−06 | 1.63 |
| Box–Cox λ=0.5 | x>0 | 40 992 | 1e−05 | 1e+05 |
| Box–Cox λ=0.5, expm1/log1p rewrite | x>0 | 40 994 (**unchanged**) | — | — |
| Yeo–Johnson λ=0.5 | ℝ | 4.4×10⁹ | 1e−05 | 1e+05 |
| logit/sigmoid | (0,1) | 16.1 | 4.0 | 1e+16 |
| μ-law (μ=255), naive inverse | [−1,1] | 3.7×10⁸ | 0.18 | 46 |
| μ-law, expm1 inverse | [−1,1] | **3.0** | 0.18 | 46 |
| tanh/atanh | ℝ | 6.3×10¹² | 1.1e−15 | 0.99 |
| softsign | ℝ | 1.2×10¹¹ | 1e−24 | 1.0 |

Findings (all predicted by Proposition 4, §5):

- **A1. The Box–Cox shift is inherently harmful.** (x^λ−1)/λ stores the information about small
  x in the low bits of a value near −1/λ; rounding the *stored* value destroys it. No rewriting
  fixes this (the expm1/log1p variant is bit-for-bit as bad). The un-shifted power x^λ
  round-trips at 1 ulp (sqrt row). The −1/λ shift exists for *statistical* continuity in λ; it
  is numerically toxic near the boundary. Practical rule: **store x^λ (or log x), apply the
  affine shift symbolically.** We did not find this spelled out in the Box–Cox literature —
  novelty class: *useful implementation contribution* (§9).
- **A2. μ-law and Yeo–Johnson defects are implementation bugs, not representation defects.**
  One `expm1` repairs μ-law from 3.7×10⁸ to 3.0 ulps. The same rewrite family (expm1/log1p at
  every `pow(·)−1` and `1+·` site) applies to YJ's positive branch.
- **A3. Saturating sigmoids lose injectivity in FP.** tanh(x) = 1.0 exactly for x ≥ 19.0615 in
  binary64 (atanh(1) = ∞): tanh/softsign are *not* valid invertible representations outside a
  narrow core; they are clipping encoders (Branch E objects) beyond it.
- **A4. Log-family round-trip cost is |φ(x)|·ε** — log/asinh cost ~250–475 ulps at the extreme
  of the f64 range; harmless for f64, but at bfloat16 (ε ≈ 8×10⁻³) the same law makes log-domain
  storage of wide-range data lose whole digits: format context changes the verdict (Branch C).

### Branch B — learned numeric representations

Prior art is dense: Gaussianization (Chen–Gopinath 2000), nonparanormal, warped GPs, input
warping, GDN (a *learned divisive normalization built to Gaussianize* — Ballé–Laparra–Simoncelli
2016), neural spline flows, and — for the mission's "learned monotone map before search" —
learned indexes (Kraska et al. 2018), where a monotone CDF model *is* the representation and the
operator (lookup) is redesigned around it. What we did not find: monotone learned transforms
shipped with **certified inverse-error bounds** (interval/κ_rt certificates per spline segment)
for use inside classical numerical pipelines. That combination (flow-style RQ splines +
Proposition-4 certificates + dev/test protocol) is a *straightforward engineering combination*
— potentially valuable in a deterministic Rust stack, but not new mathematics (§9).

### Branch C — finite-precision-aware design (Experiment X2, Stage 2)

Summation shootout, n = 10⁵–1.8×10⁵ (script X2):

| method | (b) positive, 60-decade range | (c) signed, cancellation-heavy |
|---|---:|---:|
| naive | 2.3×10⁻¹⁴ | 1.2×10¹ (12 400 % error) |
| sorted ascending | 6.3×10⁻¹⁶ | — |
| pairwise | 0 | 2.1 |
| Kahan | 0 | 7.2×10⁻¹ |
| Neumaier | 0 | **2.2×10⁻¹⁴** |
| log-domain (sequential LSE) | 1.7×10⁻¹² | undefined (signs) |

plus the canonical `[1, 1e100, 1, −1e100]`: naive = Kahan = 0, Neumaier = fsum = 2 (exact).

**Finding C1 (cancellation no-go for value warping).** Monotone value warping cannot repair
signed cancellation: the damaging rounding happens when inputs/partials are *stored*, before any
transform sees them; and log-domain arithmetic is undefined for signed sums (sign-tracked LSE
moves the cancellation into expm1 rather than removing it). The representation that *does* fix
cancellation is the **unevaluated-sum (expansion) representation** of error-free
transformations — TwoSum pairs, double-double, Shewchuk expansions — i.e., an R2-style
representation co-designed with the operation (Ogita–Rump–Oishi 2005; Shewchuk 1997). X2 shows
Neumaier (EFT-based, 11 ms) dominating log-domain (39 ms, 100× less accurate here, positive-only).
**Baseline rule confirmed: compensated algorithms are the bar that any Branch-C proposal must
beat; none of the scalar warps do.**

Formats: LNS/SLI/posit/block-FP are the representation-level territory [TSA §4 row 10; §2.3
here]. The modern industrial instance of *adaptive* representation is block scaling (MXFP: a
shared E8M0 scale per 32-element block, selected per block from the data — OCP MX v1.0, 2023):
route R3 in silicon. A software MX/SLI scalar type remains the one worthwhile SciRust format
experiment [TSA §18 Phase 2].

### Branch D — operator redesign in transformed space

This is route R1 = §2.1's table; nothing tested here is unclaimed. The equivalences the mission
asks for: transformed-domain thresholding ≡ wavelet/VST shrinkage; learned shrinkage ≡
SURE-LET/LISTA; warped-distance NN ≡ metric learning / PQ-with-ADC; modified gradient updates in
a warped parameterization ≡ mirror descent/natural gradient [TSA §4 rows 6–7];
representation-dependent attention ≡ entmax/Performer [TSA §4 row 21]; nonlinear estimators
matched to transformed noise ≡ GLM/VST pipelines. Verdict: **route R1 is real and valuable and
entirely owned by existing fields**; contributions here can only be new *instances* with
measured wins against §12 baselines.

### Branch E — non-invertible and degenerate transforms (Experiment X4)

The algebraic boundary (max-plus, morphology, idempotent limits) was settled in [TSA §2.3]:
genuinely new structure exists there and is the existing field of tropical mathematics. The
*numerical* boundary is quantization with randomness. X4 (p = 10-bit significand, 2×10⁵
increments of 10⁻⁴, true sum 20):

- round-to-nearest: **stagnates at 0.25** (once the increment < ulp(s)/2, every update rounds
  away — the classic stagnation phenomenon);
- stochastic rounding: **20.16 ± 0.56** over 5 seeds — unbiased, no stagnation, error O(√n·u)
  w.h.p., exactly as the theory predicts (Croci et al. 2022).

Verdict: the "stochastic encoder" branch is confirmed valuable and confirmed *known* (dithering
1962 → SR revival 2015–2022). For SciRust it is an implementation item (deterministic SR needs a
counter-based RNG to preserve the repo's reproducibility guarantees). Rank/sign encoders are
robust statistics and 1-bit sketching; clipping/saturation is the overload region of
quantization theory. No unclaimed mathematics found at this boundary beyond what tropical
geometry already owns.

### Branch F — hypercomplex transformed representations (Experiment X5, Stage 5)

The algebra-transport question is closed by [TSA §7]. The legitimate reformulation — do
*nonlinear charts* (polar, log) of quaternion data improve concrete estimators? — was tested on
rotation averaging (100 observations/trial, 20 trials, von-Mises-like angular noise):

| noise σ (rad) | componentwise mean + renorm | chordal mean (Markley) | Karcher mean (log/exp chart) |
|---|---|---|---|
| 0.2 | 1.071 ± 0.480° | 1.062 ± 0.478° | 1.074 ± 0.481° |
| 0.8 | 4.651 ± 2.369° | 3.815 ± 1.837° | 4.455 ± 1.994° |
| 1.5 | 23.11 ± 43.80° (fails) | **6.706 ± 2.911°** | 9.959 ± 3.594° |

Findings: (i) at small noise all three coincide (the charts agree to second order); (ii) at
large noise the naive componentwise mean fails catastrophically (hemisphere wrapping); (iii) the
**linear-algebraic chordal mean beats the nonlinear log-chart (Karcher) mean** at high noise —
the "more geometric" representation is *not* automatically the better estimator. This entire
territory is standard attitude estimation (Markley et al. 2007; invariant filtering,
Barrau–Bonnabel 2017). Radial-angular transforms E(q) = ρ(|q|)·q/|q| decompose into a scalar
Branch-A problem on |q| plus a unit-sphere chart problem — both covered above. Verdict: **no new
hypercomplex framework; concrete estimator comparisons like X5 are useful engineering notes at
most.** For RGB/multispectral/IMU targets, quaternion methods must additionally beat independent-
channel baselines (mission §11 Stage 5 requirement — kept in the roadmap).

---

## 5. Mathematical analyses (deliverable 5)

**P1 (universality).** Lemma 1, §1.1. Consequence: all novelty claims must attach to a
restricted quadruple (ℛ, ℱ, S, J), never to the pipeline form itself.

**P2 (conjugation invariants).** [TSA Props. 1–2]: fixed bijective E with F = E∘A∘D changes no
exact-arithmetic behavior. All ATRA content therefore lives in routes R1–R3.

**P3 (cancellation no-go / EFT sufficiency).** Let x, y be binary64 inputs. Any componentwise
pre-transform φ acts on already-rounded values; the information lost in rounding x, y is not
recoverable by any φ (data processing). Within one subtraction, Sterbenz's lemma makes
fl(x−y) exact when y/2 ≤ x ≤ 2y — the damage in long sums comes from *accumulated* rounding,
and the error term of each addition is exactly representable and recoverable by TwoSum
(Knuth/Dekker), which is why compensated methods (Kahan, Neumaier, Ogita–Rump–Oishi) achieve
~ε-level error where warping cannot (X2c: Neumaier 2×10⁻¹⁴ vs naive 12.4). The effective
representation is the *expansion* (unevaluated sum), an R2 representation co-designed with +.
Classification: known theory; X2 is an instance.

**P4 (round-trip design law).** For strictly monotone C¹ φ, storing y = fl(φ(x)) with relative
error ε and inverting exactly gives recovered x̂ with, to first order,
```
|x̂ − x| / |x|  ≈  ε · κ_rt(x),      κ_rt(x) = |φ(x)| / |x · φ′(x)|.
```
*Proof:* δy = ε·|φ(x)|; δx = δy/|φ′(x)|; divide by x. ∎
Predictions vs X1 measurements: log at x = 10³⁰⁰: κ_rt = |ln x| = 691 → predicted ≤ ~700 ulps,
measured 254; tanh at x ≈ 18: κ_rt ≈ 1/(x(1−tanh²x)) ≈ 6×10¹³, measured 6.3×10¹²; Box–Cox
λ=0.5 at x = 10⁻¹⁰: κ_rt = |x^λ−1|/(λ·x^λ) ≈ (1/λ)·x^{−λ} = 2×10⁵, measured 4.1×10⁴; Anscombe
at x = 10⁻⁶: κ_rt ≈ (2√0.375)·(√0.375/x)/2… ≈ 3.75×10⁵ scale, measured 1.9×10⁵. All within the
first-order factor. **Design rules that follow:** (i) a representation for data accumulating at
x₀ must satisfy φ(x₀) = 0 (else κ_rt → ∞ there): log1p, asinh, μ-law, x^λ pass; Box–Cox and
Anscombe (shifted) fail at 0; (ii) κ_rt must be bounded on the data support at the *deployment
format's* ε, not f64's. Classification: elementary/folklore analysis (standard perturbation
calculus); its value is as a certification API, not as a theorem.

**P5 (quantizer matching is solved theory).** High-resolution theory gives compander distortion
D ≈ (Δ²/12)·E[g′(X)⁻²] (Bennett 1948) minimized by point density ∝ p^{1/3} (Panter–Dite 1951),
and Lloyd–Max iteration attains local MSE optimality. X3 instantiates it: dev-selected Box–Cox
(44.88 ± 0.23 dB) reaches within 0.22 dB of compander-initialized Lloyd–Max (45.10 ± 0.14 dB),
both far above fixed transforms (uniform 34.82, log 37.98, μ-law 42.68 dB; 3 seeds, L = 256,
clipped lognormal σ = 1.5). Methodological note preserved per §13: Lloyd from naive quantile
init stalled at 24–39 dB — the *selection/initialization procedure*, not the optimality theory,
is where implementations fail.

**P6 (stochastic rounding).** E[SR(x)] = x exactly; variance bounded by grid step²/4;
inner-product error O(√n·u) w.h.p. vs O(n·u) for RN, and immunity to stagnation (Croci et al.
2022, Thm-level results). X4 instantiates stagnation (RN stuck at 0.25/20.0) and unbiasedness
(SR 20.16 ± 0.56).

**P7 (monotone equivariance).** Rank/order-based operators are invariant under monotone
componentwise maps [TSA §2.4, E9]: no Branch-A transform can change median/rank/argmax
pipelines. Kill criterion for several §7 rows.

**P8 (noise-model correspondence).** VST design is the delta-method integral φ ∝ ∫dμ/√v(μ);
bias of D after nonlinear E is the Jensen gap, corrected by exact unbiased inverses
(Mäkitalo–Foi 2011) [TSA §8, E9/E9b including the negative result].

**P9 (lossy E is statistics).** When E is many-to-one, E(x) is a statistic; D is an estimator;
sufficiency characterizes when nothing is lost (Rao–Blackwell); JL/sketching bounds quantify
what survives dimension loss. The mission's Branch E is exactly estimation theory + sketching.

---

## 6. Candidate shortlist with the 10-question grid (deliverable 4)

Six candidates survive triage (all others in §7). Grid: (1) representation (2) match reason
(3) operator redesigned? (4) invariant kept (5) info lost (6) failure mode reduced (7) new
failure mode (8) closest method (9) baseline to beat (10) falsifier.

**C1 — Representation-selection autotuner for numerical kernels** (route R3).
(1) dictionary {id, log, log1p, asinh, x^λ, μ-law, Anscombe/GAT, quantile} × format {f64, f32,
f16, MX-block} (2) matched to measured input statistics + J (3) no — operators stay standard;
selection is the contribution (4) order (5) none (monotone members) (6) quantization/overflow/
range waste (X3: +10 dB over direct) (7) dev/test mismatch, selection overfitting (8) Precimonious/
Herbie (format/expression level), Box–Cox selection (statistics) (9) hand-chosen log + Lloyd–Max
oracle + Herbie output (10) if on ≥3 real SciRust workloads the tool never beats hand-chosen
log/VST by a pre-registered margin, kill.

**C2 — κ_rt-certified stable transform-pair library** (Branch A/C engineering).
(1) the X1 table's transforms with expm1/log1p-stable kernels and per-domain κ_rt certificates
(2) P4 law makes accuracy statable in docs/tests (3) no (4) order + certified round-trip bound
(5) none (6) silent 10⁴–10¹² ulp losses (X1: μ-law 3.7×10⁸→3.0) (7) none (pure fix) (8) libm
design practice (9) naive textbook formulas (10) trivially validated; not falsifiable — pure
engineering.

**C3 — Deterministic stochastic-rounding + EFT reduction kernels** (Branch C/E).
(1) p-bit values + counter-based RNG; expansions for accumulators (2) P3/P6 (3) yes — accumulation
redesigned (compensated/SR), not conjugated (4) unbiasedness; reproducibility (5) per-sample
determinism of RN (6) stagnation (X4), cancellation (X2) (7) variance, RNG management (8) Croci
et al.; Ogita–Rump–Oishi (9) Kahan/Neumaier/pairwise at same cost (10) if SR+EFT never beats
Neumaier-in-f32 on embedded workloads, drop SR from scope.

**C4 — VST + exact-unbiased-inverse + strong Gaussian denoiser pipeline** (route R1, carryover
[TSA §18 Phase 1]). Closest: Mäkitalo–Foi. Baseline: direct-domain Poisson denoisers. Falsifier
already partially fired ([TSA E9b] negative result at naive thresholding); survives only with
BM3D-class backends at low counts.

**C5 — Software MX/SLI scalar types for embedded Rust** (Branch C, carryover [TSA §18 Phase 2]).
Closest: OCP MX, Clenshaw–Olver/Turner. Baseline: f64 + manual rescaling; f32 + Neumaier.
Falsifier: <2× robustness-or-speed on a real reliability/fatigue workload.

**C6 — Estimator-selection note for rotation statistics** (Branch F).
(1) unit-quaternion charts (2) X5 crossover: chordal vs Karcher vs componentwise depends on σ
(3) yes (chordal eigen-operator) (4) rotation invariance (5) none (6) hemisphere failure of
naive averaging (23° → 6.7° at σ=1.5) (7) none (8) Markley 2007 (9) chordal mean is the
baseline and currently wins (10) already essentially settled by X5 — publishable only as a
SciRust doc/bench note.

---

## 7. Counterexamples and kill criteria (deliverable 6)

1. **ATRA-as-a-class:** unfalsifiable by Lemma 1 — killed as a field claim; only quadruples
   (ℛ, ℱ, S, J) may be claimed.
2. **Value warping against cancellation:** killed by P3 + X2c (log-domain undefined for signs;
   Neumaier 2×10⁻¹⁴ vs every warp's N/A).
3. **Box–Cox as a numeric storage format near 0:** killed by P4 + X1-A1 (inherent, rewrite-proof
   4×10⁴-ulp loss; use unshifted x^λ).
4. **tanh/softsign as invertible representations:** killed beyond |x| ≈ 19 (X1-A3, exact FP
   saturation).
5. **Rank/order pipelines with any monotone warp:** killed by P7 (exact no-op).
6. **"More geometric is better" for quaternion averaging:** killed by X5 (chordal linear mean
   beats the log-chart Karcher mean at σ = 1.5).
7. **Naive-threshold VST pipelines:** killed by [TSA E9b] in three count regimes.
8. **Naive Lloyd (quantile init) as an oracle:** killed by X3 methodology (stalls 6–20 dB below
   optimum; companding init required) — retained as a reproducibility lesson.
9. **Special-function transforms (Γ, ζ, Bessel, Airy) in any branch:** remain killed by
   [TSA §3] (non-injectivity, κ_rt divergence, cost, no intertwiner); nothing in the adaptive
   reframing rescues them — adaptivity selects *among* valid representations and they fail entry.
10. **New algebra from lossy scalar limits:** already owned by tropical/idempotent mathematics
    [TSA §2.3]; no second instance found.

## 8. Experimental roadmap (deliverable 7)

Protocol for every stage (mission §13): fixed seeds; ≥3 seeds with mean ± sd for anything
stochastic; dev/test splits for any selection (as in X3); pre-registered baselines from §12 of
the mission (identity, direct, log-domain, standard normalization, Box–Cox/YJ, VST,
Kahan/pairwise, preconditioning, learned normalization, standard multichannel); machine-readable
outputs; negative results retained in this document's lineage.

- **Stage 1 (done here):** X1 table + P4 certificates. Extend: f32/f16/bfloat16 κ_rt tables
  (the format is context — A4). Kill: none (measurement).
- **Stage 2 (done here):** X2 (reductions), X3 (quantization), X4 (SR). Extend: dot products,
  softmax/LSE, likelihood accumulation in f16; C3 prototypes vs Neumaier-f32. Kill criteria as
  in C1/C3.
- **Stage 3 (partially done in [TSA E9/E9b]):** Gaussian/Poisson/impulsive/multiplicative noise
  × {direct, VST, homomorphic} × {linear, median, threshold, NLM-class} with the P7 no-op rows
  excluded a priori. Kill: C4's criterion.
- **Stage 4:** ill-conditioned LS and GD under {identity, reparameterization, mirror maps,
  Jacobi/diag preconditioning} — expectation from theory: reparameterization ≡ metric change
  [TSA §6]; measure only where prior work leaves constants unquantified in low precision.
- **Stage 5:** X5 extension to IMU/RGB tasks with independent-channel baselines; report only if
  quaternion representations beat them (mission §11 requirement).

## 9. Novelty assessment (deliverable 8)

| Item | Class (mission §9 ladder) |
|---|---|
| ATRA framework as such | known methods under different terminology (transform coding / autotuning / sketching), plus Lemma-1 vacuity for the unrestricted form |
| §5 optimization formulation | exact rediscovery (R–D Lagrangian; error-constrained autotuning) |
| Branch A table + κ_rt law (P4) | known analysis; **useful implementation contribution** as a certified library (C2) |
| Box–Cox shift toxicity (A1) | **useful implementation contribution** (not found stated in the transform literature; elementary once seen) |
| μ-law/YJ expm1 fixes (A2) | straightforward engineering |
| Cancellation no-go + EFT framing (P3, X2) | known method (EFT literature); packaging is engineering |
| Adaptive quantizer selection (X3, C1) | straightforward engineering combination; **potential benchmark-paper material** in a Rust scientific stack |
| SR/dither kernels (X4, C3) | known method; deterministic-SR-in-Rust is an implementation contribution |
| VST pipeline (C4) | known method (carryover) |
| MX/SLI software types (C5) | known formats; Rust software implementation contribution |
| Quaternion estimator crossover (X5, C6) | known domain (Markley); minor empirical note |
| Any "potentially novel theorem" found | **none** |
| Any "potentially novel algorithm" found | none at algorithm level; C1's *tool* is the closest, at the engineering/benchmark tier |

## 10. SciRust integration roadmap (deliverable 9)

Do **not** create seven new crates; the evidence supports four bounded work items in existing
crates plus at most one new one:

1. `scirust-core` / `scirust-special`: **stable transform pairs** (C2) — log1p/asinh/x^λ/μ-law/
   logit/Anscombe with expm1/log1p kernels, documented κ_rt per domain, property tests asserting
   the X1 bounds; explicit "not valid beyond |x|≈19" guards for saturating maps. Deterministic:
   uses only elementary functions already under the repo's determinism policy.
2. `scirust-simd` / `scirust-stats`: **EFT reductions** (TwoSum/Neumaier/pairwise, X2-validated)
   and **deterministic stochastic rounding** (counter-based RNG, e.g. Philox, so results are
   reproducible bit-for-bit given the seed) (C3).
3. `scirust-signal` / `scirust-vision`: **VST + exact unbiased inverse** feeding the existing
   denoisers (C4), with the [TSA E9b] kill criterion pre-registered.
4. New crate only if C1 proceeds: `scirust-transform-search` — the representation autotuner
   (dictionary + dev/test protocol + report artifact). Depends on items 1–2 as its dictionary.
5. Deferred indefinitely: `scirust-hypercomplex-transforms` (no evidence of need — X5's outcome
   is a doc note in the existing quaternion module); special-function transforms (killed).

## 11. Final decision (deliverable 10)

| Option | Decision |
|---|---|
| no further work | wrong — measurable engineering value exists (X1–X4) |
| **an engineering module** | **yes — items 1–3 of §10** |
| **a benchmark paper** | **yes, optionally — C1 + the Stage-2/3 protocol, positioned as benchmark/tooling, not theory** |
| a survey paper | optional; merge with [TSA]'s survey recommendation (one survey, not two) |
| a genuine research program | **no** — Lemma 1 vacuity for the general claim; every restricted branch is owned; zero candidate theorems survived |

On terminology (mission §1): the acronym ATRA adds no content over existing names. Where a name
is needed, use the field-standard ones per context — *nonlinear transform coding* (data
compression), *variance stabilization* (statistics), *precision autotuning /
representation-aware kernels* (numerics). Recommended internal name for items 1–4:
**representation-aware numerical kernels**.

---

## Appendix A — Experiments summary

Scripts: `docs/research/atra_experiments/atra_experiments.py` (X1–X4),
`docs/research/atra_experiments/atra_quaternion.py` (X5). Python 3.11.15, Linux x86-64,
binary64; seeds fixed in-source; multi-seed dispersion reported for X3 (3 seeds), X4 (5 seeds),
X5 (20 trials).

| Exp | Question | Outcome |
|---|---|---|
| X1 | Stage-1 properties of Branch-A transforms | table §4.A; A1–A4 findings; P4 validated |
| X2 | can representation beat compensation for sums? | **no** — Neumaier dominates; log-domain positive-only, 3.5× slower, 100× less accurate here |
| X3 | is adaptive representation selection worth it for quantization? | **yes** (+10 dB over direct; within 0.22 dB of Lloyd–Max) and fully classical |
| X4 | do stochastic encoders beat RN at low precision? | **yes** (20.16±0.56 vs stagnation at 0.25/20) and fully known (SR) |
| X5 | do nonlinear quaternion charts beat linear estimators? | **no at high noise** (chordal 6.7° vs Karcher 10.0°, componentwise 23°) |

## Appendix B — References added in this phase (verified ✓ where linked)

✓ Ballé et al., "Nonlinear transform coding," *IEEE JSTSP* 15(2):339–353, 2021 (and GDN:
Ballé–Laparra–Simoncelli, ICLR 2016/2017) — via [survey listings](https://www.sciencedirect.com/science/article/abs/pii/S1051200425008139).
✓ Croci, Fasi, Higham, Mary, Mikaitis, "Stochastic rounding: implementation, error analysis and
applications," [*R. Soc. Open Sci.* 9:211631 (2022)](https://royalsocietypublishing.org/rsos/article/9/3/211631/96937/Stochastic-rounding-implementation-error-analysis).
✓ OCP Microscaling Formats (MX) Specification v1.0, Sept 2023 — MXFP8/6/4, MXINT8, 32-element
blocks with shared E8M0 scale — via [format overview](https://fprox.substack.com/p/ocp-mx-scaling-formats).
✓ Kraska, Beutel, Chi, Dean, Polyzotis, "The case for learned index structures,"
[SIGMOD 2018, pp. 489–504](https://www.cl.cam.ac.uk/~ey204/teaching/ACS/R244_2018_2019/papers/Kraska_SIGMOD_2018.pdf).
✓ Jégou, Douze, Schmid, "Product quantization for nearest neighbor search,"
[*IEEE TPAMI* 33(1):117–128, 2011](https://inria.hal.science/inria-00514462).
✓ Rubio-González et al., "Precimonious: tuning assistant for floating-point precision,"
[SC 2013](https://web.cs.ucdavis.edu/~rubio/includes/sc13.pdf).
✓ Darvish Rouhani et al. (MSFP), NeurIPS 2020; Micikevicius et al. (mixed precision), ICLR 2018.
✓ Markley, Cheng, Crassidis, Oshman, "Averaging quaternions," [*JGCD* 30(4):1193–1197,
2007](https://arc.aiaa.org/doi/10.2514/1.28949).
Standard references: Goyal 2001; Huang–Schultheiss 1963; Coifman–Wickerhauser 1992;
Chen–Donoho–Saunders 1998; Donoho–Johnstone 1994; Blu–Luisier 2007; Olshausen–Field 1996;
Aharon–Elad–Bruckstein 2006; Gregor–LeCun 2010; Venkatakrishnan et al. 2013;
Romano–Elad–Milanfar 2017; Barrau–Bonnabel 2017; Johnson–Lindenstrauss 1984;
Alon–Matias–Szegedy 1996; Charikar 2002; Candès–Romberg–Tao 2006; Donoho 2006;
Boufounos–Baraniuk 2008; Dekker 1971; Ogita–Rump–Oishi 2005; Shewchuk 1997; Neumaier 1974;
Kahan 1965; Roberts 1962; Gray–Stockham 1993; Daubechies–DeVore 2003; Shoham–Gersho 1988;
Yeo–Johnson 2000; Chiang et al. 2017; Darulova–Kuncak 2014/2018; Wang et al. 2019; Dong et al.
2019; Gupta et al. 2015; Durkan et al. 2019; Lloyd 1957/1982; Max 1960; Gersho 1979;
Gray–Neuhoff 1998; plus all of [TSA Appendix B].

## Appendix C — Search-term log for claimed gaps (mission §10 requirement)

| Claimed gap / verification target | Query used | Closest prior work found |
|---|---|---|
| framework name unclaimed | `"transformed scalar algorithms" OR "transformed scalar arithmetic"` (prev. phase) | none (name unclaimed; content owned) |
| learned E/F/D with joint objective | `Ballé "nonlinear transform coding" IEEE JSTSP 2021 GDN` | Ballé et al. 2021 — owns the territory |
| stochastic encoders in numerics | `Croci Fasi Higham Mary Mikaitis stochastic rounding survey 2022` | Croci et al. 2022 — owns it |
| adaptive block-scaled formats in hardware | `OCP microscaling formats MX MXFP8 MXFP4 shared block exponent 2023` | OCP MX v1.0 — in production silicon |
| learned monotone map before search | `Kraska learned index structures SIGMOD 2018 recursive model index` | RMI — owns it |
| operator redesigned for quantized codes | `Jégou product quantization TPAMI 2011 asymmetric distance` | PQ/ADC — owns it |
| per-variable representation autotuning | `Precimonious floating-point precision SC 2013` | Precimonious/FPTuner/Daisy — format-level; transform-dictionary level at kernel granularity not found → C1 gap (engineering) |
| quaternion chart estimator comparison | `Markley averaging quaternions JGCD 2007` | Markley et al. — owns it |
| Box–Cox numerical-boundary analysis | (this phase, X1 + P4; prior phase searches on inverse-Γ/level-index) | not found stated → A1 classified implementation contribution |
| level-index / LNS / inverse-Γ / Ben-Tal / Litvinov / warped GP / Mäkitalo–Foi | see [TSA Appendix B] (verified in phase 1) | all owned |
