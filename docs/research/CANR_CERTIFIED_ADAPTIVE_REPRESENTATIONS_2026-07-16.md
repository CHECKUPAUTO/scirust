# Certified Adaptive Numerical Representations (CANR) — Scientific Evaluation

**Date:** 2026-07-16
**Status:** Research investigation (phase 3). Design + evidence; implementation deferred per mission §7/§16.
**Companion documents:** `TSA_TRANSFORMED_SCALAR_ALGORITHMS_2026-07-16.md` **[TSA]** and
`ATRA_ADAPTIVE_TRANSFORMED_REPRESENTATIONS_2026-07-16.md` **[ATRA]** in this directory. Their
conclusions are presupposed, not revisited (mission §1).
**Method:** hypotheses H1–H5 tested independently; every claimed certificate derived (§3) and
validated empirically (experiments Y1–Y6, `docs/research/canr_experiments/canr_experiments.py`,
pure stdlib, **Decimal prec-60 multiprecision reference**, fixed seeds, dev/held-out splits);
primary sources verified online; cumulative search log in Appendix C.

---

## 0. Verdict

**Certified adaptive representation selection works as a systems capability, at exactly the
scale the mission proposes and no larger.** The evidence of this phase:

- **H1 (conditioning certificates): CONFIRMED, and the formula is standard.**
  κ_rt(x) = |φ(x)/(x·φ′(x))| is precisely the relative condition number of the *decoder* φ⁻¹
  evaluated at y = φ(x) (equivalently 1/cond(φ)(x)) — classical inverse-function conditioning
  (proof in §3.1; the honest classification is *known theorem, new packaging*). What was
  validated empirically: the per-point certificate `RT_ulps ≤ 8·(κ_rt+1)` held at **123/123
  valid grid points with zero violations** across five Box–Cox exponents over 24 decades of x
  (Y1), and the invalid-region rule `κ_rt·u ≥ ½ ⇒ no recoverable digits` predicted **exactly**
  the two grid points where the shifted representation is literally non-decodable (`log1p(−1)`
  domain error). Certificates of this kind are machine-checkable and can gate a selector.
- **H2 (selection beats static): CONFIRMED in prototype.** A certificate-driven selector using
  the classical summation condition number C_sum = Σ|xᵢ|/|Σxᵢ| chose {naive, pairwise,
  superaccumulator} on three workload families and **passed held-out validation on all three**
  while using 1 flop/element where 40 were unnecessary (Y6). Caveat discovered: analytic-only
  selection is sound but conservative (it picked the superaccumulator where Neumaier would have
  sufficed empirically) — the hybrid analytic+empirical mode of mission §4 is the right design.
- **H3 (joint representation+operator): CONFIRMED trivially and decisively.** Log
  representation with the wrong operator (exp-then-sum) returns 0.0 where log+LSE returns the
  correct −787.324 (Y3); phase-2 evidence (VST+matched denoiser, PQ+ADC) already showed the
  same. Representations must be selected *as pairs with operators*; ℱ_r in the instance tuple
  is load-bearing.
- **H4 (reproducibility as an objective): CONFIRMED with an important distinction.** Across 10
  random permutations of 20 000 wide-range terms: naive summation produced 10 distinct results,
  pairwise 9, Kahan 5, Neumaier/Klein/superaccumulator 1 each (Y4). But only the
  superaccumulator (and the binned methods of Ahrens–Demmel–Nguyen) carry a **proof** of
  order-invariance; Neumaier/Klein's invariance here is a property of this dataset, not a
  guarantee, and must never be advertised as one. A determinism ladder (§6.2) makes the
  distinction precise.
- **H5 (application-specific hypercomplex): NOT ADVANCED this phase.** Phase-2 X5 stands
  (chordal linear mean beats the log-chart mean at high noise); the benchmark protocol (§9)
  carries the Stage-5 tasks; no new claim is made.
- **Box–Cox/μ-law independent verification (mission §9): done against a true multiprecision
  reference, with one correction to phase 2.** The Box–Cox *forward* computation via plain
  `pow` is already excellent (≤ 1.6 ulps — better than the expm1 route's ≤ 14); the pathology
  is 100 % representational (the stored −1/λ shift), reaching 5.6×10¹⁴ ulps round-trip at
  λ = 1.5 and total decode failure below x ≈ ε^(1/λ). Unshifted x^λ storage round-trips at
  ≤ 13 ulps everywhere (κ_rt ≡ 1/λ, flat). Downstream (as §9 demanded): identical at standard
  scale, 67× better unshifted at tiny scale (1.97×10⁻¹⁴ vs 2.96×10⁻¹⁶). μ-law: naive inverse
  up to 1.85×10¹² ulps on [10⁻¹⁵, 1); `expm1` inverse: 2 ulps. Phase-2 findings confirmed,
  sharpened, and in one respect corrected.

**Recommendation (deliverable 12): engineering module + benchmark tool.** Novelty ceiling
reached: level 6 of the mission's ladder (a new autotuning tool niche); no level-7/8 item
survived scrutiny. Details in §12.

---

## 1. Restricted formal framework (deliverable 1)

An admissible CANR instance is ℐ = (ℛ, ℱ, S, J, C, P) with all six elements explicit
(mission §3). The worked instance used throughout this report:

```
ℛ  = { f64-direct, f64-pairwise(fixed tree), f64-Neumaier, f64-Klein,
       integer superaccumulator, log-domain f64 }
ℱ  = { sum, positive product, log-likelihood accumulation }   (ℱ_r per representation:
       log-domain admits product-as-sum and sum-as-LSE only)
S  = certificate prefilter (C_sum bound per method) → cheapest passing candidate
     → held-out empirical validation
J  = flops/element (hardware-portable cost proxy), ties broken by memory
C  = relative error ≤ τ (declared per workload); determinism level ≥ declared level
P  = dev set for selection; 3 held-out seeds for validation; exact reference via
     rational arithmetic; machine-readable result rows
```

**Error decomposition** (mission §6). For a pipeline x̂ = D(F(E(x))) the total error separates,
to first order, into five attributable terms:

```
e_total ≈ e_repr   (encode rounding: ≈ B_enc·u, amplified downstream by cond(F))
        + e_op     (operator's own forward error in the encoded domain)
        + e_dec    (decode rounding ≈ B_dec·u, plus κ_rt·(upstream error) amplification)
        + e_model  (statistical mismatch: e.g., VST bias, estimator bias for lossy E)
        + e_sel    (selection error: choosing on dev data that misrepresents deployment)
```

Certificates in §3 bound the first three; P bounds e_sel (held-out validation, CIs); e_model
is bounded only by statistical assumptions that must be declared (e.g., exact unbiased inverse
for Anscombe under Poisson). This decomposition is the report's accounting discipline.

## 2. Literature review — primary sources (deliverable 2)

Verified this phase (✓ = checked online today; full list with links in Appendix B):

| Area | Primary source | Role here |
|---|---|---|
| Compensated summation | Kahan 1965; Neumaier 1974; **Klein, *Computing* 76:279–293, 2006 ✓** | ℛ members; certificates §3.5 |
| Accurate/exact summation | Ogita–Rump–Oishi, *SIAM J. Sci. Comput.* 2005; Rump et al. 2008; Shewchuk 1997; Kulisch accumulators | expansions/superaccumulator semantics |
| Reproducible summation | **Ahrens–Demmel–Nguyen, *ACM TOMS* 46(3):22, 2020 ✓** (binned 6-word accumulator); ReproBLAS (Demmel–Nguyen 2013–15) | the *proved* D1 rung of the determinism ladder |
| Verified FP error bounds | **FPTaylor: Solovyev et al., *TOPLAS* 2018 ✓** (HOL Light certificates); Daisy (Darulova et al.); Rosa | machine-checkable certificate precedent |
| Certified elementary functions | **Gappa: de Dinechin–Lauter–Melquiond, *IEEE TC* 60(2):242–253, 2011 ✓**; MetaLibm (Kupriianova–Lauter 2014); CORE-MATH; RLIBM | libm ulp budgets B_enc/B_dec must come from here |
| Precision autotuning | Precimonious (SC 2013); FPTuner (POPL 2017); Herbie (PLDI 2015) [all verified phase 2] | closest prior tools to §8 |
| Performance autotuning | **FFTW3: Frigo–Johnson, *Proc. IEEE* 93(2):216–231, 2005 ✓**; ATLAS: Whaley–Petitet–Dongarra, *Parallel Comput.* 27:3–35, 2001 ✓ | planner architecture precedent |
| Algorithm configuration | SMAC (Hutter et al., LION 2011); irace (López-Ibáñez et al. 2016) | S when the space is large |
| Stochastic rounding | Croci et al., *R. Soc. Open Sci.* 2022 [verified phase 2] | D2 rung; §3.6 |
| Summation error theory | Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed. 2002, ch. 4 | the C_sum certificates |
| Transform/statistics prior art | [TSA §4] (24 rows), [ATRA §2] (3 route tables) | not repeated |

Closest prior system per contribution: transform-pair certificates → Gappa/MetaLibm (per
function, not per *pair* with κ_rt API); representation×operator selection under an accuracy
constraint → Precimonious (types only, no operators) and FFTW (speed only, no accuracy
constraint). The *combination* — accuracy-constrained, certificate-gated, reproducibility-aware
selection over (representation, operator) pairs — appears unoccupied as a tool niche (level 6).

## 3. Certificates: derivations and validation (deliverable 3)

**3.1 κ_rt is standard inverse-function conditioning (H1).** For strictly monotone C¹ φ with
y = φ(x): cond_rel(φ)(x) = |x·φ′(x)/φ(x)| and cond_rel(φ⁻¹)(y) = |y·(φ⁻¹)′(y)/φ⁻¹(y)| =
|φ(x)/(x·φ′(x))| = κ_rt(x), using (φ⁻¹)′(y) = 1/φ′(x). Hence κ_rt(x) = cond(φ⁻¹)(φ(x)) =
1/cond(φ)(x). ∎  — The formula is textbook conditioning; no novelty is claimed for it.

**3.2 Round-trip certificate (error model made explicit).** Assume encode returns
ŷ = φ(x)(1+δ₁), |δ₁| ≤ B_enc·u (faithful libm with documented ulp budget), decode returns
φ⁻¹(ŷ)(1+δ₂), |δ₂| ≤ B_dec·u, input x exact. First-order expansion:

```
|x̂−x|/|x| ≤ ( κ_rt(x)·B_enc + B_dec )·u + O(u²).
```

Relation to forward/backward error: decode is *backward stable in y* by construction (it
returns the exact inverse of a nearby ŷ); the forward error in x is governed by κ_rt. Validity
requires (i) monotone single branch, (ii) sub-exponential second-order terms — checked per
family, and (iii) *documented* B_enc/B_dec, which for Rust means either SciRust's own
deterministic kernels or Gappa/CORE-MATH-audited libm. **Validation (Y1):** with budget 8
(covering pow/log/exp at ≤ few ulps each), all 123 valid grid points satisfied
RT_ulps ≤ 8·(κ_rt+1); zero violations.

**3.3 Invalid-region rule.** κ_rt(x)·u ≥ ½ ⇒ the stored y carries < 1 significant bit about
x ⇒ decode may fail entirely. **Validation (Y1, λ=1.5):** the two grid points with
κ_rt·u ≥ ½ were exactly the two points where decode raised `log1p(−1)` domain errors
(x^λ underflowed against the −1/λ shift). A selector must treat κ_rt·u ≥ ½ as a hard domain
exclusion, and κ_rt·u ≥ τ as the soft constraint.

**3.4 Interval certification of a whole domain (H1).** For every §8 transform, κ_rt is
piecewise monotone in x with closed-form derivative, so sup over an interval is attained at
endpoints or at explicitly computable critical points (e.g., Box–Cox: κ_rt = |x^λ−1|/(λx^λ)
is strictly decreasing on (0,1) and increasing towards |ln x|-like growth for x>1 as λ→0). A
`kappa_rt_sup(interval)` API is therefore implementable in exact arithmetic per transform. For
*approximate* transforms (rational-quadratic splines), κ_rt bounds follow from per-segment
rational bounds on φ, φ′ plus an additive approximation-error term δ: the certificate stays
valid with B_enc·u → B_enc·u + δ; still useful iff δ is certified per segment.

**3.5 Summation certificates (Higham 2002, ch. 4).** With C_sum = Σ|xᵢ|/|Σxᵢ| (the condition
number of summation): naive ≤ (n−1)u·C_sum + O(u²); fixed-tree pairwise ≤ ⌈log₂n⌉u·C_sum·(1+o(1));
Kahan/Neumaier ≤ (2u + O(nu²))·C_sum with Neumaier robust to |xᵢ| > |sᵢ| (phase-2 X2a
counterexample); Klein adds a second-order compensation (Klein 2006); the integer
superaccumulator is exact up to a truncation ≤ n·2^{E₀} with E₀ = E_max − guard_bits (proof:
each addend's discarded tail < 2^{E₀}; integer addition exact). **Validation (Y5/Y6):** all
observed errors within bounds; the selector's PASS/FAIL on held-out data matched certificates 3/3.

**3.6 Stochastic rounding.** E[SR(x)] = x; single-op variance ≤ (grid step)²/4; Σ error
O(√n·u) w.h.p.; stagnation immunity — all from Croci et al. 2022 (not re-proved; phase-2 X4
instantiates). Determinism: given (seed, order, counter-based RNG), SR is bitwise reproducible
(D2 in the ladder below).

## 4. Taxonomy: representations × operators (deliverable 4)

| Representation r | ℱ_r (compatible ops) | Certificate available | Determinism | Notes |
|---|---|---|---|---|
| f64/f32 direct | all | Higham bounds | D0 | baseline |
| fixed-tree pairwise | reductions | ⌈log₂n⌉u·C_sum | D0 (per declared tree/width) | SIMD-native |
| Kahan/Neumaier/Klein | reductions | 2u·C_sum (+2nd order) | D0 | Neumaier default choice |
| FP expansions (TwoSum cascades) | reductions, dot | exact per EFT | D0 | branchy; k-term memory |
| integer superaccumulator | reductions, dot | exact − n·2^{E₀} | **D1 (proved)** | 40+ flops/elt; big state |
| binned (Ahrens–Demmel–Nguyen) | reductions, BLAS-1 | proved reproducible + error bound | **D1 (proved)** | 6-word accumulator; the production answer |
| log-domain f64 | product→sum, sum→LSE only | κ_rt = 1 for ×; LSE bounds (Blanchard–Higham–Higham) | D0 | positive data only |
| unshifted power x^λ | quasi-arithmetic ops | κ_rt ≡ 1/λ (flat) | D0 | replaces shifted Box–Cox storage |
| μ-law/A-law (expm1 kernels) | quantize, compand | κ_rt bounded, ≤2 ulps RT (Y2) | D0 | audio/8-bit |
| Anscombe/GAT | Gaussian-denoiser ops + unbiased inverse | delta-method + Mäkitalo–Foi inverse | D0 | e_model must be declared |
| block scaling (MX-style) | GEMM/conv accumulate | per-block exact scale; element quantization bound | D0 per blocking | hardware-aligned |
| stochastic rounding (seeded) | low-precision accumulate | unbiased; √n·u w.h.p. | D2 | counter-based RNG required |
| quaternion charts | means/filters/slerp | chart-dependent (X5) | D0 | chordal default per X5 |

## 5. Box–Cox and μ-law: independent verification (deliverable 5) — Experiment Y1/Y2

Reference: Decimal prec-60 (exp/ln), grid λ ∈ {0.1, 0.25, 0.5, 0.75, 1.5} × x ∈ 25 log-spaced
points in [10⁻¹², 10¹²]:

| λ | fwd ULP (pow) | fwd ULP (expm1) | RT ULP shifted | RT ULP x^λ | max κ_rt | certificate |
|---|---:|---:|---:|---:|---:|---|
| 0.1 | 1.6 | 2.4 | 75 | 13 | 148 | 25/25, 0 viol |
| 0.25 | 0.6 | 3.0 | 299 | 2 | 4×10³ | 25/25, 0 viol |
| 0.5 | 0.5 | 5.0 | 2.8×10⁵ | 1 | 2×10⁶ | 25/25, 0 viol |
| 0.75 | 0.7 | 7.0 | 1.9×10⁸ | 13 | 1.3×10⁹ | 25/25, 0 viol |
| 1.5 | 0.5 | 14.0 | 5.6×10¹⁴ | 13 | 6.7×10¹⁷ | 23/25 valid, 0 viol; **2 predicted-invalid = 2 observed total losses** |

Findings, including one correction to [ATRA]:

1. **Forward Box–Cox via plain `pow` is already accurate (≤ 1.6 ulps)** — in fact *more*
   accurate than the `expm1(λ log x)/λ` route (≤ 14 ulps, from the rounding of the λ·log x
   argument). [ATRA A2] over-generalized the expm1 rewrite's benefit; for Box–Cox forward it is
   unnecessary (and mildly harmful). The rewrite matters where the *stored* value is near 0 and
   the naive formula subtracts 1 from a `pow` result — i.e., for μ-law's inverse, confirmed
   below.
2. **The round-trip pathology is purely representational** — identical for both forward
   implementations, growing exactly like κ_rt, and ending in outright non-decodability
   (log1p(−1)) below x ≈ ε^{1/λ}. The certificate machinery of §3 captures all of it, including
   the failure points.
3. **Unshifted x^λ storage: κ_rt ≡ 1/λ, flat; ≤ 13 ulps round-trip everywhere.**
4. **Downstream test (as §9 required before adopting a storage policy):** Box–Cox
   quasi-arithmetic mean, λ = 0.25, n = 4096, 3 seeds, Decimal reference: standard-scale data —
   shifted and unshifted identical (2.35×10⁻¹⁶); tiny-scale data (×10⁻⁸) — shifted 1.97×10⁻¹⁴
   vs unshifted 2.96×10⁻¹⁶ (67× better). **Policy: store x^λ (or log x), apply the −1/λ shift
   symbolically at aggregation boundaries; the statistical parameterization (needed for
   λ-estimation continuity and Jacobian normalization à la Box–Cox 1964) is unaffected because
   the shift is affine and can be applied exactly on the aggregate.**
5. **μ-law (Y2), against the Decimal reference:** encode 1.4 ulps; naive `pow(1+μ,y)−1` inverse
   up to **1.85×10¹² ulps** (worst at |x| = 10⁻¹⁵); `expm1` inverse **2 ulps**. Phase-2 finding
   confirmed and sharpened with a true reference.

## 6. Robust reduction design study (deliverable 6) — Experiments Y4/Y5

### 6.1 Method matrix

| method | error certificate (×C_sum) | flops/elt | memory | SIMD | parallel | determinism |
|---|---|---:|---|---|---|---|
| naive sequential | (n−1)u | 1 | O(1) | reassociates ⇒ width-dependent | linear scan | D0 |
| fixed-tree pairwise | ⌈log₂n⌉u | 1 | O(log n) stack | excellent (tree = declared blocking) | excellent | D0 per declared tree |
| Kahan | 2u + O(nu²), fails on |xᵢ|>|sᵢ| | 4 | O(1) | per-lane + compensated combine | good | D0 |
| Neumaier | 2u + O(nu²), robust | 7 (branchless select) | O(1) | per-lane + combine | good | D0 |
| Klein | 2nd-order compensation (Klein 2006) | ~14 | O(1) | as Neumaier | good | D0 |
| TwoSum/FastTwoSum EFT | exact error term per op | 6/3 | O(1) | good | good | D0 |
| expansions (Shewchuk) | exact, adaptive length | O(k) | O(k) | poor (branchy) | fair | D0 |
| superaccumulator (int) | exact − n·2^{E₀} | ~40 | ~2 words–KiB | fair (integer lanes) | reduction of accumulators | **D1 proved** |
| binned (ADN 2020) | proved bound, order-invariant | ~7–12 | 6 words | good | designed for it | **D1 proved** |
| exact dot (Ogita–Rump–Oishi) | exact/faithful | ~10–25 | O(1) | good | good | D0 |
| SR / deterministic seeded SR | unbiased, √n·u w.h.p. | 2 + RNG | O(1) | counter-based RNG vectorizes | good | D2 |

**Determinism ladder** (H4): **D0** bitwise for a fixed evaluation order/blocking; **D1**
bitwise for *any* order (proof required — only superaccumulator/binned qualify); **D2** bitwise
given (seed, order) for stochastic methods with counter-based RNG; **D3** statistical only.
SIMD consequence for SciRust: a vectorized naive/pairwise sum changes results across SIMD
widths unless the tree/blocking is part of the declared semantics — so `scirust-simd` should
either pin the blocking (D0-per-width, documented) or offer a D1 binned mode for cross-
architecture claims.

### 6.2 Measurements

Order-invariance across 10 random permutations, 20 000 mixed-sign wide-range terms (Y4):
naive **10 distinct results** (max rel err 1.8×10⁻¹⁴), pairwise 9 (1.1×10⁻¹⁵), Kahan 5
(4.6×10⁻¹⁶), Neumaier 1 (exact here), Klein 1 (exact here), superaccumulator 1 (exact,
**by proof**). Caution recorded: Neumaier/Klein unanimity is an empirical property of this
dataset — compensated summation is *not* reproducible-by-construction and must not be
documented as D1.

Accuracy/cost (Y5): wide-range positive n=10⁵ — everything except naive is exact to the
rational reference; signed cancellation n=6×10⁴ — naive 1.2×10¹ (unusable), pairwise 2.1,
Kahan 7.2×10⁻¹, Neumaier 2.2×10⁻¹⁴, Klein exact, superaccumulator exact. Python timings are
indicative only; the flops/element column is the portable cost model.

**Crate placement (mission §10):** default `sum`/`dot` in `scirust-core` = Neumaier (7
flops/elt, robust, O(1) state); SIMD fixed-tree pairwise + vectorized Neumaier in
`scirust-simd`; D1 binned/superaccumulator behind a `repro` feature (used by anything claiming
cross-architecture determinism); LSE/likelihood kernels in `scirust-stats`; **no new crate for
reductions**.

## 7. Transform-pair API assessment (deliverable 7)

The proposed `CertifiedTransform<T>` trait is a good seed but not mathematically honest as
written:

1. **Static methods prevent parameterized transforms** (Box–Cox λ, μ-law μ, spline knots).
   Methods must take `&self`; the transform is a *value*, context lives in the constructor.
2. **`T` is the wrong type for bounds.** Error bounds are dimensionless ulp/relative budgets;
   returning them in `T` conflates value space with error space. Use a dedicated
   `Bound { ulps: f64, valid: Interval }`.
3. **`roundtrip_condition(x) -> Option<T>`** is ambiguous (None = outside domain? unbounded?).
   Replace with total functions over a declared `domain() -> Interval` plus
   `kappa_rt_sup(iv: Interval) -> f64` (§3.4 makes this implementable exactly).
4. **Honesty precondition:** every bound is conditional on B_enc/B_dec libm budgets. Rust's
   `std` libm accuracy is unspecified and platform-dependent; the certificates are honest only
   over SciRust's own deterministic kernels (already present in `scirust-simd`) or an audited
   libm (CORE-MATH/Gappa route). This must be a documented contract of the trait.
5. **One trait cannot carry the whole taxonomy.** Separate abstractions required:
   - `CertifiedMonotone` — bijective single-branch pairs (the table in §4, rows 1–10);
   - `BranchedTransform` — decode takes a branch witness (e.g., y ↦ x² inversion); the branch
     must be an explicit value, never an undocumented convention (mission §13 kill criterion);
   - `LossyEncoder`/`Estimator` — many-to-one E with estimator decode carrying bias/variance
     certificates instead of inverse-error bounds (quantizers, rank/sign, sketches);
   - `StochasticEncoder` — encode takes an explicit counter/key (Philox-style) so D2 holds;
   - `BlockTransform` — slice-level encode with shared parameters (block scaling/MX), where
     per-element certificates depend on the block statistic.

Sketch (design only, per mission):

```rust
pub struct Interval { pub lo: f64, pub hi: f64 }
pub struct Bound { pub ulps: f64 }               // valid under the crate's libm budget contract

pub trait CertifiedMonotone {
    type Enc;
    fn domain(&self) -> Interval;
    fn encode(&self, x: f64) -> Option<Self::Enc>;      // None outside domain
    fn decode(&self, y: Self::Enc) -> f64;
    fn kappa_rt(&self, x: f64) -> f64;                  // = cond(decode) at encode(x)
    fn kappa_rt_sup(&self, iv: Interval) -> f64;        // exact endpoint/critical-point eval
    fn roundtrip_bound(&self, iv: Interval) -> Bound;   // (kappa_rt_sup·B_enc + B_dec) ulps
}
```

## 8. Representation-autotuner architecture (deliverable 8) — `scirust-transform-search`

Pipeline (validated in miniature by Y6):

```
inputs: kernel, scalar type, sample/distribution, tau, hardware descr., runtime budget,
        determinism level, candidate (r, f) pairs from the section-4 table
S1 certificate gate : drop (r,f) whose analytic bound or domain/kappa_rt check fails tau
                      or whose determinism level < required          [sound, cheap]
S2 cost-model rank  : flops/elt + memory traffic per candidate        [portable proxy]
S3 dev measurement  : successive halving on measured J over dev data  [handles S1 conservatism]
S4 held-out check   : fresh seeds, CI on achieved error & runtime; reject on any violation
S5 report artifact  : chosen pair, parameters, certificates, measured metrics with CIs,
                      rejected candidates WITH reasons, selection cost vs. projected savings
```

Evidence: the Y6 prototype (S1+S2 only) selected {naive | pairwise | superaccumulator} for
{benign | wide-range | cancellation} workloads under τ = {10⁻⁹, 10⁻¹³, 10⁻⁸} and passed all
held-out checks; its one inefficiency — choosing the superaccumulator (40 flops/elt) where
Neumaier (7) empirically sufficed — is precisely what S3 exists to repair. Overfitting guards:
S4 is mandatory; selection-cost accounting: report `t_search / (t_static − t_selected)`
break-even counts. Search strategy: exhaustive for |candidates| ≤ ~50 (the §4 table); SMAC/
irace-style configuration only if the space grows parameters (spline knots, block sizes).
Closest prior systems and the delta: FFTW planner (speed-only), Precimonious (types-only),
Herbie (expressions-only) — none selects (representation, operator) pairs under an accuracy
constraint with certificates and a determinism level. Novelty level 6; nothing higher.

## 9. Canonical benchmark protocol (deliverable 9)

Suite (mission §12) — every entry: 3+ seeds, dev/held-out split, CIs, exact or Decimal
reference where feasible, JSON rows `{kernel, dataset, method, seed, metric, value, ci, cert}`:

| group | kernels | primary metric | strong baselines |
|---|---|---|---|
| scalar | long products, signed sums (X2/Y5 sets), dot, norms, softmax/LSE, likelihood, polynomial eval (Horner vs compensated Horner) | rel err vs exact; flops | Neumaier, pairwise, fsum, LSE |
| signal | Gaussian/Poisson/impulsive/multiplicative denoising, multichannel correlation | MSE/SSIM vs ground truth | direct-domain denoisers, [TSA E9b] protocol |
| optimization | ill-conditioned LS, logistic regression, low-precision GD, mixed-precision iterative refinement | final loss; iterations | f64 static, Jacobi-preconditioned |
| search/ML | NN distance, cosine, PQ+ADC, attention accumulation | recall@k; ulp drift | f32 direct, PQ reference |
| hypercomplex | quaternion averaging (X5), orientation filtering, slerp, IMU fusion, multichannel denoise | angular error | chordal mean, per-channel real processing |

Negative results are retained in-tree (this file and its predecessors are the template).

## 10. Kill criteria and negative results (deliverable 10)

Fired to date (each with its §13 clause): shifted Box–Cox storage on domains reaching
|x^λ| ≪ 1 — *inverse too ill-conditioned* + *undocumented branch/domain failure* (Y1);
naive μ-law inverse — *dominated on accuracy at equal cost* (Y2); log-domain for signed sums —
*mathematically unable to meet tolerance* ([ATRA P3], Y3 shows the operator, not the
representation, is decisive); naive-threshold VST — *gain disappears against strong baseline*
([TSA E9b]); componentwise quaternion mean at high noise — *dominated* ([ATRA X5]);
compensated-summation-as-reproducible — *breaks required determinism claim* (Y4 caution);
analytic-only selection — flagged *selection cost/conservatism*, mitigated by S3/S4 (Y6).
Standing kill criteria for future candidates: as mission §13, adopted verbatim into the
benchmark protocol.

## 11. SciRust integration plan (deliverable 11)

Phase A (engineering, no research risk): `scirust-core::reduce` — Neumaier default, Klein
option, `fsum`-equivalent exact mode; `scirust-simd` — fixed-tree pairwise + vectorized
Neumaier with declared blocking (D0), binned D1 mode behind `repro` feature (Ahrens et al.
design); `scirust-core::transforms` — `CertifiedMonotone` impls for log/log1p/asinh/x^λ
(unshifted)/μ-law-expm1/logit/Anscombe with κ_rt docs and property tests asserting the Y1/Y2
bounds; `scirust-stats` — LSE, log-likelihood accumulators, VST + exact unbiased inverse.
Phase B (tool): `scirust-transform-search` per §8, seeded by the §4 table. Phase C (bench):
the §9 suite as a benchmark crate/CI job publishing JSON artifacts. Determinism contract
throughout: budgets B pinned to SciRust's own deterministic transcendentals; det-SR via
counter-based RNG only.

## 12. Final recommendation (deliverable 12) and novelty ladder

**Decision: engineering module (Phase A) + benchmark tool (Phases B–C).** Not "stop" or
"documentation only" — Y1–Y6 demonstrate measurable, certifiable wins with modest effort. Not
"application-specific research" beyond the benchmark's Stage-5 slots (H5 produced no new
claims). Not a "broader research program" — the ceiling of three phases of investigation is
firmly at ladder level 6.

Ladder assignment (mission §14): κ_rt identity — level 1 (known theorem) packaged at level 3;
summation/EFT/reproducible methods — levels 2–3; Box–Cox storage policy + μ-law kernels —
level 4 (with the Y1 downstream measurement as level-5 benchmark methodology); determinism
ladder + certificate-gated selector — level 6 (new tool niche, prior-art-adjacent to FFTW/
Precimonious/Herbie); levels 7–8 — **none** (no primary-source-defensible candidate emerged).

Terminology (mission §0): no new public acronym is warranted; "certified representation-aware
numerical kernels" describes Phase A, and the tool is simply `scirust-transform-search`.

---

## Appendix A — Experiment index

`docs/research/canr_experiments/canr_experiments.py` (Python 3.11.15, ~6 s runtime):
Y1 Box–Cox vs Decimal-60 (grid, certificates, downstream means); Y2 μ-law vs Decimal-60;
Y3 joint representation+operator; Y4 permutation reproducibility; Y5 reduction accuracy/cost;
Y6 certificate-driven selector with held-out validation. All numbers quoted in §§0–10 come
from the committed script's output on 2026-07-16.

## Appendix B — Verified sources (this phase)

- Klein, "A generalized Kahan–Babuška summation algorithm," [*Computing* 76:279–293 (2006)](https://link.springer.com/article/10.1007/s00607-005-0139-x).
- Ahrens, Demmel, Nguyen, "Algorithms for efficient reproducible floating point summation," [*ACM TOMS* 46(3):22 (2020)](https://dl.acm.org/doi/10.1145/3389360); ReproBLAS.
- Solovyev, Baranowski, Briggs, Jacobsen, Rakamarić, Gopalakrishnan, "Rigorous estimation of floating-point round-off errors with Symbolic Taylor Expansions," [*TOPLAS* (2018)](https://dl.acm.org/doi/10.1145/3230733) (FPTaylor, HOL Light certificates).
- de Dinechin, Lauter, Melquiond, "Certifying the floating-point implementation of an elementary function using Gappa," [*IEEE TC* 60(2):242–253 (2011)](https://inria.hal.science/inria-00533968/en); [MetaLibm](https://link.springer.com/chapter/10.1007/978-3-662-44199-2_106).
- Frigo, Johnson, "The design and implementation of FFTW3," [*Proc. IEEE* 93(2):216–231 (2005)](https://math.mit.edu/~stevenj/papers/FrigoJo05.pdf); Whaley, Petitet, Dongarra, ATLAS, *Parallel Comput.* 27:3–35 (2001).
- Phase-1/2 verified sources: see [TSA Appendix B] and [ATRA Appendix B].
- Standard: Higham 2002 (ch. 4); Ogita–Rump–Oishi 2005; Rump et al. 2008; Shewchuk 1997;
  Kulisch; Hutter et al. (SMAC) 2011; López-Ibáñez et al. (irace) 2016; Blanchard–Higham–Higham
  2021; Croci et al. 2022; Mäkitalo–Foi 2011.

## Appendix C — Cumulative search log (3 phases)

Phase-3 additions to the [ATRA Appendix C] log:

| target | query | closest prior found |
|---|---|---|
| Klein summation primary source | `Klein "generalized Kahan-Babuska summation" Computing 2006` | Klein 2006 — confirmed |
| reproducible summation with proofs | `Ahrens Demmel Nguyen reproducible floating point summation TOMS 2020 binned ReproBLAS` | ADN 2020 — owns D1 |
| machine-checkable FP certificates | `FPTaylor Solovyev symbolic Taylor certified bounds` | FPTaylor TOPLAS 2018 — owns verified bounds |
| certified elementary functions | `Gappa de Dinechin Lauter Melquiond certified elementary function MetaLibm` | Gappa 2011 / MetaLibm — own libm budgets |
| autotuning planner precedent | `FFTW planner Frigo Johnson Proceedings IEEE 2005 ATLAS Whaley irace SMAC` | FFTW3 2005, ATLAS 2001 — own speed-only selection; accuracy-constrained (r,f)-pair selection unoccupied → §8 tool niche (level 6) |
