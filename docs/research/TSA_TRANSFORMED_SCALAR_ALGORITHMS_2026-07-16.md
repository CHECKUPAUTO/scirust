# Transformed Scalar Algorithms (TSA) — Scientific Evaluation

**Date:** 2026-07-16
**Status:** Research investigation. No implementation (per mission statement).
**Method:** Falsificationist. Every claim below was either proved, refuted by counterexample,
checked numerically (reproducible scripts in `docs/research/tsa_experiments/`), or traced to
prior literature (references verified online where risk of misattribution existed).

---

## 0. Verdict (read this first)

**The central hypothesis — that conjugating existing algorithms by an invertible scalar
transform, `A_φ = φ⁻¹ ∘ A ∘ φ`, produces a genuinely new algorithmic paradigm — is false as
stated.** The construction is the classical *conjugation / change-of-variables principle*. It is
not merely "similar to" existing work: it **is** existing work, independently mature in at least
fifteen fields under fifteen different names (§4). In exact arithmetic the transformed algorithm
is *isomorphic* to the original — same outputs up to relabeling, same fixed points, same
asymptotic convergence rate (proved in §2.2, checked numerically in Experiment E6), same
complexity class (§11).

What genuinely changes under conjugation, and where all the real value known to science lives,
is exactly three things:

1. **Finite-precision behavior** — dynamic range, cancellation, rounding, operation cost. This
   is the territory of log-domain computation, logarithmic number systems, level-index
   arithmetic, and posits (§10).
2. **Statistical semantics** — when the transform is applied to *data* around a *nonlinear or
   noise-sensitive* estimator, the estimator changes. This is the territory of
   variance-stabilizing transforms, GLM link functions, warped Gaussian processes, and
   homomorphic filtering (§8).
3. **Degenerate limits** — when the transform family loses invertibility in a limit, genuinely
   new algebra appears. This happened exactly once at scale in history and created tropical /
   idempotent mathematics via Maslov dequantization (§2.3). It is the strongest evidence both
   *for* the spirit of the idea and *against* its novelty.

**The Γ / fractional-factorial transform family specifically is a dead end** as a general-purpose
scalar transform: `x ↦ x!` is not injective on (0,1) (counterexample: `0.1! = 0.868463811992!`,
Experiment E2), its inverse is catastrophically ill-conditioned near the minimum
x\* ≈ 0.4616 (relative-error amplification up to ~10⁸ measured, E3/E5), it overflows binary64
for x > 170.62 (E4), its inverse has no closed form and needs Newton iteration seeded by a
Lambert-W asymptotic, and — decisively — Γ possesses **no algebraic intertwining identity**
(§2.5): it does not turn any expensive operation into a cheap one, which is the single property
that made log, Fourier, Legendre and Mellin transforms algorithmically valuable. By Hölder's
theorem (1887) Γ satisfies no algebraic differential equation at all, so no such identity is
waiting to be found.

**Hypercomplex TSA is mathematically vacuous** by transport of structure: the quaternion algebra
over transformed scalars is *isomorphic* to the ordinary quaternion algebra, and zero divisors,
alternativity, Moufang identities, and norm composition are isomorphism invariants — they cannot
change (proved in §7). Applying φ componentwise *without* transporting the operations does not
yield an algebra at all (the only field automorphism of ℝ is the identity). What survives is a
numeric-representation question (e.g., quaternions in a logarithmic number system), which is
engineering, not new geometry.

**Recommendation:** TSA does not merit founding as a new field. It merits:
(a) a **survey/taxonomy paper** unifying the fifteen scattered literatures — genuinely missing
and useful; (b) an **engineering track** in SciRust (log-domain kernels, VST utilities,
correctly-rounded special functions, an experiment in automated transform selection); and
(c) two or three **focused open questions** listed in §18. Section 19 answers the mission's
final question directly.

---

## 1. Problem statement and method

The mission asks whether

```
A_φ(x) = φ⁻¹( A( φ(x) ) ),     more generally   A_T = T⁻¹ ∘ A ∘ T,   T invertible,
```

applied with scalar transforms φ : ℝ → ℝ (especially Γ-family and other special functions)
yields new algorithmic families, new filters, new hypercomplex geometry, new metrics, and new
numerical behavior.

Method used here:

- **Proofs and counterexamples first.** Sections 2 and 7 contain the structural results; they
  settle most of the questionnaire before any experiment is needed.
- **Numerical falsification experiments** E1–E9b (pure-stdlib Python, scripts committed next to
  this report), covering injectivity, conditioning, overflow, round-trip error, convergence-rate
  invariance, log-domain wins, quasi-arithmetic Γ-means, and transform-domain denoising
  including an honest negative result.
- **Literature verification.** Every load-bearing citation whose details I was not certain of
  was verified against the online record (Clenshaw–Olver level-index, inverse-Γ literature,
  Ben-Tal 1977, Litvinov–Maslov, Coleman ELM, warped GPs, Mäkitalo–Foi). A search for the term
  "transformed scalar algorithms" finds no existing field of that name — the *name* is
  unclaimed; the *content* is not.

---

## 2. Mathematical foundations

### 2.1 Taxonomy: four things "transform before computing" can mean

The mission conflates several distinct constructions. Separating them resolves most of the
apparent novelty:

| # | Construction | What is warped | Canonical examples |
|---|---|---|---|
| T1 | **Value (range) warping** — `φ⁻¹∘A∘φ` applied pointwise to data values | amplitude axis | companding (μ-law), gamma correction, log-domain products, VST, homomorphic filtering |
| T2 | **Domain warping** — resample the independent variable | time/frequency/space axis | mel/Bark scales, warped DFT, constant-Q transform, dynamic time warping, NUFFT |
| T3 | **Structure transport** — redefine operations `x ⊕ y = φ⁻¹(φx + φy)` | the arithmetic itself | quasi-arithmetic means, log semiring, tropical algebra, Tsallis q-arithmetic, non-Newtonian calculus |
| T4 | **Operator/functional calculus** — transform the spectrum of an operator, `f(A)` | operator spectrum | shift-invert Lanczos, Chebyshev filtering, matrix functions, preconditioning |

TSA as stated is T1 + T3 (+ T4 when lifted from scalars to operators). All four rows are
individually mature fields. Baraniuk & Jones (1995) even built the general unitary framework
`U A U⁻¹` for signal processing explicitly [BJ95]. The literature table in §4 maps each row.

### 2.2 The conjugation principle and its invariants (why exact-arithmetic novelty is impossible)

**Proposition 1 (algorithmic equivalence).** Let T be a bijection between domains and
`A_T = T⁻¹ ∘ A ∘ T`. Then:
(i) iterates commute with conjugation: `(A_T)ⁿ = T⁻¹ ∘ Aⁿ ∘ T` for all n;
(ii) p is a fixed point of A iff `T⁻¹(p)` is a fixed point of `A_T`;
(iii) any property of A expressible through composition and equality transfers verbatim.
*Proof:* (i) telescoping `T∘T⁻¹ = id`; (ii)–(iii) immediate. ∎

This is the definition of **topological conjugacy** in dynamical systems (Schröder 1870,
Koenigs 1884): conjugate maps have *identical dynamics up to relabeling of the state space*.
An algorithm viewed as an iterated map therefore cannot gain new behavior from conjugation.

**Proposition 2 (convergence-rate invariance).** If φ is C¹ with φ′ ≠ 0, A has fixed point p
with A′(p) = λ, then `A_φ = φ⁻¹∘A∘φ` has fixed point q = φ⁻¹(p) with `(A_φ)′(q) = λ` — the
*same* multiplier, hence the same asymptotic linear convergence factor.
*Proof:* chain rule: `(A_φ)′(q) = [φ′(q)]⁻¹ · A′(p) · φ′(q) = A′(p)` (scalars commute). In n
dimensions the Jacobians are similar matrices, hence have equal spectra. ∎

**Experiment E6** (fixed-point iteration `x ← cos x`, conjugated by `exp`): theoretical rate
|A′(p)| = 0.673612…; measured rate plain = 0.6736158, measured rate conjugated = 0.6736175.
Identical, as proved. **Conjugation cannot accelerate an algorithm asymptotically.** Transient
(non-asymptotic) behavior and basins of attraction *can* differ — that is real but it is the
classical study of "which coordinates should I solve my problem in," i.e. preconditioning and
reparameterization (§4 rows 12–15).

Two important non-equivalences must be kept clearly separate from Proposition 1, because they
are where every real effect lives:

- **Newton's method is not conjugation-covariant.** Newton applied to `g = f ∘ φ` is a
  *different* iteration from the conjugate of Newton applied to f. Deuflhard's affine-invariance
  theory characterizes exactly which transformations Newton respects (affine ones) [Deu04].
  Solving in transformed variables is therefore genuinely different — and is the existing field
  of nonlinear preconditioning (ASPIN, Cai & Keyes 2002 [CK02]) and of reparameterized
  optimization (§6).
- **Statistical estimators are not conjugation-covariant** when noise enters before the
  transform: `E[φ⁻¹(F(φ(X)))] ≠ E[X]` for nonlinear φ. This asymmetry is precisely what VST,
  link functions, and warped GPs exploit and correct for (§8).

### 2.3 Structure transport, quasi-arithmetic operations, and the one historical success

Define transported addition `x ⊕_φ y := φ⁻¹(φ(x) + φ(y))` and the transformed mean
`M_φ(x₁…x_n) = φ⁻¹( (1/n) Σ φ(x_i) )`. This is the **quasi-arithmetic (Kolmogorov–Nagumo) mean**,
axiomatized in 1930 [Kol30, Nag30, dF31], with the characterization theory completed by Aczél
(1948) and the classical treatment in Hardy–Littlewood–Pólya (1934, Ch. III). Every "transformed
sum/mean/reduction" the mission proposes lands in this class; Rényi built his entropy on exactly
this construction (1961). φ = log gives the geometric mean and the **log semiring**
(⊕ = log-add-exp) which has been the production workhorse of HMM/CRF/speech decoding for
decades; Mohri's semiring framework systematizes it for graph algorithms [Moh02].

**Proposition 3 (transport rigidity).** Transporting the *entire* field structure of ℝ through a
bijection φ yields a field isomorphic to ℝ — by construction, φ is the isomorphism. No new
algebra can arise this way.

Genuinely new algebra arises only when the transform family **degenerates**: with
`x ⊕_h y = h·log(e^{x/h} + e^{y/h})`, the limit h → 0 gives `x ⊕ y = max(x,y)` — idempotent,
non-invertible, *not* transported from (ℝ,+) by any bijection. This is **Maslov
dequantization**, and it created idempotent/tropical mathematics: Viterbi decoding, shortest
paths, mathematical morphology (max-plus convolution = dilation; Maragos's slope transform is
the Legendre-transform analogue of Fourier analysis for these systems [Mar95]), tropical
geometry [Lit05]. This is simultaneously the best historical validation of the TSA *instinct*
and the proof that the instinct was acted on decades ago: the payoff was found at the
*non-invertible boundary* of the transform family, not in its invertible interior — where
Proposition 1 guarantees there is nothing.

### 2.4 Equivariance classes: which algorithms even notice a scalar transform

The mission's per-algorithm questionnaire (§6) collapses onto three classes:

- **Class I — monotone-equivariant (TSA is exactly a no-op).** Any algorithm built from
  comparisons only: sorting, ranking, median/rank filters, min/max/argmax, max-pooling,
  order statistics, comparison-based search. For monotone φ, `median(φ(x)) = φ(median(x))`,
  so `φ⁻¹∘median∘φ = median` *identically*. **Experiment E9 confirms this to the last bit:**
  direct median filtering of Poisson data and Anscombe-domain median filtering (with the exact
  algebraic inverse) give MSE 2.2561 vs 2.2561 — the same numbers. Any TSA claim about
  rank-based algorithms is void.
- **Class II — linear/affine-equivariant.** Linear filters, plain sums, OLS: conjugation by a
  *nonlinear* φ changes the estimator's semantics (a box filter in φ-domain is a quasi-arithmetic
  moving mean, biased in the original domain). The change is real but fully described by
  quasi-arithmetic mean theory + bias analysis (delta method); in E9 the box filter gained
  nothing from the transform (MSE 1.4815 direct vs 1.4852 transformed).
- **Class III — nonlinearity- or noise-model-sensitive.** Thresholding denoisers, k-means,
  gradient methods, Kalman filters, kernels. Here the transform genuinely changes behavior — and
  in every case investigated, the change coincides with a named existing technique: Bregman
  clustering [Ban05], mirror descent [NY83], VST [Ans48], warped GPs [SRG03], etc.

### 2.5 The intertwining criterion: what makes a transform algorithmically useful

Every historically successful T1/T3 transform possesses a **homomorphism (intertwining)
identity** turning an expensive/unstable operation into a cheap/stable one:

| Transform | Identity | Payoff |
|---|---|---|
| log / exp | log(xy) = log x + log y | products → sums (LNS, likelihoods) |
| Fourier | F(f∗g) = F(f)·F(g) | convolution → pointwise product |
| Mellin | M(f ⋆ g) = M(f)·M(g) (multiplicative conv.) | scale-invariant matching |
| Legendre–Fenchel | (f □ g)\* = f\* + g\* (inf-convolution) | morphology, Hamilton–Jacobi, tropical |
| z-transform | shift → multiplication | difference equations → algebra |
| logit | odds multiply | Bayesian evidence accumulation |

**Γ has no such identity.** Its functional equations (Γ(x+1) = xΓ(x), reflection, Gauss
multiplication) relate Γ at *shifted arguments*, not Γ of a *binary operation* to an operation
on Γ-values. Hölder's theorem (1887) — Γ satisfies no algebraic differential equation — is
strong evidence that no hidden algebraic intertwiner exists. The same holds for ζ, Bessel, and
Airy used as *pointwise scalar maps*. (Bessel and Airy have rich *integral-transform* theory —
Hankel transforms etc. — but that is row T2/T4 machinery, not scalar value warping; conflating
the two is a category error the mission should avoid.)

**Conservation of difficulty.** Even a perfect intertwiner only *moves* hardness: in LNS,
multiplication becomes exact addition but addition becomes a nonlinear table lookup
(Gaussian logarithms). Net benefit therefore depends on the algorithm's **operation mix** — the
European Logarithmic Microprocessor was competitive precisely for multiply/divide/accumulate-
dominated DSP workloads [Col00]. This "operation-mix" accounting, not any new mathematics, is
the correct decision procedure for when a TSA-style rewrite pays (§10).

---

## 3. The Γ / fractional-factorial family: detailed evaluation (Experiments E1–E5, E8)

**E1 — values.** 0.2! = 0.918169, 0.5! = √π/2 = 0.886227, 0.9! = 0.961766, (1/π)! = 0.894864,
√2! = 1.253815, e! = 4.260820, π! = 7.188083.

**E2 — non-injectivity (fatal).** Γ(x+1) on (0,1) has an interior minimum at
x\* = 0.4616321545, Γ(x\*+1) = 0.8856031944. Hence `0.1! = 0.951350769867 = 0.868463811992!`
— **x ↦ x! is not injective on (0,1) and is not a valid scalar transform there.** TSA requires
T invertible; the flagship example fails the entry condition. One must restrict to x ≥ x\*
(or use lgamma on a branch), surrendering the very interval (0,1) the mission highlights.

**E3/E5 — conditioning.** Forward relative condition κ_φ(x) = |x·ψ(x+1)| grows like x·ln x
(κ = 23.5 at x = 10, 461 at x = 100): the forward map is increasingly *error-amplifying*. The
inverse condition 1/κ_φ diverges at x\*: measured amplification of a 1-ulp perturbation of y
into relative error of the recovered x reaches ~10⁻¹¹ (i.e., ~10⁵ ulps) at x\*+10⁻⁶ and grows
unboundedly closer to x\*. Away from x\* the Newton round-trip is clean (~1e-16, E5), but the
transform pair costs ~10–50× an exp/log pair (lgamma + digamma per Newton step; no closed-form
inverse exists — the literature offers only asymptotic inverses via Stirling + Lambert W
[Can01, JDC17] and complex-analytic branch studies [Ped16]).

**E4 — dynamic range (opposite of compression).** Γ(x+1) overflows binary64 for
x > 170.624377. On (0,1) the map *compresses* the whole interval into [0.8856, 1.0] — a range
of width 0.114, wasting ~3 bits of relative precision before the ill-conditioned inverse
re-expands it. For x > 1 the map *expands* super-exponentially. Both directions are the wrong
direction: useful range-engineering transforms (μ-law, PQ curve, log) compress *large* ranges
monotonically. lgamma(x+1) ≈ x ln x − x is monotone on the branch and well-behaved — but then
the transform is asymptotically a re-scaled x·log x, and every algorithmic payoff it could offer
is already delivered more cheaply by log itself.

**E8 — the "Γ-mean."** The quasi-arithmetic mean with generator lgamma(x+1) is perfectly
well-defined on the increasing branch (data {1,2,3,10}: arithmetic 4.0, geometric 2.783,
Γ-mean 4.768). It is a legitimate new *member* of the Kolmogorov–Nagumo class — as is every
strictly monotone function's mean. It is not translation-equivariant, not homogeneous, and
carries no operational identity; nothing distinguishes it favorably within the class. Novelty
within a fully-characterized 1930s class is not novelty.

**Digamma, polygamma, ζ, Bessel, Airy.** ψ is the only Γ-family member that is a monotone
bijection on (0,∞) → ℝ — a genuinely valid transform — but ψ(x) = ln x − 1/(2x) + O(x⁻²): it is
asymptotically log at ~5× the cost. Its one legitimate role is statistical: for
X ~ Gamma(k,θ), E[ln X] = ψ(k) + ln θ, so ψ appears in maximum-likelihood estimation for
Gamma-family models — which is standard exponential-family/GLM theory, not a new algorithm.
ζ restricted to (1,∞) is a monotone bijection onto (1,∞) — invertible but identity-free and
expensive; dominated by elementary maps with the same shape (e.g., 1 + 1/(x−1)). Bessel J and
Airy Ai oscillate — non-injective on any large domain — and are excluded as scalar transforms
outright; their value lives in integral transforms (T2/T4).

**Spectral behavior.** A pointwise nonlinearity applied to a signal generates harmonic and
intermodulation distortion (its Chebyshev expansion gives the harmonic mixing coefficients).
φ-filter-φ⁻¹ pipelines control this only when φ intertwines the noise/signal composition law
(log for multiplicative or convolved signals — homomorphic SP). Γ-pipelines have no such story:
the distortion is uncontrolled. No new filter family arises (§8).

**Conclusion of §3: the Γ/fractional-factorial direction should be closed.** Every measurable
property (injectivity, conditioning, range, cost, identities, spectral control) is equal to or
worse than log/exp, and the two legitimate appearances of Γ-family functions in computation —
sufficient statistics of Gamma models, and lgamma as a *computational primitive* for
combinatorial quantities — are both classical.

---

## 4. Existing literature: the many names of TSA

The single most important falsification result: **each block of the mission's program already
has a name, a founding paper, and decades of development.**

| # | Field / name | Transform | Conjugated algorithm | Key references |
|---|---|---|---|---|
| 1 | Homomorphic signal processing | log (on spectra) | linear filtering | Oppenheim 1967; Oppenheim–Schafer–Stockham 1968; cepstrum: Bogert–Healy–Tukey 1963; images: Stockham 1972 |
| 2 | Companding / perceptual coding | μ-law, A-law, γ-curves, PQ | quantization | Bennett 1948; Smith 1957; ITU-T G.711; SMPTE ST 2084 (Miller et al. 2013) |
| 3 | Variance-stabilizing transforms | √, Anscombe, Box–Cox, Fisher z, arcsine | any Gaussian denoiser/estimator | Bartlett 1936; Anscombe 1948; Box–Cox 1964; exact unbiased inverse: Mäkitalo–Foi 2011 ×2 |
| 4 | GLM link functions; transform-both-sides regression | logit, probit, log, reciprocal, ψ | regression | Nelder–Wedderburn 1972; Carroll–Ruppert 1988 |
| 5 | Quasi-arithmetic means | any monotone φ | sums, means, reductions | Kolmogorov 1930; Nagumo 1930; de Finetti 1931; Aczél 1948; HLP 1934; Rényi 1961 |
| 6 | Mirror descent / exponentiated gradient / natural gradient | mirror map ∇ψ | gradient descent | Nemirovski–Yudin 1983; Beck–Teboulle 2003; Kivinen–Warmuth 1997; Amari 1998 |
| 7 | Bregman clustering / quasi-arithmetic centroids | convex generator | k-means, centroids | Bregman 1967; Banerjee et al. 2005; Nielsen–Nock |
| 8 | Geometric programming; hidden convexity | log of variables & values | convex optimization | Duffin–Peterson–Zener 1967; Boyd et al. 2007; Ben-Tal 1977 ((h,φ)-convexity) |
| 9 | Non-Newtonian / multiplicative calculus | bijection α | calculus itself | Grossman–Katz 1972 (widely criticized as ordinary calculus in disguise — Proposition 3 is the criticism) |
| 10 | Logarithmic number systems; level-index; posits | log; iterated exp; tapered log | hardware arithmetic | Kingsbury–Rayner 1971; Swartzlander–Alexopoulos 1975; Coleman et al. 2000 (ELM); Clenshaw–Olver 1984; Clenshaw–Turner 1987 (SLI); Gustafson–Yonemoto 2017 |
| 11 | Idempotent / tropical mathematics; log semiring | exp/log family, degenerate limit | shortest paths, Viterbi, morphology, algebraic geometry | Maslov; Litvinov 2005 [math/0507014]; Viterbi 1967; Mohri 2002; Maragos 1995 (slope transform) |
| 12 | Warped Gaussian processes; input warping for BO | learned monotone φ (incl. Beta CDF — the mission's "beta transform") | GP regression, Bayesian optimization | Snelson–Rasmussen–Ghahramani NIPS 2003; Snoek et al. ICML 2014 |
| 13 | Gaussianization; nonparanormal; copulas; quantile/rank transforms | marginal CDFs | density estimation, graphical models, PCA | Chen–Gopinath 2000; Liu–Lafferty–Wasserman 2009; Sklar 1959 |
| 14 | Compositional data analysis | log-ratio | all multivariate statistics | Aitchison 1982 |
| 15 | Log-Euclidean framework (matrix-level TSA) | matrix log | interpolation, averaging, PCA on SPD matrices | Arsigny et al. 2006 |
| 16 | Functional calculus / spectral transformation (operator-level TSA) | rational/polynomial f | eigensolvers, preconditioning | Ericsson–Ruhe 1980; Higham 2008 (*Functions of Matrices*) |
| 17 | Unitary equivalence in SP; frequency warping | allpass/unitary U | any LTI processing | Baraniuk–Jones 1995; Oppenheim–Johnson–Steiglitz 1971; Härmä et al. 2000; Brown 1991 (constant-Q) |
| 18 | Conjugacy in dynamics; fractional iteration | Schröder/Abel/Böttcher coordinates | iteration itself | Schröder 1870; Koenigs 1884; Kneser 1950 |
| 19 | Nonlinear preconditioning | nonlinear change of unknowns | Newton–Krylov | Cai–Keyes 2002 (ASPIN); Deuflhard 2004 |
| 20 | Deformed (q-, κ-) arithmetic in statistical physics | q-exp, κ-exp | thermostatistics, entropies | Nivanen et al. 2003; Borges 2004; Kaniadakis 2002; Naudts 2011 |
| 21 | Transformed attention / softmax | Tsallis q-exponential | attention, argmax relaxation | Martins–Astudillo 2016 (sparsemax); Peters et al. 2019 (entmax); Performer (Choromanski et al. 2021) |
| 22 | Automated FP rewriting | expression-level algebraic + transcendental rewrites | accuracy optimization | Panchekha et al. 2015 (Herbie) |
| 23 | Log/lognormal kriging; log-Kalman; log-weights particle filters | log | interpolation, filtering | Journel–Huijbregts 1978; standard practice; LSE normalization; square-root filters: Potter 1963 |
| 24 | Metric transforms; warped kernels | concave d ↦ f(d); pullback kernels | distances, kernel methods | Schoenberg 1938; Sampson–Guttorp 1992; pre-image problem: Mika et al. 1999 |

Rows 5, 9, 20 are *literally* the mission's T3 ("transformed sums/arithmetic"); rows 1–3, 12, 23
are *literally* its filter program (§8); rows 6–8 its optimization program; row 10 its
numerical-hardware program; row 11 its "new algebraic families" program; rows 16 and 15 the
operator- and matrix-level versions. **Nothing in the mission's algorithm list lands outside
this table** (§6 gives the row-by-row mapping).

---

## 5. Similar approaches and the precise delta

What TSA would add over the table above, stated as sharply as possible:

1. **A unifying name and taxonomy.** No survey we could locate spans all 24 rows; the closest
   are Litvinov's idempotent survey (row 11 only), Baraniuk–Jones (row 17 only), and the
   quasi-arithmetic-means literature (row 5 + information geometry). A cross-field survey
   organized by Propositions 1–3 and the intertwining criterion would be a real, publishable
   contribution — *as a survey*, not as new mathematics.
2. **Exotic special-function transforms (Γ, ζ, Bessel, Airy) as the φ.** Genuinely absent from
   the literature — and §3 shows *why*: they fail invertibility, conditioning, cost, or
   identity requirements. Absence of prior art here reflects a dead end, not an opportunity.
   (The mission's beta-function instinct, by contrast, exists and works: Beta-CDF input warping,
   row 12 — because a CDF is monotone, bounded, cheap, and statistically motivated.)
3. **Systematic equivariance classification** (§2.4) of standard algorithms under scalar
   reparameterization. Scattered results exist (affine invariance of Newton [Deu04], monotone
   equivariance of order statistics — folklore); a systematic treatment is thin in the
   literature. Modest but real.
4. **Automated transform selection** for numerical accuracy at the *algorithm* level (Herbie
   works at expression level, row 22). Partially open engineering research (§18).

---

## 6. The mission's algorithm-family questionnaire, answered

Questions per family: (1) meaningful? (2) inverse required? (3) mathematically equivalent?
(4) numerics changed? (5) robustness changed? (6) genuinely new?

| Algorithm family | Answers | Where it already lives |
|---|---|---|
| Sums, reductions, means | (1) yes (2) yes (3) no — different mean (4) yes (5) yes (outliers) (6) **no** — quasi-arithmetic means | row 5 |
| Dot products, norms, cosine | (1) partially — φ destroys bilinearity; `φ⁻¹(⟨φx,φy⟩)` is a kernel-like similarity (2) optional (3) no (4) yes (5) monotone-rescaling effects (6) **no** — 1-D feature maps / warped kernels | row 24 |
| Distances | (1) yes (2) no (rankings suffice) (3) no (4) yes (5) yes (6) **no** — pullback metrics (flat, §9), metric transforms | rows 13, 24 |
| Convolution, correlation | (1) only with an intertwiner (2) yes (3) no (4) yes (5) — (6) **no** — homomorphic deconvolution (log), morphology (max-plus) | rows 1, 11 |
| FFT, wavelets | (1) pointwise φ destroys linearity; useful compositions are log-between-transforms (cepstrum) or axis warping (2)–(6) **no** | rows 1, 17 |
| Kalman filter | (1) yes for positive/multiplicative states (2) yes (3) no (4)–(5) yes (6) **no** — log-state filters, square-root filters, UKF for the nonlinearity | row 23; Julier–Uhlmann 1997 |
| Particle filters | log-weights + LSE are already universal practice | row 23 |
| Interpolation | φ-domain interpolation = quasi-arithmetic interpolation; geodesics of pullback metric (§9) | rows 5, 23 (lognormal kriging), 15 (Log-Euclidean) |
| Optimization, GD, SGD | reparameterization = mirror descent / EG / natural gradient / geometric programming | rows 6, 8 |
| Least squares | transformed LS = GLM / transform-both-sides; changes loss ⇒ different estimator (real, known) | row 4 |
| Eigensolvers | scalar φ on matrix entries destroys spectra; the correct move is functional calculus on the operator — huge existing field | row 16 |
| PCA | marginal transforms → nonparanormal/copula PCA; matrix log → Log-Euclidean PCA; nonlinear feature maps → kernel PCA (with the pre-image problem = the cost of T⁻¹) | rows 13, 15, 24 |
| Clustering | k-means in φ-space = Bregman/quasi-arithmetic clustering | row 7 |
| Nearest neighbors, similarity search | monotone per-coordinate maps = feature scaling; rank-based structures are Class-I invariant | §2.4 |
| Attention, embeddings | softmax *is* an exp-domain normalization computed via LSE; deformed variants exist (entmax); kernelized attention (Performer) | row 21 |
| Graph algorithms | semiring reformulations (min-plus shortest paths, Viterbi) — the tropical success story | row 11 |

Summary: **(3) is "yes, equivalent" whenever the algorithm is used exactly and φ is applied
end-to-end (Prop. 1); every "no" in column (3) corresponds to a named existing method.**
Column (6) is "no" across the board.

---

## 7. Hypercomplex TSA

**Proposition 4 (transport isomorphism).** Let 𝔸 be any ℝ-algebra (quaternions ℍ, octonions 𝕆,
sedenions 𝕊, any Cayley–Dickson or Clifford algebra) and Φ = φ×…×φ the componentwise extension of
a bijection φ. Define transformed operations `a ⊞ b = Φ⁻¹(Φa + Φb)`, `a ⊠ b = Φ⁻¹(Φa · Φb)`.
Then (𝔸_φ, ⊞, ⊠) is **isomorphic to 𝔸** — Φ is the isomorphism. Consequently:

- zero divisors: `a ⊠ b = Φ⁻¹(0)` iff `Φa · Φb = 0` — the zero-divisor structure is the
  image under Φ⁻¹ of the original (sedenion zero divisors, classified by Moreno 1998, map over
  verbatim);
- alternativity, Moufang identities, associativity, power-associativity: preserved verbatim
  (they are equational properties, Prop. 1(iii));
- norm and conjugation: the transported norm `N_φ = N∘Φ` satisfies the transported composition
  law `N_φ(a ⊠ b) = N_φ(a)·N_φ(b)` exactly when N is multiplicative on 𝔸 — Hurwitz's theorem
  (1898) still bites: only ℝ, ℂ, ℍ, 𝕆 up to isomorphism, and Frobenius (1877) still limits the
  associative division algebras;
- inverses: `a^{⊠-1} = Φ⁻¹((Φa)⁻¹)`.

**The geometry of the algebra cannot be modified by any invertible scalar transform.** The
mission's questions (multiplication, norm, conjugation, inverse, zero divisors, alternativity,
Moufang, geometry) all receive the same answer: *unchanged up to relabeling*.

**Proposition 5 (componentwise rigidity).** Alternatively, keep the ordinary quaternion
operations but feed them transformed components, i.e., ask that `Quaternion(φa, φb, φc, φd)`
behave compatibly with quaternion arithmetic. Compatibility with + and × forces φ to be a field
automorphism of ℝ; the only field automorphism of ℝ is the identity (order is algebraically
definable via squares, automorphisms fix ℚ, monotone + fixing ℚ ⇒ identity). So componentwise
transformation without transport **destroys the algebra structure entirely** — the object is
just a relabeled 4-tuple, and any "multiplication" defined on it either transports (→ Prop. 4,
isomorphic) or fails distributivity/bilinearity.

**What survives: numerics only.** Cost of transported multiplication = 8 forward transforms +
1 quaternion product + 4 inverse transforms — strictly more work for bit-identical mathematics.
The only defensible variant is *representation-level*: storing components in LNS/posit/SLI
formats. A quaternion product is 16 multiplications + 12 additions; LNS makes the 16
multiplications exact and nearly free but converts the 12 additions into table interpolations —
the operation-mix accounting of §2.5 applies, and the niche (if any) is narrow embedded/FPGA
territory. Note that the genuinely active field of **hypercomplex numerical analysis**
(quaternion QR and FFT — Sangwine and others; Clifford analysis) concerns matrix algorithms
*over* ℍ, and is untouched by scalar transforms for the reasons above. Note also that the
legitimate exp/log that *does* appear in quaternion practice — slerp, `to_axis_angle`, Lie
group log/exp maps (present in SciRust's own `scirust-simd` quaternion module) — is the group
exponential, an intrinsic geometric map, not a scalar TSA transform.

---

## 8. TSA filters (Experiments E9, E9b)

The proposed pipeline transform → filter → inverse-transform **is** two classical families:

- **Homomorphic filtering** (Oppenheim–Schafer–Stockham 1968): φ = log chosen because it
  intertwines the *signal composition law* (multiplicative illumination, convolutive reverb)
  with addition, where linear filters separate components. The cepstrum (Bogert–Healy–Tukey
  1963) composes log between two Fourier transforms. SciRust's own recently-merged MMSE-LSA
  denoiser (Ephraim–Malah log-spectral-amplitude estimator) is a production example of
  log-domain estimation already in this repository.
- **Variance stabilization** (Anscombe 1948 → Mäkitalo–Foi 2011): φ chosen from the *noise
  model* (φ(x) = ∫ dμ/√var(μ), delta method) so that transformed noise is homoscedastic
  Gaussian, enabling any off-the-shelf Gaussian denoiser; the modern refinement is the exact
  unbiased inverse.

Our experiments add two honest data points that sharpen — and partially deflate — the standard
narrative:

**E9 (median & box filters, Poisson signal λ ∈ [5, 20]):** direct median MSE 2.2561; Anscombe +
median + algebraic inverse MSE 2.2561 — *bit-identical*, because the median is Class-I
monotone-equivariant (§2.4); the transform is provably useless there. The first-order unbiased
inverse changes the constant and lands at 2.2074 — a bias correction, not transform-domain
filtering. Box filter: 1.4815 direct vs 1.4852 transformed — a *linear* filter also gains
nothing on a smooth signal.

**E9b (Haar universal-threshold denoising, three count regimes):**

| λ range | raw MSE | direct-domain threshold | Anscombe + threshold + unbiased inverse |
|---|---|---|---|
| [1, 20] | 10.74 | **3.50** | 5.19 |
| [2, 100] | 51.07 | **35.46** | 46.73 |
| [5, 35] | 20.13 | **10.69** | 11.28 |

The naive VST pipeline **lost in all three regimes**: the square root compresses signal features
as much as it homogenizes noise, so at fixed universal threshold the signal-to-threshold ratio
deteriorates; meanwhile a MAD-estimated global σ in the direct domain partially adapts to the
average noise level. The VST pipeline's documented wins (Mäkitalo–Foi) come from pairing it with
*strong* Gaussian denoisers (BM3D-class) at low counts — the transform is an enabler for a
better denoiser, not an improvement in itself. **Conclusion: "transformed-scalar filters" are
not a new family, and their benefit is conditional and denoiser-dependent, not intrinsic.**
Comparisons the mission requested: bilateral and NLM attack signal-dependence *adaptively*
(no global transform, no inverse-bias problem); Wiener/Kalman handle it via explicit
noise-covariance modeling; wavelet thresholding is the component VST is designed to feed.

---

## 9. Information geometry

For φ : ℝ → ℝ smooth and increasing, the pullback of the Euclidean metric is
`g(x) = φ′(x)² dx²`. In one dimension — and for componentwise transforms in n dimensions
(product metrics) — this metric is **flat**: φ is an isometry onto Euclidean space. Hence:

- **geodesics** are φ-preimages of straight lines, `γ(t) = φ⁻¹((1−t)φ(a) + tφ(b))` — precisely
  the quasi-arithmetic interpolation of §2.3; no new geodesic phenomena (no curvature, no
  conjugate points) can occur;
- **neighborhoods/topology** are unchanged (monotone φ is a homeomorphism); only the metric
  *sizes* of balls change — i.e., a reweighting;
- **similarity measures** `(φ(x) − φ(y))²` are Bregman-type/pullback divergences — the fully
  developed machinery of Bregman divergences [Bre67, Ban05], f-divergences, and Amari's
  α-representations already contains every scalar-transform-induced divergence;
- **kernels** `k(x,y) = k₀(φ(x), φ(y))` are valid PD kernels (pullback along a feature-space
  reparameterization) — this is nonstationary-kernel design by input warping, classical in
  spatial statistics (Sampson–Guttorp 1992) and Bayesian optimization (Snoek et al. 2014).

Genuinely new geometry requires *non-separable* (curvature-creating) or *non-invertible*
transforms — both outside TSA's scalar hypothesis. **No new information geometry arises from
invertible scalar transforms.**

---

## 10. Numerical analysis

**Error calculus of the round trip.** In floating point, `Â_φ = fl(φ⁻¹(fl(A(fl(φ(x))))))`
carries three extra rounding stages. To first order the relative error of the result is

```
ε_total ≈ κ_{φ⁻¹}(y) · [ ε_{φ} · κ_A + ε_A ] + ε_{φ⁻¹},     y = A(φ(x)),
```

so TSA is a net *loss* of ~1–2 ulps **unless** the transform eliminates a catastrophic failure
mode of A itself. The catalogue of catastrophic failures it can eliminate is short and known:

- **overflow/underflow of long products** — E7: the naive product of 10⁵ uniform factors
  underflows to 0; the log-domain sum gives log(∏) = −100013.5247 exactly to ~10⁻¹¹ relative.
  This is why likelihoods, HMMs, and particle filters live in log space;
- **softmax/LSE cancellation** — correctly analyzed by Blanchard–Higham–Higham 2021;
- **extreme dynamic range as a system property** — LNS (row 10) and level-index arithmetic
  (Clenshaw–Olver 1984; SLI Clenshaw–Turner 1987), the latter being exactly an *iterated-log*
  scalar representation that provably never overflows — the most radical TSA-style number
  system in existence, published forty years ago.

**Cancellation:** unchanged or worsened by generic φ; improved only by problem-specific
rewrites (the domain of Herbie-style tools, row 22). **Conditioning:** Propositions 1–2 mean the
*mathematical* conditioning of the problem is invariant; only the *algorithmic* error routing
changes. **Determinism:** a TSA pipeline is exactly as reproducible as its transcendental
kernels; ordinary libm `lgamma`/`tgamma` are not correctly rounded and differ across platforms,
so deterministic TSA would require correctly-rounded special functions (RLIBM/CORE-MATH-style),
which currently cover elementary functions, not Γ. This connects directly to SciRust's
deterministic-SIMD-transcendentals work: a `lgamma`-based kernel would today *break* the repo's
determinism guarantees. **SIMD/GPU:** forward pointwise transforms vectorize perfectly;
Newton-based inverses (inverse-Γ) introduce data-dependent iteration counts → warp divergence;
LNS addition is table/gather-bound. **Cache/locality:** pointwise maps are stream-friendly and
neutral.

## 11. Complexity

Conjugation adds Θ(n) pre/post work and multiplies inner scalar-operation costs by constants;
it **cannot change asymptotic complexity** (Prop. 1 makes A_φ's operation graph identical to
A's). Claims of complexity improvement via TSA are impossible in exact arithmetic and reduce, in
finite precision, to constant-factor hardware arguments (row 10). Memory complexity: +Θ(n) if
the transformed copy is materialized, 0 if fused. Parallelization potential: identical to A.

---

## 12. Candidate TSA algorithms — the honest shortlist

Ranked by (expected value × probability of novelty):

1. **Automated transform selection at algorithm level** ("Herbie for pipelines"): given an
   algorithm + input distribution + FP format, search a dictionary of monotone transforms
   (log, logit, √, Box–Cox, Anscombe, μ-law, PQ) minimizing measured error/overflow. Engineering
   research; prior art at expression level only. Deliverable: a SciRust `scirust-automl`-style
   tool.
2. **Learned monotone scalar transforms as numeric preconditioners**: 1-D monotone rational-
   quadratic splines (normalizing-flow components) fitted to make data "nice" (Gaussian,
   homoscedastic, bounded-range) before classical numerics — the numerics-facing use of
   Gaussianization is thinner than its ML-facing use. Moderate novelty; must cite rows 12–13.
3. **VST utilities done right** in `scirust-signal`/`scirust-vision`: Anscombe + generalized
   Anscombe with exact unbiased inverses feeding the existing NLM/BM3D-class denoisers. Zero
   research novelty, real engineering value; E9b shows why the *unbiased inverse* and *strong
   denoiser* are both mandatory.
4. **Log-domain / LSE kernels** in `scirust-stats`/`scirust-estimation` (log-likelihood
   accumulators, log-weight particle filtering, log-Kalman for positive states). Standard
   practice; the repo should have it as infrastructure.
5. **Level-index/SLI or LNS experimental scalar type** for `scirust-simd` — a `RealScalar`
   implementation with overflow-free semantics for reliability/fatigue-style extreme products.
   Niche, but genuinely differentiating for an embedded-focused crate; prior art is hardware-
   oriented, a well-tested Rust software SLI is rare.
6. **Γ-family transforms:** only as statistical links for Gamma-distributed data (ψ in ML
   estimators) — i.e., ordinary GLM support in `scirust-stats`. As generic scalar transforms:
   **closed** (§3).

## 13. Candidate TSA filters

No new family exists to propose (§8). The defensible roadmap items are exactly: Anscombe/GAT +
exact unbiased inverse + BM3D-class Gaussian backend (measured, not assumed, against direct
Poisson denoisers at each count regime), and homomorphic (log-spectral) pipelines the repo has
already begun (MMSE-LSA). A "Γ-domain filter" has no noise model that induces it: VST generators
come from variance functions `∫dμ/√v(μ)`, and no standard noise family yields Γ or ζ as its
stabilizer.

## 14. Hypercomplex TSA — closed

Per §7: isomorphic by transport (Prop. 4), impossible componentwise (Prop. 5). The only open
sliver is representation-level (LNS quaternion arithmetic for FPGA/embedded — operation-mix
dependent, likely unfavorable because quaternion workloads are addition-heavy after the
multiply). Recommendation: do not invest, beyond possibly a benchmark note.

## 15. Potential applications (of the *valid* subset)

- Poisson/low-count imaging (microscopy, astronomy, LiDAR) — VST pipelines (§13).
- Embedded/edge inference and DSP with extreme dynamic range — LNS/SLI scalar types (§12.5).
- Probabilistic inference at scale — log-domain infrastructure (§12.4).
- Bayesian optimization / GP modeling of non-stationary engineering responses — input/output
  warping (rows 12, 24), relevant to `scirust-gp` and `scirust-automl`.
- Reliability/fatigue extreme-value products (`scirust-reliability`, `scirust-fatigue`) —
  overflow-free accumulation.

## 16. Possible patents — assessment: thin to none

The conjugation principle, log-domain computation, companding curves, VST pipelines, LNS/SLI
arithmetic, mirror descent, and warped kernels are all decades-old prior art; broad claims are
indefensible. The historically *successful* patents in this space claim **specific engineered
curves tied to specific pipelines** (e.g., Dolby's perceptual quantizer EOTF standardized as
SMPTE ST 2084). A comparable narrow opportunity would require: a specific novel transfer curve +
a specific measured advantage + a specific application context — none of which this
investigation found. The Γ-family direction is unpatentable *and* technically unsound. A
freedom-to-operate check matters more than filing: μ-law/PQ-adjacent territory is crowded.

## 17. Dead ends (each with its killing argument)

1. **x! as a transform on (0,1):** non-injective (E2 counterexample). Fatal by definition.
2. **Γ/ζ/Bessel/Airy as generic scalar transforms:** no intertwining identity (Hölder 1887);
   conditioning and cost strictly dominated by log/exp (§3, E3–E5).
3. **TSA for rank/order-based algorithms** (median filters, sorting, argmax, quantiles,
   comparison-based NN): provably a no-op (Class I, §2.4; E9 exact tie).
4. **TSA acceleration of iterative solvers:** impossible asymptotically (Prop. 2; E6).
5. **New hypercomplex geometry via scalar transforms:** impossible (Props. 4–5).
6. **New information geometry from scalar/componentwise transforms:** pullback metrics are
   flat (§9).
7. **Complexity-class improvements:** impossible (§11).
8. **New semirings from invertible transforms:** transport rigidity (Prop. 3); the only
   escape (degenerate limits) is the existing tropical field.
9. **VST as an automatic denoising win:** refuted in three regimes by E9b; benefit is
   denoiser- and regime-conditional.
10. **"Non-Newtonian calculus" style reboots:** row 9's fate is the cautionary tale — a
    transported calculus is ordinary calculus in disguise, and the mathematical community has
    said so for fifty years.

## 18. Experimental roadmap (if the valid subset is pursued)

**Phase 0 — infrastructure (weeks).** Log-domain accumulators + LSE + log-weights in
`scirust-stats`/`scirust-estimation`; correctly-rounded `lgamma`/`digamma` audit in
`scirust-special` (determinism gate, §10). *Kill criterion:* none — pure infrastructure.

**Phase 1 — VST pipeline benchmark (weeks).** Anscombe/GAT + exact unbiased inverse feeding
`scirust-signal`/`scirust-vision` denoisers (NLM, MMSE-LSA, NeighBlock); benchmark vs direct
Poisson denoisers across λ ∈ {0.1–2, 2–20, 20–200} with MSE/SSIM. *Kill criterion:* if the VST
arm cannot beat direct-domain methods at low counts (the regime where the literature says it
should), drop VST from the roadmap and record the negative result.

**Phase 2 — SLI/LNS scalar type (1–2 months).** Software symmetric level-index and/or LNS
`RealScalar` in `scirust-simd`; benchmark on reliability/fatigue extreme-product workloads vs
f64 + manual rescaling and vs f128 emulation. *Kill criterion:* < 2× robustness-or-speed
advantage on a real workload.

**Phase 3 — automated transform selection (research, 3–6 months).** Search over a transform
dictionary per pipeline stage with measured-error objective; compare against Herbie-style
expression rewriting on SciRust example pipelines. This is the only phase with publication
potential beyond a survey. *Kill criterion:* selected transforms never beat the hand-chosen
log/VST baselines.

**Explicitly not scheduled:** Γ-transform algorithms, hypercomplex TSA, TSA "filters" as a new
family, TSA metrics/geodesics — closed by §17.

## 19. Scientific risk assessment

- **Rediscovery risk: ~certainty** for the core idea (already realized — see §4). Any paper
  framing conjugation itself as new will be rejected or, worse, published and then discredited
  (the non-Newtonian-calculus trajectory). Mitigation: frame all output as survey + engineering
  + the narrow open questions of §18.
- **Technical risk of the valid subset: low** (Phases 0–2 are established practice) to
  **moderate** (Phase 3 search may not beat hand-tuning).
- **Opportunity cost:** the largest risk. The Γ/hypercomplex/geometry branches consumed most of
  the mission statement and are provably empty; continued investment there has expected value
  zero.
- **Reputational note:** the strongest positive finding of this investigation is a *negative
  result with proofs* (Props. 1–5 + E-series). Publishing "why transformed scalars cannot give
  you X, and the complete map of where they already give you Y" is honest, useful, and safe.

## 20. Most promising research directions (ranked)

1. Cross-field **survey/taxonomy** built on Propositions 1–5, the intertwining criterion, and
   the equivariance classes — the unclaimed contribution this investigation found.
2. **Automated transform selection** for numerical pipelines (Phase 3).
3. **Learned monotone transforms as numeric preconditioners** (flows → classical numerics).
4. **Software SLI/LNS scalar types** in Rust for embedded scientific computing (Phase 2).
5. **Degenerate-limit exploration** — a mathematically serious but hard question: are there
   useful idempotent-type limits of transform families other than max/min-plus? (The idempotent-
   analysis literature suggests the answer is essentially no for scalar semirings, which is
   itself worth establishing rigorously.)

## 21. Conclusion — the mission's final question

> *Determine whether TSA can become a coherent mathematical and algorithmic framework, or
> whether the idea reduces to already-known coordinate changes.*

**It reduces to already-known coordinate changes.** The reduction is not partial: the
invertible-transform core is exactly conjugation (Props. 1–2), the transformed-arithmetic
extension is exactly transport of structure and quasi-arithmetic means (Prop. 3), the
hypercomplex extension is exactly transport isomorphism (Props. 4–5), the filter program is
exactly homomorphic filtering + variance stabilization, and the one historical route to genuine
novelty (degenerate limits) was taken in the 1980s–90s and became tropical mathematics. The
specific special-function transforms proposed (Γ and fractional factorials foremost) fail the
entry requirements of the framework they were meant to power — invertibility, conditioning,
cost, and the intertwining property — and are dominated by log/exp on every axis measured.

What deserves to exist is not a field but: one survey, three engineering deliverables, and one
open engineering-research question — all enumerated in §18 and §20, all compatible with this
repository's existing structure, and none requiring the TSA name to carry ontological weight.

---

## Appendix A — Experiments (reproducible)

Scripts: `docs/research/tsa_experiments/tsa_experiments.py` (E1–E9),
`docs/research/tsa_experiments/tsa_e9b_wavelet.py` (E9b). Pure Python 3 stdlib; fixed seeds.
Key outputs quoted in §§3, 8, 10 above were produced by these scripts on 2026-07-16
(Python 3.11.15, Linux x86-64, IEEE-754 binary64).

| Exp | Claim tested | Result |
|---|---|---|
| E1 | fractional factorial values | table in §3 |
| E2 | injectivity of x! on (0,1) | **refuted**: 0.1! = 0.868463811992! |
| E3 | conditioning of Γ(x+1) and inverse | κ_fwd ~ x ln x; κ_inv → ∞ at x\* (10⁸ at x\*+10⁻⁸) |
| E4 | dynamic range | overflow for x > 170.624377 (binary64) |
| E5 | round-trip accuracy | ~1e-16 away from x\*; 1 ulp → 7×10⁻¹¹ near x\* |
| E6 | conjugation preserves convergence rate | confirmed to 6 digits |
| E7 | log-domain fixes product underflow | confirmed (10⁵ factors) |
| E8 | Γ-mean is a quasi-arithmetic mean, nothing more | confirmed; not translation-equivariant |
| E9 | median filter transform-equivariance | confirmed: MSE identical to 16 digits |
| E9b | "VST pipeline always helps" | **refuted** in 3 regimes with universal-threshold Haar |

## Appendix B — References

Verified online during this investigation (2026-07-16):

- Clenshaw & Olver, level-index arithmetic (1984); Clenshaw & Turner, symmetric level-index
  (1987). Survey: [Level-Index Arithmetic: An Introductory Survey](https://link.springer.com/chapter/10.1007/BFb0085718); [overview](https://en.wikipedia.org/wiki/Level-index_system).
- Inverse gamma function: [Wikipedia](https://en.m.wikipedia.org/wiki/Inverse_gamma_function);
  Cantrell's Lambert-W asymptotic via [J. D. Cook (2017)](https://www.johndcook.com/blog/2017/02/11/approximate-inverse-of-the-gamma-function/);
  H. L. Pedersen, "Inverses of gamma functions," *Constr. Approx.*
- Ben-Tal (1977), "On generalized means and generalized convex functions," *JOTA* — via
  [generalized convexity literature](https://www.cambridge.org/core/journals/bulletin-of-the-australian-mathematical-society/article/generalized-convexity-in-mathematical-programming/F93093B5064E77DA42C6C914C1283001).
- Litvinov, "The Maslov dequantization, idempotent and tropical mathematics: a brief
  introduction," [arXiv:math/0507014](https://arxiv.org/abs/math/0507014).
- Coleman, Chester, Softley, Kadlec, "Arithmetic on the European Logarithmic Microprocessor,"
  *IEEE Trans. Computers* 49(7):702–715 (2000), [IEEE](https://ieeexplore.ieee.org/document/863040/).
- Snelson, Rasmussen, Ghahramani, "Warped Gaussian Processes," *NIPS 2003*,
  [paper](https://papers.nips.cc/paper/2481-warped-gaussian-processes).
- Mäkitalo & Foi, "Optimal inversion of the Anscombe transformation in low-count Poisson image
  denoising," *IEEE TIP* 20(1):99–109 (2011); "A closed-form approximation of the exact unbiased
  inverse of the Anscombe variance-stabilizing transformation," *IEEE TIP* 20(9):2697–2698
  (2011), [IEEE](https://ieeexplore.ieee.org/document/5721817/); [software page](https://webpages.tuni.fi/foi/invansc/index.html).

Standard references cited from the established literature:

[Ans48] Anscombe, *Biometrika* 35 (1948). — [Bar36] Bartlett (1936), √-transform. —
[BC64] Box & Cox, *JRSS B* 26 (1964). — [BJ95] Baraniuk & Jones, "Unitary equivalence: a new
twist on signal processing," *IEEE Trans. SP* 43 (1995). — [Ban05] Banerjee, Merugu, Dhillon,
Ghosh, "Clustering with Bregman divergences," *JMLR* 6 (2005). — [Bre67] Bregman (1967). —
[BHT63] Bogert, Healy, Tukey, cepstrum (1963). — [Ben48] Bennett, "Spectra of quantized
signals," *BSTJ* 27 (1948). — [Boyd07] Boyd, Kim, Vandenberghe, Hassibi, "A tutorial on
geometric programming," *Optim. Eng.* 8 (2007). — [Bro91] Brown, constant-Q transform, *JASA*
89 (1991). — [Can01] Cantrell, sci.math note on inverse Γ (2001). — [CK02] Cai & Keyes, ASPIN,
*SIAM J. Sci. Comput.* 24 (2002). — [CG00] Chen & Gopinath, "Gaussianization," *NIPS* (2000). —
[Cho21] Choromanski et al., Performer, *ICLR* (2021). — [CR88] Carroll & Ruppert,
*Transformation and Weighting in Regression* (1988). — [Deu04] Deuflhard, *Newton Methods for
Nonlinear Problems* (2004). — [DPZ67] Duffin, Peterson, Zener, *Geometric Programming* (1967).
— [dF31] de Finetti (1931). — [ER80] Ericsson & Ruhe, spectral transformation Lanczos, *Math.
Comp.* 35 (1980). — [GK72] Grossman & Katz, *Non-Newtonian Calculus* (1972). — [GY17] Gustafson
& Yonemoto, posits (2017). — [Här00] Härmä et al., "Frequency-warped signal processing for
audio," *JAES* 48 (2000). — [Hig08] Higham, *Functions of Matrices* (2008). — [BHH21]
Blanchard, Higham, Higham, "Accurately computing the log-sum-exp and softmax functions," *IMA
J. Numer. Anal.* 41 (2021). — [HLP34] Hardy, Littlewood, Pólya, *Inequalities* (1934). —
[Höl87] Hölder, Γ hypertranscendence (1887). — [JH78] Journel & Huijbregts, *Mining
Geostatistics* (1978). — [JU97] Julier & Uhlmann, UKF (1997). — [Kan02] Kaniadakis, *PRE* 66
(2002). — [KR71] Kingsbury & Rayner, log arithmetic, *Electron. Lett.* 7 (1971). — [KW97]
Kivinen & Warmuth, exponentiated gradient, *Inf. Comput.* 132 (1997). — [Kol30] Kolmogorov
(1930). — [Kne50] Kneser (1950). — [LLW09] Liu, Lafferty, Wasserman, nonparanormal, *JMLR* 10
(2009). — [Mar95] Maragos, slope transforms, *IEEE Trans. SP* (1995). — [MA16] Martins &
Astudillo, sparsemax, *ICML* (2016). — [Mik99] Mika et al., kernel PCA pre-images, *NIPS*
(1999). — [Mil13] Miller, Nezamabadi, Daly, PQ curve, *SMPTE Mot. Imag. J.* (2013); SMPTE ST
2084. — [Moh02] Mohri, "Semiring frameworks and algorithms for shortest-distance problems"
(2002). — [Mor98] Moreno, zero divisors of Cayley–Dickson algebras (1998). — [Nag30] Nagumo
(1930). — [Nau11] Naudts, *Generalised Thermostatistics* (2011). — [NW72] Nelder & Wedderburn,
GLM, *JRSS A* 135 (1972). — [Niv03] Nivanen, Le Méhauté, Wang, q-algebra (2003); Borges (2004).
— [NY83] Nemirovski & Yudin (1983); Beck & Teboulle, *Oper. Res. Lett.* 31 (2003). — [OJS71]
Oppenheim, Johnson, Steiglitz, *Proc. IEEE* 59 (1971). — [OSS68] Oppenheim, Schafer, Stockham,
"Nonlinear filtering of multiplied and convolved signals," *Proc. IEEE* 56 (1968); Oppenheim,
*Inf. Control* 11 (1967); Stockham, *Proc. IEEE* 60 (1972). — [Pan15] Panchekha et al., Herbie,
*PLDI* (2015). — [Pet19] Peters, Niculae, Martins, entmax, *ACL* (2019). — [Pot63] Potter,
square-root filtering (1963). — [Rén61] Rényi (1961). — [SG92] Sampson & Guttorp, *JASA* 87
(1992). — [Sch38] Schoenberg, *Trans. AMS* 44 (1938). — [Skl59] Sklar (1959). — [Smi57] Smith,
"Instantaneous companding," *BSTJ* 36 (1957). — [Sno14] Snoek, Swersky, Zemel, Adams, "Input
warping for Bayesian optimization," *ICML* (2014). — [SA75] Swartzlander & Alexopoulos,
sign/logarithm number system, *IEEE Trans. Comput.* (1975). — [Vit67] Viterbi (1967). — [Ait82]
Aitchison, log-ratio analysis, *JRSS B* 44 (1982). — [Ars06] Arsigny, Fillard, Pennec, Ayache,
Log-Euclidean metrics (2006). — [Ama98] Amari, natural gradient, *Neural Comput.* 10 (1998);
Amari & Nagaoka, *Methods of Information Geometry* (2000). — [Acz48] Aczél (1948). — [BM22]
Bohr & Mollerup (1922). — [Fro1877] Frobenius (1877). — [Hur1898] Hurwitz (1898). — [Sch1870]
Schröder (1870); Koenigs (1884).
