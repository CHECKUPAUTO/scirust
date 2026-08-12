# Research program — Transformed-Scalar Hypercomplex Filters (TSHF)

**Date**: 2026-07-16 · **Status**: investigation complete, verdict rendered
**Reproducibility**: all measurements in this report come from
`cargo run -p scirust-signal --example tshf_experiments` (deterministic, fixed seed).

## Executive summary

The TSHF proposal — pointwise transform the scalar (`φ(x)` with
`φ ∈ {1/Γ(x+1), ln Γ(x+1), signed log, power law, tanh, sigmoid, atan,
softsign, …}`), embed in a hypercomplex algebra (quaternion/octonion/sedenion),
filter, then invert — was subjected to a mathematical analysis, six blocks of
numerical falsification experiments, and an extensive literature review.

**Verdict: the proposal, as a "new family of filters", is not
scientifically defensible.** Its two components are separable and each is,
either already covered by an abundant literature (variance stabilization,
homomorphic filtering, companding, quasi-arithmetic means), or contradicted by
our own measurements (Γ transforms, higher-dimensional embeddings without a
coupling operator). A **narrow, already-known subset** on the other hand
deserves an implementation in SciRust: the variance-stabilizing transforms
(Anscombe, Box-Cox/signed log, signed root) **with bias-corrected inverse**,
for signal-dependent noise — see §12–13.

The salient points, with supporting figures:

- `1/Γ(x+1)` is **non-injective** (φ(0) = φ(1) = 1): immediate rejection — no
  reconstruction possible (E2).
- `ln Γ(x+1)` is non-monotonic below x ≈ 0.4616 and its numerical inversion
  amplifies noise ×27: rejected for a pipeline with reconstruction (E2).
- For **additive Gaussian** noise, the identity beats *all* tested transforms,
  on the three filters tested (E3) — consistent with theory (the noise is
  already stationary; any non-affine φ makes it level-dependent, E1).
- The **median is invariant** under any monotone φ: the median column of E3
  is constant to 10⁻¹² — the TSHF pipeline is mathematically a no-op there.
- The saturating transforms (tanh/sigmoid) inflict a **retransformation bias**
  (Jensen) measured up to −0.13 on a level of 2 (−6.5 %) and a noise
  amplification at inversion ×22–×101 (E2, E4); tanh + wavelets destroys the
  signal (10.4 dB < 12.6 dB raw, E3).
- The hypercomplex embedding is **orthogonal** to the question: any
  R-linear filter applied component-wise is identical to per-channel filtering
  (algebraic identity, E5a); the transformation/embedding order only matters
  if the transform couples the coordinates (E5c); and on our correlated-impulse
  fixture, the vector (joint) median *lost* to the per-channel median
  (12.7 dB vs 14.6 dB, E5b) — higher dimension does not help by default.

---

## 1. Mathematical foundations

### 1.1 The pipeline studied

Two architectures were analyzed:

```
(A)  x → φ(x) → embedding H → filter L → φ⁻¹ → x̂
(B)  x → embedding H → φ (component-wise or coupled) → filter L → inverses → x̂
```

**Proposition 1 (separability).** If φ acts component-wise and if L is
R-linear applied component-wise, then (A) ≡ (B) ≡ per-channel filtering of
φ(x): the hypercomplex embedding is transparent. *Proof*: a
coordinates-to-coordinates embedding is a permutation of the data order; a
component-wise R-linear operator commutes with it. Verified
numerically (E5a, E5c: exact identity). The order only becomes significant if
φ couples the coordinates (e.g. `v ↦ v·tanh(‖v‖)/‖v‖`, E5c: 0.426 ≠ 0.462) or if L
uses the hypercomplex product.

**Corollary.** The possible "novelty" of TSHF cannot come from the φ +
embedding combination per se; it must come either from φ (a classic
variance-stabilization question, §2), or from operators that genuinely exploit
the algebra's product (QFT, widely-linear filters — established quaternionic
literature, §3).

**Proposition 2 (median invariance).** For any strictly monotone φ,
`median(φ(x_i)) = φ(median(x_i))`, hence `φ⁻¹ ∘ median ∘ φ = median`: the TSHF
pipeline with a rank filter is the identity filter. Confirmed by E3 (median
column constant across the 8 transforms). A whole class of filters
(rank/order) is therefore *off-topic* for TSHF.

**Proposition 3 (quasi-arithmetic means).** For L = moving average,
`φ⁻¹(MA(φ(x)))` is the Kolmogorov-Nagumo quasi-arithmetic mean with generator
φ (log → geometric mean, x⁻¹ → harmonic…). The TSHF-MA pipeline is therefore a
mathematical object known since 1930, not a new construction.

### 1.2 Effect on the noise statistics (E1)

First-order expansion: for x = s + n, `φ(x) − φ(s) ≈ φ′(s)·n`, hence
`σ_φ(s) ≈ |φ′(s)|·σ(s)`. Three regimes measured (σ after transformation,
levels s = 0.6 / 1.2 / 1.8 / 2.4):

| φ | additive σ=0.3 | multiplicative 0.3·s | Poisson-like 0.3·√s |
|---|---|---|---|
| identity | 0.300 / 0.300 / 0.300 / 0.300 (flat ✓) | 0.18→0.72 (×4) | 0.23→0.46 (×2) |
| signed log | 0.195→0.089 (de-stabilizes) | 0.114→0.226 (×2, flattened) | — |
| signed root | 0.252→0.098 (de-stabilizes) | — | 0.173→0.153 (**flat ✓**) |
| Anscombe | 0.306→0.181 | 0.185→0.447 | 0.240→0.283 (**near flat ✓**) |
| tanh | 0.214→0.013 (squashes) | — | — |
| ln Γ(x+1) | 0.065→0.320 (**amplifies** the dependence) | 0.029→0.759 (worse than identity) | — |

Answers to scientific questions 1–2: **yes**, a scalar transform modifies the
noise statistics — but in both directions. On additive noise (already
stationary), any non-affine φ *creates* a level dependence, which degrades
global-threshold filters. Stabilization only helps if the original noise is
signal-dependent AND φ is *matched to the noise model* (root ↔ Poisson,
log ↔ multiplicative) — which is exactly the definition of the classic
variance-stabilizing transforms. `ln Γ` does the opposite of what is expected
of a VST on all tested models.

### 1.3 Invertibility, domains, branches, conditioning (E2)

| φ | injective? | domain | max \|dφ⁻¹/dy\| over x∈[−3,3] | round-trip error |
|---|---|---|---|---|
| identity | yes | ℝ | 1.0 | 0 |
| signed log | yes | ℝ | 4.0 | 7·10⁻¹⁶ |
| signed root | yes | ℝ | 3.5 | 4·10⁻¹⁶ |
| Anscombe | yes | x ≥ −0.375 | 1.8 | 4·10⁻¹⁶ |
| atan | yes | ℝ | 10.0 | 9·10⁻¹⁶ |
| softsign | yes | ℝ | 16.0 | 9·10⁻¹⁶ |
| sigmoid | yes | ℝ | 22.1 | 3·10⁻¹⁵ |
| tanh | yes | ℝ | **101.4** | 2·10⁻¹⁴ |
| ln Γ(x+1) | **no** (monotone only x > 0.4616) | x > −1 | 27.4 (+ Newton inversion) | 4·10⁻¹⁴ |
| 1/Γ(x+1) | **NO**: φ(0) = φ(1) = 1, max at x ≈ 0.4616 | — | — | reconstruction impossible |

The factor `max |dφ⁻¹/dy|` bounds the residual-noise amplification at
reconstruction (Lipschitz behavior of the inverse). The saturating transforms
concentrate this amplification exactly where the strong signal lives — the
worst place. Questions 5–6: **yes**, numerical instability and artifacts are
amplified, and structurally (not fixable by implementation).

### 1.4 Retransformation bias (E4)

`E[φ⁻¹(L(φ(x)))] ≠ s` for non-affine φ (Jensen inequality). Measured on flat
signal s = 2, noise g = 0.4, MA(9): identity −0.001; Anscombe −0.017;
signed root −0.020; signed log −0.026; softsign −0.050; sigmoid −0.054;
atan −0.060; **tanh −0.131** (−6.5 % of the level); ln Γ +0.030. This is the
known retransformation bias of the literature (Duan's smearing, Mäkitalo-Foi
exact unbiased inverse, §2): the naive algebraic inverse of the TSHF statement
is precisely the variant the literature has shown to be defective.

---

## 2–4. Literature review, existing work, already-published concepts

*(Section established by extensive bibliographic research — see the references
in fine; each TSHF component is set against the prior art.)*

*Review conducted via extensive web research (dedicated agent); kept in English, the language of the sources. Negative searches are documented with their exact scope, so that an absence of results is data rather than an assumption.*

**Method under evaluation:** apply pointwise analytic scalar transform φ (1/Γ(x+1), log Γ(x+1), signed log, power law, tanh, sigmoid, atan, softsign) → optionally embed in quaternion/octonion/sedenion algebra → filter → invert φ.

**Bottom line up front:** The architectural skeleton φ⁻¹ ∘ L ∘ φ is one of the oldest and most thoroughly developed ideas in signal processing (homomorphic filtering, 1968; variance stabilization, 1948; nonlinear mean filters, 1980s; Kolmogorov–Nagumo means, 1930/1948). Hypercomplex filtering is likewise a mature field (1990s–present). Both halves are heavily covered prior art. What I could **not** find anywhere is (a) the specific use of 1/Γ(x+1) or log Γ(x+1) as the pointwise transform, and (b) any paper explicitly combining a scalar pointwise pre-transform with hypercomplex-algebra filtering as a unified framework. Neither gap looks like a *conceptual* novelty — they are unexplored parameter choices inside a well-known template, and the literature already explains why the interesting design question is not "which φ" but "which φ matches the noise model, and how do you invert without bias."

---

## 1. Variance-stabilizing transforms (VST) + denoise + inverse

- **Anscombe, F.J. (1948), "The transformation of Poisson, binomial and negative-binomial data," *Biometrika* 35:246–254.** The transform 2√(x+3/8) makes Poisson data approximately unit-variance Gaussian; the canonical instance of "transform → treat as Gaussian → invert." Overlap with TSHF: identical three-step pipeline, with φ chosen *for a statistical reason* rather than as a free menu item.
- **Murtagh, Starck & Bijaoui (1995); Starck, Murtagh & Bijaoui, *Image Processing and Data Analysis: The Multiscale Approach* (CUP, 1998).** Generalized Anscombe Transformation (GAT) for mixed Poisson–Gaussian noise. Overlap: extends the same φ-pipeline to a two-parameter noise model — evidence the field parameterizes φ by noise physics.
- **Mäkitalo & Foi (2011), "Optimal inversion of the Anscombe transformation in low-count Poisson image denoising," *IEEE Trans. Image Processing* 20(1):99–109; also the closed-form approximation note, IEEE TIP 20(9):2697–2698 (2011); and "Optimal inversion of the GAT for Poisson-Gaussian noise," IEEE TIP 22(1):91–103 (2013).** Shows the naive algebraic inverse φ⁻¹ is *biased* (because E[φ(x)] ≠ φ(E[x])) and derives the exact unbiased inverse; VST+BM3D with this inverse is competitive with dedicated Poisson denoisers. Overlap: this is precisely the reconstruction step of TSHF, and it demonstrates that "just apply φ⁻¹" — as TSHF proposes — is the known-wrong way to do it.
- **Freeman & Tukey (1950), *Annals of Mathematical Statistics*** — √(x)+√(x+1) variant for Poisson-like data; **Box & Cox (1964), *JRSS-B*** — power-law/log family (TSHF's "power law" and "log" candidates are literally the Box-Cox family). 
- **Fryzlewicz & Nason (2004), "A Haar-Fisz algorithm for Poisson intensity estimation," *J. Comput. Graph. Statist.* 13:621–638.** Multiscale (data-driven) variance stabilization — shows the field moved beyond fixed pointwise φ a decade ago.

**Verdict:** the "scalar transform → filter → invert" pipeline for denoising is 78 years old and its inverse-bias problem is solved; TSHF adds no structure here.

## 2. Homomorphic filtering

- **Oppenheim, Schafer & Stockham (1968), "Nonlinear filtering of multiplied and convolved signals," *Proc. IEEE* 56(8):1264–1291** (building on Oppenheim's 1964 MIT thesis). The general theory of homomorphic systems: map signals through an invertible nonlinearity (log) into a vector space where the "noise combination rule" becomes addition, filter linearly, invert. Overlap: this *is* the TSHF template, stated with an explicit algebraic justification for the choice of φ — a homomorphism between signal-combination operations.
- **Cepstral processing (Bogert, Healy & Tukey 1963; Oppenheim & Schafer)** — filtering after log of the spectrum; the same idea in the Fourier domain.
- **Homomorphic wavelet despeckling:** e.g., **Gupta, Chauhan & Saxena (2005), "Homomorphic wavelet thresholding technique for denoising medical ultrasound images," *J. Med. Eng. Technol.*/PubMed 16126580**, and a large SAR literature (log-transform speckle → additive → wavelet threshold → exp). Overlap: exactly "φ = log, filter = wavelet shrinkage, invert" — with the extra, well-documented caveat that log-speckle has a nonzero mean (−ψ(L)+log L terms, trigamma variance) requiring debiasing before exponentiation, again anticipating TSHF's inversion problem.
- **Pitas & Venetsanopoulos, *Nonlinear Digital Filters: Principles and Applications* (Kluwer, 1990), ch. on homomorphic and nonlinear mean filters.** Textbook treatment: geometric mean filter = exp(mean(log x)), harmonic mean = 1/mean(1/x), Lp/contraharmonic means = power-law φ. Overlap: the entire "nonlinear mean filter" chapter is a catalogue of φ⁻¹∘(local average)∘φ for φ ∈ {log, 1/x, x^p} — i.e., three of TSHF's eight candidate transforms were textbook material by 1990.

## 3. Companding

- **μ-law/A-law (ITU-T G.711; Smith 1957 for μ-law theory).** Compress–process–expand for quantization-noise shaping; "compander" is the canonical name for the φ/φ⁻¹ sandwich. Overlap: the TSHF signed-log candidate is essentially μ-law.
- **Nonlinear companding transforms for OFDM PAPR reduction:** **Wang, Tjhung & Ng (1999)** (μ-law companding), **Jiang et al. (2005), "Exponential companding technique for PAPR reduction in OFDM systems," *IEEE Trans. Broadcasting* 51(2)**, plus published **tanh-companding** and log-companding variants (survey: Anoh et al., *J. Inf. Telecommun.* 2019). Overlap: tanh/sigmoid-family φ applied pointwise to signals, with explicit inverse at the receiver — TSHF's tanh/sigmoid/atan/softsign candidates are all companders of this type; none is used because of Γ-like analytic structure but because of amplitude-distribution shaping.
- **Durand & Dorsey (2002), "Fast bilateral filtering for the display of high-dynamic-range images," *ACM Trans. Graphics* (SIGGRAPH).** Bilateral filtering performed in the log-luminance domain, then inverted — a mainstream example of "range-compress, filter, expand" in imaging.

## 4. Quaternion signal/image processing

- **Sangwine (1996, *Electronics Letters*, quaternion FT of colour images); Ell & Sangwine (2007), "Hypercomplex Fourier transforms of color images," *IEEE Trans. Image Processing* 16(1):22–35.** Holistic frequency-domain processing of RGB as pure quaternions; computable via two complex FFTs. Overlap: TSHF's "embed in quaternion, filter" step is this literature's founding move.
- **Cheong Took & Mandic (2009), "The quaternion LMS algorithm for adaptive filtering of hypercomplex processes," *IEEE Trans. Signal Processing* 57(4); and (2010) "A quaternion widely linear adaptive filter," *IEEE TSP* 58(8).** Augmented quaternion statistics, widely-linear QLMS. **Jahanchahi & Mandic (2014), "A class of quaternion Kalman filters," *IEEE Trans. Neural Netw. Learn. Syst.* 25(3).** Overlap: mature linear-systems theory *inside* the quaternion algebra — what TSHF would need for its "filter in transformed coordinates" stage to be principled.
- **Chan, Choi & Baraniuk (2008), "Coherent multiscale image processing using dual-tree quaternion wavelets," *IEEE TIP* 17(7):1069–1082**; plus quaternion-wavelet denoising follow-ons (Yin et al. 2012, *Math. Probl. Eng.*).
- **Astola, Haavisto & Neuvo (1990), "Vector median filters," *Proc. IEEE* 78(4):678–689.** Nonlinear multichannel impulse-noise filtering treating color samples as vectors. Overlap: the standard answer to "multichannel impulses" that TSHF would have to beat.
- **Yu, Zhou et al. (2019), "Quaternion-based weighted nuclear norm minimization for color image denoising," *Neurocomputing*.** State-of-the-art low-rank quaternion denoising; shows the quaternion-denoising bar TSHF must clear is high and recent.

## 5. Octonion/sedenion signal processing

- **Błaszczyk & Snopek (2017–2020):** "Octonion Fourier transform of real-valued functions of three variables" (*Bull. Pol. Acad. Sci.* 2018); **Błaszczyk (2020), "A generalization of the octonion Fourier transform to 3-D octonion-valued signals," *Multidim. Syst. Signal Process.* 31** (arXiv:1905.12631); discrete OFT in *Comput. Appl. Math.* 39 (2020). Core claim: OFT is well-defined and most FT properties survive, *but* non-associativity forces careful left-to-right multiplication order and kills some marginal/convolution identities. Overlap: directly answers TSHF's octonion step — the transform exists, and the literature explicitly documents the price of non-associativity for LTI theory.
- **Alfsmann, Göckler, Sangwine & Ell (2007), "Hypercomplex algebras in digital signal processing: benefits and drawbacks," EUSIPCO 2007, pp. 1322–1326.** Survey concluding that beyond quaternions, loss of associativity/division-algebra structure (octonions non-associative; sedenions with zero divisors) severely limits linear-systems constructs. This is the standing skeptical position TSHF's sedenion variant must answer: **sedenions have zero divisors**, so "filtering" can annihilate nonzero signal content.
- **Popa (2016), "Octonion-valued neural networks," ICANN/Springer LNCS**; **Wu et al. (2020), "Deep octonion networks," *Neurocomputing***; **Saoud & Al-Marzouqi (2020), "Metacognitive sedenion-valued neural network and its learning algorithm," *IEEE Access*.** Octonion/sedenion algebra used in learning systems — mostly as parameter-compression devices, not as filtering domains.
- **Cariow & Cariowa (2013), "An algorithm for fast multiplication of sedenions," *Inf. Process. Lett.* 113:324–331.** Computational-cost prior art for sedenion arithmetic.

## 6. Gamma-function-based signal transforms — **the genuine gap**

I searched hard: web searches on "reciprocal gamma function" + signal processing/pointwise transform/filtering; "log gamma"/"lgamma" + intensity transform/denoising/companding; "gamma function companding"; "factorial transform" + amplitude; and arXiv full-text searches for `"reciprocal gamma function" + denoising/filtering` and `"log-gamma transform"` (the latter returned **zero results on all of arXiv**). Findings:

- **No published work uses 1/Γ(x+1) or log Γ(x+1) as a pointwise amplitude transform before filtering.** The closest hits are unrelated: (i) "gamma transform/correction" in imaging, which is the *power law* x^γ (fully covered prior art, Poynton/standard textbooks); (ii) the **gamma filter** of Principe, de Vries & de Oliveira (1993, *IEEE TSP*), an IIR structure named after the gamma *kernel*, not a pointwise Γ transform; (iii) log-Gamma *distributions* in statistics (Bartlett & Kendall 1946 on log of variance estimates) and Bayesian data augmentation involving reciprocal gamma functions (Hamura, Irie & Sugasawa 2022) — statistical modeling, not sample-wise transforms.
- Note the skeptical corollary: nothing in the VST/homomorphic literature suggests Γ-based φ *should* work — φ is chosen to match a noise model (variance function or combination rule), and no standard noise model has 1/Γ(x+1) as its stabilizer. On [0,∞), 1/Γ(x+1) is also **non-monotonic** (increases to x≈0.4616, then decays), hence not globally invertible — a disqualifying property for the φ⁻¹∘L∘φ template that the prior art all quietly requires (log Γ(x+1) has the same non-monotonicity issue on [0, 1.46]).

## 7. Exact phrase / concept searches

- `"transformed scalar hypercomplex"` — **no exact match exists**; nearest hits are Angulo's "From scalar-valued images to hypercomplex representations… morphological operators" (2011ish, morphological ordering, unrelated mechanism) and hypercomplex wavelet-filter-bank papers.
- `"TSHF"` + filter/denoising — **no match** (the acronym appears only in unrelated contexts, e.g. hyperspectral denoising networks).
- `"hypercomplex denoising"` — matches exist but all mean *hypercomplex-valued data* denoising (quaternion wavelets, octonion dictionary learning for multispectral images), never "scalar transform then hypercomplex embed."
- `"nonlinear pre-transform filtering"` / `"filtering in transformed coordinates"` — no established phrase; concept fully covered under homomorphic/VST vocabulary.

## 8. Nonlinear-transform + linear-filter theory (deepest prior art)

- **Kolmogorov (1930) & Nagumo (1930); Aczél (1948), "On mean values," *Bull. AMS* 54:392–400.** Quasi-arithmetic (f-)means M(x)=f⁻¹(Σwᵢf(xᵢ)) axiomatized nearly a century ago; Aczél's bisymmetry characterization. Overlap: **any TSHF with a linear smoothing filter L is exactly a weighted quasi-arithmetic mean with generator φ** — TSHF's core object has a 1930 name.
- **Wadbro & Hägg (2015), "On quasi-arithmetic mean based filters and their fast evaluation for large-scale topology optimization," *Struct. Multidisc. Optim.* 52.** Explicitly treats filters as f-means with arbitrary generator φ and gives fast evaluation — a modern engineering paper doing generic-φ filtering as a *framework*.
- **Arsigny, Fillard, Pennec & Ayache (2006), "Log-Euclidean metrics for fast and simple calculus on diffusion tensors," *Magnetic Resonance in Medicine* 56(2):411–421.** Filter SPD matrices by matrix-log → Euclidean processing → matrix-exp. Overlap: φ⁻¹∘L∘φ generalized beyond scalars to a matrix manifold — strictly more general than TSHF's scalar φ.
- **Bergmann & Laus et al. (2018), "Recent advances in denoising of manifold-valued images" (arXiv:1812.08540; survey);** Laus et al. (2017) NL-means via Karcher/Fréchet means on Riemannian manifolds. Overlap: the fully intrinsic version of "filter in transformed coordinates" — averaging defined by geodesics rather than by a global chart φ; subsumes the TSHF idea whenever φ is a chart.
- (Also: Pitas & Venetsanopoulos 1990 nonlinear-mean-filter chapter, cited in §2, is the signal-processing instantiation of exactly this theory.)

## 9. Bias of nonlinear inversion

- **Duan, N. (1983), "Smearing estimate: a nonparametric retransformation method," *JASA* 78(383):605–610.** Consistent nonparametric correction for the bias of back-transforming from a transformed regression scale — the statistics community's standard fix for exactly TSHF's inversion step.
- **Mäkitalo & Foi (2011, 2013)** (§1) — the image-processing instantiation: exact unbiased inverses because algebraic inversion of the Anscombe/GAT is biased at low counts.
- **Jensen-gap literature** (e.g., the standard result that E[φ⁻¹(Y)] ≠ φ⁻¹(E[Y]) with gap ∝ curvature × variance; textbook + expository treatments; also **Xie et al.-style log-speckle mean-bias corrections in SAR**, where E[log] = ψ(L)−log L must be subtracted before exp). Overlap: a naive TSHF that applies φ⁻¹ directly inherits a bias the field has been correcting since 1983 (statistics) and 2011 (imaging); any TSHF paper that doesn't address this is behind the state of the art, not ahead of it.

---

## Summary table

| TSHF component | Closest prior art | Novelty |
|---|---|---|
| φ = log, filter, exp | Oppenheim/Schafer/Stockham 1968 homomorphic filtering; homomorphic wavelet despeckling | **None** |
| φ = square-root family (VST) | Anscombe 1948; GAT (Murtagh/Starck 1995); Mäkitalo–Foi 2011–13; Haar-Fisz 2004 | **None** |
| φ = power law | Box-Cox 1964; gamma correction; Lp/contraharmonic mean filters (Pitas–Venetsanopoulos 1990) | **None** |
| φ = signed log | μ-law companding (G.711); asinh/IHS transform (Burbidge–Magee–Robb 1988, *JASA*) | **None** |
| φ = tanh / sigmoid / atan / softsign | tanh-, exponential-, μ-law-companding for OFDM (Jiang et al. 2005 etc.); log-domain bilateral (Durand–Dorsey 2002) | **None** (as companders; softsign specifically unattested in filtering but trivially a compander variant) |
| φ = 1/Γ(x+1) or log Γ(x+1) | **Nothing found** (searched web + arXiv full text; only power-law "gamma transform", gamma-kernel IIR filters, log-Gamma distributions) | **Possibly novel as a parameter choice** — but non-monotonic on the natural signal range, hence not properly invertible, and motivated by no noise model |
| General framework φ⁻¹∘L∘φ | Kolmogorov–Nagumo 1930 / Aczél 1948 quasi-arithmetic means; nonlinear mean filters 1986–90; f-mean filters (Wadbro–Hägg 2015); log-Euclidean (Arsigny 2006); manifold denoising surveys (2018) | **None** — this is the *most* covered part |
| Quaternion embedding + filtering | Sangwine/Ell QFT 1996–2007; Took–Mandic WL-QLMS/QKF 2009–14; quaternion wavelets (Chan et al. 2008); QWNNM 2019; vector median 1990 | **None** |
| Octonion filtering | Błaszczyk–Snopek OFT 2017–2020; Alfsmann–Göckler 2007 (drawbacks) | **None**, and the non-associativity cost is already published |
| Sedenion filtering | Cariow–Cariowa 2013 (arithmetic); Saoud–Al-Marzouqi 2020 (NN) — **no sedenion filtering/FT literature found** | **Possibly novel, likely for good reason**: zero divisors break linear-systems theory (documented in Alfsmann–Göckler 2007) |
| Scalar φ-transform **combined with** hypercomplex filtering, as a named framework | **No exact combination found** (searched "quaternion homomorphic", "hypercomplex companding", quaternion+Anscombe/VST) | **Incremental at best** — a Cartesian product of two mature toolboxes; both factors are standard, and nothing found suggests the combination has new emergent theory |
| Inversion step (plain φ⁻¹) | Duan 1983; Mäkitalo–Foi exact unbiased inverses; log-speckle mean-bias corrections | **Negative novelty** — TSHF as stated uses the version the literature has already superseded |

## Aspects with NO prior art found (with search provenance)

1. **1/Γ(x+1) or log Γ(x+1) as a pointwise pre-filtering transform.** Searched: general web ("reciprocal gamma function" + signal processing/filtering; "log gamma"/"lgamma" + denoising/companding; "gamma function companding"; factorial transform) and arXiv full-text (`"reciprocal gamma function"` + denoising/filtering → 12 results, all special-function theory/Bayesian stats; `"log-gamma transform"` → **zero results**). Absence is meaningful but unflattering: these φ are non-monotonic near the origin, so they fail the invertibility requirement every prior pipeline imposes.
2. **The exact phrase/acronym "transformed scalar hypercomplex" / TSHF.** No match anywhere.
3. **Sedenion-domain signal *filtering*** (as opposed to sedenion arithmetic and sedenion NNs). No sedenion Fourier transform or sedenion filter paper found; the published position (Alfsmann–Göckler 2007) is that zero divisors make it ill-suited.
4. **An explicit paper unifying "scalar compand → hypercomplex filter → expand."** Searched "quaternion homomorphic filtering", "hypercomplex companding", quaternion+variance-stabilizing. Nothing. However, since homomorphic pre-transforms and quaternion filters are each routine, this reads as an unclaimed *combination*, not an unclaimed *idea*.

**Caveats on verification:** I relied on search snippets and abstracts; I could not access full texts behind IEEE/Elsevier paywalls (e.g., the Alfsmann–Göckler PDF, Pitas–Venetsanopoulos book chapters), so page-level claims there are from secondary descriptions. arXiv full-text search covers only arXiv; a negative there does not rule out non-arXiv venues, though the general web searches also came up empty. Google Scholar was not directly queryable from this environment.

**Skeptical conclusion:** TSHF is best described as re-instantiating the homomorphic/VST/quasi-arithmetic-mean template with (i) an eccentric and mathematically problematic pair of new φ choices (Γ-based, non-invertible on part of the range, motivated by no noise model), and (ii) an optional hypercomplex embedding that is independently standard. The only defensible novelty claims are narrow parameter-level ones ("nobody has used log Γ(x+1) as a compander"; "nobody filters in sedenions"), and for each of those the literature already contains the reason nobody has: φ must be a monotone bijection matched to the noise statistics, φ⁻¹ must be bias-corrected (Duan 1983; Mäkitalo–Foi 2011), and algebras past the quaternions surrender the associativity/division structure that linear filtering theory needs (Alfsmann–Göckler 2007; Błaszczyk–Snopek 2018–2020). A new-filter-family claim would require demonstrating a noise model or signal class for which Γ-based φ is the *correct* stabilizer/homomorphism — nothing in the prior art or the proposal as stated supplies one.

Sources (key URLs used): [Mäkitalo–Foi optimal inversion (IEEE TIP)](https://dl.acm.org/doi/10.1109/TIP.2010.2056693) · [closed-form unbiased inverse (PubMed)](https://pubmed.ncbi.nlm.nih.gov/21356615/) · [Foi invansc page](https://webpages.tuni.fi/foi/invansc/index.html) · [Anscombe transform (Wikipedia)](https://en.wikipedia.org/wiki/Anscombe_transform) · [GAT optimal inversion (IEEE)](https://ieeexplore.ieee.org/document/6212354/) · [Homomorphic filtering (Wikipedia)](https://en.wikipedia.org/wiki/Homomorphic_filtering) · [homomorphic wavelet ultrasound (PubMed)](https://pubmed.ncbi.nlm.nih.gov/16126580/) · [μ-law (Wikipedia)](https://en.wikipedia.org/wiki/%CE%9C-law_algorithm) · [Durand–Dorsey 2002](https://history.siggraph.org/learning/fast-bilateral-filtering-for-the-display-of-high-dynamic-range-images-by-durand-and-dorsey/) · [Ell–Sangwine hypercomplex FT (IEEE TIP)](https://dl.acm.org/doi/abs/10.1109/TIP.2006.884955) · [quaternion widely linear filter](https://www.researchgate.net/publication/224131196_A_Quaternion_Widely_Linear_Adaptive_Filter) · [quaternion Kalman filters (PubMed)](https://pubmed.ncbi.nlm.nih.gov/24807449/) · [Chan–Choi–Baraniuk quaternion wavelets](https://www.semanticscholar.org/paper/Coherent-image-processing-using-quaternion-wavelets-Chan-Choi/d83167950ea306c6f058b44a7405a46c2ddccd72) · [Błaszczyk OFT generalization](https://arxiv.org/abs/1905.12631) · [discrete OFT](https://link.springer.com/article/10.1007/s40314-020-01373-7) · [Alfsmann–Göckler EUSIPCO 2007](https://www.semanticscholar.org/paper/Hypercomplex-algebras-in-digital-signal-processing:-Alfsmann-G%C3%B6ckler/8e7b49cb759711182f11fcfb28f4a7b92d307a3d) · [Deep Octonion Networks](https://arxiv.org/abs/1903.08478) · [sedenion NN](https://www.researchgate.net/publication/343255151_Metacognitive_Sedenion-Valued_Neural_Network_and_Its_Learning_Algorithm) · [Cariow sedenion multiplication](https://www.sciencedirect.com/science/article/abs/pii/S0020019013000653) · [Sedenion zero divisors (Wikipedia)](https://en.wikipedia.org/wiki/Sedenion) · [quasi-arithmetic mean (Wikipedia)](https://en.wikipedia.org/wiki/Quasi-arithmetic_mean) · [f-mean filters, topology optimization](https://link.springer.com/article/10.1007/s00158-015-1273-5) · [Aczél characterization lineage](https://arxiv.org/abs/1501.02857) · [Arsigny log-Euclidean (MRM 2006)](https://onlinelibrary.wiley.com/doi/10.1002/mrm.20965) · [manifold-valued denoising survey](https://arxiv.org/pdf/1812.08540) · [Duan smearing (JASA 1983)](https://www.tandfonline.com/doi/abs/10.1080/01621459.1983.10478017) · [Jensen gap exposition](https://medium.com/data-science/mind-the-jensen-gap-c54e0eb9e1b7) · [Burbidge–Magee–Robb IHS (JASA 1988)](https://www.tandfonline.com/doi/abs/10.1080/01621459.1988.10478575) · [exponential companding OFDM](https://ieeexplore.ieee.org/document/1433083/) · [companding survey](https://www.tandfonline.com/doi/full/10.1080/24751839.2019.1606878) · [Haar-Fisz (Fryzlewicz–Nason)](http://stats.lse.ac.uk/fryzlewicz/Poisson/jcgs.pdf) · [QWNNM denoising](https://www.sciencedirect.com/science/article/abs/pii/S0925231218314887) · [vector median filters context](https://link.springer.com/chapter/10.1007/978-3-662-04186-4_2) · [Pitas–Venetsanopoulos book](https://link.springer.com/book/10.1007/978-1-4757-6017-0) · [reciprocal gamma function (Wikipedia)](https://en.wikipedia.org/wiki/Reciprocal_gamma_function) · [Freeman-Tukey](https://www.statsref.com/HTML/freeman-tukey.html) · [Box-Cox review](https://projecteuclid.org/journals/statistical-science/volume-36/issue-2/The-BoxCox-Transformation-Review-and-Extensions/10.1214/20-STS778.pdf)

---

## 5. Actually novel aspects

After extensive research, the only elements without identified precedent are:

1. **The use of `1/Γ(x+1)` or `ln Γ(x+1)` as a pointwise denoising
   transform** — no precedent found. Our measurements show *why*: the
   first is non-injective, the second non-monotonic, ill-conditioned, and
   anti-stabilizing (E1, E2). The absence of precedent here reflects an absence of
   merit, not an opportunity.
2. **The term "Transformed-Scalar Hypercomplex Filters" and the marketing
   assembly of the two ideas** — the assembly has no precedent *as a named
   family*, but Proposition 1 (§1.1) shows it decomposes into two independent
   questions, each classic.

No new mathematical property, no new empirical gain emerged from the
experiments.

## 6. Weaknesses

1. **Separability** (Prop. 1): the core of the proposal factorizes into two
   independent, already-studied ideas; the assembly adds nothing by itself.
2. **Self-neutralization on rank filters** (Prop. 2).
3. On additive Gaussian noise — the most common case — the pipeline can only
   lose (E1, E3): Gauss-Markov cannot be improved by a pointwise change of
   coordinates followed by the same filter.
4. The naive inverse is biased (E4); correcting the bias requires exactly the
   machinery (exact unbiased inverse) published by the prior art.
5. The most "original" proposed transforms (Γ family) are mathematically
   disqualified (E2).
6. Octonions/sedenions: the loss of associativity removes the faithful matrix
   representation, so linear-systems theory (transfer functions, z-transform)
   does not carry over; no filtering benefit demonstrated in the literature,
   and the SIMD cost is already documented in SciRust (#513/#517). Our E5b
   shows that even a joint operator can *lose* to per-channel processing.

## 7. Potential applications (of the viable subset)

- **Photon counting / low-flux imaging** (Poisson): Anscombe + Gaussian
  denoiser + corrected inverse — standard pipeline, relevant for
  `scirust-vision`.
- **Radar/ultrasound speckle, industrial multiplicative noise**: signed log
  (homomorphic filtering) + bias correction.
- **Level-dependent-noise sensors** (photodiodes, gauges): Box-Cox/root.
- **Correlated multichannel** (color, quaternionic IMU): *genuine* quaternion
  filters (QFT, widely-linear) — a distinct path from TSHF, already mapped.

## 8. Proposed architecture (viable subset only)

```
x (signal-dependent noise, identified model)
  → matched VST (anscombe | boxcox(λ) | signed_log)
  → any existing Gaussian denoiser of denoise::*
  → BIAS-CORRECTED inverse (exact-unbiased for Anscombe; smearing for log)
  → x̂
```

Suggested module: `denoise::vst` — three (φ, corrected φ⁻¹) pairs, a selector by
`NoiseProfile` (the classifier already detects level dependence via
per-band variance), and integration into `denoise_auto` as a conditional
pre/post-step. **No** hypercomplex embedding in this module (Prop. 1).

## 9. Experimental protocol (for any follow-up)

1. Synthetic fixtures with controlled model: low-count Poisson (λ ∈ 1–20),
   multiplicative 10–40 %, mixed Poisson-Gaussian (the Starck/Murtagh case).
2. Comparisons: identity vs naive-VST vs corrected-VST, on MA/wavelets/BM3D-1D
   (`collab1d`), metrics §10.
3. Regime sweep: plot VST gain vs intensity of the signal dependence
   (our E3 shows a gain ≈ zero in the mild ×2 regime; the literature locates it
   in the strong regime — verify the crossover threshold).
4. Recommended public datasets (identification only, no download):
   images — BSD68/Set12, FMD (fluorescence, true low-flux Poisson), SIDD;
   audio — VoiceBank-DEMAND; hyperspectral — Indian Pines, Pavia; IMU —
   UCI-HAR; radar — SAR speckle signals (Sentinel-1 patches); medical —
   MIT-BIH (ECG), CHB-MIT (EEG); industrial vibrations — CWRU Bearing,
   NASA IMS. Each covers a distinct noise model of the protocol.

## 10. Validation methodology

Metrics: SNR/PSNR/RMSE/MAE (in the *original* coordinates), SSIM (2-D),
mean bias (the retransformation defect, cf. E4), energy conservation
`‖x̂‖/‖s‖`, per-channel norm distortion, edge preservation (E6),
inter-channel correlation preservation (E5b), φ∘φ⁻¹ round-trip error (E2),
bit-for-bit determinism (SciRust convention), time/memory. Success thresholds
to set *before* measurement; any gain < 0.5 dB is declared null.

## 11. Risks

- **Scientific risk**: reclassifying known work as novelty — mitigated by this
  report (explicit citations, §2–4).
- **Engineering risk**: the bias correction depends on the noise model; a bad
  match (log on additive) *degrades* (E1, E3) — the selector must be
  conservative (default = identity).
- **Numerical risk**: domains (Anscombe x ≥ −3/8; log x > 0) → documented
  clamps, never silent.
- **Scope risk**: the octonion/sedenion temptation — no result justifies it;
  engaging it would consume SIMD effort with no filtering benefit.

## 12. Recommendations

1. **Reject** the TSHF family as a "new family of filters"; do not implement
   any Γ pipeline or octonion/sedenion filtering embedding.
2. **Implement the viable, honestly named subset**: module
   `denoise::vst` (Anscombe + exact unbiased inverse, Box-Cox/log + smearing,
   signed root), wired to the classifier — Phase 1 of the plan below.
3. The saturating transforms (tanh/sigmoid/softsign/atan): reserve for uses
   *without reconstruction* (robust features, display compression) — never as
   the φ of an inverted pipeline (E2/E4/E6).
4. The legitimate quaternionic path (color QFT, widely-linear, vector median
   for *desynchronized* impulses) is a separate workstream, to evaluate on its
   own fixtures — our E5b shows it does not win by default.

### Roadmap (viable subset)

- **Phase 1 — pure scalar prototype**: `denoise::vst` (3 corrected φ/φ⁻¹ pairs,
  per-profile selector, Poisson/multiplicative oracles where corrected-VST >
  identity by at least 1 dB in the strong regime — that is the acceptance
  criterion).
- **Phase 2 — quaternion**: only the operators that genuinely couple the
  channels (vector median, widely-linear Wiener) on multichannel fixtures;
  gate: beat the per-channel on ≥ 2 realistic fixtures.
- **Phase 3 — octonion**: *conditional* on a literature result or a Phase 2
  experiment demonstrating a need for 8 coupled channels; otherwise abandon.
- **Phase 4 — SIMD**: vectorize the φ/φ⁻¹ (elementary functions) only if
  Phase 1 is adopted and profiled as expensive.
- **Phase 5 — GPU**: not justified by SciRust's current volumes; re-evaluate
  with real image/hyperspectral workloads.

## 13. Does the concept deserve an implementation in SciRust?

**As TSHF: no.** The experiments reveal no regime where the generic pipeline
beats the existing state, its original components (Γ) are mathematically
disqualified, and everything that works already has a name in the literature.

**As a targeted VST module: yes** — the scope of Phase 1 above, with a
quantified acceptance criterion and honest naming (Anscombe/Box-Cox, not
"TSHF").

---

*Report produced within the SciRust research program. Experiments:
`scirust-signal/examples/tshf_experiments.rs` (E1–E6, deterministic). Method:
falsification first — each experimental block was designed to be able to
contradict the hypothesis, and several did.*

---

## Addendum — execution of the recommendations (2026-07-16, same day)

Status of each §12 item and of the roadmap, with the acceptance measurements
obtained:

- **Reco 1 (rejection of TSHF/Γ/octonion-sedenion)**: respected — nothing of
  the sort was implemented.
- **Reco 2 / Phase 1 — executed**: module `denoise::vst` (Anscombe + the
  Mäkitalo-Foi exact unbiased inverse in closed form; signed log + Duan
  smearing; signed root; manual Box-Cox(λ)), conservative selector
  `detect_noise_model` (default = identity), integration as a conditional
  pre/post-step of `denoise_auto`. **§12 gates passed**: Poisson λ∈[1,12]:
  +5.02 dB vs identity (criterion ≥ +1 dB), corrected inverse > naive by
  +3.90 dB; strong 30 % multiplicative: +4.88 dB; mild regime: +0.04 dB (the
  ≈ zero gain predicted in §9.3 — and no loss); residual bias 0.015 (naive:
  0.268 ≈ the Jensen gap of 0.25 predicted). Execution note: the internal
  denoiser retained is `stft_wiener_auto` — the global-threshold wavelet,
  although the "classic" VST beneficiary, *lost* ~1 dB after stabilization on
  level-correlated signals (the raw MAD calibration acted as an accidentally
  adaptive threshold); consistent with the §11 principle "never degrade".
- **Reco 3 — executed**: `denoise::compand` (`soft_clip`, `soft_clip_robust`;
  tanh/atan/softsign), no inverse by design.
- **Reco 4 / Phase 2 — executed, split verdict**: module
  `denoise::multichannel`, gate "beat the per-channel on ≥ 2 fixtures":
  `wiener_spatial` (joint spatial Wiener ≡ real widely-linear) **passes** —
  +2.48 dB (4 correlated channels) and +3.67 dB (rank-1 stereo) against its
  diagonal restriction; `vector_median` **fails** (0/2) — −1.81 dB on
  synchronized impulses (E5b reproduced) and −2.02 dB on *desynchronized*
  impulses: the §12.4 conjecture ("vector median for desynchronized
  impulses") is **falsified** — the vector median returns an observed vector
  whose entire background noise survives, whereas the scalar median averages
  that noise. Its inter-channel correlation preservation is also inferior
  (error 4.4e-3 vs 2.8e-3). Kept as a reference implementation, verdict in
  the doc; figures reproducible via `phase2_gate_report()`.
- **Phase 3 (octonion) — not triggered**: the condition ("a demonstrated need
  for 8 coupled channels") is not met; Phase 2 on the contrary falsified the
  rank joint operator.
- **Phase 4 (SIMD of φ/φ⁻¹) — not triggered**: φ/φ⁻¹ are O(n) passes of
  elementary functions, negligible before the internal denoisers' cost.
- **Phase 5 (GPU) — not triggered**: volumes unchanged since the report.

## Addendum 2 — §9 protocol executed, GAT and 2-D extensions (2026-07-16)

The §9 experimental protocol is now replayable
(`cargo run --release -p scirust-signal --example vst_protocol`, deterministic
blocks P1–P5) and its open questions are **measured**:

- **§9.3, crossover threshold (P4)**: at ×10 level dynamic range, the VST
  gain is already material (≥ +0.5 dB) at 2 % multiplicative noise (the
  crossover is ≤ 2 %); at 30 % noise, the crossover in *level dynamic range*
  is at **≈ ×3** — and at ×2 the VST is a material loss of −0.77 dB, which
  *refines and strengthens* the §9.3 "≈ zero gain in the mild ×2 regime".
  Consequence coded: the dynamic-range gate of the `detect_noise_model`
  selector is tightened from ×2 to **×3** (constant `DETECT_MIN_RANGE`,
  documented by this measurement).
- **Carrier regime (P5, new limitation)**: the Anscombe gain collapses from
  +5.17 dB (carrier at 3 cycles/4096) to **−0.93 dB (40 cycles)** — a
  pointwise φ does not commute with the spectrum: the root converts a fast
  carrier into a stack of harmonics that the internal linear shrinkage crops.
  Documented in the `vst` module doc ("Known limitation: fast carriers")
  and pinned by a test. The VST targets *slow* intensities.
- **GAT (§9.1c, the Starck-Murtagh case)**: `VstKind::Gat { gain, sigma }` with
  the exact unbiased inverse in closed form (Mäkitalo-Foi 2013): +1.54 to
  +2.87 dB depending on calibration (worst case: read-dominated (1.3, 1.5));
  gain=1, σ=0 reduces exactly to Anscombe. Notable scalability fact: the
  constant-σ/gain calibrations are exact rescalings (the GAT normalizes the
  gain, each stage is scale-equivariant).
- **2-D transposition (`scirust_vision::denoise`)**: `vst_denoise2d` /
  `vst_denoise2d_auto`. Three 1-D results transpose as-is: 2-D VisuShrink
  loses under stabilization (−0.6 dB — raw MAD calibration accidentally
  adaptive); the 2-D median is *invariant* bit for bit
  (report Prop. 2, confirmed empirically); the best measured partner is
  **2-D NLM** (+5.4 dB Poisson, +3.0 dB GAT — its patch distances with
  global `h` assume exactly the homoscedasticity that the VST restores). The
  1-D detector works on smooth images in line segments, with tighter
  correlation (the log-level dispersion is the limiting factor); misses fall
  back to Identity (safe).
- **Wavelet arm (P2)**: VisuShrink benefits from stabilization on *no*
  tested fraction (−4.4 to −1.2 dB) — the choice of `stft_wiener_auto` (1-D)
  / `nlm2d` (2-D) as internal partners is confirmed.
