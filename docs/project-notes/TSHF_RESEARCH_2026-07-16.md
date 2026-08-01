# Programme de recherche — Filtres Hypercomplexes à Scalaires Transformés (TSHF)

**Date** : 2026-07-16 · **Statut** : investigation terminée, verdict rendu
**Reproductibilité** : toutes les mesures de ce rapport proviennent de
`cargo run -p scirust-signal --example tshf_experiments` (déterministe, graine fixe).

## Résumé exécutif

La proposition TSHF — transformer ponctuellement le scalaire (`φ(x)` avec
`φ ∈ {1/Γ(x+1), ln Γ(x+1), log signé, loi de puissance, tanh, sigmoïde, atan,
softsign, …}`), plonger dans une algèbre hypercomplexe (quaternion/octonion/sédénion),
filtrer, puis inverser — a été soumise à une analyse mathématique, à six blocs
d'expériences numériques de falsification et à une revue de littérature extensive.

**Verdict : la proposition, en tant que « nouvelle famille de filtres », n'est pas
scientifiquement défendable.** Ses deux composantes sont séparables et chacune est,
soit déjà couverte par une littérature abondante (stabilisation de variance,
filtrage homomorphe, companding, moyennes quasi-arithmétiques), soit contredite par
nos propres mesures (transformées Γ, plongements de dimension supérieure sans
opérateur couplant). Un **sous-ensemble étroit et déjà connu** mérite en revanche
une implémentation dans SciRust : les transformées stabilisatrices de variance
(Anscombe, Box-Cox/log signé, racine signée) **avec inverse à biais corrigé**, pour
le bruit dépendant du signal — voir §12–13.

Les points saillants, chiffres à l'appui :

- `1/Γ(x+1)` est **non injective** (φ(0) = φ(1) = 1) : rejet immédiat — aucune
  reconstruction possible (E2).
- `ln Γ(x+1)` est non monotone sous x ≈ 0,4616 et son inversion numérique amplifie
  le bruit ×27 : rejet pour un pipeline avec reconstruction (E2).
- Pour du bruit **additif gaussien**, l'identité bat *toutes* les transformées
  testées, sur les trois filtres testés (E3) — conforme à la théorie (le bruit est
  déjà stationnaire ; toute φ non affine le rend dépendant du niveau, E1).
- La **médiane est invariante** sous toute φ monotone : la colonne médiane de E3
  est constante à 10⁻¹² près — le pipeline TSHF y est mathématiquement un no-op.
- Les transformées saturantes (tanh/sigmoïde) infligent un **biais de
  retransformation** (Jensen) mesuré jusqu'à −0,13 sur un niveau de 2 (−6,5 %) et
  une amplification du bruit à l'inversion ×22–×101 (E2, E4) ; tanh + ondelettes
  détruit le signal (10,4 dB < 12,6 dB brut, E3).
- Le plongement hypercomplexe est **orthogonal** à la question : tout filtre
  R-linéaire appliqué composante par composante est identique au filtrage par canal
  (identité algébrique, E5a) ; l'ordre transformation/plongement ne compte que si la
  transformée couple les coordonnées (E5c) ; et sur notre fixture d'impulsions
  corrélées, le médian vectoriel (joint) a *perdu* contre le médian par canal
  (12,7 dB vs 14,6 dB, E5b) — la dimension supérieure n'aide pas par défaut.

---

## 1. Fondations mathématiques

### 1.1 Le pipeline étudié

Deux architectures ont été analysées :

```
(A)  x → φ(x) → plongement H → filtre L → φ⁻¹ → x̂
(B)  x → plongement H → φ (par composante ou couplée) → filtre L → inverses → x̂
```

**Proposition 1 (séparabilité).** Si φ agit composante par composante et si L est
R-linéaire appliqué composante par composante, alors (A) ≡ (B) ≡ filtrage par canal
de φ(x) : le plongement hypercomplexe est transparent. *Preuve* : un plongement
coordonnées-vers-coordonnées est une permutation de l'ordre des données ; un
opérateur R-linéaire composante par composante commute avec elle. Vérifié
numériquement (E5a, E5c : identité exacte). L'ordre ne devient significatif que si
φ couple les coordonnées (ex. `v ↦ v·tanh(‖v‖)/‖v‖`, E5c : 0,426 ≠ 0,462) ou si L
utilise le produit hypercomplexe.

**Corollaire.** La « nouveauté » éventuelle du TSHF ne peut PAS venir de la
combinaison φ + plongement en soi ; elle doit venir soit de φ (question classique
de stabilisation de variance, §2), soit d'opérateurs exploitant réellement le
produit de l'algèbre (QFT, filtres widely-linear — littérature quaternionique
établie, §3).

**Proposition 2 (invariance de la médiane).** Pour toute φ strictement monotone,
`median(φ(x_i)) = φ(median(x_i))`, donc `φ⁻¹ ∘ median ∘ φ = median` : le pipeline
TSHF avec un filtre de rang est l'identité du filtre. Confirmé par E3 (colonne
médiane constante sur les 8 transformées). Toute une classe de filtres
(rang/ordre) est donc *hors sujet* pour le TSHF.

**Proposition 3 (moyennes quasi-arithmétiques).** Pour L = moyenne mobile,
`φ⁻¹(MA(φ(x)))` est la moyenne quasi-arithmétique de Kolmogorov-Nagumo de
générateur φ (log → moyenne géométrique, x⁻¹ → harmonique…). Le pipeline TSHF-MA
est donc un objet mathématique connu depuis 1930, pas une construction nouvelle.

### 1.2 Effet sur les statistiques du bruit (E1)

Développement au premier ordre : pour x = s + n, `φ(x) − φ(s) ≈ φ′(s)·n`, donc
`σ_φ(s) ≈ |φ′(s)|·σ(s)`. Trois régimes mesurés (σ après transformation, niveaux
s = 0,6 / 1,2 / 1,8 / 2,4) :

| φ | additif σ=0,3 | multiplicatif 0,3·s | Poisson-like 0,3·√s |
|---|---|---|---|
| identité | 0,300 / 0,300 / 0,300 / 0,300 (plat ✓) | 0,18→0,72 (×4) | 0,23→0,46 (×2) |
| log signé | 0,195→0,089 (dé-stabilise) | 0,114→0,226 (×2, aplati) | — |
| racine signée | 0,252→0,098 (dé-stabilise) | — | 0,173→0,153 (**plat ✓**) |
| Anscombe | 0,306→0,181 | 0,185→0,447 | 0,240→0,283 (**quasi plat ✓**) |
| tanh | 0,214→0,013 (écrase) | — | — |
| ln Γ(x+1) | 0,065→0,320 (**amplifie** la dépendance) | 0,029→0,759 (pire que l'identité) | — |

Réponses aux questions scientifiques 1–2 : **oui**, une transformée scalaire
modifie les statistiques du bruit — mais dans les deux sens. Sur bruit additif
(déjà stationnaire), toute φ non affine *crée* une dépendance au niveau, ce qui
dégrade les filtres à seuil global. La stabilisation n'apporte quelque chose que si
le bruit d'origine est dépendant du signal ET si φ est *appariée au modèle de
bruit* (racine ↔ Poisson, log ↔ multiplicatif) — ce qui est la définition exacte
des transformées stabilisatrices de variance classiques. `ln Γ` fait l'inverse de
ce qu'on attend d'une VST sur tous les modèles testés.

### 1.3 Inversibilité, domaines, branches, conditionnement (E2)

| φ | injective ? | domaine | max \|dφ⁻¹/dy\| sur x∈[−3,3] | erreur aller-retour |
|---|---|---|---|---|
| identité | oui | ℝ | 1,0 | 0 |
| log signé | oui | ℝ | 4,0 | 7·10⁻¹⁶ |
| racine signée | oui | ℝ | 3,5 | 4·10⁻¹⁶ |
| Anscombe | oui | x ≥ −0,375 | 1,8 | 4·10⁻¹⁶ |
| atan | oui | ℝ | 10,0 | 9·10⁻¹⁶ |
| softsign | oui | ℝ | 16,0 | 9·10⁻¹⁶ |
| sigmoïde | oui | ℝ | 22,1 | 3·10⁻¹⁵ |
| tanh | oui | ℝ | **101,4** | 2·10⁻¹⁴ |
| ln Γ(x+1) | **non** (monotone seulement x > 0,4616) | x > −1 | 27,4 (+ inversion Newton) | 4·10⁻¹⁴ |
| 1/Γ(x+1) | **NON** : φ(0) = φ(1) = 1, max en x ≈ 0,4616 | — | — | reconstruction impossible |

Le facteur `max |dφ⁻¹/dy|` borne l'amplification du bruit résiduel à la
reconstruction (comportement de Lipschitz de l'inverse). Les transformées
saturantes concentrent cette amplification exactement là où vit le signal fort —
le pire endroit. Questions 5–6 : **oui**, l'instabilité numérique et les artefacts
sont amplifiés, et de façon structurelle (pas corrigeable par implémentation).

### 1.4 Biais de retransformation (E4)

`E[φ⁻¹(L(φ(x)))] ≠ s` pour φ non affine (inégalité de Jensen). Mesuré sur signal
plat s = 2, bruit g = 0,4, MA(9) : identité −0,001 ; Anscombe −0,017 ;
racine signée −0,020 ; log signé −0,026 ; softsign −0,050 ; sigmoïde −0,054 ;
atan −0,060 ; **tanh −0,131** (−6,5 % du niveau) ; ln Γ +0,030. C'est le biais de
retransformation connu de la littérature (smearing de Duan, inverse exact non
biaisé de Mäkitalo-Foi, §2) : l'inverse algébrique naïf de l'énoncé TSHF est
précisément la variante que la littérature a montrée défectueuse.

---

## 2–4. Revue de littérature, travaux existants, concepts déjà publiés

*(Section établie par recherche bibliographique extensive — voir les références in
fine ; chaque composante du TSHF est mise en regard de l'art antérieur.)*

*Revue conduite par recherche web extensive (agent dédié) ; conservée en anglais, la langue des sources. Les recherches négatives sont documentées avec leur périmètre exact, pour que l'absence de résultat soit une donnée et non une supposition.*

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

## 5. Aspects réellement nouveaux

Après recherche extensive, les seuls éléments sans antécédent identifié sont :

1. **L'usage de `1/Γ(x+1)` ou `ln Γ(x+1)` comme transformée ponctuelle de
   débruitage** — aucun antécédent trouvé. Nos mesures montrent *pourquoi* : la
   première est non injective, la seconde non monotone, mal conditionnée, et
   anti-stabilisatrice (E1, E2). L'absence d'antécédent reflète ici une absence de
   mérite, pas une opportunité.
2. **Le terme « Transformed-Scalar Hypercomplex Filters » et l'assemblage
   marketing des deux idées** — l'assemblage n'a pas d'antécédent *en tant que
   famille nommée*, mais la Proposition 1 (§1.1) montre qu'il se décompose en deux
   questions indépendantes, chacune classique.

Aucune propriété mathématique nouvelle, aucun gain empirique nouveau n'a émergé
des expériences.

## 6. Faiblesses

1. **Séparabilité** (Prop. 1) : le cœur de la proposition se factorise en deux
   idées indépendantes déjà étudiées ; l'assemblage n'ajoute rien par lui-même.
2. **Auto-neutralisation sur les filtres de rang** (Prop. 2).
3. Sur bruit additif gaussien — le cas le plus courant — le pipeline ne peut que
   perdre (E1, E3) : Gauss-Markov ne se laisse pas améliorer par un changement de
   coordonnées ponctuel suivi du même filtre.
4. L'inverse naïf est biaisé (E4) ; corriger le biais exige exactement la
   machinerie (inverse exact non biaisé) publiée par l'art antérieur.
5. Les transformées proposées les plus « originales » (famille Γ) sont
   mathématiquement disqualifiées (E2).
6. Octonions/sédénions : la perte d'associativité supprime la représentation
   matricielle fidèle, donc la théorie des systèmes linéaires (fonctions de
   transfert, z-transformée) ne s'y transporte pas ; aucun bénéfice de filtrage
   démontré dans la littérature, et le coût SIMD est déjà documenté dans SciRust
   (#513/#517). Notre E5b montre qu'un opérateur joint peut même *perdre* contre
   le traitement par canal.

## 7. Applications potentielles (du sous-ensemble viable)

- **Comptage photonique / imagerie faible flux** (Poisson) : Anscombe + débruiteur
  gaussien + inverse corrigé — pipeline standard, pertinent pour `scirust-vision`.
- **Speckle radar/échographie, bruit multiplicatif industriel** : log signé
  (filtrage homomorphe) + correction de biais.
- **Capteurs à bruit dépendant du niveau** (photodiodes, jauges) : Box-Cox/racine.
- **Multicanal corrélé** (couleur, IMU quaternionique) : filtres quaternioniques
  *authentiques* (QFT, widely-linear) — voie distincte du TSHF, déjà balisée.

## 8. Architecture proposée (sous-ensemble viable uniquement)

```
x (bruit dépendant du signal, modèle identifié)
  → VST appariée (anscombe | boxcox(λ) | signed_log)
  → n'importe quel débruiteur gaussien existant de denoise::*
  → inverse À BIAIS CORRIGÉ (exact-unbiased pour Anscombe ; smearing pour log)
  → x̂
```

Module suggéré : `denoise::vst` — trois paires (φ, φ⁻¹ corrigé), un sélecteur par
`NoiseProfile` (le classificateur détecte déjà la dépendance au niveau via la
variance par bande), et l'intégration à `denoise_auto` comme pré/post-étape
conditionnelle. **Pas** de plongement hypercomplexe dans ce module (Prop. 1).

## 9. Protocole expérimental (pour la suite éventuelle)

1. Fixtures synthétiques à modèle contrôlé : Poisson à faible comptage (λ ∈ 1–20),
   multiplicatif 10–40 %, mixte Poisson-gaussien (le cas Starck/Murtagh).
2. Comparaisons : identité vs VST-naïve vs VST-corrigée, sur MA/ondelettes/BM3D-1D
   (`collab1d`), métriques §10.
3. Balayage du régime : tracer le gain VST vs intensité de la dépendance au signal
   (notre E3 montre un gain ≈ nul en régime doux ×2 ; la littérature le situe en
   régime fort — vérifier le seuil de croisement).
4. Datasets publics recommandés (identification seule, pas de téléchargement) :
   images — BSD68/Set12, FMD (fluorescence, vrai Poisson faible flux), SIDD ;
   audio — VoiceBank-DEMAND ; hyperspectral — Indian Pines, Pavia ; IMU —
   UCI-HAR ; radar — signaux SAR speckle (Sentinel-1 patchs) ; médical —
   MIT-BIH (ECG), CHB-MIT (EEG) ; vibrations industrielles — CWRU Bearing,
   NASA IMS. Chacun couvre un modèle de bruit distinct du protocole.

## 10. Méthodologie de validation

Métriques : SNR/PSNR/RMSE/MAE (dans les coordonnées *d'origine*), SSIM (2-D),
biais moyen (le défaut de retransformation, cf. E4), conservation d'énergie
`‖x̂‖/‖s‖`, distorsion de norme par canal, préservation d'arêtes (E6),
préservation de corrélation inter-canaux (E5b), erreur aller-retour φ∘φ⁻¹ (E2),
déterminisme bit-à-bit (convention SciRust), temps/mémoire. Seuils de succès à
fixer *avant* mesure ; tout gain < 0,5 dB est déclaré nul.

## 11. Risques

- **Risque scientifique** : requalifier du travail connu en nouveauté — mitigé par
  ce rapport (citations explicites, §2–4).
- **Risque d'ingénierie** : la correction de biais dépend du modèle de bruit ; un
  mauvais appariement (log sur additif) *dégrade* (E1, E3) — le sélecteur doit
  être conservateur (défaut = identité).
- **Risque numérique** : domaines (Anscombe x ≥ −3/8 ; log x > 0) → clamps
  documentés, jamais silencieux.
- **Risque de périmètre** : la tentation octonion/sédénion — aucun résultat ne la
  justifie ; s'y engager consommerait l'effort SIMD sans bénéfice de filtrage.

## 12. Recommandations

1. **Rejeter** la famille TSHF comme « nouvelle famille de filtres » ; ne pas
   implémenter de pipeline Γ ni de plongement octonion/sédénion de filtrage.
2. **Implémenter le sous-ensemble viable et honnêtement nommé** : module
   `denoise::vst` (Anscombe + inverse exact non biaisé, Box-Cox/log + smearing,
   racine signée), branché au classificateur — Phase 1 du plan ci-dessous.
3. Les transformées saturantes (tanh/sigmoïde/softsign/atan) : à réserver aux
   usages *sans reconstruction* (features robustes, compression d'affichage) —
   jamais comme φ de pipeline inversé (E2/E4/E6).
4. La voie quaternionique légitime (QFT couleur, widely-linear, médian vectoriel
   pour impulsions *désynchronisées*) est un chantier séparé, à évaluer sur ses
   propres fixtures — notre E5b montre qu'elle ne gagne pas par défaut.

### Feuille de route (sous-ensemble viable)

- **Phase 1 — prototype scalaire pur** : `denoise::vst` (3 paires φ/φ⁻¹ corrigées,
  sélecteur par profil, oracles Poisson/multiplicatif où VST-corrigée > identité
  d'au moins 1 dB en régime fort — c'est le critère d'acceptation).
- **Phase 2 — quaternion** : uniquement les opérateurs qui couplent réellement les
  canaux (médian vectoriel, Wiener widely-linear) sur fixtures multicanal ; gate :
  battre le par-canal sur ≥ 2 fixtures réalistes.
- **Phase 3 — octonion** : *conditionnelle* à un résultat de littérature ou une
  expérience Phase 2 démontrant un besoin à 8 canaux couplés ; sinon abandonner.
- **Phase 4 — SIMD** : vectoriser les φ/φ⁻¹ (fonctions élémentaires) seulement si
  Phase 1 est adoptée et profilée comme coûteuse.
- **Phase 5 — GPU** : non justifiée par les volumes actuels de SciRust ; réévaluer
  avec des charges image/hyperspectrales réelles.

## 13. Le concept mérite-t-il une implémentation dans SciRust ?

**En tant que TSHF : non.** Les expériences ne dégagent aucun régime où le
pipeline générique bat l'existant, ses composantes originales (Γ) sont
mathématiquement disqualifiées, et tout ce qui fonctionne porte déjà un nom dans
la littérature.

**En tant que module VST ciblé : oui** — périmètre de la Phase 1 ci-dessus, avec
critère d'acceptation chiffré et nommage honnête (Anscombe/Box-Cox, pas « TSHF »).

---

*Rapport produit dans le cadre du programme de recherche SciRust. Expériences :
`scirust-signal/examples/tshf_experiments.rs` (E1–E6, déterministes). Méthode :
falsification d'abord — chaque bloc expérimental a été conçu pour pouvoir
contredire l'hypothèse, et plusieurs l'ont fait.*

---

## Addendum — exécution des recommandations (2026-07-16, même jour)

Statut de chaque élément du §12 et de la feuille de route, avec les mesures
d'acceptation obtenues :

- **Reco 1 (rejet TSHF/Γ/octonion-sédénion)** : respectée — rien de tel n'a été
  implémenté.
- **Reco 2 / Phase 1 — exécutée** : module `denoise::vst` (Anscombe + inverse
  exact non biaisé de Mäkitalo-Foi en forme close ; log signé + smearing de Duan ;
  racine signée ; Box-Cox(λ) manuel), sélecteur conservateur `detect_noise_model`
  (défaut = identité), intégration en pré/post-étape conditionnelle de
  `denoise_auto`. **Portes du §12 franchies** : Poisson λ∈[1,12] : +5,02 dB vs
  identité (critère ≥ +1 dB), inverse corrigé > naïf de +3,90 dB ; multiplicatif
  30 % fort : +4,88 dB ; régime doux : +0,04 dB (le gain ≈ nul prédit au §9.3 —
  et aucune perte) ; biais résiduel 0,015 (naïf : 0,268 ≈ le gap de Jensen prédit
  de 0,25). Note d'exécution : le débruiteur interne retenu est
  `stft_wiener_auto` — l'ondelette à seuil global, pourtant la bénéficiaire
  « classique » d'une VST, *perdait* ~1 dB après stabilisation sur signaux
  corrélés au niveau (le calibrage MAD brut agissait comme seuil
  accidentellement adaptatif) ; conforme au principe §11 « ne jamais dégrader ».
- **Reco 3 — exécutée** : `denoise::compand` (`soft_clip`, `soft_clip_robust` ;
  tanh/atan/softsign), sans inverse par conception.
- **Reco 4 / Phase 2 — exécutée, verdict partagé** : module
  `denoise::multichannel`, porte « battre le par-canal sur ≥ 2 fixtures » :
  `wiener_spatial` (Wiener spatial joint ≡ widely-linear réel) **passe** —
  +2,48 dB (4 canaux corrélés) et +3,67 dB (stéréo rang-1) contre sa restriction
  diagonale ; `vector_median` **échoue** (0/2) — −1,81 dB sur impulsions
  synchronisées (E5b reproduit) et −2,02 dB sur impulsions *désynchronisées* :
  la conjecture du §12.4 (« médian vectoriel pour impulsions désynchronisées »)
  est **falsifiée** — le médian vectoriel restitue un vecteur observé dont tout
  le bruit de fond survit, quand la médiane scalaire moyenne ce bruit. Sa
  préservation de corrélation inter-canaux est elle aussi inférieure (erreur
  4,4e-3 vs 2,8e-3). Conservé comme implémentation de référence, verdict en doc ;
  chiffres reproductibles par `phase2_gate_report()`.
- **Phase 3 (octonion) — non déclenchée** : la condition (« un besoin démontré à
  8 canaux couplés ») n'est pas remplie ; la Phase 2 a au contraire falsifié
  l'opérateur joint de rang.
- **Phase 4 (SIMD des φ/φ⁻¹) — non déclenchée** : φ/φ⁻¹ sont des passes O(n) de
  fonctions élémentaires, négligeables devant le coût des débruiteurs internes.
- **Phase 5 (GPU) — non déclenchée** : volumes inchangés depuis le rapport.

## Addendum 2 — protocole §9 exécuté, extensions GAT et 2-D (2026-07-16)

Le protocole expérimental du §9 est désormais rejouable
(`cargo run --release -p scirust-signal --example vst_protocol`, blocs P1–P5
déterministes) et ses questions ouvertes sont **mesurées** :

- **§9.3, seuil de croisement (P4)** : à ×10 de dynamique de niveaux, le gain
  VST est déjà matériel (≥ +0,5 dB) à 2 % de bruit multiplicatif (le croisement
  est ≤ 2 %) ; à 30 % de bruit, le croisement en *dynamique de niveaux* est à
  **≈ ×3** — et à ×2 la VST est une perte matérielle de −0,77 dB, ce qui
  *précise et durcit* le « gain ≈ nul en régime doux ×2 » du §9.3. Conséquence
  codée : la porte de dynamique du sélecteur `detect_noise_model` est resserrée
  de ×2 à **×3** (constante `DETECT_MIN_RANGE`, documentée par cette mesure).
- **Régime de porteuse (P5, limitation nouvelle)** : le gain Anscombe s'effondre
  de +5,17 dB (porteuse à 3 cycles/4096) à **−0,93 dB (40 cycles)** — une φ
  ponctuelle ne commute pas avec le spectre : la racine convertit une porteuse
  rapide en pile d'harmoniques que le rétrécissement linéaire interne rogne.
  Documenté dans la doc du module `vst` (« Known limitation: fast carriers »)
  et épinglé par test. La VST s'adresse aux intensités *lentes*.
- **GAT (§9.1c, le cas Starck-Murtagh)** : `VstKind::Gat { gain, sigma }` avec
  l'inverse exact non biaisé en forme close (Mäkitalo-Foi 2013) : +1,54 à
  +2,87 dB selon la calibration (pire cas : lecture dominante (1.3, 1.5)) ;
  gain=1, σ=0 se réduit exactement à Anscombe. Fait notable de scalabilité : les
  calibrations à σ/gain constant sont des re-mises à l'échelle exactes (la GAT
  normalise le gain, chaque étage est équivariant d'échelle).
- **Transposition 2-D (`scirust_vision::denoise`)** : `vst_denoise2d` /
  `vst_denoise2d_auto`. Trois résultats 1-D se transposent tels quels : le
  VisuShrink 2-D perd sous stabilisation (−0,6 dB — calibrage MAD brut
  accidentellement adaptatif) ; la médiane 2-D est *invariante* bit à bit
  (Prop. 2 du rapport, confirmée empiriquement) ; le meilleur partenaire mesuré
  est **NLM 2-D** (+5,4 dB Poisson, +3,0 dB GAT — ses distances de patchs à `h`
  global supposent exactement l'homoscédasticité que la VST restaure). Le
  détecteur 1-D fonctionne sur images lisses en segments de lignes, avec une
  corrélation plus serrée (la dispersion log-niveau est le facteur limitant) ;
  les ratés échouent vers Identity (sûr).
- **Bras ondelette (P2)** : VisuShrink ne bénéficie de la stabilisation sur
  *aucune* fraction testée (−4,4 à −1,2 dB) — le choix `stft_wiener_auto` (1-D)
  / `nlm2d` (2-D) comme partenaires internes est confirmé.
