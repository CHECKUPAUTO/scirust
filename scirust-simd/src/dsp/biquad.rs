// scirust-simd/src/dsp/biquad.rs
//
// # Filtre biquadratique générique `Biquad<T>`
//
// Une section du second ordre (deux pôles, deux zéros), **générique sur le
// scalaire** : le même code traite `Biquad<f32>`, `Biquad<f64>` **et**
// `Biquad<FixedI32<FRAC>>` (virgule fixe déterministe). Structure **Direct-Form
// II transposée** (DF2T), la plus robuste numériquement en précision finie.
//
// Fonction de transfert :
//
// ```text
//   H(z) = (b0 + b1 z⁻¹ + b2 z⁻²) / (1 + a1 z⁻¹ + a2 z⁻²)
// ```
//
// Récurrence DF2T (état `s1, s2`) :
//
// ```text
//   y = b0·x + s1
//   s1 = b1·x − a1·y + s2
//   s2 = b2·x − a2·y
// ```
//
// Le traitement n'utilise que l'anneau ([`NumericScalar`]) : en virgule fixe il
// est **déterministe bit-à-bit** (le filtrage donne les mêmes bits sur toute
// architecture, indépendamment du matériel flottant). Les coefficients sont
// **déjà normalisés** (`a0 = 1`).
//
// ## Conception de coefficients
//
// Les constructeurs [`Biquad::lowpass`], [`highpass`](Biquad::highpass),
// [`bandpass`](Biquad::bandpass) implémentent le « cookbook » RBJ (Robert
// Bristow-Johnson) et n'exigent que [`RealScalar`] (`sin`/`cos`/`recip`) — ils
// fonctionnent donc aussi en virgule fixe. Note : la **précision** des
// coefficients dépend de `FRAC` ; pour des filtres à `Q` élevé ou à très basse
// fréquence, préférer un `FRAC` large (p. ex. `Q8_24`).
//
// ## Filtres d'ordre supérieur (Butterworth) — [`BiquadCascade`]
//
// Une seule section du second ordre (`Biquad::lowpass`/`highpass` à
// `Q ≈ 0.707`) ne descend qu'à 12 dB/octave. [`BiquadCascade::butterworth_lowpass`]/
// [`butterworth_highpass`](BiquadCascade::butterworth_highpass) construisent un
// filtre Butterworth d'ordre pair `n` **quelconque** en cascadant `n/2`
// [`Biquad`] — **aucun nouveau filtre bas niveau**, seule la conception des
// coefficients (facteur de qualité par section) et l'enchaînement sont
// nouveaux. Le facteur de qualité de la `k`-ième section (`k = 1..n/2`) est
// `Qₖ = 1/(2·cos((2k−1)·π/(2n)))` — placement classique des pôles de
// Butterworth sur le cercle unité, répercuté section par section (`n = 2` ↦
// `Q₁ = 1/(2·cos(π/4)) = 1/√2`, le cas RBJ usuel). Réservé aux ordres **pairs**
// (un ordre impair exigerait une section du premier ordre, hors périmètre ici).
//
// ## Filtres de Chebyshev de type I — [`BiquadCascade::chebyshev1_lowpass`]/[`chebyshev1_highpass`](BiquadCascade::chebyshev1_highpass)
//
// Le Butterworth est **maximalement plat** en bande passante mais sa coupure
// est progressive. Chebyshev de type I accepte une **ondulation contrôlée**
// (`ripple_db`) en bande passante en échange d'une coupure bien plus raide à
// ordre égal — le compromis classique lorsque la platitude importe moins que
// la sélectivité. Ses pôles ne sont plus sur le cercle unité mais sur une
// **ellipse** : pour `ε = √(10^{ripple_db/10} − 1)` et `a = asinh(1/ε)/n`,
// le pôle `k` (`k = 1..n/2`, ordre pair comme Butterworth) a pour partie
// réelle `−sinh(a)·sin(θₖ)` et imaginaire `cosh(a)·cos(θₖ)`
// (`θₖ = (2k−1)π/(2n)`, même angle que Butterworth). `sinh`/`cosh`/`asinh`
// ne sont pas des méthodes de [`RealScalar`] : ils se déduisent localement
// de `exp`/`ln`/`sqrt` (déjà génériques et éprouvés), sans élargir le trait
// pour un seul usage.
//
// Chaque paire de pôles conjugués donne une section de fréquence propre
// `ωₙₖ = |pôleₖ|` et de facteur de qualité `Qₖ = ωₙₖ/(2·|Re(pôleₖ)|)` —
// contrairement à Butterworth, `ωₙₖ` **varie** d'une section à l'autre (les
// pôles ne sont pas à rayon unité), donc chaque section doit être
// **dénormalisée** par son propre `ωₙₖ` avant l'appel RBJ. Passe-bas :
// `cutoff·ωₙₖ` (mise à l'échelle directe). Passe-haut : la substitution
// analogique classique `s → 1/s` (passe-bas → passe-haut) préserve `Qₖ`
// mais **inverse** la fréquence propre (`ωₙₖ → 1/ωₙₖ`, démonstration dans le
// commentaire de [`chebyshev1_pole_params`]) — d'où `cutoff/ωₙₖ`, une
// division réelle (`/`), jamais `.recip()` mis en cache.
//
// Pour un ordre **pair**, la réponse touche le plancher d'ondulation
// (`−ripple_db`) **exactement** au continu **et** à la coupure (les deux
// sont des extrema du polynôme de Chebyshev sous-jacent) — propriété exacte
// exploitée par les tests plutôt qu'une tolérance ad hoc.

use core::ops::Div;

use crate::fixed::{NumericScalar, RealScalar};

/// Filtre biquadratique (section du second ordre), forme directe II transposée.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Biquad<T> {
    b0: T,
    b1: T,
    b2: T,
    a1: T,
    a2: T,
    s1: T,
    s2: T,
}

impl<T: NumericScalar> Biquad<T> {
    /// Construit depuis des coefficients **déjà normalisés** (`a0 = 1`). L'état
    /// interne est initialisé à zéro.
    #[inline]
    pub fn new(b0: T, b1: T, b2: T, a1: T, a2: T) -> Self {
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            s1: T::zero(),
            s2: T::zero(),
        }
    }

    /// Filtre « passe-tout » identité (`H(z) = 1`) : la sortie égale l'entrée.
    #[inline]
    pub fn identity() -> Self {
        Self::new(T::one(), T::zero(), T::zero(), T::zero(), T::zero())
    }

    /// Remet l'état interne à zéro (redémarre le filtre).
    #[inline]
    pub fn reset(&mut self) {
        self.s1 = T::zero();
        self.s2 = T::zero();
    }

    /// Traite un échantillon et met à jour l'état (DF2T).
    #[inline]
    pub fn process(&mut self, x: T) -> T {
        let y = self.b0 * x + self.s1;
        self.s1 = self.b1 * x - self.a1 * y + self.s2;
        self.s2 = self.b2 * x - self.a2 * y;
        y
    }

    /// Filtre un bloc `input` vers `out` (même longueur). Panique sinon.
    #[inline]
    pub fn process_block(&mut self, input: &[T], out: &mut [T]) {
        assert_eq!(
            input.len(),
            out.len(),
            "process_block: longueurs différentes"
        );
        for (o, &x) in out.iter_mut().zip(input)
        {
            *o = self.process(x);
        }
    }

    /// Coefficients normalisés `(b0, b1, b2, a1, a2)`.
    #[inline]
    pub fn coefficients(&self) -> (T, T, T, T, T) {
        (self.b0, self.b1, self.b2, self.a1, self.a2)
    }
}

impl<T: RealScalar> Biquad<T> {
    /// Pulsation normalisée `ω₀ = 2π·f₀/fs` et `(cos ω₀, sin ω₀, α)` avec
    /// `α = sin ω₀ / (2Q)`. Base commune des conceptions RBJ.
    #[inline]
    fn rbj_prelude(sample_rate: T, cutoff: T, q: T) -> (T, T, T) {
        let two = T::from_i32(2);
        let w0 = two * T::pi() * cutoff * sample_rate.recip();
        let (sn, cs) = (w0.sin(), w0.cos());
        let alpha = sn * (two * q).recip();
        (cs, sn, alpha)
    }

    /// Passe-bas RBJ à `cutoff` (Hz), échantillonné à `sample_rate` (Hz),
    /// facteur de qualité `q` (`≈0.707` pour Butterworth). Gain unité au continu.
    #[inline]
    pub fn lowpass(sample_rate: T, cutoff: T, q: T) -> Self {
        let (cs, _sn, alpha) = Self::rbj_prelude(sample_rate, cutoff, q);
        let one = T::one();
        let two = T::from_i32(2);
        let inv_a0 = (one + alpha).recip();
        let b1 = (one - cs) * inv_a0;
        let b0 = b1 * two.recip(); // (1−cos)/2 · 1/a0
        let b2 = b0;
        let a1 = (-(two * cs)) * inv_a0;
        let a2 = (one - alpha) * inv_a0;
        Self::new(b0, b1, b2, a1, a2)
    }

    /// Passe-haut RBJ. Gain unité en haute fréquence, nul au continu.
    #[inline]
    pub fn highpass(sample_rate: T, cutoff: T, q: T) -> Self {
        let (cs, _sn, alpha) = Self::rbj_prelude(sample_rate, cutoff, q);
        let one = T::one();
        let two = T::from_i32(2);
        let inv_a0 = (one + alpha).recip();
        let one_plus = (one + cs) * inv_a0;
        let b0 = one_plus * two.recip(); // (1+cos)/2 /a0
        let b1 = -one_plus; // −(1+cos)/a0
        let b2 = b0;
        let a1 = (-(two * cs)) * inv_a0;
        let a2 = (one - alpha) * inv_a0;
        Self::new(b0, b1, b2, a1, a2)
    }

    /// Passe-bande RBJ (gain crête = `Q`, forme « constant skirt gain »).
    #[inline]
    pub fn bandpass(sample_rate: T, cutoff: T, q: T) -> Self {
        let (cs, sn, alpha) = Self::rbj_prelude(sample_rate, cutoff, q);
        let one = T::one();
        let two = T::from_i32(2);
        let inv_a0 = (one + alpha).recip();
        let b0 = (sn * two.recip()) * inv_a0; // sin/2 /a0
        let b1 = T::zero();
        let b2 = -b0;
        let a1 = (-(two * cs)) * inv_a0;
        let a2 = (one - alpha) * inv_a0;
        Self::new(b0, b1, b2, a1, a2)
    }
}

/// Facteurs de qualité des `order/2` sections d'un filtre de Butterworth
/// d'ordre pair `order` : `Qₖ = 1/(2·cos((2k−1)·π/(2·order)))` (cf. en-tête de
/// module). La division par `2·order` (pas une puissance de deux) utilise
/// l'opérateur `/`, pas `recip()` — même précaution que
/// [`crate::geometry::Quaternion::from_rotation_matrix`].
///
/// Panique si `order < 2` ou `order` est impair.
pub(crate) fn butterworth_qs<T: RealScalar + Div<Output = T>>(order: usize) -> Vec<T> {
    assert!(order >= 2, "butterworth : ordre {order} doit être ≥ 2");
    assert_eq!(
        order % 2,
        0,
        "butterworth : ordre {order} doit être pair (une section du premier ordre serait nécessaire pour un ordre impair)"
    );
    let n_sections = order / 2;
    let two = T::from_i32(2);
    let order_t = T::from_i32(order as i32);
    (1..=n_sections)
        .map(|k| {
            let angle = T::from_i32((2 * k - 1) as i32) * T::pi() / (two * order_t);
            T::one() / (two * angle.cos())
        })
        .collect()
}

/// `(ωₙₖ, Qₖ)` des `order/2` sections du prototype Chebyshev de type I
/// normalisé (coupure `= 1`), `k = 1..order/2` (cf. en-tête de module pour la
/// dérivation). `ripple_db` est l'ondulation crête-à-plancher en bande
/// passante (typiquement `0,1` à `3` dB).
///
/// Dérivation de l'inversion de fréquence propre en passe-haut (`s → 1/s`) :
/// une section passe-bas normalisée `H(s) = ωₙ²/(s² + (ωₙ/Q)·s + ωₙ²)`
/// devient, en substituant `s → 1/s` puis en multipliant haut et bas par
/// `s²` puis en divisant par `ωₙ²` :
/// `H(s) = s² / (s² + (1/(Q·ωₙ))·s + (1/ωₙ)²)` — une section passe-haut de
/// même `Q` mais de fréquence propre `1/ωₙ`.
///
/// Panique si `order < 2` ou `order` est impair (mêmes conditions que
/// [`butterworth_qs`]).
pub(crate) fn chebyshev1_pole_params<T: RealScalar + Div<Output = T>>(
    order: usize,
    ripple_db: T,
) -> Vec<(T, T)> {
    assert!(order >= 2, "chebyshev1 : ordre {order} doit être ≥ 2");
    assert_eq!(
        order % 2,
        0,
        "chebyshev1 : ordre {order} doit être pair (une section du premier ordre serait nécessaire pour un ordre impair)"
    );
    let one = T::one();
    let two = T::from_i32(2);
    let ten = T::from_i32(10);
    let n_sections = order / 2;
    let order_t = T::from_i32(order as i32);

    // ε = √(10^{ripple_db/10} − 1), 10^x = e^{x·ln 10}.
    let epsilon = ((ripple_db / ten * ten.ln()).exp() - one).sqrt();
    // a = asinh(1/ε)/n, asinh(x) = ln(x + √(x²+1)).
    let inv_eps = one / epsilon;
    let a = (inv_eps + (inv_eps * inv_eps + one).sqrt()).ln() / order_t;
    let half = two.recip(); // puissance de deux : recip() exact.
    let sinh_a = (a.exp() - (-a).exp()) * half;
    let cosh_a = (a.exp() + (-a).exp()) * half;

    (1..=n_sections)
        .map(|k| {
            let theta = T::from_i32((2 * k - 1) as i32) * T::pi() / (two * order_t);
            let sigma = sinh_a * theta.sin(); // |Re(pôle)|, pôle = −sigma ± j·omega.
            let omega = cosh_a * theta.cos();
            let wn = (sigma * sigma + omega * omega).sqrt();
            let q = wn / (two * sigma);
            (wn, q)
        })
        .collect()
}

/// Cascade de [`Biquad`] : filtre d'ordre pair `2·n` (`n` sections du second
/// ordre enchaînées), générique sur le scalaire comme [`Biquad`].
#[derive(Debug, Clone, PartialEq)]
pub struct BiquadCascade<T> {
    stages: Vec<Biquad<T>>,
}

impl<T: NumericScalar> BiquadCascade<T> {
    /// Construit depuis des sections déjà conçues (au moins une).
    #[inline]
    pub fn new(stages: Vec<Biquad<T>>) -> Self {
        assert!(
            !stages.is_empty(),
            "BiquadCascade::new : au moins une section requise"
        );
        Self { stages }
    }

    /// Ordre du filtre (`2 × nombre de sections`).
    #[inline]
    #[must_use]
    pub fn order(&self) -> usize {
        self.stages.len() * 2
    }

    /// Sections de la cascade (lecture seule) — permet d'inspecter les
    /// coefficients de chaque section (cf. [`Biquad::coefficients`]).
    #[inline]
    #[must_use]
    pub fn stages(&self) -> &[Biquad<T>] {
        &self.stages
    }

    /// Remet à zéro l'état interne de toutes les sections.
    #[inline]
    pub fn reset(&mut self) {
        for stage in &mut self.stages
        {
            stage.reset();
        }
    }

    /// Traite un échantillon à travers toute la cascade (chaque section
    /// consomme la sortie de la précédente).
    #[inline]
    pub fn process(&mut self, x: T) -> T {
        let mut y = x;
        for stage in &mut self.stages
        {
            y = stage.process(y);
        }
        y
    }

    /// Filtre un bloc `input` vers `out` (même longueur) à travers toute la
    /// cascade. Panique sinon.
    #[inline]
    pub fn process_block(&mut self, input: &[T], out: &mut [T]) {
        assert_eq!(
            input.len(),
            out.len(),
            "BiquadCascade::process_block : longueurs différentes"
        );
        for (o, &x) in out.iter_mut().zip(input)
        {
            *o = self.process(x);
        }
    }
}

impl<T: RealScalar + Div<Output = T>> BiquadCascade<T> {
    /// Filtre de Butterworth **passe-bas**, ordre pair `order`, à `cutoff`
    /// (Hz), échantillonné à `sample_rate` (Hz) : réponse maximalement plate
    /// dans la bande passante, gain unité au continu. Cascade de `order/2`
    /// [`Biquad::lowpass`] aux facteurs de qualité de Butterworth (cf.
    /// en-tête de module).
    ///
    /// Panique si `order < 2` ou `order` est impair.
    #[must_use]
    pub fn butterworth_lowpass(sample_rate: T, cutoff: T, order: usize) -> Self {
        let stages = butterworth_qs::<T>(order)
            .into_iter()
            .map(|q| Biquad::lowpass(sample_rate, cutoff, q))
            .collect();
        Self::new(stages)
    }

    /// Filtre de Butterworth **passe-haut**, ordre pair `order` — symétrique
    /// de [`Self::butterworth_lowpass`] (cf. [`Biquad::highpass`]).
    ///
    /// Panique si `order < 2` ou `order` est impair.
    #[must_use]
    pub fn butterworth_highpass(sample_rate: T, cutoff: T, order: usize) -> Self {
        let stages = butterworth_qs::<T>(order)
            .into_iter()
            .map(|q| Biquad::highpass(sample_rate, cutoff, q))
            .collect();
        Self::new(stages)
    }

    /// Filtre de Chebyshev de type I **passe-bas**, ordre pair `order`, à
    /// `cutoff` (Hz), échantillonné à `sample_rate` (Hz), ondulation
    /// `ripple_db` (dB, crête-à-plancher) en bande passante : coupure plus
    /// raide que Butterworth au même ordre, au prix d'une ondulation
    /// contrôlée plutôt qu'une platitude maximale (cf. en-tête de module).
    /// Chaque section est dénormalisée par sa propre fréquence propre
    /// (`cutoff·ωₙₖ`, cf. [`chebyshev1_pole_params`]), puis le gain global de
    /// la cascade est corrigé (cf. [`apply_ripple_floor_gain`]).
    ///
    /// Panique si `order < 2` ou `order` est impair.
    #[must_use]
    pub fn chebyshev1_lowpass(sample_rate: T, cutoff: T, order: usize, ripple_db: T) -> Self {
        let mut stages: Vec<Biquad<T>> = chebyshev1_pole_params::<T>(order, ripple_db)
            .into_iter()
            .map(|(wn, q)| Biquad::lowpass(sample_rate, cutoff * wn, q))
            .collect();
        apply_ripple_floor_gain(&mut stages, ripple_db);
        Self::new(stages)
    }

    /// Filtre de Chebyshev de type I **passe-haut**, ordre pair `order` —
    /// mêmes pôles prototypes que [`Self::chebyshev1_lowpass`], mais chaque
    /// section est dénormalisée par l'**inverse** de sa fréquence propre
    /// (`cutoff/ωₙₖ`, substitution `s → 1/s`, cf.
    /// [`chebyshev1_pole_params`]) et utilise [`Biquad::highpass`], puis le
    /// gain global est corrigé (cf. [`apply_ripple_floor_gain`]).
    ///
    /// Panique si `order < 2` ou `order` est impair.
    #[must_use]
    pub fn chebyshev1_highpass(sample_rate: T, cutoff: T, order: usize, ripple_db: T) -> Self {
        let mut stages: Vec<Biquad<T>> = chebyshev1_pole_params::<T>(order, ripple_db)
            .into_iter()
            .map(|(wn, q)| Biquad::highpass(sample_rate, cutoff / wn, q))
            .collect();
        apply_ripple_floor_gain(&mut stages, ripple_db);
        Self::new(stages)
    }
}

/// Corrige le gain global d'une cascade de Chebyshev de type I (ordre pair).
///
/// [`Biquad::lowpass`]/[`highpass`](Biquad::highpass) normalisent **chacun**
/// leur propre section à gain unité à leur extremum de référence (continu
/// pour passe-bas, haute fréquence pour passe-haut — cf. leur documentation)
/// : le produit de `n/2` sections ainsi normalisées a donc **toujours** un
/// gain de cet extremum égal à `1` (`1 × 1 × ⋯ × 1`), quels que soient les
/// pôles. Or un filtre de Chebyshev d'ordre **pair** a un gain exactement
/// `−ripple_db` (pas `0` dB) à cet extremum — les deux polynômes de
/// Chebyshev sous-jacents y valent `±1` (un extremum), donnant
/// `|H|² = 1/(1+ε²)`. Puisque les sections normalisées et la cascade
/// correctement mise à l'échelle ont le **même dénominateur** (mêmes pôles),
/// elles ne diffèrent que d'une **constante multiplicative** — reporter
/// cette constante (`10^{−ripple_db/20}`) sur la **première** section (peu
/// importe laquelle, un produit en cascade) corrige le gain global sans
/// toucher à la forme de l'ondulation.
fn apply_ripple_floor_gain<T: RealScalar + Div<Output = T>>(
    stages: &mut [Biquad<T>],
    ripple_db: T,
) {
    let two = T::from_i32(2);
    let ten = T::from_i32(10);
    let floor = (-ripple_db / (two * ten) * ten.ln()).exp(); // 10^{−ripple_db/20}.
    if let Some(first) = stages.first_mut()
    {
        let (b0, b1, b2, a1, a2) = first.coefficients();
        *first = Biquad::new(b0 * floor, b1 * floor, b2 * floor, a1, a2);
    }
}
