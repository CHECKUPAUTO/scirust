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
}
