// scirust-simd/src/dsp/freqz.rs
//
// # Réponse en fréquence — évaluation de `H(e^{jω})`
//
// `dsp` sait **concevoir** des filtres ([`super::biquad`], [`super::fir`])
// mais rien n'en évaluait la réponse en fréquence : ce module comble ce vide
// (l'équivalent de `scipy.signal.freqz`), **générique sur le scalaire** comme
// le reste de `dsp`.
//
// ## Évaluation
//
// La fonction de transfert est évaluée directement au point `z = e^{jω}` du
// cercle unité (`ω` en radians/échantillon, `ω ∈ [0, π]` couvre tout le
// spectre utile par symétrie hermitienne d'un filtre à coefficients réels) :
//
// * [`super::biquad::Biquad`] : `H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (1 + a1·z⁻¹ + a2·z⁻²)`,
//   évaluation directe (numérateur et dénominateur, division complexe via
//   [`super::fft::Complex::recip`]).
// * [`super::biquad::BiquadCascade`] : produit des réponses de chaque
//   section (les fonctions de transfert d'une cascade se multiplient).
// * [`super::fir::Fir`] : `H(z) = Σₖ hₖ·z⁻ᵏ`, schéma de Horner (aucune
//   division, un FIR n'a pas de pôles).
//
// ## Magnitude, phase, délai de groupe
//
// [`magnitude`]/[`magnitude_db`] et [`phase`] sont des fonctions libres sur
// [`super::fft::Complex`] (pas spécifiques à un type de filtre). `phase`
// utilise [`crate::fixed::RealScalar::atan2`], à valeurs dans `(−π, π]` —
// une courbe de phase présente donc des discontinuités artificielles de
// `2π` à chaque franchissement de cette coupure, que [`unwrap_phase`]
// élimine (technique standard, `numpy.unwrap`) : ajouter/soustraire le
// multiple de `2π` nécessaire pour que chaque saut consécutif reste `≤ π` en
// valeur absolue.
//
// [`group_delay`] (`τ(ω) = −dφ/dω`) s'obtient par différences finies sur la
// phase **déballée** (une phase encore repliée produirait des sauts
// artificiels de plusieurs échantillons) : centrées à l'intérieur de la
// grille, décentrées aux deux extrémités. Un FIR à coefficients symétriques
// a un délai de groupe **constant** `= (N−1)/2` (cf. [`super::fir`]) —
// propriété exacte exploitée par les tests plutôt qu'une référence séparée.

use core::ops::Div;

use crate::fixed::RealScalar;

use super::biquad::{Biquad, BiquadCascade};
use super::fft::Complex;
use super::fir::Fir;

/// `z⁻¹ = e^{−jω} = cos ω − j·sin ω`, le retard unité évalué sur le cercle
/// unité à la pulsation normalisée `ω` (radians/échantillon).
#[inline]
fn unit_delay<T: RealScalar>(omega: T) -> Complex<T> {
    Complex::new(omega.cos(), -omega.sin())
}

/// Module `|H|` d'une réponse complexe.
#[inline]
#[must_use]
pub fn magnitude<T: RealScalar>(h: Complex<T>) -> T {
    h.norm_sqr().sqrt()
}

/// Module en décibels `20·log₁₀|H| = (20/ln 10)·ln|H|` (pas de `log10` dans
/// [`RealScalar`] : `ln` seul, même schéma que `10^x = e^{x·ln 10}` déjà
/// utilisé par [`super::biquad::chebyshev1_pole_params`]).
#[inline]
#[must_use]
pub fn magnitude_db<T: RealScalar + Div<Output = T>>(h: Complex<T>) -> T {
    let ten = T::from_i32(10);
    let twenty = T::from_i32(20);
    (twenty / ten.ln()) * magnitude(h).ln()
}

/// Phase `arg(H) = atan2(im, re) ∈ (−π, π]` d'une réponse complexe — repliée
/// à cette coupure, cf. [`unwrap_phase`] pour une courbe continue.
#[inline]
#[must_use]
pub fn phase<T: RealScalar>(h: Complex<T>) -> T {
    h.im.atan2(h.re)
}

/// Déballe en place une suite de phases (`numpy.unwrap`) : ajoute/soustrait
/// le multiple de `2π` nécessaire pour que chaque saut consécutif reste
/// `≤ π` en valeur absolue, éliminant les discontinuités artificielles de
/// [`phase`] à la coupure `±π` (cf. en-tête de module). Suppose `phases`
/// échantillonné assez finement pour que la **vraie** variation de phase
/// entre deux points consécutifs reste `< π` (sinon ambiguë, comme
/// `numpy.unwrap`).
pub fn unwrap_phase<T: RealScalar>(phases: &mut [T]) {
    let two_pi = T::from_i32(2) * T::pi();
    for i in 1..phases.len()
    {
        let mut diff = phases[i] - phases[i - 1];
        while diff > T::pi()
        {
            phases[i] = phases[i] - two_pi;
            diff = diff - two_pi;
        }
        while diff < -T::pi()
        {
            phases[i] = phases[i] + two_pi;
            diff = diff + two_pi;
        }
    }
}

/// Délai de groupe `τ(ω) = −dφ/dω` (cf. en-tête de module), estimé par
/// différences finies sur `unwrapped_phase` — phase **déjà déballée**
/// ([`unwrap_phase`]) échantillonnée sur une grille de pulsations
/// **régulièrement espacées** de pas `d_omega` : centrées aux points
/// intérieurs, décentrées (avant/arrière) aux deux extrémités.
///
/// Panique si `unwrapped_phase.len() < 2`.
#[must_use]
pub fn group_delay<T: RealScalar + Div<Output = T>>(unwrapped_phase: &[T], d_omega: T) -> Vec<T> {
    let n = unwrapped_phase.len();
    assert!(n >= 2, "group_delay : au moins 2 points requis");
    let two = T::from_i32(2);
    let mut out = vec![T::zero(); n];
    out[0] = -((unwrapped_phase[1] - unwrapped_phase[0]) / d_omega);
    out[n - 1] = -((unwrapped_phase[n - 1] - unwrapped_phase[n - 2]) / d_omega);
    for i in 1..n - 1
    {
        out[i] = -((unwrapped_phase[i + 1] - unwrapped_phase[i - 1]) / (two * d_omega));
    }
    out
}

impl<T: RealScalar> Biquad<T> {
    /// Réponse en fréquence `H(e^{jω})` à la pulsation normalisée `ω`
    /// (radians/échantillon, cf. en-tête de module).
    #[must_use]
    pub fn frequency_response(&self, omega: T) -> Complex<T> {
        let z_inv = unit_delay(omega);
        let z_inv2 = z_inv * z_inv;
        let (b0, b1, b2, a1, a2) = self.coefficients();
        let num = Complex::from_real(b0)
            + Complex::from_real(b1) * z_inv
            + Complex::from_real(b2) * z_inv2;
        let den = Complex::from_real(T::one())
            + Complex::from_real(a1) * z_inv
            + Complex::from_real(a2) * z_inv2;
        num / den
    }

    /// [`Self::frequency_response`] sur toute une grille de pulsations.
    #[must_use]
    pub fn frequency_response_grid(&self, omegas: &[T]) -> Vec<Complex<T>> {
        omegas.iter().map(|&w| self.frequency_response(w)).collect()
    }
}

impl<T: RealScalar + Div<Output = T>> Biquad<T> {
    /// [`Self::frequency_response`] à `freq_hz` (Hz), échantillonné à
    /// `sample_rate` (Hz) : `ω = 2π·freq_hz/sample_rate` (même convention que
    /// [`Biquad::lowpass`]/[`highpass`](Biquad::highpass)).
    #[must_use]
    pub fn frequency_response_hz(&self, sample_rate: T, freq_hz: T) -> Complex<T> {
        let omega = T::from_i32(2) * T::pi() * freq_hz / sample_rate;
        self.frequency_response(omega)
    }
}

impl<T: RealScalar> BiquadCascade<T> {
    /// Réponse en fréquence de la cascade entière à la pulsation normalisée
    /// `ω` : produit des réponses de chaque section (cf. en-tête de module).
    #[must_use]
    pub fn frequency_response(&self, omega: T) -> Complex<T> {
        self.stages()
            .iter()
            .fold(Complex::from_real(T::one()), |acc, stage| {
                acc * stage.frequency_response(omega)
            })
    }

    /// [`Self::frequency_response`] sur toute une grille de pulsations.
    #[must_use]
    pub fn frequency_response_grid(&self, omegas: &[T]) -> Vec<Complex<T>> {
        omegas.iter().map(|&w| self.frequency_response(w)).collect()
    }
}

impl<T: RealScalar + Div<Output = T>> BiquadCascade<T> {
    /// [`Self::frequency_response`] à `freq_hz` (Hz), échantillonné à
    /// `sample_rate` (Hz) — cf. [`Biquad::frequency_response_hz`].
    #[must_use]
    pub fn frequency_response_hz(&self, sample_rate: T, freq_hz: T) -> Complex<T> {
        let omega = T::from_i32(2) * T::pi() * freq_hz / sample_rate;
        self.frequency_response(omega)
    }
}

impl<T: RealScalar, const N: usize> Fir<T, N> {
    /// Réponse en fréquence `H(e^{jω}) = Σₖ hₖ·z⁻ᵏ` à la pulsation normalisée
    /// `ω` (schéma de Horner, cf. en-tête de module — aucune division, un FIR
    /// n'a pas de pôles).
    #[must_use]
    pub fn frequency_response(&self, omega: T) -> Complex<T> {
        let z_inv = unit_delay(omega);
        let mut acc = Complex::zero();
        for &h in self.taps().iter().rev()
        {
            acc = acc * z_inv + Complex::from_real(h);
        }
        acc
    }

    /// [`Self::frequency_response`] sur toute une grille de pulsations.
    #[must_use]
    pub fn frequency_response_grid(&self, omegas: &[T]) -> Vec<Complex<T>> {
        omegas.iter().map(|&w| self.frequency_response(w)).collect()
    }
}
