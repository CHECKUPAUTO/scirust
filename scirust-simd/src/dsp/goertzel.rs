// scirust-simd/src/dsp/goertzel.rs
//
// # Algorithme de Goertzel — DFT à une seule fréquence
//
// Quand on ne veut **qu'un** bin du spectre (détection de tonalité, DTMF,
// mesure de puissance à une raie précise), une FFT complète est du gaspillage.
// L'algorithme de Goertzel évalue `X(ω) = Σₙ x[n]·e^{−jωn}` en `O(N)` avec une
// **récurrence à deux termes** (un seul multiplieur réel par échantillon),
// sans table de twiddles ni tampon — idéal en embarqué. **Générique sur le
// scalaire** comme le reste de `dsp`.
//
// ## Récurrence
//
// Avec `coeff = 2·cos ω`, on itère l'état `s[n] = x[n] + coeff·s[n−1] − s[n−2]`
// (état initial nul). Après les `N` échantillons, le filtre de sortie
// `y = s[N−1] − e^{−jω}·s[N−2]` se lit sur les deux derniers états
// `q₁ = s[N−1]`, `q₂ = s[N−2]` :
//
// ```text
// Re(y) = q₁ − q₂·cos ω ,  Im(y) = q₂·sin ω
// ```
//
// mais `y = e^{jω(N−1)}·X(ω)` : le pôle unique `e^{jω}` de la section a une
// réponse impulsionnelle `e^{jωn}`, d'où un **facteur de phase** `e^{jω(N−1)}`
// à retirer pour obtenir la vraie valeur de DFT. [`goertzel`] applique donc la
// correction `X(ω) = y·e^{−jω(N−1)}` (Goertzel généralisé, Sysel & Rajmic) —
// une seule paire `cos`/`sin` supplémentaire, pas par échantillon.
//
// La **puissance** `|X|² = |y|² = q₁² + q₂² − coeff·q₁·q₂` est en revanche
// insensible à ce facteur (`|e^{jφ}| = 1`) : [`goertzel_power`] l'obtient sans
// aucune multiplication trigonométrique finale (identité algébrique exacte).
//
// ## Fréquence arbitraire (Goertzel généralisé)
//
// `ω` (radians/échantillon) est **quelconque** : nul besoin qu'il tombe sur un
// bin `2πk/N`. C'est le Goertzel « généralisé » — on peut viser exactement une
// fréquence physique (via [`goertzel_hz`]) plutôt que le bin le plus proche.
//
// ## Précision en virgule fixe
//
// La récurrence est purement un anneau (`+ − ×`) : le seul arrondi vient de
// `coeff = 2·cos ω` (un unique `cos`) et de l'accumulation de `s`. Pour un
// signal borné et `N` modéré, `Q16_16` suit le flottant de près. Attention :
// `s` peut croître jusqu'à `≈ N·max|x|/|sin ω|` près de la résonance — préférer
// un `FRAC` large ou normaliser pour de longs blocs. Aucun `unsafe`.

use super::fft::Complex;
use crate::fixed::RealScalar;

/// Valeur complexe de la DFT `X(ω) = Σₙ x[n]·e^{−jωn}` à la pulsation
/// normalisée `omega` (radians/échantillon), par la récurrence de Goertzel
/// (cf. en-tête de module). `x` peut avoir une longueur quelconque.
#[must_use]
pub fn goertzel<T: RealScalar>(x: &[T], omega: T) -> Complex<T> {
    let cos_w = omega.cos();
    let sin_w = omega.sin();
    let coeff = T::from_i32(2) * cos_w;
    let (mut q1, mut q2) = (T::zero(), T::zero());
    for &v in x
    {
        let q0 = v + coeff * q1 - q2;
        q2 = q1;
        q1 = q0;
    }
    // q1 = s[N−1], q2 = s[N−2] ; y = s[N−1] − e^{−jω}·s[N−2].
    let y = Complex::new(q1 - q2 * cos_w, q2 * sin_w);
    // Retire le facteur de phase : X = y·e^{−jω(N−1)} (cf. en-tête de module).
    let n = x.len();
    if n <= 1
    {
        return y;
    }
    let phase = omega * T::from_i32((n - 1) as i32);
    let correction = Complex::new(phase.cos(), -phase.sin());
    y * correction
}

/// Puissance `|X(ω)|² = q₁² + q₂² − 2·cos ω·q₁·q₂` à la pulsation normalisée
/// `omega` (radians/échantillon), sans les deux multiplications
/// trigonométriques finales de [`goertzel`] (identité algébrique exacte, cf.
/// en-tête de module) — la voie économique pour un simple détecteur de
/// présence de tonalité.
#[must_use]
pub fn goertzel_power<T: RealScalar>(x: &[T], omega: T) -> T {
    let cos_w = omega.cos();
    let coeff = T::from_i32(2) * cos_w;
    let (mut q1, mut q2) = (T::zero(), T::zero());
    for &v in x
    {
        let q0 = v + coeff * q1 - q2;
        q2 = q1;
        q1 = q0;
    }
    q1 * q1 + q2 * q2 - coeff * q1 * q2
}

/// [`goertzel`] visant une fréquence physique `freq_hz` (Hz) sur un signal
/// échantillonné à `sample_rate` (Hz) : `ω = 2π·freq_hz/sample_rate` (même
/// convention que le reste de `dsp`, cf. [`super::biquad::Biquad::lowpass`]).
#[must_use]
pub fn goertzel_hz<T: RealScalar>(x: &[T], sample_rate: T, freq_hz: T) -> Complex<T> {
    let omega = T::from_i32(2) * T::pi() * freq_hz * sample_rate.recip();
    goertzel(x, omega)
}

/// [`goertzel_power`] visant une fréquence physique `freq_hz` (Hz) — cf.
/// [`goertzel_hz`].
#[must_use]
pub fn goertzel_power_hz<T: RealScalar>(x: &[T], sample_rate: T, freq_hz: T) -> T {
    let omega = T::from_i32(2) * T::pi() * freq_hz * sample_rate.recip();
    goertzel_power(x, omega)
}
