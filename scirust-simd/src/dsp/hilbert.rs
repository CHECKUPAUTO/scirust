// scirust-simd/src/dsp/hilbert.rs
//
// # Transformée de Hilbert & signal analytique (par FFT)
//
// `dsp` savait déjà passer au domaine fréquentiel ([`super::fft`]) mais rien
// n'en tirait le **signal analytique** — la brique de base de toute
// démodulation d'amplitude/de phase. Ce module comble ce vide (l'équivalent de
// `scipy.signal.hilbert`), **générique sur le scalaire** comme le reste de
// `dsp` : le même code sert `f32`, `f64` **et** la virgule fixe déterministe.
//
// ## Signal analytique
//
// Le signal analytique de `x` réel est `z = x + i·H{x}`, où `H{x}` est la
// transformée de Hilbert (déphasage de −90° de chaque composante fréquentielle
// positive). On l'obtient par la méthode fréquentielle standard :
//
// 1. `X = FFT(x)` ;
// 2. on **annule les fréquences négatives** et on **double les positives** en
//    multipliant `X[k]` par `h[k]` :
//    * `h[0] = 1` (continu) et `h[N/2] = 1` (Nyquist) — inchangés ;
//    * `h[k] = 2` pour `1 ≤ k < N/2` — fréquences positives doublées ;
//    * `h[k] = 0` pour `N/2 < k < N` — fréquences négatives annulées ;
// 3. `z = IFFT(X·h)`.
//
// La partie réelle de `z` reconstruit `x` ; sa partie imaginaire **est** `H{x}`.
// La longueur doit être une **puissance de 2** (contrainte de [`super::fft`]).
//
// ## Quantités instantanées
//
// À partir de `z = A·e^{iφ}` on lit directement :
//
// * l'**enveloppe** `A = |z|` ([`envelope`]) — la détection d'enveloppe d'un
//   signal AM ;
// * la **phase instantanée** `φ = arg(z)` déballée ([`instantaneous_phase`],
//   via [`super::freqz::unwrap_phase`]) ;
// * la **fréquence instantanée** `f = (1/2π)·dφ/dt` ([`instantaneous_frequency`],
//   différences finies sur la phase déballée) — la démodulation FM.
//
// ## Précision en virgule fixe
//
// Deux FFT de longueur `N` en cascade (directe puis inverse) : chaque étage
// accumule l'arrondi des twiddles (cf. en-tête de [`super::fft`]). Pour un
// signal borné dans `[−1, 1]` et `N ≤ 2¹²`, `Q16_16` suffit ; pour de longues
// transformées, préférer un `FRAC` large. Aucun `unsafe`.

use super::fft::{Complex, fft, ifft};
use crate::fixed::RealScalar;

/// Signal analytique `z = x + i·H{x}` de `x` réel, calculé par FFT (cf. en-tête
/// de module). `z[n].re` reconstruit `x[n]`, `z[n].im` est la transformée de
/// Hilbert `H{x}[n]`.
///
/// Panique si `x.len()` n'est pas une puissance de 2 `≥ 2` (contrainte de
/// [`super::fft::fft`]).
#[must_use]
pub fn analytic_signal<T: RealScalar>(x: &[T]) -> Vec<Complex<T>> {
    let n = x.len();
    assert!(
        n.is_power_of_two() && n >= 2,
        "analytic_signal : longueur = puissance de 2 ≥ 2"
    );
    let mut spec: Vec<Complex<T>> = x.iter().map(|&v| Complex::from_real(v)).collect();
    fft(&mut spec);

    // Multiplieur h : double les fréquences positives, annule les négatives,
    // laisse le continu (k=0) et Nyquist (k=N/2) inchangés.
    let two = T::from_i32(2);
    let half = n / 2;
    for (k, c) in spec.iter_mut().enumerate()
    {
        if k == 0 || k == half
        {
            // h = 1 : inchangé.
        }
        else if k < half
        {
            *c = c.scale(two); // h = 2 : fréquence positive.
        }
        else
        {
            *c = Complex::zero(); // h = 0 : fréquence négative.
        }
    }

    ifft(&mut spec);
    spec
}

/// Transformée de Hilbert `H{x}` de `x` réel : partie imaginaire du signal
/// analytique ([`analytic_signal`]). `H` déphase chaque composante de fréquence
/// positive de `−90°` (par exemple `H{cos} = sin`).
///
/// Panique selon les préconditions d'[`analytic_signal`].
#[must_use]
pub fn hilbert<T: RealScalar>(x: &[T]) -> Vec<T> {
    analytic_signal(x).iter().map(|z| z.im).collect()
}

/// Enveloppe instantanée `A[n] = |z[n]|` (module du signal analytique) — la
/// détection d'enveloppe d'un signal modulé en amplitude.
///
/// Panique selon les préconditions d'[`analytic_signal`].
#[must_use]
pub fn envelope<T: RealScalar>(x: &[T]) -> Vec<T> {
    analytic_signal(x)
        .iter()
        .map(|z| z.norm_sqr().sqrt())
        .collect()
}

/// Phase instantanée `φ[n] = arg(z[n])`, **déballée** ([`super::freqz::unwrap_phase`])
/// pour éliminer les sauts artificiels de `2π` à la coupure `±π` de `atan2`.
///
/// Panique selon les préconditions d'[`analytic_signal`].
#[must_use]
pub fn instantaneous_phase<T: RealScalar>(x: &[T]) -> Vec<T> {
    let z = analytic_signal(x);
    let mut phases: Vec<T> = z.iter().map(|c| c.im.atan2(c.re)).collect();
    super::freqz::unwrap_phase(&mut phases);
    phases
}

/// Fréquence instantanée `f[n] = (1/2π)·dφ/dt` (Hz), estimée par différences
/// finies sur la phase instantanée **déballée** ([`instantaneous_phase`]) et
/// mise à l'échelle par `sample_rate` (Hz) : centrées à l'intérieur, décentrées
/// aux deux extrémités (même schéma que [`super::freqz::group_delay`]). C'est
/// la démodulation FM.
///
/// Panique selon les préconditions d'[`analytic_signal`].
#[must_use]
pub fn instantaneous_frequency<T: RealScalar>(x: &[T], sample_rate: T) -> Vec<T> {
    let phases = instantaneous_phase(x);
    let n = phases.len();
    // f = (fs/2π)·dφ/dn  (dt = 1/fs entre deux échantillons consécutifs).
    let two_pi = T::from_i32(2) * T::pi();
    let scale = sample_rate * two_pi.recip();
    let half = T::from_i32(2).recip();
    let mut out = vec![T::zero(); n];
    out[0] = scale * (phases[1] - phases[0]);
    out[n - 1] = scale * (phases[n - 1] - phases[n - 2]);
    for i in 1..n - 1
    {
        out[i] = scale * (phases[i + 1] - phases[i - 1]) * half;
    }
    out
}
