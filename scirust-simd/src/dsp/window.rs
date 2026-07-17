// scirust-simd/src/dsp/window.rs
//
// # Fenêtres d'apodisation génériques (analyse spectrale)
//
// Fenêtres de pondération temporelle usuelles à appliquer avant une FFT
// ([`super::fft`]) pour réduire la fuite spectrale (« leakage ») causée par la
// troncature brutale d'un signal non périodique dans la fenêtre d'analyse.
// Génériques sur [`RealScalar`] (`cos`) : même code pour le flottant et la
// virgule fixe déterministe (`FixedI32<FRAC>`).
//
// ## Convention : périodique, pas symétrique
//
// `w[n] = f(2·π·n / len)` (dénominateur `len`, pas `len − 1`) — la convention
// **périodique**, adaptée à l'analyse spectrale via FFT : elle évite de
// dupliquer un point identique aux deux bords de la fenêtre (contrairement à
// la convention « symétrique » utilisée pour la conception de filtres FIR).
//
// ## Fenêtres fournies
//
// | Fenêtre | Formule | Lobe principal | Atténuation des lobes secondaires |
// |---|---|---|---|
// | [`hann`] | `0.5 − 0.5·cos(θ)` | étroit | modérée (~31 dB) |
// | [`hamming`] | `0.54 − 0.46·cos(θ)` | étroit | meilleure au premier lobe (~43 dB) |
// | [`blackman`] | `0.42 − 0.5·cos(θ) + 0.08·cos(2θ)` | plus large | excellente (~58 dB) |
// | [`blackman_harris`] | `0.36 − 0.49·cos(θ) + 0.14·cos(2θ) − 0.01·cos(3θ)` | plus large encore | quasi optimale (~92 dB) |
//
// Les constantes (`0.54`, `0.46`, `0.42`, `0.08`, `0.36`, `0.49`, `0.14`,
// `0.01`) sont construites en ratios d'entiers (`from_i32(a) *
// from_i32(b).recip()`), comme pour toute constante générique du crate —
// exactes en flottant, résolues à la résolution de `T` en virgule fixe.
// Celles de Blackman-Harris sont arrondies au centième (`0.35875 → 0.36`,
// etc., toujours de somme `1`) : les coefficients théoriques ont un
// dénominateur de `100000`, hors de la plage représentable par les formats
// virgule fixe les plus étroits testés ici (ex. `Q8_24`, entiers `≤ 127`).

use crate::fixed::RealScalar;

/// `2·π·n / len`. Panique si `len == 0`.
#[inline]
fn angle<T: RealScalar>(n: usize, len: usize) -> T {
    assert!(len >= 1, "fenêtre : len doit être ≥ 1");
    let two_pi = T::from_i32(2) * T::pi();
    let inv_len = T::from_i32(len as i32).recip();
    two_pi * T::from_i32(n as i32) * inv_len
}

/// Coefficient `n` (sur `len`) de la fenêtre de Hann périodique.
#[inline]
#[must_use]
pub fn hann_coeff<T: RealScalar>(n: usize, len: usize) -> T {
    let half = T::from_i32(2).recip();
    half - half * angle::<T>(n, len).cos()
}

/// Coefficient `n` (sur `len`) de la fenêtre de Hamming périodique.
#[inline]
#[must_use]
pub fn hamming_coeff<T: RealScalar>(n: usize, len: usize) -> T {
    let a = T::from_i32(27) * T::from_i32(50).recip(); // 0.54
    let b = T::from_i32(23) * T::from_i32(50).recip(); // 0.46
    a - b * angle::<T>(n, len).cos()
}

/// Coefficient `n` (sur `len`) de la fenêtre de Blackman périodique.
#[inline]
#[must_use]
pub fn blackman_coeff<T: RealScalar>(n: usize, len: usize) -> T {
    let th = angle::<T>(n, len);
    let a0 = T::from_i32(21) * T::from_i32(50).recip(); // 0.42
    let a1 = T::from_i32(2).recip(); // 0.5
    let a2 = T::from_i32(4) * T::from_i32(50).recip(); // 0.08
    a0 - a1 * th.cos() + a2 * (th + th).cos()
}

/// Coefficient `n` (sur `len`) de la fenêtre de Blackman-Harris (4 termes)
/// périodique.
#[inline]
#[must_use]
pub fn blackman_harris_coeff<T: RealScalar>(n: usize, len: usize) -> T {
    let th = angle::<T>(n, len);
    let a0 = T::from_i32(36) * T::from_i32(100).recip(); // 0.36
    let a1 = T::from_i32(49) * T::from_i32(100).recip(); // 0.49
    let a2 = T::from_i32(14) * T::from_i32(100).recip(); // 0.14
    let a3 = T::from_i32(100).recip(); // 0.01
    a0 - a1 * th.cos() + a2 * (th + th).cos() - a3 * (th + th + th).cos()
}

/// Remplit `out` avec la fenêtre de Hann périodique de longueur `out.len()`.
#[inline]
pub fn hann_into<T: RealScalar>(out: &mut [T]) {
    let len = out.len();
    for (n, w) in out.iter_mut().enumerate()
    {
        *w = hann_coeff(n, len);
    }
}

/// Remplit `out` avec la fenêtre de Hamming périodique de longueur `out.len()`.
#[inline]
pub fn hamming_into<T: RealScalar>(out: &mut [T]) {
    let len = out.len();
    for (n, w) in out.iter_mut().enumerate()
    {
        *w = hamming_coeff(n, len);
    }
}

/// Remplit `out` avec la fenêtre de Blackman périodique de longueur
/// `out.len()`.
#[inline]
pub fn blackman_into<T: RealScalar>(out: &mut [T]) {
    let len = out.len();
    for (n, w) in out.iter_mut().enumerate()
    {
        *w = blackman_coeff(n, len);
    }
}

/// Remplit `out` avec la fenêtre de Blackman-Harris périodique de longueur
/// `out.len()`.
#[inline]
pub fn blackman_harris_into<T: RealScalar>(out: &mut [T]) {
    let len = out.len();
    for (n, w) in out.iter_mut().enumerate()
    {
        *w = blackman_harris_coeff(n, len);
    }
}

/// Fenêtre de Hann périodique de longueur `len`.
#[must_use]
pub fn hann<T: RealScalar>(len: usize) -> Vec<T> {
    let mut w = vec![T::zero(); len];
    hann_into(&mut w);
    w
}

/// Fenêtre de Hamming périodique de longueur `len`.
#[must_use]
pub fn hamming<T: RealScalar>(len: usize) -> Vec<T> {
    let mut w = vec![T::zero(); len];
    hamming_into(&mut w);
    w
}

/// Fenêtre de Blackman périodique de longueur `len`.
#[must_use]
pub fn blackman<T: RealScalar>(len: usize) -> Vec<T> {
    let mut w = vec![T::zero(); len];
    blackman_into(&mut w);
    w
}

/// Fenêtre de Blackman-Harris (4 termes) périodique de longueur `len`.
#[must_use]
pub fn blackman_harris<T: RealScalar>(len: usize) -> Vec<T> {
    let mut w = vec![T::zero(); len];
    blackman_harris_into(&mut w);
    w
}

/// Applique `window` à `signal` en place (produit élément par élément).
///
/// Panique si `signal.len() != window.len()`.
#[inline]
pub fn apply<T: RealScalar>(signal: &mut [T], window: &[T]) {
    assert_eq!(
        signal.len(),
        window.len(),
        "apply : signal de longueur {} ≠ fenêtre de longueur {}",
        signal.len(),
        window.len()
    );
    for (s, &w) in signal.iter_mut().zip(window)
    {
        *s = *s * w;
    }
}
