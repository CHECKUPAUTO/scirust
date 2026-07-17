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
// | [`kaiser`] | `I₀(β·√(1−r²)) / I₀(β)`, `r = 2n/len − 1` | réglable (`β`) | réglable (`β`) |
//
// Les constantes (`0.54`, `0.46`, `0.42`, `0.08`, `0.36`, `0.49`, `0.14`,
// `0.01`) sont construites en ratios d'entiers (`from_i32(a) *
// from_i32(b).recip()`), comme pour toute constante générique du crate —
// exactes en flottant, résolues à la résolution de `T` en virgule fixe.
// Celles de Blackman-Harris sont arrondies au centième (`0.35875 → 0.36`,
// etc., toujours de somme `1`) : les coefficients théoriques ont un
// dénominateur de `100000`, hors de la plage représentable par les formats
// virgule fixe les plus étroits testés ici (ex. `Q8_24`, entiers `≤ 127`).
//
// ## Fenêtre de Kaiser : un paramètre, un compromis réglable
//
// Contrairement aux fenêtres ci-dessus (formule fixe), [`kaiser`] prend un
// paramètre de forme `β` : plus `β` est grand, plus le lobe principal
// s'élargit et plus les lobes secondaires s'atténuent — un seul réglage entre
// les deux extrêmes que les autres fenêtres fixent chacune à un point. Repose
// sur [`RealScalar::bessel_i0`] (fonction de Bessel modifiée, ordre 0) : `r =
// 2n/len − 1` n'étant pas construit à partir d'un dénominateur fixe (`len` est
// une valeur d'exécution quelconque, pas nécessairement une puissance de
// deux), `kaiser_coeff`/`kaiser_into`/`kaiser` demandent `T: RealScalar +
// Div<Output = T>` et utilisent la division réelle (`/`), pas `recip()` —
// même raison que [`crate::dsp::mel`].
//
// En virgule fixe, le domaine garanti de `bessel_i0` est `β ∈ [0, 12]`
// (au-delà, `I₀(β)` dépasse la plage représentable de `Q16.16` et sature).

use core::ops::Div;

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

/// Coefficient `n` (sur `len`) de la fenêtre de Kaiser périodique, paramètre
/// de forme `beta` (typiquement `4` à `9` ; `0` dégénère en fenêtre
/// rectangulaire, `I₀(0)/I₀(0) = 1` partout).
///
/// Panique si `len == 0`.
#[inline]
#[must_use]
pub fn kaiser_coeff<T: RealScalar + Div<Output = T>>(n: usize, len: usize, beta: T) -> T {
    assert!(len >= 1, "fenêtre : len doit être ≥ 1");
    // Diviser *avant* de multiplier par `n` : `2n/len ∈ [0, 2)` tient dans `T`,
    // mais le produit intermédiaire `2·n` peut le dépasser largement pour un
    // format étroit et une fenêtre longue (même piège que `dsp::mel`).
    let two_over_len = T::from_i32(2) / T::from_i32(len as i32);
    let r = T::from_i32(n as i32) * two_over_len - T::one();
    let arg = beta * (T::one() - r * r).sqrt();
    arg.bessel_i0() / beta.bessel_i0()
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

/// Remplit `out` avec la fenêtre de Kaiser périodique (paramètre `beta`) de
/// longueur `out.len()`.
#[inline]
pub fn kaiser_into<T: RealScalar + Div<Output = T>>(out: &mut [T], beta: T) {
    let len = out.len();
    for (n, w) in out.iter_mut().enumerate()
    {
        *w = kaiser_coeff(n, len, beta);
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

/// Fenêtre de Kaiser périodique (paramètre `beta`) de longueur `len`.
#[must_use]
pub fn kaiser<T: RealScalar + Div<Output = T>>(len: usize, beta: T) -> Vec<T> {
    let mut w = vec![T::zero(); len];
    kaiser_into(&mut w, beta);
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
