// scirust-simd/src/transformed/special.rs
//
// # Fonctions spéciales (famille Gamma), 100 % Rust
//
// `ln_gamma`, `gamma` et `digamma` (ψ) pour argument réel **strictement
// positif**, sans FFI ni dépendance externe. Ces primitives servent les
// transformations `ReciprocalGamma` et `LogGamma` (cf. modules frères).
//
// ## Méthodes (standard, non inventées)
//
// * `ln_gamma` : approximation de **Lanczos** (`g = 7`, 9 coefficients), la
//   formule de référence pour `ln Γ(z)`, `z > 0`. Erreur relative ≲ 1e-15 sur
//   `z ∈ [0.5, ∞)` (mesurée dans les tests contre des valeurs exactes connues :
//   `Γ(n) = (n−1)!`, `Γ(½) = √π`).
// * `gamma` = `exp(ln_gamma)` (notre domaine `z > 0` donne `Γ(z) > 0`, donc pas
//   de gestion de signe).
// * `digamma` : récurrence `ψ(z) = ψ(z+1) − 1/z` jusqu'à `z ≥ 6`, puis série
//   **asymptotique** `ψ(z) ≈ ln z − 1/2z − Σ B₂ₖ/(2k·z²ᵏ)`.
//
// ## Domaine
//
// Toutes exigent `z > 0`. En dehors, le comportement n'est **pas** défini ici :
// les transformations appelantes restreignent leur domaine (`x > −1` ⇒
// `z = x+1 > 0`) et renvoient une erreur de domaine explicite au préalable.

/// `ln(2π)`, constante de la formule de Lanczos.
const LN_2PI: f64 = 1.837_877_066_409_345_3;

/// Paramètre `g` de Lanczos (décalage).
const LANCZOS_G: f64 = 7.0;

/// Coefficients de Lanczos (`g = 7`, `n = 9`) — série de référence.
const LANCZOS_C: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// Argument `z*` où `Γ(z)` (et donc `ln Γ`, `1/Γ`) atteint son **extremum**
/// sur `(0, ∞)` : l'unique zéro de la digamma, `ψ(z*) = 0`.
///
/// `Γ` y est **minimale** (`Γ(z*) ≈ 0.885603`). C'est le point de séparation des
/// deux branches monotones utilisé par le décodage des transformations Gamma.
pub const GAMMA_ARGMIN: f64 = 1.461_632_144_968_362_3;

/// `ln Γ(z)` pour `z > 0` (approximation de Lanczos).
#[inline]
#[must_use]
pub fn ln_gamma(z: f64) -> f64 {
    debug_assert!(z > 0.0, "ln_gamma: z doit être > 0");
    let z = z - 1.0;
    let mut acc = LANCZOS_C[0];
    for (i, &c) in LANCZOS_C.iter().enumerate().skip(1)
    {
        acc += c / (z + i as f64);
    }
    let t = z + LANCZOS_G + 0.5;
    0.5 * LN_2PI + (z + 0.5) * t.ln() - t + acc.ln()
}

/// `Γ(z)` pour `z > 0`.
#[inline]
#[must_use]
pub fn gamma(z: f64) -> f64 {
    ln_gamma(z).exp()
}

/// `ψ(z) = d/dz ln Γ(z)` (digamma) pour `z > 0`.
#[inline]
#[must_use]
pub fn digamma(mut z: f64) -> f64 {
    debug_assert!(z > 0.0, "digamma: z doit être > 0");
    let mut acc = 0.0;
    // Récurrence ψ(z) = ψ(z+1) − 1/z jusqu'à z ≥ 10 (série asymptotique alors
    // précise à ~1e-13 : le reste tronqué est en O(1/z⁸)).
    while z < 10.0
    {
        acc -= 1.0 / z;
        z += 1.0;
    }
    // ψ(z) ≈ ln z − 1/(2z) − Σ B₂ₖ/(2k z²ᵏ) (B₂=1/6, B₄=−1/30, B₆=1/42).
    let f = 1.0 / (z * z);
    acc + z.ln() - 0.5 / z - f * (1.0 / 12.0 - f * (1.0 / 120.0 - f * (1.0 / 252.0)))
}
