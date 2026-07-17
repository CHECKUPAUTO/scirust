// scirust-simd/src/fixed/norm.rs
//
// # Normalisations quantifiées déterministes (RMSNorm, LayerNorm)
//
// [`rmsnorm`]/[`layer_norm`] : les deux normalisations Transformer usuelles,
// en virgule fixe déterministe — le pendant quantifié du module flottant
// [`crate::norm`] (vectorisé AVX-512, non déterministe). Combinées à
// [`super::attention`] et [`super::layer`], elles complètent le bloc
// Transformer (Attention → Add & Norm → FFN → Add & Norm) entièrement en
// virgule fixe, reproductible bit-à-bit.
//
// * [`rmsnorm`] : `y = x / √(moyenne(x²) + eps) · γ` (LLaMA, Mistral…).
// * [`layer_norm`] : `y = (x − μ) / √(σ² + eps) · γ + β`.
//
// Génériques sur `T: FixedReducible + NumericScalar` — donc sur les deux
// stockages `i32`/`i64` (aucune transcendante Q32 requise ici, juste `sqrt` et
// des divisions, disponibles pour tout stockage).
//
// ## Division réelle, pas réciproque isolée
//
// Chaque élément de sortie divise par l'écart-type (ou la racine de la
// moyenne des carrés) via une division réelle **vérifiée**
// ([`FixedReducible::checked_div`]), jamais une réciproque isolée calculée
// une fois puis multipliée `d` fois : `d` divisions exactes (chacune à
// ≤ 0.5 ULP par l'accumulateur élargi) valent mieux qu'une réciproque
// quantifiée une seule fois et réutilisée, qui introduirait un biais commun
// à tous les canaux (même leçon que `dsp::mel`, les décompositions de
// `fixed::linalg`, `fixed::attention`).
//
// `None` si un écart-type (ou une racine quadratique moyenne) plus `eps` vaut
// zéro pour une ligne (normalisation indéfinie) ou si une division déborde —
// propriété des **données**, pas un bug d'appelant.

use super::reductions::{FixedReducible, dot, sum};
use super::traits::NumericScalar;

/// RMSNorm par ligne : `y[r,:] = x[r,:] / √(moyenne(x[r,:]²) + eps) · gamma`.
///
/// `x`/sortie sont `rows × d` row-major ; `gamma` a `d` éléments (gain par
/// canal). `None` si une ligne a une racine quadratique moyenne (+ `eps`)
/// nulle, ou en cas de débordement d'une division. Panique si les longueurs
/// de slice ne correspondent pas aux dimensions annoncées.
#[must_use]
pub fn rmsnorm<T>(x: &[T], rows: usize, d: usize, gamma: &[T], eps: T) -> Option<Vec<T>>
where
    T: FixedReducible + NumericScalar,
{
    assert_eq!(
        x.len(),
        rows * d,
        "rmsnorm : x de longueur {} ≠ {rows}×{d}",
        x.len()
    );
    assert_eq!(
        gamma.len(),
        d,
        "rmsnorm : gamma de longueur {} ≠ {d}",
        gamma.len()
    );

    let d_t = T::from_i32(d as i32);
    let mut y = vec![T::ZERO; rows * d];
    for r in 0..rows
    {
        let row = &x[r * d..r * d + d];
        let mean_sq = dot(row, row).checked_div(d_t)?;
        let rms = (mean_sq + eps).sqrt();
        for (i, &xi) in row.iter().enumerate()
        {
            y[r * d + i] = xi.checked_div(rms)? * gamma[i];
        }
    }
    Some(y)
}

/// LayerNorm par ligne : `y[r,:] = (x[r,:] − μ) / √(σ² + eps) · gamma + beta`,
/// `μ`/`σ²` étant la moyenne/variance de la ligne.
///
/// `x`/sortie sont `rows × d` row-major ; `gamma`/`beta` ont `d` éléments.
/// `None` si l'écart-type (+ `eps`) d'une ligne est nul, ou en cas de
/// débordement d'une division. Panique si les longueurs de slice ne
/// correspondent pas aux dimensions annoncées.
#[must_use]
pub fn layer_norm<T>(
    x: &[T],
    rows: usize,
    d: usize,
    gamma: &[T],
    beta: &[T],
    eps: T,
) -> Option<Vec<T>>
where
    T: FixedReducible + NumericScalar,
{
    assert_eq!(
        x.len(),
        rows * d,
        "layer_norm : x de longueur {} ≠ {rows}×{d}",
        x.len()
    );
    assert_eq!(
        gamma.len(),
        d,
        "layer_norm : gamma de longueur {} ≠ {d}",
        gamma.len()
    );
    assert_eq!(
        beta.len(),
        d,
        "layer_norm : beta de longueur {} ≠ {d}",
        beta.len()
    );

    let d_t = T::from_i32(d as i32);
    let mut y = vec![T::ZERO; rows * d];
    let mut centered = vec![T::ZERO; d];
    for r in 0..rows
    {
        let row = &x[r * d..r * d + d];
        let mean = sum(row).checked_div(d_t)?;
        for (c, &xi) in centered.iter_mut().zip(row)
        {
            *c = xi - mean;
        }
        let var = dot(&centered, &centered).checked_div(d_t)?;
        let denom = (var + eps).sqrt();
        for (i, &c) in centered.iter().enumerate()
        {
            y[r * d + i] = c.checked_div(denom)? * gamma[i] + beta[i];
        }
    }
    Some(y)
}
