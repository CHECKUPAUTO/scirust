// scirust-simd/src/fixed/norm.rs
//
// # Normalisations et encodage positionnel quantifiés déterministes
//
// [`rmsnorm`]/[`layer_norm`]/[`rope_apply`] : les briques Transformer usuelles
// complémentaires du GEMM/attention, en virgule fixe déterministe — le
// pendant quantifié du module flottant [`crate::norm`] (vectorisé AVX-512,
// non déterministe). Combinées à [`super::attention`] et [`super::layer`],
// elles complètent le bloc Transformer (RoPE sur `Q`/`K` → Attention →
// Add & Norm → FFN → Add & Norm) entièrement en virgule fixe, reproductible
// bit-à-bit.
//
// * [`rmsnorm`] : `y = x / √(moyenne(x²) + eps) · γ` (LLaMA, Mistral…).
// * [`layer_norm`] : `y = (x − μ) / √(σ² + eps) · γ + β`.
// * [`rope_apply`] : rotation des paires `(x[2i], x[2i+1])` par un angle
//   dépendant de la position (rotary positional embedding).
//
// `rmsnorm`/`layer_norm` sont génériques sur `T: FixedReducible +
// NumericScalar` — donc sur les deux stockages `i32`/`i64` (aucune
// transcendante Q32 requise, juste `sqrt` et des divisions, disponibles pour
// tout stockage). `rope_apply` est **réservé au stockage `i32`** : les angles
// font intervenir `log2`/`exp2`/`sin`/`cos` ([`super::traits::RealScalar`]),
// réservées à la précision interne Q32.
//
// ## Division réelle, pas réciproque isolée
//
// Chaque élément de sortie de `rmsnorm`/`layer_norm` divise par l'écart-type
// (ou la racine de la moyenne des carrés) via une division réelle **vérifiée**
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
use super::traits::{NumericScalar, RealScalar};
use super::types::Fixed;

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

// ------------------------------------------------------------------ //
//  RoPE — rotary positional embedding                                 //
// ------------------------------------------------------------------ //

/// Applique RoPE en place à chaque ligne de `x` (`rows × d`, `d` pair),
/// déterministe.
///
/// La ligne `r` est à la position `pos_offset + r`. On fait tourner les
/// paires `(x[2i], x[2i+1])` par l'angle `θᵢ·pos`, avec `θᵢ = base^(−2i/d)`
/// (convention du papier RoPE original, `base = 10000` typiquement) :
///
/// ```text
/// x'[2i]   = x[2i]·cos − x[2i+1]·sin
/// x'[2i+1] = x[2i]·sin + x[2i+1]·cos
/// ```
///
/// `θᵢ = exp2(−2i/d · log2(base))` : ne dépend que de `i` (pas de la ligne),
/// donc calculé **une seule fois** pour les `d/2` paires puis réutilisé pour
/// chaque ligne — contrairement au module flottant [`crate::norm::rope_apply`],
/// qui le recalcule à chaque ligne (même résultat, travail redondant en moins
/// ici). `−2i/d` utilise une division réelle (`d` n'est pas nécessairement
/// une puissance de deux).
///
/// Se combine avec [`super::attention`] en appliquant RoPE à `Q` et `K` avant
/// [`super::attention::attention`]. Réservé au stockage `i32`. Panique si
/// `x.len() != rows·d` ou si `d` est impair.
pub fn rope_apply<const FRAC: u32>(
    x: &mut [Fixed<i32, FRAC>],
    rows: usize,
    d: usize,
    base: Fixed<i32, FRAC>,
    pos_offset: usize,
) {
    assert_eq!(
        x.len(),
        rows * d,
        "rope_apply : x de longueur {} ≠ {rows}×{d}",
        x.len()
    );
    assert_eq!(d % 2, 0, "rope_apply : d doit être pair (reçu {d})");
    let half = d / 2;

    let log2_base = base.log2();
    let d_t = Fixed::<i32, FRAC>::from_i32(d as i32);
    let thetas: Vec<Fixed<i32, FRAC>> = (0..half)
        .map(|i| {
            let ratio = Fixed::<i32, FRAC>::from_i32(-2 * i as i32) / d_t;
            (ratio * log2_base).exp2()
        })
        .collect();

    for r in 0..rows
    {
        let pos = Fixed::<i32, FRAC>::from_i32((pos_offset + r) as i32);
        let row = &mut x[r * d..r * d + d];
        for (i, &theta) in thetas.iter().enumerate()
        {
            let angle = pos * theta;
            let (s, c) = (angle.sin(), angle.cos());
            let a = row[2 * i];
            let b = row[2 * i + 1];
            row[2 * i] = a * c - b * s;
            row[2 * i + 1] = a * s + b * c;
        }
    }
}
