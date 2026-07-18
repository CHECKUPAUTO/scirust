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
// [`batch_norm`]/[`batch_norm_batched`] complètent ce module côté CNN
// ([`super::conv2d`]/[`super::pool2d`]) plutôt que Transformer : même forme
// `(x − μ)/√(σ² + eps)·γ + β`, mais `μ`/`σ²` (« running mean/var ») sont
// **précalculées à l'entraînement** et figées à l'inférence — un paramètre
// **par canal**, pas une statistique recalculée par ligne comme
// `layer_norm`. S'insère dans la chaîne `conv2d → batch_norm → activation`
// (avant [`super::pool2d`]).
//
// `rmsnorm`/`layer_norm`/`batch_norm` sont génériques sur `T: FixedReducible +
// NumericScalar` — donc sur les deux stockages `i32`/`i64` (aucune
// transcendante Q32 requise, juste `sqrt` et des divisions, disponibles pour
// tout stockage). `rope_apply` est **réservé au stockage `i32`** : les angles
// font intervenir `log2`/`exp2`/`sin`/`cos` ([`super::traits::RealScalar`]),
// réservées à la précision interne Q32.
//
// ## Division réelle, pas réciproque isolée
//
// Chaque élément de sortie de `rmsnorm`/`layer_norm`/`batch_norm` divise par
// l'écart-type (ou la racine de la moyenne des carrés) via une division
// réelle **vérifiée** ([`FixedReducible::checked_div`]), jamais une
// réciproque isolée calculée une fois puis multipliée `d` fois : `d`
// divisions exactes (chacune à ≤ 0.5 ULP par l'accumulateur élargi) valent
// mieux qu'une réciproque quantifiée une seule fois et réutilisée, qui
// introduirait un biais commun à tous les canaux (même leçon que `dsp::mel`,
// les décompositions de `fixed::linalg`, `fixed::attention`). `batch_norm`
// calcule néanmoins `√(σ² + eps)` **une seule fois par canal** (pas par
// position spatiale) : la racine ne dépend que du canal, la recalculer à
// chaque position serait un travail redondant, pas une économie de précision
// — la division elle-même, elle, reste individuelle à chaque élément.
//
// `None` si un écart-type (ou une racine quadratique moyenne) plus `eps` vaut
// zéro pour une ligne/un canal (normalisation indéfinie) ou si une division
// déborde — propriété des **données**, pas un bug d'appelant.

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
//  BatchNorm (inférence) — chaîne CNN                                 //
// ------------------------------------------------------------------ //

/// Normalisation par lot (BatchNorm), **inférence** : `y[c,:] = (x[c,:] −
/// running_mean[c]) / √(running_var[c] + eps) · gamma[c] + beta[c]`.
///
/// `x`/sortie sont `channels × spatial` row-major (même convention que
/// [`super::conv2d`]/[`super::pool2d`] — `spatial = height·width` pour des
/// données 2D, `spatial = length` pour du 1D, `spatial = 1` après un
/// aplatissement global) ; `running_mean`/`running_var`/`gamma`/`beta` ont
/// `channels` éléments (statistiques figées à l'entraînement, **pas**
/// recalculées ici — à la différence de [`layer_norm`], cf. en-tête de
/// module).
///
/// `None` si `running_var[c] + eps ≤ 0` pour un canal (racine indéfinie), ou
/// en cas de débordement d'une division. Panique si les longueurs de slice
/// ne correspondent pas aux dimensions annoncées.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn batch_norm<T>(
    x: &[T],
    channels: usize,
    spatial: usize,
    running_mean: &[T],
    running_var: &[T],
    gamma: &[T],
    beta: &[T],
    eps: T,
) -> Option<Vec<T>>
where
    T: FixedReducible + NumericScalar,
{
    assert_eq!(
        x.len(),
        channels * spatial,
        "batch_norm : x de longueur {} ≠ {channels}×{spatial}",
        x.len()
    );
    assert_eq!(
        running_mean.len(),
        channels,
        "batch_norm : running_mean de longueur {} ≠ {channels}",
        running_mean.len()
    );
    assert_eq!(
        running_var.len(),
        channels,
        "batch_norm : running_var de longueur {} ≠ {channels}",
        running_var.len()
    );
    assert_eq!(
        gamma.len(),
        channels,
        "batch_norm : gamma de longueur {} ≠ {channels}",
        gamma.len()
    );
    assert_eq!(
        beta.len(),
        channels,
        "batch_norm : beta de longueur {} ≠ {channels}",
        beta.len()
    );

    let mut y = vec![T::ZERO; channels * spatial];
    for c in 0..channels
    {
        let denom = (running_var[c] + eps).sqrt();
        let (mean, g, b) = (running_mean[c], gamma[c], beta[c]);
        let row = &x[c * spatial..(c + 1) * spatial];
        let out_row = &mut y[c * spatial..(c + 1) * spatial];
        for (o, &xi) in out_row.iter_mut().zip(row)
        {
            *o = (xi - mean).checked_div(denom)? * g + b;
        }
    }
    Some(y)
}

/// [`batch_norm`] **par lot** : `x` est `batch × channels × spatial` (un
/// échantillon par bloc, contigu), les statistiques et le gain/décalage
/// restant **partagés** entre tous les échantillons du lot (propriété de
/// BatchNorm — contrairement à [`super::conv2d::conv2d_batch`], ce n'est pas
/// un GEMM à fusionner, juste `batch` applications indépendantes de
/// [`batch_norm`]). Résultat **identique bit-à-bit** à `batch` appels de
/// [`batch_norm`] concaténés (vérifié par test).
///
/// Panique si `x.len() != batch·channels·spatial`, ou selon les
/// préconditions de [`batch_norm`].
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn batch_norm_batched<T>(
    x: &[T],
    batch: usize,
    channels: usize,
    spatial: usize,
    running_mean: &[T],
    running_var: &[T],
    gamma: &[T],
    beta: &[T],
    eps: T,
) -> Option<Vec<T>>
where
    T: FixedReducible + NumericScalar,
{
    let sample_len = channels * spatial;
    assert_eq!(
        x.len(),
        batch * sample_len,
        "batch_norm_batched : x de longueur {} ≠ {batch}×{channels}×{spatial}",
        x.len()
    );
    let mut y = Vec::with_capacity(batch * sample_len);
    for sample in x.chunks_exact(sample_len)
    {
        y.extend(batch_norm(
            sample,
            channels,
            spatial,
            running_mean,
            running_var,
            gamma,
            beta,
            eps,
        )?);
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
