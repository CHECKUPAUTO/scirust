//! # Normalisations et encodages positionnels (x86_64)
//!
//! Briques Transformer complémentaires du GEMM/attention :
//!
//! * **RMSNorm** — `y = x / √(moyenne(x²) + eps) · γ` (LLaMA, Mistral…).
//! * **LayerNorm** — `y = (x − μ) / √(σ² + eps) · γ + β`.
//! * **RoPE** — rotary positional embedding, rotation des paires `(2i, 2i+1)`
//!   par un angle dépendant de la position.
//!
//! Les normalisations sont **vectorisées AVX-512** : réductions (somme, somme
//! des carrés) par `_mm512_reduce_add_ps`, puis mise à l'échelle par voie ;
//! repli scalaire garanti. Toutes opèrent **par ligne** (`rows × d`, row-major)
//! et en place. Vérifiées contre une référence scalaire dans les tests.

#![allow(clippy::missing_safety_doc)]

// ===================================================================== //
//  RMSNorm                                                               //
// ===================================================================== //

/// RMSNorm en place, par ligne : `x[r,:] ← x[r,:] / √(moyenne(x²) + eps) · γ`.
/// `gamma` est le gain par canal (longueur `d`).
pub fn rmsnorm(x: &mut [f32], rows: usize, d: usize, gamma: &[f32], eps: f32) {
    assert_eq!(x.len(), rows * d, "rmsnorm: shape mismatch");
    assert_eq!(gamma.len(), d, "rmsnorm: gamma length != d");
    if d == 0
    {
        return;
    }
    for r in 0..rows
    {
        let row = &mut x[r * d..r * d + d];
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f")
            {
                // SAFETY: gated by the runtime detection just above.
                unsafe { rmsnorm_row_avx512(row, gamma, eps) };
                continue;
            }
        }
        rmsnorm_row_scalar(row, gamma, eps);
    }
}

fn rmsnorm_row_scalar(row: &mut [f32], gamma: &[f32], eps: f32) {
    let d = row.len();
    let ss: f32 = row.iter().map(|&v| v * v).sum();
    let inv = 1.0 / (ss / d as f32 + eps).sqrt();
    for (v, &g) in row.iter_mut().zip(gamma)
    {
        *v = *v * inv * g;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn rmsnorm_row_avx512(row: &mut [f32], gamma: &[f32], eps: f32) {
    use core::arch::x86_64::*;
    let d = row.len();
    // Σ x².
    let mut acc = _mm512_setzero_ps();
    let mut i = 0;
    while i + 16 <= d
    {
        let v = _mm512_loadu_ps(row.as_ptr().add(i));
        acc = _mm512_fmadd_ps(v, v, acc);
        i += 16;
    }
    let rem = d - i;
    if rem > 0
    {
        let mask = (1u16 << rem) - 1;
        let v = _mm512_maskz_loadu_ps(mask, row.as_ptr().add(i));
        acc = _mm512_fmadd_ps(v, v, acc);
    }
    let ss = _mm512_reduce_add_ps(acc);
    let inv = 1.0 / (ss / d as f32 + eps).sqrt();

    // x = x · inv · γ.
    let invv = _mm512_set1_ps(inv);
    let mut i = 0;
    while i + 16 <= d
    {
        let v = _mm512_loadu_ps(row.as_ptr().add(i));
        let g = _mm512_loadu_ps(gamma.as_ptr().add(i));
        _mm512_storeu_ps(
            row.as_mut_ptr().add(i),
            _mm512_mul_ps(_mm512_mul_ps(v, invv), g),
        );
        i += 16;
    }
    if rem > 0
    {
        let mask = (1u16 << rem) - 1;
        let v = _mm512_maskz_loadu_ps(mask, row.as_ptr().add(i));
        let g = _mm512_maskz_loadu_ps(mask, gamma.as_ptr().add(i));
        _mm512_mask_storeu_ps(
            row.as_mut_ptr().add(i),
            mask,
            _mm512_mul_ps(_mm512_mul_ps(v, invv), g),
        );
    }
}

// ===================================================================== //
//  LayerNorm                                                             //
// ===================================================================== //

/// LayerNorm en place, par ligne :
/// `x[r,:] ← (x[r,:] − μ) / √(σ² + eps) · γ + β`, avec `μ`/`σ²` la moyenne et
/// la variance de la ligne. `gamma`/`beta` de longueur `d`.
pub fn layernorm(x: &mut [f32], rows: usize, d: usize, gamma: &[f32], beta: &[f32], eps: f32) {
    assert_eq!(x.len(), rows * d, "layernorm: shape mismatch");
    assert_eq!(gamma.len(), d, "layernorm: gamma length != d");
    assert_eq!(beta.len(), d, "layernorm: beta length != d");
    if d == 0
    {
        return;
    }
    for r in 0..rows
    {
        let row = &mut x[r * d..r * d + d];
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f")
            {
                // SAFETY: gated by the runtime detection just above.
                unsafe { layernorm_row_avx512(row, gamma, beta, eps) };
                continue;
            }
        }
        layernorm_row_scalar(row, gamma, beta, eps);
    }
}

fn layernorm_row_scalar(row: &mut [f32], gamma: &[f32], beta: &[f32], eps: f32) {
    let d = row.len() as f32;
    let mean: f32 = row.iter().sum::<f32>() / d;
    let var: f32 = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / d;
    let inv = 1.0 / (var + eps).sqrt();
    for ((v, &g), &b) in row.iter_mut().zip(gamma).zip(beta)
    {
        *v = (*v - mean) * inv * g + b;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn layernorm_row_avx512(row: &mut [f32], gamma: &[f32], beta: &[f32], eps: f32) {
    use core::arch::x86_64::*;
    let d = row.len();
    let dinv = 1.0 / d as f32;

    // Σ x et Σ x² en un passage.
    let mut s = _mm512_setzero_ps();
    let mut sq = _mm512_setzero_ps();
    let mut i = 0;
    while i + 16 <= d
    {
        let v = _mm512_loadu_ps(row.as_ptr().add(i));
        s = _mm512_add_ps(s, v);
        sq = _mm512_fmadd_ps(v, v, sq);
        i += 16;
    }
    let rem = d - i;
    if rem > 0
    {
        let mask = (1u16 << rem) - 1;
        let v = _mm512_maskz_loadu_ps(mask, row.as_ptr().add(i));
        s = _mm512_add_ps(s, v);
        sq = _mm512_fmadd_ps(v, v, sq);
    }
    let sum = _mm512_reduce_add_ps(s);
    let sumsq = _mm512_reduce_add_ps(sq);
    let mean = sum * dinv;
    let var = sumsq * dinv - mean * mean; // E[x²] − E[x]²
    let inv = 1.0 / (var + eps).sqrt();

    // x = (x − μ)·inv·γ + β.
    let meanv = _mm512_set1_ps(mean);
    let invv = _mm512_set1_ps(inv);
    let mut i = 0;
    while i + 16 <= d
    {
        let v = _mm512_loadu_ps(row.as_ptr().add(i));
        let g = _mm512_loadu_ps(gamma.as_ptr().add(i));
        let b = _mm512_loadu_ps(beta.as_ptr().add(i));
        let norm = _mm512_mul_ps(_mm512_mul_ps(_mm512_sub_ps(v, meanv), invv), g);
        _mm512_storeu_ps(row.as_mut_ptr().add(i), _mm512_add_ps(norm, b));
        i += 16;
    }
    if rem > 0
    {
        let mask = (1u16 << rem) - 1;
        let v = _mm512_maskz_loadu_ps(mask, row.as_ptr().add(i));
        let g = _mm512_maskz_loadu_ps(mask, gamma.as_ptr().add(i));
        let b = _mm512_maskz_loadu_ps(mask, beta.as_ptr().add(i));
        let norm = _mm512_mul_ps(_mm512_mul_ps(_mm512_sub_ps(v, meanv), invv), g);
        _mm512_mask_storeu_ps(row.as_mut_ptr().add(i), mask, _mm512_add_ps(norm, b));
    }
}

// ===================================================================== //
//  RoPE — rotary positional embedding                                    //
// ===================================================================== //

/// Applique RoPE en place à chaque ligne de `x` (`rows × d`, `d` pair).
///
/// La ligne `r` est à la position `pos_offset + r`. On rote les paires
/// `(x[2i], x[2i+1])` par l'angle `θ_i · pos`, avec `θ_i = base^(−2i/d)`
/// (convention du papier RoPE original, `base = 10000` typiquement) :
///
/// ```text
/// x'[2i]   = x[2i]·cos − x[2i+1]·sin
/// x'[2i+1] = x[2i]·sin + x[2i+1]·cos
/// ```
///
/// Les angles font intervenir `sin`/`cos` (transcendantes) : implémentation
/// scalaire correcte (les tables `cos`/`sin` dominent le coût ; la rotation
/// elle-même est mémoire-bornée). Se combine avec l'attention en appliquant
/// RoPE à `Q` et `K` avant le produit `Q·Kᵀ`.
pub fn rope_apply(x: &mut [f32], rows: usize, d: usize, base: f32, pos_offset: usize) {
    assert_eq!(x.len(), rows * d, "rope_apply: shape mismatch");
    assert_eq!(d % 2, 0, "rope_apply: d doit être pair");
    let half = d / 2;
    for r in 0..rows
    {
        let pos = (pos_offset + r) as f32;
        let row = &mut x[r * d..r * d + d];
        for i in 0..half
        {
            let theta = base.powf(-2.0 * i as f32 / d as f32);
            let angle = pos * theta;
            let (sin, cos) = angle.sin_cos();
            let a = row[2 * i];
            let b = row[2 * i + 1];
            row[2 * i] = a * cos - b * sin;
            row[2 * i + 1] = a * sin + b * cos;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: &[f32], b: &[f32], rel: f32, ctx: &str) {
        assert_eq!(a.len(), b.len(), "{ctx}: len");
        for i in 0..a.len()
        {
            let tol = rel * (1.0 + b[i].abs());
            assert!(
                (a[i] - b[i]).abs() <= tol,
                "{ctx}: idx {i}: {} vs {}",
                a[i],
                b[i]
            );
        }
    }

    #[test]
    fn rmsnorm_matches_scalar_all_lengths() {
        for d in [1usize, 3, 8, 16, 17, 33, 64]
        {
            let rows = 4;
            let x0: Vec<f32> = (0..rows * d)
                .map(|i| (i as f32 * 0.3).sin() * 5.0)
                .collect();
            let gamma: Vec<f32> = (0..d).map(|i| 0.5 + (i as f32) * 0.03).collect();
            let eps = 1e-5;

            let mut got = x0.clone();
            rmsnorm(&mut got, rows, d, &gamma, eps);

            let mut want = x0.clone();
            for r in 0..rows
            {
                rmsnorm_row_scalar(&mut want[r * d..r * d + d], &gamma, eps);
            }
            approx(&got, &want, 1e-5, &format!("rmsnorm d={d}"));
        }
    }

    #[test]
    fn layernorm_matches_scalar_and_normalizes() {
        for d in [2usize, 8, 16, 17, 40]
        {
            let rows = 3;
            let x0: Vec<f32> = (0..rows * d)
                .map(|i| (i as f32 * 0.21).cos() * 3.0 + 1.0)
                .collect();
            let gamma: Vec<f32> = (0..d).map(|i| 1.0 + (i as f32) * 0.01).collect();
            let beta: Vec<f32> = (0..d).map(|i| (i as f32) * 0.02 - 0.1).collect();
            let eps = 1e-5;

            let mut got = x0.clone();
            layernorm(&mut got, rows, d, &gamma, &beta, eps);

            let mut want = x0.clone();
            for r in 0..rows
            {
                layernorm_row_scalar(&mut want[r * d..r * d + d], &gamma, &beta, eps);
            }
            approx(&got, &want, 1e-4, &format!("layernorm d={d}"));
        }

        // Avec γ=1, β=0 : moyenne ≈ 0 et variance ≈ 1 par ligne.
        let (rows, d) = (5, 32);
        let mut x: Vec<f32> = (0..rows * d)
            .map(|i| (i as f32 * 0.7).sin() * 2.0 + 3.0)
            .collect();
        let g = vec![1.0f32; d];
        let b = vec![0.0f32; d];
        layernorm(&mut x, rows, d, &g, &b, 1e-6);
        for r in 0..rows
        {
            let row = &x[r * d..r * d + d];
            let mean: f32 = row.iter().sum::<f32>() / d as f32;
            let var: f32 = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
            assert!(mean.abs() <= 1e-3, "row {r} mean {mean}");
            assert!((var - 1.0).abs() <= 1e-2, "row {r} var {var}");
        }
    }

    #[test]
    fn rope_preserves_norm_and_matches_formula() {
        let (rows, d) = (6, 8);
        let base = 10000.0f32;
        let x0: Vec<f32> = (0..rows * d).map(|i| (i as f32 * 0.13).sin()).collect();

        let mut got = x0.clone();
        rope_apply(&mut got, rows, d, base, 0);

        // Référence explicite + conservation de la norme par paire.
        let half = d / 2;
        for r in 0..rows
        {
            let pos = r as f32;
            for i in 0..half
            {
                let theta = base.powf(-2.0 * i as f32 / d as f32);
                let (sin, cos) = (pos * theta).sin_cos();
                let a = x0[r * d + 2 * i];
                let b = x0[r * d + 2 * i + 1];
                let e0 = a * cos - b * sin;
                let e1 = a * sin + b * cos;
                assert!((got[r * d + 2 * i] - e0).abs() <= 1e-5);
                assert!((got[r * d + 2 * i + 1] - e1).abs() <= 1e-5);
                // rotation → norme de la paire conservée.
                let n_in = a * a + b * b;
                let n_out = got[r * d + 2 * i] * got[r * d + 2 * i]
                    + got[r * d + 2 * i + 1] * got[r * d + 2 * i + 1];
                assert!((n_in - n_out).abs() <= 1e-4, "norm pair r={r} i={i}");
            }
        }
    }

    #[test]
    fn rope_position_zero_is_identity() {
        // À la position 0, tous les angles valent 0 → rotation identité.
        let (rows, d) = (1, 16);
        let x0: Vec<f32> = (0..d).map(|i| (i as f32) - 8.0).collect();
        let mut got = x0.clone();
        rope_apply(&mut got, rows, d, 10000.0, 0);
        approx(&got, &x0, 1e-6, "rope pos0 identity");
    }
}
