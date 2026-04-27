// scirust-simd/src/portable.rs
//
// Portable SIMD kernels using std::simd (nightly).
// Un seul source → le compilateur émet AVX2/SSE2/NEON/SVE
// selon la cible, sans #[target_feature] explicite par branche.
//
// Requis dans Cargo.toml :
//   [features]
//   portable-simd = []
//
// Et dans lib.rs :
//   #![cfg_attr(feature = "portable-simd", feature(portable_simd))]

#![cfg_attr(feature = "portable-simd", feature(portable_simd))]

#[cfg(feature = "portable-simd")]
pub mod simd_ops {
    use std::simd::{
        f32x8, f64x4,
        SimdFloat, StdFloat,
        num::SimdFloat as _,
    };

    // ------------------------------------------------------------------ //
    //  ADDITION                                                            //
    // ------------------------------------------------------------------ //

    /// Additionne deux slices f32 en place : dst[i] += src[i]
    /// Traite 8 éléments par cycle SIMD, scalaire pour le reste.
    #[inline]
    pub fn add_f32_inplace(dst: &mut [f32], src: &[f32]) {
        assert_eq!(dst.len(), src.len(), "add_f32_inplace: longueurs différentes");

        let (pre, mid_dst, suf_dst) = dst.as_simd_mut::<8>();
        let (_, mid_src, _)         = src.as_simd::<8>();

        // Prologue scalaire (alignement)
        for (d, s) in pre.iter_mut().zip(src.iter()) {
            *d += s;
        }

        // Cœur SIMD — 8 f32 en parallèle
        for (vd, vs) in mid_dst.iter_mut().zip(mid_src.iter()) {
            *vd += vs;
        }

        // Épilogue scalaire
        let offset = pre.len() + mid_dst.len() * 8;
        for (d, s) in suf_dst.iter_mut().zip(src[offset..].iter()) {
            *d += s;
        }
    }

    /// Additionne deux slices f64 en place : dst[i] += src[i]
    #[inline]
    pub fn add_f64_inplace(dst: &mut [f64], src: &[f64]) {
        assert_eq!(dst.len(), src.len());

        let (pre, mid_dst, suf_dst) = dst.as_simd_mut::<4>();
        let (_, mid_src, _)         = src.as_simd::<4>();

        for (d, s) in pre.iter_mut().zip(src.iter()) { *d += s; }
        for (vd, vs) in mid_dst.iter_mut().zip(mid_src.iter()) { *vd += vs; }
        let offset = pre.len() + mid_dst.len() * 4;
        for (d, s) in suf_dst.iter_mut().zip(src[offset..].iter()) { *d += s; }
    }

    // ------------------------------------------------------------------ //
    //  MULTIPLICATION SCALAIRE (scale)                                     //
    // ------------------------------------------------------------------ //

    /// Multiplie chaque élément par un scalaire : v[i] *= alpha
    #[inline]
    pub fn scale_f32(v: &mut [f32], alpha: f32) {
        let splat = f32x8::splat(alpha);
        let (pre, mid, suf) = v.as_simd_mut::<8>();
        for x in pre.iter_mut()  { *x *= alpha; }
        for vx in mid.iter_mut() { *vx *= splat; }
        for x in suf.iter_mut()  { *x *= alpha; }
    }

    #[inline]
    pub fn scale_f64(v: &mut [f64], alpha: f64) {
        let splat = f64x4::splat(alpha);
        let (pre, mid, suf) = v.as_simd_mut::<4>();
        for x in pre.iter_mut()  { *x *= alpha; }
        for vx in mid.iter_mut() { *vx *= splat; }
        for x in suf.iter_mut()  { *x *= alpha; }
    }

    // ------------------------------------------------------------------ //
    //  PRODUIT SCALAIRE (dot product)                                      //
    // ------------------------------------------------------------------ //

    /// Produit scalaire SIMD avec accumulation parallèle (8 accus f32).
    /// Réduit le risque de dépendance séquentielle sur l'accumulateur.
    #[inline]
    pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "dot_f32: longueurs différentes");

        let mut acc = f32x8::splat(0.0);

        let (pre_a, mid_a, suf_a) = a.as_simd::<8>();
        let (_, mid_b, _)         = b.as_simd::<8>();

        // Prologue
        let mut scalar_acc = 0.0f32;
        for (x, y) in pre_a.iter().zip(b.iter()) {
            scalar_acc += x * y;
        }

        // Cœur SIMD
        for (va, vb) in mid_a.iter().zip(mid_b.iter()) {
            acc += va * vb;
        }

        // Épilogue
        let offset = pre_a.len() + mid_a.len() * 8;
        for (x, y) in suf_a.iter().zip(b[offset..].iter()) {
            scalar_acc += x * y;
        }

        // Réduction horizontale du vecteur SIMD
        scalar_acc + acc.reduce_sum()
    }

    #[inline]
    pub fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());

        let mut acc = f64x4::splat(0.0);
        let (pre_a, mid_a, suf_a) = a.as_simd::<4>();
        let (_, mid_b, _)         = b.as_simd::<4>();

        let mut scalar_acc = 0.0f64;
        for (x, y) in pre_a.iter().zip(b.iter()) { scalar_acc += x * y; }
        for (va, vb) in mid_a.iter().zip(mid_b.iter()) { acc += va * vb; }
        let offset = pre_a.len() + mid_a.len() * 4;
        for (x, y) in suf_a.iter().zip(b[offset..].iter()) { scalar_acc += x * y; }

        scalar_acc + acc.reduce_sum()
    }

    // ------------------------------------------------------------------ //
    //  FMA — Fused Multiply-Add : dst[i] = a[i]*b[i] + c[i]               //
    //  Exploite les instructions VFMADD231PS/FMLA d'AVX2/NEON              //
    // ------------------------------------------------------------------ //

    #[inline]
    pub fn fma_f32(dst: &mut [f32], a: &[f32], b: &[f32], c: &[f32]) {
        assert!(dst.len() == a.len() && a.len() == b.len() && b.len() == c.len());

        let (pre, mid_dst, suf) = dst.as_simd_mut::<8>();
        let (_, mid_a, _) = a.as_simd::<8>();
        let (_, mid_b, _) = b.as_simd::<8>();
        let (_, mid_c, _) = c.as_simd::<8>();

        for i in 0..pre.len() { pre[i] = a[i] * b[i] + c[i]; }

        for i in 0..mid_dst.len() {
            // mul_add est mappé sur VFMADD quand disponible
            mid_dst[i] = mid_a[i].mul_add(mid_b[i], mid_c[i]);
        }

        let offset = pre.len() + mid_dst.len() * 8;
        for (i, d) in suf.iter_mut().enumerate() {
            *d = a[offset + i] * b[offset + i] + c[offset + i];
        }
    }

    // ------------------------------------------------------------------ //
    //  NORMALISATION L2                                                     //
    // ------------------------------------------------------------------ //

    /// Normalise un vecteur en norme L2 in-place.
    pub fn normalize_f32(v: &mut [f32]) {
        let norm = dot_f32(v, v).sqrt();
        if norm > f32::EPSILON {
            scale_f32(v, 1.0 / norm);
        }
    }

    // ------------------------------------------------------------------ //
    //  ACTIVATION : ReLU et Sigmoid vectorisés                             //
    // ------------------------------------------------------------------ //

    /// ReLU SIMD : max(0, x) pour chaque élément
    #[inline]
    pub fn relu_f32(v: &mut [f32]) {
        let zero = f32x8::splat(0.0);
        let (pre, mid, suf) = v.as_simd_mut::<8>();
        for x in pre.iter_mut()  { *x = x.max(0.0); }
        for vx in mid.iter_mut() { *vx = vx.simd_max(zero); }
        for x in suf.iter_mut()  { *x = x.max(0.0); }
    }
}

// ------------------------------------------------------------------ //
//  Fallback scalar — compilé quand portable-simd n'est pas activé    //
// ------------------------------------------------------------------ //
#[cfg(not(feature = "portable-simd"))]
pub mod simd_ops {
    #[inline]
    pub fn add_f32_inplace(dst: &mut [f32], src: &[f32]) {
        for (d, s) in dst.iter_mut().zip(src.iter()) { *d += s; }
    }
    #[inline]
    pub fn add_f64_inplace(dst: &mut [f64], src: &[f64]) {
        for (d, s) in dst.iter_mut().zip(src.iter()) { *d += s; }
    }
    #[inline]
    pub fn scale_f32(v: &mut [f32], alpha: f32) {
        for x in v.iter_mut() { *x *= alpha; }
    }
    #[inline]
    pub fn scale_f64(v: &mut [f64], alpha: f64) {
        for x in v.iter_mut() { *x *= alpha; }
    }
    #[inline]
    pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
    #[inline]
    pub fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
    #[inline]
    pub fn fma_f32(dst: &mut [f32], a: &[f32], b: &[f32], c: &[f32]) {
        for i in 0..dst.len() { dst[i] = a[i] * b[i] + c[i]; }
    }
    #[inline]
    pub fn normalize_f32(v: &mut [f32]) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON { for x in v.iter_mut() { *x /= norm; } }
    }
    #[inline]
    pub fn relu_f32(v: &mut [f32]) {
        for x in v.iter_mut() { *x = x.max(0.0); }
    }
}

// ------------------------------------------------------------------ //
//  Tests unitaires                                                    //
// ------------------------------------------------------------------ //
#[cfg(test)]
mod tests {
    use super::simd_ops::*;

    #[test]
    fn test_add_f32_inplace() {
        let mut dst = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let src     = vec![0.5f32; 9];
        add_f32_inplace(&mut dst, &src);
        for (i, x) in dst.iter().enumerate() {
            assert!((x - (i as f32 + 1.5)).abs() < 1e-6, "add_f32 failed at {i}");
        }
    }

    #[test]
    fn test_dot_f32() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![4.0f32, 3.0, 2.0, 1.0];
        let result = dot_f32(&a, &b);
        assert!((result - 20.0).abs() < 1e-5, "dot = {result}");
    }

    #[test]
    fn test_relu() {
        let mut v = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
        relu_f32(&mut v);
        assert_eq!(v, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_normalize() {
        let mut v = vec![3.0f32, 4.0];
        normalize_f32(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }
}
