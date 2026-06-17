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
//
// ## Safety
//
// This module uses `std::simd` which provides a safe abstraction over platform SIMD.
// The `as_simd` and `as_simd_mut` methods are `unsafe` in std but are wrapped by
// safe public APIs here. Safety invariants:
// - **Slice bounds**: The `as_simd`/`as_simd_mut` calls are guarded by the slice length
//   checks (`assert_eq!` in public functions). The split into (pre, mid, suf) is guaranteed
//   by the standard library to partition the slice without overlap or out-of-bounds access.
// - **Alignment**: `std::simd` handles alignment requirements internally; the fallback
//   scalar loops handle any misaligned prefix/suffix.
// - **No raw pointers escape**: All SIMD operations stay within the safe abstraction.
//   The `unsafe` blocks in std's implementation are sound because the slice metadata
//   (ptr, len) is valid Rust slice data.
//
// The fallback scalar implementations (when `portable-simd` feature is disabled) are
// entirely safe Rust with no `unsafe` code.
//
// The `portable_simd` feature is declared once at the crate root (`lib.rs`);
// no need to repeat it here.

#[cfg(feature = "portable-simd")]
pub mod simd_ops {
    use std::simd::{StdFloat, f32x8, f64x4, num::SimdFloat};

    // ------------------------------------------------------------------ //
    //  ADDITION                                                            //
    // ------------------------------------------------------------------ //

    /// Additionne deux slices f32 en place : dst[i] += src[i]
    /// Traite 8 éléments par cycle SIMD, scalaire pour le reste.
    #[inline]
    pub fn add_f32_inplace(dst: &mut [f32], src: &[f32]) {
        assert_eq!(
            dst.len(),
            src.len(),
            "add_f32_inplace: longueurs différentes"
        );

        // Chunk both slices identically (chunk k = elements [8k..8k+8] of each),
        // so the pairing is correct regardless of the slices' runtime alignment.
        let mut dc = dst.chunks_exact_mut(8);
        let mut sc = src.chunks_exact(8);
        for (d, s) in dc.by_ref().zip(sc.by_ref())
        {
            let mut vd = f32x8::from_slice(d);
            vd += f32x8::from_slice(s);
            vd.copy_to_slice(d);
        }
        for (d, s) in dc.into_remainder().iter_mut().zip(sc.remainder())
        {
            *d += s;
        }
    }

    /// Additionne deux slices f64 en place : dst[i] += src[i]
    #[inline]
    pub fn add_f64_inplace(dst: &mut [f64], src: &[f64]) {
        assert_eq!(dst.len(), src.len());

        let mut dc = dst.chunks_exact_mut(4);
        let mut sc = src.chunks_exact(4);
        for (d, s) in dc.by_ref().zip(sc.by_ref())
        {
            let mut vd = f64x4::from_slice(d);
            vd += f64x4::from_slice(s);
            vd.copy_to_slice(d);
        }
        for (d, s) in dc.into_remainder().iter_mut().zip(sc.remainder())
        {
            *d += s;
        }
    }

    // ------------------------------------------------------------------ //
    //  MULTIPLICATION SCALAIRE (scale)                                     //
    // ------------------------------------------------------------------ //

    /// Multiplie chaque élément par un scalaire : v[i] *= alpha
    #[inline]
    pub fn scale_f32(v: &mut [f32], alpha: f32) {
        let splat = f32x8::splat(alpha);
        let (pre, mid, suf) = v.as_simd_mut::<8>();
        for x in pre.iter_mut()
        {
            *x *= alpha;
        }
        for vx in mid.iter_mut()
        {
            *vx *= splat;
        }
        for x in suf.iter_mut()
        {
            *x *= alpha;
        }
    }

    #[inline]
    pub fn scale_f64(v: &mut [f64], alpha: f64) {
        let splat = f64x4::splat(alpha);
        let (pre, mid, suf) = v.as_simd_mut::<4>();
        for x in pre.iter_mut()
        {
            *x *= alpha;
        }
        for vx in mid.iter_mut()
        {
            *vx *= splat;
        }
        for x in suf.iter_mut()
        {
            *x *= alpha;
        }
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
        let mut ac = a.chunks_exact(8);
        let mut bc = b.chunks_exact(8);
        for (ca, cb) in ac.by_ref().zip(bc.by_ref())
        {
            acc += f32x8::from_slice(ca) * f32x8::from_slice(cb);
        }
        let mut scalar_acc = 0.0f32;
        for (x, y) in ac.remainder().iter().zip(bc.remainder())
        {
            scalar_acc += x * y;
        }
        // Réduction horizontale du vecteur SIMD.
        scalar_acc + acc.reduce_sum()
    }

    #[inline]
    pub fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());

        let mut acc = f64x4::splat(0.0);
        let mut ac = a.chunks_exact(4);
        let mut bc = b.chunks_exact(4);
        for (ca, cb) in ac.by_ref().zip(bc.by_ref())
        {
            acc += f64x4::from_slice(ca) * f64x4::from_slice(cb);
        }
        let mut scalar_acc = 0.0f64;
        for (x, y) in ac.remainder().iter().zip(bc.remainder())
        {
            scalar_acc += x * y;
        }
        scalar_acc + acc.reduce_sum()
    }

    // ------------------------------------------------------------------ //
    //  FMA — Fused Multiply-Add : dst[i] = a[i]*b[i] + c[i]               //
    //  Exploite les instructions VFMADD231PS/FMLA d'AVX2/NEON              //
    // ------------------------------------------------------------------ //

    #[inline]
    pub fn fma_f32(dst: &mut [f32], a: &[f32], b: &[f32], c: &[f32]) {
        assert!(dst.len() == a.len() && a.len() == b.len() && b.len() == c.len());

        let mut dc = dst.chunks_exact_mut(8);
        let mut ac = a.chunks_exact(8);
        let mut bc = b.chunks_exact(8);
        let mut cc = c.chunks_exact(8);
        for (((d, ca), cb), ci) in dc
            .by_ref()
            .zip(ac.by_ref())
            .zip(bc.by_ref())
            .zip(cc.by_ref())
        {
            // mul_add est mappé sur VFMADD/FMLA quand disponible.
            let v = f32x8::from_slice(ca).mul_add(f32x8::from_slice(cb), f32x8::from_slice(ci));
            v.copy_to_slice(d);
        }
        let (dr, ar, br, cr) = (
            dc.into_remainder(),
            ac.remainder(),
            bc.remainder(),
            cc.remainder(),
        );
        for (i, d) in dr.iter_mut().enumerate()
        {
            *d = ar[i] * br[i] + cr[i];
        }
    }

    // ------------------------------------------------------------------ //
    //  NORMALISATION L2                                                     //
    // ------------------------------------------------------------------ //

    /// Normalise un vecteur en norme L2 in-place.
    pub fn normalize_f32(v: &mut [f32]) {
        let norm = dot_f32(v, v).sqrt();
        if norm > f32::EPSILON
        {
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
        for x in pre.iter_mut()
        {
            *x = x.max(0.0);
        }
        for vx in mid.iter_mut()
        {
            *vx = vx.simd_max(zero);
        }
        for x in suf.iter_mut()
        {
            *x = x.max(0.0);
        }
    }
}

// ------------------------------------------------------------------ //
//  Fallback scalar — compilé quand portable-simd n'est pas activé    //
// ------------------------------------------------------------------ //
#[cfg(not(feature = "portable-simd"))]
pub mod simd_ops {
    #[inline]
    pub fn add_f32_inplace(dst: &mut [f32], src: &[f32]) {
        for (d, s) in dst.iter_mut().zip(src.iter())
        {
            *d += s;
        }
    }
    #[inline]
    pub fn add_f64_inplace(dst: &mut [f64], src: &[f64]) {
        for (d, s) in dst.iter_mut().zip(src.iter())
        {
            *d += s;
        }
    }
    #[inline]
    pub fn scale_f32(v: &mut [f32], alpha: f32) {
        for x in v.iter_mut()
        {
            *x *= alpha;
        }
    }
    #[inline]
    pub fn scale_f64(v: &mut [f64], alpha: f64) {
        for x in v.iter_mut()
        {
            *x *= alpha;
        }
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
        for i in 0..dst.len()
        {
            dst[i] = a[i] * b[i] + c[i];
        }
    }
    #[inline]
    pub fn normalize_f32(v: &mut [f32]) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON
        {
            for x in v.iter_mut()
            {
                *x /= norm;
            }
        }
    }
    #[inline]
    pub fn relu_f32(v: &mut [f32]) {
        for x in v.iter_mut()
        {
            *x = x.max(0.0);
        }
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
        let src = vec![0.5f32; 9];
        add_f32_inplace(&mut dst, &src);
        for (i, x) in dst.iter().enumerate()
        {
            assert!((x - (i as f32 + 1.5)).abs() < 1e-6, "add_f32 failed at {i}");
        }
    }

    /// Regression: the multi-slice kernels must be correct **regardless of the
    /// operands' relative alignment**. The previous `as_simd`-per-slice split
    /// paired mismatched SIMD lanes when operands had different alignment,
    /// producing wrong results *nondeterministically* (allocation-dependent).
    /// Slicing from varied offsets forces every relative alignment.
    #[test]
    fn simd_kernels_correct_under_any_alignment() {
        let n = 40usize;
        for d_off in 0..9
        {
            for s_off in 0..9
            {
                let mut dbuf: Vec<f32> = (0..n + d_off).map(|i| (i as f32 * 0.7).sin()).collect();
                let sbuf: Vec<f32> = (0..n + s_off).map(|i| (i as f32 * 0.3).cos()).collect();
                let cbuf: Vec<f32> = (0..n + s_off).map(|i| (i as f32 * 0.11) - 0.5).collect();

                // add_f32_inplace: dst += src.
                let d0: Vec<f32> = dbuf[d_off..d_off + n].to_vec();
                add_f32_inplace(&mut dbuf[d_off..d_off + n], &sbuf[s_off..s_off + n]);
                for k in 0..n
                {
                    let want = d0[k] + sbuf[s_off + k];
                    assert!(
                        (dbuf[d_off + k] - want).abs() < 1e-6,
                        "add d_off={d_off} s_off={s_off} k={k}"
                    );
                }

                // dot_f32: a·b.
                let dot = dot_f32(&sbuf[s_off..s_off + n], &cbuf[s_off..s_off + n]);
                let want_dot: f32 = (0..n).map(|k| sbuf[s_off + k] * cbuf[s_off + k]).sum();
                assert!(
                    (dot - want_dot).abs() < 1e-3,
                    "dot d_off={d_off} s_off={s_off}: {dot} vs {want_dot}"
                );

                // fma_f32: dst = a*b + c (a = the just-updated dbuf slice).
                let mut fbuf = vec![0.0f32; n + d_off];
                fma_f32(
                    &mut fbuf[d_off..d_off + n],
                    &dbuf[d_off..d_off + n],
                    &sbuf[s_off..s_off + n],
                    &cbuf[s_off..s_off + n],
                );
                for k in 0..n
                {
                    let want = dbuf[d_off + k] * sbuf[s_off + k] + cbuf[s_off + k];
                    assert!(
                        (fbuf[d_off + k] - want).abs() < 1e-5,
                        "fma d_off={d_off} s_off={s_off} k={k}"
                    );
                }
            }
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
