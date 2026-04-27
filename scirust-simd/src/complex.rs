// scirust-simd/src/complex.rs
//
// Kernels SIMD pour Complex<f32> et Complex<f64>.
// Critique pour DSP, FFT, MIMO, propagation EM, optique...
//
// Représentation mémoire : interleaved [re0, im0, re1, im1, ...]
// avec #[repr(C)] pour garantir le layout. Cela permet de transmuter
// un &[Complex<f32>] en &[f32] de longueur double.

#![cfg_attr(feature = "portable-simd", feature(portable_simd))]

// ------------------------------------------------------------------ //
//  Type Complex                                                       //
// ------------------------------------------------------------------ //

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complex<T> {
    pub re: T,
    pub im: T,
}

impl<T: Copy> Complex<T> {
    #[inline]
    pub const fn new(re: T, im: T) -> Self { Self { re, im } }
}

impl Complex<f32> {
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    pub const ONE:  Self = Self { re: 1.0, im: 0.0 };
    pub const I:    Self = Self { re: 0.0, im: 1.0 };

    #[inline]
    pub fn norm_sqr(self) -> f32 { self.re * self.re + self.im * self.im }

    #[inline]
    pub fn conj(self) -> Self { Self { re: self.re, im: -self.im } }

    #[inline]
    pub fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

// ------------------------------------------------------------------ //
//  Helpers — transmuter slice de Complex en slice de scalaires        //
// ------------------------------------------------------------------ //

#[inline]
fn as_f32(s: &[Complex<f32>]) -> &[f32] {
    // SAFETY : Complex<f32> est #[repr(C)] avec 2 f32 contigus
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const f32, s.len() * 2) }
}

#[inline]
fn as_f32_mut(s: &mut [Complex<f32>]) -> &mut [f32] {
    unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut f32, s.len() * 2) }
}

// ------------------------------------------------------------------ //
//  ADDITION COMPLEXE — trivial (additif lane-par-lane sur f32)        //
// ------------------------------------------------------------------ //

#[cfg(feature = "portable-simd")]
pub fn complex_add_f32(dst: &mut [Complex<f32>], src: &[Complex<f32>]) {
    use std::simd::f32x8;
    assert_eq!(dst.len(), src.len());

    let dst_f = as_f32_mut(dst);
    let src_f = as_f32(src);

    let (pre, mid_dst, suf_dst) = dst_f.as_simd_mut::<8>();
    let (_, mid_src, _)         = src_f.as_simd::<8>();

    for (d, s) in pre.iter_mut().zip(src_f.iter()) { *d += s; }
    for (vd, vs) in mid_dst.iter_mut().zip(mid_src.iter()) { *vd += vs; }
    let offset = pre.len() + mid_dst.len() * 8;
    for (d, s) in suf_dst.iter_mut().zip(src_f[offset..].iter()) { *d += s; }
}

#[cfg(not(feature = "portable-simd"))]
pub fn complex_add_f32(dst: &mut [Complex<f32>], src: &[Complex<f32>]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        d.re += s.re;
        d.im += s.im;
    }
}

// ------------------------------------------------------------------ //
//  MULTIPLICATION COMPLEXE — non-trivial, utilise des shuffles        //
// ------------------------------------------------------------------ //
//
//  (a + bi)(c + di) = (ac - bd) + (ad + bc)i
//
//  Layout pour 4 complexes (8 lanes f32) :
//    a_buf = [a0_re, a0_im, a1_re, a1_im, a2_re, a2_im, a3_re, a3_im]
//    b_buf = [b0_re, b0_im, b1_re, b1_im, b2_re, b2_im, b3_re, b3_im]
//
//  Étapes :
//    1) a_re_dup = [a0_re, a0_re, a1_re, a1_re, ...]   broadcast pairs even
//    2) a_im_dup = [a0_im, a0_im, a1_im, a1_im, ...]   broadcast pairs odd
//    3) b_swap   = [b0_im, b0_re, b1_im, b1_re, ...]   swap dans chaque pair
//    4) part1    = a_re_dup * b
//    5) part2    = a_im_dup * b_swap
//    6) Combine avec sign_mask = [-1, +1, -1, +1, ...] sur part2 :
//       result = part1 + part2 * sign_mask
//                = [a0_re*b0_re - a0_im*b0_im, a0_re*b0_im + a0_im*b0_re, ...]

#[cfg(feature = "portable-simd")]
pub fn complex_mul_f32(
    dst: &mut [Complex<f32>],
    a: &[Complex<f32>],
    b: &[Complex<f32>],
) {
    use std::simd::{f32x8, simd_swizzle};
    assert!(dst.len() == a.len() && a.len() == b.len());

    let a_f = as_f32(a);
    let b_f = as_f32(b);
    let dst_f = as_f32_mut(dst);

    // Sign mask pour la combinaison finale
    let sign_mask = f32x8::from_array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]);

    // 4 complexes traités par cycle SIMD (= 8 lanes f32)
    let chunks = a.len() / 4;
    for c in 0..chunks {
        let av = f32x8::from_slice(&a_f[c * 8..]);
        let bv = f32x8::from_slice(&b_f[c * 8..]);

        // Duplique les parties réelles : [re, re, re, re, re, re, re, re]
        // depuis [re, im, re, im, re, im, re, im]
        let a_re_dup: f32x8 = simd_swizzle!(av, [0, 0, 2, 2, 4, 4, 6, 6]);
        let a_im_dup: f32x8 = simd_swizzle!(av, [1, 1, 3, 3, 5, 5, 7, 7]);

        // Swap pairs de b : [im, re, im, re, ...]
        let b_swap: f32x8 = simd_swizzle!(bv, [1, 0, 3, 2, 5, 4, 7, 6]);

        // (a_re * b)  +  (a_im * b_swap) * sign_mask
        let part1 = a_re_dup * bv;
        let part2 = a_im_dup * b_swap;
        let result = part1 + part2 * sign_mask;

        result.copy_to_slice(&mut dst_f[c * 8..]);
    }

    // Reste scalaire
    for i in (chunks * 4)..a.len() {
        dst[i] = a[i].mul(b[i]);
    }
}

#[cfg(not(feature = "portable-simd"))]
pub fn complex_mul_f32(
    dst: &mut [Complex<f32>],
    a: &[Complex<f32>],
    b: &[Complex<f32>],
) {
    for i in 0..a.len() { dst[i] = a[i].mul(b[i]); }
}

// ------------------------------------------------------------------ //
//  PRODUIT SCALAIRE HERMITIEN : sum(a_i * conj(b_i))                  //
//  Forme la norme L2 de signaux complexes, projection orthogonale...  //
// ------------------------------------------------------------------ //
//
//  a_i * conj(b_i) = (a_re + i a_im)(b_re - i b_im)
//                  = (a_re*b_re + a_im*b_im) + i(a_im*b_re - a_re*b_im)
//
//  Note : pour la norme |a|² = sum(a_i * conj(a_i)) on a juste le réel.

pub fn complex_dot_hermitian_f32(a: &[Complex<f32>], b: &[Complex<f32>]) -> Complex<f32> {
    assert_eq!(a.len(), b.len());

    #[cfg(feature = "portable-simd")]
    {
        use std::simd::{f32x8, simd_swizzle, num::SimdFloat};

        let a_f = as_f32(a);
        let b_f = as_f32(b);

        // Accumulateurs séparés re et im pour éviter la contention horizontale
        let mut acc_re = f32x8::splat(0.0);
        let mut acc_im = f32x8::splat(0.0);
        let sign = f32x8::from_array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);

        let chunks = a.len() / 4;
        for c in 0..chunks {
            let av = f32x8::from_slice(&a_f[c * 8..]);
            let bv = f32x8::from_slice(&b_f[c * 8..]);

            // Re part : a_re*b_re + a_im*b_im
            // = somme alternée de av * bv (lanes 0,2,4,6 = re*re, lanes 1,3,5,7 = im*im)
            // → on accumule juste av * bv ; à la fin on additionne lanes même + lanes impair
            let prod_re_part = av * bv;

            // Im part : a_im*b_re - a_re*b_im
            // = av_swap * bv * sign_mask
            let av_swap: f32x8 = simd_swizzle!(av, [1, 0, 3, 2, 5, 4, 7, 6]);
            let prod_im_part = av_swap * bv * sign;

            acc_re += prod_re_part;
            acc_im += prod_im_part;
        }

        // Réduction : pour acc_re, somme tous les lanes (re et im mélangés mais c'est OK)
        // En fait : prod_re_part = [a_re*b_re, a_im*b_im, a_re*b_re, ...] sur les lanes
        // pairs et impairs. Leur somme totale = sum(a_re*b_re + a_im*b_im) donc OK.
        let mut re_total = acc_re.reduce_sum();
        let mut im_total = acc_im.reduce_sum();

        // Reste scalaire
        for i in (chunks * 4)..a.len() {
            re_total += a[i].re * b[i].re + a[i].im * b[i].im;
            im_total += a[i].im * b[i].re - a[i].re * b[i].im;
        }

        return Complex::new(re_total, im_total);
    }

    #[cfg(not(feature = "portable-simd"))]
    {
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for i in 0..a.len() {
            re += a[i].re * b[i].re + a[i].im * b[i].im;
            im += a[i].im * b[i].re - a[i].re * b[i].im;
        }
        Complex::new(re, im)
    }
}

// ------------------------------------------------------------------ //
//  Norme L2 d'un vecteur complexe                                     //
// ------------------------------------------------------------------ //

pub fn complex_norm_l2_f32(v: &[Complex<f32>]) -> f32 {
    // |v|² = sum |v_i|² = sum (re² + im²) = sum(v · v) (interleaved)
    #[cfg(feature = "portable-simd")]
    {
        use std::simd::{f32x8, num::SimdFloat};
        let v_f = as_f32(v);
        let mut acc = f32x8::splat(0.0);
        let (pre, mid, suf) = v_f.as_simd::<8>();
        let mut scalar_acc = 0.0f32;
        for x in pre { scalar_acc += x * x; }
        for vx in mid { acc += vx * vx; }
        for x in suf { scalar_acc += x * x; }
        (scalar_acc + acc.reduce_sum()).sqrt()
    }
    #[cfg(not(feature = "portable-simd"))]
    {
        v.iter().map(|c| c.re * c.re + c.im * c.im).sum::<f32>().sqrt()
    }
}

// ------------------------------------------------------------------ //
//  Tests                                                              //
// ------------------------------------------------------------------ //
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_add() {
        let mut dst = vec![Complex::new(1.0f32, 2.0); 5];
        let src     = vec![Complex::new(3.0f32, -1.0); 5];
        complex_add_f32(&mut dst, &src);
        for c in &dst {
            assert!((c.re - 4.0).abs() < 1e-6);
            assert!((c.im - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_complex_mul_basic() {
        // (1+2i)(3+4i) = 3 + 4i + 6i - 8 = -5 + 10i
        let a = vec![Complex::new(1.0f32, 2.0); 1];
        let b = vec![Complex::new(3.0f32, 4.0); 1];
        let mut dst = vec![Complex::ZERO; 1];
        complex_mul_f32(&mut dst, &a, &b);
        assert!((dst[0].re - (-5.0)).abs() < 1e-5);
        assert!((dst[0].im - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_complex_mul_simd_chunk() {
        // Test avec 4 complexes (= 1 chunk SIMD)
        let a: Vec<_> = (0..4).map(|i| Complex::new(i as f32, (i + 1) as f32)).collect();
        let b: Vec<_> = (0..4).map(|i| Complex::new((i * 2) as f32, (i + 3) as f32)).collect();
        let mut dst = vec![Complex::ZERO; 4];
        complex_mul_f32(&mut dst, &a, &b);

        for i in 0..4 {
            let expected = a[i].mul(b[i]);
            assert!((dst[i].re - expected.re).abs() < 1e-4,
                    "re mismatch at {i}: {} vs {}", dst[i].re, expected.re);
            assert!((dst[i].im - expected.im).abs() < 1e-4,
                    "im mismatch at {i}: {} vs {}", dst[i].im, expected.im);
        }
    }

    #[test]
    fn test_complex_dot_hermitian_self() {
        // <a, a> = sum |a_i|² (réel pur)
        let a = vec![
            Complex::new(3.0f32, 4.0),    // |.|² = 25
            Complex::new(1.0f32, 0.0),    // |.|² = 1
            Complex::new(0.0f32, 2.0),    // |.|² = 4
        ];
        let result = complex_dot_hermitian_f32(&a, &a);
        assert!((result.re - 30.0).abs() < 1e-5);
        assert!(result.im.abs() < 1e-5);
    }

    #[test]
    fn test_complex_norm() {
        let v = vec![Complex::new(3.0f32, 4.0)]; // |.| = 5
        let n = complex_norm_l2_f32(&v);
        assert!((n - 5.0).abs() < 1e-6);
    }
}
