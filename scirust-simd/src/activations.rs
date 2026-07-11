//! # Activations transcendantes vectorisées (x86_64 AVX-512)
//!
//! Approximations SIMD des activations non linéaires usuelles des réseaux de
//! neurones — `sigmoid`, `tanh`, `SiLU`/swish, `GELU` — bâties sur une
//! **exponentielle vectorisée** `exp_ps` (réduction d'argument `x = k·ln2 + r`
//! puis polynôme minimax sur `r`, reconstruction `2^k` par `_mm512_scalef_ps`).
//!
//! Chaque entrée publique fait de la détection *runtime* et retombe sur la
//! `libm` scalaire, donc le résultat est correct sur tout CPU. Les helpers
//! `__m512` (`pub(crate)`) sont réutilisés par l'épilogue fusionné du GEMM
//! (couche dense + activation, voir [`crate::gemm`]).
//!
//! Précision : erreur relative ≲ 1e-6 sur `exp` dans la plage utile, ce qui
//! suffit très largement pour l'inférence/entraînement `f32` (dont le bruit de
//! quantification dépasse cette borne).

#![allow(clippy::missing_safety_doc)]

// Constantes partagées. `LOG2_E == 1/ln2` vient de la lib standard.
// Décomposition de ln2 en deux parties (Cephes) pour la réduction d'argument :
// `LN2_HI = 355/512` est exact en f32, `LN2_LO` capture le reste.
#[cfg(target_arch = "x86_64")]
const LN2_HI: f32 = 0.693_359_4;
#[cfg(target_arch = "x86_64")]
const LN2_LO: f32 = -2.121_944e-4;
#[cfg(target_arch = "x86_64")]
const EXP_HI: f32 = 88.3762; // bornes anti-overflow de expf
#[cfg(target_arch = "x86_64")]
const EXP_LO: f32 = -88.3762;
const SQRT_2_OVER_PI: f32 = 0.797_884_6; // sqrt(2/pi), pour GELU tanh
const GELU_C: f32 = 0.044_715;

// ===================================================================== //
//  Helpers __m512 (AVX-512) — réutilisés par le GEMM fusionné            //
// ===================================================================== //

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn exp_ps(x: core::arch::x86_64::__m512) -> core::arch::x86_64::__m512 {
    use core::arch::x86_64::*;
    // Clamp pour éviter overflow/NaN.
    let x = _mm512_min_ps(_mm512_set1_ps(EXP_HI), x);
    let x = _mm512_max_ps(_mm512_set1_ps(EXP_LO), x);

    // k = round(x / ln2) ; r = x - k·ln2 (deux parties).
    let kf =
        _mm512_roundscale_ps::<0x00>(_mm512_mul_ps(x, _mm512_set1_ps(core::f32::consts::LOG2_E)));
    let r = _mm512_fnmadd_ps(kf, _mm512_set1_ps(LN2_HI), x);
    let r = _mm512_fnmadd_ps(kf, _mm512_set1_ps(LN2_LO), r);

    // exp(r) ≈ Σ r^n/n! (Horner, degré 6).
    let mut p = _mm512_set1_ps(1.0 / 720.0);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0 / 120.0));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0 / 24.0));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0 / 6.0));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.5));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0));

    // 2^k · exp(r).
    _mm512_scalef_ps(p, kf)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn sigmoid_ps(x: core::arch::x86_64::__m512) -> core::arch::x86_64::__m512 {
    use core::arch::x86_64::*;
    // 1 / (1 + exp(-x)).
    let e = exp_ps(_mm512_sub_ps(_mm512_setzero_ps(), x));
    _mm512_div_ps(_mm512_set1_ps(1.0), _mm512_add_ps(_mm512_set1_ps(1.0), e))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn tanh_ps(x: core::arch::x86_64::__m512) -> core::arch::x86_64::__m512 {
    use core::arch::x86_64::*;
    // tanh(x) = 2·sigmoid(2x) − 1.
    let s = sigmoid_ps(_mm512_mul_ps(_mm512_set1_ps(2.0), x));
    _mm512_fmsub_ps(_mm512_set1_ps(2.0), s, _mm512_set1_ps(1.0))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn silu_ps(x: core::arch::x86_64::__m512) -> core::arch::x86_64::__m512 {
    use core::arch::x86_64::*;
    // x · sigmoid(x).
    _mm512_mul_ps(x, sigmoid_ps(x))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn gelu_ps(x: core::arch::x86_64::__m512) -> core::arch::x86_64::__m512 {
    use core::arch::x86_64::*;
    // Approximation tanh : 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))).
    let x2 = _mm512_mul_ps(x, x);
    let x3 = _mm512_mul_ps(x2, x);
    let inner = _mm512_fmadd_ps(_mm512_set1_ps(GELU_C), x3, x);
    let t = tanh_ps(_mm512_mul_ps(_mm512_set1_ps(SQRT_2_OVER_PI), inner));
    let half_x = _mm512_mul_ps(_mm512_set1_ps(0.5), x);
    _mm512_mul_ps(half_x, _mm512_add_ps(_mm512_set1_ps(1.0), t))
}

// ===================================================================== //
//  Références scalaires (repli + oracle de test)                         //
// ===================================================================== //

/// Sigmoïde scalaire.
#[inline]
pub fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// SiLU / swish scalaire : `x · sigmoid(x)`.
#[inline]
pub fn silu_scalar(x: f32) -> f32 {
    x * sigmoid_scalar(x)
}

/// GELU scalaire (approximation tanh, comme PyTorch `approximate="tanh"`).
#[inline]
pub fn gelu_scalar(x: f32) -> f32 {
    let inner = SQRT_2_OVER_PI * (x + GELU_C * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

// ===================================================================== //
//  API publiques sur tranches (in-place)                                 //
// ===================================================================== //

macro_rules! define_slice_activation {
    ($name:ident, $ps:ident, $scalar:expr, $doc:literal) => {
        #[doc = $doc]
        pub fn $name(data: &mut [f32]) {
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avx512f")
                {
                    // SAFETY: gated by the runtime detection just above.
                    unsafe { apply_avx512(data, $ps) };
                    return;
                }
            }
            for x in data.iter_mut()
            {
                *x = $scalar(*x);
            }
        }
    };
}

define_slice_activation!(
    sigmoid_inplace,
    sigmoid_ps,
    sigmoid_scalar,
    "Sigmoïde en place, vectorisée AVX-512 avec repli scalaire."
);
define_slice_activation!(
    silu_inplace,
    silu_ps,
    silu_scalar,
    "SiLU/swish en place, vectorisée AVX-512 avec repli scalaire."
);
define_slice_activation!(
    gelu_inplace,
    gelu_ps,
    gelu_scalar,
    "GELU (approx. tanh) en place, vectorisée AVX-512 avec repli scalaire."
);
define_slice_activation!(
    tanh_inplace,
    tanh_ps,
    |x: f32| x.tanh(),
    "tanh en place, vectorisée AVX-512 avec repli scalaire."
);

/// `exp` en place (AVX-512 avec repli scalaire).
pub fn exp_inplace(data: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f")
        {
            // SAFETY: gated by the runtime detection just above.
            unsafe { apply_avx512(data, exp_ps) };
            return;
        }
    }
    for x in data.iter_mut()
    {
        *x = x.exp();
    }
}

/// Applique un noyau `__m512 -> __m512` sur toute la tranche (épilogue masqué).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_avx512(
    data: &mut [f32],
    f: unsafe fn(core::arch::x86_64::__m512) -> core::arch::x86_64::__m512,
) {
    use core::arch::x86_64::*;
    let n = data.len();
    let mut i = 0;
    while i + 16 <= n
    {
        let v = _mm512_loadu_ps(data.as_ptr().add(i));
        _mm512_storeu_ps(data.as_mut_ptr().add(i), f(v));
        i += 16;
    }
    let r = n - i;
    if r > 0
    {
        let mask = (1u16 << r) - 1;
        let v = _mm512_maskz_loadu_ps(mask, data.as_ptr().add(i));
        _mm512_mask_storeu_ps(data.as_mut_ptr().add(i), mask, f(v));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(got: f32, want: f32, rel: f32, ctx: &str) {
        let tol = rel * (1.0 + want.abs());
        assert!(
            (got - want).abs() <= tol,
            "{ctx}: got {got}, want {want} (tol {tol})"
        );
    }

    #[test]
    fn exp_matches_libm() {
        let xs: Vec<f32> = (0..200).map(|i| (i as f32) * 0.4 - 40.0).collect();
        let mut got = xs.clone();
        exp_inplace(&mut got);
        for (i, &x) in xs.iter().enumerate()
        {
            assert_close(got[i], x.exp(), 1e-5, &format!("exp({x})"));
        }
    }

    #[test]
    fn sigmoid_tanh_match_libm() {
        let xs: Vec<f32> = (0..300).map(|i| (i as f32) * 0.1 - 15.0).collect();
        let mut sg = xs.clone();
        sigmoid_inplace(&mut sg);
        let mut th = xs.clone();
        tanh_inplace(&mut th);
        for (i, &x) in xs.iter().enumerate()
        {
            assert_close(sg[i], sigmoid_scalar(x), 1e-5, &format!("sigmoid({x})"));
            assert_close(th[i], x.tanh(), 1e-5, &format!("tanh({x})"));
        }
    }

    #[test]
    fn silu_gelu_match_scalar() {
        let xs: Vec<f32> = (0..300).map(|i| (i as f32) * 0.1 - 15.0).collect();
        let mut si = xs.clone();
        silu_inplace(&mut si);
        let mut ge = xs.clone();
        gelu_inplace(&mut ge);
        for (i, &x) in xs.iter().enumerate()
        {
            assert_close(si[i], silu_scalar(x), 1e-5, &format!("silu({x})"));
            assert_close(ge[i], gelu_scalar(x), 1e-5, &format!("gelu({x})"));
        }
    }

    #[test]
    fn handles_all_lengths_remainder() {
        // Couvre chaque taille d'épilogue masqué 0..=33.
        for len in 0..=33usize
        {
            let xs: Vec<f32> = (0..len).map(|i| (i as f32) * 0.3 - 3.0).collect();
            let mut got = xs.clone();
            sigmoid_inplace(&mut got);
            for (i, &x) in xs.iter().enumerate()
            {
                assert_close(got[i], sigmoid_scalar(x), 1e-5, &format!("len {len} i {i}"));
            }
        }
    }
}
