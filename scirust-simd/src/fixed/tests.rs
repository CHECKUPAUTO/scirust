// scirust-simd/src/fixed/tests.rs
//
// Batterie de validation du sous-système virgule fixe. Tous les tests sont
// **indépendants de l'architecture** (aucune dépendance au matériel SIMD :
// std::simd produit les mêmes bits partout). On combine :
//  * assertions **exactes** sur des cas construits (arrondi, overflow, bits) ;
//  * comparaison à une référence `f64` à quelques ULP pour mul/div/math ;
//  * égalité stricte **SIMD == scalaire**.

use super::activation as act;
use super::attention::{attention, causal_attention, multi_head_attention};
use super::conv::{Conv1dShape, conv1d, conv1d_batch};
use super::conv2d::{
    Conv2dShape, Conv2dTransposeShape, conv2d, conv2d_batch, conv2d_transpose, depthwise_conv2d,
    separable_conv2d,
};
use super::kv_cache::KvCache;
use super::layer::Linear;
use super::linalg;
use super::math::{reciprocal, rsqrt, sqrt};
use super::model::TransformerModel;
use super::norm::{layer_norm, rmsnorm, rope_apply};
use super::pool::{Pool1dShape, avg_pool1d, max_pool1d};
use super::pool2d::{Pool2dShape, avg_pool2d, max_pool2d};
use super::reductions as red;
use super::rescale::{rescale, rescale_saturating, rescale_wrapping};
use super::simd::{FixedI16x8, FixedI32x8, FixedI64x4};
use super::transcendental as tr;
use super::transformer::{TransformerBlock, rope_apply_heads};
use super::{
    FixedI16, FixedI32, FixedI64, NumericScalar, OverflowMode, Q1_15, Q8_8, Q8_24, Q16_16, Q24_8,
    Q32_32, RealScalar, RoundingMode,
};
use crate::geometry::Quaternion;

// LCG déterministe.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    /// Brut i32 dans une plage modérée (évite les débordements de test).
    fn raw_i32(&mut self) -> i32 {
        (self.next() >> 40) as i32 - (1 << 23)
    }
}

/// Q16.16 depuis un flottant (arrondi au pair), pour les cas connus.
fn q16(v: f64) -> Q16_16 {
    Q16_16::try_from(v).unwrap()
}

// ------------------------------------------------------------------ //
//  Représentation & constantes                                        //
// ------------------------------------------------------------------ //

#[test]
fn layout_is_transparent() {
    use core::mem::{align_of, size_of};
    assert_eq!(size_of::<Q16_16>(), size_of::<i32>());
    assert_eq!(align_of::<Q16_16>(), align_of::<i32>());
    assert_eq!(size_of::<Q32_32>(), size_of::<i64>());
    assert_eq!(size_of::<[Q16_16; 8]>(), size_of::<[i32; 8]>());
}

#[test]
fn constants_and_raw() {
    assert_eq!(Q16_16::zero().to_raw(), 0);
    assert_eq!(Q16_16::one().to_raw(), 1 << 16);
    assert_eq!(Q16_16::resolution().to_raw(), 1);
    assert_eq!(Q16_16::one().to_f64(), 1.0);
    assert_eq!(Q16_16::resolution().to_f64(), 1.0 / 65536.0);
    assert_eq!(Q8_24::one().to_raw(), 1 << 24);
    assert_eq!(Q32_32::one().to_raw(), 1i64 << 32);
    assert_eq!(Q24_8::one().to_raw(), 1 << 8);
}

// ------------------------------------------------------------------ //
//  Conversions                                                        //
// ------------------------------------------------------------------ //

#[test]
fn convert_int_saturating() {
    assert_eq!(Q16_16::from(3).to_f64(), 3.0);
    assert_eq!(Q16_16::from(-5).to_f64(), -5.0);
    // Q16.16 sature à ±32768 : 100000 → MAX.
    assert_eq!(Q16_16::from(100_000), Q16_16::max_value());
    assert_eq!(Q16_16::from(-100_000), Q16_16::min_value());
    // Variante checked.
    assert!(Q16_16::from_int_checked(100_000).is_none());
    assert!(Q16_16::from_int_checked(30_000).is_some());
}

#[test]
fn convert_float_roundtrip_and_errors() {
    for &v in &[0.0, 1.0, -1.0, 0.5, -0.25, 123.4567, -9999.99]
    {
        let f = q16(v);
        // Aller-retour à moins d'une résolution.
        assert!((f.to_f64() - v).abs() <= 1.0 / 65536.0, "v={v}");
    }
    // NaN / infini / hors plage → Err.
    assert!(Q16_16::try_from(f64::NAN).is_err());
    assert!(Q16_16::try_from(f64::INFINITY).is_err());
    assert!(Q16_16::try_from(1e9).is_err());
    // f32 lossless vers f64 puis fixe.
    assert_eq!(Q16_16::try_from(0.5f32).unwrap(), q16(0.5));
    // Into<f32>/<f64>.
    let x = q16(3.25);
    let as_f32: f32 = x.into();
    let as_f64: f64 = x.into();
    assert_eq!(as_f32, 3.25);
    assert_eq!(as_f64, 3.25);
}

#[test]
fn convert_float_rounding_modes() {
    // 0.5 en résolution Q0 : construit une valeur à mi-chemin exact.
    // raw cible = round(2.5) sous différents modes (échelle 1 pour lisibilité).
    let v = 2.5f64 / 65536.0; // vaut exactement 2.5 ULP
    assert_eq!(
        Q16_16::from_f64(v, RoundingMode::TowardZero)
            .unwrap()
            .to_raw(),
        2
    );
    assert_eq!(Q16_16::from_f64(v, RoundingMode::Ceil).unwrap().to_raw(), 3);
    assert_eq!(
        Q16_16::from_f64(v, RoundingMode::Floor).unwrap().to_raw(),
        2
    );
    // 2.5 → pair le plus proche = 2.
    assert_eq!(
        Q16_16::from_f64(v, RoundingMode::NearestEven)
            .unwrap()
            .to_raw(),
        2
    );
    // 3.5 → pair le plus proche = 4.
    let v2 = 3.5f64 / 65536.0;
    assert_eq!(
        Q16_16::from_f64(v2, RoundingMode::NearestEven)
            .unwrap()
            .to_raw(),
        4
    );
}

// ------------------------------------------------------------------ //
//  Addition / soustraction / négation                                 //
// ------------------------------------------------------------------ //

#[test]
fn add_sub_neg_exact() {
    let a = q16(1.5);
    let b = q16(2.25);
    assert_eq!((a + b).to_f64(), 3.75);
    assert_eq!((a - b).to_f64(), -0.75);
    assert_eq!((-a).to_f64(), -1.5);
    // Commutativité & associativité exactes.
    let c = q16(-0.125);
    assert_eq!(a + b, b + a);
    assert_eq!((a + b) + c, a + (b + c));
    // AddAssign / SubAssign.
    let mut m = a;
    m += b;
    m -= c;
    assert_eq!(m, a + b - c);
}

#[test]
fn overflow_policies() {
    let max = Q16_16::max_value();
    let one = Q16_16::one();
    // Wrapping (opérateur) enveloppe : MAX.raw + (1<<16) modulo 2^32.
    assert_eq!(max + one, Q16_16::from_raw(i32::MAX.wrapping_add(1 << 16)));
    // Checked détecte.
    assert!(max.checked_add(one).is_none());
    assert_eq!(q16(1.0).checked_add(one).unwrap(), q16(2.0));
    // Saturating sature.
    assert_eq!(max.saturating_add(one), max);
    assert_eq!(Q16_16::min_value().saturating_sub(one), Q16_16::min_value());
    // Négation de MIN.
    assert_eq!(Q16_16::min_value().wrapping_neg(), Q16_16::min_value());
    assert!(Q16_16::min_value().checked_neg().is_none());
    assert_eq!(Q16_16::min_value().saturating_neg(), Q16_16::max_value());
}

// ------------------------------------------------------------------ //
//  Multiplication                                                     //
// ------------------------------------------------------------------ //

#[test]
fn mul_exact_and_reference() {
    // Cas exacts.
    assert_eq!((q16(0.5) * q16(0.5)).to_f64(), 0.25);
    assert_eq!((q16(2.0) * q16(3.0)).to_f64(), 6.0);
    assert_eq!((q16(-1.5) * q16(4.0)).to_f64(), -6.0);
    assert_eq!(q16(7.0) * Q16_16::one(), q16(7.0)); // x·1 = x
    assert_eq!(q16(7.0) * Q16_16::zero(), Q16_16::zero()); // x·0 = 0
    // Référence f64 à 1 ULP (troncature vers zéro).
    let mut rng = Lcg(0x1234);
    for _ in 0..2000
    {
        let a = Q16_16::from_raw(rng.raw_i32());
        let b = Q16_16::from_raw(rng.raw_i32());
        let got = (a * b).to_raw() as i64;
        let expected = ((a.to_raw() as i64) * (b.to_raw() as i64)) >> 16; // troncature ≥ 0
        let expected_tz = if (a.to_raw() as i64) * (b.to_raw() as i64) < 0
        {
            // vers zéro pour les négatifs
            let p = (a.to_raw() as i64) * (b.to_raw() as i64);
            -((-p) >> 16)
        }
        else
        {
            expected
        };
        assert_eq!(got, expected_tz, "a={a}, b={b}");
    }
}

#[test]
fn mul_rounding_modes() {
    // a = 2^-16 (raw 1), b tel que le produit ait un reste = 1/2 exactement.
    let a = Q16_16::from_raw(1);
    let half_tie = Q16_16::from_raw(1 << 15); // reste = 0x8000 = moitié, quotient 0
    assert_eq!(
        a.mul_rounded(half_tie, RoundingMode::TowardZero, OverflowMode::Wrap)
            .unwrap()
            .to_raw(),
        0
    );
    assert_eq!(
        a.mul_rounded(half_tie, RoundingMode::Ceil, OverflowMode::Wrap)
            .unwrap()
            .to_raw(),
        1
    );
    // quotient impair (1) + moitié → pair le plus proche = 2.
    let odd_tie = Q16_16::from_raw((1 << 16) + (1 << 15)); // quotient 1, reste 1/2
    assert_eq!(
        a.mul_rounded(odd_tie, RoundingMode::NearestEven, OverflowMode::Wrap)
            .unwrap()
            .to_raw(),
        2
    );
    assert_eq!(
        a.mul_rounded(odd_tie, RoundingMode::TowardZero, OverflowMode::Wrap)
            .unwrap()
            .to_raw(),
        1
    );
}

#[test]
fn mul_overflow_policies() {
    let big = q16(200.0); // 200·200 = 40000 > 32768 ⇒ déborde Q16.16
    assert!(big.checked_mul(big).is_none());
    assert_eq!(big.saturating_mul(big), Q16_16::max_value());
    let negbig = q16(-200.0);
    assert_eq!(big.saturating_mul(negbig), Q16_16::min_value());
}

// ------------------------------------------------------------------ //
//  Division                                                           //
// ------------------------------------------------------------------ //

#[test]
fn div_exact_and_reference() {
    assert_eq!((q16(6.0) / q16(2.0)).to_f64(), 3.0);
    assert_eq!((q16(1.0) / q16(4.0)).to_f64(), 0.25);
    assert_eq!((q16(-9.0) / q16(3.0)).to_f64(), -3.0);
    // 1/3 : proche de 0.3333 à 1 ULP.
    let third = q16(1.0) / q16(3.0);
    assert!((third.to_f64() - 1.0 / 3.0).abs() <= 1.0 / 65536.0);
    // Référence f64.
    let mut rng = Lcg(0x77);
    for _ in 0..1000
    {
        let a = Q16_16::from_raw(rng.raw_i32());
        let mut d = rng.raw_i32();
        if d == 0
        {
            d = 1;
        }
        let b = Q16_16::from_raw(d);
        let got = a / b;
        let expected = a.to_f64() / b.to_f64();
        assert!(
            (got.to_f64() - expected).abs() <= 2.0 / 65536.0,
            "a={a} b={b} got={got} exp={expected}"
        );
    }
}

#[test]
fn div_by_zero() {
    assert!(q16(1.0).checked_div(Q16_16::zero()).is_none());
}

#[test]
#[should_panic(expected = "zéro")]
fn div_by_zero_operator_panics() {
    let _ = q16(1.0) / Q16_16::zero();
}

// ------------------------------------------------------------------ //
//  Comparaison / min / max / clamp / abs                              //
// ------------------------------------------------------------------ //

#[test]
fn ordering_and_minmax() {
    assert!(q16(-1.0) < q16(0.0));
    assert!(q16(2.5) > q16(2.499));
    assert_eq!(q16(2.0).min(q16(3.0)), q16(2.0));
    assert_eq!(q16(2.0).max(q16(3.0)), q16(3.0));
    assert_eq!(q16(5.0).clamp(q16(0.0), q16(4.0)), q16(4.0));
    assert_eq!(q16(-5.0).clamp(q16(0.0), q16(4.0)), q16(0.0));
    assert_eq!(q16(-3.0).abs(), q16(3.0));
    assert_eq!(Q16_16::min_value().abs(), Q16_16::max_value()); // saturant
}

// ------------------------------------------------------------------ //
//  SIMD == scalaire                                                   //
// ------------------------------------------------------------------ //

#[test]
fn simd_i16x8_matches_scalar() {
    // Deux formats pour exercer des `FRAC` distincts : Q8.8 (plage modérée) et
    // Q1.15 (audio). L'égalité doit être stricte, y compris sur les cas
    // enveloppants (le produit élargi `i16→i32` est exact, la troncature finale
    // `i32→i16` enveloppe exactement comme le scalaire).
    let mut rng = Lcg(0xC0FFEE);
    for _ in 0..500
    {
        let a88: [Q8_8; 8] = core::array::from_fn(|_| Q8_8::from_raw(rng.next() as i16));
        let b88: [Q8_8; 8] = core::array::from_fn(|_| Q8_8::from_raw(rng.next() as i16));
        let va = FixedI16x8::from_array(a88);
        let vb = FixedI16x8::from_array(b88);

        let add = (va + vb).to_array();
        let sub = (va - vb).to_array();
        let mul = (va * vb).to_array();
        let neg = (-va).to_array();
        let mn = va.min(vb).to_array();
        let mx = va.max(vb).to_array();
        let ab = va.abs().to_array();
        let fma = va.mul_add(vb, va).to_array();
        for i in 0..8
        {
            assert_eq!(add[i], a88[i] + b88[i], "add lane {i}");
            assert_eq!(sub[i], a88[i] - b88[i], "sub lane {i}");
            assert_eq!(mul[i], a88[i] * b88[i], "mul lane {i}");
            assert_eq!(neg[i], -a88[i], "neg lane {i}");
            assert_eq!(mn[i], a88[i].min(b88[i]), "min lane {i}");
            assert_eq!(mx[i], a88[i].max(b88[i]), "max lane {i}");
            assert_eq!(ab[i], a88[i].abs(), "abs lane {i}");
            assert_eq!(fma[i], a88[i] * b88[i] + a88[i], "mul_add lane {i}");
        }

        // Même produit sur Q1.15 (FRAC = 15).
        let a15: [Q1_15; 8] = core::array::from_fn(|_| Q1_15::from_raw(rng.next() as i16));
        let b15: [Q1_15; 8] = core::array::from_fn(|_| Q1_15::from_raw(rng.next() as i16));
        let mul15 = (FixedI16x8::from_array(a15) * FixedI16x8::from_array(b15)).to_array();
        for i in 0..8
        {
            assert_eq!(mul15[i], a15[i] * b15[i], "mul Q1.15 lane {i}");
        }
    }
}

#[test]
fn simd_i16x8_select_cmp_reduce() {
    let a = FixedI16x8::from_array(core::array::from_fn(|i| Q8_8::from(i as i16)));
    let b = FixedI16x8::splat(Q8_8::from(3));
    let mask = a.simd_lt(b); // lanes 0,1,2 < 3
    let sel = FixedI16x8::select(mask, a, b).to_array();
    for (i, &got) in sel.iter().enumerate()
    {
        let expected = if i < 3
        {
            Q8_8::from(i as i16)
        }
        else
        {
            Q8_8::from(3)
        };
        assert_eq!(got, expected, "select lane {i}");
    }
    assert!(a.simd_le(a).all(), "self ≤ self");
    assert!(a.simd_eq(a).all(), "self == self");

    // Réduction horizontale exacte : Σ 0..7 = 28 (addition virgule fixe exacte).
    let sum = FixedI16x8::from_array(core::array::from_fn(|i| Q8_8::from(i as i16))).reduce_sum();
    assert_eq!(sum, Q8_8::from(28), "reduce_sum");
}

#[test]
fn simd_i32x8_matches_scalar() {
    let mut rng = Lcg(0xABCDEF);
    for _ in 0..500
    {
        let a: [Q16_16; 8] = core::array::from_fn(|_| Q16_16::from_raw(rng.raw_i32()));
        let b: [Q16_16; 8] = core::array::from_fn(|_| Q16_16::from_raw(rng.raw_i32()));
        let va = FixedI32x8::from_array(a);
        let vb = FixedI32x8::from_array(b);

        let add = (va + vb).to_array();
        let sub = (va - vb).to_array();
        let mul = (va * vb).to_array();
        let neg = (-va).to_array();
        let mn = va.min(vb).to_array();
        let mx = va.max(vb).to_array();
        let ab = va.abs().to_array();
        for i in 0..8
        {
            assert_eq!(add[i], a[i] + b[i], "add lane {i}");
            assert_eq!(sub[i], a[i] - b[i], "sub lane {i}");
            assert_eq!(mul[i], a[i] * b[i], "mul lane {i}");
            assert_eq!(neg[i], -a[i], "neg lane {i}");
            assert_eq!(mn[i], a[i].min(b[i]), "min lane {i}");
            assert_eq!(mx[i], a[i].max(b[i]), "max lane {i}");
            assert_eq!(ab[i], a[i].abs(), "abs lane {i}");
        }
    }
}

#[test]
fn simd_i64x4_matches_scalar() {
    let mut rng = Lcg(0x5A5A);
    for _ in 0..500
    {
        let a: [Q32_32; 4] = core::array::from_fn(|_| FixedI64::from_raw(rng.next() as i64));
        let b: [Q32_32; 4] = core::array::from_fn(|_| FixedI64::from_raw(rng.next() as i64));
        let va = FixedI64x4::from_array(a);
        let vb = FixedI64x4::from_array(b);
        let add = (va + vb).to_array();
        let mul = (va * vb).to_array();
        let mn = va.min(vb).to_array();
        for i in 0..4
        {
            assert_eq!(add[i], a[i] + b[i], "add lane {i}");
            assert_eq!(mul[i], a[i] * b[i], "mul lane {i}");
            assert_eq!(mn[i], a[i].min(b[i]), "min lane {i}");
        }
    }
}

#[test]
fn simd_select_and_cmp() {
    let a = FixedI32x8::from_array(core::array::from_fn(|i| Q16_16::from(i as i32)));
    let b = FixedI32x8::splat(q16(3.0));
    let mask = a.simd_lt(b); // lanes 0,1,2 < 3
    let sel = FixedI32x8::select(mask, a, b).to_array();
    for (i, &got) in sel.iter().enumerate()
    {
        let expected = if (i as f64) < 3.0
        {
            Q16_16::from(i as i32)
        }
        else
        {
            q16(3.0)
        };
        assert_eq!(got, expected, "lane {i}");
    }
}

// ------------------------------------------------------------------ //
//  Réductions & déterminisme                                          //
// ------------------------------------------------------------------ //

#[test]
fn reductions_exact() {
    let data: Vec<Q16_16> = (1..=100).map(|i| q16(i as f64 * 0.5)).collect();
    // Σ 0.5·i pour i=1..100 = 0.5·5050 = 2525.
    assert_eq!(red::sum(&data).to_f64(), 2525.0);
    assert_eq!(red::l1_norm(&data).to_f64(), 2525.0);
    assert_eq!(red::max(&data).unwrap().to_f64(), 50.0);
    assert_eq!(red::min(&data).unwrap().to_f64(), 0.5);
    assert_eq!(red::argmax(&data), Some(99));
    assert_eq!(red::argmin(&data), Some(0));
    // Toutes valeurs positives : L∞ = max.
    assert_eq!(red::linf_norm(&data).to_f64(), 50.0);

    // dot([1,2,3],[4,5,6]) = 32.
    let x: Vec<Q16_16> = [1.0, 2.0, 3.0].iter().map(|&v| q16(v)).collect();
    let y: Vec<Q16_16> = [4.0, 5.0, 6.0].iter().map(|&v| q16(v)).collect();
    assert_eq!(red::dot(&x, &y).to_f64(), 32.0);
    // ‖[3,4]‖ = 5, ‖[3,4]‖∞ = 4.
    let v: Vec<Q16_16> = [3.0, 4.0].iter().map(|&t| q16(t)).collect();
    assert_eq!(red::l2_norm(&v).to_f64(), 5.0);
    assert_eq!(red::linf_norm(&v).to_f64(), 4.0);
    // L∞ prend la valeur absolue : ‖[-9, 1, 2]‖∞ = 9.
    let neg: Vec<Q16_16> = [-9.0, 1.0, 2.0].iter().map(|&t| q16(t)).collect();
    assert_eq!(red::linf_norm(&neg).to_f64(), 9.0);
    // cos(x, x) ≈ 1.
    let c = red::cosine_similarity(&v, &v);
    assert!((c.to_f64() - 1.0).abs() < 1e-3, "cos={c}");
}

#[test]
fn sum_is_order_independent() {
    // La somme virgule fixe est exacte : indépendante de l'ordre (déterminisme).
    let mut rng = Lcg(0xD);
    let mut data: Vec<Q32_32> = (0..257)
        .map(|_| FixedI64::from_raw((rng.raw_i32() as i64) << 8))
        .collect();
    let forward = red::sum(&data);
    data.reverse();
    let backward = red::sum(&data);
    assert_eq!(forward, backward, "somme dépendante de l'ordre !");
}

#[test]
fn reductions_simd_vs_scalar_dot() {
    // dot SIMD (chemin i32, N non multiple de 8) == référence scalaire exacte.
    let mut rng = Lcg(0x9);
    let n = 1000;
    let a: Vec<Q16_16> = (0..n)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 8))
        .collect();
    let b: Vec<Q16_16> = (0..n)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 8))
        .collect();
    let simd = red::dot(&a, &b);
    let mut acc: i128 = 0;
    for i in 0..n
    {
        acc += a[i].wrapping_mul(b[i]).to_raw() as i128;
    }
    assert_eq!(simd.to_raw() as i128, acc);
}

// ------------------------------------------------------------------ //
//  Algèbre linéaire : GEMM déterministe                               //
// ------------------------------------------------------------------ //

/// Référence naïve : triple boucle avec opérateurs enveloppants de `Fixed`.
/// Doit coïncider **bit-à-bit** avec `linalg::matmul` (somme enveloppante
/// associative + arrondi de produit identique).
fn naive_matmul<const F: u32>(
    a: &[FixedI32<F>],
    b: &[FixedI32<F>],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<FixedI32<F>> {
    let mut c = vec![FixedI32::<F>::from_raw(0); m * n];
    for i in 0..m
    {
        for j in 0..n
        {
            let mut acc = FixedI32::<F>::from_raw(0);
            for l in 0..k
            {
                acc += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

#[test]
fn transpose_roundtrip_and_known() {
    // 2×3 connu.
    let a = [1i32, 2, 3, 4, 5, 6].map(Q16_16::from);
    let t = linalg::transpose(&a, 2, 3); // 3×2
    let expected = [1i32, 4, 2, 5, 3, 6].map(Q16_16::from);
    assert_eq!(t, expected);
    // Double transposition = identité.
    let tt = linalg::transpose(&t, 3, 2);
    assert_eq!(tt, a.to_vec());
}

#[test]
fn matmul_known_small() {
    // A = [[1,2,3],[4,5,6]] (2×3), B = [[7,8],[9,10],[11,12]] (3×2).
    // C = [[58,64],[139,154]].
    let a = [1i32, 2, 3, 4, 5, 6].map(Q16_16::from);
    let b = [7i32, 8, 9, 10, 11, 12].map(Q16_16::from);
    let c = linalg::matmul(&a, &b, 2, 3, 2);
    let expected = [58i32, 64, 139, 154].map(Q16_16::from);
    assert_eq!(c, expected);
}

#[test]
fn matvec_known_small() {
    // A = [[1,2,3],[4,5,6]] (2×3), x = [1,0,-1] → y = [1-3, 4-6] = [-2,-2].
    let a = [1i32, 2, 3, 4, 5, 6].map(Q16_16::from);
    let x = [1i32, 0, -1].map(Q16_16::from);
    let y = linalg::matvec(&a, &x, 2, 3);
    assert_eq!(y, [(-2i32), -2].map(Q16_16::from));
}

#[test]
fn matmul_matches_naive_bit_exact() {
    // Égalité stricte SIMD/dot vs triple boucle enveloppante, tailles variées
    // (dont non multiples de 8 pour couvrir le reste scalaire du dot).
    let mut rng = Lcg(0xBEEF);
    for &(m, k, n) in &[
        (1, 1, 1),
        (3, 8, 5),
        (7, 13, 4),
        (8, 16, 8),
        (5, 1, 9),
        (4, 3, 1),
    ]
    {
        let a: Vec<Q16_16> = (0..m * k)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 8))
            .collect();
        let b: Vec<Q16_16> = (0..k * n)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 8))
            .collect();
        let got = linalg::matmul(&a, &b, m, k, n);
        let want = naive_matmul(&a, &b, m, k, n);
        assert_eq!(got, want, "matmul {m}×{k}·{k}×{n}");
        // matvec == matmul avec n = 1.
        let x: Vec<Q16_16> = (0..k)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 8))
            .collect();
        let yv = linalg::matvec(&a, &x, m, k);
        let ym = linalg::matmul(&a, &x, m, k, 1);
        assert_eq!(yv, ym, "matvec vs matmul(n=1) {m}×{k}");
    }
}

#[test]
fn matmul_float_reference() {
    // Valeurs modérées (pas d'overflow) : le GEMM virgule fixe approche la
    // référence f64 à quelques résolutions près (arrondi par produit).
    let mut rng = Lcg(0x1357);
    let (m, k, n) = (6, 10, 7);
    let a: Vec<Q16_16> = (0..m * k)
        .map(|_| Q16_16::try_from((rng.raw_i32() >> 20) as f64 / 8.0).unwrap())
        .collect();
    let b: Vec<Q16_16> = (0..k * n)
        .map(|_| Q16_16::try_from((rng.raw_i32() >> 20) as f64 / 8.0).unwrap())
        .collect();
    let c = linalg::matmul(&a, &b, m, k, n);
    for i in 0..m
    {
        for j in 0..n
        {
            let mut fref = 0.0f64;
            for l in 0..k
            {
                fref += a[i * k + l].to_f64() * b[l * n + j].to_f64();
            }
            let got = c[i * n + j].to_f64();
            // k produits, chacun arrondi à ≤ 1 résolution → borne k·2^-16.
            assert!(
                (got - fref).abs() <= (k as f64) / 65536.0,
                "C[{i},{j}] = {got} vs {fref}"
            );
        }
    }
}

#[test]
fn matmul_zero_dims() {
    // k = 0 : produit de matrices « vides » → C nulle m×n.
    let a: [Q16_16; 0] = [];
    let b: [Q16_16; 0] = [];
    let c = linalg::matmul(&a, &b, 2, 0, 3);
    assert_eq!(c, vec![Q16_16::zero(); 6]);
    // m = 0 → sortie vide.
    assert!(linalg::matmul(&a, &b, 0, 0, 3).is_empty());
}

#[test]
fn matmul_i64_storage() {
    // Le chemin de stockage i64 (dot scalaire, accumulation i128) donne aussi
    // l'égalité stricte avec la référence naïve.
    let a = [1i64, 2, 3, 4].map(Q32_32::from);
    let b = [5i64, 6, 7, 8].map(Q32_32::from);
    let c = linalg::matmul(&a, &b, 2, 2, 2);
    // [[1,2],[3,4]]·[[5,6],[7,8]] = [[19,22],[43,50]].
    let expected = [19i64, 22, 43, 50].map(Q32_32::from);
    assert_eq!(c, expected);
}

#[test]
#[should_panic(expected = "matmul")]
fn matmul_dim_mismatch_panics() {
    let a = [Q16_16::one(); 6]; // annoncé 2×3
    let b = [Q16_16::one(); 6]; // annoncé 3×2
    let _ = linalg::matmul(&a, &b, 2, 3, 3); // b.len()=6 ≠ 3×3=9 → panique
}

#[test]
fn matmul_bt_matches_matmul_with_explicit_transpose() {
    let mut rng = Lcg(0xB7_5A51);
    for &(m, k, n) in &[(1, 1, 1), (3, 8, 5), (7, 13, 4), (8, 16, 8), (5, 1, 9)]
    {
        let a: Vec<Q16_16> = (0..m * k)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 8))
            .collect();
        // bt joue le rôle de Bᵀ (n × k) : b (k × n) en est la transposée.
        let bt: Vec<Q16_16> = (0..n * k)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 8))
            .collect();
        let b = linalg::transpose(&bt, n, k);
        assert_eq!(
            linalg::matmul_bt(&a, &bt, m, k, n),
            linalg::matmul(&a, &b, m, k, n),
            "m={m} k={k} n={n}"
        );
    }
}

#[test]
fn matmul_bt_matches_naive_reference() {
    // A = [[1,2,3],[4,5,6]] (2×3) ; Bᵀ = [[1,0,-1],[2,1,0]] (2×3, donc B 3×2).
    // C = A·Bᵀᵀ : C[0,0] = 1-3 = -2, C[0,1] = 2+2 = 4,
    //             C[1,0] = 4-6 = -2, C[1,1] = 8+5 = 13.
    let a = [1i32, 2, 3, 4, 5, 6].map(Q16_16::from);
    let bt = [1i32, 0, -1, 2, 1, 0].map(Q16_16::from);
    let c = linalg::matmul_bt(&a, &bt, 2, 3, 2);
    assert_eq!(c, [-2i32, 4, -2, 13].map(Q16_16::from));
}

#[test]
#[should_panic(expected = "matmul_bt")]
fn matmul_bt_dim_mismatch_panics() {
    let a = [Q16_16::one(); 6]; // annoncé 2×3
    let bt = [Q16_16::one(); 6]; // annoncé 3×3 (attendu n×k = 3×3 = 9)
    let _ = linalg::matmul_bt(&a, &bt, 2, 3, 3);
}

// ------------------------------------------------------------------ //
//  Décompositions : Cholesky, LU à pivot partiel, déterminant          //
// ------------------------------------------------------------------ //

/// Matrice `n×n` symétrique définie positive aléatoire : `A = BᵀB + n·I`
/// (`B` à coefficients modestes pour éviter tout débordement). Le terme
/// `n·I` garantit un conditionnement raisonnable, pas seulement la
/// définie-positivité stricte.
fn random_spd(rng: &mut Lcg, n: usize) -> Vec<Q16_16> {
    let b: Vec<Q16_16> = (0..n * n)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 10))
        .collect();
    let bt = linalg::transpose(&b, n, n);
    let mut spd = linalg::matmul(&bt, &b, n, n, n);
    for i in 0..n
    {
        spd[i * n + i] += Q16_16::from_i32(n as i32);
    }
    spd
}

/// Matrice `n×n` à diagonale strictement dominante (donc inversible et bien
/// conditionnée), coefficients hors diagonale modestes.
fn random_diag_dominant(rng: &mut Lcg, n: usize) -> Vec<Q16_16> {
    let mut a: Vec<Q16_16> = (0..n * n)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 12))
        .collect();
    for i in 0..n
    {
        a[i * n + i] = Q16_16::from_i32(4 * n as i32);
    }
    a
}

/// Reconstruit `L·U` (`n×n`) à partir du buffer combiné renvoyé par
/// [`linalg::lu_decompose`] (diagonale unité de `L` rendue explicite).
fn lu_reconstruct(lu: &[Q16_16], n: usize) -> Vec<Q16_16> {
    let mut l = vec![Q16_16::zero(); n * n];
    let mut u = vec![Q16_16::zero(); n * n];
    for i in 0..n
    {
        l[i * n + i] = Q16_16::one();
        for j in 0..i
        {
            l[i * n + j] = lu[i * n + j];
        }
        for j in i..n
        {
            u[i * n + j] = lu[i * n + j];
        }
    }
    linalg::matmul(&l, &u, n, n, n)
}

#[test]
fn cholesky_known_exact() {
    // A = [[4,6],[6,25]] = L·Lᵀ avec L = [[2,0],[3,4]] (3²+4²=25) : exact en
    // entiers, aucun arrondi.
    let a = [4i32, 6, 6, 25].map(Q16_16::from);
    let l = linalg::cholesky(&a, 2).expect("SPD");
    let expected = [2i32, 0, 3, 4].map(Q16_16::from);
    assert_eq!(l, expected);
}

#[test]
fn cholesky_rejects_indefinite() {
    // det([[1,2],[2,1]]) = 1 - 4 = -3 < 0 : pas définie positive.
    let a = [1i32, 2, 2, 1].map(Q16_16::from);
    assert!(linalg::cholesky(&a, 2).is_none());
}

#[test]
fn cholesky_rejects_zero_matrix() {
    let a = vec![Q16_16::zero(); 9];
    assert!(linalg::cholesky(&a, 3).is_none());
}

#[test]
fn forward_back_substitution_known() {
    // L = [[2,0],[3,4]], b = [4, 22] → y : 2y0=4→y0=2 ; 3·2+4y1=22→y1=4.
    let l = [2i32, 0, 3, 4].map(Q16_16::from);
    let b = [4i32, 22].map(Q16_16::from);
    let y = linalg::forward_substitution(&l, &b, 2).unwrap();
    assert_eq!(y, [2i32, 4].map(Q16_16::from));

    // U = [[2,3],[0,4]], y = [2,4] → x : 4x1=4→x1=1 ; 2x0+3=2→x0=... solve
    // back : x1 = y1/U11 = 1 ; x0 = (y0 - U01·x1)/U00 = (2-3)/2 = -0.5.
    let u = [2i32, 3, 0, 4].map(Q16_16::from);
    let x = linalg::back_substitution(&u, &y, 2).unwrap();
    assert_eq!(x[1], Q16_16::one());
    assert_eq!(x[0], q16(-0.5));
}

#[test]
fn forward_substitution_singular_is_none() {
    let l = [0i32, 0, 3, 4].map(Q16_16::from); // L[0,0] = 0
    let b = [1i32, 1].map(Q16_16::from);
    assert!(linalg::forward_substitution(&l, &b, 2).is_none());
}

#[test]
fn cholesky_solve_matches_matvec_random_spd() {
    let mut rng = Lcg(0xC401_5ED0);
    for &n in &[1usize, 2, 3, 4, 6]
    {
        let a = random_spd(&mut rng, n);
        let x_true: Vec<Q16_16> = (0..n)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 10))
            .collect();
        let b = linalg::matvec(&a, &x_true, n, n);
        let x = linalg::cholesky_solve(&a, &b, n).expect("A construite SPD");
        let b_check = linalg::matvec(&a, &x, n, n);
        let tol = (n as f64) * 16.0 / 65536.0;
        for i in 0..n
        {
            let diff = (b_check[i].to_f64() - b[i].to_f64()).abs();
            assert!(diff <= tol, "n={n} i={i}: résidu {diff} > {tol}");
        }
    }
}

#[test]
fn cholesky_i64_storage_known() {
    // Même exemple exact que `cholesky_known_exact`, stockage i64 (Q32_32).
    let a = [4i64, 6, 6, 25].map(Q32_32::from);
    let l = linalg::cholesky(&a, 2).expect("SPD");
    let expected = [2i64, 0, 3, 4].map(Q32_32::from);
    assert_eq!(l, expected);
}

#[test]
fn lu_decompose_reconstructs_permuted_a() {
    // Nécessite un pivot dès k=0 : colonne 0 = [2,4,8], le maximum (8) est
    // en ligne 2 → swap(0,2).
    let a = [
        2i32, 1, 1, //
        4, 3, 3, //
        8, 7, 9,
    ]
    .map(Q16_16::from);
    let (lu, perm) = linalg::lu_decompose(&a, 3).expect("inversible");
    assert_eq!(perm[0], 2, "le pivot de la colonne 0 doit être la ligne 2");
    let recon = lu_reconstruct(&lu, 3);
    for i in 0..3
    {
        for j in 0..3
        {
            let want = a[perm[i] * 3 + j].to_f64();
            let got = recon[i * 3 + j].to_f64();
            assert!(
                (got - want).abs() <= 4.0 / 65536.0,
                "L·U[{i},{j}]={got} vs A[perm[{i}],{j}]={want}"
            );
        }
    }
}

#[test]
fn lu_decompose_random_diag_dominant_reconstructs() {
    let mut rng = Lcg(0x10CC_0FF5);
    for &n in &[1usize, 2, 3, 5, 7]
    {
        let a = random_diag_dominant(&mut rng, n);
        let (lu, perm) = linalg::lu_decompose(&a, n).expect("diag. dominante ⇒ inversible");
        let recon = lu_reconstruct(&lu, n);
        let tol = (n as f64) * 8.0 / 65536.0;
        for i in 0..n
        {
            for j in 0..n
            {
                let want = a[perm[i] * n + j].to_f64();
                let got = recon[i * n + j].to_f64();
                assert!(
                    (got - want).abs() <= tol,
                    "n={n} L·U[{i},{j}]={got} vs {want}"
                );
            }
        }
    }
}

#[test]
fn lu_decompose_detects_singular() {
    // Ligne 1 = 2 · ligne 0 : colonnes liées, matrice singulière.
    let a = [1i32, 2, 2, 4].map(Q16_16::from);
    assert!(linalg::lu_decompose(&a, 2).is_none());
}

#[test]
fn lu_solve_matches_matvec_random() {
    let mut rng = Lcg(0x5A1E_5010);
    for &n in &[1usize, 2, 3, 4, 6]
    {
        let a = random_diag_dominant(&mut rng, n);
        let x_true: Vec<Q16_16> = (0..n)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 10))
            .collect();
        let b = linalg::matvec(&a, &x_true, n, n);
        let x = linalg::lu_solve(&a, &b, n).expect("diag. dominante ⇒ inversible");
        let b_check = linalg::matvec(&a, &x, n, n);
        let tol = (n as f64) * 8.0 / 65536.0;
        for i in 0..n
        {
            let diff = (b_check[i].to_f64() - b[i].to_f64()).abs();
            assert!(diff <= tol, "n={n} i={i}: résidu {diff} > {tol}");
        }
    }
}

#[test]
fn determinant_known_values() {
    // Identité : déterminant 1.
    let id3 = [
        1i32, 0, 0, //
        0, 1, 0, //
        0, 0, 1,
    ]
    .map(Q16_16::from);
    assert_eq!(linalg::determinant(&id3, 3), Q16_16::one());

    // [[2,3],[1,5]] : pas de pivot (|2| ≥ |1| en colonne 0), facteur
    // d'élimination 1/2 exact en Q16.16 → déterminant exact 2·5 − 3·1 = 7.
    let a = [2i32, 3, 1, 5].map(Q16_16::from);
    assert_eq!(linalg::determinant(&a, 2), Q16_16::from_i32(7));

    // Matrice de transposition pure : un pivot ⇒ déterminant −1.
    let swap = [0i32, 1, 1, 0].map(Q16_16::from);
    assert_eq!(linalg::determinant(&swap, 2), -Q16_16::one());
}

#[test]
fn determinant_zero_for_singular() {
    let a = [1i32, 2, 2, 4].map(Q16_16::from);
    assert_eq!(linalg::determinant(&a, 2), Q16_16::zero());
}

// ------------------------------------------------------------------ //
//  QR (Householder) : moindres carrés                                 //
// ------------------------------------------------------------------ //

/// Matrice `m×n` (`m ≥ n`) de rang `n` garanti : les `n` premières lignes
/// forment un bloc `n×n` à diagonale dominante (donc inversible, cf.
/// `random_diag_dominant`), les `m−n` lignes suivantes ajoutent des
/// coefficients modestes qui ne peuvent pas faire chuter le rang.
fn random_full_rank(rng: &mut Lcg, m: usize, n: usize) -> Vec<Q16_16> {
    let mut a = vec![Q16_16::zero(); m * n];
    let top = random_diag_dominant(rng, n);
    a[..n * n].copy_from_slice(&top);
    for i in n..m
    {
        for j in 0..n
        {
            a[i * n + j] = Q16_16::from_raw(rng.raw_i32() >> 12);
        }
    }
    a
}

#[test]
fn qr_reconstructs_a_and_is_orthonormal() {
    let mut rng = Lcg(0x9A0F_00D5);
    for &(m, n) in &[(2usize, 2usize), (3, 2), (5, 3), (7, 4), (4, 4)]
    {
        let a = random_full_rank(&mut rng, m, n);
        let (q, r) = linalg::qr_decompose(&a, m, n).expect("rang plein");
        let tol = ((m + n) as f64) * 512.0 / 65536.0;

        // Reconstruction : Q·R ≈ A.
        let recon = linalg::matmul(&q, &r, m, n, n);
        for i in 0..m
        {
            for j in 0..n
            {
                let want = a[i * n + j].to_f64();
                let got = recon[i * n + j].to_f64();
                assert!(
                    (got - want).abs() <= tol,
                    "m={m} n={n} Q·R[{i},{j}]={got} vs {want}"
                );
            }
        }

        // Orthonormalité : Qᵀ·Q ≈ Iₙ.
        let qt = linalg::transpose(&q, m, n);
        let qtq = linalg::matmul(&qt, &q, n, m, n);
        for i in 0..n
        {
            for j in 0..n
            {
                let want = if i == j { 1.0 } else { 0.0 };
                let got = qtq[i * n + j].to_f64();
                assert!(
                    (got - want).abs() <= tol,
                    "m={m} n={n} QᵀQ[{i},{j}]={got} vs {want}"
                );
            }
        }
    }
}

#[test]
fn qr_solve_recovers_exact_solution_for_consistent_system() {
    let mut rng = Lcg(0x5057_5057);
    for &(m, n) in &[(3usize, 2usize), (5, 3), (8, 4)]
    {
        let a = random_full_rank(&mut rng, m, n);
        let x_true: Vec<Q16_16> = (0..n)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 10))
            .collect();
        let b = linalg::matvec(&a, &x_true, m, n);
        let x = linalg::qr_solve(&a, &b, m, n).expect("rang plein");
        let tol = (n as f64) * 32.0 / 65536.0;
        for i in 0..n
        {
            let diff = (x[i].to_f64() - x_true[i].to_f64()).abs();
            assert!(diff <= tol, "m={m} n={n} i={i}: {diff} > {tol}");
        }
    }
}

#[test]
fn qr_solve_matches_lu_solve_on_square_system() {
    let mut rng = Lcg(0x5A51_5A51);
    for &n in &[1usize, 2, 3, 4, 6]
    {
        let a = random_diag_dominant(&mut rng, n);
        let b: Vec<Q16_16> = (0..n)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 10))
            .collect();
        let x_qr = linalg::qr_solve(&a, &b, n, n).expect("diag. dominante ⇒ inversible");
        let x_lu = linalg::lu_solve(&a, &b, n).expect("diag. dominante ⇒ inversible");
        let tol = (n as f64) * 16.0 / 65536.0;
        for i in 0..n
        {
            let diff = (x_qr[i].to_f64() - x_lu[i].to_f64()).abs();
            assert!(
                diff <= tol,
                "n={n} i={i}: qr={} lu={}",
                x_qr[i].to_f64(),
                x_lu[i].to_f64()
            );
        }
    }
}

#[test]
fn qr_solve_residual_is_orthogonal_to_column_space() {
    // b quelconque (pas nécessairement dans l'image de A, système
    // surdéterminé inconsistant) : la condition d'optimalité des moindres
    // carrés impose Aᵀ·(A·x − b) ≈ 0.
    let mut rng = Lcg(0x0271_10DA);
    for &(m, n) in &[(5usize, 2usize), (8, 3), (10, 4)]
    {
        let a = random_full_rank(&mut rng, m, n);
        let b: Vec<Q16_16> = (0..m)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 8))
            .collect();
        let x = linalg::qr_solve(&a, &b, m, n).expect("rang plein");
        let ax = linalg::matvec(&a, &x, m, n);
        let residual: Vec<Q16_16> = (0..m).map(|i| ax[i] - b[i]).collect();
        let at = linalg::transpose(&a, m, n);
        let atr = linalg::matvec(&at, &residual, n, m);
        let tol = ((m + n) as f64) * 512.0 / 65536.0;
        for (j, &atrj) in atr.iter().enumerate()
        {
            assert!(
                atrj.to_f64().abs() <= tol,
                "m={m} n={n} j={j}: {}",
                atrj.to_f64()
            );
        }
    }
}

#[test]
#[should_panic(expected = "qr_decompose")]
fn qr_decompose_requires_m_ge_n() {
    let a = vec![Q16_16::zero(); 2 * 3]; // m=2, n=3 : m<n.
    let _ = linalg::qr_decompose(&a, 2, 3);
}

#[test]
#[should_panic(expected = "qr_decompose")]
fn qr_decompose_dim_mismatch_panics() {
    let a = vec![Q16_16::zero(); 5]; // annoncé 3×2 = 6 ≠ 5.
    let _ = linalg::qr_decompose(&a, 3, 2);
}

// ------------------------------------------------------------------ //
//  Jacobi (matrices symétriques) — décomposition spectrale            //
// ------------------------------------------------------------------ //

/// Matrice symétrique aléatoire `n×n` (pas nécessairement définie positive,
/// contrairement à `random_spd`) : `A = B + Bᵀ`, `B` à coefficients modestes.
fn random_symmetric(rng: &mut Lcg, n: usize) -> Vec<Q16_16> {
    let b: Vec<Q16_16> = (0..n * n)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 12))
        .collect();
    let bt = linalg::transpose(&b, n, n);
    (0..n * n).map(|i| b[i] + bt[i]).collect()
}

#[test]
fn jacobi_eigen_reconstructs_and_is_orthonormal() {
    let mut rng = Lcg(0xE16E_0001);
    for &n in &[1usize, 2, 3, 4, 5]
    {
        let a = random_symmetric(&mut rng, n);
        let (eigenvalues, v, sweeps) = linalg::jacobi_eigen(&a, n, q16(1e-4), 60)
            .expect("pas de débordement pour cette échelle de données");
        assert!(sweeps <= 60, "n={n}: sweeps={sweeps} > max_sweeps");

        // Reconstruction : A ≈ V · diag(λ) · Vᵀ.
        let mut lambda_v = vec![Q16_16::zero(); n * n];
        for i in 0..n
        {
            for j in 0..n
            {
                lambda_v[i * n + j] = v[i * n + j] * eigenvalues[j];
            }
        }
        let vt = linalg::transpose(&v, n, n);
        let reconstructed = linalg::matmul(&lambda_v, &vt, n, n, n);
        let tol = (n as f64) * 64.0 / 65536.0;
        for i in 0..n * n
        {
            let diff = (reconstructed[i].to_f64() - a[i].to_f64()).abs();
            assert!(
                diff <= tol,
                "n={n} i={i}: reconstruction {} vs {} (écart {diff} > {tol})",
                reconstructed[i].to_f64(),
                a[i].to_f64()
            );
        }

        // Orthonormalité : V · Vᵀ ≈ I.
        let vvt = linalg::matmul(&v, &vt, n, n, n);
        for i in 0..n
        {
            for j in 0..n
            {
                let want = if i == j { 1.0 } else { 0.0 };
                let diff = (vvt[i * n + j].to_f64() - want).abs();
                assert!(
                    diff <= tol,
                    "n={n} i={i} j={j}: V·Vᵀ {} vs {want} (écart {diff} > {tol})",
                    vvt[i * n + j].to_f64()
                );
            }
        }
    }
}

#[test]
fn jacobi_eigen_known_2x2_diagonal_converges_in_one_sweep() {
    // Déjà diagonale : aucune rotation nécessaire, converge dès la première
    // passe (qui sert aussi à vérifier la convergence).
    let a = [3i32, 0, 0, 7].map(Q16_16::from);
    let (eigenvalues, v, sweeps) = linalg::jacobi_eigen(&a, 2, q16(1e-4), 20).expect("diagonale");
    assert_eq!(eigenvalues, [q16(3.0), q16(7.0)]);
    assert_eq!(
        v,
        [Q16_16::one(), Q16_16::zero(), Q16_16::zero(), Q16_16::one()]
    );
    assert_eq!(sweeps, 1);
}

#[test]
fn jacobi_eigen_n1_trivial() {
    let a = [q16(5.0)];
    let (eigenvalues, v, sweeps) = linalg::jacobi_eigen(&a, 1, q16(1e-4), 10).expect("n=1");
    assert_eq!(eigenvalues, vec![q16(5.0)]);
    assert_eq!(v, vec![Q16_16::one()]);
    assert_eq!(sweeps, 1); // aucune paire (p,q) à n=1 : converge trivialement.
}

#[test]
fn jacobi_eigen_only_reads_lower_triangle() {
    // Partie supérieure incohérente (jamais lue), comme `cholesky`.
    let a_bogus_upper = [2i32, 999, 1, 2].map(Q16_16::from);
    let a_symmetric = [2i32, 1, 1, 2].map(Q16_16::from);
    let (ev1, v1, _) = linalg::jacobi_eigen(&a_bogus_upper, 2, q16(1e-4), 30).unwrap();
    let (ev2, v2, _) = linalg::jacobi_eigen(&a_symmetric, 2, q16(1e-4), 30).unwrap();
    assert_eq!(ev1, ev2);
    assert_eq!(v1, v2);
}

#[test]
fn jacobi_eigen_i64_storage() {
    // Même exemple exact que `jacobi_eigen_known_2x2_diagonal_converges_in_one_sweep`,
    // stockage i64 (Q32_32) : aucune transcendante requise (cf. en-tête de
    // module), donc généralisable au second stockage sans réécriture.
    let a = [3i64, 0, 0, 7].map(Q32_32::from);
    let (eigenvalues, v, sweeps) =
        linalg::jacobi_eigen(&a, 2, Q32_32::zero(), 20).expect("diagonale");
    assert_eq!(eigenvalues, [3i64, 7].map(Q32_32::from));
    assert_eq!(
        v,
        [Q32_32::one(), Q32_32::zero(), Q32_32::zero(), Q32_32::one()]
    );
    assert_eq!(sweeps, 1);
}

#[test]
#[should_panic(expected = "jacobi_eigen")]
fn jacobi_eigen_dim_mismatch_panics() {
    let a = vec![Q16_16::zero(); 5]; // annoncé 2×2 = 4 ≠ 5.
    let _ = linalg::jacobi_eigen(&a, 2, q16(1e-4), 10);
}

// ------------------------------------------------------------------ //
//  SVD (via Jacobi sur AᵀA)                                           //
// ------------------------------------------------------------------ //

#[test]
fn svd_known_diagonal_matrix() {
    let a = [3i32, 0, 0, 1].map(Q16_16::from);
    let (u, sigma, vt, sweeps) = linalg::svd(&a, 2, 2, q16(1e-4), 20).expect("pas de débordement");
    assert_eq!(sigma, [q16(3.0), q16(1.0)]);
    assert_eq!(
        u,
        [Q16_16::one(), Q16_16::zero(), Q16_16::zero(), Q16_16::one()]
    );
    assert_eq!(
        vt,
        [Q16_16::one(), Q16_16::zero(), Q16_16::zero(), Q16_16::one()]
    );
    assert_eq!(sweeps, 1);
}

#[test]
fn svd_reconstructs_a_and_is_orthonormal() {
    let mut rng = Lcg(0x5F4D_0001);
    for &(m, n) in &[(2usize, 2usize), (4, 2), (5, 3), (6, 4)]
    {
        let a = random_full_rank(&mut rng, m, n);
        let (u, sigma, vt, sweeps) =
            linalg::svd(&a, m, n, q16(1e-4), 60).expect("pas de débordement pour cette échelle");
        assert!(sweeps <= 60, "m={m} n={n}: sweeps={sweeps} > max_sweeps");

        for i in 1..n
        {
            assert!(
                sigma[i - 1] >= sigma[i],
                "m={m} n={n}: sigma non triée décroissante : {sigma:?}"
            );
        }

        // Reconstruction : A ≈ U · diag(σ) · Vᵀ.
        let mut u_sigma = vec![Q16_16::zero(); m * n];
        for i in 0..m
        {
            for j in 0..n
            {
                u_sigma[i * n + j] = u[i * n + j] * sigma[j];
            }
        }
        let reconstructed = linalg::matmul(&u_sigma, &vt, m, n, n);
        let tol = ((m + n) as f64) * 256.0 / 65536.0;
        for i in 0..m * n
        {
            let diff = (reconstructed[i].to_f64() - a[i].to_f64()).abs();
            assert!(
                diff <= tol,
                "m={m} n={n} i={i}: reconstruction {} vs {} (écart {diff} > {tol})",
                reconstructed[i].to_f64(),
                a[i].to_f64()
            );
        }

        // Orthonormalité : Uᵀ·U ≈ I_n et V·Vᵀ ≈ I_n (V = transpose(Vᵀ)).
        let ut = linalg::transpose(&u, m, n);
        let utu = linalg::matmul(&ut, &u, n, m, n);
        let v = linalg::transpose(&vt, n, n);
        let vvt = linalg::matmul(&v, &vt, n, n, n);
        for i in 0..n
        {
            for j in 0..n
            {
                let want = if i == j { 1.0 } else { 0.0 };
                let diff_u = (utu[i * n + j].to_f64() - want).abs();
                assert!(
                    diff_u <= tol,
                    "m={m} n={n} Uᵀ·U[{i},{j}]: {} vs {want}",
                    utu[i * n + j].to_f64()
                );
                let diff_v = (vvt[i * n + j].to_f64() - want).abs();
                assert!(
                    diff_v <= tol,
                    "m={m} n={n} V·Vᵀ[{i},{j}]: {} vs {want}",
                    vvt[i * n + j].to_f64()
                );
            }
        }
    }
}

#[test]
fn svd_solve_matches_qr_solve_on_full_rank_system() {
    let mut rng = Lcg(0x5F4D_0002);
    for &(m, n) in &[(4usize, 3usize), (6, 4)]
    {
        let a = random_full_rank(&mut rng, m, n);
        let b: Vec<Q16_16> = (0..m)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 8))
            .collect();
        let x_qr = linalg::qr_solve(&a, &b, m, n).expect("rang plein");
        let x_svd = linalg::svd_solve(&a, &b, m, n, q16(1e-4), 60).expect("pas de débordement");
        let tol = (m as f64) * 256.0 / 65536.0;
        for i in 0..n
        {
            let diff = (x_qr[i].to_f64() - x_svd[i].to_f64()).abs();
            assert!(
                diff <= tol,
                "m={m} n={n} i={i}: qr_solve {} vs svd_solve {} (écart {diff} > {tol})",
                x_qr[i].to_f64(),
                x_svd[i].to_f64()
            );
        }
    }
}

#[test]
fn svd_solve_gives_minimum_norm_solution_for_rank_deficient_system() {
    // Colonne 1 = 2× colonne 0 : rang 1 (déficient, n=2). Système consistant :
    // b = colonne 0 exactement (x=(1,0) est une solution parmi une infinité,
    // toutes de la forme (1,0) + t·(2,−1), (2,−1) engendrant l'espace nul).
    let a = [1i32, 2, 2, 4, 3, 6].map(Q16_16::from);
    let (m, n) = (3usize, 2usize);
    let b = [q16(1.0), q16(2.0), q16(3.0)];

    let x = linalg::svd_solve(&a, &b, m, n, q16(1e-3), 60)
        .expect("svd_solve doit réussir malgré le rang déficient");

    // Solution valide des moindres carrés : A·x ≈ b.
    let ax = linalg::matvec(&a, &x, m, n);
    for i in 0..m
    {
        let diff = (ax[i].to_f64() - b[i].to_f64()).abs();
        assert!(
            diff <= 0.05,
            "i={i}: A·x = {} vs b = {}",
            ax[i].to_f64(),
            b[i].to_f64()
        );
    }

    // Solution de **norme minimale** : orthogonale à l'espace nul (2,−1),
    // propriété distinctive de la pseudo-inverse de Moore-Penrose.
    let null_dot = x[0].to_f64() * 2.0 - x[1].to_f64();
    assert!(
        null_dot.abs() <= 0.05,
        "x devrait être orthogonal à l'espace nul de A : produit scalaire {null_dot}"
    );
}

#[test]
fn svd_i64_storage() {
    // Même exemple exact que `svd_known_diagonal_matrix`, stockage i64 (Q32_32).
    let a = [3i64, 0, 0, 1].map(Q32_32::from);
    let (u, sigma, vt, sweeps) =
        linalg::svd(&a, 2, 2, Q32_32::zero(), 20).expect("pas de débordement");
    assert_eq!(sigma, [3i64, 1].map(Q32_32::from));
    assert_eq!(
        u,
        [Q32_32::one(), Q32_32::zero(), Q32_32::zero(), Q32_32::one()]
    );
    assert_eq!(
        vt,
        [Q32_32::one(), Q32_32::zero(), Q32_32::zero(), Q32_32::one()]
    );
    assert_eq!(sweeps, 1);
}

#[test]
#[should_panic(expected = "svd")]
fn svd_requires_m_ge_n() {
    let a = vec![Q16_16::zero(); 2 * 3]; // m=2, n=3 : m<n.
    let _ = linalg::svd(&a, 2, 3, q16(1e-4), 20);
}

#[test]
#[should_panic(expected = "svd")]
fn svd_dim_mismatch_panics() {
    let a = vec![Q16_16::zero(); 5]; // annoncé 3×2 = 6 ≠ 5.
    let _ = linalg::svd(&a, 3, 2, q16(1e-4), 20);
}

// ------------------------------------------------------------------ //
//  Problème aux valeurs propres généralisé (A·x = λ·B·x)              //
// ------------------------------------------------------------------ //

#[test]
fn generalized_eig_symmetric_matches_jacobi_eigen_when_b_is_identity() {
    // B = I : la réduction de Cholesky est triviale (L = I), le problème
    // généralisé coïncide donc **bit à bit** avec jacobi_eigen(A) direct —
    // preuve croisée indépendante, sans référence séparée.
    let mut rng = Lcg(0x6E16_0001);
    for &n in &[1usize, 2, 3, 4, 5]
    {
        let a = random_symmetric(&mut rng, n);
        let mut b = vec![Q16_16::zero(); n * n];
        for i in 0..n
        {
            b[i * n + i] = Q16_16::one();
        }
        let (ev_want, v_want, sweeps_want) =
            linalg::jacobi_eigen(&a, n, q16(1e-4), 60).expect("pas de débordement");
        let (ev_got, v_got, sweeps_got) =
            linalg::generalized_eig_symmetric(&a, &b, n, q16(1e-4), 60)
                .expect("pas de débordement");
        assert_eq!(ev_got, ev_want, "n={n}: valeurs propres");
        assert_eq!(v_got, v_want, "n={n}: vecteurs propres");
        assert_eq!(sweeps_got, sweeps_want, "n={n}: sweeps");
    }
}

#[test]
fn generalized_eig_symmetric_known_diagonal_case() {
    // A, B diagonales : λᵢ = aᵢᵢ/bᵢᵢ, vecteur propre eᵢ/√bᵢᵢ (calcul à la
    // main, cf. en-tête de fonction pour la B-orthonormalité) — A = diag(6,2),
    // B = diag(4,1) ⟹ λ = (1.5, 2.0), vecteurs (0.5,0) et (0,1).
    let a = [6i32, 0, 0, 2].map(Q16_16::from);
    let b = [4i32, 0, 0, 1].map(Q16_16::from);
    let (eigenvalues, x, sweeps) =
        linalg::generalized_eig_symmetric(&a, &b, 2, q16(1e-4), 20).expect("pas de débordement");
    assert_eq!(eigenvalues, [q16(1.5), q16(2.0)]);
    assert_eq!(x, [q16(0.5), q16(0.0), q16(0.0), q16(1.0)]);
    assert_eq!(sweeps, 1); // déjà diagonale : aucune rotation nécessaire.
}

#[test]
fn generalized_eig_symmetric_solves_the_eigenvalue_equation_and_is_b_orthonormal() {
    // Propriété caractéristique (cf. en-tête de fonction), vérifiée sur des
    // A/B aléatoires plutôt qu'une référence indépendante réimplémentée :
    //  * A·xᵢ ≈ λᵢ·B·xᵢ pour chaque vecteur propre (l'équation elle-même) ;
    //  * Xᵀ·B·X ≈ I (B-orthonormalité, pas l'orthonormalité usuelle).
    let mut rng = Lcg(0x6E16_0002);
    for &n in &[2usize, 3, 4, 5, 6]
    {
        let a = random_symmetric(&mut rng, n);
        let b = random_spd(&mut rng, n);
        let (eigenvalues, x, sweeps) = linalg::generalized_eig_symmetric(&a, &b, n, q16(1e-4), 80)
            .expect("pas de débordement pour cette échelle");
        assert!(sweeps <= 80, "n={n}: sweeps={sweeps} > max_sweeps");

        let tol = (n * n) as f64 * 32.0 / 65536.0;

        // A·X et B·X·diag(λ) (colonne par colonne).
        let ax = linalg::matmul(&a, &x, n, n, n);
        let bx = linalg::matmul(&b, &x, n, n, n);
        for i in 0..n
        {
            for j in 0..n
            {
                let want = bx[i * n + j].to_f64() * eigenvalues[j].to_f64();
                let got = ax[i * n + j].to_f64();
                assert!(
                    (got - want).abs() <= tol,
                    "n={n} i={i} j={j}: A·X = {got} vs B·X·diag(λ) = {want} (écart > {tol})"
                );
            }
        }

        // Xᵀ·B·X ≈ I.
        let xt = linalg::transpose(&x, n, n);
        let xtb = linalg::matmul(&xt, &b, n, n, n);
        let xtbx = linalg::matmul(&xtb, &x, n, n, n);
        for i in 0..n
        {
            for j in 0..n
            {
                let want = if i == j { 1.0 } else { 0.0 };
                let got = xtbx[i * n + j].to_f64();
                assert!(
                    (got - want).abs() <= tol,
                    "n={n} i={i} j={j}: Xᵀ·B·X = {got} vs {want}"
                );
            }
        }
    }
}

#[test]
fn generalized_eig_symmetric_i64_storage() {
    // Même exemple exact que `generalized_eig_symmetric_known_diagonal_case`,
    // stockage i64 (Q32_32).
    let a = [6i64, 0, 0, 2].map(Q32_32::from);
    let b = [4i64, 0, 0, 1].map(Q32_32::from);
    let (eigenvalues, x, sweeps) = linalg::generalized_eig_symmetric(&a, &b, 2, Q32_32::zero(), 20)
        .expect("pas de débordement");
    assert_eq!(
        eigenvalues,
        [Q32_32::try_from(1.5).unwrap(), Q32_32::from(2i64)]
    );
    assert_eq!(
        x,
        [
            Q32_32::try_from(0.5).unwrap(),
            Q32_32::zero(),
            Q32_32::zero(),
            Q32_32::one()
        ]
    );
    assert_eq!(sweeps, 1);
}

#[test]
fn generalized_eig_symmetric_returns_none_for_non_spd_b() {
    // B non définie positive (mineur principal négatif) : cholesky(B)
    // échoue, generalized_eig_symmetric doit donc renvoyer None plutôt que
    // paniquer ou produire un résultat incohérent.
    let a = [Q16_16::one(); 4];
    let b = [-1i32, 0, 0, -1].map(Q16_16::from);
    assert_eq!(
        linalg::generalized_eig_symmetric(&a, &b, 2, q16(1e-4), 20),
        None
    );
}

#[test]
#[should_panic(expected = "generalized_eig_symmetric")]
fn generalized_eig_symmetric_a_dim_mismatch_panics() {
    let a = vec![Q16_16::zero(); 5]; // annoncé 2×2 = 4 ≠ 5.
    let b = vec![Q16_16::zero(); 4];
    let _ = linalg::generalized_eig_symmetric(&a, &b, 2, q16(1e-4), 20);
}

#[test]
#[should_panic(expected = "generalized_eig_symmetric")]
fn generalized_eig_symmetric_b_dim_mismatch_panics() {
    let a = vec![Q16_16::zero(); 4];
    let b = vec![Q16_16::zero(); 5]; // annoncé 2×2 = 4 ≠ 5.
    let _ = linalg::generalized_eig_symmetric(&a, &b, 2, q16(1e-4), 20);
}

// ------------------------------------------------------------------ //
//  Hessenberg + QR décalé (matrices quelconques, non symétriques)     //
// ------------------------------------------------------------------ //

/// Matrice quelconque `n×n` aléatoire (pas symétrisée, contrairement à
/// `random_symmetric`) : coefficients modestes, évite les débordements de test.
fn random_general(rng: &mut Lcg, n: usize) -> Vec<Q16_16> {
    (0..n * n)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 12))
        .collect()
}

fn eigenvalue_re_f64(e: linalg::Eigenvalue<Q16_16>) -> f64 {
    match e
    {
        linalg::Eigenvalue::Real(x) => x.to_f64(),
        linalg::Eigenvalue::Complex(re, _im) => re.to_f64(),
    }
}

#[test]
fn hessenberg_zero_below_first_subdiagonal() {
    let mut rng = Lcg(0xEC33_0001);
    for &n in &[1usize, 2, 3, 4, 5, 6]
    {
        let a = random_general(&mut rng, n);
        let h = linalg::hessenberg(&a, n).expect("pas de débordement pour cette échelle");
        for i in 0..n
        {
            for j in 0..n
            {
                if j + 1 < i
                {
                    assert_eq!(
                        h[i * n + j],
                        Q16_16::zero(),
                        "n={n} i={i} j={j}: H[i,j] devrait être exactement nul (Hessenberg)"
                    );
                }
            }
        }
    }
}

#[test]
fn hessenberg_preserves_trace() {
    let mut rng = Lcg(0xEC33_0002);
    for &n in &[1usize, 2, 3, 4, 5]
    {
        let a = random_general(&mut rng, n);
        let h = linalg::hessenberg(&a, n).expect("pas de débordement pour cette échelle");
        let trace_a: f64 = (0..n).map(|i| a[i * n + i].to_f64()).sum();
        let trace_h: f64 = (0..n).map(|i| h[i * n + i].to_f64()).sum();
        let tol = ((n * n) as f64) * 32.0 / 65536.0;
        assert!(
            (trace_a - trace_h).abs() <= tol,
            "n={n}: trace(A)={trace_a} vs trace(H)={trace_h}"
        );
    }
}

#[test]
fn hessenberg_n_below_3_is_identity() {
    // Toute matrice 0×0/1×1/2×2 est déjà de Hessenberg : réduction = copie.
    let a2 = [q16(1.0), q16(2.0), q16(3.0), q16(4.0)];
    assert_eq!(linalg::hessenberg(&a2, 2).unwrap(), a2.to_vec());
    let a1 = [q16(5.0)];
    assert_eq!(linalg::hessenberg(&a1, 1).unwrap(), a1.to_vec());
    let a0: [Q16_16; 0] = [];
    assert_eq!(linalg::hessenberg(&a0, 0).unwrap(), a0.to_vec());
}

#[test]
#[should_panic(expected = "hessenberg")]
fn hessenberg_dim_mismatch_panics() {
    let a = vec![Q16_16::zero(); 5]; // annoncé 3×3 = 9 ≠ 5.
    let _ = linalg::hessenberg(&a, 3);
}

#[test]
fn eigenvalues_general_matches_jacobi_eigen_on_symmetric_matrices() {
    // Sur une entrée symétrique, les deux algorithmes doivent s'accorder :
    // preuve croisée indépendante de eigenvalues_general contre jacobi_eigen,
    // déjà validé (reconstruction + orthonormalité).
    let mut rng = Lcg(0xEC33_0003);
    for &n in &[1usize, 2, 3, 4, 5]
    {
        let a = random_symmetric(&mut rng, n);
        let (want, _, _) = linalg::jacobi_eigen(&a, n, q16(1e-4), 60).expect("pas de débordement");
        let got = linalg::eigenvalues_general(&a, n, q16(1e-4), 200)
            .expect("pas de débordement / convergence");

        let mut want_sorted: Vec<f64> = want.iter().map(|w| w.to_f64()).collect();
        let mut got_sorted: Vec<f64> = got.into_iter().map(eigenvalue_re_f64).collect();
        want_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        got_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let tol = (n as f64) * 128.0 / 65536.0;
        for (w, g) in want_sorted.iter().zip(&got_sorted)
        {
            assert!(
                (w - g).abs() <= tol,
                "n={n}: valeurs propres non symétrique {got_sorted:?} vs Jacobi {want_sorted:?}"
            );
        }
    }
}

#[test]
fn eigenvalues_general_upper_triangular_gives_diagonal_directly() {
    // Triangulaire supérieure : sous-diagonale de Hessenberg déjà nulle sur
    // toute la matrice — chaque valeur propre se déflate immédiatement
    // (bloc 1×1), sans aucune itération QR (teste la recherche de déflation
    // seule, indépendamment de shifted_qr_step).
    let n = 4;
    #[rustfmt::skip]
    let a = [
        q16(1.0), q16(2.0), q16(3.0), q16(4.0),
        q16(0.0), q16(5.0), q16(6.0), q16(7.0),
        q16(0.0), q16(0.0), q16(8.0), q16(9.0),
        q16(0.0), q16(0.0), q16(0.0), q16(10.0),
    ];
    let got = linalg::eigenvalues_general(&a, n, q16(1e-4), 50).expect("triangulaire : direct");
    let mut got_sorted: Vec<f64> = got.into_iter().map(eigenvalue_re_f64).collect();
    got_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(got_sorted, vec![1.0, 5.0, 8.0, 10.0]);
}

#[test]
fn eigenvalues_general_companion_matrix_known_real_roots() {
    // Matrice compagnon du polynôme (x−1)(x−2)(x−3) = x³ − 6x² + 11x − 6 :
    // ses valeurs propres sont exactement les racines {1, 2, 3}.
    let n = 3;
    #[rustfmt::skip]
    let a = [
        q16(0.0), q16(0.0), q16(6.0),
        q16(1.0), q16(0.0), q16(-11.0),
        q16(0.0), q16(1.0), q16(6.0),
    ];
    let got = linalg::eigenvalues_general(&a, n, q16(1e-4), 200).expect("compagnon : converge");
    let mut got_sorted: Vec<f64> = got.into_iter().map(eigenvalue_re_f64).collect();
    got_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let tol = 5e-3;
    for (g, want) in got_sorted.iter().zip(&[1.0, 2.0, 3.0])
    {
        assert!((g - want).abs() <= tol, "racines {got_sorted:?} vs [1,2,3]");
    }
}

#[test]
fn eigenvalues_general_rotation_matrix_gives_complex_conjugate_pair() {
    // R(θ) = [[cos θ, −sin θ], [sin θ, cos θ]] : valeurs propres cos θ ± i·sin θ.
    let (cos_t, sin_t) = (0.5, 3f64.sqrt() / 2.0); // θ = π/3.
    let a = [q16(cos_t), q16(-sin_t), q16(sin_t), q16(cos_t)];
    let got = linalg::eigenvalues_general(&a, 2, q16(1e-4), 50).expect("2×2 : direct");
    let tol = 5e-3;
    for e in got
    {
        match e
        {
            linalg::Eigenvalue::Complex(re, im) =>
            {
                assert!((re.to_f64() - cos_t).abs() <= tol, "re={}", re.to_f64());
                assert!(
                    (im.to_f64().abs() - sin_t).abs() <= tol,
                    "im={}",
                    im.to_f64()
                );
            },
            linalg::Eigenvalue::Real(x) =>
            {
                panic!(
                    "rotation θ=π/3 : valeur propre réelle inattendue {}",
                    x.to_f64()
                )
            },
        }
    }
}

#[test]
fn eigenvalues_general_block_diagonal_two_rotations_no_iteration_needed() {
    // Bloc-diagonale de deux rotations 2×2 : couplage inter-blocs nul dès le
    // départ, donc chaque bloc se déflate directement (aucune itération QR).
    let n = 4;
    let (c1, s1) = (0.5, 3f64.sqrt() / 2.0); // θ₁ = π/3.
    let (c2, s2) = (0.0, 1.0); // θ₂ = π/2.
    #[rustfmt::skip]
    let a = [
        q16(c1), q16(-s1), q16(0.0), q16(0.0),
        q16(s1), q16(c1),  q16(0.0), q16(0.0),
        q16(0.0), q16(0.0), q16(c2), q16(-s2),
        q16(0.0), q16(0.0), q16(s2), q16(c2),
    ];
    let got = linalg::eigenvalues_general(&a, n, q16(1e-4), 50).expect("bloc-diagonale : direct");
    let mut pairs: Vec<(f64, f64)> = got
        .into_iter()
        .map(|e| match e
        {
            linalg::Eigenvalue::Complex(re, im) => (re.to_f64(), im.to_f64().abs()),
            linalg::Eigenvalue::Real(x) =>
            {
                panic!("bloc rotation : réelle inattendue {}", x.to_f64())
            },
        })
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let tol = 5e-3;
    let want = {
        let mut w = vec![(c1, s1), (c1, s1), (c2, s2), (c2, s2)];
        w.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        w
    };
    for ((re, im), (wre, wim)) in pairs.iter().zip(&want)
    {
        assert!(
            (re - wre).abs() <= tol && (im - wim).abs() <= tol,
            "{pairs:?} vs {want:?}"
        );
    }
}

#[test]
fn eigenvalues_general_preserves_trace_on_random_matrices() {
    // Somme des parties réelles des n valeurs propres renvoyées = trace(A)
    // (invariant valable que les valeurs propres soient réelles ou en paires
    // complexes conjuguées, cf. doc de fonction) — vérifie les cas génériques
    // nécessitant plusieurs itérations QR, pour lesquels aucune racine
    // fermée n'est disponible.
    let mut rng = Lcg(0xEC33_0004);
    for &n in &[2usize, 3, 4, 5, 6]
    {
        let a = random_general(&mut rng, n);
        let got = linalg::eigenvalues_general(&a, n, q16(1e-4), 300)
            .expect("pas de débordement / convergence");
        let sum_re: f64 = got.into_iter().map(eigenvalue_re_f64).sum();
        let trace_a: f64 = (0..n).map(|i| a[i * n + i].to_f64()).sum();
        // Tolérance proportionnelle à l'échelle réelle des données (l'erreur
        // absolue accumulée sur plusieurs itérations QR dépend de l'échelle
        // des coefficients, pas seulement de `n`) plus un plancher absolu.
        let max_abs: f64 = a.iter().map(|x| x.to_f64().abs()).fold(0.0, f64::max);
        let tol = (n as f64) * 1024.0 / 65536.0 + max_abs * (n as f64) * 0.1;
        assert!(
            (sum_re - trace_a).abs() <= tol,
            "n={n}: Σ Re(λ) = {sum_re} vs trace(A) = {trace_a}"
        );
    }
}

#[test]
fn eigenvalues_general_i64_storage() {
    // Même exemple (triangulaire connue) que les tests ci-dessus, stockage
    // i64 (Q32_32) : aucune transcendante requise, généralisable sans
    // réécriture (cf. jacobi_eigen/svd).
    let a = [3i64, 0, 0, 7].map(Q32_32::from);
    let got = linalg::eigenvalues_general(&a, 2, Q32_32::zero(), 20).expect("diagonale");
    let mut got_sorted: Vec<i64> = got
        .into_iter()
        .map(|e| match e
        {
            linalg::Eigenvalue::Real(x) => x.to_f64() as i64,
            linalg::Eigenvalue::Complex(re, _) => re.to_f64() as i64,
        })
        .collect();
    got_sorted.sort_unstable();
    assert_eq!(got_sorted, vec![3, 7]);
}

#[test]
fn eigenvalues_general_n0_and_n1_trivial() {
    let a0: [Q16_16; 0] = [];
    assert_eq!(
        linalg::eigenvalues_general(&a0, 0, q16(1e-4), 10).unwrap(),
        Vec::new()
    );
    let a1 = [q16(5.0)];
    assert_eq!(
        linalg::eigenvalues_general(&a1, 1, q16(1e-4), 10).unwrap(),
        vec![linalg::Eigenvalue::Real(q16(5.0))]
    );
}

#[test]
#[should_panic(expected = "eigenvalues_general")]
fn eigenvalues_general_dim_mismatch_panics() {
    let a = vec![Q16_16::zero(); 5]; // annoncé 3×3 = 9 ≠ 5.
    let _ = linalg::eigenvalues_general(&a, 3, q16(1e-4), 10);
}

#[test]
fn eigenvalues_general_converges_on_many_random_matrices() {
    // Robustesse du décalage de Wilkinson + décalage ad hoc de secours :
    // aucune non-convergence/débordement sur un large échantillon de
    // matrices aléatoires (pas seulement les cas structurés ci-dessus).
    for seed in 0..200u64
    {
        let mut rng = Lcg(0xABCD_0000u64.wrapping_add(seed));
        for &n in &[3usize, 4, 5, 6, 7, 8]
        {
            let a = random_general(&mut rng, n);
            let got = linalg::eigenvalues_general(&a, n, q16(1e-4), 500);
            assert!(
                got.is_some(),
                "seed={seed} n={n} : non-convergence ou débordement"
            );
        }
    }
}

#[test]
fn eigenvalues_general_converges_at_larger_scale() {
    // À `n` plus grand, le nombre d'itérations QR nécessaires croît avec le
    // nombre de valeurs propres à isoler une à une : `max_iter` doit croître
    // en proportion (convention classique ~30·n, ici généreusement 100·n).
    for seed in 0..5u64
    {
        let mut rng = Lcg(0xF16E_0000u64.wrapping_add(seed));
        for &n in &[16usize, 32, 48]
        {
            let a = random_general(&mut rng, n);
            let got = linalg::eigenvalues_general(&a, n, q16(1e-4), 100 * n);
            assert!(
                got.is_some(),
                "seed={seed} n={n} : non-convergence ou débordement"
            );
        }
    }
}

// ------------------------------------------------------------------ //
//  Vecteurs propres réels (itération inverse)                        //
// ------------------------------------------------------------------ //

/// Matrice `n×n` à valeurs propres RÉELLES connues et distinctes
/// (`eigenvalues`) : `A = P·diag(eigenvalues)·P⁻¹` pour `P` aléatoire
/// générique (coefficients modestes, diagonale légèrement renforcée pour
/// garantir l'inversibilité **sans** rendre `P` quasi diagonale) — les
/// colonnes de `P` sont alors exactement les vecteurs propres de `A`
/// (`A·P = P·D` ⟹ `A·pᵢ = λᵢ·pᵢ`), référence indépendante de la
/// construction pour [`linalg::eigenvector_real`].
///
/// Délibérément **pas** [`random_diag_dominant`] : une `P` quasi diagonale
/// rendrait `A` elle-même quasi diagonale, ce qui déclenche une déflation
/// immédiate (1×1) de `eigenvalues_general` dès la réduction de Hessenberg —
/// avant toute itération QR de raffinement — sur la base d'une
/// sous-diagonale déjà petite par construction plutôt que par convergence,
/// avec une précision dégradée sur la valeur propre lue. Une `P` générique
/// (coefficients comparables) évite ce cas limite.
fn matrix_with_known_real_eigenvectors(
    rng: &mut Lcg,
    n: usize,
    eigenvalues: &[Q16_16],
) -> (Vec<Q16_16>, Vec<Q16_16>) {
    assert_eq!(eigenvalues.len(), n);
    let mut p: Vec<Q16_16> = (0..n * n)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 8))
        .collect();
    for i in 0..n
    {
        p[i * n + i] += Q16_16::from_i32(3);
    }
    let mut p_inv = vec![Q16_16::zero(); n * n];
    for j in 0..n
    {
        let mut e_j = vec![Q16_16::zero(); n];
        e_j[j] = Q16_16::one();
        let col = linalg::lu_solve(&p, &e_j, n).expect("P inversible");
        for i in 0..n
        {
            p_inv[i * n + j] = col[i];
        }
    }
    let mut d = vec![Q16_16::zero(); n * n];
    for i in 0..n
    {
        d[i * n + i] = eigenvalues[i];
    }
    let pd = linalg::matmul(&p, &d, n, n, n);
    let a = linalg::matmul(&pd, &p_inv, n, n, n);
    (a, p)
}

/// Aligne `got` sur `want` à un signe près (l'itération inverse ne fixe un
/// vecteur propre qu'à un scalaire près) et renvoie l'écart max composante
/// par composante.
fn signed_max_diff(got: &[Q16_16], want: &[f64]) -> f64 {
    let dot_sign: f64 = got.iter().zip(want).map(|(&g, &w)| g.to_f64() * w).sum();
    let sign = if dot_sign < 0.0 { -1.0 } else { 1.0 };
    got.iter()
        .zip(want)
        .map(|(&g, &w)| (g.to_f64() - sign * w).abs())
        .fold(0.0, f64::max)
}

#[test]
fn eigenvector_real_matches_known_construction() {
    let mut rng = Lcg(0xE164_0001);
    let n = 3;
    let eigenvalues = [q16(2.0), q16(5.0), q16(9.0)]; // bien séparées
    let (a, p) = matrix_with_known_real_eigenvectors(&mut rng, n, &eigenvalues);

    for k in 0..n
    {
        let lambda = eigenvalues[k];
        let got = linalg::eigenvector_real(&a, n, lambda, q16(1e-5), 50)
            .expect("itération inverse converge");

        // Propriété directe, indépendante de la construction P : A·v ≈ λ·v.
        let av = linalg::matvec(&a, &got, n, n);
        for i in 0..n
        {
            let diff = (av[i].to_f64() - lambda.to_f64() * got[i].to_f64()).abs();
            assert!(
                diff <= 5e-3,
                "k={k} i={i}: A·v={} vs λ·v={}",
                av[i].to_f64(),
                lambda.to_f64() * got[i].to_f64()
            );
        }

        // Direction : colonne k de P, normalisée, à un signe près.
        let mut want: Vec<f64> = (0..n).map(|i| p[i * n + k].to_f64()).collect();
        let norm: f64 = want.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in want.iter_mut()
        {
            *x /= norm;
        }
        let diff = signed_max_diff(&got, &want);
        assert!(diff <= 5e-3, "k={k}: écart de direction {diff}");
    }
}

#[test]
fn eigenvector_real_converges_from_approximate_eigenvalue() {
    // L'itération inverse converge vers le vecteur propre de la valeur
    // propre la PLUS PROCHE de l'estimation fournie (propriété classique de
    // la méthode, cf. en-tête de fonction) : un eigenvalue légèrement
    // perturbé doit encore donner le bon vecteur, tant que la perturbation
    // reste petite devant l'écart aux autres valeurs propres (2,0/9,0 ici).
    let mut rng = Lcg(0xE164_0002);
    let n = 3;
    let eigenvalues = [q16(2.0), q16(5.0), q16(9.0)];
    let (a, p) = matrix_with_known_real_eigenvectors(&mut rng, n, &eigenvalues);

    let got = linalg::eigenvector_real(&a, n, q16(5.1), q16(1e-5), 50)
        .expect("converge malgré l'approximation");

    let mut want: Vec<f64> = (0..n).map(|i| p[i * n + 1].to_f64()).collect();
    let norm: f64 = want.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in want.iter_mut()
    {
        *x /= norm;
    }
    let diff = signed_max_diff(&got, &want);
    assert!(diff <= 1e-2, "écart de direction {diff}");
}

#[test]
fn eigenvector_real_matches_jacobi_eigen_known_2x2() {
    // A = [[2,1],[1,2]] : valeurs propres 3 et 1 (A=[[a,b],[b,a]] ⟹ λ=a±b,
    // calcul à la main), vecteurs propres (1,1)/√2 et (1,−1)/√2 — comparé
    // ici à jacobi_eigen (déjà validé), pas à la solution analytique
    // directement, preuve croisée entre les deux algorithmes.
    let a = [q16(2.0), q16(1.0), q16(1.0), q16(2.0)];
    let (eigenvalues, v, _) = linalg::jacobi_eigen(&a, 2, q16(1e-4), 20).expect("2×2 direct");

    for k in 0..2
    {
        let lambda = eigenvalues[k];
        let got = linalg::eigenvector_real(&a, 2, lambda, q16(1e-6), 30)
            .expect("itération inverse converge");
        let want: Vec<f64> = (0..2).map(|i| v[i * 2 + k].to_f64()).collect();
        let diff = signed_max_diff(&got, &want);
        assert!(diff <= 1e-3, "k={k}: écart de direction {diff}");
    }
}

#[test]
fn eigenvectors_general_matches_eigenvalue_equation() {
    let mut rng = Lcg(0xE164_0004);
    let n = 4;
    let eigenvalues_known = [q16(2.0), q16(5.0), q16(9.0), q16(13.0)];
    let (a, _p) = matrix_with_known_real_eigenvectors(&mut rng, n, &eigenvalues_known);

    let eigenvalues = linalg::eigenvalues_general(&a, n, q16(1e-4), 200)
        .expect("pas de débordement / convergence");
    let vectors = linalg::eigenvectors_general(&a, n, &eigenvalues, q16(1e-5), 50);

    assert_eq!(vectors.len(), n);
    for (i, (ev, v)) in eigenvalues.iter().zip(&vectors).enumerate()
    {
        let linalg::Eigenvalue::Real(lambda) = *ev
        else
        {
            panic!(
                "i={i}: valeur propre complexe inattendue pour une matrice à spectre réel connu"
            );
        };
        let v = v
            .as_ref()
            .unwrap_or_else(|| panic!("i={i}: itération inverse doit converger"));
        let av = linalg::matvec(&a, v, n, n);
        for j in 0..n
        {
            let diff = (av[j].to_f64() - lambda.to_f64() * v[j].to_f64()).abs();
            assert!(
                diff <= 1e-2,
                "i={i} j={j}: A·v={} vs λ·v={}",
                av[j].to_f64(),
                lambda.to_f64() * v[j].to_f64()
            );
        }
    }
}

#[test]
fn eigenvectors_general_returns_none_for_complex_eigenvalues() {
    // Rotation 90° : valeurs propres ±i, purement complexes — aucune valeur
    // propre réelle, donc aucun vecteur propre réel à récupérer.
    let n = 2;
    let a = [q16(0.0), q16(-1.0), q16(1.0), q16(0.0)];
    let eigenvalues = linalg::eigenvalues_general(&a, n, q16(1e-4), 50).expect("2×2 direct");
    assert!(
        eigenvalues
            .iter()
            .all(|e| matches!(e, linalg::Eigenvalue::Complex(_, _)))
    );

    let vectors = linalg::eigenvectors_general(&a, n, &eigenvalues, q16(1e-5), 50);
    assert_eq!(vectors, vec![None, None]);
}

#[test]
fn eigenvector_real_i64_storage() {
    // Diagonale : valeurs propres = coefficients diagonaux, vecteur propre
    // de 7 = base canonique (0,1) à un signe près (calcul immédiat).
    let a = [3i64, 0, 0, 7].map(Q32_32::from);
    let got =
        linalg::eigenvector_real(&a, 2, Q32_32::from(7i64), Q32_32::zero(), 20).expect("converge");
    assert!(
        got[0].to_f64().abs() <= 1e-4,
        "composante 0 : {}",
        got[0].to_f64()
    );
    assert!(
        (got[1].to_f64().abs() - 1.0).abs() <= 1e-4,
        "composante 1 : {}",
        got[1].to_f64()
    );
}

#[test]
#[should_panic(expected = "eigenvector_real")]
fn eigenvector_real_dim_mismatch_panics() {
    let a = vec![Q16_16::zero(); 5]; // annoncé 3×3 = 9 ≠ 5.
    let _ = linalg::eigenvector_real(&a, 3, Q16_16::zero(), q16(1e-4), 10);
}

// ------------------------------------------------------------------ //
//  Racines de polynôme (matrice compagnon + eigenvalues_general)      //
// ------------------------------------------------------------------ //

fn sorted_real_parts(roots: Vec<linalg::Eigenvalue<Q16_16>>) -> Vec<f64> {
    let mut v: Vec<f64> = roots.into_iter().map(eigenvalue_re_f64).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

#[test]
fn companion_matrix_known_cubic_example() {
    // (x−1)(x−2)(x−3) = x³ − 6x² + 11x − 6 : même exemple que
    // eigenvalues_general_companion_matrix_known_real_roots, construit ici
    // via companion_matrix plutôt qu'à la main — preuve croisée directe.
    let coeffs = [q16(1.0), q16(-6.0), q16(11.0), q16(-6.0)];
    let got = linalg::companion_matrix(&coeffs);
    #[rustfmt::skip]
    let want = [
        q16(0.0), q16(0.0), q16(6.0),
        q16(1.0), q16(0.0), q16(-11.0),
        q16(0.0), q16(1.0), q16(6.0),
    ];
    assert_eq!(got, want);
}

#[test]
fn poly_roots_quadratic_known_real_roots() {
    // x² − 5x + 6 = (x−2)(x−3).
    let coeffs = [q16(1.0), q16(-5.0), q16(6.0)];
    let got = linalg::poly_roots(&coeffs, q16(1e-4), 50).expect("quadratique : converge");
    let got = sorted_real_parts(got);
    for (g, want) in got.iter().zip(&[2.0, 3.0])
    {
        assert!((g - want).abs() <= 5e-3, "racines {got:?} vs [2,3]");
    }
}

#[test]
fn poly_roots_cubic_known_real_roots() {
    // (x−1)(x−2)(x−3) = x³ − 6x² + 11x − 6.
    let coeffs = [q16(1.0), q16(-6.0), q16(11.0), q16(-6.0)];
    let got = linalg::poly_roots(&coeffs, q16(1e-4), 200).expect("cubique : converge");
    let got = sorted_real_parts(got);
    for (g, want) in got.iter().zip(&[1.0, 2.0, 3.0])
    {
        assert!((g - want).abs() <= 5e-3, "racines {got:?} vs [1,2,3]");
    }
}

#[test]
fn poly_roots_quintic_well_separated_real_roots() {
    // (x−1)(x−2)(x−3)(x−4)(x−5) = x⁵ − 15x⁴ + 85x³ − 225x² + 274x − 120 :
    // degré plus élevé, racines réelles bien séparées — plusieurs itérations
    // QR nécessaires, aucune racine fermée disponible autrement.
    let coeffs = [
        q16(1.0),
        q16(-15.0),
        q16(85.0),
        q16(-225.0),
        q16(274.0),
        q16(-120.0),
    ];
    let got = linalg::poly_roots(&coeffs, q16(1e-4), 1000).expect("quintique : converge");
    let got = sorted_real_parts(got);
    for (g, want) in got.iter().zip(&[1.0, 2.0, 3.0, 4.0, 5.0])
    {
        assert!((g - want).abs() <= 5e-2, "racines {got:?} vs [1,2,3,4,5]");
    }
}

#[test]
fn poly_roots_complex_conjugate_pair() {
    // x² + 1 = 0 : racines ±i.
    let coeffs = [q16(1.0), q16(0.0), q16(1.0)];
    let got = linalg::poly_roots(&coeffs, q16(1e-4), 50).expect("x²+1 : direct");
    let tol = 5e-3;
    for e in got
    {
        match e
        {
            linalg::Eigenvalue::Complex(re, im) =>
            {
                assert!(re.to_f64().abs() <= tol, "re={}", re.to_f64());
                assert!((im.to_f64().abs() - 1.0).abs() <= tol, "im={}", im.to_f64());
            },
            linalg::Eigenvalue::Real(x) =>
            {
                panic!("x²+1 : racine réelle inattendue {}", x.to_f64())
            },
        }
    }
}

#[test]
fn poly_roots_non_monic_leading_coefficient() {
    // 2x² − 10x + 12 = 2·(x−2)·(x−3) : mêmes racines que la forme monique,
    // le coefficient dominant non unitaire doit être normalisé correctement.
    let coeffs = [q16(2.0), q16(-10.0), q16(12.0)];
    let got = linalg::poly_roots(&coeffs, q16(1e-4), 50).expect("non monique : converge");
    let got = sorted_real_parts(got);
    for (g, want) in got.iter().zip(&[2.0, 3.0])
    {
        assert!((g - want).abs() <= 5e-3, "racines {got:?} vs [2,3]");
    }
}

#[test]
fn poly_roots_repeated_root() {
    // (x−2)² = x² − 4x + 4 : racine double, cas potentiellement délicat pour
    // une itération QR (valeurs propres confondues).
    let coeffs = [q16(1.0), q16(-4.0), q16(4.0)];
    let got = linalg::poly_roots(&coeffs, q16(1e-4), 100).expect("racine double : converge");
    let got = sorted_real_parts(got);
    for &g in &got
    {
        assert!((g - 2.0).abs() <= 1e-2, "racines {got:?} vs [2,2]");
    }
}

#[test]
fn poly_roots_linear_degree_one() {
    // 3x − 6 = 0 : racine unique x = 2 (bloc 1×1 direct, aucune itération).
    let coeffs = [q16(3.0), q16(-6.0)];
    let got = linalg::poly_roots(&coeffs, q16(1e-4), 10).expect("linéaire : direct");
    assert_eq!(got.len(), 1);
    match got[0]
    {
        linalg::Eigenvalue::Real(x) => assert!((x.to_f64() - 2.0).abs() <= 1e-3),
        other => panic!("linéaire : racine complexe inattendue {other:?}"),
    }
}

#[test]
fn poly_roots_i64_storage() {
    // Même exemple que poly_roots_quadratic_known_real_roots, stockage i64
    // (Q32_32) : aucune transcendante requise, généralisable sans réécriture
    // (cf. jacobi_eigen/svd/eigenvalues_general).
    let coeffs = [
        Q32_32::try_from(1.0).unwrap(),
        Q32_32::try_from(-5.0).unwrap(),
        Q32_32::try_from(6.0).unwrap(),
    ];
    let got = linalg::poly_roots(&coeffs, Q32_32::zero(), 50).expect("quadratique i64 : converge");
    let mut got_f64: Vec<f64> = got
        .into_iter()
        .map(|e| match e
        {
            linalg::Eigenvalue::Real(x) => Q32_32::to_f64(x),
            linalg::Eigenvalue::Complex(re, _) => Q32_32::to_f64(re),
        })
        .collect();
    got_f64.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for (g, want) in got_f64.iter().zip(&[2.0, 3.0])
    {
        assert!((g - want).abs() <= 5e-3, "racines {got_f64:?} vs [2,3]");
    }
}

#[test]
#[should_panic(expected = "companion_matrix")]
fn companion_matrix_rejects_too_few_coefficients() {
    let coeffs = [q16(5.0)]; // un seul coefficient : degré 0, non supporté.
    let _ = linalg::companion_matrix(&coeffs);
}

#[test]
#[should_panic(expected = "companion_matrix")]
fn companion_matrix_rejects_zero_leading_coefficient() {
    let coeffs = [Q16_16::zero(), q16(1.0), q16(2.0)];
    let _ = linalg::companion_matrix(&coeffs);
}

#[test]
#[should_panic(expected = "companion_matrix")]
fn poly_roots_rejects_too_few_coefficients() {
    let coeffs = [q16(5.0)];
    let _ = linalg::poly_roots(&coeffs, q16(1e-4), 10);
}

#[test]
fn poly_roots_converges_at_larger_degree() {
    // Comme eigenvalues_general_converges_at_larger_scale : robustesse à un
    // degré plus élevé (coefficients aléatoires modestes, pas de racine
    // connue à l'avance — seule la convergence est vérifiée ici).
    for seed in 0..5u64
    {
        let mut rng = Lcg(0x9012_0000u64.wrapping_add(seed));
        for &degree in &[8usize, 16, 24]
        {
            let mut coeffs = vec![Q16_16::zero(); degree + 1];
            coeffs[0] = Q16_16::one(); // coefficient dominant non nul (forme monique directe).
            for c in coeffs.iter_mut().skip(1)
            {
                *c = Q16_16::from_raw(rng.raw_i32() >> 12);
            }
            let got = linalg::poly_roots(&coeffs, q16(1e-4), 100 * degree);
            assert!(
                got.is_some(),
                "seed={seed} degree={degree} : non-convergence ou débordement"
            );
        }
    }
}

// ------------------------------------------------------------------ //
//  Exponentielle de matrice (mise à l'échelle et carrés répétés)      //
// ------------------------------------------------------------------ //

#[test]
fn matrix_exp_zero_is_identity() {
    for &n in &[1usize, 2, 3, 5]
    {
        let a = vec![Q16_16::zero(); n * n];
        let got = linalg::matrix_exp(&a, n).expect("zéro : pas de débordement");
        for i in 0..n
        {
            for j in 0..n
            {
                let want = if i == j { 1.0 } else { 0.0 };
                let diff = (got[i * n + j].to_f64() - want).abs();
                assert!(
                    diff <= 1e-3,
                    "n={n} [{i},{j}] = {} vs {want}",
                    got[i * n + j].to_f64()
                );
            }
        }
    }
}

#[test]
fn matrix_exp_diagonal_matches_scalar_exp() {
    #[rustfmt::skip]
    let a = [
        q16(0.3), q16(0.0), q16(0.0),
        q16(0.0), q16(-0.5), q16(0.0),
        q16(0.0), q16(0.0), q16(0.2),
    ];
    let got = linalg::matrix_exp(&a, 3).expect("diagonale : pas de débordement");
    let want = [0.3f64.exp(), (-0.5f64).exp(), 0.2f64.exp()];
    for i in 0..3
    {
        let diff = (got[i * 3 + i].to_f64() - want[i]).abs();
        assert!(
            diff <= 5e-3,
            "diag[{i}] = {} vs {}",
            got[i * 3 + i].to_f64(),
            want[i]
        );
        for j in 0..3
        {
            if i != j
            {
                assert!(
                    got[i * 3 + j].to_f64().abs() <= 5e-3,
                    "hors-diagonale [{i},{j}] = {}",
                    got[i * 3 + j].to_f64()
                );
            }
        }
    }
}

#[test]
fn matrix_exp_so3_generator_matches_rotation_matrix() {
    // e^{θ·K} = matrice de rotation de Quaternion::from_axis_angle (même
    // axe/angle) : application exponentielle classique so(3) → SO(3).
    let theta = 0.6; // rad, modéré (bien approximé par le Padé [3/3]).
    #[rustfmt::skip]
    let k = [
        q16(0.0), q16(-1.0), q16(0.0),
        q16(1.0), q16(0.0), q16(0.0),
        q16(0.0), q16(0.0), q16(0.0),
    ];
    let theta_k: Vec<Q16_16> = k.iter().map(|&x| x * q16(theta)).collect();
    let got = linalg::matrix_exp(&theta_k, 3).expect("so(3) : pas de débordement");

    let quat =
        Quaternion::from_axis_angle([Q16_16::zero(), Q16_16::zero(), Q16_16::one()], q16(theta));
    let want = quat.to_rotation_matrix();
    for i in 0..3
    {
        for j in 0..3
        {
            let diff = (got[i * 3 + j].to_f64() - want[i][j].to_f64()).abs();
            assert!(
                diff <= 1e-2,
                "R[{i},{j}] = {} vs {}",
                got[i * 3 + j].to_f64(),
                want[i][j].to_f64()
            );
        }
    }
}

#[test]
fn matrix_exp_inverse_is_negative_exp() {
    let a = [q16(0.2), q16(0.1), q16(-0.15), q16(0.05)];
    let neg_a: Vec<Q16_16> = a.iter().map(|&x| Q16_16::zero() - x).collect();
    let ea = linalg::matrix_exp(&a, 2).expect("A : pas de débordement");
    let ena = linalg::matrix_exp(&neg_a, 2).expect("-A : pas de débordement");
    let prod = linalg::matmul(&ea, &ena, 2, 2, 2);
    for i in 0..2
    {
        for j in 0..2
        {
            let want = if i == j { 1.0 } else { 0.0 };
            let diff = (prod[i * 2 + j].to_f64() - want).abs();
            assert!(
                diff <= 1e-2,
                "prod[{i},{j}] = {} vs {want}",
                prod[i * 2 + j].to_f64()
            );
        }
    }
}

#[test]
fn matrix_exp_commuting_doubling() {
    // e^A · e^A = e^{2A} (A commute avec elle-même).
    let a = [q16(0.15), q16(-0.1), q16(0.2), q16(0.05)];
    let two_a: Vec<Q16_16> = a.iter().map(|&x| x + x).collect();
    let ea = linalg::matrix_exp(&a, 2).expect("A : pas de débordement");
    let e2a_direct = linalg::matrix_exp(&two_a, 2).expect("2A : pas de débordement");
    let e2a_squared = linalg::matmul(&ea, &ea, 2, 2, 2);
    for i in 0..4
    {
        let diff = (e2a_direct[i].to_f64() - e2a_squared[i].to_f64()).abs();
        assert!(
            diff <= 1e-2,
            "i={i}: {} vs {}",
            e2a_direct[i].to_f64(),
            e2a_squared[i].to_f64()
        );
    }
}

#[test]
fn matrix_exp_i64_storage() {
    let a = [
        Q32_32::try_from(0.1).unwrap(),
        Q32_32::zero(),
        Q32_32::zero(),
        Q32_32::try_from(0.2).unwrap(),
    ];
    let got = linalg::matrix_exp(&a, 2).expect("i64 : pas de débordement");
    let want = [0.1f64.exp(), 0.2f64.exp()];
    for i in 0..2
    {
        let diff = (Q32_32::to_f64(got[i * 2 + i]) - want[i]).abs();
        assert!(
            diff <= 5e-3,
            "diag[{i}] = {} vs {}",
            Q32_32::to_f64(got[i * 2 + i]),
            want[i]
        );
    }
}

#[test]
fn matrix_exp_n0_trivial() {
    let a: [Q16_16; 0] = [];
    assert_eq!(linalg::matrix_exp(&a, 0).unwrap(), Vec::new());
}

#[test]
#[should_panic(expected = "matrix_exp")]
fn matrix_exp_dim_mismatch_panics() {
    let a = vec![Q16_16::zero(); 5]; // annoncé 3×3 = 9 ≠ 5.
    let _ = linalg::matrix_exp(&a, 3);
}

// ------------------------------------------------------------------ //
//  Activations quantifiées                                            //
// ------------------------------------------------------------------ //

#[test]
fn activation_relu_family_exact() {
    // relu / relu6 / clamp / hardtanh : exacts en virgule fixe (min/max/affine).
    assert_eq!(act::relu(q16(2.5)), q16(2.5));
    assert_eq!(act::relu(q16(-2.5)), Q16_16::zero());
    assert_eq!(act::relu(Q16_16::zero()), Q16_16::zero());

    assert_eq!(act::relu6(q16(10.0)), q16(6.0));
    assert_eq!(act::relu6(q16(3.5)), q16(3.5));
    assert_eq!(act::relu6(q16(-1.0)), Q16_16::zero());

    assert_eq!(act::clamp(q16(5.0), q16(-2.0), q16(2.0)), q16(2.0));
    assert_eq!(act::clamp(q16(-5.0), q16(-2.0), q16(2.0)), q16(-2.0));
    assert_eq!(act::clamp(q16(1.0), q16(-2.0), q16(2.0)), q16(1.0));

    assert_eq!(act::hardtanh(q16(4.0), q16(-1.0), q16(1.0)), q16(1.0));
    assert_eq!(act::hardtanh(q16(-4.0), q16(-1.0), q16(1.0)), q16(-1.0));
    assert_eq!(act::hardtanh(q16(0.25), q16(-1.0), q16(1.0)), q16(0.25));

    // leaky_relu : branche positive identité, négative pentée (exacte ici).
    assert_eq!(act::leaky_relu(q16(3.0), q16(0.5)), q16(3.0));
    assert_eq!(act::leaky_relu(q16(-4.0), q16(0.5)), q16(-2.0));

    // Même famille sur le stockage i16 (Q8.8, NumericScalar).
    let a = Q8_8::from(10);
    assert_eq!(act::relu6(a), Q8_8::from(6));
    assert_eq!(act::relu(-a), Q8_8::zero());
}

#[test]
fn activation_relu_family_over_f64() {
    // Généricité flottante : les mêmes fonctions, exactes sur f64.
    assert_eq!(act::relu(-2.0_f64), 0.0);
    assert_eq!(act::relu(2.0_f64), 2.0);
    assert_eq!(act::relu6(9.0_f64), 6.0);
    assert_eq!(act::clamp(5.0_f64, -1.0, 1.0), 1.0);
    assert_eq!(act::leaky_relu(-4.0_f64, 0.1), -0.4);
}

#[test]
fn activation_hardsigmoid_hardswish() {
    // Régions saturées (nettement au-delà de ±3, insensibles à l'arrondi de
    // recip) : ≤ −3 → 0, ≥ 3 → 1 (hardsigmoid).
    assert_eq!(act::hardsigmoid(q16(-4.0)), Q16_16::zero());
    assert_eq!(act::hardsigmoid(q16(-10.0)), Q16_16::zero());
    assert_eq!(act::hardsigmoid(q16(4.0)), Q16_16::one());
    assert_eq!(act::hardsigmoid(q16(10.0)), Q16_16::one());

    // hardswish : nulle bien sous −3, identité bien au-dessus de 3 (x·1 exact).
    assert_eq!(act::hardswish(q16(-5.0)), Q16_16::zero());
    assert_eq!(act::hardswish(q16(5.0)), q16(5.0));

    // Zone affine : comparaison à la référence f64 à quelques résolutions
    // (1/6 et 1/2 sont approchés par recip en virgule fixe).
    let hs_ref = |x: f64| (x / 6.0 + 0.5).clamp(0.0, 1.0);
    for &x in &[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    {
        let got = act::hardsigmoid(q16(x)).to_f64();
        assert!(
            (got - hs_ref(x)).abs() <= 6.0 / 65536.0,
            "hardsigmoid({x}) = {got} vs {}",
            hs_ref(x)
        );
        let gotw = act::hardswish(q16(x)).to_f64();
        let refw = x * hs_ref(x);
        assert!(
            (gotw - refw).abs() <= 12.0 / 65536.0,
            "hardswish({x}) = {gotw} vs {refw}"
        );
    }

    // Cohérence flottante : hardsigmoid(0) = 0.5 exact sur f64.
    assert_eq!(act::hardsigmoid(0.0_f64), 0.5);
    assert_eq!(act::hardswish(0.0_f64), 0.0);
}

#[test]
fn activation_gelu() {
    // GELU(0) = 0 exact (erf(0) = 0).
    assert_eq!(act::gelu(Q16_16::zero()), Q16_16::zero());
    assert_eq!(act::gelu(0.0_f64), 0.0);

    // Comparaison à la référence exacte 0.5·x·(1+erf(x/√2)), via la série
    // indépendante `erf_series_ref` (définie plus bas dans ce module).
    let gelu_ref = |x: f64| 0.5 * x * (1.0 + erf_series_ref(x / core::f64::consts::SQRT_2));
    for &x in &[-3.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0]
    {
        let want = gelu_ref(x);
        let got = act::gelu(q16(x)).to_f64();
        assert!(
            (got - want).abs() <= 8.0 * ULP16,
            "gelu({x}) = {got} vs {want}"
        );
        let gotf = act::gelu(x);
        assert!(
            (gotf - want).abs() < 1e-6,
            "gelu_f64({x}) = {gotf} vs {want}"
        );
    }

    // Loin à gauche : x·Φ(x) → 0 (Φ décroît plus vite que |x| ne croît).
    assert!(act::gelu(q16(-6.0)).to_f64().abs() < 1e-2);
    // Loin à droite : GELU(x) → x (Φ(x) → 1).
    assert!((act::gelu(q16(6.0)).to_f64() - 6.0).abs() < 1e-2);
}

#[test]
fn activation_apply_inplace_on_layer_output() {
    // Applique relu à la sortie d'un GEMM (couche linéaire quantifiée).
    let w = [1i32, -2, 3, -4, 5, -6].map(Q16_16::from); // 2×3
    let x = [1i32, 1, 1].map(Q16_16::from);
    let mut y = linalg::matvec(&w, &x, 2, 3); // [1-2+3, -4+5-6] = [2, -5]
    assert_eq!(y, [q16(2.0), q16(-5.0)]);
    act::apply_inplace(&mut y, act::relu);
    assert_eq!(y, [q16(2.0), Q16_16::zero()]); // relu écrase le négatif
}

// ------------------------------------------------------------------ //
//  Couche linéaire quantifiée                                         //
// ------------------------------------------------------------------ //

#[test]
fn linear_forward_known_small() {
    // W = [[1,2,3],[4,5,6]] (2×3), b = [10, -1], x = [1,0,-1].
    // W·x = [1-3, 4-6] = [-2,-2] ; +b = [8, -3].
    let w = [1i32, 2, 3, 4, 5, 6].map(Q16_16::from);
    let b = [10i32, -1].map(Q16_16::from);
    let x = [1i32, 0, -1].map(Q16_16::from);
    let layer = Linear::new(w.to_vec(), b.to_vec(), 2, 3);
    assert_eq!(layer.out_features(), 2);
    assert_eq!(layer.in_features(), 3);
    let y = layer.forward(&x);
    assert_eq!(y, [q16(8.0), q16(-3.0)]);
}

#[test]
fn linear_forward_activated_matches_forward_then_apply() {
    // forward_activated(f) doit coïncider exactement avec forward puis
    // apply_inplace(f) séparément (même calcul, juste composé).
    let mut rng = Lcg(0xFEED);
    let (out_f, in_f) = (5, 7);
    let w: Vec<Q16_16> = (0..out_f * in_f)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 6))
        .collect();
    let b: Vec<Q16_16> = (0..out_f)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 6))
        .collect();
    let x: Vec<Q16_16> = (0..in_f)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 6))
        .collect();
    let layer = Linear::new(w, b, out_f, in_f);

    let mut expected = layer.forward(&x);
    act::apply_inplace(&mut expected, act::relu);
    let got = layer.forward_activated(&x, act::relu);
    assert_eq!(got, expected);

    // Cohérence : W·x + b sans activation == forward().
    let plain = layer.forward(&x);
    assert_eq!(plain.len(), out_f);
}

#[test]
fn linear_i64_storage() {
    // Chemin de stockage i64 (Q32.32) : mêmes garanties.
    let w = [1i64, 2, 3, 4].map(Q32_32::from); // 2×2
    let b = [100i64, -100].map(Q32_32::from);
    let x = [1i64, 1].map(Q32_32::from);
    let layer = Linear::new(w.to_vec(), b.to_vec(), 2, 2);
    // W·x = [1+2, 3+4] = [3,7] ; +b = [103, -93].
    assert_eq!(
        layer.forward(&x),
        [Q32_32::from(103i64), Q32_32::from(-93i64)]
    );
}

#[test]
#[should_panic(expected = "Linear::new")]
fn linear_dim_mismatch_panics() {
    let w = vec![Q16_16::one(); 6]; // annoncé 2×3
    let b = vec![Q16_16::zero(); 3]; // devrait être 2
    let _ = Linear::new(w, b, 2, 3);
}

#[test]
fn linear_predict_class_known() {
    // 3 classes, 2 features. Choisi pour que la classe 1 gagne nettement.
    let w = [1i32, 0, 0, 1, -1, -1].map(Q16_16::from); // 3×2
    let b = [0i32, 5, 0].map(Q16_16::from);
    let x = [1i32, 1].map(Q16_16::from);
    let layer = Linear::new(w.to_vec(), b.to_vec(), 3, 2);
    // Logits : [1, 1+5, -1-1] = [1, 6, -2] → argmax = 1.
    assert_eq!(layer.predict_class(&x), Some(1));
}

#[test]
fn linear_predict_class_matches_argmax_of_forward() {
    // predict_class doit toujours coïncider avec argmax(forward(x)), y compris
    // sur le stockage i64.
    let mut rng = Lcg(0xACE1);
    let (out_f, in_f) = (6, 4);
    let w: Vec<Q32_32> = (0..out_f * in_f)
        .map(|_| Q32_32::from_raw(rng.next() as i64))
        .collect();
    let b: Vec<Q32_32> = (0..out_f)
        .map(|_| Q32_32::from_raw(rng.next() as i64))
        .collect();
    let x: Vec<Q32_32> = (0..in_f)
        .map(|_| Q32_32::from_raw(rng.next() as i64))
        .collect();
    let layer = Linear::new(w, b, out_f, in_f);
    let logits = layer.forward(&x);
    assert_eq!(layer.predict_class(&x), red::argmax(&logits));
}

#[test]
fn linear_predict_class_empty_output_is_none() {
    let layer: Linear<Q16_16> = Linear::new(vec![], vec![], 0, 3);
    assert_eq!(layer.predict_class(&[Q16_16::zero(); 3]), None);
}

#[test]
fn linear_predict_proba_argmax_matches_predict_class() {
    // Propriété centrale : argmax(softmax(z)) == argmax(z). predict_proba et
    // predict_class doivent donc toujours désigner la même classe.
    let mut rng = Lcg(0xB16B00B5);
    for _ in 0..200
    {
        let (out_f, in_f) = (5, 3);
        let w: Vec<Q16_16> = (0..out_f * in_f)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 10))
            .collect();
        let b: Vec<Q16_16> = (0..out_f)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 10))
            .collect();
        let x: Vec<Q16_16> = (0..in_f)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 10))
            .collect();
        let layer = Linear::new(w, b, out_f, in_f);

        let proba = layer.predict_proba(&x);
        // Les probabilités somment à ~1 (résolution Q16.16 près).
        let total: f64 = proba.iter().map(|p| p.to_f64()).sum();
        assert!((total - 1.0).abs() <= 1e-3, "Σproba = {total}");

        let proba_argmax = red::argmax(&proba);
        assert_eq!(proba_argmax, layer.predict_class(&x));
    }
}

// ------------------------------------------------------------------ //
//  Couche linéaire : inférence par lot                                //
// ------------------------------------------------------------------ //

/// Concatène `batch` appels indépendants de `f` (référence non batchée).
fn looped<T, R>(
    x: &[T],
    batch: usize,
    in_features: usize,
    mut f: impl FnMut(&[T]) -> Vec<R>,
) -> Vec<R> {
    let mut out = Vec::new();
    for row in x.chunks_exact(in_features).take(batch)
    {
        out.extend(f(row));
    }
    out
}

#[test]
fn linear_forward_batch_matches_looped_forward() {
    let mut rng = Lcg(0xBA7C_0001);
    let (out_f, in_f, batch) = (5, 7, 6);
    let w: Vec<Q16_16> = (0..out_f * in_f)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 6))
        .collect();
    let b: Vec<Q16_16> = (0..out_f)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 6))
        .collect();
    let x: Vec<Q16_16> = (0..batch * in_f)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 6))
        .collect();
    let layer = Linear::new(w, b, out_f, in_f);

    let got = layer.forward_batch(&x, batch);
    let want = looped(&x, batch, in_f, |row| layer.forward(row));
    assert_eq!(
        got, want,
        "forward_batch doit coïncider bit-à-bit avec la boucle"
    );
}

#[test]
fn linear_forward_batch_activated_matches_looped() {
    let mut rng = Lcg(0xBA7C_0002);
    let (out_f, in_f, batch) = (4, 5, 8);
    let w: Vec<Q16_16> = (0..out_f * in_f)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 6))
        .collect();
    let b: Vec<Q16_16> = (0..out_f)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 6))
        .collect();
    let x: Vec<Q16_16> = (0..batch * in_f)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 6))
        .collect();
    let layer = Linear::new(w, b, out_f, in_f);

    let got = layer.forward_batch_activated(&x, batch, act::relu);
    let want = looped(&x, batch, in_f, |row| {
        layer.forward_activated(row, act::relu)
    });
    assert_eq!(got, want);
}

#[test]
fn linear_predict_class_batch_matches_looped() {
    let mut rng = Lcg(0xBA7C_0003);
    let (out_f, in_f, batch) = (6, 4, 10);
    let w: Vec<Q32_32> = (0..out_f * in_f)
        .map(|_| Q32_32::from_raw(rng.next() as i64))
        .collect();
    let b: Vec<Q32_32> = (0..out_f)
        .map(|_| Q32_32::from_raw(rng.next() as i64))
        .collect();
    let x: Vec<Q32_32> = (0..batch * in_f)
        .map(|_| Q32_32::from_raw(rng.next() as i64))
        .collect();
    let layer = Linear::new(w, b, out_f, in_f);

    let got = layer.predict_class_batch(&x, batch);
    let want = looped(&x, batch, in_f, |row| vec![layer.predict_class(row)]);
    assert_eq!(got, want);
}

#[test]
fn linear_predict_proba_batch_matches_looped() {
    let mut rng = Lcg(0xBA7C_0004);
    let (out_f, in_f, batch) = (5, 3, 12);
    let w: Vec<Q16_16> = (0..out_f * in_f)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 10))
        .collect();
    let b: Vec<Q16_16> = (0..out_f)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 10))
        .collect();
    let x: Vec<Q16_16> = (0..batch * in_f)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 10))
        .collect();
    let layer = Linear::new(w, b, out_f, in_f);

    let got = layer.predict_proba_batch(&x, batch);
    let want = looped(&x, batch, in_f, |row| layer.predict_proba(row));
    assert_eq!(got, want);
}

#[test]
#[should_panic(expected = "Linear::forward_batch")]
fn linear_forward_batch_dim_mismatch_panics() {
    let layer: Linear<Q16_16> = Linear::new(vec![Q16_16::one(); 6], vec![Q16_16::zero(); 2], 2, 3);
    let x = vec![Q16_16::zero(); 7]; // ni 1×3 ni 2×3 ni aucun multiple de 3
    let _ = layer.forward_batch(&x, 2);
}

// ------------------------------------------------------------------ //
//  Requantification (rescale)                                         //
// ------------------------------------------------------------------ //

#[test]
fn rescale_finer_resolution_exact() {
    // Q16.16 → Q24.8 : FRAC diminue (16 → 8), résolution plus grossière.
    // Attends un arrondi vers zéro par défaut (TowardZero/Wrap).
    let x = q16(3.5); // raw = 3*2^16 + 2^15
    let y: Q24_8 = rescale_wrapping(x);
    assert_eq!(y.to_f64(), 3.5);

    // Q24.8 → Q16.16 : FRAC augmente (8 → 16), résolution plus fine, exact
    // (décalage à gauche, aucune perte).
    let a = Q24_8::try_from(7.25).unwrap();
    let b: Q16_16 = rescale_wrapping(a);
    assert_eq!(b.to_f64(), 7.25);
}

#[test]
fn rescale_round_trip_within_resolution() {
    // Aller-retour Q16.16 → Q8.24 → Q16.16 (les deux stockages i32) doit
    // rester à moins d'une résolution Q16.16 (8.24 est plus fin, sans perte
    // à l'aller ; le retour perd à la résolution Q16.16 uniquement).
    let mut rng = Lcg(0xC0DE);
    for _ in 0..500
    {
        let x = Q16_16::from_raw(rng.raw_i32() >> 8); // évite le débordement Q8.24
        let up: Q8_24 = rescale_wrapping(x);
        let back: Q16_16 = rescale_wrapping(up);
        assert_eq!(back, x, "aller-retour 16.16→8.24→16.16 doit être exact");
    }
}

#[test]
fn rescale_reduce_rounding_modes() {
    // Q16.16 raw = 3 (donc 3/2^16), rescale vers FRAC=0 (entier) : le reste
    // est < 1/2 → TowardZero et NearestEven arrondissent tous deux vers 0.
    let tiny = Q16_16::from_raw(3);
    let z: FixedI32<0> = rescale(tiny, RoundingMode::TowardZero, OverflowMode::Wrap).unwrap();
    assert_eq!(z.to_raw(), 0);

    // raw = 1<<15 exactement (moitié) : NearestEven arrondit au pair (0),
    // Ceil arrondit vers +∞ (1).
    let half = Q16_16::from_raw(1 << 15);
    let ne: FixedI32<0> = rescale(half, RoundingMode::NearestEven, OverflowMode::Wrap).unwrap();
    assert_eq!(ne.to_raw(), 0);
    let ceil: FixedI32<0> = rescale(half, RoundingMode::Ceil, OverflowMode::Wrap).unwrap();
    assert_eq!(ceil.to_raw(), 1);
}

#[test]
fn rescale_overflow_policies_on_coarser_integer_range() {
    // Q8_8 (±128) → Q1_15 (±1) : FRAC augmente (8→15), la plage entière
    // rétrécit énormément — une valeur hors [-1,1) déborde.
    let big = Q8_8::from(5); // hors plage de Q1.15
    assert!(rescale::<i16, 8, 15>(big, RoundingMode::TowardZero, OverflowMode::Checked).is_none());
    let sat: Q1_15 = rescale_saturating(big);
    assert_eq!(sat, Q1_15::max_value());

    let neg_big = Q8_8::from(-5);
    let sat_neg: Q1_15 = rescale_saturating(neg_big);
    assert_eq!(sat_neg, Q1_15::min_value());

    // Une valeur dans la plage cible passe sans encombre.
    let small = Q8_8::try_from(0.25).unwrap();
    let ok: Q1_15 = rescale_saturating(small);
    assert!((ok.to_f64() - 0.25).abs() <= Q1_15::resolution().to_f64());
}

#[test]
fn rescale_identity_same_frac() {
    // TO == FROM : identité (les deux branches du décalage coïncident à 0).
    let x = q16(-12.375);
    let y: Q16_16 = rescale_wrapping(x);
    assert_eq!(y, x);
}

#[test]
fn rescale_i64_storage() {
    // Chemin de stockage i64 (Q32.32 ↔ FixedI64<16>) : mêmes garanties.
    let x = Q32_32::try_from(100.5).unwrap();
    let y: FixedI64<16> = rescale_wrapping(x);
    assert_eq!(y.to_f64(), 100.5);
    let back: Q32_32 = rescale_wrapping(y);
    assert_eq!(back, x);
}

#[test]
fn rescale_feeds_linear_layer() {
    // Usage réaliste : un accumulateur haute résolution (Q8.24) requantifié en
    // Q16.16 avant d'alimenter la couche linéaire suivante (même stockage i32,
    // requis par `FixedReducible`/`Linear`, mais résolution différente).
    let wide = [
        Q8_24::try_from(1.0).unwrap(),
        Q8_24::try_from(-2.0).unwrap(),
        Q8_24::try_from(0.5).unwrap(),
    ];
    let narrow: Vec<Q16_16> = wide.iter().map(|&v| rescale_wrapping(v)).collect();
    let w = [1i32, 1, 1].map(Q16_16::from);
    let b = [0i32].map(Q16_16::from);
    let layer = Linear::new(w.to_vec(), b.to_vec(), 1, 3);
    let y = layer.forward(&narrow);
    assert_eq!(y[0].to_f64(), 1.0 - 2.0 + 0.5);
}

// ------------------------------------------------------------------ //
//  Convolution 1D (im2col + GEMM)                                     //
// ------------------------------------------------------------------ //

/// Référence naïve : triple boucle directe (canal de sortie, position,
/// canal d'entrée × noyau), opérateurs enveloppants de `Fixed`. Doit coïncider
/// **bit-à-bit** avec `conv1d` (même produits arrondis, même somme exacte).
fn naive_conv1d<const F: u32>(
    x: &[FixedI32<F>],
    weights: &[FixedI32<F>],
    bias: &[FixedI32<F>],
    shape: Conv1dShape,
) -> Vec<FixedI32<F>> {
    let length_out = shape.length_out();
    let mut y = vec![FixedI32::<F>::from_raw(0); shape.out_channels * length_out];
    for co in 0..shape.out_channels
    {
        for j in 0..length_out
        {
            let mut acc = bias[co];
            for ci in 0..shape.in_channels
            {
                for k in 0..shape.kernel_size
                {
                    let w = weights
                        [co * (shape.in_channels * shape.kernel_size) + ci * shape.kernel_size + k];
                    let xv = x[ci * shape.length + j * shape.stride + k];
                    acc += w * xv;
                }
            }
            y[co * length_out + j] = acc;
        }
    }
    y
}

#[test]
fn conv1d_known_small() {
    // 1 canal d'entrée, 1 canal de sortie, noyau [1, -1], stride 1 : chaque
    // sortie est x[j]·1 + x[j+1]·(−1) = x[j] − x[j+1] (corrélation croisée,
    // sans retournement du noyau — convention standard des CNN).
    // x = [1,3,6,10] → y = [1-3, 3-6, 6-10] = [-2,-3,-4] (+ biais 0).
    let x = [1i32, 3, 6, 10].map(Q16_16::from);
    let w = [1i32, -1].map(Q16_16::from);
    let b = [Q16_16::zero()];
    let shape = Conv1dShape {
        in_channels: 1,
        length: 4,
        out_channels: 1,
        kernel_size: 2,
        stride: 1,
    };
    let y = conv1d(&x, &w, &b, shape);
    assert_eq!(y, [-2i32, -3, -4].map(Q16_16::from));
}

#[test]
fn conv1d_multi_channel_matches_naive_bit_exact() {
    // 2 canaux d'entrée, 3 canaux de sortie, noyau de taille 3, stride 2,
    // tailles variées : égalité stricte avec la triple boucle naïve.
    let mut rng = Lcg(0xD00D);
    for &(in_channels, length, out_channels, kernel_size, stride) in &[
        (2, 10, 3, 3, 2),
        (1, 5, 1, 5, 1),
        (4, 20, 2, 4, 3),
        (3, 7, 5, 2, 1),
    ]
    {
        let shape = Conv1dShape {
            in_channels,
            length,
            out_channels,
            kernel_size,
            stride,
        };
        let length_out = shape.length_out();
        let x: Vec<Q16_16> = (0..in_channels * length)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 6))
            .collect();
        let w: Vec<Q16_16> = (0..out_channels * in_channels * kernel_size)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 6))
            .collect();
        let b: Vec<Q16_16> = (0..out_channels)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 6))
            .collect();

        let got = conv1d(&x, &w, &b, shape);
        let want = naive_conv1d(&x, &w, &b, shape);
        assert_eq!(got.len(), out_channels * length_out);
        assert_eq!(got, want, "conv1d shape={shape:?}");
    }
}

#[test]
fn conv1d_stride_one_is_full_overlap() {
    // Stride 1, noyau de taille 1 : identité canal à canal pondérée par un
    // scalaire (pas de recouvrement de fenêtre, length_out == length).
    let x = [2i32, 4, 6].map(Q16_16::from); // 1×3
    let w = [q16(0.5)]; // 1×1×1
    let b = [Q16_16::zero()];
    let shape = Conv1dShape {
        in_channels: 1,
        length: 3,
        out_channels: 1,
        kernel_size: 1,
        stride: 1,
    };
    let y = conv1d(&x, &w, &b, shape);
    assert_eq!(y, [q16(1.0), q16(2.0), q16(3.0)]);
}

#[test]
#[should_panic(expected = "Conv1dShape::length_out")]
fn conv1d_kernel_larger_than_length_panics() {
    let x = [Q16_16::one(); 3];
    let w = [Q16_16::one(); 5];
    let b = [Q16_16::zero()];
    let shape = Conv1dShape {
        in_channels: 1,
        length: 3,
        out_channels: 1,
        kernel_size: 5,
        stride: 1,
    };
    let _ = conv1d(&x, &w, &b, shape);
}

#[test]
#[should_panic(expected = "conv1d")]
fn conv1d_dim_mismatch_panics() {
    let x = [Q16_16::one(); 4]; // annoncé 1×4
    let w = [Q16_16::one(); 3]; // annoncé 1×1×2 → devrait être longueur 2
    let b = [Q16_16::zero()];
    let shape = Conv1dShape {
        in_channels: 1,
        length: 4,
        out_channels: 1,
        kernel_size: 2,
        stride: 1,
    };
    let _ = conv1d(&x, &w, &b, shape);
}

#[test]
fn conv1d_i64_storage() {
    // Chemin de stockage i64 (Q32.32) : mêmes garanties.
    let x = [1i64, 2, 3, 4].map(Q32_32::from); // 1×4
    let w = [1i64, 1].map(Q32_32::from); // noyau [1,1], somme glissante
    let b = [Q32_32::from(10i64)];
    let shape = Conv1dShape {
        in_channels: 1,
        length: 4,
        out_channels: 1,
        kernel_size: 2,
        stride: 1,
    };
    let y = conv1d(&x, &w, &b, shape);
    // Sommes glissantes [1+2, 2+3, 3+4] = [3,5,7], + biais 10 → [13,15,17].
    assert_eq!(y, [13i64, 15, 17].map(Q32_32::from));
}

#[test]
fn conv1d_batch_matches_looped_conv1d() {
    let mut rng = Lcg(0xBA7C_1001);
    for &(batch, in_channels, length, out_channels, kernel_size, stride) in &[
        (4, 2, 10, 3, 3, 2),
        (1, 1, 5, 1, 5, 1),
        (6, 4, 20, 2, 4, 3),
        (3, 3, 7, 5, 2, 1),
    ]
    {
        let shape = Conv1dShape {
            in_channels,
            length,
            out_channels,
            kernel_size,
            stride,
        };
        let length_out = shape.length_out();
        let x: Vec<Q16_16> = (0..batch * in_channels * length)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 6))
            .collect();
        let w: Vec<Q16_16> = (0..out_channels * in_channels * kernel_size)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 6))
            .collect();
        let b: Vec<Q16_16> = (0..out_channels)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 6))
            .collect();

        let got = conv1d_batch(&x, batch, &w, &b, shape);
        let mut want = Vec::with_capacity(batch * out_channels * length_out);
        for sample in x.chunks_exact(in_channels * length)
        {
            want.extend(conv1d(sample, &w, &b, shape));
        }
        assert_eq!(got, want, "batch={batch} shape={shape:?}");

        // La sortie, repliée (batch·out_channels) × length_out, alimente
        // max_pool1d/avg_pool1d **sans code supplémentaire** : vérifie que le
        // résultat coïncide avec un pooling par échantillon (cf. doc de tête
        // de module).
        if length_out >= 2
        {
            let pool_shape = Pool1dShape {
                channels: batch * out_channels,
                length: length_out,
                window: 2,
                stride: 2,
            };
            let pooled_batch = max_pool1d(&got, pool_shape);
            let mut pooled_want = Vec::new();
            for sample_out in want.chunks_exact(out_channels * length_out)
            {
                pooled_want.extend(max_pool1d(
                    sample_out,
                    Pool1dShape {
                        channels: out_channels,
                        length: length_out,
                        window: 2,
                        stride: 2,
                    },
                ));
            }
            assert_eq!(pooled_batch, pooled_want, "pooling replié, batch={batch}");
        }
    }
}

#[test]
#[should_panic(expected = "conv1d_batch")]
fn conv1d_batch_dim_mismatch_panics() {
    let x = vec![Q16_16::one(); 7]; // ni 2×1×3 (=6) ni aucun multiple valide
    let w = [Q16_16::one(); 3]; // 1×1×3
    let b = [Q16_16::zero()];
    let shape = Conv1dShape {
        in_channels: 1,
        length: 3,
        out_channels: 1,
        kernel_size: 3,
        stride: 1,
    };
    let _ = conv1d_batch(&x, 2, &w, &b, shape);
}

// ------------------------------------------------------------------ //
//  Pooling 1D                                                         //
// ------------------------------------------------------------------ //

#[test]
fn max_pool1d_known_small() {
    // 1 canal, x = [1,5,3,2,8,4], fenêtre 2, stride 2 (non chevauchant) :
    // y = [max(1,5), max(3,2), max(8,4)] = [5,3,8].
    let x = [1i32, 5, 3, 2, 8, 4].map(Q16_16::from);
    let shape = Pool1dShape {
        channels: 1,
        length: 6,
        window: 2,
        stride: 2,
    };
    let y = max_pool1d(&x, shape);
    assert_eq!(y, [5i32, 3, 8].map(Q16_16::from));
}

#[test]
fn avg_pool1d_known_small() {
    // 1 canal, x = [1,3,5,7], fenêtre 2, stride 2 : y = [(1+3)/2, (5+7)/2] = [2,6].
    let x = [1i32, 3, 5, 7].map(Q16_16::from);
    let shape = Pool1dShape {
        channels: 1,
        length: 4,
        window: 2,
        stride: 2,
    };
    let y = avg_pool1d(&x, shape);
    assert_eq!(y, [2i32, 6].map(Q16_16::from));
}

#[test]
fn pool1d_multi_channel_overlapping_window() {
    // 2 canaux, fenêtre chevauchante (stride < window). Sommes de fenêtre
    // choisies multiples de `window` pour une moyenne entière exacte (la
    // division virgule fixe calcule le quotient réel, pas une division
    // entière : (1+4+2)/3 = 2.333…, pas 2 — ces valeurs évitent toute
    // ambiguïté d'arrondi dans un test « connu »).
    let x = [3i32, 6, 9, 12, /* canal 2 */ 30, 60, 90, 120].map(Q16_16::from);
    let shape = Pool1dShape {
        channels: 2,
        length: 4,
        window: 3,
        stride: 1,
    };
    // length_out = (4-3)/1+1 = 2.
    let mx = max_pool1d(&x, shape);
    assert_eq!(mx, [9i32, 12, /* canal 2 */ 90, 120].map(Q16_16::from));
    let avg = avg_pool1d(&x, shape);
    // canal 1 : (3+6+9)/3=6, (6+9+12)/3=9 ; canal 2 : (30+60+90)/3=60, (60+90+120)/3=90.
    assert_eq!(avg, [6i32, 9, 60, 90].map(Q16_16::from));
}

#[test]
fn max_pool1d_matches_reductions_max_reference() {
    let mut rng = Lcg(0xF00D);
    for &(channels, length, window, stride) in
        &[(3, 10, 3, 2), (1, 8, 8, 1), (4, 15, 4, 3), (2, 7, 2, 1)]
    {
        let shape = Pool1dShape {
            channels,
            length,
            window,
            stride,
        };
        let length_out = shape.length_out();
        let x: Vec<Q16_16> = (0..channels * length)
            .map(|_| Q16_16::from_raw(rng.raw_i32()))
            .collect();
        let got = max_pool1d(&x, shape);
        for c in 0..channels
        {
            for j in 0..length_out
            {
                let start = c * length + j * stride;
                let want = red::max(&x[start..start + window]).unwrap();
                assert_eq!(got[c * length_out + j], want, "max_pool c={c} j={j}");
            }
        }
    }
}

#[test]
fn avg_pool1d_matches_sum_div_window_reference() {
    let mut rng = Lcg(0xFACE);
    for &(channels, length, window, stride) in
        &[(3, 10, 3, 2), (1, 8, 8, 1), (4, 15, 4, 3), (2, 7, 2, 1)]
    {
        let shape = Pool1dShape {
            channels,
            length,
            window,
            stride,
        };
        let length_out = shape.length_out();
        let x: Vec<Q16_16> = (0..channels * length)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 4))
            .collect();
        let got = avg_pool1d(&x, shape);
        let divisor = Q16_16::from(window as i32);
        for c in 0..channels
        {
            for j in 0..length_out
            {
                let start = c * length + j * stride;
                let want = red::sum(&x[start..start + window])
                    .checked_div(divisor)
                    .unwrap();
                assert_eq!(got[c * length_out + j], want, "avg_pool c={c} j={j}");
            }
        }
    }
}

#[test]
fn pool1d_stride_equals_window_is_disjoint_tiling() {
    // stride == window : les fenêtres ne se chevauchent pas (pavage exact).
    let x = [1i32, 2, 3, 4, 5, 6, 7, 8].map(Q16_16::from);
    let shape = Pool1dShape {
        channels: 1,
        length: 8,
        window: 4,
        stride: 4,
    };
    assert_eq!(shape.length_out(), 2);
    let y = max_pool1d(&x, shape);
    assert_eq!(y, [4i32, 8].map(Q16_16::from));
}

#[test]
#[should_panic(expected = "Pool1dShape::length_out")]
fn pool1d_window_larger_than_length_panics() {
    let x = [Q16_16::one(); 3];
    let shape = Pool1dShape {
        channels: 1,
        length: 3,
        window: 5,
        stride: 1,
    };
    let _ = max_pool1d(&x, shape);
}

#[test]
#[should_panic(expected = "max_pool1d")]
fn pool1d_dim_mismatch_panics() {
    let x = [Q16_16::one(); 5]; // annoncé 1×5
    let shape = Pool1dShape {
        channels: 1,
        length: 6,
        window: 2,
        stride: 1,
    };
    let _ = max_pool1d(&x, shape);
}

#[test]
fn pool1d_i64_storage() {
    // Chemin de stockage i64 (Q32.32) : mêmes garanties.
    let x = [1i64, 9, 3, 7].map(Q32_32::from);
    let shape = Pool1dShape {
        channels: 1,
        length: 4,
        window: 2,
        stride: 2,
    };
    assert_eq!(max_pool1d(&x, shape), [9i64, 7].map(Q32_32::from));
    assert_eq!(avg_pool1d(&x, shape), [5i64, 5].map(Q32_32::from));
}

#[test]
fn pool1d_feeds_conv1d_chain() {
    // Usage réaliste : conv1d → max_pool1d, comme dans un CNN léger.
    let x = [1i32, 2, 3, 4, 5, 6].map(Q16_16::from); // 1 canal, longueur 6
    let w = [1i32, -1].map(Q16_16::from); // différence discrète
    let b = [Q16_16::zero()];
    let conv_shape = Conv1dShape {
        in_channels: 1,
        length: 6,
        out_channels: 1,
        kernel_size: 2,
        stride: 1,
    };
    let conv_out = conv1d(&x, &w, &b, conv_shape); // x[j]-x[j+1] = [-1,-1,-1,-1,-1], longueur 5
    let pool_shape = Pool1dShape {
        channels: 1,
        length: conv_out.len(),
        window: 2,
        stride: 2,
    };
    let pooled = max_pool1d(&conv_out, pool_shape);
    assert_eq!(pooled, [q16(-1.0), q16(-1.0)]);
}

// ------------------------------------------------------------------ //
//  Convolution 2D (im2col + GEMM)                                     //
// ------------------------------------------------------------------ //

/// Référence naïve : quintuple boucle directe (canal de sortie, position `oh`,
/// `ow`, canal d'entrée × noyau), opérateurs enveloppants de `Fixed`. Doit
/// coïncider **bit-à-bit** avec `conv2d`.
fn naive_conv2d<const F: u32>(
    x: &[FixedI32<F>],
    weights: &[FixedI32<F>],
    bias: &[FixedI32<F>],
    shape: Conv2dShape,
) -> Vec<FixedI32<F>> {
    let height_out = shape.height_out();
    let width_out = shape.width_out();
    let mut y = vec![FixedI32::<F>::from_raw(0); shape.out_channels * height_out * width_out];
    for co in 0..shape.out_channels
    {
        for oh in 0..height_out
        {
            for ow in 0..width_out
            {
                let mut acc = bias[co];
                for ci in 0..shape.in_channels
                {
                    for kh in 0..shape.kernel_h
                    {
                        for kw in 0..shape.kernel_w
                        {
                            let w_idx = co * (shape.in_channels * shape.kernel_h * shape.kernel_w)
                                + ci * (shape.kernel_h * shape.kernel_w)
                                + kh * shape.kernel_w
                                + kw;
                            let h = oh * shape.stride_h + kh;
                            let w = ow * shape.stride_w + kw;
                            let x_idx = ci * (shape.height * shape.width) + h * shape.width + w;
                            acc += weights[w_idx] * x[x_idx];
                        }
                    }
                }
                y[co * (height_out * width_out) + oh * width_out + ow] = acc;
            }
        }
    }
    y
}

#[test]
fn conv2d_known_small() {
    // 1 canal, x = arithmétique 1..9 (3×3), noyau 2×2 [[1,0],[0,-1]], stride
    // (1,1) : y[oh,ow] = x[oh,ow] - x[oh+1,ow+1] = -4 partout (différences
    // constantes de la suite arithmétique), + biais 10 → 6 partout.
    let x = [1i32, 2, 3, 4, 5, 6, 7, 8, 9].map(Q16_16::from);
    let w = [1i32, 0, 0, -1].map(Q16_16::from);
    let b = [Q16_16::from(10)];
    let shape = Conv2dShape {
        in_channels: 1,
        height: 3,
        width: 3,
        out_channels: 1,
        kernel_h: 2,
        kernel_w: 2,
        stride_h: 1,
        stride_w: 1,
    };
    assert_eq!(shape.height_out(), 2);
    assert_eq!(shape.width_out(), 2);
    let y = conv2d(&x, &w, &b, shape);
    assert_eq!(y, [6i32, 6, 6, 6].map(Q16_16::from));
}

#[test]
fn conv2d_pointwise_1x1_kernel() {
    // Noyau 1×1 : combinaison linéaire canal à canal, point par point (aucun
    // recouvrement spatial, height_out == height, width_out == width).
    let x = [1i32, 2, 3, 4, /* canal 2 */ 10, 20, 30, 40].map(Q16_16::from);
    let w = [q16(2.0), q16(0.5)]; // (co=0,ci=0)=2, (co=0,ci=1)=0.5
    let b = [Q16_16::zero()];
    let shape = Conv2dShape {
        in_channels: 2,
        height: 2,
        width: 2,
        out_channels: 1,
        kernel_h: 1,
        kernel_w: 1,
        stride_h: 1,
        stride_w: 1,
    };
    let y = conv2d(&x, &w, &b, shape);
    // y[h,w] = 2·x0[h,w] + 0.5·x1[h,w].
    assert_eq!(y, [q16(7.0), q16(14.0), q16(21.0), q16(28.0)]);
}

#[test]
fn conv2d_multi_channel_matches_naive_bit_exact() {
    let mut rng = Lcg(0xC0DE2D);
    let cases = [
        (2, 8, 8, 3, 3, 3, 2, 2),
        (1, 5, 6, 1, 5, 5, 1, 1),
        (3, 10, 7, 2, 3, 2, 2, 1),
        (1, 4, 4, 1, 2, 2, 1, 1),
    ];
    for &(in_channels, height, width, out_channels, kernel_h, kernel_w, stride_h, stride_w) in
        &cases
    {
        let shape = Conv2dShape {
            in_channels,
            height,
            width,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
        };
        let height_out = shape.height_out();
        let width_out = shape.width_out();
        let x: Vec<Q16_16> = (0..in_channels * height * width)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 6))
            .collect();
        let w: Vec<Q16_16> = (0..out_channels * in_channels * kernel_h * kernel_w)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 6))
            .collect();
        let b: Vec<Q16_16> = (0..out_channels)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 6))
            .collect();

        let got = conv2d(&x, &w, &b, shape);
        let want = naive_conv2d(&x, &w, &b, shape);
        assert_eq!(got.len(), out_channels * height_out * width_out);
        assert_eq!(got, want, "conv2d shape={shape:?}");
    }
}

#[test]
#[should_panic(expected = "Conv2dShape::height_out")]
fn conv2d_kernel_taller_than_input_panics() {
    let x = [Q16_16::one(); 9]; // 1×3×3
    let w = [Q16_16::one(); 4]; // 1×1×4×1 → hauteur de noyau 4 > 3
    let b = [Q16_16::zero()];
    let shape = Conv2dShape {
        in_channels: 1,
        height: 3,
        width: 3,
        out_channels: 1,
        kernel_h: 4,
        kernel_w: 1,
        stride_h: 1,
        stride_w: 1,
    };
    let _ = conv2d(&x, &w, &b, shape);
}

#[test]
#[should_panic(expected = "conv2d")]
fn conv2d_dim_mismatch_panics() {
    let x = [Q16_16::one(); 9]; // annoncé 1×3×3
    let w = [Q16_16::one(); 3]; // annoncé 1×1×2×2 → devrait être longueur 4
    let b = [Q16_16::zero()];
    let shape = Conv2dShape {
        in_channels: 1,
        height: 3,
        width: 3,
        out_channels: 1,
        kernel_h: 2,
        kernel_w: 2,
        stride_h: 1,
        stride_w: 1,
    };
    let _ = conv2d(&x, &w, &b, shape);
}

#[test]
fn conv2d_i64_storage() {
    // Chemin de stockage i64 (Q32.32) : mêmes garanties. 1 canal, 2×2, noyau
    // 2×2 identité en (0,0) uniquement : y = x[0,0] partout où défini (une
    // seule position de sortie ici, height_out=width_out=1).
    let x = [1i64, 2, 3, 4].map(Q32_32::from); // 2×2
    let w = [1i64, 0, 0, 0].map(Q32_32::from); // ne garde que x[0,0]
    let b = [Q32_32::from(100i64)];
    let shape = Conv2dShape {
        in_channels: 1,
        height: 2,
        width: 2,
        out_channels: 1,
        kernel_h: 2,
        kernel_w: 2,
        stride_h: 1,
        stride_w: 1,
    };
    let y = conv2d(&x, &w, &b, shape);
    assert_eq!(y, [Q32_32::from(101i64)]); // x[0,0]=1 + biais 100
}

#[test]
fn conv2d_batch_matches_looped_conv2d() {
    let mut rng = Lcg(0xBA7C_2001);
    let cases = [
        (4, 2, 8, 8, 3, 3, 3, 2, 2),
        (1, 1, 5, 6, 1, 5, 5, 1, 1),
        (5, 3, 10, 7, 2, 3, 2, 2, 1),
    ];
    for &(
        batch,
        in_channels,
        height,
        width,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
    ) in &cases
    {
        let shape = Conv2dShape {
            in_channels,
            height,
            width,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
        };
        let height_out = shape.height_out();
        let width_out = shape.width_out();
        let x: Vec<Q16_16> = (0..batch * in_channels * height * width)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 6))
            .collect();
        let w: Vec<Q16_16> = (0..out_channels * in_channels * kernel_h * kernel_w)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 6))
            .collect();
        let b: Vec<Q16_16> = (0..out_channels)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 6))
            .collect();

        let got = conv2d_batch(&x, batch, &w, &b, shape);
        let mut want = Vec::with_capacity(batch * out_channels * height_out * width_out);
        for sample in x.chunks_exact(in_channels * height * width)
        {
            want.extend(conv2d(sample, &w, &b, shape));
        }
        assert_eq!(got, want, "batch={batch} shape={shape:?}");

        // Repliée (batch·out_channels) × height_out × width_out, la sortie
        // alimente max_pool2d sans code supplémentaire (cf. doc de tête de
        // module) : coïncide avec un pooling par échantillon.
        if height_out >= 2 && width_out >= 2
        {
            let pool_shape = Pool2dShape {
                channels: batch * out_channels,
                height: height_out,
                width: width_out,
                window_h: 2,
                window_w: 2,
                stride_h: 2,
                stride_w: 2,
            };
            let pooled_batch = max_pool2d(&got, pool_shape);
            let mut pooled_want = Vec::new();
            for sample_out in want.chunks_exact(out_channels * height_out * width_out)
            {
                pooled_want.extend(max_pool2d(
                    sample_out,
                    Pool2dShape {
                        channels: out_channels,
                        height: height_out,
                        width: width_out,
                        window_h: 2,
                        window_w: 2,
                        stride_h: 2,
                        stride_w: 2,
                    },
                ));
            }
            assert_eq!(pooled_batch, pooled_want, "pooling replié, batch={batch}");
        }
    }
}

#[test]
#[should_panic(expected = "conv2d_batch")]
fn conv2d_batch_dim_mismatch_panics() {
    let x = vec![Q16_16::one(); 7]; // ni 2×1×2×2 (=8) ni aucun multiple valide
    let w = [Q16_16::one(); 4]; // 1×1×2×2
    let b = [Q16_16::zero()];
    let shape = Conv2dShape {
        in_channels: 1,
        height: 2,
        width: 2,
        out_channels: 1,
        kernel_h: 2,
        kernel_w: 2,
        stride_h: 1,
        stride_w: 1,
    };
    let _ = conv2d_batch(&x, 2, &w, &b, shape);
}

// ------------------------------------------------------------------ //
//  Convolution séparable en profondeur (depthwise, MobileNet)          //
// ------------------------------------------------------------------ //

#[test]
fn depthwise_conv2d_known_small() {
    // 2 canaux 3×3, noyaux 2×2 distincts par canal, stride (1,1).
    // Canal 0 : x = 1..9, noyau [[1,0],[0,-1]] → y = x[oh,ow]-x[oh+1,ow+1] = -4 partout.
    // Canal 1 : x = 10× (1..9), noyau [[1,0],[0,1]] → y = x[oh,ow]+x[oh+1,ow+1].
    let x0 = [1i32, 2, 3, 4, 5, 6, 7, 8, 9];
    let x1: Vec<i32> = x0.iter().map(|v| v * 10).collect();
    let x: Vec<Q16_16> = x0
        .iter()
        .chain(x1.iter())
        .copied()
        .map(Q16_16::from)
        .collect();
    let w = [1i32, 0, 0, -1, /* canal 1 */ 1, 0, 0, 1].map(Q16_16::from);
    let b = [Q16_16::zero(), Q16_16::zero()];
    let shape = Conv2dShape {
        in_channels: 2,
        height: 3,
        width: 3,
        out_channels: 2,
        kernel_h: 2,
        kernel_w: 2,
        stride_h: 1,
        stride_w: 1,
    };
    let y = depthwise_conv2d(&x, &w, &b, shape);
    // Canal 0 : -4 partout (2×2 sorties). Canal 1 : x[oh,ow]+x[oh+1,ow+1] pour
    // x1 = 10,20,30,40,50,60,70,80,90 → (10+50,20+60,40+80,50+90) = (60,80,120,140).
    assert_eq!(y, [-4i32, -4, -4, -4, 60, 80, 120, 140].map(Q16_16::from));
}

#[test]
fn depthwise_conv2d_matches_block_diagonal_full_conv() {
    // Une convolution profonde équivaut exactement à une convolution dense
    // (`conv2d`) dont le tenseur de poids est bloc-diagonal :
    // full[co,ci,kh,kw] = depthwise[ci,kh,kw] si co==ci, sinon 0 — vérifie
    // depthwise_conv2d contre conv2d (déjà testée), sans référence indépendante.
    let mut rng = Lcg(0xDEDE_2001);
    let cases = [
        (2usize, 8usize, 8usize, 3usize, 3usize, 2usize, 2usize),
        (3, 10, 7, 3, 2, 2, 1),
        (1, 5, 6, 5, 5, 1, 1),
    ];
    for &(channels, height, width, kernel_h, kernel_w, stride_h, stride_w) in &cases
    {
        let shape = Conv2dShape {
            in_channels: channels,
            height,
            width,
            out_channels: channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
        };
        let x: Vec<Q16_16> = (0..channels * height * width)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 6))
            .collect();
        let depthwise_w: Vec<Q16_16> = (0..channels * kernel_h * kernel_w)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 6))
            .collect();
        let b: Vec<Q16_16> = (0..channels)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 6))
            .collect();

        let mut full_w = vec![Q16_16::zero(); channels * channels * kernel_h * kernel_w];
        let kernel_size = kernel_h * kernel_w;
        for ci in 0..channels
        {
            full_w[(ci * channels + ci) * kernel_size..(ci * channels + ci + 1) * kernel_size]
                .copy_from_slice(&depthwise_w[ci * kernel_size..(ci + 1) * kernel_size]);
        }

        let got = depthwise_conv2d(&x, &depthwise_w, &b, shape);
        let want = conv2d(&x, &full_w, &b, shape);
        assert_eq!(got, want, "channels={channels} shape={shape:?}");
    }
}

#[test]
#[should_panic(expected = "depthwise_conv2d")]
fn depthwise_conv2d_requires_out_eq_in_channels() {
    let x = vec![Q16_16::zero(); 2 * 3 * 3];
    let w = vec![Q16_16::zero(); 2 * 2 * 2];
    let b = vec![Q16_16::zero(); 2];
    let shape = Conv2dShape {
        in_channels: 2,
        height: 3,
        width: 3,
        out_channels: 3, // devrait valoir in_channels = 2
        kernel_h: 2,
        kernel_w: 2,
        stride_h: 1,
        stride_w: 1,
    };
    let _ = depthwise_conv2d(&x, &w, &b, shape);
}

#[test]
#[should_panic(expected = "depthwise_conv2d")]
fn depthwise_conv2d_weights_dim_mismatch_panics() {
    let x = vec![Q16_16::zero(); 2 * 3 * 3];
    let w = vec![Q16_16::zero(); 5]; // devrait être 2×2×2 = 8
    let b = vec![Q16_16::zero(); 2];
    let shape = Conv2dShape {
        in_channels: 2,
        height: 3,
        width: 3,
        out_channels: 2,
        kernel_h: 2,
        kernel_w: 2,
        stride_h: 1,
        stride_w: 1,
    };
    let _ = depthwise_conv2d(&x, &w, &b, shape);
}

#[test]
fn separable_conv2d_matches_depthwise_then_pointwise() {
    let mut rng = Lcg(0xDEDE_2002);
    let (in_channels, out_channels, height, width, kernel_h, kernel_w) =
        (3usize, 5usize, 6usize, 6usize, 3usize, 3usize);
    let shape = Conv2dShape {
        in_channels,
        height,
        width,
        out_channels: in_channels,
        kernel_h,
        kernel_w,
        stride_h: 1,
        stride_w: 1,
    };
    let x: Vec<Q16_16> = (0..in_channels * height * width)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 6))
        .collect();
    let dw: Vec<Q16_16> = (0..in_channels * kernel_h * kernel_w)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 6))
        .collect();
    let db: Vec<Q16_16> = (0..in_channels)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 6))
        .collect();
    let pw: Vec<Q16_16> = (0..out_channels * in_channels)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 6))
        .collect();
    let pb: Vec<Q16_16> = (0..out_channels)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 6))
        .collect();

    let got = separable_conv2d(&x, &dw, &db, &pw, &pb, shape, out_channels);

    let depthwise_out = depthwise_conv2d(&x, &dw, &db, shape);
    let pointwise_shape = Conv2dShape {
        in_channels,
        height: shape.height_out(),
        width: shape.width_out(),
        out_channels,
        kernel_h: 1,
        kernel_w: 1,
        stride_h: 1,
        stride_w: 1,
    };
    let want = conv2d(&depthwise_out, &pw, &pb, pointwise_shape);
    assert_eq!(got, want);
}

// ------------------------------------------------------------------ //
//  Convolution transposée (déconvolution/suréchantillonnage)          //
// ------------------------------------------------------------------ //

#[test]
fn conv2d_transpose_shape_formula() {
    let cases = [
        (2usize, 2usize, 2usize, 2usize, 1usize, 1usize),
        (3, 3, 3, 3, 2, 2),
        (4, 5, 3, 2, 2, 3),
        (1, 1, 4, 4, 1, 1),
    ];
    for &(height, width, kernel_h, kernel_w, stride_h, stride_w) in &cases
    {
        let shape = Conv2dTransposeShape {
            in_channels: 1,
            height,
            width,
            out_channels: 1,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
        };
        assert_eq!(shape.height_out(), (height - 1) * stride_h + kernel_h);
        assert_eq!(shape.width_out(), (width - 1) * stride_w + kernel_w);
    }
}

#[test]
#[should_panic(expected = "Conv2dTransposeShape::height_out")]
fn conv2d_transpose_zero_stride_h_panics() {
    let shape = Conv2dTransposeShape {
        in_channels: 1,
        height: 2,
        width: 2,
        out_channels: 1,
        kernel_h: 2,
        kernel_w: 2,
        stride_h: 0,
        stride_w: 1,
    };
    let _ = shape.height_out();
}

#[test]
#[should_panic(expected = "Conv2dTransposeShape::width_out")]
fn conv2d_transpose_zero_width_panics() {
    let shape = Conv2dTransposeShape {
        in_channels: 1,
        height: 2,
        width: 0,
        out_channels: 1,
        kernel_h: 2,
        kernel_w: 2,
        stride_h: 1,
        stride_w: 1,
    };
    let _ = shape.width_out();
}

#[test]
fn conv2d_transpose_known_small() {
    // 1 canal, entrée 2×2 = [[1,2],[3,4]], noyau 2×2 = [[10,20],[30,40]],
    // stride 1, biais nul : chaque élément d'entrée diffuse le noyau entier
    // (pondéré par sa valeur) sur la sortie 3×3, les recouvrements
    // s'additionnent — calcul à la main (cf. en-tête de module) :
    //   [10,  40,  40 ]
    //   [60,  200, 160]
    //   [90,  240, 160]
    let x = [1i32, 2, 3, 4].map(Q16_16::from);
    let w = [10i32, 20, 30, 40].map(Q16_16::from);
    let b = [Q16_16::zero()];
    let shape = Conv2dTransposeShape {
        in_channels: 1,
        height: 2,
        width: 2,
        out_channels: 1,
        kernel_h: 2,
        kernel_w: 2,
        stride_h: 1,
        stride_w: 1,
    };
    let y = conv2d_transpose(&x, &w, &b, shape);
    assert_eq!(
        y,
        [10i32, 40, 40, 60, 200, 160, 90, 240, 160].map(Q16_16::from)
    );
}

#[test]
fn conv2d_transpose_adds_bias_per_channel() {
    let x = [1i32, 2, 3, 4].map(Q16_16::from);
    let w = [10i32, 20, 30, 40].map(Q16_16::from);
    let b = [Q16_16::from(1000i32)];
    let shape = Conv2dTransposeShape {
        in_channels: 1,
        height: 2,
        width: 2,
        out_channels: 1,
        kernel_h: 2,
        kernel_w: 2,
        stride_h: 1,
        stride_w: 1,
    };
    let y = conv2d_transpose(&x, &w, &b, shape);
    assert_eq!(
        y,
        [1010i32, 1040, 1040, 1060, 1200, 1160, 1090, 1240, 1160].map(Q16_16::from)
    );
}

#[test]
fn conv2d_transpose_stride_scatters_without_overlap() {
    // stride == kernel : aucune fenêtre de sortie ne se recouvre — chaque
    // bloc de sortie kernel_h×kernel_w est exactement `x[ih,iw] * noyau`,
    // sans accumulation croisée (cas limite utile, distinct du recouvrement
    // additif de `conv2d_transpose_known_small`).
    let x = [1i32, 2, 3, 4].map(Q16_16::from);
    let w = [1i32, 2, 3, 4].map(Q16_16::from);
    let b = [Q16_16::zero()];
    let shape = Conv2dTransposeShape {
        in_channels: 1,
        height: 2,
        width: 2,
        out_channels: 1,
        kernel_h: 2,
        kernel_w: 2,
        stride_h: 2,
        stride_w: 2,
    };
    let y = conv2d_transpose(&x, &w, &b, shape);
    assert_eq!(shape.height_out(), 4);
    assert_eq!(shape.width_out(), 4);
    #[rustfmt::skip]
    let want = [
        1, 2,  2, 4,
        3, 4,  6, 8,
        3, 6,  4, 8,
        9, 12, 12, 16,
    ]
    .map(Q16_16::from);
    assert_eq!(y, want);
}

#[test]
fn conv2d_transpose_is_adjoint_of_conv2d() {
    // ⟨conv2d(x, W), g⟩ == ⟨x, conv2d_transpose(g, W)⟩ (cf. en-tête de
    // module) : **même** tableau `W`, aucune réindexation — seuls
    // `in_channels`/`out_channels` sont échangés dans `Conv2dTransposeShape`.
    // Biais nuls des deux côtés (l'identité porte sur l'opérateur linéaire
    // seul). Pas d'égalité bit-à-bit attendue : chaque multiplication
    // virgule fixe est arrondie indépendamment, et les deux membres ne
    // groupent pas les produits dans le même ordre (contrairement à
    // `conv2d_batch`, qui répète littéralement le même calcul) — l'écart
    // reste borné par le nombre de termes accumulés fois la résolution.
    let mut rng = Lcg(0xC0FE_7001);
    let cases = [
        (
            2usize, 3usize, 6usize, 6usize, 2usize, 2usize, 2usize, 2usize,
        ),
        (3, 2, 7, 5, 3, 3, 1, 1),
        (1, 4, 8, 6, 2, 3, 2, 1),
    ];
    for &(in_channels, out_channels, height, width, kernel_h, kernel_w, stride_h, stride_w) in
        &cases
    {
        let shape = Conv2dShape {
            in_channels,
            height,
            width,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
        };
        let height_out = shape.height_out();
        let width_out = shape.width_out();
        // Choisi pour que le format aller-retour retombe exactement sur
        // (height, width), sans troncature de la division entière côté
        // conv2d — sinon conv2d_transpose (qui ignore le remplissage
        // manquant) reconstruirait une entrée légèrement plus grande.
        assert_eq!((height_out - 1) * stride_h + kernel_h, height);
        assert_eq!((width_out - 1) * stride_w + kernel_w, width);

        let x: Vec<Q16_16> = (0..in_channels * height * width)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 8))
            .collect();
        let w: Vec<Q16_16> = (0..out_channels * in_channels * kernel_h * kernel_w)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 8))
            .collect();
        let g: Vec<Q16_16> = (0..out_channels * height_out * width_out)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 8))
            .collect();
        let zero_bias_out = vec![Q16_16::zero(); out_channels];
        let zero_bias_in = vec![Q16_16::zero(); in_channels];

        let y = conv2d(&x, &w, &zero_bias_out, shape);

        let transpose_shape = Conv2dTransposeShape {
            in_channels: out_channels,
            height: height_out,
            width: width_out,
            out_channels: in_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
        };
        let u = conv2d_transpose(&g, &w, &zero_bias_in, transpose_shape);

        let lhs = red::dot(&y, &g).to_f64();
        let rhs = red::dot(&x, &u).to_f64();
        // Tolérance proportionnelle au nombre de termes accumulés avant le
        // dernier arrondi (la somme intermédiaire la plus longue des deux
        // côtés) — mesurée empiriquement à 1-3 résolutions Q16.16
        // (1/65536) sur ces cas, ce facteur 8 laisse une marge confortable
        // sans masquer une vraie régression.
        let inner_terms =
            (in_channels * kernel_h * kernel_w).max(out_channels * kernel_h * kernel_w) as f64;
        let tol = 8.0 * inner_terms / 65536.0;
        assert!(
            (lhs - rhs).abs() <= tol,
            "adjoint mismatch shape={shape:?} lhs={lhs} rhs={rhs} tol={tol}"
        );
    }
}

#[test]
fn conv2d_transpose_i64_storage() {
    let x = [1i64, 2, 3, 4].map(Q32_32::from);
    let w = [10i64, 20, 30, 40].map(Q32_32::from);
    let b = [Q32_32::from(1000i64)];
    let shape = Conv2dTransposeShape {
        in_channels: 1,
        height: 2,
        width: 2,
        out_channels: 1,
        kernel_h: 2,
        kernel_w: 2,
        stride_h: 1,
        stride_w: 1,
    };
    let y = conv2d_transpose(&x, &w, &b, shape);
    assert_eq!(
        y,
        [1010i64, 1040, 1040, 1060, 1200, 1160, 1090, 1240, 1160].map(Q32_32::from)
    );
}

#[test]
#[should_panic(expected = "conv2d_transpose")]
fn conv2d_transpose_x_dim_mismatch_panics() {
    let x = [Q16_16::one(); 3]; // annoncé 1×2×2 = 4
    let w = [Q16_16::one(); 4];
    let b = [Q16_16::zero()];
    let shape = Conv2dTransposeShape {
        in_channels: 1,
        height: 2,
        width: 2,
        out_channels: 1,
        kernel_h: 2,
        kernel_w: 2,
        stride_h: 1,
        stride_w: 1,
    };
    let _ = conv2d_transpose(&x, &w, &b, shape);
}

#[test]
#[should_panic(expected = "conv2d_transpose")]
fn conv2d_transpose_weights_dim_mismatch_panics() {
    let x = [Q16_16::one(); 4]; // 1×2×2
    let w = [Q16_16::one(); 3]; // devrait être 1×1×2×2 = 4
    let b = [Q16_16::zero()];
    let shape = Conv2dTransposeShape {
        in_channels: 1,
        height: 2,
        width: 2,
        out_channels: 1,
        kernel_h: 2,
        kernel_w: 2,
        stride_h: 1,
        stride_w: 1,
    };
    let _ = conv2d_transpose(&x, &w, &b, shape);
}

#[test]
#[should_panic(expected = "conv2d_transpose")]
fn conv2d_transpose_bias_dim_mismatch_panics() {
    let x = [Q16_16::one(); 4];
    let w = [Q16_16::one(); 4];
    let b = [Q16_16::zero(); 2]; // devrait être 1
    let shape = Conv2dTransposeShape {
        in_channels: 1,
        height: 2,
        width: 2,
        out_channels: 1,
        kernel_h: 2,
        kernel_w: 2,
        stride_h: 1,
        stride_w: 1,
    };
    let _ = conv2d_transpose(&x, &w, &b, shape);
}

// ------------------------------------------------------------------ //
//  Pooling 2D                                                         //
// ------------------------------------------------------------------ //

/// Référence naïve, indépendante de l'implémentation (pas de tampon partagé,
/// pas d'appel à `reductions::max`/`sum`) : doit coïncider **bit-à-bit**.
fn naive_max_pool2d<const F: u32>(x: &[FixedI32<F>], shape: Pool2dShape) -> Vec<FixedI32<F>> {
    let height_out = shape.height_out();
    let width_out = shape.width_out();
    let mut y = Vec::with_capacity(shape.channels * height_out * width_out);
    for c in 0..shape.channels
    {
        for oh in 0..height_out
        {
            for ow in 0..width_out
            {
                let base = c * (shape.height * shape.width);
                let mut best = x[base + (oh * shape.stride_h) * shape.width + ow * shape.stride_w];
                for kh in 0..shape.window_h
                {
                    for kw in 0..shape.window_w
                    {
                        let h = oh * shape.stride_h + kh;
                        let w = ow * shape.stride_w + kw;
                        let v = x[base + h * shape.width + w];
                        if v > best
                        {
                            best = v;
                        }
                    }
                }
                y.push(best);
            }
        }
    }
    y
}

fn naive_avg_pool2d<const F: u32>(x: &[FixedI32<F>], shape: Pool2dShape) -> Vec<FixedI32<F>> {
    let height_out = shape.height_out();
    let width_out = shape.width_out();
    let divisor = FixedI32::<F>::from((shape.window_h * shape.window_w) as i32);
    let mut y = Vec::with_capacity(shape.channels * height_out * width_out);
    for c in 0..shape.channels
    {
        for oh in 0..height_out
        {
            for ow in 0..width_out
            {
                let base = c * (shape.height * shape.width);
                let mut acc = FixedI32::<F>::from_raw(0);
                for kh in 0..shape.window_h
                {
                    for kw in 0..shape.window_w
                    {
                        let h = oh * shape.stride_h + kh;
                        let w = ow * shape.stride_w + kw;
                        acc += x[base + h * shape.width + w];
                    }
                }
                y.push(acc.checked_div(divisor).unwrap());
            }
        }
    }
    y
}

#[test]
fn pool2d_known_small_disjoint_tiling() {
    // 1 canal, x = arithmétique 1..16 (4×4), fenêtre 2×2, stride (2,2)
    // (pavage disjoint, sans chevauchement).
    let x = (1i32..=16).map(Q16_16::from).collect::<Vec<_>>();
    let shape = Pool2dShape {
        channels: 1,
        height: 4,
        width: 4,
        window_h: 2,
        window_w: 2,
        stride_h: 2,
        stride_w: 2,
    };
    assert_eq!(shape.height_out(), 2);
    assert_eq!(shape.width_out(), 2);
    // Fenêtres : {1,2,5,6}, {3,4,7,8}, {9,10,13,14}, {11,12,15,16}.
    let mx = max_pool2d(&x, shape);
    assert_eq!(mx, [6i32, 8, 14, 16].map(Q16_16::from));
    let avg = avg_pool2d(&x, shape);
    // Sommes 14,22,46,54, ÷4 = 3.5, 5.5, 11.5, 13.5 (division réelle exacte).
    assert_eq!(avg, [q16(3.5), q16(5.5), q16(11.5), q16(13.5)]);
}

#[test]
fn pool2d_overlapping_window_multi_channel() {
    // 2 canaux, fenêtre 2×2 chevauchante (stride (1,1) < fenêtre), 3×3 par
    // canal. Sommes choisies multiples de 4 pour une moyenne entière exacte.
    let c0 = [4i32, 8, 12, 16, 20, 24, 28, 32, 36]; // 3×3, pas constant
    let c1 = [40i32, 80, 120, 160, 200, 240, 280, 320, 360];
    let x: Vec<Q16_16> = c0
        .iter()
        .chain(c1.iter())
        .copied()
        .map(Q16_16::from)
        .collect();
    let shape = Pool2dShape {
        channels: 2,
        height: 3,
        width: 3,
        window_h: 2,
        window_w: 2,
        stride_h: 1,
        stride_w: 1,
    };
    assert_eq!(shape.height_out(), 2);
    assert_eq!(shape.width_out(), 2);
    let got_max = max_pool2d(&x, shape);
    let want_max = naive_max_pool2d(&x, shape);
    assert_eq!(got_max, want_max);
    let got_avg = avg_pool2d(&x, shape);
    let want_avg = naive_avg_pool2d(&x, shape);
    assert_eq!(got_avg, want_avg);
}

#[test]
fn pool2d_matches_naive_reference_varied_shapes() {
    let mut rng = Lcg(0xFEED2D);
    let cases = [
        (3, 10, 8, 3, 3, 2, 2),
        (1, 6, 6, 6, 6, 1, 1),
        (2, 9, 7, 2, 3, 3, 2),
        (4, 5, 5, 2, 2, 1, 1),
    ];
    for &(channels, height, width, window_h, window_w, stride_h, stride_w) in &cases
    {
        let shape = Pool2dShape {
            channels,
            height,
            width,
            window_h,
            window_w,
            stride_h,
            stride_w,
        };
        let height_out = shape.height_out();
        let width_out = shape.width_out();
        let x: Vec<Q16_16> = (0..channels * height * width)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 4))
            .collect();

        let got_max = max_pool2d(&x, shape);
        let want_max = naive_max_pool2d(&x, shape);
        assert_eq!(got_max.len(), channels * height_out * width_out);
        assert_eq!(got_max, want_max, "max_pool2d shape={shape:?}");

        let got_avg = avg_pool2d(&x, shape);
        let want_avg = naive_avg_pool2d(&x, shape);
        assert_eq!(got_avg, want_avg, "avg_pool2d shape={shape:?}");
    }
}

#[test]
#[should_panic(expected = "Pool2dShape::height_out")]
fn pool2d_window_taller_than_input_panics() {
    let x = [Q16_16::one(); 9]; // 1×3×3
    let shape = Pool2dShape {
        channels: 1,
        height: 3,
        width: 3,
        window_h: 4,
        window_w: 1,
        stride_h: 1,
        stride_w: 1,
    };
    let _ = max_pool2d(&x, shape);
}

#[test]
#[should_panic(expected = "avg_pool2d")]
fn pool2d_dim_mismatch_panics() {
    let x = [Q16_16::one(); 8]; // annoncé 1×3×3 attendu 9, fourni 8
    let shape = Pool2dShape {
        channels: 1,
        height: 3,
        width: 3,
        window_h: 2,
        window_w: 2,
        stride_h: 1,
        stride_w: 1,
    };
    let _ = avg_pool2d(&x, shape);
}

#[test]
fn pool2d_i64_storage() {
    // Chemin de stockage i64 (Q32.32) : mêmes garanties.
    let x = [1i64, 9, 3, 7].map(Q32_32::from); // 2×2, une seule fenêtre 2×2
    let shape = Pool2dShape {
        channels: 1,
        height: 2,
        width: 2,
        window_h: 2,
        window_w: 2,
        stride_h: 1,
        stride_w: 1,
    };
    assert_eq!(max_pool2d(&x, shape), [Q32_32::from(9i64)]);
    assert_eq!(avg_pool2d(&x, shape), [Q32_32::from(5i64)]); // (1+9+3+7)/4=5
}

#[test]
fn pool2d_feeds_conv2d_chain() {
    // Usage réaliste : conv2d → max_pool2d, comme dans un CNN léger 2D.
    let x = (1i32..=16).map(Q16_16::from).collect::<Vec<_>>(); // 1×4×4
    let w = [1i32, 0, 0, -1].map(Q16_16::from); // diagonale 2×2
    let b = [Q16_16::zero()];
    let conv_shape = Conv2dShape {
        in_channels: 1,
        height: 4,
        width: 4,
        out_channels: 1,
        kernel_h: 2,
        kernel_w: 2,
        stride_h: 1,
        stride_w: 1,
    };
    // y[oh,ow] = x[oh,ow] - x[oh+1,ow+1] = -(width+1) = -5 partout (largeur 4).
    let conv_out = conv2d(&x, &w, &b, conv_shape); // 1×3×3, toutes valeurs = -5
    let pool_shape = Pool2dShape {
        channels: 1,
        height: 3,
        width: 3,
        window_h: 2,
        window_w: 2,
        stride_h: 1,
        stride_w: 1,
    };
    let pooled = max_pool2d(&conv_out, pool_shape);
    assert_eq!(pooled, vec![q16(-5.0); 4]); // toutes les fenêtres valent -5
}

// ------------------------------------------------------------------ //
//  Math : sqrt / rsqrt / reciprocal                                   //
// ------------------------------------------------------------------ //

#[test]
fn sqrt_reference() {
    assert_eq!(sqrt(q16(4.0)).to_f64(), 2.0);
    assert_eq!(sqrt(q16(0.25)).to_f64(), 0.5);
    assert_eq!(sqrt(Q16_16::zero()), Q16_16::zero());
    assert_eq!(sqrt(q16(-1.0)), Q16_16::zero()); // convention : ≤0 → 0
    let mut rng = Lcg(0x321);
    for _ in 0..1000
    {
        let raw = (rng.raw_i32().unsigned_abs() as i32) & i32::MAX;
        let x = Q16_16::from_raw(raw);
        let got = sqrt(x).to_f64();
        let expected = x.to_f64().sqrt();
        assert!(
            (got - expected).abs() <= 1.0 / 65536.0,
            "√{x}: {got} vs {expected}"
        );
    }
}

#[test]
fn reciprocal_and_rsqrt() {
    assert_eq!(reciprocal(q16(2.0)).unwrap().to_f64(), 0.5);
    assert_eq!(reciprocal(q16(0.25)).unwrap().to_f64(), 4.0);
    assert!(reciprocal(Q16_16::zero()).is_none());
    // rsqrt(4) = 0.5.
    assert!((rsqrt(q16(4.0)).unwrap().to_f64() - 0.5).abs() <= 2.0 / 65536.0);
    assert!(rsqrt(Q16_16::zero()).is_none());
    // Q8.24 : plus haute résolution.
    let r = reciprocal(Q8_24::try_from(8.0).unwrap()).unwrap();
    assert!((r.to_f64() - 0.125).abs() <= 1.0 / (1 << 24) as f64);
}

// ------------------------------------------------------------------ //
//  Bits, extrêmes, signes, puissances de deux                         //
// ------------------------------------------------------------------ //

#[test]
fn bit_stability_and_extremes() {
    // Bruts figés : la représentation ne doit jamais changer silencieusement.
    assert_eq!(q16(1.0).to_raw(), 65536);
    assert_eq!(q16(-1.0).to_raw(), -65536);
    assert_eq!(q16(0.5).to_raw(), 32768);
    assert_eq!(Q8_24::try_from(1.0).unwrap().to_raw(), 1 << 24);
    // Puissances de deux : multiplication = décalage exact.
    for k in 0..10
    {
        let p = q16((1 << k) as f64);
        assert_eq!((p * q16(3.0)).to_f64(), (3 * (1 << k)) as f64);
    }
    // Extrêmes.
    assert_eq!(Q16_16::max_value().to_raw(), i32::MAX);
    assert_eq!(Q16_16::min_value().to_raw(), i32::MIN);
    // Zéro signé n'existe pas : -0 == 0.
    assert_eq!(-Q16_16::zero(), Q16_16::zero());
}

#[test]
fn display_is_exact() {
    assert_eq!(format!("{}", q16(1.5)), "1.5");
    assert_eq!(format!("{}", q16(-2.25)), "-2.25");
    assert_eq!(format!("{}", Q16_16::zero()), "0");
    assert_eq!(format!("{}", q16(3.0)), "3");
    // 0.5 = exact ; 0.1 n'est pas binaire-exact mais termine en ≤16 chiffres.
    assert_eq!(format!("{}", q16(0.5)), "0.5");
    assert_eq!(format!("{}", Q16_16::resolution()), "0.0000152587890625");
}

// ------------------------------------------------------------------ //
//  Généricité : NumericScalar sur f32/f64/Fixed                       //
// ------------------------------------------------------------------ //

/// Polynôme générique : `x² + x + 1`.
fn poly<T: NumericScalar>(x: T) -> T {
    x * x + x + T::one()
}

#[test]
fn numeric_scalar_generic() {
    // Le MÊME code s'instancie sur flottant et virgule fixe.
    assert_eq!(poly(2.0f64), 7.0);
    assert_eq!(poly(2.0f32), 7.0);
    assert_eq!(poly(q16(2.0)).to_f64(), 7.0);
    assert_eq!(poly(Q32_32::try_from(3.0).unwrap()).to_f64(), 13.0);
    // abs / from_i32 génériques.
    assert_eq!(<Q16_16 as NumericScalar>::from_i32(-4).abs().to_f64(), 4.0);
    assert_eq!(<Q16_16 as NumericScalar>::zero(), Q16_16::zero());
    assert_eq!(<Q16_16 as NumericScalar>::one(), Q16_16::one());
}

#[test]
fn q24_8_wide_range() {
    // Q24.8 : grande plage, faible résolution.
    let big = FixedI32::<8>::from(1_000_000);
    assert_eq!(big.to_f64(), 1_000_000.0);
    assert_eq!((big + FixedI32::<8>::from(1)).to_f64(), 1_000_001.0);
}

// ------------------------------------------------------------------ //
//  Transcendantes : valeurs connues + bornes ULP prouvées             //
// ------------------------------------------------------------------ //

/// 1 ULP Q16.16 en valeur réelle.
const ULP16: f64 = 1.0 / 65536.0;

/// Erreur maximale (en ULP Q16.16) d'une transcendante virgule fixe vs `f64`,
/// balayée sur `[lo, hi]` en `steps + 1` points. La référence `f64` est évaluée
/// sur l'entrée **réellement représentée** (`x.to_f64()`), de sorte que l'on
/// mesure l'erreur de l'algorithme et de la quantification de sortie, pas celle
/// de la quantification d'entrée. Renvoie l'ULP maximal ET l'entrée fautive.
fn sweep_ulp<F, G>(lo: f64, hi: f64, steps: i64, fx: F, reff: G) -> (f64, f64)
where
    F: Fn(Q16_16) -> Q16_16,
    G: Fn(f64) -> f64,
{
    let mut worst = 0.0f64;
    let mut worst_at = lo;
    for s in 0..=steps
    {
        let v = lo + (hi - lo) * (s as f64) / (steps as f64);
        let x = q16(v);
        let got = fx(x).to_f64();
        let want = reff(x.to_f64());
        let ulp = (got - want).abs() / ULP16;
        if ulp > worst
        {
            worst = ulp;
            worst_at = x.to_f64();
        }
    }
    (worst, worst_at)
}

/// `assert` que le balayage reste sous `bound` ULP, avec diagnostic.
fn assert_ulp<F, G>(name: &str, bound: f64, lo: f64, hi: f64, steps: i64, fx: F, reff: G)
where
    F: Fn(Q16_16) -> Q16_16,
    G: Fn(f64) -> f64,
{
    let (worst, at) = sweep_ulp(lo, hi, steps, fx, reff);
    assert!(
        worst <= bound,
        "{name}: erreur max {worst:.3} ULP > {bound} (à x = {at})"
    );
}

#[test]
fn transcendental_known_values() {
    // Cas exacts (aucun arrondi résiduel attendu après quantification).
    assert_eq!(tr::exp(Q16_16::zero()), Q16_16::one()); // e⁰ = 1
    assert_eq!(tr::exp2(q16(3.0)), q16(8.0)); // 2³ = 8
    assert_eq!(tr::exp2(q16(-2.0)), q16(0.25)); // 2⁻² = 1/4
    assert_eq!(tr::sin(Q16_16::zero()), Q16_16::zero()); // sin 0 = 0
    assert_eq!(tr::cos(Q16_16::zero()), Q16_16::one()); // cos 0 = 1
    assert_eq!(tr::tanh(Q16_16::zero()), Q16_16::zero()); // tanh 0 = 0
    assert_eq!(tr::sigmoid(Q16_16::zero()), q16(0.5)); // σ(0) = 1/2

    // Cas connus à ≤ 2 ULP (arrondi de la réduction/du polynôme).
    let near = |a: Q16_16, b: f64| (a.to_f64() - b).abs() <= 2.0 * ULP16;
    assert!(near(tr::ln(Q16_16::one()), 0.0)); // ln 1 = 0
    assert!(near(tr::log2(q16(8.0)), 3.0)); // log₂ 8 = 3
    assert!(near(tr::log2(q16(1024.0)), 10.0)); // log₂ 1024 = 10
    assert!(near(tr::ln(q16(std::f64::consts::E)), 1.0)); // ln e = 1
    assert!(near(tr::sin(q16(std::f64::consts::FRAC_PI_2)), 1.0)); // sin(π/2)=1
    assert!(near(tr::cos(q16(std::f64::consts::PI)), -1.0)); // cos π = -1
    assert!(near(tr::exp(q16(1.0)), std::f64::consts::E)); // e¹ = e
}

#[test]
fn transcendental_ulp_bounds() {
    // Bornes prouvées par balayage dense sur tout le domaine actif de Q16.16.
    // Valeurs mesurées (maillage 40 001 points) :
    //   exp 3.24  exp2 5.01  ln 0.50  log2 0.50  sin 0.52  cos 0.52
    //   tanh 0.50  sigmoid 0.50 (ULP Q16.16).
    // Les bornes assertées gardent une marge et documentent le pire cas réel :
    // exp/exp2 croissent près du sommet de la plage (l'erreur relative du
    // minimax × la magnitude), les autres restent sous 1 ULP partout.
    let n = 40_000;
    // exp : domaine où eˣ ∈ (résolution, max Q16.16). e¹⁰ ≈ 22026 < 32768.
    assert_ulp("exp", 4.0, -10.0, 10.0, n, tr::exp, f64::exp);
    // exp2 : 2ˣ, 2¹⁴·⁵ ≈ 23170 < 32768 (pire cas au sommet de la plage).
    assert_ulp("exp2", 6.0, -14.0, 14.5, n, tr::exp2, f64::exp2);
    // ln / log2 : x > 0, jusqu'au sommet de la plage.
    assert_ulp("ln", 1.0, ULP16, 32000.0, n, tr::ln, f64::ln);
    assert_ulp("log2", 1.0, ULP16, 32000.0, n, tr::log2, f64::log2);
    // sin / cos : large domaine → exerce la réduction d'argument mod 2π.
    assert_ulp("sin", 1.0, -100.0, 100.0, n, tr::sin, f64::sin);
    assert_ulp("cos", 1.0, -100.0, 100.0, n, tr::cos, f64::cos);
    // tanh / sigmoid : saturent hors de ±~12, bornés partout.
    assert_ulp("tanh", 1.0, -12.0, 12.0, n, tr::tanh, f64::tanh);
    assert_ulp("sigmoid", 1.0, -16.0, 16.0, n, tr::sigmoid, |x| {
        1.0 / (1.0 + (-x).exp())
    });
}

#[test]
fn inverse_trig_ulp_bounds() {
    // Mesuré (maillage 40 001 pts) : atan 0.50, asin 0.51, acos 0.51 ULP.
    let n = 40_000;
    // atan : borné (±π/2). Domaine large pour exercer la réduction |x|>1.
    assert_ulp("atan", 1.0, -128.0, 128.0, n, tr::atan, f64::atan);
    // asin / acos : dérivée → ∞ en ±1, donc l'erreur ULP croît près des bords ;
    // on teste [-0.999, 0.999] (le domaine exploitable) + les bords traités à
    // part par `inverse_trig_known_values`.
    assert_ulp("asin", 1.0, -0.999, 0.999, n, tr::asin, f64::asin);
    assert_ulp("acos", 1.0, -0.999, 0.999, n, tr::acos, f64::acos);
    // atan2 le long du cercle unité : angle(cosθ, sinθ) doit rendre θ.
    let mut worst = 0.0f64;
    for k in 0..=n
    {
        let theta = -std::f64::consts::PI + 2.0 * std::f64::consts::PI * (k as f64) / (n as f64);
        let (s, c) = (theta.sin(), theta.cos());
        let got = tr::atan2(q16(s), q16(c)).to_f64();
        // repli d'angle pour comparer près de ±π.
        let want = q16(s).to_f64().atan2(q16(c).to_f64());
        let mut d = (got - want).abs();
        if d > std::f64::consts::PI
        {
            d = (2.0 * std::f64::consts::PI - d).abs();
        }
        worst = worst.max(d / ULP16);
    }
    assert!(worst <= 2.0, "atan2 cercle: {worst:.3} ULP");
}

#[test]
fn inverse_trig_known_values() {
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};
    let near = |a: Q16_16, b: f64| (a.to_f64() - b).abs() <= 2.0 * ULP16;
    // atan.
    assert_eq!(tr::atan(Q16_16::zero()), Q16_16::zero());
    assert!(near(tr::atan(q16(1.0)), FRAC_PI_4));
    assert!(near(tr::atan(q16(-1.0)), -FRAC_PI_4));
    // asin / acos aux bornes (traitées par saturation du domaine).
    assert!(near(tr::asin(Q16_16::zero()), 0.0));
    assert!(near(tr::asin(q16(1.0)), FRAC_PI_2));
    assert!(near(tr::asin(q16(-1.0)), -FRAC_PI_2));
    assert!(near(tr::acos(q16(1.0)), 0.0));
    assert!(near(tr::acos(Q16_16::zero()), FRAC_PI_2));
    assert!(near(tr::acos(q16(-1.0)), PI));
    assert!(near(tr::asin(q16(0.5)), (0.5f64).asin()));
    // Hors domaine : saturation propre, sans panique.
    assert!(near(tr::asin(q16(2.0)), FRAC_PI_2));
    assert!(near(tr::acos(q16(-2.0)), PI));
    // atan2 : quadrants.
    assert!(near(tr::atan2(q16(1.0), q16(1.0)), FRAC_PI_4));
    assert!(near(tr::atan2(q16(1.0), Q16_16::zero()), FRAC_PI_2));
    assert!(near(tr::atan2(Q16_16::zero(), q16(-1.0)), PI));
    assert!(near(tr::atan2(q16(-1.0), q16(-1.0)), -(PI - FRAC_PI_4)));
    assert_eq!(tr::atan2(Q16_16::zero(), Q16_16::zero()), Q16_16::zero());
}

#[test]
fn inverse_trig_generic_real_scalar() {
    // atan/atan2/asin/acos exposés via RealScalar, cohérents f32/f64/fixe.
    assert!((RealScalar::atan(1.0f32) - std::f32::consts::FRAC_PI_4).abs() < 1e-6);
    assert!((RealScalar::atan2(1.0f64, 1.0f64) - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
    assert!((RealScalar::asin(q16(0.5)).to_f64() - 0.5f64.asin()).abs() < 2e-3);
    assert!((RealScalar::acos(q16(0.25)).to_f64() - 0.25f64.acos()).abs() < 2e-3);
}

// ------------------------------------------------------------------ //
//  bessel_i0 (fonction de Bessel modifiée, ordre 0)                   //
// ------------------------------------------------------------------ //

/// Référence indépendante : somme directe de la série `I₀(x) = Σₖ
/// (x/2)^{2k}/(k!)²`, sans lien avec le polynôme minimax de
/// `transcendental::bessel_i0` ni la formule rationnelle d'Abramowitz &
/// Stegun de l'impl `f32`/`f64` — validation croisée véritablement
/// indépendante.
fn i0_series_ref(x: f64) -> f64 {
    let mut term = 1.0f64;
    let mut total = 1.0f64;
    for k in 1..100
    {
        term *= (x / (2.0 * k as f64)).powi(2);
        total += term;
        if term < 1e-18 * total
        {
            break;
        }
    }
    total
}

#[test]
fn bessel_i0_known_values() {
    let near = |a: Q16_16, b: f64| (a.to_f64() - b).abs() <= 2.0 * ULP16;
    assert!(near(tr::bessel_i0(Q16_16::zero()), 1.0)); // I₀(0) = 1
    // I₀ est paire.
    assert_eq!(tr::bessel_i0(q16(3.0)), tr::bessel_i0(q16(-3.0)));
    assert_eq!(tr::bessel_i0(q16(9.0)), tr::bessel_i0(q16(-9.0)));
}

#[test]
fn bessel_i0_ulp_bounds() {
    // Mesuré (maillage 40 001 pts) : 131.3 ULP sur tout [0, 12] (dominé par
    // la magnitude au sommet, I₀(12) ≈ 18949), 1.84 ULP sur [0, 8.9]
    // (I₀(x) ≤ 1024) — même phénomène que exp/exp2, voir la doc du module.
    let n = 40_000;
    assert_ulp(
        "bessel_i0",
        135.0,
        0.0,
        12.0,
        n,
        tr::bessel_i0,
        i0_series_ref,
    );
    assert_ulp(
        "bessel_i0 (I0<=1024)",
        2.0,
        0.0,
        8.9,
        n,
        tr::bessel_i0,
        i0_series_ref,
    );
}

#[test]
fn bessel_i0_generic_real_scalar() {
    // Exposée via RealScalar, cohérente f32/f64/virgule fixe.
    // Impl f32/f64 : formule rationnelle d'Abramowitz & Stegun, erreur
    // relative ≤ ~1.6e-7 (pas une identité std exacte, contrairement à
    // atan/atan2/...).
    assert!((RealScalar::bessel_i0(0.0f32) - 1.0).abs() < 1e-6);
    assert!((RealScalar::bessel_i0(3.0f64) - i0_series_ref(3.0)).abs() < 1e-6);
    assert!((RealScalar::bessel_i0(q16(5.0)).to_f64() - i0_series_ref(5.0)).abs() < 2e-3);
}

// ------------------------------------------------------------------ //
//  erf / erfc (fonction d'erreur)                                    //
// ------------------------------------------------------------------ //

/// Référence indépendante : série de Maclaurin directe `erf(x) = (2/√π)·Σₙ
/// (-1)ⁿx^{2n+1}/(n!(2n+1))`, sans lien avec le polynôme minimax de
/// `transcendental::erf` ni la formule d'Abramowitz & Stegun de l'impl
/// `f32`/`f64`.
fn erf_series_ref(x: f64) -> f64 {
    let mut term = x;
    let mut sum = x;
    for n in 1..300
    {
        let nf = n as f64;
        term *= -x * x / nf * (2.0 * nf - 1.0) / (2.0 * nf + 1.0);
        sum += term;
        if term.abs() < 1e-18 * sum.abs().max(1e-300)
        {
            break;
        }
    }
    sum * 2.0 / std::f64::consts::PI.sqrt()
}

#[test]
fn erf_known_values() {
    let near = |a: Q16_16, b: f64| (a.to_f64() - b).abs() <= 2.0 * ULP16;
    assert!(near(tr::erf(Q16_16::zero()), 0.0)); // erf(0) = 0
    assert!(near(tr::erfc(Q16_16::zero()), 1.0)); // erfc(0) = 1
    // erf est impaire.
    assert_eq!(tr::erf(q16(1.5)), -tr::erf(q16(-1.5)));
    assert_eq!(tr::erf(q16(3.0)), -tr::erf(q16(-3.0)));
}

#[test]
fn erf_ulp_bounds() {
    // Mesuré (maillage 40 001 pts) : ≤ 0.52 ULP sur [-4.5, 4.5] — fonction
    // bien conditionnée (sortie bornée dans [-1, 1]), contrairement à
    // bessel_i0. Domaine borné à ±4.5, pas ±6 : au-delà, `erf_series_ref`
    // elle-même perd sa précision par annulation catastrophique en f64
    // (Σ de termes alternés de grande magnitude avant convergence), et cesse
    // d'être une référence fiable — indépendamment de la justesse de
    // `transcendental::erf`.
    let n = 40_000;
    assert_ulp("erf", 1.0, -4.5, 4.5, n, tr::erf, erf_series_ref);
    assert_ulp("erfc", 1.0, -4.5, 4.5, n, tr::erfc, |x| {
        1.0 - erf_series_ref(x)
    });
}

#[test]
fn erf_saturates_beyond_domain() {
    // Au-delà du domaine ajusté ([0, 4]), erf/erfc doivent rester proches de
    // leurs limites ±1/0 (saturation propre), sans dépendre d'une référence
    // fragile aux grands |x| (voir `erf_ulp_bounds`).
    for &x in &[5.0, 6.0, 8.0, 10.0]
    {
        let e = tr::erf(q16(x)).to_f64();
        assert!((e - 1.0).abs() < 1e-3, "erf({x}) = {e}, attendu ≈ 1");
        let ec = tr::erfc(q16(x)).to_f64();
        assert!(ec.abs() < 1e-3, "erfc({x}) = {ec}, attendu ≈ 0");
        // Impaire / symétrique.
        assert_eq!(tr::erf(q16(-x)), -tr::erf(q16(x)));
    }
}

#[test]
fn erf_generic_real_scalar() {
    // Exposée via RealScalar, cohérente f32/f64/virgule fixe. Impl f32/f64 :
    // formule rationnelle d'Abramowitz & Stegun, erreur ≤ ~1.5e-7.
    assert!((RealScalar::erf(0.0f32) - 0.0).abs() < 1e-6);
    assert!((RealScalar::erf(1.0f64) - erf_series_ref(1.0)).abs() < 1e-6);
    assert!((RealScalar::erfc(1.0f64) - (1.0 - erf_series_ref(1.0))).abs() < 1e-6);
    assert!((RealScalar::erf(q16(2.0)).to_f64() - erf_series_ref(2.0)).abs() < 2e-3);
}

#[test]
fn transcendental_high_resolution_q8_24() {
    // La même implémentation générique sert un autre FRAC (Q8.24, résolution
    // 6e-8) : les identités de base tiennent sur ce format haute précision.
    type Q = Q8_24;
    let one = Q::one();
    let ulp = 1.0 / (1u64 << 24) as f64;
    let near = |a: Q, b: f64, tol_ulp: f64| (a.to_f64() - b).abs() <= tol_ulp * ulp;
    assert!(near(tr::exp(Q::zero()), 1.0, 1.0));
    assert!(near(tr::exp2(Q::try_from(2.0).unwrap()), 4.0, 4.0));
    assert!(near(tr::ln(one), 0.0, 4.0));
    assert!(near(tr::sigmoid(Q::zero()), 0.5, 1.0));
    // sin(π/6) = 1/2.
    let pi6 = Q::try_from(std::f64::consts::PI / 6.0).unwrap();
    assert!(near(tr::sin(pi6), 0.5, 16.0));
}

#[test]
fn real_scalar_generic_over_float_and_fixed() {
    // Une activation générique (SiLU : x·σ(x)) s'instancie identiquement sur
    // flottant et virgule fixe — c'est l'intérêt de RealScalar.
    fn silu<T: RealScalar>(x: T) -> T {
        x * x.sigmoid()
    }
    let silu_f = silu(1.5f32) as f64;
    let silu_x = silu(q16(1.5)).to_f64();
    assert!(
        (silu_f - silu_x).abs() < 2e-3,
        "SiLU flottant {silu_f} vs fixe {silu_x}"
    );

    // Toutes les méthodes RealScalar sur virgule fixe, cohérentes avec f64.
    let approx = |a: Q16_16, b: f64| (a.to_f64() - b).abs() < 2e-3;
    assert!(approx(RealScalar::sqrt(q16(2.0)), 2f64.sqrt()));
    assert!(approx(RealScalar::recip(q16(4.0)), 0.25));
    assert!(approx(RealScalar::exp(q16(2.0)), 2f64.exp()));
    assert!(approx(RealScalar::ln(q16(10.0)), 10f64.ln()));
    assert!(approx(RealScalar::tanh(q16(0.5)), 0.5f64.tanh()));

    // Délégation flottante de RealScalar (sigmoïde dérivée).
    assert!((RealScalar::sigmoid(0.0f32) - 0.5).abs() < 1e-6);
    assert!((RealScalar::sigmoid(0.0f64) - 0.5).abs() < 1e-12);
}

#[test]
fn softmax_normalized_and_order_independent() {
    let input = [q16(1.0), q16(2.0), q16(3.0), q16(-1.0)];
    let mut out = [Q16_16::zero(); 4];
    tr::softmax_into(&input, &mut out);

    // Somme ≈ 1 (probabilités).
    let sum: f64 = out.iter().map(|p| p.to_f64()).sum();
    assert!((sum - 1.0).abs() < 1e-3, "Σ softmax = {sum}");
    // Monotone : plus grand logit ⇒ plus grande probabilité.
    assert!(out[2] > out[1] && out[1] > out[0] && out[0] > out[3]);
    // Valeurs vs référence f64 stable (max-subtract).
    let logits = [1.0f64, 2.0, 3.0, -1.0];
    let mx = 3.0f64;
    let denom: f64 = logits.iter().map(|l| (l - mx).exp()).sum();
    for (o, l) in out.iter().zip(logits)
    {
        assert!((o.to_f64() - (l - mx).exp() / denom).abs() < 1e-3);
    }

    // Déterminisme bit-à-bit : la somme est accumulée en i128 (exacte, donc
    // indépendante de l'ordre). Une permutation de l'entrée permute la sortie
    // à l'identique — aucun résidu d'arrondi dépendant de l'ordre.
    let perm = [q16(3.0), q16(1.0), q16(-1.0), q16(2.0)];
    let mut out2 = [Q16_16::zero(); 4];
    tr::softmax_into(&perm, &mut out2);
    assert_eq!(out2[0], out[2]);
    assert_eq!(out2[1], out[0]);
    assert_eq!(out2[2], out[3]);
    assert_eq!(out2[3], out[1]);
}

#[test]
fn transcendental_saturates_without_panic() {
    // Pas d'infini/NaN en virgule fixe : les cas limites saturent proprement.
    assert_eq!(tr::ln(Q16_16::zero()), Q16_16::min_value()); // ln 0 → min
    assert_eq!(tr::ln(q16(-1.0)), Q16_16::min_value()); // ln(<0) → min
    assert_eq!(tr::log2(Q16_16::zero()), Q16_16::min_value());
    // exp d'un grand argument sature au max au lieu de déborder.
    assert_eq!(tr::exp(q16(1000.0)), Q16_16::max_value());
    assert_eq!(tr::exp2(q16(1000.0)), Q16_16::max_value());
    // exp d'un grand argument négatif tend vers 0.
    assert_eq!(tr::exp(q16(-1000.0)), Q16_16::zero());
    // tanh/sigmoid saturent dans [−1,1] / [0,1].
    assert!(tr::tanh(q16(50.0)).to_f64() <= 1.0);
    assert!((tr::tanh(q16(50.0)).to_f64() - 1.0).abs() < 1e-3);
    assert!((tr::sigmoid(q16(50.0)).to_f64() - 1.0).abs() < 1e-3);
    assert!(tr::sigmoid(q16(-50.0)).to_f64() < 1e-3);
}

// ------------------------------------------------------------------ //
//  Stockage i16 (FixedI16) — validation de la généricité             //
// ------------------------------------------------------------------ //

#[test]
fn fixed_i16_constants_and_layout() {
    use core::mem::size_of;
    assert_eq!(size_of::<Q1_15>(), size_of::<i16>());
    assert_eq!(size_of::<Q8_8>(), size_of::<i16>());
    // Q8.8 : 1.0 = 2^8 = 256 (représentable).
    assert_eq!(Q8_8::one().to_raw(), 1 << 8);
    assert_eq!(Q8_8::resolution().to_raw(), 1);
    assert_eq!(Q8_8::resolution().to_f64(), 1.0 / 256.0);
    // Q1.15 : échantillons dans [−1, 1). 1.0 non représentable (documenté).
    assert_eq!(Q1_15::resolution().to_f64(), 1.0 / 32768.0);
    assert_eq!(Q1_15::max_value().to_raw(), i16::MAX);
    assert_eq!(Q1_15::min_value().to_f64(), -1.0);
    assert!(Q1_15::max_value().to_f64() < 1.0);
}

#[test]
fn fixed_i16_conversions_saturating() {
    // Q8.8 : conversions entières saturantes.
    assert_eq!(Q8_8::from(3).to_f64(), 3.0);
    assert_eq!(Q8_8::from(-5).to_f64(), -5.0);
    assert_eq!(Q8_8::from(1000), Q8_8::max_value()); // 1000 > 127.99 → sature
    assert_eq!(Q8_8::from(-1000), Q8_8::min_value());
    // Aller-retour flottant à la résolution.
    for &v in &[0.0, 0.5, -0.25, 3.75, -12.5]
    {
        let f = Q8_8::try_from(v).unwrap();
        assert!((f.to_f64() - v).abs() <= 1.0 / 256.0, "v={v}");
    }
    // Q1.15 : plage échantillon.
    assert!((Q1_15::try_from(0.5).unwrap().to_f64() - 0.5).abs() <= 1.0 / 32768.0);
    assert!((Q1_15::try_from(-0.75).unwrap().to_f64() + 0.75).abs() <= 1.0 / 32768.0);
    assert!(Q1_15::try_from(1.5).is_err()); // hors plage
}

#[test]
fn fixed_i16_multiply_uses_i32_wide_exactly() {
    // Le produit passe par l'accumulateur i32 (élargi de i16) : exact avant
    // arrondi. On compare à une référence i32 (troncature vers zéro).
    let mut rng = Lcg(0xF16);
    for _ in 0..2000
    {
        let ra = (rng.next() >> 49) as i16; // i16 modéré
        let rb = (rng.next() >> 49) as i16;
        let a = FixedI16::<8>::from_raw(ra);
        let b = FixedI16::<8>::from_raw(rb);
        let got = (a * b).to_raw() as i64;
        let p = (ra as i64) * (rb as i64);
        let expected = if p < 0 { -((-p) >> 8) } else { p >> 8 };
        assert_eq!(got, expected as i16 as i64, "a={ra} b={rb}");
    }
}

#[test]
fn fixed_i16_audio_range_and_overflow() {
    // Q1.15 : le produit de deux échantillons de [−1, 1) reste dans la plage.
    let a = Q1_15::try_from(0.8).unwrap();
    let b = Q1_15::try_from(-0.6).unwrap();
    assert!(((a * b).to_f64() - (-0.48)).abs() < 2.0 / 32768.0);
    // Overflow des opérateurs : enveloppe déterministe.
    let max = Q8_8::max_value();
    assert_eq!(
        max + Q8_8::one(),
        Q8_8::from_raw(i16::MAX.wrapping_add(1 << 8))
    );
    assert!(max.checked_add(Q8_8::one()).is_none());
    assert_eq!(max.saturating_add(Q8_8::one()), max);
    // Négation de MIN sature.
    assert_eq!(Q1_15::min_value().saturating_neg(), Q1_15::max_value());
}

#[test]
fn fixed_i16_is_numeric_scalar_generic() {
    // La MÊME fonction générique (déjà utilisée pour f32/i32/i64) s'instancie
    // sur le stockage i16 sans réécriture — c'est tout l'objet du lot.
    assert_eq!(poly(Q8_8::try_from(2.0).unwrap()).to_f64(), 7.0); // x²+x+1 en 2 = 7
    assert_eq!(<Q8_8 as NumericScalar>::from_i32(-4).abs().to_f64(), 4.0);
    assert_eq!(<Q8_8 as NumericScalar>::one().to_f64(), 1.0);
    // Déterminisme bit-à-bit d'un petit calcul.
    let run = || {
        let mut acc = Q8_8::zero();
        for i in 0..20
        {
            acc += Q8_8::try_from((i as f64) * 0.1 - 1.0).unwrap() * Q8_8::try_from(0.5).unwrap();
        }
        acc.to_raw()
    };
    assert_eq!(run(), run());
}

// ------------------------------------------------------------------ //
//  Attention produit-scalaire mise à l'échelle (déterministe)         //
// ------------------------------------------------------------------ //

fn to_f64_vec(x: &[Q16_16]) -> Vec<f64> {
    x.iter().map(|v| v.to_f64()).collect()
}

/// Référence f64 indépendante : `Attention(Q,K,V) = softmax(scale·Q·Kᵀ)·V`.
fn attention_ref_f64(
    q: &[f64],
    s: usize,
    d: usize,
    k: &[f64],
    t: usize,
    v: &[f64],
    scale: f64,
) -> Vec<f64> {
    let mut scores = vec![0.0f64; s * t];
    for i in 0..s
    {
        for j in 0..t
        {
            let mut acc = 0.0;
            for e in 0..d
            {
                acc += q[i * d + e] * k[j * d + e];
            }
            scores[i * t + j] = scale * acc;
        }
    }
    for row in scores.chunks_exact_mut(t)
    {
        let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for x in row.iter_mut()
        {
            *x = (*x - m).exp();
            sum += *x;
        }
        for x in row.iter_mut()
        {
            *x /= sum;
        }
    }
    let mut out = vec![0.0f64; s * d];
    for i in 0..s
    {
        for e in 0..d
        {
            let mut acc = 0.0;
            for j in 0..t
            {
                acc += scores[i * t + j] * v[j * d + e];
            }
            out[i * d + e] = acc;
        }
    }
    out
}

#[test]
fn attention_matches_f64_reference() {
    let mut rng = Lcg(0xA77E_0001);
    for &(s, d, t) in &[(1usize, 1usize, 1usize), (3, 4, 5), (5, 6, 4)]
    {
        let q: Vec<Q16_16> = (0..s * d)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 12))
            .collect();
        let k: Vec<Q16_16> = (0..t * d)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 12))
            .collect();
        let v: Vec<Q16_16> = (0..t * d)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 12))
            .collect();
        let scale = q16(1.0 / (d as f64).sqrt());

        let got = attention(&q, s, d, &k, t, &v, scale);
        let want = attention_ref_f64(
            &to_f64_vec(&q),
            s,
            d,
            &to_f64_vec(&k),
            t,
            &to_f64_vec(&v),
            scale.to_f64(),
        );

        for i in 0..s * d
        {
            let diff = (got[i].to_f64() - want[i]).abs();
            assert!(
                diff <= 1e-2,
                "s={s} d={d} t={t} i={i}: {} vs {}",
                got[i].to_f64(),
                want[i]
            );
        }
    }
}

#[test]
#[should_panic(expected = "attention")]
fn attention_dim_mismatch_panics() {
    let q = [Q16_16::one(); 4]; // annoncé 2×2
    let k = [Q16_16::one(); 3]; // devrait être 2×2 = 4
    let v = [Q16_16::one(); 4];
    let _ = attention(&q, 2, 2, &k, 2, &v, Q16_16::one());
}

/// Référence f64 indépendante pour l'attention causale : softmax sur les
/// clés `0..=i` seulement.
fn causal_attention_ref_f64(
    q: &[f64],
    s: usize,
    d: usize,
    k: &[f64],
    v: &[f64],
    scale: f64,
) -> Vec<f64> {
    let mut out = vec![0.0f64; s * d];
    for i in 0..s
    {
        let t_eff = i + 1;
        let mut row = vec![0.0f64; t_eff];
        for (j, r) in row.iter_mut().enumerate()
        {
            let mut acc = 0.0;
            for e in 0..d
            {
                acc += q[i * d + e] * k[j * d + e];
            }
            *r = scale * acc;
        }
        let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for r in row.iter_mut()
        {
            *r = (*r - m).exp();
            sum += *r;
        }
        for e in 0..d
        {
            let mut acc = 0.0;
            for (j, &r) in row.iter().enumerate()
            {
                acc += r * v[j * d + e];
            }
            out[i * d + e] = acc / sum;
        }
    }
    out
}

#[test]
fn causal_attention_matches_f64_reference() {
    let mut rng = Lcg(0xA77E_0002);
    for &(s, d) in &[(1usize, 1usize), (4, 3), (6, 5)]
    {
        let q: Vec<Q16_16> = (0..s * d)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 12))
            .collect();
        let k: Vec<Q16_16> = (0..s * d)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 12))
            .collect();
        let v: Vec<Q16_16> = (0..s * d)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 12))
            .collect();
        let scale = q16(1.0 / (d as f64).sqrt());

        let got = causal_attention(&q, s, d, &k, &v, scale);
        let want = causal_attention_ref_f64(
            &to_f64_vec(&q),
            s,
            d,
            &to_f64_vec(&k),
            &to_f64_vec(&v),
            scale.to_f64(),
        );
        for i in 0..s * d
        {
            let diff = (got[i].to_f64() - want[i]).abs();
            assert!(
                diff <= 1e-2,
                "s={s} d={d} i={i}: {} vs {}",
                got[i].to_f64(),
                want[i]
            );
        }
    }
}

#[test]
fn causal_attention_first_query_is_value_row_zero() {
    // Requête 0 ne voit que la clé 0 → softmax trivial → out[0] == v[0].
    let (s, d) = (4usize, 3usize);
    let mut rng = Lcg(0xA77E_0003);
    let q: Vec<Q16_16> = (0..s * d)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 10))
        .collect();
    let k: Vec<Q16_16> = (0..s * d)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 10))
        .collect();
    let v: Vec<Q16_16> = (0..s * d)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 10))
        .collect();
    let out = causal_attention(&q, s, d, &k, &v, q16(0.3));
    for e in 0..d
    {
        let diff = (out[e].to_f64() - v[e].to_f64()).abs();
        assert!(diff <= 1e-3, "out[0][{e}] != v[0][{e}]");
    }
}

#[test]
#[should_panic(expected = "causal_attention")]
fn causal_attention_dim_mismatch_panics() {
    let q = [Q16_16::one(); 4]; // annoncé 2×2
    let k = [Q16_16::one(); 3]; // devrait être 2×2 = 4
    let v = [Q16_16::one(); 4];
    let _ = causal_attention(&q, 2, 2, &k, &v, Q16_16::one());
}

#[test]
fn multi_head_attention_matches_per_head() {
    let mut rng = Lcg(0xA77E_0004);
    for &(s, t, h, dh) in &[(3usize, 5usize, 2usize, 4usize), (6, 6, 3, 4)]
    {
        let dm = h * dh;
        let q: Vec<Q16_16> = (0..s * dm)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 12))
            .collect();
        let k: Vec<Q16_16> = (0..t * dm)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 12))
            .collect();
        let v: Vec<Q16_16> = (0..t * dm)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 12))
            .collect();
        let scale = q16(1.0 / (dh as f64).sqrt());

        let got = multi_head_attention(&q, s, t, h, dh, &k, &v, scale, false);

        // Référence : chaque tête via `attention` sur ses colonnes extraites
        // manuellement — même calcul que ce que fait `multi_head_attention`
        // en interne, donc égalité **exacte** attendue (pas une tolérance).
        for head in 0..h
        {
            let off = head * dh;
            let qh: Vec<Q16_16> = (0..s)
                .flat_map(|r| q[r * dm + off..r * dm + off + dh].to_vec())
                .collect();
            let kh: Vec<Q16_16> = (0..t)
                .flat_map(|r| k[r * dm + off..r * dm + off + dh].to_vec())
                .collect();
            let vh: Vec<Q16_16> = (0..t)
                .flat_map(|r| v[r * dm + off..r * dm + off + dh].to_vec())
                .collect();
            let want_h = attention(&qh, s, dh, &kh, t, &vh, scale);
            for r in 0..s
            {
                for e in 0..dh
                {
                    assert_eq!(
                        got[r * dm + off + e],
                        want_h[r * dh + e],
                        "head {head} r={r} e={e}"
                    );
                }
            }
        }
    }
}

#[test]
#[should_panic(expected = "multi_head_attention")]
fn multi_head_attention_causal_requires_equal_s_t_panics() {
    let q = [Q16_16::one(); 8]; // s=2, dm=4 (h=2,dh=2)
    let k = [Q16_16::one(); 12]; // t=3, dm=4 (t != s)
    let v = [Q16_16::one(); 12];
    let _ = multi_head_attention(&q, 2, 3, 2, 2, &k, &v, Q16_16::one(), true);
}

// ------------------------------------------------------------------ //
//  RMSNorm / LayerNorm (déterministe)                                 //
// ------------------------------------------------------------------ //

#[test]
fn rmsnorm_known_small() {
    // x=[1,1,1,1] : moyenne(x²)=1, rms=1 (exact) → y = x/rms · gamma = gamma.
    let x = [1i32, 1, 1, 1].map(Q16_16::from);
    let gamma = [2i32, 3, 4, 5].map(Q16_16::from);
    let y = rmsnorm(&x, 1, 4, &gamma, Q16_16::zero()).unwrap();
    assert_eq!(y, gamma);
}

fn rmsnorm_ref_f64(x: &[f64], rows: usize, d: usize, gamma: &[f64], eps: f64) -> Vec<f64> {
    let mut y = vec![0.0f64; rows * d];
    for r in 0..rows
    {
        let row = &x[r * d..r * d + d];
        let ms: f64 = row.iter().map(|v| v * v).sum::<f64>() / d as f64;
        let rms = (ms + eps).sqrt();
        for i in 0..d
        {
            y[r * d + i] = row[i] / rms * gamma[i];
        }
    }
    y
}

#[test]
fn rmsnorm_matches_f64_reference() {
    let mut rng = Lcg(0x2222_0001);
    for &(rows, d) in &[(1usize, 1usize), (3, 5), (4, 8)]
    {
        let x: Vec<Q16_16> = (0..rows * d)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 8))
            .collect();
        let gamma: Vec<Q16_16> = (0..d)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 10))
            .collect();
        let eps = q16(1e-3);
        let got = rmsnorm(&x, rows, d, &gamma, eps).expect("rms + eps > 0");
        let want = rmsnorm_ref_f64(&to_f64_vec(&x), rows, d, &to_f64_vec(&gamma), eps.to_f64());
        for i in 0..rows * d
        {
            let diff = (got[i].to_f64() - want[i]).abs();
            assert!(
                diff <= 1e-2,
                "rows={rows} d={d} i={i}: {} vs {}",
                got[i].to_f64(),
                want[i]
            );
        }
    }
}

#[test]
fn rmsnorm_zero_row_and_eps_returns_none() {
    // Ligne nulle, eps=0 → rms=0 → division indéfinie.
    let x = [Q16_16::zero(); 4];
    let gamma = [Q16_16::one(); 4];
    assert!(rmsnorm(&x, 1, 4, &gamma, Q16_16::zero()).is_none());
}

#[test]
fn rmsnorm_i64_storage() {
    let x = [3i64, 4].map(Q32_32::from);
    let gamma = [Q32_32::from(1i64); 2];
    let y = rmsnorm(&x, 1, 2, &gamma, Q32_32::zero()).unwrap();
    let rms = 12.5f64.sqrt(); // moyenne(9+16)/2 = 12.5
    assert!((y[0].to_f64() - 3.0 / rms).abs() < 1e-6);
    assert!((y[1].to_f64() - 4.0 / rms).abs() < 1e-6);
}

#[test]
#[should_panic(expected = "rmsnorm")]
fn rmsnorm_dim_mismatch_panics() {
    let x = vec![Q16_16::one(); 5]; // annoncé 1×5
    let gamma = vec![Q16_16::one(); 4]; // devrait être 5
    let _ = rmsnorm(&x, 1, 5, &gamma, Q16_16::zero());
}

#[test]
fn layer_norm_known_small() {
    // x=[0,0,4,4] : moyenne=2, variance=4 (exact), écart-type=2 (eps=0) →
    // y=(x-2)/2 = [-1,-1,1,1].
    let x = [0i32, 0, 4, 4].map(Q16_16::from);
    let gamma = [Q16_16::one(); 4];
    let beta = [Q16_16::zero(); 4];
    let y = layer_norm(&x, 1, 4, &gamma, &beta, Q16_16::zero()).unwrap();
    assert_eq!(y, [-1i32, -1, 1, 1].map(Q16_16::from));
}

fn layer_norm_ref_f64(
    x: &[f64],
    rows: usize,
    d: usize,
    gamma: &[f64],
    beta: &[f64],
    eps: f64,
) -> Vec<f64> {
    let mut y = vec![0.0f64; rows * d];
    for r in 0..rows
    {
        let row = &x[r * d..r * d + d];
        let mean: f64 = row.iter().sum::<f64>() / d as f64;
        let var: f64 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / d as f64;
        let denom = (var + eps).sqrt();
        for i in 0..d
        {
            y[r * d + i] = (row[i] - mean) / denom * gamma[i] + beta[i];
        }
    }
    y
}

#[test]
fn layer_norm_matches_f64_reference() {
    let mut rng = Lcg(0x2222_0002);
    for &(rows, d) in &[(1usize, 1usize), (3, 5), (4, 8)]
    {
        let x: Vec<Q16_16> = (0..rows * d)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 8))
            .collect();
        let gamma: Vec<Q16_16> = (0..d)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 10))
            .collect();
        let beta: Vec<Q16_16> = (0..d)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 10))
            .collect();
        let eps = q16(1e-3);
        let got = layer_norm(&x, rows, d, &gamma, &beta, eps).expect("var + eps > 0");
        let want = layer_norm_ref_f64(
            &to_f64_vec(&x),
            rows,
            d,
            &to_f64_vec(&gamma),
            &to_f64_vec(&beta),
            eps.to_f64(),
        );
        for i in 0..rows * d
        {
            let diff = (got[i].to_f64() - want[i]).abs();
            assert!(
                diff <= 1e-2,
                "rows={rows} d={d} i={i}: {} vs {}",
                got[i].to_f64(),
                want[i]
            );
        }
    }
}

#[test]
fn layer_norm_zero_variance_returns_none() {
    // Ligne constante, eps=0 → variance nulle → division indéfinie.
    let x = [Q16_16::from(5); 4];
    let gamma = [Q16_16::one(); 4];
    let beta = [Q16_16::zero(); 4];
    assert!(layer_norm(&x, 1, 4, &gamma, &beta, Q16_16::zero()).is_none());
}

#[test]
fn layer_norm_i64_storage() {
    let x = [0i64, 0, 4, 4].map(Q32_32::from);
    let gamma = [Q32_32::from(1i64); 4];
    let beta = [Q32_32::zero(); 4];
    let y = layer_norm(&x, 1, 4, &gamma, &beta, Q32_32::zero()).unwrap();
    assert_eq!(y, [-1i64, -1, 1, 1].map(Q32_32::from));
}

#[test]
#[should_panic(expected = "layer_norm")]
fn layer_norm_dim_mismatch_panics() {
    let x = vec![Q16_16::one(); 4];
    let gamma = vec![Q16_16::one(); 4];
    let beta = vec![Q16_16::one(); 3]; // devrait être 4
    let _ = layer_norm(&x, 1, 4, &gamma, &beta, Q16_16::zero());
}

// ------------------------------------------------------------------ //
//  RoPE (rotary positional embedding), déterministe                   //
// ------------------------------------------------------------------ //

/// Référence f64 indépendante : mêmes conventions que `rope_apply`.
fn rope_ref_f64(x: &[f64], rows: usize, d: usize, base: f64, pos_offset: usize) -> Vec<f64> {
    let half = d / 2;
    let mut y = x.to_vec();
    for r in 0..rows
    {
        let pos = (pos_offset + r) as f64;
        let row = &mut y[r * d..r * d + d];
        for i in 0..half
        {
            let theta = base.powf(-2.0 * i as f64 / d as f64);
            let angle = pos * theta;
            let (s, c) = angle.sin_cos();
            let a = row[2 * i];
            let b = row[2 * i + 1];
            row[2 * i] = a * c - b * s;
            row[2 * i + 1] = a * s + b * c;
        }
    }
    y
}

#[test]
fn rope_matches_f64_reference() {
    let mut rng = Lcg(0x3333_0001);
    for &(rows, d, pos_offset) in &[(1usize, 2usize, 0usize), (5, 8, 0), (3, 16, 5)]
    {
        let mut x: Vec<Q16_16> = (0..rows * d)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 8))
            .collect();
        let base = q16(10000.0);
        let x0 = to_f64_vec(&x);
        rope_apply(&mut x, rows, d, base, pos_offset);
        let want = rope_ref_f64(&x0, rows, d, base.to_f64(), pos_offset);
        for i in 0..rows * d
        {
            let diff = (x[i].to_f64() - want[i]).abs();
            assert!(
                diff <= 1e-2,
                "rows={rows} d={d} pos_offset={pos_offset} i={i}: {} vs {}",
                x[i].to_f64(),
                want[i]
            );
        }
    }
}

#[test]
fn rope_preserves_pair_norm() {
    // Une rotation conserve la norme de chaque paire — invariant structurel,
    // indépendant de toute référence externe.
    let mut rng = Lcg(0x3333_0002);
    let (rows, d) = (4usize, 8usize);
    let mut x: Vec<Q16_16> = (0..rows * d)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 8))
        .collect();
    let before = to_f64_vec(&x);
    rope_apply(&mut x, rows, d, q16(10000.0), 7);
    for r in 0..rows
    {
        for i in 0..d / 2
        {
            let (a0, b0) = (before[r * d + 2 * i], before[r * d + 2 * i + 1]);
            let norm0 = (a0 * a0 + b0 * b0).sqrt();
            let (a1, b1) = (x[r * d + 2 * i].to_f64(), x[r * d + 2 * i + 1].to_f64());
            let norm1 = (a1 * a1 + b1 * b1).sqrt();
            assert!(
                (norm0 - norm1).abs() <= 1e-2,
                "r={r} i={i}: norme {norm0} → {norm1}"
            );
        }
    }
}

#[test]
fn rope_position_zero_is_identity() {
    // La ligne à la position 0 (angle nul pour toutes les paires) doit
    // rester inchangée.
    let mut rng = Lcg(0x3333_0003);
    let (rows, d) = (3usize, 8usize);
    let x: Vec<Q16_16> = (0..rows * d)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 8))
        .collect();
    let mut got = x.clone();
    rope_apply(&mut got, rows, d, q16(10000.0), 0);
    for i in 0..d
    {
        let diff = (got[i].to_f64() - x[i].to_f64()).abs();
        assert!(
            diff <= 1e-3,
            "position 0 : composante {i} modifiée ({} vs {})",
            got[i].to_f64(),
            x[i].to_f64()
        );
    }
}

#[test]
#[should_panic(expected = "rope_apply")]
fn rope_apply_dim_mismatch_panics() {
    let mut x = vec![Q16_16::zero(); 7]; // ni 1×8 ni aucun multiple valide de 8
    rope_apply(&mut x, 1, 8, q16(10000.0), 0);
}

#[test]
#[should_panic(expected = "rope_apply")]
fn rope_apply_odd_d_panics() {
    let mut x = vec![Q16_16::zero(); 6]; // 2×3 : d=3 impair
    rope_apply(&mut x, 2, 3, q16(10000.0), 0);
}

// ------------------------------------------------------------------ //
//  Cache KV (décodage autoregressif incrémental), déterministe        //
// ------------------------------------------------------------------ //

#[test]
fn kv_cache_incremental_matches_batched_causal() {
    let mut rng = Lcg(0x4444_0001);
    for &(s, h, dh) in &[(1usize, 1usize, 2usize), (4, 2, 4), (10, 3, 6)]
    {
        let dm = h * dh;
        let q: Vec<Q16_16> = (0..s * dm)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 12))
            .collect();
        let k: Vec<Q16_16> = (0..s * dm)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 12))
            .collect();
        let v: Vec<Q16_16> = (0..s * dm)
            .map(|_| Q16_16::from_raw(rng.raw_i32() >> 12))
            .collect();
        let scale = q16(1.0 / (dh as f64).sqrt());

        // Référence : attention causale multi-tête en bloc.
        let expected = multi_head_attention(&q, s, s, h, dh, &k, &v, scale, true);

        // Incrémental : empile puis décode chaque token.
        let mut cache: KvCache<16> = KvCache::new(s, dm);
        let mut got = Vec::with_capacity(s * dm);
        for i in 0..s
        {
            cache.append(&k[i * dm..i * dm + dm], &v[i * dm..i * dm + dm]);
            got.extend(cache.decode_step(&q[i * dm..i * dm + dm], h, dh, scale));
        }

        // Bit-exact (pas une tolérance) : contrairement au flottant, la
        // somme virgule fixe est exacte et associative — nourrir le cache
        // incrémentalement ou calculer en bloc effectue exactement la même
        // séquence d'opérations sur les mêmes données.
        assert_eq!(
            got, expected,
            "s={s} h={h} dh={dh} : incrémental doit être bit-exact vis-à-vis du batch causal"
        );
    }
}

#[test]
fn kv_cache_first_token_attends_to_itself() {
    // Un seul token en cache → softmax trivial (un seul score) → out == v.
    let (h, dh) = (2usize, 3usize);
    let dm = h * dh;
    let mut rng = Lcg(0x4444_0002);
    let q: Vec<Q16_16> = (0..dm)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 10))
        .collect();
    let k: Vec<Q16_16> = (0..dm)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 10))
        .collect();
    let v: Vec<Q16_16> = (0..dm)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 10))
        .collect();
    let mut cache: KvCache<16> = KvCache::new(4, dm);
    cache.append(&k, &v);
    let out = cache.decode_step(&q, h, dh, q16(0.3));
    assert_eq!(out, v);
}

#[test]
fn kv_cache_decode_step_empty_returns_zero() {
    let cache: KvCache<16> = KvCache::new(3, 4);
    let q = [q16(1.0); 4];
    let out = cache.decode_step(&q, 2, 2, q16(0.5));
    assert_eq!(out, vec![Q16_16::zero(); 4]);
}

#[test]
fn kv_cache_len_clear_and_capacity() {
    let mut cache: KvCache<16> = KvCache::new(3, 4);
    assert!(cache.is_empty());
    assert_eq!(cache.capacity(), 3);
    cache.append(&[Q16_16::one(); 4], &[q16(5.0); 4]);
    assert_eq!(cache.len(), 1);
    cache.clear();
    assert!(cache.is_empty());
}

#[test]
#[should_panic(expected = "KvCache::append")]
fn kv_cache_append_dim_mismatch_panics() {
    let mut cache: KvCache<16> = KvCache::new(4, 6);
    let k = vec![Q16_16::zero(); 5]; // devrait être 6
    let v = vec![Q16_16::zero(); 6];
    cache.append(&k, &v);
}

#[test]
#[should_panic(expected = "cache plein")]
fn kv_cache_append_beyond_capacity_panics() {
    let mut cache: KvCache<16> = KvCache::new(1, 2);
    cache.append(&[Q16_16::zero(); 2], &[Q16_16::zero(); 2]);
    cache.append(&[Q16_16::zero(); 2], &[Q16_16::zero(); 2]); // cache plein (cap=1)
}

#[test]
#[should_panic(expected = "KvCache::decode_step")]
fn kv_cache_decode_step_dim_mismatch_panics() {
    let mut cache: KvCache<16> = KvCache::new(4, 6);
    cache.append(&[Q16_16::zero(); 6], &[Q16_16::zero(); 6]);
    let q = vec![Q16_16::zero(); 5]; // devrait être 6
    let _ = cache.decode_step(&q, 2, 3, Q16_16::one());
}

// ------------------------------------------------------------------ //
//  Bloc Transformer décodeur pre-norm (assemblage complet)             //
// ------------------------------------------------------------------ //

/// Génère des coefficients déterministes bornés (`[-0.5, 0.5]`), même
/// construction que le test scalaire de [`crate::transformer`] (module
/// flottant).
fn mk_f64(n: usize, seed: f64) -> Vec<f64> {
    (0..n)
        .map(|i| ((i as f64 + seed) * 0.017).sin() * 0.5)
        .collect()
}

/// Référence f64 indépendante de `y = W·x + b` (`W` : `out×in` row-major,
/// même convention que [`Linear`]).
fn linear_ref_f64(
    x: &[f64],
    rows: usize,
    in_f: usize,
    w: &[f64],
    out_f: usize,
    b: &[f64],
) -> Vec<f64> {
    let mut y = vec![0.0f64; rows * out_f];
    for r in 0..rows
    {
        for o in 0..out_f
        {
            let mut acc = b[o];
            for i in 0..in_f
            {
                acc += x[r * in_f + i] * w[o * in_f + i];
            }
            y[r * out_f + o] = acc;
        }
    }
    y
}

/// [`rope_ref_f64`] appliquée indépendamment à chaque tête (cf.
/// [`rope_apply_heads`]).
fn rope_apply_heads_ref_f64(
    x: &[f64],
    s: usize,
    h: usize,
    dh: usize,
    base: f64,
    pos_offset: usize,
) -> Vec<f64> {
    let dm = h * dh;
    let mut y = x.to_vec();
    for head in 0..h
    {
        let off = head * dh;
        let mut head_x = vec![0.0f64; s * dh];
        for r in 0..s
        {
            head_x[r * dh..r * dh + dh].copy_from_slice(&x[r * dm + off..r * dm + off + dh]);
        }
        let head_y = rope_ref_f64(&head_x, s, dh, base, pos_offset);
        for r in 0..s
        {
            y[r * dm + off..r * dm + off + dh].copy_from_slice(&head_y[r * dh..r * dh + dh]);
        }
    }
    y
}

fn silu_ref_f64(x: f64) -> f64 {
    x * (1.0 / (1.0 + (-x).exp()))
}

/// Attention causale multi-tête, référence f64 indépendante (même algorithme
/// que [`multi_head_attention`] avec `causal = true`, écrit séparément).
fn mha_causal_ref_f64(q: &[f64], k: &[f64], v: &[f64], s: usize, h: usize, dh: usize) -> Vec<f64> {
    let dm = h * dh;
    let scale = 1.0 / (dh as f64).sqrt();
    let mut out = vec![0.0f64; s * dm];
    for hh in 0..h
    {
        let off = hh * dh;
        for i in 0..s
        {
            let mut row = vec![0.0f64; i + 1];
            for (j, r) in row.iter_mut().enumerate()
            {
                let mut acc = 0.0;
                for e in 0..dh
                {
                    acc += q[i * dm + off + e] * k[j * dm + off + e];
                }
                *r = scale * acc;
            }
            let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0;
            for r in row.iter_mut()
            {
                *r = (*r - m).exp();
                sum += *r;
            }
            for e in 0..dh
            {
                let mut acc = 0.0;
                for (j, &p) in row.iter().enumerate()
                {
                    acc += p * v[j * dm + off + e];
                }
                out[i * dm + off + e] = acc / sum;
            }
        }
    }
    out
}

/// Référence f64 indépendante du bloc décodeur complet, dans le même ordre
/// exact d'opérations que [`TransformerBlock::forward`].
#[allow(clippy::too_many_arguments)]
fn transformer_forward_ref_f64(
    x0: &[f64],
    s: usize,
    d: usize,
    h: usize,
    dff: usize,
    wq: &[f64],
    wk: &[f64],
    wv: &[f64],
    wo: &[f64],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    norm1: &[f64],
    norm2: &[f64],
    eps: f64,
    base: f64,
) -> Vec<f64> {
    let dh = d / h;
    let zero_d = vec![0.0f64; d];
    let mut x = x0.to_vec();

    let hn = rmsnorm_ref_f64(&x, s, d, norm1, eps);
    let mut q = linear_ref_f64(&hn, s, d, wq, d, &zero_d);
    let mut k = linear_ref_f64(&hn, s, d, wk, d, &zero_d);
    let v = linear_ref_f64(&hn, s, d, wv, d, &zero_d);
    q = rope_apply_heads_ref_f64(&q, s, h, dh, base, 0);
    k = rope_apply_heads_ref_f64(&k, s, h, dh, base, 0);
    let attn = mha_causal_ref_f64(&q, &k, &v, s, h, dh);
    let o = linear_ref_f64(&attn, s, d, wo, d, &zero_d);
    for i in 0..s * d
    {
        x[i] += o[i];
    }

    let hn2 = rmsnorm_ref_f64(&x, s, d, norm2, eps);
    let mut f1 = linear_ref_f64(&hn2, s, d, w1, dff, b1);
    for v in f1.iter_mut()
    {
        *v = silu_ref_f64(*v);
    }
    let f2 = linear_ref_f64(&f1, s, dff, w2, d, &zero_d);
    for i in 0..s * d
    {
        x[i] += f2[i];
    }
    x
}

/// Construit un [`TransformerBlock`] de test (`d=8`, `h=2`, `dff=16`, biais
/// nul sauf `b1`, gains de normalisation croissants) et son vecteur d'entrée
/// `x0`, en f64 **et** en `Q16_16` — pour comparer bloc fixe et référence f64.
fn build_test_block(
    causal: bool,
) -> (
    TransformerBlock<16>,
    Vec<f64>,
    Vec<Q16_16>,
    usize,
    usize,
    usize,
    usize,
) {
    let (s, d, h, dff) = (6usize, 8usize, 2usize, 16usize);
    let eps = 1e-3;
    let base = 10000.0;

    let wq_f = mk_f64(d * d, 1.0);
    let wk_f = mk_f64(d * d, 2.0);
    let wv_f = mk_f64(d * d, 3.0);
    let wo_f = mk_f64(d * d, 4.0);
    let w1_f = mk_f64(d * dff, 5.0);
    let b1_f = mk_f64(dff, 6.0);
    let w2_f = mk_f64(dff * d, 7.0);
    let norm1_f: Vec<f64> = (0..d).map(|i| 1.0 + i as f64 * 0.01).collect();
    let norm2_f: Vec<f64> = (0..d).map(|i| 0.9 + i as f64 * 0.02).collect();
    let x0_f: Vec<f64> = (0..s * d).map(|i| (i as f64 * 0.05).cos()).collect();

    let to_q = |v: &[f64]| -> Vec<Q16_16> { v.iter().map(|&x| q16(x)).collect() };
    let block = TransformerBlock::new(
        d,
        h,
        dff,
        Linear::new(to_q(&wq_f), vec![Q16_16::zero(); d], d, d),
        Linear::new(to_q(&wk_f), vec![Q16_16::zero(); d], d, d),
        Linear::new(to_q(&wv_f), vec![Q16_16::zero(); d], d, d),
        Linear::new(to_q(&wo_f), vec![Q16_16::zero(); d], d, d),
        Linear::new(to_q(&w1_f), to_q(&b1_f), dff, d),
        Linear::new(to_q(&w2_f), vec![Q16_16::zero(); d], d, dff),
        to_q(&norm1_f),
        to_q(&norm2_f),
        q16(eps),
        q16(base),
        causal,
    );
    let x0_q = to_q(&x0_f);
    (block, x0_f, x0_q, s, d, h, dff)
}

#[test]
fn transformer_block_matches_f64_reference() {
    let (block, x0_f, x0_q, s, d, h, dff) = build_test_block(true);

    let mut got = x0_q.clone();
    block.forward(&mut got, s).expect("rmsnorm bien défini");

    let want = transformer_forward_ref_f64(
        &x0_f,
        s,
        d,
        h,
        dff,
        &mk_f64(d * d, 1.0),
        &mk_f64(d * d, 2.0),
        &mk_f64(d * d, 3.0),
        &mk_f64(d * d, 4.0),
        &mk_f64(d * dff, 5.0),
        &mk_f64(dff, 6.0),
        &mk_f64(dff * d, 7.0),
        &(0..d).map(|i| 1.0 + i as f64 * 0.01).collect::<Vec<_>>(),
        &(0..d).map(|i| 0.9 + i as f64 * 0.02).collect::<Vec<_>>(),
        1e-3,
        10000.0,
    );

    for i in 0..s * d
    {
        let diff = (got[i].to_f64() - want[i]).abs();
        let tol = 5e-2 * (1.0 + want[i].abs());
        assert!(
            diff <= tol,
            "i={i}: bloc fixe {} vs référence f64 {} (écart {diff}, tolérance {tol})",
            got[i].to_f64(),
            want[i]
        );
    }
}

#[test]
fn decode_matches_prefill_bit_exact() {
    // Comme pour KvCache seul (`kv_cache_incremental_matches_batched_causal`),
    // décoder un token à la fois via `forward_decode` + `KvCache` reproduit
    // **bit-à-bit** (pas à une tolérance près) le préremplissage `forward`
    // causal sur toute la séquence : chaque brique composée (RMSNorm, Linear,
    // RoPE, attention via KvCache) est locale à sa ligne / sa position
    // absolue, jamais du regroupement en lot — garantie strictement plus
    // forte que le module flottant `crate::transformer` (tolérance `2e-3`).
    let (block, _x0_f, x0_q, s, d, _h, _dff) = build_test_block(true);

    let mut prefill = x0_q.clone();
    block.forward(&mut prefill, s).expect("rmsnorm bien défini");

    let mut cache: KvCache<16> = KvCache::new(s, d);
    let mut decoded = vec![Q16_16::zero(); s * d];
    for t in 0..s
    {
        let mut row = x0_q[t * d..t * d + d].to_vec();
        block
            .forward_decode(&mut row, t, &mut cache)
            .expect("rmsnorm bien défini");
        decoded[t * d..t * d + d].copy_from_slice(&row);
    }
    assert_eq!(cache.len(), s);

    assert_eq!(
        decoded, prefill,
        "décodage incrémental doit être bit-exact vis-à-vis du préremplissage causal"
    );
}

#[test]
fn rope_apply_heads_matches_per_head_f64_reference() {
    let (s, h, dh) = (4usize, 3usize, 4usize);
    let dm = h * dh;
    let mut rng = Lcg(0x5555_0001);
    let mut x: Vec<Q16_16> = (0..s * dm)
        .map(|_| Q16_16::from_raw(rng.raw_i32() >> 8))
        .collect();
    let x0 = to_f64_vec(&x);
    rope_apply_heads(&mut x, s, h, dh, q16(10000.0), 0);
    let want = rope_apply_heads_ref_f64(&x0, s, h, dh, 10000.0, 0);
    for i in 0..s * dm
    {
        let diff = (x[i].to_f64() - want[i]).abs();
        assert!(diff <= 1e-2, "i={i}: {} vs {}", x[i].to_f64(), want[i]);
    }
}

#[test]
#[should_panic(expected = "d_model")]
fn transformer_block_new_rejects_non_divisible_heads() {
    let d = 8;
    let zero = vec![Q16_16::zero(); d];
    let _ = TransformerBlock::<16>::new(
        d,
        3, // 8 non divisible par 3
        16,
        Linear::new(vec![Q16_16::zero(); d * d], zero.clone(), d, d),
        Linear::new(vec![Q16_16::zero(); d * d], zero.clone(), d, d),
        Linear::new(vec![Q16_16::zero(); d * d], zero.clone(), d, d),
        Linear::new(vec![Q16_16::zero(); d * d], zero.clone(), d, d),
        Linear::new(
            vec![Q16_16::zero(); d * 16],
            vec![Q16_16::zero(); 16],
            16,
            d,
        ),
        Linear::new(vec![Q16_16::zero(); 16 * d], zero.clone(), d, 16),
        zero.clone(),
        zero,
        q16(1e-3),
        q16(10000.0),
        true,
    );
}

#[test]
#[should_panic(expected = "wq.in_features")]
fn transformer_block_new_rejects_wrong_projection_shape() {
    let d = 8;
    let zero = vec![Q16_16::zero(); d];
    let _ = TransformerBlock::<16>::new(
        d,
        2,
        16,
        Linear::new(vec![Q16_16::zero(); 4 * 4], vec![Q16_16::zero(); 4], 4, 4), // devrait être d×d
        Linear::new(vec![Q16_16::zero(); d * d], zero.clone(), d, d),
        Linear::new(vec![Q16_16::zero(); d * d], zero.clone(), d, d),
        Linear::new(vec![Q16_16::zero(); d * d], zero.clone(), d, d),
        Linear::new(
            vec![Q16_16::zero(); d * 16],
            vec![Q16_16::zero(); 16],
            16,
            d,
        ),
        Linear::new(vec![Q16_16::zero(); 16 * d], zero.clone(), d, 16),
        zero.clone(),
        zero,
        q16(1e-3),
        q16(10000.0),
        true,
    );
}

#[test]
#[should_panic(expected = "TransformerBlock::forward")]
fn transformer_block_forward_dim_mismatch_panics() {
    let (block, _x0_f, _x0_q, s, d, _h, _dff) = build_test_block(true);
    let mut x = vec![Q16_16::zero(); s * d - 1]; // longueur incorrecte
    let _ = block.forward(&mut x, s);
}

// ------------------------------------------------------------------ //
//  Modèle Transformer multi-couche (pile de TransformerBlock)          //
// ------------------------------------------------------------------ //

/// Construit un [`TransformerBlock`] de test (`d=8`, `h=2`, `dff=16`, causal)
/// avec des poids décalés par `seed_base` — pour empiler plusieurs couches
/// distinctes dans un [`TransformerModel`] (même dimension modèle, poids
/// différents, comme des couches réellement entraînées indépendamment).
fn build_fixed_block_seeded(seed_base: f64) -> TransformerBlock<16> {
    let (d, dff) = (8usize, 16usize);
    // Poids réduits d'un facteur 10 par rapport à `mk_f64` (utilisé pour le
    // test à un seul bloc) : empilés sur plusieurs couches **et** réinjectés
    // à travers plusieurs pas de génération (`generate_hidden`), le flux
    // résiduel non normalisé par gain (poids non entraînés) croît sinon assez
    // vite pour déborder la plage `Q16.16` (~±32768) après quelques passes.
    let to_q = |v: Vec<f64>| -> Vec<Q16_16> { v.into_iter().map(|x| q16(x * 0.1)).collect() };
    TransformerBlock::new(
        d,
        2,
        dff,
        Linear::new(
            to_q(mk_f64(d * d, seed_base + 1.0)),
            vec![Q16_16::zero(); d],
            d,
            d,
        ),
        Linear::new(
            to_q(mk_f64(d * d, seed_base + 2.0)),
            vec![Q16_16::zero(); d],
            d,
            d,
        ),
        Linear::new(
            to_q(mk_f64(d * d, seed_base + 3.0)),
            vec![Q16_16::zero(); d],
            d,
            d,
        ),
        Linear::new(
            to_q(mk_f64(d * d, seed_base + 4.0)),
            vec![Q16_16::zero(); d],
            d,
            d,
        ),
        Linear::new(
            to_q(mk_f64(d * dff, seed_base + 5.0)),
            to_q(mk_f64(dff, seed_base + 6.0)),
            dff,
            d,
        ),
        Linear::new(
            to_q(mk_f64(dff * d, seed_base + 7.0)),
            vec![Q16_16::zero(); d],
            d,
            dff,
        ),
        to_q((0..d).map(|i| 1.0 + i as f64 * 0.01).collect()),
        to_q((0..d).map(|i| 0.9 + i as f64 * 0.02).collect()),
        q16(1e-3),
        q16(10000.0),
        true,
    )
}

#[test]
fn transformer_model_stack_decode_matches_prefill_bit_exact() {
    // Comme pour un bloc seul (`decode_matches_prefill_bit_exact`), décoder
    // token par token à travers toute la pile (un cache par couche)
    // reproduit **bit-à-bit** le préremplissage causal en bloc de toute la
    // pile : propriété propagée sans perte depuis `fixed::transformer`, cf.
    // en-tête de module.
    let (s, d, n_layers) = (6usize, 8usize, 3usize);
    let blocks: Vec<_> = (0..n_layers)
        .map(|l| build_fixed_block_seeded(l as f64 * 10.0))
        .collect();
    let model = TransformerModel::new(blocks);
    assert_eq!(model.n_layers(), n_layers);
    assert_eq!(model.d_model(), d);

    let x0: Vec<Q16_16> = (0..s * d).map(|i| q16((i as f64 * 0.05).cos())).collect();

    let mut prefill = x0.clone();
    model.prefill(&mut prefill, s).expect("rmsnorm bien défini");

    let mut caches = model.new_caches(s);
    let mut decoded = vec![Q16_16::zero(); s * d];
    for t in 0..s
    {
        let mut row = x0[t * d..t * d + d].to_vec();
        model
            .decode_step(&mut row, t, &mut caches)
            .expect("rmsnorm bien défini");
        decoded[t * d..t * d + d].copy_from_slice(&row);
    }

    assert_eq!(
        decoded, prefill,
        "décodage empilé doit être bit-exact vis-à-vis du préremplissage"
    );
}

#[test]
fn transformer_model_generate_hidden_is_deterministic() {
    let (d, n_layers) = (8usize, 2usize);
    let blocks: Vec<_> = (0..n_layers)
        .map(|l| build_fixed_block_seeded(l as f64 * 5.0))
        .collect();
    let model = TransformerModel::new(blocks);

    let (prompt_len, n_new) = (3usize, 4usize);
    let prompt: Vec<Q16_16> = (0..prompt_len * d)
        .map(|i| q16((i as f64 * 0.1).sin()))
        .collect();

    let a = model
        .generate_hidden(&prompt, prompt_len, n_new)
        .expect("rmsnorm bien défini");
    assert_eq!(a.len(), n_new * d);

    // Déterministe par construction : deux exécutions identiques.
    let b = model
        .generate_hidden(&prompt, prompt_len, n_new)
        .expect("rmsnorm bien défini");
    assert_eq!(a, b, "génération non déterministe");
}

#[test]
#[should_panic(expected = "au moins un bloc")]
fn transformer_model_new_rejects_empty() {
    let _: TransformerModel<16> = TransformerModel::new(Vec::new());
}

#[test]
#[should_panic(expected = "d_model doit être homogène")]
fn transformer_model_new_rejects_heterogeneous_d_model() {
    let b1 = build_fixed_block_seeded(0.0); // d_model = 8
    let (d2, dff2) = (4usize, 8usize);
    let zero2 = vec![Q16_16::zero(); d2];
    let b2 = TransformerBlock::new(
        d2,
        2,
        dff2,
        Linear::new(vec![Q16_16::zero(); d2 * d2], zero2.clone(), d2, d2),
        Linear::new(vec![Q16_16::zero(); d2 * d2], zero2.clone(), d2, d2),
        Linear::new(vec![Q16_16::zero(); d2 * d2], zero2.clone(), d2, d2),
        Linear::new(vec![Q16_16::zero(); d2 * d2], zero2.clone(), d2, d2),
        Linear::new(
            vec![Q16_16::zero(); d2 * dff2],
            vec![Q16_16::zero(); dff2],
            dff2,
            d2,
        ),
        Linear::new(vec![Q16_16::zero(); dff2 * d2], zero2.clone(), d2, dff2),
        zero2.clone(),
        zero2,
        q16(1e-3),
        q16(10000.0),
        true,
    );
    let _ = TransformerModel::new(vec![b1, b2]);
}

#[test]
#[should_panic(expected = "cache(s) fourni(s)")]
fn transformer_model_decode_step_wrong_cache_count_panics() {
    let blocks: Vec<_> = (0..2)
        .map(|l| build_fixed_block_seeded(l as f64 * 5.0))
        .collect();
    let model = TransformerModel::new(blocks);
    let d = model.d_model();
    let mut caches = model.new_caches(4);
    caches.pop(); // 1 seul cache pour 2 couches
    let mut row = vec![Q16_16::zero(); d];
    let _ = model.decode_step(&mut row, 0, &mut caches);
}
