// scirust-simd/src/fixed/tests.rs
//
// Batterie de validation du sous-système virgule fixe. Tous les tests sont
// **indépendants de l'architecture** (aucune dépendance au matériel SIMD :
// std::simd produit les mêmes bits partout). On combine :
//  * assertions **exactes** sur des cas construits (arrondi, overflow, bits) ;
//  * comparaison à une référence `f64` à quelques ULP pour mul/div/math ;
//  * égalité stricte **SIMD == scalaire**.

use super::activation as act;
use super::conv::{Conv1dShape, conv1d, conv1d_batch};
use super::conv2d::{Conv2dShape, conv2d, conv2d_batch};
use super::layer::Linear;
use super::linalg;
use super::math::{reciprocal, rsqrt, sqrt};
use super::pool::{Pool1dShape, avg_pool1d, max_pool1d};
use super::pool2d::{Pool2dShape, avg_pool2d, max_pool2d};
use super::reductions as red;
use super::rescale::{rescale, rescale_saturating, rescale_wrapping};
use super::simd::{FixedI16x8, FixedI32x8, FixedI64x4};
use super::transcendental as tr;
use super::{
    FixedI16, FixedI32, FixedI64, NumericScalar, OverflowMode, Q1_15, Q8_8, Q8_24, Q16_16, Q24_8,
    Q32_32, RealScalar, RoundingMode,
};

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
