// scirust-simd/src/fixed/tests.rs
//
// Batterie de validation du sous-système virgule fixe. Tous les tests sont
// **indépendants de l'architecture** (aucune dépendance au matériel SIMD :
// std::simd produit les mêmes bits partout). On combine :
//  * assertions **exactes** sur des cas construits (arrondi, overflow, bits) ;
//  * comparaison à une référence `f64` à quelques ULP pour mul/div/math ;
//  * égalité stricte **SIMD == scalaire**.

use super::linalg;
use super::math::{reciprocal, rsqrt, sqrt};
use super::reductions as red;
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

    // dot([1,2,3],[4,5,6]) = 32.
    let x: Vec<Q16_16> = [1.0, 2.0, 3.0].iter().map(|&v| q16(v)).collect();
    let y: Vec<Q16_16> = [4.0, 5.0, 6.0].iter().map(|&v| q16(v)).collect();
    assert_eq!(red::dot(&x, &y).to_f64(), 32.0);
    // ‖[3,4]‖ = 5.
    let v: Vec<Q16_16> = [3.0, 4.0].iter().map(|&t| q16(t)).collect();
    assert_eq!(red::l2_norm(&v).to_f64(), 5.0);
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
