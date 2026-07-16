// scirust-simd/src/fixed/tests.rs
//
// Batterie de validation du sous-système virgule fixe. Tous les tests sont
// **indépendants de l'architecture** (aucune dépendance au matériel SIMD :
// std::simd produit les mêmes bits partout). On combine :
//  * assertions **exactes** sur des cas construits (arrondi, overflow, bits) ;
//  * comparaison à une référence `f64` à quelques ULP pour mul/div/math ;
//  * égalité stricte **SIMD == scalaire**.

use super::math::{reciprocal, rsqrt, sqrt};
use super::reductions as red;
use super::simd::{FixedI32x8, FixedI64x4};
use super::{
    FixedI32, FixedI64, NumericScalar, OverflowMode, Q8_24, Q16_16, Q24_8, Q32_32, RoundingMode,
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
