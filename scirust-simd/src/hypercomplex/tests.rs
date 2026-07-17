// scirust-simd/src/hypercomplex/tests.rs
//
// Validation mathématique rigoureuse des algèbres hypercomplexes SIMD.
//
// Stratégie : tous les vecteurs de test sont à coefficients ENTIERS
// petits. Les produits hypercomplexes n'impliquent alors que des entiers
// très inférieurs à 2²⁴, exactement représentables en f32 — toutes les
// égalités ci-dessous sont donc EXACTES (==), sans tolérance flottante.
// Les constantes attendues ont été dérivées en arithmétique rationnelle
// exacte avec la même convention de Cayley-Dickson :
//   (a, b)(c, d) = (a·c − d̄·b, d·a + b·c̄).

use std::simd::f32x4;

use super::dual::{DualOctonion, DualSedenion};
use super::octonion::OctonionSimd;
use super::quat::{quat_conj, quat_mul};
use super::scalar;
use super::sedenion::SedenionSimd;

/// Générateur congruentiel déterministe → petits entiers dans [-5, 5].
/// (Pas de dépendance `rand`, reproductible bit à bit.)
struct Lcg(u64);

impl Lcg {
    fn next_small_int(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) % 11) as f32 - 5.0
    }

    fn octonion(&mut self) -> OctonionSimd {
        let mut c = [0.0f32; 8];
        for x in &mut c
        {
            *x = self.next_small_int();
        }
        OctonionSimd::from_array(c)
    }

    fn sedenion(&mut self) -> SedenionSimd {
        let mut c = [0.0f32; 16];
        for x in &mut c
        {
            *x = self.next_small_int();
        }
        SedenionSimd::from_array(c)
    }
}

// ------------------------------------------------------------------ //
//  Layout mémoire                                                     //
// ------------------------------------------------------------------ //

#[test]
fn layout_alignment_and_size() {
    use core::mem::{align_of, size_of};
    // 256 bits alignés 32 : chargeables par vmovaps ymm.
    assert_eq!(size_of::<OctonionSimd>(), 32);
    assert_eq!(align_of::<OctonionSimd>(), 32);
    // 512 bits alignés 64 : une ligne de cache, chargeable par vmovaps zmm.
    assert_eq!(size_of::<SedenionSimd>(), 64);
    assert_eq!(align_of::<SedenionSimd>(), 64);
}

// ------------------------------------------------------------------ //
//  Quaternions (cas de base vectorisé)                                //
// ------------------------------------------------------------------ //

#[test]
fn quat_mul_matches_scalar_reference() {
    let mut rng = Lcg(0xDEADBEEF);
    for _ in 0..200
    {
        let mut p = [0.0f32; 4];
        let mut q = [0.0f32; 4];
        for x in &mut p
        {
            *x = rng.next_small_int();
        }
        for x in &mut q
        {
            *x = rng.next_small_int();
        }
        let simd = quat_mul(f32x4::from_array(p), f32x4::from_array(q));
        assert_eq!(simd.to_array(), scalar::quat_mul(p, q));
    }
}

#[test]
fn quat_hamilton_table() {
    // i·j = k, j·k = i, k·i = j, i² = j² = k² = −1.
    let e = |i: usize| {
        let mut c = [0.0f32; 4];
        c[i] = 1.0;
        f32x4::from_array(c)
    };
    let neg = |v: f32x4| (-v).to_array();
    assert_eq!(quat_mul(e(1), e(2)).to_array(), e(3).to_array()); // i·j = k
    assert_eq!(quat_mul(e(2), e(3)).to_array(), e(1).to_array()); // j·k = i
    assert_eq!(quat_mul(e(3), e(1)).to_array(), e(2).to_array()); // k·i = j
    assert_eq!(quat_mul(e(1), e(1)).to_array(), neg(e(0))); // i² = −1
    assert_eq!(quat_mul(e(2), e(1)).to_array(), neg(e(3))); // j·i = −k
    assert_eq!(quat_conj(e(1)).to_array(), neg(e(1))); // ī = −i
}

// ------------------------------------------------------------------ //
//  Octonions : correction                                             //
// ------------------------------------------------------------------ //

#[test]
fn octonion_basis_table_matches_scalar_reference() {
    // Les 64 produits eᵢ·eⱼ : SIMD == référence scalaire, et chaque
    // produit est un monôme ±e_k (structure d'algèbre de base).
    for i in 0..8
    {
        for j in 0..8
        {
            let simd = (OctonionSimd::unit(i) * OctonionSimd::unit(j)).to_array();
            let reference = scalar::oct_mul(
                OctonionSimd::unit(i).to_array(),
                OctonionSimd::unit(j).to_array(),
            );
            assert_eq!(simd, reference, "e{i}·e{j}");
            let nonzero: Vec<f32> = simd.iter().copied().filter(|&c| c != 0.0).collect();
            assert_eq!(nonzero.len(), 1, "e{i}·e{j} doit être un monôme");
            assert!(nonzero[0] == 1.0 || nonzero[0] == -1.0);
        }
    }
}

#[test]
fn octonion_fixed_product_vector() {
    // Vecteur de contrôle dérivé en arithmétique rationnelle exacte.
    let x = OctonionSimd::from_array([1.0, 2.0, -1.0, 3.0, 0.0, -2.0, 1.0, -1.0]);
    let y = OctonionSimd::from_array([-2.0, 1.0, 4.0, 0.0, -1.0, 2.0, -3.0, 1.0]);
    let expected = [8.0, -15.0, 10.0, -2.0, -9.0, -8.0, -7.0, 13.0];
    assert_eq!((x * y).to_array(), expected);
}

#[test]
fn octonion_simd_matches_scalar_on_random_inputs() {
    let mut rng = Lcg(0x0C70);
    for _ in 0..500
    {
        let x = rng.octonion();
        let y = rng.octonion();
        assert_eq!(
            (x * y).to_array(),
            scalar::oct_mul(x.to_array(), y.to_array())
        );
    }
}

#[test]
fn octonion_conjugation_is_an_antiautomorphism() {
    // conj(x·y) = conj(y)·conj(x) — exact sur entiers.
    let mut rng = Lcg(0xC0817);
    for _ in 0..100
    {
        let x = rng.octonion();
        let y = rng.octonion();
        assert_eq!((x * y).conj(), y.conj() * x.conj());
    }
}

#[test]
fn octonion_norm_is_multiplicative() {
    // 𝕆 est une algèbre de composition : ‖x·y‖² = ‖x‖²·‖y‖².
    // (Propriété qui échoue pour 𝕊 — voir le test des diviseurs de zéro.)
    let mut rng = Lcg(0x2077);
    for _ in 0..100
    {
        let x = rng.octonion();
        let y = rng.octonion();
        assert_eq!((x * y).norm_sqr(), x.norm_sqr() * y.norm_sqr());
    }
}

// ------------------------------------------------------------------ //
//  Octonions : non-associativité & alternativité                      //
// ------------------------------------------------------------------ //

#[test]
fn octonion_non_associativity_on_basis_elements() {
    // Cas dérivé exactement : (e₁·e₂)·e₄ = +e₇ mais e₁·(e₂·e₄) = −e₇.
    let e1 = OctonionSimd::unit(1);
    let e2 = OctonionSimd::unit(2);
    let e4 = OctonionSimd::unit(4);
    let e7 = OctonionSimd::unit(7);

    let left = (e1 * e2) * e4;
    let right = e1 * (e2 * e4);
    assert_eq!(left, e7);
    assert_eq!(right, -e7);
    assert_ne!(left, right, "l'associateur [e1, e2, e4] doit être non nul");
}

#[test]
fn octonion_non_associativity_on_dense_values() {
    // Cas non trivial à 8 composantes pleines : l'associateur
    // (x·y)·z − x·(y·z) est massivement non nul.
    let x = OctonionSimd::from_array([1.0, 2.0, -1.0, 3.0, 0.0, -2.0, 1.0, -1.0]);
    let y = OctonionSimd::from_array([-2.0, 1.0, 4.0, 0.0, -1.0, 2.0, -3.0, 1.0]);
    let z = OctonionSimd::from_array([3.0, -1.0, 0.0, 2.0, 1.0, 1.0, -2.0, 4.0]);

    let associator = (x * y) * z - x * (y * z);
    assert!(
        associator.norm_sqr() > 0.0,
        "associateur nul : les octonions sembleraient associatifs"
    );
}

#[test]
fn octonion_alternativity_left_and_right() {
    // 𝕆 est alternatif : x·(x·y) = (x·x)·y et (y·x)·x = y·(x·x).
    // Exact sur entiers (identité de Moufang / sous-algèbre engendrée
    // par 2 éléments associative — théorème d'Artin).
    let mut rng = Lcg(0xA17E12);
    for _ in 0..200
    {
        let x = rng.octonion();
        let y = rng.octonion();
        assert_eq!(x * (x * y), (x * x) * y, "alternativité gauche violée");
        assert_eq!((y * x) * x, y * (x * x), "alternativité droite violée");
    }
}

// ------------------------------------------------------------------ //
//  Octonions : norme, normalisation, inverse                          //
// ------------------------------------------------------------------ //

/// Écart maximal composante à composante entre deux tableaux `f32`.
fn max_abs_diff<const N: usize>(a: [f32; N], b: [f32; N]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn octonion_normalize_has_unit_norm() {
    let mut rng = Lcg(0x0C7A);
    for _ in 0..200
    {
        let o = rng.octonion();
        if o.norm_sqr() == 0.0
        {
            continue;
        }
        let n = o.normalize().norm();
        assert!((n - 1.0).abs() < 1e-4, "‖normalize(o)‖ = {n}");
    }
}

#[test]
fn octonion_inverse_is_two_sided() {
    // 𝕆 alternatif ⇒ ō·o = o·ō = ‖o‖²·1 exactement, donc o⁻¹·o = o·o⁻¹ = 1.
    let mut rng = Lcg(0x1_1170_ABCD);
    for _ in 0..200
    {
        let o = rng.octonion();
        if o.norm_sqr() == 0.0
        {
            continue;
        }
        let inv = o.inverse();
        let left = (inv * o).to_array();
        let right = (o * inv).to_array();
        assert!(
            max_abs_diff(left, OctonionSimd::ONE.to_array()) < 1e-3,
            "o⁻¹·o ≠ 1 : {left:?}"
        );
        assert!(
            max_abs_diff(right, OctonionSimd::ONE.to_array()) < 1e-3,
            "o·o⁻¹ ≠ 1 : {right:?}"
        );
    }
}

// ------------------------------------------------------------------ //
//  Sédénions : correction                                             //
// ------------------------------------------------------------------ //

#[test]
fn sedenion_basis_table_matches_scalar_reference() {
    for i in 0..16
    {
        for j in 0..16
        {
            let simd = (SedenionSimd::unit(i) * SedenionSimd::unit(j)).to_array();
            let reference = scalar::sed_mul(
                SedenionSimd::unit(i).to_array(),
                SedenionSimd::unit(j).to_array(),
            );
            assert_eq!(simd, reference, "e{i}·e{j}");
        }
    }
}

#[test]
fn sedenion_fixed_product_vector() {
    // Vecteur de contrôle dérivé en arithmétique rationnelle exacte.
    let x = SedenionSimd::from_array([
        1.0, -1.0, 2.0, 0.0, 3.0, -2.0, 1.0, 1.0, 0.0, 2.0, -1.0, 1.0, -3.0, 0.0, 2.0, -1.0,
    ]);
    let y = SedenionSimd::from_array([
        2.0, 1.0, 0.0, -1.0, 1.0, 3.0, -2.0, 0.0, 1.0, -1.0, 2.0, 0.0, 1.0, -2.0, 0.0, 3.0,
    ]);
    let expected = [
        18.0, 6.0, -1.0, 3.0, 9.0, -4.0, -11.0, 10.0, -20.0, 15.0, -3.0, -12.0, 3.0, -7.0, 14.0,
        2.0,
    ];
    assert_eq!((x * y).to_array(), expected);
}

#[test]
fn sedenion_simd_matches_scalar_on_random_inputs() {
    let mut rng = Lcg(0x5ED);
    for _ in 0..500
    {
        let x = rng.sedenion();
        let y = rng.sedenion();
        assert_eq!(
            (x * y).to_array(),
            scalar::sed_mul(x.to_array(), y.to_array())
        );
    }
}

#[test]
fn scalar_table_matches_recursive_reference() {
    // La baseline « boucle par boucle » des benchmarks est elle-même
    // validée contre l'oracle récursif.
    let oct_table = scalar::oct_table();
    let sed_table = scalar::sed_table();
    let mut rng = Lcg(0x7AB1E);
    for _ in 0..200
    {
        let (x, y) = (rng.octonion().to_array(), rng.octonion().to_array());
        assert_eq!(oct_table.mul(&x, &y), scalar::oct_mul(x, y));
        let (s, t) = (rng.sedenion().to_array(), rng.sedenion().to_array());
        assert_eq!(sed_table.mul(&s, &t), scalar::sed_mul(s, t));
    }
}

// ------------------------------------------------------------------ //
//  Sédénions : pathologies caractéristiques                           //
// ------------------------------------------------------------------ //

#[test]
fn sedenion_zero_divisors() {
    // Diviseurs de zéro : (e₁ + e₁₀)·(e₄ − e₁₅) = 0 EXACTEMENT,
    // avec les deux facteurs non nuls (‖·‖² = 2 chacun).
    // Dérivé en arithmétique exacte avec la convention CD du module.
    let x = SedenionSimd::unit(1) + SedenionSimd::unit(10);
    let y = SedenionSimd::unit(4) - SedenionSimd::unit(15);

    assert_eq!(x.norm_sqr(), 2.0, "x doit être non nul");
    assert_eq!(y.norm_sqr(), 2.0, "y doit être non nul");

    let product = x * y;
    // Zéro EXACT sur les 16 lanes — pas une tolérance : les produits de
    // ±1 s'annulent bit à bit dans le pipeline SIMD.
    assert_eq!(
        product.to_array(),
        [0.0f32; 16],
        "(e1 + e10)·(e4 − e15) doit être le sédénion nul"
    );

    // Corollaire : la norme n'est PAS multiplicative sur 𝕊
    // (‖x·y‖² = 0 ≠ 4 = ‖x‖²·‖y‖²) — 𝕊 n'est pas une algèbre de composition.
    assert_ne!(product.norm_sqr(), x.norm_sqr() * y.norm_sqr());
}

#[test]
fn sedenion_zero_divisor_factors_are_still_individually_invertible() {
    // Les DEUX facteurs du diviseur de zéro ci-dessus (x·y = 0, x,y ≠ 0)
    // sont chacun parfaitement inversibles des deux côtés : l'identité
    // s̄·s = s·s̄ = ‖s‖²·1 tient à tout niveau de Cayley-Dickson, y compris
    // 𝕊. Ceci ne contredit pas l'existence du diviseur de zéro : l'argument
    // « s inversible et s·t = 0, t ≠ 0, sont incompatibles » repose sur
    // l'associativité (s⁻¹·(s·t) = (s⁻¹·s)·t), qui échoue sur 𝕊.
    let x = SedenionSimd::unit(1) + SedenionSimd::unit(10);
    let y = SedenionSimd::unit(4) - SedenionSimd::unit(15);
    assert_eq!((x * y).to_array(), [0.0f32; 16]); // rappel du diviseur de zéro

    for s in [x, y]
    {
        let inv = s.inverse();
        let left = (inv * s).to_array();
        let right = (s * inv).to_array();
        assert!(
            max_abs_diff(left, SedenionSimd::ONE.to_array()) < 1e-4,
            "s⁻¹·s ≠ 1 : {left:?}"
        );
        assert!(
            max_abs_diff(right, SedenionSimd::ONE.to_array()) < 1e-4,
            "s·s⁻¹ ≠ 1 : {right:?}"
        );
    }
}

#[test]
fn sedenion_normalize_has_unit_norm() {
    let mut rng = Lcg(0x50D1);
    for _ in 0..200
    {
        let s = rng.sedenion();
        if s.norm_sqr() == 0.0
        {
            continue;
        }
        let n = s.normalize().norm();
        assert!((n - 1.0).abs() < 1e-4, "‖normalize(s)‖ = {n}");
    }
}

#[test]
fn sedenion_inverse_is_two_sided_on_random_inputs() {
    let mut rng = Lcg(0x5ED_1234);
    for _ in 0..200
    {
        let s = rng.sedenion();
        if s.norm_sqr() == 0.0
        {
            continue;
        }
        let inv = s.inverse();
        let left = (inv * s).to_array();
        let right = (s * inv).to_array();
        assert!(
            max_abs_diff(left, SedenionSimd::ONE.to_array()) < 1e-3,
            "s⁻¹·s ≠ 1 : {left:?}"
        );
        assert!(
            max_abs_diff(right, SedenionSimd::ONE.to_array()) < 1e-3,
            "s·s⁻¹ ≠ 1 : {right:?}"
        );
    }
}

#[test]
fn sedenion_alternativity_failure() {
    // Perte d'alternativité : pour x = e₁ + e₁₀ et y = e₄,
    //   x·(x·y) = −2·e₄ − 2·e₁₅   mais   (x·x)·y = −2·e₄.
    // (Valeurs dérivées en arithmétique exacte.)
    let x = SedenionSimd::unit(1) + SedenionSimd::unit(10);
    let y = SedenionSimd::unit(4);

    let lhs = x * (x * y);
    let rhs = (x * x) * y;

    let expected_lhs = (SedenionSimd::unit(4) + SedenionSimd::unit(15)).scale(-2.0);
    let expected_rhs = SedenionSimd::unit(4).scale(-2.0);
    assert_eq!(lhs, expected_lhs);
    assert_eq!(rhs, expected_rhs);
    assert_ne!(lhs, rhs, "l'alternativité gauche doit échouer sur 𝕊");
}

// ------------------------------------------------------------------ //
//  Différenciation automatique forward-mode                           //
// ------------------------------------------------------------------ //

#[test]
fn dual_octonion_product_rule_is_exact() {
    // f(x) = x·x ⇒ Df(x₀)[v] = v·x₀ + x₀·v (Leibniz non commutatif).
    let mut rng = Lcg(0xD0A1);
    for _ in 0..100
    {
        let x0 = rng.octonion();
        let v = rng.octonion();
        let x = DualOctonion::variable(x0, v);
        let fx = x * x;
        assert_eq!(fx.val, x0 * x0);
        assert_eq!(fx.eps, v * x0 + x0 * v);
    }
}

#[test]
fn dual_octonion_constants_have_zero_derivative() {
    // f(x) = c·x·c′ (c, c′ constantes) ⇒ Df(x₀)[v] = c·v·c′ — les
    // parenthèses comptent (non-associativité) : on fixe (c·x)·c′.
    let c = OctonionSimd::from_array([1.0, 0.0, 2.0, -1.0, 0.0, 3.0, 0.0, 1.0]);
    let c2 = OctonionSimd::from_array([-1.0, 1.0, 0.0, 0.0, 2.0, 0.0, -2.0, 0.0]);
    let x0 = OctonionSimd::from_array([2.0, -1.0, 1.0, 3.0, -2.0, 0.0, 1.0, -1.0]);
    let v = OctonionSimd::from_array([0.0, 1.0, -1.0, 0.0, 2.0, 1.0, 0.0, 3.0]);

    let f =
        (DualOctonion::constant(c) * DualOctonion::variable(x0, v)) * DualOctonion::constant(c2);
    assert_eq!(f.val, (c * x0) * c2);
    assert_eq!(f.eps, (c * v) * c2);
}

#[test]
fn dual_octonion_epsilon_is_nilpotent() {
    // ε² = 0 : le produit de deux duaux purs (valeur nulle) est nul.
    let d1 = DualOctonion::new(OctonionSimd::ZERO, OctonionSimd::unit(3));
    let d2 = DualOctonion::new(OctonionSimd::ZERO, OctonionSimd::unit(5));
    let p = d1 * d2;
    assert_eq!(p.val, OctonionSimd::ZERO);
    assert_eq!(p.eps, OctonionSimd::ZERO);
}

#[test]
fn dual_octonion_matches_finite_differences() {
    // Contrôle numérique indépendant : f(x) = (x·x)·x, dérivée duale vs
    // différence centrale (f(x₀+h·v) − f(x₀−h·v)) / 2h.
    let x0 = OctonionSimd::from_array([0.5, -0.25, 1.0, 0.75, -0.5, 0.25, -1.0, 0.5]);
    let v = OctonionSimd::from_array([1.0, 0.5, -0.5, 0.25, 0.75, -0.25, 0.5, -1.0]);

    let x = DualOctonion::variable(x0, v);
    let f = (x * x) * x;

    let h = 1.0e-3f32;
    let cube = |o: OctonionSimd| (o * o) * o;
    let plus = cube(x0 + v.scale(h));
    let minus = cube(x0 - v.scale(h));
    let fd = (plus - minus).scale(1.0 / (2.0 * h));

    let err = (f.eps - fd).norm_sqr().sqrt();
    assert!(
        err < 1.0e-2,
        "dérivée duale trop éloignée des différences finies : err = {err}"
    );
}

#[test]
fn dual_sedenion_product_rule_is_exact() {
    let mut rng = Lcg(0xD5ED);
    for _ in 0..100
    {
        let x0 = rng.sedenion();
        let v = rng.sedenion();
        let x = DualSedenion::variable(x0, v);
        let fx = x * x;
        assert_eq!(fx.val, x0 * x0);
        assert_eq!(fx.eps, v * x0 + x0 * v);
    }
}

#[test]
fn dual_sedenion_norm_sqr_gradient() {
    // (‖x‖²)' = 2⟨x₀, v⟩ — vérifié sur entiers (exact).
    let x0 = SedenionSimd::from_array([
        1.0, -2.0, 3.0, 0.0, 1.0, -1.0, 2.0, 0.0, -3.0, 1.0, 0.0, 2.0, -1.0, 1.0, 0.0, -2.0,
    ]);
    let v = SedenionSimd::from_array([
        2.0, 1.0, 0.0, -1.0, 1.0, 3.0, -2.0, 0.0, 1.0, -1.0, 2.0, 0.0, 1.0, -2.0, 0.0, 3.0,
    ]);
    let (value, deriv) = DualSedenion::variable(x0, v).norm_sqr();
    assert_eq!(value, x0.norm_sqr());
    // 2⟨x₀, v⟩ calculé à la main :
    let dot: f32 = x0
        .to_array()
        .iter()
        .zip(v.to_array().iter())
        .map(|(a, b)| a * b)
        .sum();
    assert_eq!(deriv, 2.0 * dot);
}

#[test]
fn dual_octonion_norm_normalize_inverse_match_finite_differences() {
    // x₀ non nul, v quelconque ; contrôle numérique indépendant contre la
    // différence centrale (f(x₀+h·v) − f(x₀−h·v)) / 2h, même schéma que
    // `dual_octonion_matches_finite_differences`.
    let x0 = OctonionSimd::from_array([0.5, -0.25, 1.0, 0.75, -0.5, 0.25, -1.0, 0.5]);
    let v = OctonionSimd::from_array([1.0, 0.5, -0.5, 0.25, 0.75, -0.25, 0.5, -1.0]);
    let h = 1.0e-3f32;

    let dual = DualOctonion::variable(x0, v);

    // norm : f = ‖x‖ ∈ ℝ.
    let (val_n, deriv_n) = dual.norm();
    assert_eq!(val_n, x0.norm());
    let plus_n = (x0 + v.scale(h)).norm();
    let minus_n = (x0 - v.scale(h)).norm();
    let fd_norm = (plus_n - minus_n) / (2.0 * h);
    assert!(
        (deriv_n - fd_norm).abs() < 1e-2,
        "norm : dérivée duale {deriv_n} vs différences finies {fd_norm}"
    );

    // normalize : f = x/‖x‖ ∈ 𝕆.
    let dual_normalize = dual.normalize();
    assert_eq!(dual_normalize.val, x0.normalize());
    let plus = (x0 + v.scale(h)).normalize();
    let minus = (x0 - v.scale(h)).normalize();
    let fd_normalize = (plus - minus).scale(1.0 / (2.0 * h));
    let err_normalize = (dual_normalize.eps - fd_normalize).norm_sqr().sqrt();
    assert!(err_normalize < 1e-2, "normalize : err = {err_normalize}");

    // inverse : f = x⁻¹ ∈ 𝕆.
    let dual_inverse = dual.inverse();
    assert_eq!(dual_inverse.val, x0.inverse());
    let plus = (x0 + v.scale(h)).inverse();
    let minus = (x0 - v.scale(h)).inverse();
    let fd_inverse = (plus - minus).scale(1.0 / (2.0 * h));
    let err_inverse = (dual_inverse.eps - fd_inverse).norm_sqr().sqrt();
    assert!(err_inverse < 1e-2, "inverse : err = {err_inverse}");
}

#[test]
fn dual_sedenion_norm_normalize_inverse_match_finite_differences() {
    let x0 = SedenionSimd::from_array([
        1.0, -2.0, 3.0, 0.0, 1.0, -1.0, 2.0, 0.0, -3.0, 1.0, 0.0, 2.0, -1.0, 1.0, 0.0, -2.0,
    ]);
    let v = SedenionSimd::from_array([
        2.0, 1.0, 0.0, -1.0, 1.0, 3.0, -2.0, 0.0, 1.0, -1.0, 2.0, 0.0, 1.0, -2.0, 0.0, 3.0,
    ]);
    let h = 1.0e-3f32;

    let dual = DualSedenion::variable(x0, v);

    let (val_n, deriv_n) = dual.norm();
    assert_eq!(val_n, x0.norm());
    let plus_n = (x0 + v.scale(h)).norm();
    let minus_n = (x0 - v.scale(h)).norm();
    let fd_norm = (plus_n - minus_n) / (2.0 * h);
    assert!(
        (deriv_n - fd_norm).abs() < 1e-2,
        "norm : dérivée duale {deriv_n} vs différences finies {fd_norm}"
    );

    let dual_normalize = dual.normalize();
    assert_eq!(dual_normalize.val, x0.normalize());
    let plus = (x0 + v.scale(h)).normalize();
    let minus = (x0 - v.scale(h)).normalize();
    let fd_normalize = (plus - minus).scale(1.0 / (2.0 * h));
    let err_normalize = (dual_normalize.eps - fd_normalize).norm_sqr().sqrt();
    assert!(err_normalize < 1e-2, "normalize : err = {err_normalize}");

    let dual_inverse = dual.inverse();
    assert_eq!(dual_inverse.val, x0.inverse());
    let plus = (x0 + v.scale(h)).inverse();
    let minus = (x0 - v.scale(h)).inverse();
    let fd_inverse = (plus - minus).scale(1.0 / (2.0 * h));
    let err_inverse = (dual_inverse.eps - fd_inverse).norm_sqr().sqrt();
    assert!(err_inverse < 1e-2, "inverse : err = {err_inverse}");
}
