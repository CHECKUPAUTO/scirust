//! **Charpente métallique — flambement par flexion d'une barre comprimée
//! (Eurocode 3, EN 1993-1-1 §6.3.1)** : charge critique d'Euler `Ncr`, élancement
//! réduit `λ̄`, facteur de réduction `χ` de la courbe de flambement, puis
//! résistance de calcul au flambement `Nb,Rd` d'un poteau comprimé.
//!
//! ```text
//! charge critique d'Euler   Ncr   = π²·E·I / Lcr²
//! élancement réduit         λ̄     = √(A·fy / Ncr)          (classes 1-3)
//! coefficient intermédiaire Φ     = 0,5·(1 + α·(λ̄ − 0,2) + λ̄²)
//! facteur de réduction      χ     = 1 / (Φ + √(Φ² − λ̄²))   (plafonné à 1,0)
//! résistance au flambement  Nb,Rd = χ·A·fy / γM1
//! ```
//!
//! `E` = `youngs_modulus` module d'Young de l'acier (MPa = N/mm²), `I` =
//! `second_moment` moment quadratique de la section autour de l'axe de flambement
//! (mm⁴), `Lcr` = `buckling_length` longueur de flambement (mm), `Ncr` =
//! `critical_load`/`steelcomp_euler_critical_load` charge critique d'Euler (N),
//! `A` = `area` aire brute de la section (mm²), `fy` limite d'élasticité de
//! l'acier (MPa), `λ̄` = `non_dimensional_slenderness` élancement réduit (sans
//! dimension), `α` = `imperfection_factor` facteur d'imperfection de la courbe de
//! flambement (sans dimension), `Φ` grandeur intermédiaire (sans dimension),
//! `χ` = `reduction_factor` facteur de réduction (sans dimension, `0 < χ ≤ 1`),
//! `γM1` = `gamma_m1` coefficient partiel de sécurité (sans dimension), `Nb,Rd`
//! résistance de calcul au flambement (N).
//!
//! **Convention** : N, mm, MPa (avec `1 MPa = 1 N/mm²`, donc `Ncr`, `A·fy` et
//! `Nb,Rd` sont en N) ; `λ̄`, `α`, `Φ`, `χ` et `γM1` sont sans dimension.
//! **Limite honnête** : ce module traite le seul **flambement par flexion** d'une
//! barre comprimée de section de **classe 1, 2 ou 3** (l'élancement réduit
//! emploie l'aire brute `A` ; une classe 4 exigerait l'aire efficace `Aeff`). Le
//! **flambement par torsion** et le **flambement par flexion-torsion** ne sont
//! **pas** couverts. La limite d'élasticité caractéristique `fy`, le coefficient
//! partiel de sécurité `γM1` et le facteur d'imperfection `α` de la courbe de
//! flambement adaptée (`a0`, `a`, `b`, `c` ou `d` selon la section et l'axe) sont
//! **fournis par l'appelant** d'après l'Eurocode et son Annexe Nationale ; aucune
//! valeur « par défaut » n'est inventée.

/// Charge critique d'Euler `Ncr = π²·E·I / Lcr²` (N) d'une barre comprimée, avec
/// `youngs_modulus` = `E` en MPa (N/mm²), `second_moment` = `I` en mm⁴ et
/// `buckling_length` = `Lcr` en mm.
///
/// Panique si `youngs_modulus <= 0`, `second_moment <= 0` ou
/// `buckling_length <= 0`.
pub fn steelcomp_euler_critical_load(
    youngs_modulus: f64,
    second_moment: f64,
    buckling_length: f64,
) -> f64 {
    assert!(
        youngs_modulus > 0.0,
        "le module d'Young E doit être strictement positif (MPa)"
    );
    assert!(
        second_moment > 0.0,
        "le moment quadratique I doit être strictement positif (mm⁴)"
    );
    assert!(
        buckling_length > 0.0,
        "la longueur de flambement Lcr doit être strictement positive (mm)"
    );
    core::f64::consts::PI * core::f64::consts::PI * youngs_modulus * second_moment
        / (buckling_length * buckling_length)
}

/// Élancement réduit `λ̄ = √(A·fy / Ncr)` (sans dimension, pour les classes 1 à 3),
/// avec `area` = `A` l'aire brute en mm², `fy` la limite d'élasticité en MPa et
/// `critical_load` = `Ncr` la charge critique d'Euler en N (`A·fy` est en N).
///
/// Panique si `area <= 0`, `fy <= 0` ou `critical_load <= 0`.
pub fn steelcomp_non_dimensional_slenderness(area: f64, fy: f64, critical_load: f64) -> f64 {
    assert!(
        area > 0.0,
        "l'aire A de la section doit être strictement positive (mm²)"
    );
    assert!(
        fy > 0.0,
        "la limite d'élasticité fy doit être strictement positive (MPa)"
    );
    assert!(
        critical_load > 0.0,
        "la charge critique Ncr doit être strictement positive (N)"
    );
    (area * fy / critical_load).sqrt()
}

/// Facteur de réduction `χ` de la courbe de flambement (sans dimension, plafonné à
/// `1,0`) : `Φ = 0,5·(1 + α·(λ̄ − 0,2) + λ̄²)` puis `χ = 1 / (Φ + √(Φ² − λ̄²))`,
/// avec `non_dimensional_slenderness` = `λ̄` l'élancement réduit et
/// `imperfection_factor` = `α` le facteur d'imperfection de la courbe choisie.
///
/// Panique si `non_dimensional_slenderness < 0` ou `imperfection_factor < 0`.
pub fn steelcomp_reduction_factor(
    non_dimensional_slenderness: f64,
    imperfection_factor: f64,
) -> f64 {
    assert!(
        non_dimensional_slenderness >= 0.0,
        "l'élancement réduit λ̄ doit être ≥ 0"
    );
    assert!(
        imperfection_factor >= 0.0,
        "le facteur d'imperfection α doit être ≥ 0"
    );
    let lambda_bar = non_dimensional_slenderness;
    let phi = 0.5 * (1.0 + imperfection_factor * (lambda_bar - 0.2) + lambda_bar * lambda_bar);
    let chi = 1.0 / (phi + (phi * phi - lambda_bar * lambda_bar).sqrt());
    chi.min(1.0)
}

/// Résistance de calcul au flambement `Nb,Rd = χ·A·fy / γM1` (N), avec
/// `reduction_factor` = `χ` le facteur de réduction (sans dimension), `area` = `A`
/// l'aire brute en mm², `fy` la limite d'élasticité en MPa et `gamma_m1` = `γM1`
/// le coefficient partiel de sécurité (sans dimension).
///
/// Panique si `reduction_factor < 0`, `area <= 0`, `fy <= 0` ou `gamma_m1 <= 0`.
pub fn steelcomp_buckling_resistance(
    reduction_factor: f64,
    area: f64,
    fy: f64,
    gamma_m1: f64,
) -> f64 {
    assert!(
        reduction_factor >= 0.0,
        "le facteur de réduction χ doit être ≥ 0"
    );
    assert!(
        area > 0.0,
        "l'aire A de la section doit être strictement positive (mm²)"
    );
    assert!(
        fy > 0.0,
        "la limite d'élasticité fy doit être strictement positive (MPa)"
    );
    assert!(
        gamma_m1 > 0.0,
        "le coefficient partiel γM1 doit être strictement positif"
    );
    reduction_factor * area * fy / gamma_m1
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn euler_load_scales_inverse_square_with_length() {
        // Ncr ∝ 1/Lcr² : doubler la longueur de flambement divise Ncr par 4.
        let e = 210_000.0_f64; // MPa
        let i = 1.0e7_f64; // mm⁴
        let n1 = steelcomp_euler_critical_load(e, i, 3000.0);
        let n2 = steelcomp_euler_critical_load(e, i, 6000.0);
        assert_relative_eq!(n1 / n2, 4.0, epsilon = 1e-12);
    }

    #[test]
    fn euler_load_worked_value() {
        // E = 210000 MPa, I = 1e7 mm⁴, Lcr = 3000 mm.
        // Ncr = π²·210000·1e7 / 3000² = π²·2,1e12 / 9e6 = π²·233333,333…
        //     = 9,8696044…·233333,333… ≈ 2 302 908 N (≈ 2303 kN).
        let ncr = steelcomp_euler_critical_load(210_000.0, 1.0e7, 3000.0);
        assert_relative_eq!(ncr, 2_302_908.5, epsilon = 1.0);
    }

    #[test]
    fn slenderness_identity_and_unit_case() {
        // Identité : λ̄²·Ncr = A·fy, quelles que soient les entrées.
        let area = 5000.0_f64; // mm²
        let fy = 355.0_f64; // MPa
        let ncr = 2_302_908.5_f64; // N
        let lambda_bar = steelcomp_non_dimensional_slenderness(area, fy, ncr);
        assert_relative_eq!(lambda_bar.powi(2) * ncr, area * fy, epsilon = 1e-6);
        // Cas limite : si Ncr = A·fy, alors λ̄ = √1 = 1 exactement.
        let squash = area * fy;
        assert_relative_eq!(
            steelcomp_non_dimensional_slenderness(area, fy, squash),
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn reduction_factor_unity_at_plateau_and_below() {
        // À λ̄ = 0,2, Φ = 0,5·(1 + 0 + 0,04) = 0,52 et √(0,52² − 0,04) = √0,2304
        // = 0,48 exactement, donc χ = 1/(0,52 + 0,48) = 1,0 quel que soit α.
        assert_relative_eq!(steelcomp_reduction_factor(0.2, 0.34), 1.0, epsilon = 1e-12);
        assert_relative_eq!(steelcomp_reduction_factor(0.2, 0.76), 1.0, epsilon = 1e-12);
        // En deçà du plateau (λ̄ < 0,2) le facteur est plafonné à 1,0.
        assert_relative_eq!(steelcomp_reduction_factor(0.1, 0.49), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn reduction_factor_worked_curve_b() {
        // Courbe b (α = 0,34), λ̄ = 1,0 :
        // Φ = 0,5·(1 + 0,34·0,8 + 1) = 0,5·2,272 = 1,136 ;
        // √(1,136² − 1) = √(1,290496 − 1) = √0,290496 = 0,538977 ;
        // χ = 1/(1,136 + 0,538977) = 1/1,674977 = 0,597023 (tableau EC3 : 0,5970).
        let chi = steelcomp_reduction_factor(1.0, 0.34);
        assert_relative_eq!(chi, 0.597_023, epsilon = 1e-3);
    }

    #[test]
    fn buckling_resistance_reduces_squash_load() {
        // Nb,Rd = χ·(A·fy/γM1). Avec χ = 1 on retrouve la charge plastique Npl,Rd,
        // et Nb,Rd est proportionnelle à χ.
        let area = 5000.0_f64; // mm²
        let fy = 355.0_f64; // MPa
        let gamma = 1.0_f64;
        let npl = steelcomp_buckling_resistance(1.0, area, fy, gamma);
        assert_relative_eq!(npl, area * fy / gamma, epsilon = 1e-9);
        let chi = 0.597_023_f64;
        let nb = steelcomp_buckling_resistance(chi, area, fy, gamma);
        assert_relative_eq!(nb, chi * npl, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "la longueur de flambement Lcr doit être strictement positive")]
    fn euler_load_rejects_non_positive_length() {
        // Longueur de flambement nulle : division par zéro, entrée refusée.
        steelcomp_euler_critical_load(210_000.0, 1.0e7, 0.0);
    }
}
