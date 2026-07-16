//! **Béton armé — poinçonnement d'une dalle (Eurocode 2, ELU)** : périmètre de
//! contrôle `u1` à `2d` d'un poteau rectangulaire intérieur, contrainte de
//! poinçonnement de calcul `vEd`, résistance au poinçonnement sans armatures
//! spécifiques `vRd,c` et taux d'exploitation associé.
//!
//! ```text
//! périmètre de contrôle   u1          = 2·(c1 + c2) + 2·π·(2·d)
//! contrainte              vEd         = β · VEd / (u1 · d)
//! résistance              vRd,c       = CRd,c · k · (100 · ρl · fck)^(1/3)
//! taux d'exploitation     utilisation = vEd / vRd,c
//! ```
//!
//! `u1` périmètre de contrôle au contour de référence situé à `2d` du poteau
//! (mm), `c1` = `column_width` et `c2` = `column_depth` dimensions en plan du
//! poteau rectangulaire (mm), `d` = `effective_depth` hauteur utile moyenne de
//! la dalle (mm), `vEd` contrainte de cisaillement de poinçonnement de calcul
//! (MPa), `VEd` = `design_shear_force` effort de poinçonnement de calcul (N),
//! `β` = `load_factor_beta` coefficient de majoration tenant compte de
//! l'excentricité de la charge (sans dimension, `≥ 1`), `vRd,c` résistance au
//! poinçonnement sans armatures spécifiques (MPa), `CRd,c` = `crd_c` coefficient
//! réglementaire de l'EC2 (sans dimension, usuellement `γc`-dépendant), `k` =
//! `size_factor_k` coefficient d'échelle (sans dimension), `ρl` = `rho_l` ratio
//! géométrique moyen d'armatures longitudinales tendues (sans dimension), `fck`
//! résistance caractéristique en compression du béton (MPa), `utilisation` taux
//! d'exploitation (sans dimension, `≤ 1` = vérifié).
//!
//! **Convention** : N, mm, MPa (avec `1 MPa = 1 N/mm²`), donc les contraintes
//! ressortent en **mégapascals** et les périmètres en **millimètres** ; le
//! périmètre `u1` combine les deux côtés droits `2·(c1 + c2)` et les quatre
//! quarts de cercle de rayon `2d`, soit `2·π·(2·d)`.
//!
//! **Limite honnête** : vérification au **périmètre de contrôle `u1` (à `2d`)**
//! d'un **poteau intérieur rectangulaire** uniquement ; les résistances
//! caractéristiques (`fck`, `fyk`, `fy`…) **et** les coefficients partiels de
//! sécurité (`γc`, `γs`, `γM`…), ainsi que le coefficient de majoration `β`
//! (excentricité) et les coefficients réglementaires `CRd,c`, `k` et le ratio
//! `ρl` sont **fournis par l'appelant** d'après l'Eurocode 2 et son Annexe
//! Nationale ; aucune valeur « par défaut » n'est inventée. Ce module **ne
//! dimensionne pas** les armatures de poinçonnement et ne traite ni les poteaux
//! de rive/d'angle, ni la vérification au nu du poteau (`vRd,max`).

use core::f64::consts::PI;

/// Périmètre de contrôle de référence à `2d` d'un poteau rectangulaire
/// intérieur `u1 = 2·(c1 + c2) + 2·π·(2·d)` (mm), avec `c1`, `c2` et `d` en mm.
///
/// Panique si `column_width <= 0`, si `column_depth <= 0` ou si
/// `effective_depth <= 0`.
pub fn rcpunch_control_perimeter_rectangular(
    column_width: f64,
    column_depth: f64,
    effective_depth: f64,
) -> f64 {
    assert!(
        column_width > 0.0,
        "la largeur de poteau c1 doit être strictement positive"
    );
    assert!(
        column_depth > 0.0,
        "la profondeur de poteau c2 doit être strictement positive"
    );
    assert!(
        effective_depth > 0.0,
        "la hauteur utile d doit être strictement positive"
    );
    2.0 * (column_width + column_depth) + 2.0 * PI * 2.0 * effective_depth
}

/// Contrainte de poinçonnement de calcul `vEd = β · VEd / (u1 · d)` (MPa), avec
/// `VEd` en N et `u1`, `d` en mm.
///
/// Panique si `design_shear_force < 0`, si `load_factor_beta < 1`, si
/// `control_perimeter <= 0` ou si `effective_depth <= 0` (division par zéro).
pub fn rcpunch_shear_stress(
    design_shear_force: f64,
    load_factor_beta: f64,
    control_perimeter: f64,
    effective_depth: f64,
) -> f64 {
    assert!(
        design_shear_force >= 0.0,
        "l'effort de poinçonnement VEd doit être ≥ 0"
    );
    assert!(
        load_factor_beta >= 1.0,
        "le coefficient de majoration β doit être ≥ 1"
    );
    assert!(
        control_perimeter > 0.0,
        "le périmètre de contrôle u1 doit être strictement positif"
    );
    assert!(
        effective_depth > 0.0,
        "la hauteur utile d doit être strictement positive"
    );
    load_factor_beta * design_shear_force / (control_perimeter * effective_depth)
}

/// Résistance au poinçonnement sans armatures spécifiques
/// `vRd,c = CRd,c · k · (100 · ρl · fck)^(1/3)` (MPa), avec `fck` en MPa.
///
/// Panique si `crd_c <= 0`, si `size_factor_k` n'est pas dans `[1, 2]`, si
/// `rho_l < 0` ou si `fck <= 0`.
pub fn rcpunch_resistance_without_reinforcement(
    crd_c: f64,
    size_factor_k: f64,
    rho_l: f64,
    fck: f64,
) -> f64 {
    assert!(
        crd_c > 0.0,
        "le coefficient CRd,c doit être strictement positif"
    );
    assert!(
        (1.0..=2.0).contains(&size_factor_k),
        "le coefficient d'échelle k doit être dans [1, 2]"
    );
    assert!(
        rho_l >= 0.0,
        "le ratio d'armatures longitudinales ρl doit être ≥ 0"
    );
    assert!(
        fck > 0.0,
        "la résistance fck doit être strictement positive"
    );
    crd_c * size_factor_k * (100.0 * rho_l * fck).cbrt()
}

/// Taux d'exploitation au poinçonnement `utilisation = vEd / vRd,c` (sans
/// dimension), avec `vEd` et `vRd,c` en MPa ; `≤ 1` signifie « vérifié ».
///
/// Panique si `shear_stress < 0` ou si `resistance <= 0` (division par zéro).
pub fn rcpunch_utilisation(shear_stress: f64, resistance: f64) -> f64 {
    assert!(
        shear_stress >= 0.0,
        "la contrainte de poinçonnement vEd doit être ≥ 0"
    );
    assert!(
        resistance > 0.0,
        "la résistance vRd,c doit être strictement positive"
    );
    shear_stress / resistance
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn control_perimeter_splits_into_straight_and_curved_parts() {
        // Poteau 400 × 400 mm, dalle d = 200 mm :
        //   u1 = 2·(400 + 400) + 2·π·(2·200) = 1600 + 800·π ≈ 4113,274 mm
        let u1 = rcpunch_control_perimeter_rectangular(400.0, 400.0, 200.0);
        assert_relative_eq!(u1, 1600.0 + 800.0 * PI, epsilon = 1e-9);
        // Identité : le périmètre moins les deux côtés droits vaut exactement les
        // quatre quarts de cercle de rayon 2d, soit 2·π·(2·d) = 4·π·d.
        let curved = u1 - 2.0 * (400.0 + 400.0);
        assert_relative_eq!(curved, 4.0 * PI * 200.0, epsilon = 1e-9);
    }

    #[test]
    fn control_perimeter_grows_linearly_with_depth() {
        // La partie courbe est linéaire en d : augmenter d de Δ augmente u1 de
        // 4·π·Δ, indépendamment des dimensions du poteau.
        let u_a = rcpunch_control_perimeter_rectangular(300.0, 500.0, 180.0);
        let u_b = rcpunch_control_perimeter_rectangular(300.0, 500.0, 280.0);
        assert_relative_eq!(u_b - u_a, 4.0 * PI * 100.0, epsilon = 1e-9);
    }

    #[test]
    fn shear_stress_clean_case_and_proportionality() {
        // Cas chiffré propre : β = 1, VEd = 400 000 N, u1 = 2000 mm, d = 200 mm :
        //   vEd = 1 · 400 000 / (2000 · 200) = 400 000 / 400 000 = 1,0 MPa
        let v = rcpunch_shear_stress(400_000.0, 1.0, 2000.0, 200.0);
        assert_relative_eq!(v, 1.0, epsilon = 1e-12);
        // Proportionnalité : doubler l'effort VEd double la contrainte vEd.
        let v2 = rcpunch_shear_stress(800_000.0, 1.0, 2000.0, 200.0);
        assert_relative_eq!(v2 / v, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn resistance_clean_cube_case() {
        // Cube parfait : 100 · ρl · fck = 100 · 0,01 · 27 = 27 → (27)^(1/3) = 3.
        //   vRd,c = CRd,c · k · 3 = 0,12 · 2,0 · 3 = 0,72 MPa
        let vrd = rcpunch_resistance_without_reinforcement(0.12, 2.0, 0.01, 27.0);
        assert_relative_eq!(vrd, 0.72, epsilon = 1e-12);
    }

    #[test]
    fn utilisation_is_reciprocal_of_resistance() {
        // Réciprocité : utilisation(v, R) · R = v, et utilisation(R, R) = 1.
        let v = 0.72_f64;
        let r = 0.90_f64;
        let u = rcpunch_utilisation(v, r);
        assert_relative_eq!(u * r, v, epsilon = 1e-12);
        assert_relative_eq!(rcpunch_utilisation(r, r), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn realistic_interior_column_chain() {
        // Chaîne réaliste : poteau 400 × 400 mm, dalle d = 200 mm, VEd = 500 kN,
        // β = 1,15, CRd,c = 0,12, k = 2,0, ρl = 0,0125, fck = 30 MPa.
        //   u1    = 1600 + 800·π ≈ 4113,274 mm
        //   vEd   = 1,15 · 500 000 / (4113,274 · 200) ≈ 0,69895 MPa
        //   vRd,c = 0,12 · 2,0 · (100 · 0,0125 · 30)^(1/3)
        //         = 0,24 · (37,5)^(1/3) = 0,24 · 3,347… ≈ 0,80333 MPa
        //   util  ≈ 0,69895 / 0,80333 ≈ 0,8701
        let u1 = rcpunch_control_perimeter_rectangular(400.0, 400.0, 200.0);
        let ved = rcpunch_shear_stress(500_000.0, 1.15, u1, 200.0);
        let vrd = rcpunch_resistance_without_reinforcement(0.12, 2.0, 0.0125, 30.0);
        let util = rcpunch_utilisation(ved, vrd);
        assert_relative_eq!(ved, 0.698_95, epsilon = 1e-3);
        assert_relative_eq!(vrd, 0.803_33, epsilon = 1e-3);
        assert_relative_eq!(util, 0.870_1, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(expected = "la résistance vRd,c doit être strictement positive")]
    fn utilisation_rejects_null_resistance() {
        // Division par zéro interdite : vRd,c = 0.
        rcpunch_utilisation(0.72, 0.0);
    }
}
