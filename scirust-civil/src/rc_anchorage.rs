//! **Béton armé — ancrage et recouvrement des armatures (Eurocode 2, ELU)** :
//! contrainte d'adhérence de calcul, longueur d'ancrage de base, longueur
//! d'ancrage de calcul (avec produit des coefficients de forme et d'enrobage)
//! et longueur de recouvrement des barres tendues.
//!
//! ```text
//! adhérence de calcul       fbd     = 2,25 · η1 · η2 · fctd
//! ancrage de base           lb,rqd  = (φ / 4) · (σsd / fbd)
//! ancrage de calcul         lbd     = max(∏αi · lb,rqd , lb,min)
//! recouvrement              l0      = α6 · lbd
//! ```
//!
//! `fbd` contrainte d'adhérence de calcul (MPa), `η1` coefficient de conditions
//! d'adhérence (sans dimension, `1,0` bonnes conditions, `0,7` sinon), `η2`
//! coefficient de diamètre de barre (sans dimension, `1,0` pour `φ ≤ 32 mm`,
//! `(132 − φ) / 100` au-delà), `fctd` résistance de calcul en traction du béton
//! (MPa), `lb,rqd` longueur d'ancrage de base requise (mm), `φ` diamètre de la
//! barre (mm), `σsd` contrainte dans la barre à ancrer (MPa), `lbd` longueur
//! d'ancrage de calcul (mm), `∏αi` produit des coefficients de forme, d'enrobage
//! et de confinement `α1·α2·α3·α4·α5` (sans dimension), `lb,min` longueur
//! d'ancrage minimale (mm), `l0` longueur de recouvrement (mm), `α6` coefficient
//! de recouvrement (sans dimension, usuellement `1,0 ≤ α6 ≤ 1,5`).
//!
//! **Convention** : N, mm, MPa (avec `1 MPa = 1 N/mm²`) ; toutes les longueurs
//! ressortent en **millimètres** et les contraintes sont en **mégapascals**.
//! **Limite honnête** : ancrage droit/courbe **simplifié** de l'EC2 ; la
//! contrainte d'adhérence `fbd` est reconstituée à partir des coefficients `η1`
//! (conditions d'adhérence) et `η2` (diamètre) **et** de `fctd` — tous **fournis
//! par l'appelant**. Les résistances caractéristiques (`fck`, `fctk`, `fyk`,
//! `fy`…) **et** les coefficients partiels de sécurité (`γc`, `γs`, `γM`…) — donc
//! `fctd` et la contrainte `σsd` — ainsi que le produit des coefficients de forme
//! `∏αi`, le coefficient de recouvrement `α6` et la longueur minimale `lb,min`
//! sont **fournis par l'appelant** d'après l'Eurocode 2 et son Annexe Nationale ;
//! aucune valeur « par défaut » n'est inventée. Le choix des coefficients, la
//! prise en compte des crochets et courbures, ainsi que les dispositions
//! constructives (espacement, proportion de barres en recouvrement) restent à la
//! charge de l'ingénieur.

/// Contrainte d'adhérence de calcul `fbd = 2,25 · η1 · η2 · fctd` (MPa), avec
/// `fctd` en MPa.
///
/// Panique si `eta1 <= 0`, si `eta2 <= 0` ou si `fctd <= 0`.
pub fn rcanchor_design_bond_stress(eta1: f64, eta2: f64, fctd: f64) -> f64 {
    assert!(
        eta1 > 0.0,
        "le coefficient d'adhérence η1 doit être strictement positif"
    );
    assert!(
        eta2 > 0.0,
        "le coefficient de diamètre η2 doit être strictement positif"
    );
    assert!(
        fctd > 0.0,
        "la résistance de calcul fctd doit être strictement positive"
    );
    2.25 * eta1 * eta2 * fctd
}

/// Longueur d'ancrage de base requise `lb,rqd = (φ / 4) · (σsd / fbd)` (mm),
/// avec `φ` en mm, `σsd` et `fbd` en MPa.
///
/// Panique si `bar_diameter <= 0`, si `steel_stress < 0` ou si
/// `bond_stress <= 0` (division par zéro).
pub fn rcanchor_basic_anchorage_length(
    bar_diameter: f64,
    steel_stress: f64,
    bond_stress: f64,
) -> f64 {
    assert!(
        bar_diameter > 0.0,
        "le diamètre de barre φ doit être strictement positif"
    );
    assert!(
        steel_stress >= 0.0,
        "la contrainte dans la barre σsd doit être ≥ 0"
    );
    assert!(
        bond_stress > 0.0,
        "la contrainte d'adhérence fbd doit être strictement positive"
    );
    (bar_diameter / 4.0) * (steel_stress / bond_stress)
}

/// Longueur d'ancrage de calcul `lbd = max(∏αi · lb,rqd , lb,min)` (mm), où
/// `alpha_product` est le produit des coefficients de forme et d'enrobage.
///
/// Panique si `basic_length < 0`, si `alpha_product <= 0` ou si
/// `minimum_length < 0`.
pub fn rcanchor_design_anchorage_length(
    basic_length: f64,
    alpha_product: f64,
    minimum_length: f64,
) -> f64 {
    assert!(
        basic_length >= 0.0,
        "la longueur d'ancrage de base lb,rqd doit être ≥ 0"
    );
    assert!(
        alpha_product > 0.0,
        "le produit des coefficients ∏αi doit être strictement positif"
    );
    assert!(
        minimum_length >= 0.0,
        "la longueur d'ancrage minimale lb,min doit être ≥ 0"
    );
    (alpha_product * basic_length).max(minimum_length)
}

/// Longueur de recouvrement `l0 = α6 · lbd` (mm), avec `α6` le coefficient de
/// recouvrement.
///
/// Panique si `design_anchorage_length < 0` ou si `alpha6 <= 0`.
pub fn rcanchor_lap_length(design_anchorage_length: f64, alpha6: f64) -> f64 {
    assert!(
        design_anchorage_length >= 0.0,
        "la longueur d'ancrage de calcul lbd doit être ≥ 0"
    );
    assert!(
        alpha6 > 0.0,
        "le coefficient de recouvrement α6 doit être strictement positif"
    );
    alpha6 * design_anchorage_length
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn bond_stress_clean_case_and_proportionality() {
        // Conditions d'adhérence bonnes (η1 = 1) et petit diamètre (η2 = 1) :
        //   fbd = 2,25 · 1 · 1 · 2,0 = 4,5 MPa
        let fbd = rcanchor_design_bond_stress(1.0, 1.0, 2.0);
        assert_relative_eq!(fbd, 4.5, epsilon = 1e-12);
        // Proportionnalité : fbd est linéaire en fctd ; doubler fctd double fbd.
        let fbd2 = rcanchor_design_bond_stress(1.0, 1.0, 4.0);
        assert_relative_eq!(fbd2 / fbd, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn basic_length_clean_case() {
        // φ = 16 mm, σsd = 400 MPa, fbd = 2,0 MPa :
        //   lb,rqd = (16 / 4) · (400 / 2) = 4 · 200 = 800 mm
        let lb = rcanchor_basic_anchorage_length(16.0, 400.0, 2.0);
        assert_relative_eq!(lb, 800.0, epsilon = 1e-9);
    }

    #[test]
    fn basic_length_proportional_to_stress() {
        // Proportionnalité : lb,rqd est linéaire en σsd ; doubler la contrainte
        // dans la barre double la longueur d'ancrage de base.
        let lb1 = rcanchor_basic_anchorage_length(20.0, 200.0, 2.7);
        let lb2 = rcanchor_basic_anchorage_length(20.0, 400.0, 2.7);
        assert_relative_eq!(lb2 / lb1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn design_length_alpha_and_minimum() {
        // Le produit des coefficients réduit la longueur, tant qu'il reste
        // au-dessus du plancher :
        //   ∏αi · lb,rqd = 0,7 · 800 = 560 mm > lb,min = 200 mm  →  560 mm
        let lbd = rcanchor_design_anchorage_length(800.0, 0.7, 200.0);
        assert_relative_eq!(lbd, 560.0, epsilon = 1e-9);
        // Cas où le plancher l'emporte :
        //   0,2 · 800 = 160 mm < lb,min = 300 mm  →  300 mm
        let lbd_min = rcanchor_design_anchorage_length(800.0, 0.2, 300.0);
        assert_relative_eq!(lbd_min, 300.0, epsilon = 1e-9);
    }

    #[test]
    fn lap_length_clean_case_and_proportionality() {
        // l0 = α6 · lbd = 1,5 · 560 = 840 mm
        let l0 = rcanchor_lap_length(560.0, 1.5);
        assert_relative_eq!(l0, 840.0, epsilon = 1e-9);
        // Proportionnalité : l0 est linéaire en α6.
        let l0_ref = rcanchor_lap_length(560.0, 1.0);
        assert_relative_eq!(l0 / l0_ref, 1.5, epsilon = 1e-12);
    }

    #[test]
    fn realistic_chained_case_c25_30() {
        // Chaîne complète pour un béton C25/30 et un acier B500 :
        //   fctd = 1,2 MPa (donné), η1 = η2 = 1,0
        //   fbd    = 2,25 · 1 · 1 · 1,2 = 2,7 MPa
        //   φ = 20 mm, σsd = 435 MPa
        //   lb,rqd = (20 / 4) · (435 / 2,7) = 5 · 161,1111… = 805,5556 mm
        //   lbd    = max(0,7 · 805,5556 ; 200) = 563,8889 mm
        //   l0     = 1,5 · 563,8889 = 845,8333 mm
        let fbd = rcanchor_design_bond_stress(1.0, 1.0, 1.2);
        assert_relative_eq!(fbd, 2.7, epsilon = 1e-12);
        let lb = rcanchor_basic_anchorage_length(20.0, 435.0, fbd);
        assert_relative_eq!(lb, 805.555_556, epsilon = 1e-3);
        let lbd = rcanchor_design_anchorage_length(lb, 0.7, 200.0);
        assert_relative_eq!(lbd, 563.888_889, epsilon = 1e-3);
        let l0 = rcanchor_lap_length(lbd, 1.5);
        assert_relative_eq!(l0, 845.833_333, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(expected = "la contrainte d'adhérence fbd doit être strictement positive")]
    fn basic_length_rejects_zero_bond_stress() {
        // fbd = 0 : division par zéro interdite.
        rcanchor_basic_anchorage_length(16.0, 400.0, 0.0);
    }
}
