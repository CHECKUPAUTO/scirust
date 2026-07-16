//! **Charpente métallique — déversement d'une poutre fléchie
//! (Eurocode 3, EN 1993-1-1 §6.3.2)** : élancement réduit de déversement `λ̄LT`,
//! facteur de réduction `χLT` (méthode des sections laminées ou équivalentes
//! soudées, avec plateau `λ̄LT,0` et coefficient `β`), puis moment résistant de
//! calcul au déversement `Mb,Rd` d'une poutre non maintenue latéralement.
//!
//! ```text
//! élancement réduit         λ̄LT  = √(Wpl·fy / Mcr)               (classes 1-2)
//! coefficient intermédiaire ΦLT  = 0,5·(1 + αLT·(λ̄LT − λ̄LT,0) + β·λ̄LT²)
//! facteur de réduction      χLT  = 1 / (ΦLT + √(ΦLT² − β·λ̄LT²))  (plafonné à 1,0)
//! moment résistant          Mb,Rd = χLT·Wpl·fy / γM1
//! ```
//!
//! `Wpl` = `plastic_modulus` module de flexion plastique de la section (mm³, pour
//! une section de classe 1 ou 2), `fy` limite d'élasticité de l'acier (MPa =
//! N/mm²), `Mcr` = `elastic_critical_moment` moment critique élastique de
//! déversement (N·mm), `λ̄LT` = `non_dimensional_slenderness` élancement réduit de
//! déversement (sans dimension), `αLT` = `imperfection_factor` facteur
//! d'imperfection de la courbe de déversement (sans dimension), `λ̄LT,0` =
//! `plateau_length` longueur du plateau (sans dimension), `β` = `beta` coefficient
//! de la méthode des sections laminées (sans dimension), `ΦLT` grandeur
//! intermédiaire (sans dimension), `χLT` = `reduction_factor` facteur de réduction
//! (sans dimension, `0 < χLT ≤ 1`), `γM1` = `gamma_m1` coefficient partiel de
//! sécurité (sans dimension), `Mb,Rd` moment résistant de calcul au déversement
//! (N·mm).
//!
//! **Convention** : N, mm, MPa (avec `1 MPa = 1 N/mm²`, donc `Wpl·fy`, `Mcr` et
//! `Mb,Rd` sont en N·mm) ; `λ̄LT`, `αLT`, `λ̄LT,0`, `β`, `ΦLT`, `χLT` et `γM1` sont
//! sans dimension.
//! **Limite honnête** : ce module traite le seul **déversement** (flambement
//! latéral avec torsion) d'une poutre fléchie de section de **classe 1 ou 2** (le
//! module `Wpl` employé est le module plastique ; une classe 3 utiliserait `Wel`
//! et une classe 4 le module efficace `Weff`). Le **moment critique élastique
//! `Mcr`** n'est **pas** calculé ici : il dépend du diagramme de moment, des
//! conditions de maintien et de la géométrie de la section, et est **fourni** par
//! l'appelant (Eurocode 3, méthode générale ou formule adaptée). Le facteur
//! d'imperfection `αLT` de la courbe de déversement choisie, la longueur de
//! plateau `λ̄LT,0`, le coefficient `β`, la limite d'élasticité caractéristique
//! `fy`, le module plastique `Wpl` et le coefficient partiel `γM1` sont **fournis
//! par l'appelant** d'après l'Eurocode et son Annexe Nationale ; aucune valeur
//! « par défaut » n'est inventée.

/// Élancement réduit de déversement `λ̄LT = √(Wpl·fy / Mcr)` (sans dimension, pour
/// les sections de classe 1 ou 2), avec `plastic_modulus` = `Wpl` le module de
/// flexion plastique en mm³, `fy` la limite d'élasticité en MPa et
/// `elastic_critical_moment` = `Mcr` le moment critique élastique en N·mm
/// (`Wpl·fy` est en N·mm).
///
/// Panique si `plastic_modulus <= 0`, `fy <= 0` ou `elastic_critical_moment <= 0`.
pub fn steellt_non_dimensional_slenderness(
    plastic_modulus: f64,
    fy: f64,
    elastic_critical_moment: f64,
) -> f64 {
    assert!(
        plastic_modulus > 0.0,
        "le module plastique Wpl doit être strictement positif (mm³)"
    );
    assert!(
        fy > 0.0,
        "la limite d'élasticité fy doit être strictement positive (MPa)"
    );
    assert!(
        elastic_critical_moment > 0.0,
        "le moment critique élastique Mcr doit être strictement positif (N·mm)"
    );
    (plastic_modulus * fy / elastic_critical_moment).sqrt()
}

/// Facteur de réduction au déversement `χLT` (sans dimension, plafonné à `1,0`) :
/// `ΦLT = 0,5·(1 + αLT·(λ̄LT − λ̄LT,0) + β·λ̄LT²)` puis
/// `χLT = 1 / (ΦLT + √(ΦLT² − β·λ̄LT²))`, avec
/// `non_dimensional_slenderness` = `λ̄LT` l'élancement réduit de déversement,
/// `imperfection_factor` = `αLT` le facteur d'imperfection de la courbe choisie,
/// `plateau_length` = `λ̄LT,0` la longueur du plateau et `beta` = `β` le
/// coefficient de la méthode des sections laminées.
///
/// Panique si `non_dimensional_slenderness < 0`, `imperfection_factor < 0`,
/// `plateau_length < 0` ou `beta <= 0`.
pub fn steellt_reduction_factor(
    non_dimensional_slenderness: f64,
    imperfection_factor: f64,
    plateau_length: f64,
    beta: f64,
) -> f64 {
    assert!(
        non_dimensional_slenderness >= 0.0,
        "l'élancement réduit λ̄LT doit être ≥ 0"
    );
    assert!(
        imperfection_factor >= 0.0,
        "le facteur d'imperfection αLT doit être ≥ 0"
    );
    assert!(
        plateau_length >= 0.0,
        "la longueur de plateau λ̄LT,0 doit être ≥ 0"
    );
    assert!(beta > 0.0, "le coefficient β doit être strictement positif");
    let lambda_bar = non_dimensional_slenderness;
    let phi = 0.5
        * (1.0
            + imperfection_factor * (lambda_bar - plateau_length)
            + beta * lambda_bar * lambda_bar);
    let chi = 1.0 / (phi + (phi * phi - beta * lambda_bar * lambda_bar).sqrt());
    chi.min(1.0)
}

/// Moment résistant de calcul au déversement `Mb,Rd = χLT·Wpl·fy / γM1` (N·mm),
/// avec `reduction_factor` = `χLT` le facteur de réduction (sans dimension),
/// `plastic_modulus` = `Wpl` le module plastique en mm³, `fy` la limite
/// d'élasticité en MPa et `gamma_m1` = `γM1` le coefficient partiel de sécurité
/// (sans dimension).
///
/// Panique si `reduction_factor < 0`, `plastic_modulus <= 0`, `fy <= 0` ou
/// `gamma_m1 <= 0`.
pub fn steellt_buckling_resistance_moment(
    reduction_factor: f64,
    plastic_modulus: f64,
    fy: f64,
    gamma_m1: f64,
) -> f64 {
    assert!(
        reduction_factor >= 0.0,
        "le facteur de réduction χLT doit être ≥ 0"
    );
    assert!(
        plastic_modulus > 0.0,
        "le module plastique Wpl doit être strictement positif (mm³)"
    );
    assert!(
        fy > 0.0,
        "la limite d'élasticité fy doit être strictement positive (MPa)"
    );
    assert!(
        gamma_m1 > 0.0,
        "le coefficient partiel γM1 doit être strictement positif"
    );
    reduction_factor * plastic_modulus * fy / gamma_m1
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn slenderness_identity_and_unit_case() {
        // Identité : λ̄LT²·Mcr = Wpl·fy, quelles que soient les entrées.
        let wpl = 1.0e6_f64; // mm³
        let fy = 355.0_f64; // MPa
        let mcr = 5.0e8_f64; // N·mm
        let lambda_bar = steellt_non_dimensional_slenderness(wpl, fy, mcr);
        assert_relative_eq!(lambda_bar.powi(2) * mcr, wpl * fy, epsilon = 1e-3);
        // Cas limite : si Mcr = Wpl·fy, alors λ̄LT = √1 = 1 exactement.
        let squash_moment = wpl * fy;
        assert_relative_eq!(
            steellt_non_dimensional_slenderness(wpl, fy, squash_moment),
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn slenderness_worked_value() {
        // Wpl = 1e6 mm³, fy = 355 MPa, Mcr = 5e8 N·mm.
        // Wpl·fy = 3,55e8 N·mm ; rapport = 3,55e8 / 5e8 = 0,71 ;
        // λ̄LT = √0,71 = 0,8426149…
        let lambda_bar = steellt_non_dimensional_slenderness(1.0e6, 355.0, 5.0e8);
        assert_relative_eq!(lambda_bar, 0.842_615, epsilon = 1e-3);
    }

    #[test]
    fn reduction_factor_unity_at_plateau() {
        // À λ̄LT = λ̄LT,0, on a ΦLT = 0,5·(1 + β·λ̄LT,0²) et
        // √(ΦLT² − β·λ̄LT,0²) = 1 − ΦLT, donc χLT = 1/(ΦLT + 1 − ΦLT) = 1,0
        // quel que soit le facteur d'imperfection αLT.
        // Vérif chiffrée : λ̄LT,0 = 0,4, β = 0,75 → ΦLT = 0,5·(1 + 0,12) = 0,56,
        // √(0,56² − 0,75·0,16) = √(0,3136 − 0,12) = √0,1936 = 0,44 ;
        // χLT = 1/(0,56 + 0,44) = 1,0.
        assert_relative_eq!(
            steellt_reduction_factor(0.4, 0.34, 0.4, 0.75),
            1.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            steellt_reduction_factor(0.4, 0.76, 0.4, 0.75),
            1.0,
            epsilon = 1e-12
        );
        // En deçà du plateau (λ̄LT < λ̄LT,0) le facteur est plafonné à 1,0.
        assert_relative_eq!(
            steellt_reduction_factor(0.2, 0.49, 0.4, 0.75),
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn reduction_factor_worked_rolled_section() {
        // Section laminée, courbe b (αLT = 0,34), λ̄LT,0 = 0,4, β = 0,75, λ̄LT = 1,0 :
        // ΦLT = 0,5·(1 + 0,34·(1,0 − 0,4) + 0,75·1,0²)
        //     = 0,5·(1 + 0,204 + 0,75) = 0,5·1,954 = 0,977 ;
        // √(0,977² − 0,75·1,0²) = √(0,954529 − 0,75) = √0,204529 = 0,452249 ;
        // χLT = 1/(0,977 + 0,452249) = 1/1,429249 = 0,699668.
        let chi = steellt_reduction_factor(1.0, 0.34, 0.4, 0.75);
        assert_relative_eq!(chi, 0.699_668, epsilon = 1e-3);
    }

    #[test]
    fn buckling_moment_reduces_plastic_moment() {
        // Mb,Rd = χLT·(Wpl·fy/γM1). Avec χLT = 1 et γM1 = 1 on retrouve le moment
        // résistant plastique Mpl,Rd = Wpl·fy, et Mb,Rd est proportionnel à χLT.
        let wpl = 1.0e6_f64; // mm³
        let fy = 355.0_f64; // MPa
        let gamma = 1.0_f64;
        let mpl = steellt_buckling_resistance_moment(1.0, wpl, fy, gamma);
        assert_relative_eq!(mpl, wpl * fy / gamma, epsilon = 1e-6);
        let chi = 0.699_668_f64;
        let mb = steellt_buckling_resistance_moment(chi, wpl, fy, gamma);
        assert_relative_eq!(mb, chi * mpl, epsilon = 1e-3);
    }

    #[test]
    fn buckling_moment_scales_inverse_with_gamma() {
        // Mb,Rd ∝ 1/γM1 : passer de γM1 = 1,0 à 1,10 réduit Mb,Rd du facteur 1,10.
        let m10 = steellt_buckling_resistance_moment(0.7, 1.0e6, 355.0, 1.0);
        let m11 = steellt_buckling_resistance_moment(0.7, 1.0e6, 355.0, 1.10);
        assert_relative_eq!(m10 / m11, 1.10, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "le moment critique élastique Mcr doit être strictement positif")]
    fn slenderness_rejects_non_positive_mcr() {
        // Moment critique nul : division par zéro, entrée refusée.
        steellt_non_dimensional_slenderness(1.0e6, 355.0, 0.0);
    }
}
