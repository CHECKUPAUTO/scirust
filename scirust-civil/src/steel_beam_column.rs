//! **Charpente métallique — barre en flexion composée (Eurocode 3,
//! EN 1993-1-1 §6.3.3)** : taux de travail axial `N/NRd`, taux de travail en
//! flexion `M/MRd`, puis vérification de l'interaction flexion + effort normal,
//! d'abord par un critère linéaire simplifié conservatif, ensuite par la formule
//! de stabilité 6.61/6.62 employant les facteurs d'interaction `k` fournis.
//!
//! ```text
//! taux axial            uN   = NEd / NRd
//! taux de flexion       uM   = MEd / MRd
//! critère linéaire      Ulin = uN + uMy + uMz                         (≤ 1 conservatif)
//! critère de stabilité  Usta = uN + kyy·uMy + kzz·uMz                 (6.61 / 6.62)
//! ```
//!
//! `NEd` = `design_axial` effort normal de calcul (N), `NRd` = `axial_resistance`
//! résistance axiale de calcul (N, résistance de section `Npl,Rd` ou de flambement
//! `Nb,Rd` selon le cas), `MEd` = `design_moment` moment de calcul (N·mm), `MRd` =
//! `moment_resistance` moment résistant de calcul (N·mm, résistance de section
//! `Mc,Rd` ou moment résistant au déversement `Mb,Rd`), `uN` =
//! `axial_utilisation` taux de travail axial (sans dimension), `uMy` / `uMz` =
//! `moment_utilisation_y` / `moment_utilisation_z` taux de travail en flexion
//! autour des axes fort `y` et faible `z` (sans dimension), `kyy` / `kzz` =
//! `interaction_factor_y` / `interaction_factor_z` facteurs d'interaction (sans
//! dimension), `Ulin` / `Usta` taux d'interaction (sans dimension, la vérification
//! est satisfaite si `≤ 1`).
//!
//! **Convention** : N, mm, MPa (avec `1 MPa = 1 N/mm²`, donc `NEd`, `NRd` sont en
//! N et `MEd`, `MRd` en N·mm) ; les taux `uN`, `uMy`, `uMz`, les facteurs `kyy`,
//! `kzz` et les critères `Ulin`, `Usta` sont sans dimension.
//! **Limite honnête** : ce module ne réalise que l'**assemblage** des taux de
//! travail d'une barre en **flexion composée**. Les résistances de section et de
//! flambement `NRd` (via `steel_compression`) et `MRd` (via
//! `steel_lateral_torsional`) sont **fournies par l'appelant**, de même que les
//! **facteurs d'interaction** `kyy` et `kzz` : ces derniers dépendent des facteurs
//! de moment uniforme équivalent `Cmy`, `Cmz`, `CmLT`, des élancements réduits et
//! de la classe de section, et résultent du calcul complet de l'Annexe B ou C de
//! l'EN 1993-1-1 (aucune valeur « par défaut » n'est inventée ici). Le **critère
//! linéaire** (`uN + uMy + uMz ≤ 1`) est une borne **conservative** ; la
//! **formule de stabilité** 6.61/6.62 requiert impérativement les facteurs `k`
//! issus de ce calcul complet. La limite d'élasticité caractéristique `fy` et les
//! coefficients partiels `γM0`, `γM1` sont fournis à l'appelant par l'Eurocode et
//! son Annexe Nationale.

/// Taux de travail axial `uN = NEd / NRd` (sans dimension), avec `design_axial` =
/// `NEd` l'effort normal de calcul en N et `axial_resistance` = `NRd` la
/// résistance axiale de calcul en N (résistance de section ou de flambement).
///
/// Panique si `design_axial < 0` ou `axial_resistance <= 0`.
pub fn steelbc_axial_utilisation(design_axial: f64, axial_resistance: f64) -> f64 {
    assert!(
        design_axial >= 0.0,
        "l'effort normal de calcul NEd doit être ≥ 0 (N)"
    );
    assert!(
        axial_resistance > 0.0,
        "la résistance axiale NRd doit être strictement positive (N)"
    );
    design_axial / axial_resistance
}

/// Taux de travail en flexion `uM = MEd / MRd` (sans dimension), avec
/// `design_moment` = `MEd` le moment de calcul en N·mm et `moment_resistance` =
/// `MRd` le moment résistant de calcul en N·mm (résistance de section ou moment
/// résistant au déversement).
///
/// Panique si `design_moment < 0` ou `moment_resistance <= 0`.
pub fn steelbc_bending_utilisation(design_moment: f64, moment_resistance: f64) -> f64 {
    assert!(
        design_moment >= 0.0,
        "le moment de calcul MEd doit être ≥ 0 (N·mm)"
    );
    assert!(
        moment_resistance > 0.0,
        "le moment résistant MRd doit être strictement positif (N·mm)"
    );
    design_moment / moment_resistance
}

/// Critère d'interaction linéaire simplifié conservatif
/// `Ulin = uN + uMy + uMz` (sans dimension ; la vérification est satisfaite si
/// `Ulin ≤ 1`), avec `axial_utilisation` = `uN` le taux axial et
/// `moment_utilisation_y` / `moment_utilisation_z` = `uMy` / `uMz` les taux de
/// flexion autour des axes fort et faible.
///
/// Panique si `axial_utilisation < 0`, `moment_utilisation_y < 0` ou
/// `moment_utilisation_z < 0`.
pub fn steelbc_linear_interaction(
    axial_utilisation: f64,
    moment_utilisation_y: f64,
    moment_utilisation_z: f64,
) -> f64 {
    assert!(
        axial_utilisation >= 0.0,
        "le taux de travail axial uN doit être ≥ 0"
    );
    assert!(
        moment_utilisation_y >= 0.0,
        "le taux de flexion uMy doit être ≥ 0"
    );
    assert!(
        moment_utilisation_z >= 0.0,
        "le taux de flexion uMz doit être ≥ 0"
    );
    axial_utilisation + moment_utilisation_y + moment_utilisation_z
}

/// Critère d'interaction de stabilité EN 1993-1-1 (6.61 / 6.62)
/// `Usta = uN + kyy·uMy + kzz·uMz` (sans dimension ; la vérification est satisfaite
/// si `Usta ≤ 1`), avec `axial_utilisation` = `uN` le taux axial,
/// `interaction_factor_y` / `interaction_factor_z` = `kyy` / `kzz` les facteurs
/// d'interaction fournis et `moment_utilisation_y` / `moment_utilisation_z` =
/// `uMy` / `uMz` les taux de flexion.
///
/// Panique si `axial_utilisation < 0`, `interaction_factor_y < 0`,
/// `moment_utilisation_y < 0`, `interaction_factor_z < 0` ou
/// `moment_utilisation_z < 0`.
pub fn steelbc_stability_interaction(
    axial_utilisation: f64,
    interaction_factor_y: f64,
    moment_utilisation_y: f64,
    interaction_factor_z: f64,
    moment_utilisation_z: f64,
) -> f64 {
    assert!(
        axial_utilisation >= 0.0,
        "le taux de travail axial uN doit être ≥ 0"
    );
    assert!(
        interaction_factor_y >= 0.0,
        "le facteur d'interaction kyy doit être ≥ 0"
    );
    assert!(
        moment_utilisation_y >= 0.0,
        "le taux de flexion uMy doit être ≥ 0"
    );
    assert!(
        interaction_factor_z >= 0.0,
        "le facteur d'interaction kzz doit être ≥ 0"
    );
    assert!(
        moment_utilisation_z >= 0.0,
        "le taux de flexion uMz doit être ≥ 0"
    );
    axial_utilisation
        + interaction_factor_y * moment_utilisation_y
        + interaction_factor_z * moment_utilisation_z
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn axial_utilisation_worked_value_and_reciprocity() {
        // NEd = 500 000 N, NRd = 2 000 000 N → uN = 0,25.
        let un = steelbc_axial_utilisation(500_000.0, 2_000_000.0);
        assert_relative_eq!(un, 0.25, epsilon = 1e-12);
        // Réciprocité : uN·NRd = NEd.
        assert_relative_eq!(un * 2_000_000.0, 500_000.0, epsilon = 1e-6);
        // Cas limite : NEd = NRd → uN = 1 exactement (section pleinement sollicitée).
        assert_relative_eq!(
            steelbc_axial_utilisation(2_000_000.0, 2_000_000.0),
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn bending_utilisation_worked_value_and_scaling() {
        // MEd = 100e6 N·mm, MRd = 250e6 N·mm → uMy = 0,4.
        let um = steelbc_bending_utilisation(100.0e6, 250.0e6);
        assert_relative_eq!(um, 0.4, epsilon = 1e-12);
        // Proportionnalité : doubler MEd double le taux de flexion.
        let um2 = steelbc_bending_utilisation(200.0e6, 250.0e6);
        assert_relative_eq!(um2, 2.0 * um, epsilon = 1e-12);
        // Cas nul : MEd = 0 → taux nul.
        assert_relative_eq!(
            steelbc_bending_utilisation(0.0, 250.0e6),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn linear_interaction_sum_worked_case() {
        // uN = 0,25 ; uMy = 0,40 ; uMz = 0,10 → Ulin = 0,75 (≤ 1, vérifié).
        let ulin = steelbc_linear_interaction(0.25, 0.40, 0.10);
        assert_relative_eq!(ulin, 0.75, epsilon = 1e-12);
        // Cohérence avec les taux élémentaires recalculés depuis les efforts.
        let un = steelbc_axial_utilisation(500_000.0, 2_000_000.0);
        let umy = steelbc_bending_utilisation(100.0e6, 250.0e6);
        let umz = steelbc_bending_utilisation(25.0e6, 250.0e6); // 0,10
        assert_relative_eq!(
            steelbc_linear_interaction(un, umy, umz),
            0.75,
            epsilon = 1e-12
        );
    }

    #[test]
    fn stability_reduces_to_linear_when_factors_unity() {
        // Avec kyy = kzz = 1, la formule de stabilité redonne le critère linéaire.
        let ulin = steelbc_linear_interaction(0.25, 0.40, 0.10);
        let usta = steelbc_stability_interaction(0.25, 1.0, 0.40, 1.0, 0.10);
        assert_relative_eq!(usta, ulin, epsilon = 1e-12);
    }

    #[test]
    fn stability_interaction_worked_case() {
        // uN = 0,25 ; kyy = 1,0 ; uMy = 0,40 ; kzz = 0,60 ; uMz = 0,10 :
        // Usta = 0,25 + 1,0·0,40 + 0,60·0,10 = 0,25 + 0,40 + 0,06 = 0,71 (≤ 1).
        let usta = steelbc_stability_interaction(0.25, 1.0, 0.40, 0.60, 0.10);
        assert_relative_eq!(usta, 0.71, epsilon = 1e-12);
    }

    #[test]
    fn stability_factors_amplify_bending_terms() {
        // Des facteurs k > 1 amplifient les termes de flexion : Usta ≥ Ulin.
        let ulin = steelbc_linear_interaction(0.20, 0.30, 0.15);
        let usta = steelbc_stability_interaction(0.20, 1.4, 0.30, 1.2, 0.15);
        // Usta = 0,20 + 1,4·0,30 + 1,2·0,15 = 0,20 + 0,42 + 0,18 = 0,80.
        assert_relative_eq!(usta, 0.80, epsilon = 1e-12);
        assert!(usta >= ulin);
    }

    #[test]
    #[should_panic(expected = "la résistance axiale NRd doit être strictement positive")]
    fn axial_utilisation_rejects_non_positive_resistance() {
        // Résistance axiale nulle : division par zéro, entrée refusée.
        steelbc_axial_utilisation(500_000.0, 0.0);
    }
}
