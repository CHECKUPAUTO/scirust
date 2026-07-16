//! Colonne à garnissage — hauteur par **HTU/NTU** : nombre d'unités de transfert
//! (dilué et forme complète avec facteur d'absorption), hauteur d'unité de
//! transfert et hauteur de garnissage.
//!
//! ```text
//! NTU dilué        N = ln(y_in / y_out)                                        [-]
//! NTU (facteur A)  N = 1/(1 − 1/A) · ln[ (y_in − y*)/(y_out − y*)·(1 − 1/A) + 1/A ]  [-]
//! HTU              H = G / (K_G·a)                                             [m]
//! hauteur          Z = H · N                                                   [m]
//! ```
//!
//! `N` nombre d'unités de transfert côté gaz [sans dimension], `y_in`/`y_out`
//! fractions molaires du soluté dans le gaz à l'entrée/à la sortie [sans
//! dimension], `y*` fraction molaire d'équilibre (droite d'équilibre de pente
//! constante) [sans dimension], `A` facteur d'absorption `A = L/(m·G)` [sans
//! dimension], `H` hauteur d'unité de transfert [m], `G` flux molaire surfacique
//! de gaz [mol·s⁻¹·m⁻²], `K_G·a` coefficient global de transfert volumique
//! [mol·s⁻¹·m⁻³] (produit du coefficient global `K_G` [mol·s⁻¹·m⁻²] par l'aire
//! interfaciale spécifique `a` [m²·m⁻³ = m⁻¹]), `Z` hauteur de garnissage [m].
//!
//! **Limite honnête** : ces relations valent pour une **colonne à garnissage en
//! contre-courant**, avec une **droite d'équilibre de pente constante FOURNIE**
//! par l'appelant (jamais inventée ; issue de la loi de Henry, de tables ou
//! d'essais) résumée par le **facteur d'absorption `A`** ou la fraction
//! d'équilibre `y*`. Le **coefficient global de transfert volumique `K_G·a`** et
//! le **flux molaire surfacique `G`** sont eux aussi **FOURNIS** (corrélations de
//! garnissage, essais pilotes) et jamais devinés. Le **NTU dilué** néglige la
//! droite d'équilibre (force motrice logarithmique, soluté très soluble) ; la
//! **forme complète** utilise le **facteur d'absorption `A`**. La décomposition
//! `Z = H·N` sépare la **résistance au transfert (HTU)** de la **difficulté de
//! séparation (NTU)** ; les **enthalpies de dissolution sont négligées**
//! (hypothèse isotherme, solutions diluées).

/// Nombre d'unités de transfert côté gaz en **solution diluée**
/// `N = ln(y_in / y_out)` (sans dimension), équilibre négligé (force motrice
/// logarithmique).
///
/// `inlet_fraction` (y_in) et `outlet_fraction` (y_out) fractions molaires du
/// soluté dans le gaz à l'entrée/à la sortie [sans dimension].
///
/// Panique si `inlet_fraction <= 0` ou `outlet_fraction <= 0`.
pub fn htu_number_of_transfer_units_dilute(inlet_fraction: f64, outlet_fraction: f64) -> f64 {
    assert!(
        inlet_fraction > 0.0 && outlet_fraction > 0.0,
        "y_in > 0 et y_out > 0 requis"
    );
    (inlet_fraction / outlet_fraction).ln()
}

/// Nombre d'unités de transfert côté gaz avec **droite d'équilibre** et **facteur
/// d'absorption** `A` :
/// `N = 1/(1 − 1/A) · ln[ (y_in − y*)/(y_out − y*)·(1 − 1/A) + 1/A ]` (sans
/// dimension).
///
/// `inlet_fraction` (y_in), `outlet_fraction` (y_out) et `equilibrium_fraction`
/// (y*) fractions molaires [sans dimension] ; `absorption_factor` (A = L/(m·G))
/// facteur d'absorption [sans dimension].
///
/// Panique si `absorption_factor <= 0`, si `absorption_factor ≈ 1` (terme
/// `1 − 1/A` singulier), si `outlet_fraction ≈ equilibrium_fraction` (division
/// par la force motrice de sortie nulle), ou si l'argument du logarithme est
/// `<= 0`.
pub fn htu_number_of_transfer_units_absorption(
    inlet_fraction: f64,
    outlet_fraction: f64,
    equilibrium_fraction: f64,
    absorption_factor: f64,
) -> f64 {
    assert!(absorption_factor > 0.0, "A > 0 requis");
    assert!(
        (absorption_factor - 1.0).abs() > 1.0e-9,
        "A ≠ 1 requis (terme 1 − 1/A singulier en A = 1)"
    );
    let outlet_driving = outlet_fraction - equilibrium_fraction;
    assert!(
        outlet_driving.abs() > 1.0e-12,
        "y_out ≠ y* requis (force motrice de sortie non nulle)"
    );
    let one_minus_inv_a = 1.0 - 1.0 / absorption_factor;
    let ratio = (inlet_fraction - equilibrium_fraction) / outlet_driving;
    let log_argument = ratio * one_minus_inv_a + 1.0 / absorption_factor;
    assert!(
        log_argument > 0.0,
        "argument du logarithme > 0 requis (force motrice cohérente)"
    );
    (1.0 / one_minus_inv_a) * log_argument.ln()
}

/// Hauteur d'unité de transfert `H = G / (K_G·a)` (m).
///
/// `molar_gas_flux` (G) flux molaire surfacique de gaz [mol·s⁻¹·m⁻²] ;
/// `overall_coefficient` (K_G) coefficient global de transfert [mol·s⁻¹·m⁻²] ;
/// `interfacial_area` (a) aire interfaciale spécifique [m²·m⁻³ = m⁻¹].
///
/// Panique si `molar_gas_flux < 0`, `overall_coefficient <= 0` ou
/// `interfacial_area <= 0`.
pub fn htu_height_of_transfer_unit(
    molar_gas_flux: f64,
    overall_coefficient: f64,
    interfacial_area: f64,
) -> f64 {
    assert!(molar_gas_flux >= 0.0, "G ≥ 0 requis");
    assert!(
        overall_coefficient > 0.0 && interfacial_area > 0.0,
        "K_G > 0 et a > 0 requis"
    );
    molar_gas_flux / (overall_coefficient * interfacial_area)
}

/// Hauteur de garnissage `Z = H · N` (m), produit de la hauteur d'unité de
/// transfert par le nombre d'unités de transfert.
///
/// `height_of_transfer_unit` (H) en m ; `number_of_transfer_units` (N) sans
/// dimension.
///
/// Panique si `height_of_transfer_unit < 0` ou `number_of_transfer_units < 0`.
pub fn htu_packing_height(height_of_transfer_unit: f64, number_of_transfer_units: f64) -> f64 {
    assert!(
        height_of_transfer_unit >= 0.0 && number_of_transfer_units >= 0.0,
        "H ≥ 0 et N ≥ 0 requis"
    );
    height_of_transfer_unit * number_of_transfer_units
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn dilute_ntu_matches_log_ratio() {
        // y_in = 0.1, y_out = 0.01 ⇒ N = ln(10) = LN_10 ≈ 2.302585.
        assert_relative_eq!(
            htu_number_of_transfer_units_dilute(0.1_f64, 0.01_f64),
            core::f64::consts::LN_10,
            max_relative = 1e-12
        );
        // Force motrice nulle : y_in = y_out ⇒ ln(1) = 0.
        assert_relative_eq!(
            htu_number_of_transfer_units_dilute(0.05_f64, 0.05_f64),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn absorption_ntu_realistic_case() {
        // A = 2 ⇒ 1/A = 0.5, 1 − 1/A = 0.5.
        // y_in = 0.03, y_out = 0.01, y* = 0 ⇒ ratio = 0.03/0.01 = 3.
        // argument = 3·0.5 + 0.5 = 2 ⇒ N = (1/0.5)·ln(2) = 2·LN_2 ≈ 1.386294.
        assert_relative_eq!(
            htu_number_of_transfer_units_absorption(0.03_f64, 0.01_f64, 0.0_f64, 2.0_f64),
            2.0_f64 * core::f64::consts::LN_2,
            max_relative = 1e-12
        );
    }

    #[test]
    fn absorption_ntu_reduces_to_dilute_when_factor_large() {
        // Quand A → ∞ (1/A → 0), la forme complète tend vers le NTU dilué avec
        // équilibre nul : N → ln(y_in/y_out).
        let full = htu_number_of_transfer_units_absorption(0.1_f64, 0.01_f64, 0.0_f64, 1.0e9_f64);
        let dilute = htu_number_of_transfer_units_dilute(0.1_f64, 0.01_f64);
        assert_relative_eq!(full, dilute, max_relative = 1e-3);
    }

    #[test]
    fn height_of_transfer_unit_definition() {
        // G = 2, K_G = 0.04, a = 100 ⇒ H = 2/(0.04·100) = 2/4 = 0.5.
        assert_relative_eq!(
            htu_height_of_transfer_unit(2.0_f64, 0.04_f64, 100.0_f64),
            0.5,
            max_relative = 1e-12
        );
        // Flux nul ⇒ hauteur d'unité nulle.
        assert_relative_eq!(
            htu_height_of_transfer_unit(0.0_f64, 0.04_f64, 100.0_f64),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn packing_height_is_htu_times_ntu() {
        // H = 0.5 m, N = 2·LN_2 ⇒ Z = 0.5·2·LN_2 = LN_2 ≈ 0.693147.
        let ntu = htu_number_of_transfer_units_absorption(0.03_f64, 0.01_f64, 0.0_f64, 2.0_f64);
        let htu = htu_height_of_transfer_unit(2.0_f64, 0.04_f64, 100.0_f64);
        assert_relative_eq!(
            htu_packing_height(htu, ntu),
            core::f64::consts::LN_2,
            max_relative = 1e-12
        );
        // Proportionnalité : doubler le HTU double la hauteur.
        assert_relative_eq!(
            htu_packing_height(2.0 * htu, ntu),
            2.0 * htu_packing_height(htu, ntu),
            max_relative = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "A ≠ 1 requis")]
    fn absorption_ntu_panics_at_unit_factor() {
        // A = 1 annule 1 − 1/A ⇒ terme singulier ⇒ panique.
        let _ = htu_number_of_transfer_units_absorption(0.03_f64, 0.01_f64, 0.0_f64, 1.0_f64);
    }
}
