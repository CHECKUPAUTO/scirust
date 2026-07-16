//! Absorption gaz-liquide — facteur d'absorption, équation de **Kremser** pour
//! une cascade d'étages théoriques, débit de liquide minimal (pincement à
//! l'entrée) et nombre d'unités de transfert d'une colonne à garnissage diluée.
//!
//! ```text
//! facteur d'absorption   A   = L / (m·G)                                   [-]
//! Kremser (N étages)     φ_A = (A^(N+1) − A) / (A^(N+1) − 1)     (A ≠ 1)   [-]
//! liquide minimal        Lₘ  = G·(y_in − y_out) / (y_in/m − x_in)         [mol·s⁻¹]
//! NTU dilué (colonne)    N_tOG = ln(y_in / y_out)                          [-]
//! ```
//!
//! `A` facteur d'absorption [sans dimension], `L` débit molaire de liquide
//! (solvant) [mol·s⁻¹], `G` débit molaire de gaz porteur [mol·s⁻¹], `m` pente
//! de la droite d'équilibre y = m·x [sans dimension], `N` nombre d'étages
//! théoriques [étages], `φ_A` fraction du soluté absorbée [sans dimension,
//! 0 ≤ φ_A ≤ 1], `Lₘ` débit molaire de liquide minimal [mol·s⁻¹], `y_in`/`y_out`
//! fractions molaires du soluté dans le gaz à l'entrée/à la sortie [sans
//! dimension], `x_in` fraction molaire du soluté dans le liquide entrant [sans
//! dimension], `N_tOG` nombre d'unités de transfert côté gaz [sans dimension].
//!
//! **Limite honnête** : ces relations valent pour une **absorption isotherme de
//! solutions diluées**, avec une **droite d'équilibre de pente `m` constante
//! FOURNIE par l'appelant** (jamais inventée ; issue de la loi de Henry, de
//! tables ou d'essais). Le modèle de **Kremser** suppose des **étages
//! théoriques** ; le rendement d'étage réel (efficacité de Murphree ou globale)
//! est **fourni** pour convertir en étages réels. Pour une **colonne à
//! garnissage**, le nombre d'unités de transfert `N_tOG` ci-dessus retient la
//! **force motrice pratiquement constante** (soluté très soluble, pente
//! négligeable) ; la **hauteur d'unité de transfert (HTU)** — donc les
//! coefficients de transfert de matière et les diffusivités — est **fournie**
//! par l'appelant. Le **débit minimal** correspond au **pincement à l'entrée**
//! (liquide sortant en équilibre avec le gaz entrant). Les **enthalpies de
//! dissolution sont négligées** (hypothèse isotherme).

/// Facteur d'absorption `A = L / (m·G)` (sans dimension).
///
/// `liquid_flow` (L) et `gas_flow` (G) en mol·s⁻¹ ; `equilibrium_slope` (m)
/// pente de la droite d'équilibre y = m·x [sans dimension].
///
/// Panique si `gas_flow <= 0`, `equilibrium_slope <= 0` ou `liquid_flow < 0`.
pub fn absorp_factor(liquid_flow: f64, gas_flow: f64, equilibrium_slope: f64) -> f64 {
    assert!(
        gas_flow > 0.0 && equilibrium_slope > 0.0 && liquid_flow >= 0.0,
        "G > 0, m > 0 et L ≥ 0 requis"
    );
    liquid_flow / (equilibrium_slope * gas_flow)
}

/// Fraction de soluté absorbée sur `N` étages théoriques (équation de
/// **Kremser**) `φ_A = (A^(N+1) − A) / (A^(N+1) − 1)`, valable pour `A ≠ 1`.
///
/// `absorption_factor` (A) sans dimension ; `theoretical_stages` (N) nombre
/// d'étages théoriques. Pour `A > 1`, `φ_A → 1` quand `N → ∞`.
///
/// Panique si `absorption_factor <= 0`, si `absorption_factor ≈ 1` (formule
/// singulière) ou si `theoretical_stages == 0`.
pub fn absorp_kremser_fraction_absorbed(absorption_factor: f64, theoretical_stages: u32) -> f64 {
    assert!(absorption_factor > 0.0, "A > 0 requis");
    assert!(
        (absorption_factor - 1.0).abs() > 1.0e-9,
        "A ≠ 1 requis (formule de Kremser singulière en A = 1)"
    );
    assert!(theoretical_stages >= 1, "N ≥ 1 étage théorique requis");
    let power = absorption_factor.powi(theoretical_stages as i32 + 1);
    (power - absorption_factor) / (power - 1.0)
}

/// Débit molaire de liquide **minimal** `Lₘ = G·(y_in − y_out) / (y_in/m − x_in)`
/// (mol·s⁻¹), correspondant au **pincement à l'entrée** du gaz (liquide sortant
/// en équilibre avec le gaz entrant : `x_out,max = y_in/m`).
///
/// `gas_flow` (G) en mol·s⁻¹ ; `equilibrium_slope` (m) sans dimension ;
/// `inlet_gas_fraction` (y_in), `outlet_gas_fraction` (y_out) et
/// `inlet_liquid_fraction` (x_in) fractions molaires [sans dimension].
///
/// Panique si `gas_flow <= 0`, `equilibrium_slope <= 0`, si les fractions ne
/// sont pas dans `[0, 1]`, si `outlet_gas_fraction > inlet_gas_fraction`
/// (absorption), ou si `y_in/m − x_in <= 0` (charge de liquide entrant
/// infaisable, au-delà de l'équilibre).
pub fn absorp_minimum_liquid_flow(
    gas_flow: f64,
    equilibrium_slope: f64,
    inlet_gas_fraction: f64,
    outlet_gas_fraction: f64,
    inlet_liquid_fraction: f64,
) -> f64 {
    assert!(
        gas_flow > 0.0 && equilibrium_slope > 0.0,
        "G > 0 et m > 0 requis"
    );
    assert!(
        (0.0..=1.0).contains(&inlet_gas_fraction)
            && (0.0..=1.0).contains(&outlet_gas_fraction)
            && (0.0..=1.0).contains(&inlet_liquid_fraction),
        "fractions molaires dans [0, 1] requises"
    );
    assert!(
        inlet_gas_fraction >= outlet_gas_fraction,
        "y_in ≥ y_out requis (le gaz cède du soluté)"
    );
    let driving = inlet_gas_fraction / equilibrium_slope - inlet_liquid_fraction;
    assert!(
        driving > 0.0,
        "y_in/m − x_in > 0 requis (liquide entrant sous l'équilibre du gaz entrant)"
    );
    gas_flow * (inlet_gas_fraction - outlet_gas_fraction) / driving
}

/// Nombre d'unités de transfert côté gaz `N_tOG = ln(y_in / y_out)` (sans
/// dimension), colonne à garnissage en **solution diluée** à **force motrice
/// pratiquement constante**.
///
/// `inlet_fraction` (y_in) et `outlet_fraction` (y_out) fractions molaires du
/// soluté dans le gaz à l'entrée/à la sortie [sans dimension].
///
/// Panique si `inlet_fraction <= 0` ou `outlet_fraction <= 0`.
pub fn absorp_ntu_dilute(inlet_fraction: f64, outlet_fraction: f64) -> f64 {
    assert!(
        inlet_fraction > 0.0 && outlet_fraction > 0.0,
        "y_in > 0 et y_out > 0 requis"
    );
    (inlet_fraction / outlet_fraction).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn factor_definition_and_reciprocity() {
        // L = 100, G = 50, m = 0.5 ⇒ A = 100/(0.5·50) = 100/25 = 4.
        let a = absorp_factor(100.0_f64, 50.0_f64, 0.5_f64);
        assert_relative_eq!(a, 4.0, max_relative = 1e-12);
        // Réciprocité : A·(m·G) doit redonner L.
        assert_relative_eq!(a * (0.5_f64 * 50.0_f64), 100.0, max_relative = 1e-12);
    }

    #[test]
    fn kremser_realistic_case() {
        // A = 2, N = 3 : A^(N+1) = 2^4 = 16 ; φ_A = (16 − 2)/(16 − 1) = 14/15.
        assert_relative_eq!(
            absorp_kremser_fraction_absorbed(2.0_f64, 3),
            14.0_f64 / 15.0_f64,
            max_relative = 1e-12
        );
    }

    #[test]
    fn kremser_tends_to_full_absorption() {
        // Pour A > 1, un très grand N absorbe la quasi-totalité du soluté (φ_A → 1).
        let phi = absorp_kremser_fraction_absorbed(2.0_f64, 30);
        assert!(phi < 1.0, "la fraction absorbée reste < 1");
        assert_relative_eq!(phi, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn minimum_liquid_flow_pinch_case() {
        // G = 10, m = 2, y_in = 0.1, y_out = 0.01, x_in = 0 (solvant pur) :
        // Lₘ = 10·(0.1 − 0.01)/(0.1/2 − 0) = 10·0.09/0.05 = 0.9/0.05 = 18.
        assert_relative_eq!(
            absorp_minimum_liquid_flow(10.0_f64, 2.0_f64, 0.1_f64, 0.01_f64, 0.0_f64),
            18.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn ntu_matches_log_ratio_and_reciprocity() {
        // y_in = 0.1, y_out = 0.01 ⇒ N_tOG = ln(10) ≈ 2.302585.
        assert_relative_eq!(
            absorp_ntu_dilute(0.1_f64, 0.01_f64),
            core::f64::consts::LN_10,
            max_relative = 1e-12
        );
        // Force motrice nulle : y_in = y_out ⇒ ln(1) = 0.
        assert_relative_eq!(absorp_ntu_dilute(0.05_f64, 0.05_f64), 0.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "A ≠ 1 requis")]
    fn kremser_panics_at_unit_factor() {
        // A = 1 rend la formule de Kremser singulière (0/0) ⇒ panique.
        let _ = absorp_kremser_fraction_absorbed(1.0_f64, 3);
    }
}
