//! **Air comprimé** — gaz parfait et travail de compression : masse volumique,
//! travail isotherme/adiabatique, température de refoulement et taux de compression.
//!
//! ```text
//! masse volumique  ρ = p/(R·T)                        (gaz parfait)
//! taux             τ = p₂/p₁
//! travail isotherme  W_iso = p₁·V₁·ln(p₂/p₁)
//! travail adiabat.   W_ad = (γ/(γ−1))·p₁·V₁·[(p₂/p₁)^{(γ−1)/γ} − 1]
//! refoulement adiab. T₂ = T₁·(p₂/p₁)^{(γ−1)/γ}
//! ```
//!
//! `p` pression **absolue** (Pa), `T` température absolue (K), `R` constante
//! spécifique du gaz (air ≈ 287 J·kg⁻¹·K⁻¹), `V₁` volume aspiré (m³), `γ` rapport
//! des chaleurs massiques (air ≈ 1,4), `W` travail de compression (J), `τ` taux
//! de compression.
//!
//! **Convention** : pressions **absolues**, températures en kelvin, SI.
//! **Limite honnête** : gaz **parfait** ; compression **réversible** (idéale)
//! isotherme (borne basse) ou adiabatique/isentropique (borne haute), sans
//! rendement mécanique ni pertes de charge ; la compression réelle se situe entre
//! ces deux bornes (polytropique). `R` et `γ` sont des données du gaz fournies
//! par l'appelant.

/// Masse volumique d'un gaz parfait `ρ = p/(R·T)`.
///
/// Panique si un paramètre `<= 0`.
pub fn ideal_gas_density(pressure_abs: f64, specific_gas_constant: f64, temperature: f64) -> f64 {
    assert!(
        pressure_abs > 0.0 && specific_gas_constant > 0.0 && temperature > 0.0,
        "p, R et T strictement positifs requis"
    );
    pressure_abs / (specific_gas_constant * temperature)
}

/// Taux de compression `τ = p₂/p₁` (pressions absolues).
///
/// Panique si `inlet_pressure_abs <= 0`.
pub fn compression_ratio(inlet_pressure_abs: f64, outlet_pressure_abs: f64) -> f64 {
    assert!(
        inlet_pressure_abs > 0.0,
        "la pression d'aspiration doit être strictement positive"
    );
    outlet_pressure_abs / inlet_pressure_abs
}

/// Travail de compression **isotherme** `W = p₁·V₁·ln(p₂/p₁)`.
///
/// Panique si une pression ou le volume `<= 0`.
pub fn isothermal_work(
    inlet_pressure_abs: f64,
    outlet_pressure_abs: f64,
    inlet_volume: f64,
) -> f64 {
    assert!(
        inlet_pressure_abs > 0.0 && outlet_pressure_abs > 0.0 && inlet_volume > 0.0,
        "pressions et volume strictement positifs requis"
    );
    inlet_pressure_abs * inlet_volume * (outlet_pressure_abs / inlet_pressure_abs).ln()
}

/// Travail de compression **adiabatique** (isentropique)
/// `W = (γ/(γ−1))·p₁·V₁·[(p₂/p₁)^{(γ−1)/γ} − 1]`.
///
/// Panique si une pression ou le volume `<= 0`, ou `gamma <= 1`.
pub fn adiabatic_work(
    inlet_pressure_abs: f64,
    outlet_pressure_abs: f64,
    inlet_volume: f64,
    gamma: f64,
) -> f64 {
    assert!(
        inlet_pressure_abs > 0.0 && outlet_pressure_abs > 0.0 && inlet_volume > 0.0 && gamma > 1.0,
        "p, V > 0 et γ > 1 requis"
    );
    let ratio = outlet_pressure_abs / inlet_pressure_abs;
    (gamma / (gamma - 1.0))
        * inlet_pressure_abs
        * inlet_volume
        * (ratio.powf((gamma - 1.0) / gamma) - 1.0)
}

/// Température de refoulement **adiabatique** `T₂ = T₁·(p₂/p₁)^{(γ−1)/γ}`.
///
/// Panique si une pression ou la température `<= 0`, ou `gamma <= 1`.
pub fn adiabatic_outlet_temperature(
    inlet_temperature: f64,
    inlet_pressure_abs: f64,
    outlet_pressure_abs: f64,
    gamma: f64,
) -> f64 {
    assert!(
        inlet_temperature > 0.0
            && inlet_pressure_abs > 0.0
            && outlet_pressure_abs > 0.0
            && gamma > 1.0,
        "T, p > 0 et γ > 1 requis"
    );
    inlet_temperature * (outlet_pressure_abs / inlet_pressure_abs).powf((gamma - 1.0) / gamma)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn air_density_at_stp() {
        // Air à 101 325 Pa, 288 K : ρ ≈ 1,225 kg/m³.
        let rho = ideal_gas_density(101_325.0, 287.0, 288.0);
        assert!(rho > 1.22 && rho < 1.23);
    }

    #[test]
    fn compression_ratio_is_pressure_ratio() {
        assert_relative_eq!(compression_ratio(1e5, 7e5), 7.0, epsilon = 1e-12);
    }

    #[test]
    fn adiabatic_work_exceeds_isothermal() {
        // Pour un même τ, la compression adiabatique coûte plus que l'isotherme.
        let (p1, p2, v1) = (1e5, 7e5, 1.0);
        let w_iso = isothermal_work(p1, p2, v1);
        let w_ad = adiabatic_work(p1, p2, v1, 1.4);
        assert!(w_ad > w_iso);
        assert!(w_iso > 0.0);
    }

    #[test]
    fn adiabatic_outlet_temperature_rises() {
        // Air 293 K comprimé de 1 à 7 bar → T₂ = 293·7^{0,4/1,4} ≈ 511 K.
        let t2 = adiabatic_outlet_temperature(293.0, 1e5, 7e5, 1.4);
        assert!(t2 > 505.0 && t2 < 515.0);
        assert!(t2 > 293.0);
    }

    #[test]
    fn isothermal_work_matches_formula() {
        let (p1, p2, v1) = (1e5, 5e5, 2.0);
        assert_relative_eq!(
            isothermal_work(p1, p2, v1),
            p1 * v1 * (p2 / p1).ln(),
            epsilon = 1e-6
        );
    }

    #[test]
    #[should_panic(expected = "γ > 1")]
    fn gamma_one_panics() {
        adiabatic_work(1e5, 7e5, 1.0, 1.0);
    }
}
