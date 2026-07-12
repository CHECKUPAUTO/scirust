//! Écoulement d'**air comprimé** — régime critique (**sonique/bloqué**), rapport
//! de pression critique, vitesse du son et débit-masse bloqué à travers un orifice.
//!
//! ```text
//! rapport critique  r* = (2/(γ+1))^{γ/(γ−1)}          (air : ≈ 0,528)
//! bloqué si         p_aval/p_amont ≤ r*
//! vitesse du son    a = √(γ·R·T)
//! débit-masse bloqué ṁ = Cd·A·p₀·√(γ/(R·T₀))·(2/(γ+1))^{(γ+1)/(2(γ−1))}
//! ```
//!
//! `γ` rapport des chaleurs massiques (air ≈ 1,4), `R` constante **spécifique**
//! du gaz (air ≈ 287 J·kg⁻¹·K⁻¹), `T` température absolue (K), `p₀` pression
//! **absolue** amont (Pa), `A` section au col (m²), `Cd` coefficient de débit
//! (≈ 0,6–0,9), `ṁ` débit-masse (kg/s).
//!
//! **Convention** : pressions **absolues**, températures en kelvin, SI.
//! **Limite honnête** : gaz **parfait**, écoulement **isentropique** au col,
//! formule de débit valable en régime **bloqué** (sonique). En dessous du seuil
//! critique l'écoulement est subsonique et le débit dépend du rapport de
//! pression (non couvert ici). `γ`, `R`, `Cd` sont fournis par l'appelant.

/// Rapport de pression **critique** `r* = (2/(γ+1))^{γ/(γ−1)}`.
///
/// Panique si `gamma <= 1`.
pub fn critical_pressure_ratio(gamma: f64) -> f64 {
    assert!(gamma > 1.0, "γ doit être strictement supérieur à 1");
    (2.0 / (gamma + 1.0)).powf(gamma / (gamma - 1.0))
}

/// Vrai si l'écoulement est **bloqué** (sonique) : `p_aval/p_amont ≤ r*`.
///
/// Panique si une pression `<= 0` ou `gamma <= 1`.
pub fn is_choked(downstream_abs: f64, upstream_abs: f64, gamma: f64) -> bool {
    assert!(
        downstream_abs > 0.0 && upstream_abs > 0.0,
        "pressions absolues strictement positives"
    );
    downstream_abs / upstream_abs <= critical_pressure_ratio(gamma)
}

/// Vitesse du son `a = √(γ·R·T)`.
///
/// Panique si un paramètre `<= 0` ou `gamma <= 1`.
pub fn speed_of_sound(gamma: f64, specific_gas_constant: f64, temperature: f64) -> f64 {
    assert!(
        gamma > 1.0 && specific_gas_constant > 0.0 && temperature > 0.0,
        "γ > 1, R > 0 et T > 0 requis"
    );
    (gamma * specific_gas_constant * temperature).sqrt()
}

/// Débit-masse en régime **bloqué**
/// `ṁ = Cd·A·p₀·√(γ/(R·T₀))·(2/(γ+1))^{(γ+1)/(2(γ−1))}`.
///
/// Panique si un paramètre `<= 0`, `gamma <= 1` ou `Cd` hors `]0, 1]`.
pub fn choked_mass_flow(
    discharge_coefficient: f64,
    throat_area: f64,
    upstream_pressure_abs: f64,
    upstream_temperature: f64,
    gamma: f64,
    specific_gas_constant: f64,
) -> f64 {
    assert!(
        discharge_coefficient > 0.0 && discharge_coefficient <= 1.0,
        "Cd doit être dans ]0, 1]"
    );
    assert!(
        throat_area > 0.0
            && upstream_pressure_abs > 0.0
            && upstream_temperature > 0.0
            && gamma > 1.0
            && specific_gas_constant > 0.0,
        "A, p₀, T₀ > 0 et γ > 1, R > 0 requis"
    );
    let g = gamma;
    let root = (g / (specific_gas_constant * upstream_temperature)).sqrt();
    let expo = (2.0 / (g + 1.0)).powf((g + 1.0) / (2.0 * (g - 1.0)));
    discharge_coefficient * throat_area * upstream_pressure_abs * root * expo
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn air_critical_ratio_is_0528() {
        // Pour γ=1,4, r* ≈ 0,5283.
        assert_relative_eq!(
            critical_pressure_ratio(1.4),
            0.5282817877,
            max_relative = 1e-6
        );
    }

    #[test]
    fn choked_below_threshold() {
        // 7 bar absolu amont, 1 bar aval → ratio ≈ 0,143 < 0,528 → bloqué.
        assert!(is_choked(1e5, 7e5, 1.4));
        // 7 bar amont, 6 bar aval → ratio ≈ 0,857 > 0,528 → subsonique.
        assert!(!is_choked(6e5, 7e5, 1.4));
    }

    #[test]
    fn speed_of_sound_air_at_288k() {
        // Air à 288 K : a = √(1,4·287·288) ≈ 340 m/s.
        let a = speed_of_sound(1.4, 287.0, 288.0);
        assert!(a > 339.0 && a < 341.0);
    }

    #[test]
    fn choked_flow_proportional_to_pressure_and_area() {
        // ṁ ∝ p₀ et ∝ A (à T, γ, R, Cd fixés).
        let m1 = choked_mass_flow(0.8, 1e-6, 7e5, 293.0, 1.4, 287.0);
        let m2 = choked_mass_flow(0.8, 2e-6, 14e5, 293.0, 1.4, 287.0);
        assert_relative_eq!(m2 / m1, 4.0, epsilon = 1e-9);
        assert!(m1 > 0.0);
    }

    #[test]
    #[should_panic(expected = "Cd doit être")]
    fn discharge_coefficient_above_one_panics() {
        choked_mass_flow(1.5, 1e-6, 7e5, 293.0, 1.4, 287.0);
    }
}
