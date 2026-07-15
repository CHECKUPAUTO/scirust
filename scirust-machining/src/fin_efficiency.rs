//! Rendement d'une ailette droite de refroidissement à section constante et bout adiabatique.
//!
//! ```text
//! paramètre     m   = √(h·P/(k·A_c))                (1/m)
//! rendement     η   = tanh(m·L)/(m·L)               (bout adiabatique, sans dimension)
//! efficience    ε   = η·A_fin/A_c                    (gain vs surface nue, sans dimension)
//! flux évacué   Q   = η·h·A_fin·(T_b − T_∞)          (W)
//! ```
//!
//! `h` coefficient de convection (W/(m²·K)), `P` périmètre mouillé de l'ailette (m),
//! `k` conductivité thermique (W/(m·K)), `A_c` aire de section droite (m²),
//! `L` longueur de l'ailette (m), `m` paramètre d'ailette (1/m), `A_fin` aire
//! d'échange de l'ailette (m²), `T_b` température de base (K ou °C), `T_∞`
//! température du fluide ambiant (K ou °C).
//!
//! **Convention** : unités SI cohérentes, `f64`. **Limite honnête** : ailette 1D à
//! **section constante**, régime **permanent**, **conduction longitudinale seule**,
//! extrémité **adiabatique**. Le coefficient `h` et la conductivité `k` sont
//! **uniformes et FOURNIS par l'appelant** ; aucune constante physique, propriété
//! matériau ou paramètre de procédé n'est supposée « par défaut ». Ni rayonnement,
//! ni résistance de contact base-ailette.

/// Paramètre d'ailette `m = √(h·P/(k·A_c))` (1/m).
///
/// Panique si `heat_transfer_coefficient < 0`, `perimeter < 0` ou `k·A_c <= 0`.
pub fn fineff_parameter_m(
    heat_transfer_coefficient: f64,
    perimeter: f64,
    thermal_conductivity: f64,
    cross_section_area: f64,
) -> f64 {
    assert!(
        heat_transfer_coefficient >= 0.0,
        "h doit être positif ou nul"
    );
    assert!(perimeter >= 0.0, "le périmètre P doit être positif ou nul");
    assert!(
        thermal_conductivity * cross_section_area > 0.0,
        "k·A_c doit être strictement positif"
    );
    (heat_transfer_coefficient * perimeter / (thermal_conductivity * cross_section_area)).sqrt()
}

/// Rendement d'une ailette à bout adiabatique `η = tanh(m·L)/(m·L)` (sans dimension).
///
/// Panique si `m·L <= 0`.
pub fn fineff_efficiency_adiabatic_tip(m: f64, length: f64) -> f64 {
    let ml = m * length;
    assert!(ml > 0.0, "m·L doit être strictement positif");
    ml.tanh() / ml
}

/// Efficience de l'ailette `ε = η·A_fin/A_c` (gain par rapport à la surface nue).
///
/// Panique si `efficiency < 0`, `surface_area < 0` ou `cross_section_area <= 0`.
pub fn fineff_effectiveness(efficiency: f64, surface_area: f64, cross_section_area: f64) -> f64 {
    assert!(efficiency >= 0.0, "le rendement η doit être positif ou nul");
    assert!(surface_area >= 0.0, "A_fin doit être positif ou nul");
    assert!(
        cross_section_area > 0.0,
        "A_c doit être strictement positif"
    );
    efficiency * surface_area / cross_section_area
}

/// Flux thermique évacué par l'ailette `Q = η·h·A_fin·(T_b − T_∞)` (W).
///
/// Panique si `efficiency < 0`, `heat_transfer_coefficient < 0` ou `fin_area < 0`.
pub fn fineff_heat_rate(
    efficiency: f64,
    heat_transfer_coefficient: f64,
    fin_area: f64,
    base_temp: f64,
    ambient_temp: f64,
) -> f64 {
    assert!(efficiency >= 0.0, "le rendement η doit être positif ou nul");
    assert!(
        heat_transfer_coefficient >= 0.0,
        "h doit être positif ou nul"
    );
    assert!(fin_area >= 0.0, "A_fin doit être positif ou nul");
    efficiency * heat_transfer_coefficient * fin_area * (base_temp - ambient_temp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn parameter_m_definition() {
        // h=100, P=0,1, k=200, A_c=1e-4 → m = √(100·0,1/(200·1e-4)) = √500.
        assert_relative_eq!(
            fineff_parameter_m(100.0, 0.1, 200.0, 1e-4),
            500.0_f64.sqrt(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn efficiency_identity_at_ml_unity() {
        // Pour m·L = 1, η = tanh(1)/1 = tanh(1) (identité exacte).
        assert_relative_eq!(
            fineff_efficiency_adiabatic_tip(2.0, 0.5),
            1.0_f64.tanh(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn efficiency_tends_to_one_for_short_fin() {
        // Quand m·L → 0, η → 1 (ailette courte quasi isotherme), et η < 1 toujours.
        let eta = fineff_efficiency_adiabatic_tip(1.0, 1e-4);
        assert!(eta < 1.0 && eta > 0.9999);
    }

    #[test]
    fn effectiveness_proportional_to_fin_area() {
        // ε = η·A_fin/A_c : doubler A_fin double l'efficience.
        let e1 = fineff_effectiveness(0.8, 0.005, 1e-4);
        let e2 = fineff_effectiveness(0.8, 0.010, 1e-4);
        assert_relative_eq!(e2, 2.0 * e1, epsilon = 1e-12);
        // Contrôle chiffré : 0,8·0,005/1e-4 = 40.
        assert_relative_eq!(e1, 40.0, epsilon = 1e-12);
    }

    #[test]
    fn effectiveness_matches_heat_rate_ratio() {
        // Identité : ε = Q / (h·A_c·(T_b − T_∞)) puisque Q = η·h·A_fin·ΔT.
        let (eta, h, a_fin, a_c, tb, tinf) = (0.72_f64, 100.0, 0.005, 1e-4, 80.0, 30.0);
        let q = fineff_heat_rate(eta, h, a_fin, tb, tinf);
        let eps = fineff_effectiveness(eta, a_fin, a_c);
        assert_relative_eq!(eps, q / (h * a_c * (tb - tinf)), epsilon = 1e-9);
    }

    #[test]
    fn heat_rate_realistic_value() {
        // η=0,72, h=100, A_fin=0,005, ΔT=50 → Q = 0,72·100·0,005·50 = 18 W.
        assert_relative_eq!(
            fineff_heat_rate(0.72, 100.0, 0.005, 80.0, 30.0),
            18.0,
            epsilon = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "k·A_c")]
    fn zero_conductivity_panics() {
        fineff_parameter_m(100.0, 0.1, 0.0, 1e-4);
    }
}
