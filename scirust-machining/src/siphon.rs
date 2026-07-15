//! Siphon : écoulement gravitaire d'un fluide incompressible décrit par
//! **Bernoulli** (vitesse de sortie de Torricelli, débit, pression au sommet et
//! hauteur maximale du col avant cavitation).
//!
//! ```text
//! vitesse de sortie   v = √(2·g·H)                              (Torricelli)
//! débit               Q = v·A
//! pression au col     p_col = p_atm − ρ·g·h_col − ½·ρ·v²
//! hauteur max du col  h_max = (p_atm − p_vap − ½·ρ·v²)/(ρ·g)
//! ```
//!
//! `g` pesanteur (m/s²), `H` dénivelé entre la surface amont et la sortie (m),
//! `v` vitesse de sortie (m/s), `A` section de conduite (m²), `Q` débit
//! volumique (m³/s), `ρ` masse volumique (kg/m³), `h_col` hauteur du col
//! au-dessus de la surface amont (m), `p_atm` pression atmosphérique (Pa),
//! `p_vap` pression de vapeur saturante (Pa), `p_col` pression absolue au
//! sommet (Pa), `h_max` hauteur maximale du col (m).
//!
//! **Convention** : SI cohérent, pressions **absolues**. **Limite honnête** :
//! fluide incompressible parfait, écoulement permanent le long d'une ligne de
//! courant, Bernoulli **sans pertes** (sauf si l'appelant fournit une vitesse
//! déjà réduite pour tenir compte des pertes) ; `g`, `ρ`, `p_atm` et `p_vap`
//! sont **fournis par l'appelant** — aucune valeur de fluide ou d'atmosphère
//! n'est supposée « par défaut ». Le siphon reste amorcé tant que la pression
//! au col demeure au-dessus de la pression de vapeur.

/// Vitesse de sortie idéale d'un siphon `v = √(2·g·H)` (m/s, Torricelli entre la
/// surface amont et la sortie).
///
/// Panique si `g·H < 0`.
pub fn siphon_outlet_velocity(height_drop: f64, gravity: f64) -> f64 {
    assert!(
        gravity * height_drop >= 0.0,
        "g·H doit être positif (dénivelé et pesanteur de même signe)"
    );
    (2.0_f64 * gravity * height_drop).sqrt()
}

/// Débit volumique du siphon `Q = v·A` (m³/s).
///
/// Panique si `pipe_area < 0`.
pub fn siphon_flow_rate(outlet_velocity: f64, pipe_area: f64) -> f64 {
    assert!(
        pipe_area >= 0.0,
        "la section de conduite doit être positive"
    );
    outlet_velocity * pipe_area
}

/// Pression absolue au col `p_col = p_atm − ρ·g·h_col − ½·ρ·v²` (Pa) ; elle doit
/// rester supérieure à la pression de vapeur pour éviter la cavitation.
///
/// Panique si `fluid_density < 0`.
pub fn siphon_crest_pressure(
    atmospheric_pressure: f64,
    fluid_density: f64,
    gravity: f64,
    crest_height_above_surface: f64,
    velocity: f64,
) -> f64 {
    assert!(
        fluid_density >= 0.0,
        "la masse volumique doit être positive"
    );
    atmospheric_pressure
        - fluid_density * gravity * crest_height_above_surface
        - 0.5 * fluid_density * velocity * velocity
}

/// Hauteur maximale du col avant cavitation
/// `h_max = (p_atm − p_vap − ½·ρ·v²)/(ρ·g)` (m).
///
/// Panique si `ρ·g <= 0`.
pub fn siphon_max_crest_height(
    atmospheric_pressure: f64,
    vapor_pressure: f64,
    fluid_density: f64,
    gravity: f64,
    velocity: f64,
) -> f64 {
    assert!(
        fluid_density * gravity > 0.0,
        "ρ·g doit être strictement positif"
    );
    (atmospheric_pressure - vapor_pressure - 0.5 * fluid_density * velocity * velocity)
        / (fluid_density * gravity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn outlet_velocity_matches_torricelli() {
        // v = √(2·9,81·2) = √39,24 ≈ 6,2642 m/s.
        let v = siphon_outlet_velocity(2.0, 9.81);
        assert_relative_eq!(v, (2.0_f64 * 9.81 * 2.0).sqrt(), epsilon = 1e-12);
        assert_relative_eq!(v, 6.264184532_f64, epsilon = 1e-6);
    }

    #[test]
    fn outlet_velocity_scales_with_square_root_of_drop() {
        // Multiplier le dénivelé par 4 double la vitesse (v ∝ √H).
        let v1 = siphon_outlet_velocity(1.5, 9.81);
        let v4 = siphon_outlet_velocity(6.0, 9.81);
        assert_relative_eq!(v4, 2.0 * v1, epsilon = 1e-12);
    }

    #[test]
    fn flow_rate_is_velocity_times_area() {
        // Q = 3·0,01 = 0,03 m³/s ; linéarité en section.
        assert_relative_eq!(siphon_flow_rate(3.0, 0.01), 0.03, epsilon = 1e-12);
        assert_relative_eq!(
            siphon_flow_rate(3.0, 0.02),
            2.0 * siphon_flow_rate(3.0, 0.01),
            epsilon = 1e-12
        );
    }

    #[test]
    fn crest_pressure_realistic_case() {
        // p_atm=101325, ρ=1000, g=9,81, h_col=5, v=2 :
        // 101325 − 1000·9,81·5 − ½·1000·4 = 101325 − 49050 − 2000 = 50275 Pa.
        assert_relative_eq!(
            siphon_crest_pressure(101_325.0, 1000.0, 9.81, 5.0, 2.0),
            50_275.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn crest_pressure_at_max_height_equals_vapor_pressure() {
        // Identité : à la hauteur maximale, la pression au col vaut exactement la
        // pression de vapeur (limite de cavitation).
        let (p_atm, p_vap, rho, g, v) = (101_325.0, 2339.0, 1000.0, 9.81, 1.5);
        let h_max = siphon_max_crest_height(p_atm, p_vap, rho, g, v);
        let p_col = siphon_crest_pressure(p_atm, rho, g, h_max, v);
        assert_relative_eq!(p_col, p_vap, epsilon = 1e-6);
    }

    #[test]
    fn max_crest_height_realistic_case() {
        // p_atm=101325, p_vap=2339, ρ=1000, g=9,81, v=1 :
        // (101325 − 2339 − 500)/9810 = 98486/9810 ≈ 10,03935 m.
        let h_max = siphon_max_crest_height(101_325.0, 2339.0, 1000.0, 9.81, 1.0);
        assert_relative_eq!(h_max, 98_486.0 / 9810.0, epsilon = 1e-9);
        assert_relative_eq!(h_max, 10.039347_f64, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "ρ·g doit être strictement positif")]
    fn max_crest_height_zero_density_panics() {
        siphon_max_crest_height(101_325.0, 2339.0, 0.0, 9.81, 1.0);
    }
}
