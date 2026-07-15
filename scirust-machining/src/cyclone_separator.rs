//! Cyclone dépoussiéreur — séparation gaz-solide par force centrifuge (modèle
//! de Lapple, temps de séjour) : diamètre de coupure, rendement fractionnel,
//! nombre de tours effectifs et perte de charge.
//!
//! ```text
//! diamètre de coupure d50   d50 = sqrt( 9·μ·b / (2·π·N·vi·(ρp − ρg)) )
//! nombre de tours effectifs N   = (Lb + Lc/2) / a
//! rendement fractionnel     η   = 1 / (1 + (d50/dp)²)
//! perte de charge           Δp  = Nh · ½·ρg·vi²
//! ```
//!
//! `μ` viscosité dynamique du gaz (Pa·s), `b` largeur d'entrée (m), `N` nombre
//! de tours effectifs de la veine (sans dimension), `vi` vitesse d'entrée du gaz
//! (m/s), `ρp` masse volumique des particules (kg/m³), `ρg` masse volumique du
//! gaz (kg/m³), `a` hauteur d'entrée (m), `Lb` longueur du corps cylindrique
//! (m), `Lc` longueur du cône (m), `dp` diamètre de la particule (m), `d50`
//! diamètre de coupure (m), `η` rendement de collecte fractionnel (0..1), `Nh`
//! nombre de hauteurs cinétiques d'entrée (sans dimension), `Δp` perte de charge
//! (Pa). Le `d50` est le diamètre collecté à 50 % : `η = 0,5` lorsque `dp = d50`.
//!
//! **Convention** : SI cohérent (mètre, kilogramme, seconde, pascal).
//! **Limite honnête** : modèle de Lapple fondé sur le temps de séjour ; la
//! viscosité `μ`, les masses volumiques `ρp` et `ρg`, la géométrie (`b`, `a`,
//! `Lb`, `Lc`) et le nombre de hauteurs cinétiques `Nh` sont **fournis par
//! l'appelant** (propriétés du gaz et des solides, dessin du cyclone, mesure de
//! perte de charge) — aucune constante physique, matériau ou procédé n'est
//! inventée « par défaut ». Le nombre de tours effectifs est **estimé par la
//! géométrie fournie** ; les effets de ré-entraînement et de charge en poussière
//! ne sont pas modélisés.

use core::f64::consts::PI;

/// Diamètre de coupure d50 de Lapple
/// `d50 = sqrt( 9·μ·b / (2·π·N·vi·(ρp − ρg)) )` (m) — diamètre collecté à 50 %.
///
/// Panique si `dynamic_viscosity <= 0`, `inlet_width <= 0`,
/// `effective_turns <= 0`, `inlet_velocity <= 0`, `gas_density < 0` ou
/// `particle_density <= gas_density`.
pub fn cyclone_cut_diameter(
    dynamic_viscosity: f64,
    inlet_width: f64,
    effective_turns: f64,
    inlet_velocity: f64,
    particle_density: f64,
    gas_density: f64,
) -> f64 {
    assert!(
        dynamic_viscosity > 0.0,
        "la viscosité dynamique doit être strictement positive"
    );
    assert!(
        inlet_width > 0.0,
        "la largeur d'entrée doit être strictement positive"
    );
    assert!(
        effective_turns > 0.0,
        "le nombre de tours effectifs doit être strictement positif"
    );
    assert!(
        inlet_velocity > 0.0,
        "la vitesse d'entrée doit être strictement positive"
    );
    assert!(
        gas_density >= 0.0,
        "la masse volumique du gaz ne peut pas être négative"
    );
    assert!(
        particle_density > gas_density,
        "la masse volumique des particules doit dépasser celle du gaz"
    );
    (9.0 * dynamic_viscosity * inlet_width
        / (2.0 * PI * effective_turns * inlet_velocity * (particle_density - gas_density)))
        .sqrt()
}

/// Nombre de tours effectifs de la veine gazeuse estimé par la géométrie
/// `N = (Lb + Lc/2) / a` (sans dimension).
///
/// Panique si `inlet_height <= 0`, `body_length <= 0` ou `cone_length < 0`.
pub fn cyclone_number_of_turns(inlet_height: f64, body_length: f64, cone_length: f64) -> f64 {
    assert!(
        inlet_height > 0.0,
        "la hauteur d'entrée doit être strictement positive"
    );
    assert!(
        body_length > 0.0,
        "la longueur du corps doit être strictement positive"
    );
    assert!(
        cone_length >= 0.0,
        "la longueur du cône ne peut pas être négative"
    );
    (body_length + cone_length / 2.0) / inlet_height
}

/// Rendement de collecte fractionnel de Lapple
/// `η = 1 / (1 + (d50/dp)²)` (sans dimension, 0..1).
///
/// Panique si `particle_diameter <= 0` ou `cut_diameter <= 0`.
pub fn cyclone_collection_efficiency(particle_diameter: f64, cut_diameter: f64) -> f64 {
    assert!(
        particle_diameter > 0.0,
        "le diamètre de particule doit être strictement positif"
    );
    assert!(
        cut_diameter > 0.0,
        "le diamètre de coupure doit être strictement positif"
    );
    let ratio = cut_diameter / particle_diameter;
    1.0 / (1.0 + ratio * ratio)
}

/// Perte de charge du cyclone `Δp = Nh · ½·ρg·vi²` (Pa) exprimée en nombre de
/// hauteurs cinétiques d'entrée.
///
/// Panique si `inlet_velocity_heads <= 0`, `gas_density <= 0` ou
/// `inlet_velocity <= 0`.
pub fn cyclone_pressure_drop(
    inlet_velocity_heads: f64,
    gas_density: f64,
    inlet_velocity: f64,
) -> f64 {
    assert!(
        inlet_velocity_heads > 0.0,
        "le nombre de hauteurs cinétiques doit être strictement positif"
    );
    assert!(
        gas_density > 0.0,
        "la masse volumique du gaz doit être strictement positive"
    );
    assert!(
        inlet_velocity > 0.0,
        "la vitesse d'entrée doit être strictement positive"
    );
    inlet_velocity_heads * 0.5 * gas_density * inlet_velocity * inlet_velocity
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn efficiency_is_half_at_cut_diameter() {
        // Définition du d50 : à dp = d50, le rendement fractionnel vaut 0,5.
        let d50 = 3.0e-6_f64;
        let eta = cyclone_collection_efficiency(d50, d50);
        assert_relative_eq!(eta, 0.5, epsilon = 1e-15);
    }

    #[test]
    fn cut_diameter_reference_case() {
        // Air μ=1,8e-5 Pa·s, b=0,05 m, N=5, vi=15 m/s, ρp=2000, ρg=1,2 kg/m³.
        // num = 9·1,8e-5·0,05 = 8,1e-6
        // den = 2·π·5·15·(2000−1,2) = 941912,31
        // d50 = sqrt(8,1e-6 / 941912,31) = 2,9325e-6 m.
        let d50 = cyclone_cut_diameter(1.8e-5, 0.05, 5.0, 15.0, 2000.0, 1.2);
        assert_relative_eq!(d50, 2.9325e-6, epsilon = 1e-9);
    }

    #[test]
    fn cut_diameter_scales_with_sqrt_inlet_width() {
        // d50 ∝ sqrt(b) : quadrupler la largeur d'entrée double le d50.
        let d1 = cyclone_cut_diameter(1.8e-5, 0.05, 5.0, 15.0, 2000.0, 1.2);
        let d2 = cyclone_cut_diameter(1.8e-5, 0.20, 5.0, 15.0, 2000.0, 1.2);
        assert_relative_eq!(d2 / d1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn number_of_turns_reference_case() {
        // Lb=1,0 m, Lc=1,0 m, a=0,2 m → N = (1,0 + 1,0/2)/0,2 = 7,5.
        let n = cyclone_number_of_turns(0.2, 1.0, 1.0);
        assert_relative_eq!(n, 7.5, epsilon = 1e-12);
    }

    #[test]
    fn pressure_drop_reference_case() {
        // Nh=6, ρg=1,2 kg/m³, vi=15 m/s → Δp = 6·0,5·1,2·15² = 810 Pa.
        let dp = cyclone_pressure_drop(6.0, 1.2, 15.0);
        assert_relative_eq!(dp, 810.0, epsilon = 1e-9);
    }

    #[test]
    fn pressure_drop_scales_with_velocity_squared() {
        // Δp ∝ vi² : doubler la vitesse d'entrée quadruple la perte de charge.
        let dp1 = cyclone_pressure_drop(6.0, 1.2, 15.0);
        let dp2 = cyclone_pressure_drop(6.0, 1.2, 30.0);
        assert_relative_eq!(dp2 / dp1, 4.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "masse volumique des particules")]
    fn particle_lighter_than_gas_panics() {
        cyclone_cut_diameter(1.8e-5, 0.05, 5.0, 15.0, 1.0, 1.2);
    }
}
