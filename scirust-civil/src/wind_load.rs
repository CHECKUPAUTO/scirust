//! Actions du vent sur les structures selon l'**Eurocode 1** (EN 1991-1-4) :
//! vitesse moyenne du vent, pression dynamique de pointe, pression sur une
//! surface et force globale du vent.
//!
//! ```text
//! vitesse moyenne        vm = vb·cr·co
//! pression de pointe      qp = (1 + 7·Iv)·½·ρ·vm²
//! pression sur surface    w  = qp·cp
//! force globale           Fw = cf·qp·Aref
//! ```
//!
//! `vb` vitesse de base du vent (m/s, issue de la carte de vent), `cr` facteur
//! de rugosité (–), `co` facteur d'orographie (–), `vm` vitesse moyenne (m/s),
//! `Iv` intensité (facteur) de turbulence (–), `ρ` masse volumique de l'air
//! (kg/m³), `qp` pression dynamique de pointe (Pa), `cp` coefficient de pression
//! (–, positif en pression, négatif en succion), `w` pression sur la surface
//! (Pa), `cf` coefficient de force (–), `Aref` aire de référence (m²), `Fw`
//! force globale du vent (N).
//!
//! **Convention** : SI strict et cohérent — m/s pour les vitesses, Pa pour les
//! pressions, N pour les forces, kg/m³ pour la masse volumique. Comme 1 Pa =
//! 1 N/m², une pression `qp` (Pa) multipliée par une aire (m²) donne bien des
//! newtons. Types `f64`.
//!
//! **Limite honnête** : l'action du vent suit l'Eurocode 1-4. La vitesse de
//! base `vb` (carte de vent, période de retour, altitude), les facteurs de
//! rugosité `cr`, d'orographie `co` et de turbulence `Iv` (dépendant du site et
//! de la hauteur), les coefficients aérodynamiques de pression `cp` et de force
//! `cf` (dépendant de la forme, de l'élancement, de la zone) sont **fournis par
//! l'appelant** d'après la carte de vent et les tables de l'Eurocode et de son
//! Annexe Nationale — jamais inventés. La masse volumique de l'air `ρ` est
//! également **fournie** (elle dépend de l'altitude et de la température). Ce
//! module ne calcule **pas** les coefficients aérodynamiques ni les effets
//! dynamiques de résonance (coefficient structural cscd).

/// Vitesse moyenne du vent `vm = vb·cr·co` (m/s).
///
/// Panique si `basic_velocity < 0`, `roughness_factor < 0` ou
/// `orography_factor < 0`.
pub fn wind_mean_velocity(
    basic_velocity: f64,
    roughness_factor: f64,
    orography_factor: f64,
) -> f64 {
    assert!(
        basic_velocity >= 0.0,
        "la vitesse de base vb doit être positive ou nulle"
    );
    assert!(
        roughness_factor >= 0.0,
        "le facteur de rugosité cr doit être positif ou nul"
    );
    assert!(
        orography_factor >= 0.0,
        "le facteur d'orographie co doit être positif ou nul"
    );
    basic_velocity * roughness_factor * orography_factor
}

/// Pression dynamique de pointe `qp = (1 + 7·Iv)·½·ρ·vm²` (Pa).
///
/// Panique si `turbulence_factor < 0`, `air_density <= 0` ou `mean_velocity < 0`.
pub fn wind_peak_velocity_pressure(
    turbulence_factor: f64,
    air_density: f64,
    mean_velocity: f64,
) -> f64 {
    assert!(
        turbulence_factor >= 0.0,
        "le facteur de turbulence Iv doit être positif ou nul"
    );
    assert!(
        air_density > 0.0,
        "la masse volumique de l'air ρ doit être strictement positive"
    );
    assert!(
        mean_velocity >= 0.0,
        "la vitesse moyenne vm doit être positive ou nulle"
    );
    (1.0 + 7.0 * turbulence_factor) * 0.5 * air_density * mean_velocity * mean_velocity
}

/// Pression du vent sur une surface `w = qp·cp` (Pa).
///
/// Le coefficient de pression `cp` peut être négatif (succion), aussi son signe
/// n'est pas contraint.
///
/// Panique si `peak_velocity_pressure < 0`.
pub fn wind_pressure_on_surface(peak_velocity_pressure: f64, pressure_coefficient: f64) -> f64 {
    assert!(
        peak_velocity_pressure >= 0.0,
        "la pression de pointe qp doit être positive ou nulle"
    );
    peak_velocity_pressure * pressure_coefficient
}

/// Force globale du vent `Fw = cf·qp·Aref` (N).
///
/// Panique si `force_coefficient < 0`, `peak_velocity_pressure < 0` ou
/// `reference_area < 0`.
pub fn wind_force(force_coefficient: f64, peak_velocity_pressure: f64, reference_area: f64) -> f64 {
    assert!(
        force_coefficient >= 0.0,
        "le coefficient de force cf doit être positif ou nul"
    );
    assert!(
        peak_velocity_pressure >= 0.0,
        "la pression de pointe qp doit être positive ou nulle"
    );
    assert!(
        reference_area >= 0.0,
        "l'aire de référence Aref doit être positive ou nulle"
    );
    force_coefficient * peak_velocity_pressure * reference_area
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn mean_velocity_is_product_of_factors() {
        // vm = vb·cr·co : facteurs unitaires laissent la vitesse de base inchangée.
        assert_relative_eq!(
            wind_mean_velocity(26.0, 1.0, 1.0),
            26.0,
            max_relative = 1e-12
        );
        // vb = 26, cr = 0,8, co = 1,0 → vm = 20,8 m/s.
        assert_relative_eq!(
            wind_mean_velocity(26.0, 0.8, 1.0),
            20.8,
            max_relative = 1e-12
        );
    }

    #[test]
    fn peak_pressure_scales_with_velocity_squared() {
        // qp ∝ vm² : doubler la vitesse quadruple la pression de pointe.
        let q1 = wind_peak_velocity_pressure(0.15, 1.25, 20.0);
        let q2 = wind_peak_velocity_pressure(0.15, 1.25, 40.0);
        assert_relative_eq!(q2, 4.0 * q1, max_relative = 1e-12);
    }

    #[test]
    fn no_turbulence_gives_bare_dynamic_pressure() {
        // Iv = 0 : qp = ½·ρ·vm² = 0,5·1,25·20² = 250 Pa.
        assert_relative_eq!(
            wind_peak_velocity_pressure(0.0, 1.25, 20.0),
            250.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn suction_gives_negative_pressure() {
        // cp négatif (succion) → pression négative, proportionnelle à qp.
        let qp = 500.0_f64;
        assert_relative_eq!(
            wind_pressure_on_surface(qp, -0.7),
            -350.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn force_is_linear_in_area() {
        // Fw ∝ Aref : doubler l'aire double la force (cf, qp fixés).
        let f1 = wind_force(1.3, 554.32, 10.0);
        let f2 = wind_force(1.3, 554.32, 20.0);
        assert_relative_eq!(f2, 2.0 * f1, max_relative = 1e-12);
    }

    #[test]
    fn worked_case_building_facade() {
        // Site : vb = 26 m/s, cr = 0,8, co = 1,0 → vm = 20,8 m/s.
        // vm² = 432,64. Iv = 0,15, ρ = 1,25 kg/m³.
        // qp = (1 + 7·0,15)·0,5·1,25·432,64
        //    = 2,05 · 0,625 · 432,64 = 2,05 · 270,4 = 554,32 Pa.
        // Paroi cp = 0,8 → w = 554,32·0,8 = 443,456 Pa.
        // Force sur Aref = 10 m² avec cf = 1,3 :
        // Fw = 1,3 · 554,32 · 10 = 7206,16 N.
        let vm = wind_mean_velocity(26.0, 0.8, 1.0);
        assert_relative_eq!(vm, 20.8, max_relative = 1e-9);
        let qp = wind_peak_velocity_pressure(0.15, 1.25, vm);
        assert_relative_eq!(qp, 554.32, max_relative = 1e-3);
        let w = wind_pressure_on_surface(qp, 0.8);
        assert_relative_eq!(w, 443.456, max_relative = 1e-3);
        let fw = wind_force(1.3, qp, 10.0);
        assert_relative_eq!(fw, 7206.16, max_relative = 1e-3);
    }

    #[test]
    #[should_panic(expected = "la masse volumique de l'air ρ doit être strictement positive")]
    fn zero_air_density_panics() {
        let _ = wind_peak_velocity_pressure(0.15, 0.0, 20.0);
    }
}
