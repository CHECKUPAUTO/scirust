//! Tube de **Pitot** : mesure de la vitesse d'un fluide à partir de la
//! différence entre pression d'arrêt et pression statique (Bernoulli).
//!
//! ```text
//! vitesse           v  = √( 2·(p0 − p) / ρ )
//! pression dynamique q  = ½·ρ·v²
//! pression d'arrêt   p0 = p + ½·ρ·v²
//! débit volumique    Q  = v·A
//! ```
//!
//! `p0` pression d'arrêt (de stagnation) (Pa), `p` pression statique (Pa),
//! `q = p0 − p` pression dynamique (Pa), `ρ` masse volumique du fluide (kg/m³),
//! `v` vitesse locale de l'écoulement (m/s), `A` aire de la section de passage
//! (m²), `Q` débit volumique (m³/s).
//!
//! **Convention** : SI cohérent (Pa, kg/m³, m/s, m², m³/s). **Limite honnête** :
//! écoulement **incompressible** et **permanent** (nombre de Mach faible,
//! compressibilité négligée), coefficient de sonde supposé **unitaire** ; la
//! masse volumique `ρ` est une **donnée** de l'appelant (issue de mesures ou de
//! tables) — jamais une valeur inventée. La sonde donne une vitesse **ponctuelle**
//! au point de mesure ; un profil non uniforme dans la section (Q = v·A) doit
//! être corrigé par l'appelant (facteur de profil / intégration).

/// Vitesse locale `v = √( 2·(p0 − p) / ρ )` (m/s).
///
/// Panique si `density <= 0` ou si `stagnation_pressure < static_pressure`
/// (pression dynamique négative, physiquement interdite).
pub fn pitot_velocity(stagnation_pressure: f64, static_pressure: f64, density: f64) -> f64 {
    assert!(density > 0.0, "la masse volumique doit être > 0");
    assert!(
        stagnation_pressure >= static_pressure,
        "p0 ≥ p requis (pression dynamique ≥ 0)"
    );
    (2.0 * (stagnation_pressure - static_pressure) / density).sqrt()
}

/// Pression dynamique `q = ½·ρ·v²` (Pa).
///
/// Panique si `density < 0`.
pub fn pitot_dynamic_pressure(velocity: f64, density: f64) -> f64 {
    assert!(density >= 0.0, "la masse volumique doit être ≥ 0");
    0.5 * density * velocity * velocity
}

/// Pression d'arrêt `p0 = p + ½·ρ·v²` (Pa) — inverse de [`pitot_velocity`].
///
/// Panique si `density < 0`.
pub fn pitot_stagnation_pressure(static_pressure: f64, velocity: f64, density: f64) -> f64 {
    assert!(density >= 0.0, "la masse volumique doit être ≥ 0");
    static_pressure + pitot_dynamic_pressure(velocity, density)
}

/// Débit volumique `Q = v·A` (m³/s) — vitesse ponctuelle supposée uniforme sur `A`.
///
/// Panique si `area < 0`.
pub fn pitot_volumetric_flow(velocity: f64, area: f64) -> f64 {
    assert!(area >= 0.0, "l'aire de passage doit être ≥ 0");
    velocity * area
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn velocity_and_stagnation_pressure_are_inverse() {
        // p0 reconstruit à partir de v doit redonner la pression d'arrêt initiale.
        let (p, rho) = (101_325.0, 1.225);
        let p0 = 101_425.0; // p0 − p = 100 Pa
        let v = pitot_velocity(p0, p, rho);
        assert_relative_eq!(
            pitot_stagnation_pressure(p, v, rho),
            p0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn dynamic_pressure_equals_pressure_difference() {
        // q = ½ρv² doit égaler p0 − p par construction de v.
        let (p, rho) = (95_000.0, 1.2);
        let p0 = 95_250.0; // Δp = 250 Pa
        let v = pitot_velocity(p0, p, rho);
        assert_relative_eq!(pitot_dynamic_pressure(v, rho), 250.0, max_relative = 1e-12);
    }

    #[test]
    fn velocity_scales_with_root_dynamic_pressure() {
        // v ∝ √(p0 − p) : quadrupler la dépression double la vitesse.
        let rho = 1.225;
        let v1 = pitot_velocity(100.0, 0.0, rho);
        let v2 = pitot_velocity(400.0, 0.0, rho);
        assert_relative_eq!(v2 / v1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn realistic_air_pitot_measurement() {
        // Air ρ = 1,225 kg/m³, p0 − p = 100 Pa :
        //   v = √(2·100 / 1,225) = √163,265306… = 12,777531… m/s
        // Débit dans une section A = 0,2 m² :
        //   Q = 12,777531·0,2 = 2,555506… m³/s
        let v = pitot_velocity(100.0, 0.0, 1.225);
        assert_relative_eq!(v, 12.777_531_299_998_798, max_relative = 1e-12);
        let q = pitot_volumetric_flow(v, 0.2);
        assert_relative_eq!(q, 2.555_506_259_999_76, max_relative = 1e-12);
    }

    #[test]
    fn flow_scales_linearly_with_area() {
        // À vitesse fixée, Q ∝ A.
        let v = 12.5;
        let q1 = pitot_volumetric_flow(v, 0.1);
        let q2 = pitot_volumetric_flow(v, 0.3);
        assert_relative_eq!(q2 / q1, 3.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "p0 ≥ p requis")]
    fn stagnation_below_static_panics() {
        pitot_velocity(90_000.0, 100_000.0, 1.225);
    }
}
