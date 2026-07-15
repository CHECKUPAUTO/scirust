//! **Performance aéraulique d'un ventilateur centrifuge** : elle relie le débit
//! volumique, les pressions (totale, dynamique, statique), la puissance utile de
//! l'air et les rendements ramenés à la puissance mesurée sur l'arbre.
//!
//! ```text
//! puissance utile de l'air   Pa = qv · pt
//! pression dynamique         pd = ½ · rho · v²
//! pression statique          ps = pt − pd
//! rendement total            eta_t = qv · pt / Parbre
//! rendement statique         eta_s = qv · ps / Parbre
//! ```
//!
//! `qv` débit volumique (m³/s), `pt` pression totale (Pa), `pd` pression dynamique
//! (Pa), `ps` pression statique (Pa), `rho` masse volumique de l'air (kg/m³), `v`
//! vitesse de l'air au refoulement (m/s), `Pa` puissance utile de l'air (W),
//! `Parbre` puissance mécanique sur l'arbre (W), `eta_t` et `eta_s` rendements
//! (sans dimension). Convention : unités SI cohérentes ; la pression totale est la
//! somme de la pression statique et de la pression dynamique.
//!
//! **Limite honnête** : ce bilan aéraulique suppose un air **incompressible**
//! (faible taux de compression, gaz peu échauffé) ; le débit et les pressions sont
//! **fournis** par l'appelant, ainsi que la **puissance à l'arbre** (mesurée, elle
//! englobe les pertes internes). Aucune masse volumique, aucun rendement ni aucune
//! constante n'est supposé « par défaut » : toutes les grandeurs sont fournies.
//! Distinct de [`crate::pump_affinity`] (lois de similitude, changement de point de
//! fonctionnement).

/// Puissance utile de l'air `Pa = qv · pt` (W).
///
/// C'est la puissance transmise au fluide, à distinguer de la puissance à l'arbre.
///
/// Panique si `volume_flow < 0` ou `total_pressure < 0`.
pub fn fan_air_power(volume_flow: f64, total_pressure: f64) -> f64 {
    assert!(
        volume_flow >= 0.0,
        "le débit volumique ne peut pas être négatif"
    );
    assert!(
        total_pressure >= 0.0,
        "la pression totale ne peut pas être négative"
    );
    volume_flow * total_pressure
}

/// Pression dynamique au refoulement `pd = ½ · rho · v²` (Pa).
///
/// Panique si `air_density < 0` (une masse volumique négative n'a pas de sens).
pub fn fan_velocity_pressure(air_density: f64, outlet_velocity: f64) -> f64 {
    assert!(
        air_density >= 0.0,
        "la masse volumique de l'air ne peut pas être négative"
    );
    0.5 * air_density * outlet_velocity * outlet_velocity
}

/// Pression statique `ps = pt − pd` (Pa).
///
/// La pression totale étant la somme de la statique et de la dynamique, la
/// statique s'obtient en retranchant la dynamique de la totale.
///
/// Panique si `velocity_pressure < 0` (une pression dynamique est toujours ≥ 0).
pub fn fan_static_pressure(total_pressure: f64, velocity_pressure: f64) -> f64 {
    assert!(
        velocity_pressure >= 0.0,
        "la pression dynamique ne peut pas être négative"
    );
    total_pressure - velocity_pressure
}

/// Rendement total `eta_t = qv · pt / Parbre` (sans dimension).
///
/// Rapport de la puissance utile de l'air (basée sur la pression totale) à la
/// puissance mécanique fournie sur l'arbre.
///
/// Panique si `volume_flow < 0`, `total_pressure < 0` ou `shaft_power <= 0`.
pub fn fan_total_efficiency(volume_flow: f64, total_pressure: f64, shaft_power: f64) -> f64 {
    assert!(
        volume_flow >= 0.0,
        "le débit volumique ne peut pas être négatif"
    );
    assert!(
        total_pressure >= 0.0,
        "la pression totale ne peut pas être négative"
    );
    assert!(
        shaft_power > 0.0,
        "la puissance à l'arbre doit être strictement positive"
    );
    volume_flow * total_pressure / shaft_power
}

/// Rendement statique `eta_s = qv · ps / Parbre` (sans dimension).
///
/// Rapport de la puissance utile de l'air basée sur la pression statique à la
/// puissance mécanique fournie sur l'arbre.
///
/// Panique si `volume_flow < 0` ou `shaft_power <= 0`.
pub fn fan_static_efficiency(volume_flow: f64, static_pressure: f64, shaft_power: f64) -> f64 {
    assert!(
        volume_flow >= 0.0,
        "le débit volumique ne peut pas être négatif"
    );
    assert!(
        shaft_power > 0.0,
        "la puissance à l'arbre doit être strictement positive"
    );
    volume_flow * static_pressure / shaft_power
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn pressure_decomposition_is_consistent() {
        // Identité pt = ps + pd : la statique reconstruit la totale avec la dynamique.
        let (rho, v, total) = (1.2_f64, 10.0_f64, 500.0_f64);
        let pd = fan_velocity_pressure(rho, v); // 0.5·1.2·100 = 60 Pa
        assert_relative_eq!(pd, 60.0, epsilon = 1e-12);
        let ps = fan_static_pressure(total, pd); // 500 − 60 = 440 Pa
        assert_relative_eq!(ps, 440.0, epsilon = 1e-12);
        assert_relative_eq!(ps + pd, total, epsilon = 1e-12);
    }

    #[test]
    fn velocity_pressure_scales_as_square() {
        // pd ∝ v² : doubler la vitesse quadruple la pression dynamique.
        let rho = 1.2_f64;
        let pd1 = fan_velocity_pressure(rho, 8.0);
        let pd2 = fan_velocity_pressure(rho, 16.0);
        assert_relative_eq!(pd2, 4.0 * pd1, epsilon = 1e-9);
    }

    #[test]
    fn air_power_defines_total_efficiency() {
        // Identité : eta_t = Pa / Parbre, donc Pa = eta_t · Parbre.
        let (qv, pt, shaft) = (2.0_f64, 500.0_f64, 2000.0_f64);
        let air = fan_air_power(qv, pt); // 2·500 = 1000 W
        assert_relative_eq!(air, 1000.0, epsilon = 1e-9);
        let eta_t = fan_total_efficiency(qv, pt, shaft); // 1000/2000 = 0.5
        assert_relative_eq!(eta_t, 0.5, epsilon = 1e-12);
        assert_relative_eq!(air, eta_t * shaft, epsilon = 1e-9);
    }

    #[test]
    fn static_efficiency_is_below_total() {
        // ps < pt (dynamique > 0) ⇒ eta_s < eta_t à débit et arbre identiques.
        let (qv, shaft) = (2.0_f64, 2000.0_f64);
        let pd = fan_velocity_pressure(1.2, 10.0); // 60 Pa
        let pt = 500.0_f64;
        let ps = fan_static_pressure(pt, pd); // 440 Pa
        let eta_t = fan_total_efficiency(qv, pt, shaft); // 0.5
        let eta_s = fan_static_efficiency(qv, ps, shaft); // 2·440/2000 = 0.44
        assert_relative_eq!(eta_s, 0.44, epsilon = 1e-12);
        assert!(eta_s < eta_t);
    }

    #[test]
    fn efficiency_scales_inversely_with_shaft_power() {
        // À puissance utile fixe, le rendement varie en 1/Parbre.
        let (qv, pt) = (1.5_f64, 300.0_f64);
        let eta_a = fan_total_efficiency(qv, pt, 900.0);
        let eta_b = fan_total_efficiency(qv, pt, 1800.0);
        assert_relative_eq!(eta_a, 2.0 * eta_b, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "puissance à l'arbre doit être strictement positive")]
    fn zero_shaft_power_panics() {
        let _ = fan_total_efficiency(2.0, 500.0, 0.0);
    }
}
