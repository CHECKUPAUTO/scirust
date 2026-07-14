//! **Embrayage à friction en glissement** — énergie dissipée, durée de
//! glissement et échauffement lors de la synchronisation de deux inerties.
//!
//! ```text
//! énergie de glissement   Q = ½·J·Δω²          (dissipée pour synchroniser)
//! durée de glissement     t = J·Δω / T          (couple d'embrayage constant)
//! élévation de temp.      ΔT = Q / (m·c)         (échauffement adiabatique)
//! ```
//!
//! `J` moment d'inertie ramené sur l'arbre (kg·m²), `Δω` écart de vitesse
//! angulaire initial (rad/s), `Q` énergie de glissement dissipée (J), `T`
//! couple de friction transmis par l'embrayage (N·m), `t` durée de glissement
//! (s), `m` masse chauffée des garnitures/plateaux (kg), `c` chaleur massique
//! (J/(kg·K)), `ΔT` élévation de température (K).
//!
//! **Convention** : SI. **Limite honnête** : le couple d'embrayage `T` est
//! supposé **constant** pendant tout le glissement, **toute** l'énergie de
//! glissement est convertie en **chaleur** dans l'embrayage, et l'échauffement
//! est **adiabatique** (aucune évacuation vers l'extérieur pendant la phase de
//! glissement). Le modèle synchronise une inertie unique `J` vers une consigne
//! (côté opposé de raideur/inertie infinie) ; le partage de chaleur entre les
//! deux faces et la dépendance du frottement à la température/vitesse ne sont
//! pas traités. Les propriétés matériaux (`m`, `c`) et le couple `T` sont
//! **fournis par l'appelant**. Voir [`crate::frictional_heating`] (contact
//! glissant) et [`crate::brakes`] (freinage).

/// Énergie de glissement dissipée `Q = ½·J·Δω²`.
///
/// Énergie cinétique de l'écart de vitesse, entièrement dissipée en chaleur
/// pour synchroniser l'inertie `J` (approximation à couple constant).
///
/// Panique si `inertia < 0` (le carré rend le signe de `Δω` indifférent).
pub fn clutch_slip_energy(inertia: f64, initial_speed_diff_rad: f64) -> f64 {
    assert!(inertia >= 0.0, "l'inertie J doit être positive");
    let half = 0.5_f64;
    half * inertia * initial_speed_diff_rad.powi(2)
}

/// Durée de glissement `t = J·Δω / T` (couple d'embrayage constant).
///
/// Temps nécessaire pour annuler l'écart de vitesse `Δω` sous le couple de
/// friction constant `T` appliqué à l'inertie `J`.
///
/// Panique si `inertia < 0`, `speed_diff_rad < 0` ou `torque <= 0`.
pub fn clutch_slip_time(inertia: f64, torque: f64, speed_diff_rad: f64) -> f64 {
    assert!(inertia >= 0.0, "l'inertie J doit être positive");
    assert!(
        speed_diff_rad >= 0.0,
        "l'écart de vitesse Δω doit être positif"
    );
    assert!(
        torque > 0.0,
        "le couple d'embrayage T doit être strictement positif"
    );
    inertia * speed_diff_rad / torque
}

/// Élévation de température adiabatique `ΔT = Q / (m·c)`.
///
/// Échauffement des garnitures/plateaux si toute l'énergie `Q` reste dans la
/// masse `m` de chaleur massique `c` (sans évacuation).
///
/// Panique si `slip_energy < 0`, `mass <= 0` ou `specific_heat <= 0`.
pub fn clutch_temperature_rise(slip_energy: f64, mass: f64, specific_heat: f64) -> f64 {
    assert!(
        slip_energy >= 0.0,
        "l'énergie de glissement Q doit être positive"
    );
    assert!(mass > 0.0, "la masse m doit être strictement positive");
    assert!(
        specific_heat > 0.0,
        "la chaleur massique c doit être strictement positive"
    );
    slip_energy / (mass * specific_heat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn slip_energy_scales_quadratically_with_speed_diff() {
        // Doubler Δω quadruple l'énergie dissipée (Q ∝ Δω²).
        let q1 = clutch_slip_energy(0.5, 50.0);
        let q2 = clutch_slip_energy(0.5, 100.0);
        assert_relative_eq!(q2, 4.0 * q1, epsilon = 1e-9);
    }

    #[test]
    fn slip_energy_matches_kinetic_energy() {
        // J=2 kg·m², Δω=10 rad/s → Q = ½·2·10² = 100 J.
        assert_relative_eq!(clutch_slip_energy(2.0, 10.0), 100.0, epsilon = 1e-9);
        // Le signe de Δω est indifférent (terme au carré).
        assert_relative_eq!(
            clutch_slip_energy(2.0, -10.0),
            clutch_slip_energy(2.0, 10.0),
            epsilon = 1e-12
        );
    }

    #[test]
    fn slip_time_is_inversely_proportional_to_torque() {
        // À J et Δω fixés, doubler le couple divise la durée par deux.
        let t1 = clutch_slip_time(1.5, 40.0, 200.0);
        let t2 = clutch_slip_time(1.5, 80.0, 200.0);
        assert_relative_eq!(t2, t1 / 2.0, epsilon = 1e-12);
    }

    #[test]
    fn slip_time_from_impulse_momentum() {
        // Théorème du moment cinétique : T·t = J·Δω.
        // J=1,5, T=50 N·m, Δω=30 rad/s → t = 0,9 s.
        let t = clutch_slip_time(1.5, 50.0, 30.0);
        assert_relative_eq!(t, 0.9, epsilon = 1e-9);
        assert_relative_eq!(50.0 * t, 1.5 * 30.0, epsilon = 1e-9);
    }

    #[test]
    fn temperature_rise_realistic_case() {
        // Synchronisation J=1 kg·m² depuis Δω=150 rad/s → Q = ½·1·150² = 11250 J
        // dans m=2 kg de garniture, c=1200 J/(kg·K) → ΔT = 11250/2400 ≈ 4,6875 K.
        let q = clutch_slip_energy(1.0, 150.0);
        let dt = clutch_temperature_rise(q, 2.0, 1200.0);
        assert_relative_eq!(q, 11250.0, epsilon = 1e-9);
        assert_relative_eq!(dt, 11250.0 / 2400.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "le couple d'embrayage T doit être strictement positif")]
    fn zero_torque_slip_time_panics() {
        clutch_slip_time(1.5, 0.0, 30.0);
    }
}
