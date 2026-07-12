//! Accouplements à plateaux (flasques boulonnés) — couple transmissible par les
//! boulons sur un cercle de perçage, et conversion puissance ↔ couple.
//!
//! ```text
//! couple par les boulons  C = n·F·R_bc
//! effort par boulon       F = C/(n·R_bc)
//! couple depuis puissance C = P/ω
//! ```
//!
//! `n` nombre de boulons, `F` effort de cisaillement par boulon (N), `R_bc` rayon
//! du cercle de perçage (m), `C` couple (N·m), `P` puissance (W), `ω` vitesse
//! angulaire (rad/s). Les boulons, répartis sur le cercle de perçage, reprennent
//! le couple en cisaillement.
//!
//! **Convention** : SI cohérent. **Limite honnête** : accouplement **rigide**,
//! couple **également réparti** entre boulons ajustés (pas de glissement, pas de
//! désalignement) ; le dimensionnement au cisaillement/matage de chaque boulon
//! se fait ensuite via [`crate::riveted_joints`] ou [`crate::fastener_groups`].

/// Couple transmissible par les boulons `C = n·F·R_bc` (N·m).
pub fn torque_from_bolts(bolt_count: u32, bolt_force: f64, bolt_circle_radius: f64) -> f64 {
    bolt_count as f64 * bolt_force * bolt_circle_radius
}

/// Effort de cisaillement par boulon `F = C/(n·R_bc)` (N).
///
/// Panique si `n·R_bc <= 0`.
pub fn bolt_force_from_torque(torque: f64, bolt_count: u32, bolt_circle_radius: f64) -> f64 {
    let denom = bolt_count as f64 * bolt_circle_radius;
    assert!(denom > 0.0, "n·R_bc doit être strictement positif");
    torque / denom
}

/// Couple transmis à partir de la puissance `C = P/ω` (N·m).
///
/// Panique si `omega <= 0`.
pub fn power_to_torque(power_w: f64, omega_rad_s: f64) -> f64 {
    assert!(
        omega_rad_s > 0.0,
        "la vitesse angulaire doit être strictement positive"
    );
    power_w / omega_rad_s
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn torque_and_bolt_force_are_inverse() {
        // n=6 boulons, F=5000 N, R_bc=80 mm → C = 6·5000·0,08 = 2400 N·m.
        let c = torque_from_bolts(6, 5000.0, 0.080);
        assert_relative_eq!(c, 2400.0, epsilon = 1e-9);
        // Réciproque : F = C/(n·R_bc).
        assert_relative_eq!(bolt_force_from_torque(c, 6, 0.080), 5000.0, epsilon = 1e-9);
    }

    #[test]
    fn more_bolts_share_the_load() {
        // À couple égal, doubler le nombre de boulons halve l'effort par boulon.
        let f6 = bolt_force_from_torque(2400.0, 6, 0.080);
        let f12 = bolt_force_from_torque(2400.0, 12, 0.080);
        assert_relative_eq!(f6 / f12, 2.0, epsilon = 1e-9);
    }

    #[test]
    fn power_to_torque_conversion() {
        // P=15 kW à ω=100 rad/s → C = 150 N·m.
        assert_relative_eq!(power_to_torque(15_000.0, 100.0), 150.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "n·R_bc")]
    fn zero_radius_panics() {
        bolt_force_from_torque(2400.0, 6, 0.0);
    }
}
