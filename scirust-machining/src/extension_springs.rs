//! Ressorts de **traction** (extension) hélicoïdaux — raideur, tension initiale
//! et effort/flèche au-delà du décollement des spires.
//!
//! ```text
//! raideur         k = G·d⁴/(8·D³·n)
//! flèche          x = (F − Fi)/k          (F > Fi ; jointif au repos)
//! effort          F = Fi + k·x
//! cisaillement    τ = Kw·8·F·D/(π·d³)      (facteur de Wahl Kw)
//! ```
//!
//! `G` module de cisaillement (Pa), `d` diamètre du fil (m), `D` diamètre moyen
//! (m), `n` spires actives, `Fi` **tension initiale** (les spires jointives ne
//! s'écartent qu'au-delà de `Fi`), `Kw` facteur de Wahl. La raideur suit la même
//! loi qu'un ressort de compression, mais la tension initiale décale la courbe.
//!
//! **Convention** : SI cohérent. **Limite honnête** : corps du ressort en
//! cisaillement de torsion (comme la compression) ; la **tension initiale** et
//! la contrainte de flexion des **crochets** (souvent dimensionnante) sont des
//! données fournies par l'appelant, non calculées ici.

use core::f64::consts::PI;

/// Raideur `k = G·d⁴/(8·D³·n)` (N/m).
///
/// Panique si `coil_diameter <= 0` ou `active_coils <= 0`.
pub fn rate(shear_modulus: f64, wire_diameter: f64, coil_diameter: f64, active_coils: f64) -> f64 {
    assert!(
        coil_diameter > 0.0 && active_coils > 0.0,
        "D > 0 et n > 0 requis"
    );
    shear_modulus * wire_diameter.powi(4) / (8.0 * coil_diameter.powi(3) * active_coils)
}

/// Flèche au-delà de la tension initiale `x = (F − Fi)/k` (m).
///
/// Panique si `rate <= 0` ou si `force < initial_tension` (spires encore jointives).
pub fn deflection(force: f64, initial_tension: f64, rate: f64) -> f64 {
    assert!(rate > 0.0, "la raideur doit être strictement positive");
    assert!(
        force >= initial_tension,
        "l'effort doit dépasser la tension initiale (spires jointives)"
    );
    (force - initial_tension) / rate
}

/// Effort à une flèche donnée `F = Fi + k·x` (N).
pub fn force_at_deflection(initial_tension: f64, rate: f64, deflection: f64) -> f64 {
    initial_tension + rate * deflection
}

/// Cisaillement corrigé du corps `τ = Kw·8·F·D/(π·d³)` (Pa).
///
/// Panique si `wire_diameter <= 0`.
pub fn body_shear_stress(
    wahl_factor: f64,
    force: f64,
    coil_diameter: f64,
    wire_diameter: f64,
) -> f64 {
    assert!(
        wire_diameter > 0.0,
        "le diamètre du fil doit être strictement positif"
    );
    wahl_factor * 8.0 * force * coil_diameter / (PI * wire_diameter.powi(3))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn rate_matches_helical_formula() {
        let k = rate(80e9, 0.002, 0.016, 10.0);
        assert_relative_eq!(
            k,
            80e9 * 0.002f64.powi(4) / (8.0 * 0.016f64.powi(3) * 10.0),
            epsilon = 1e-6
        );
    }

    #[test]
    fn initial_tension_offsets_the_curve() {
        // Fi=20 N, k=1000 N/m. À F=120 N → x = 100/1000 = 0,1 m.
        let x = deflection(120.0, 20.0, 1000.0);
        assert_relative_eq!(x, 0.1, epsilon = 1e-9);
        // Réciproque : F = Fi + k·x.
        assert_relative_eq!(
            force_at_deflection(20.0, 1000.0, 0.1),
            120.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn no_deflection_at_initial_tension() {
        // À F = Fi exactement, la flèche est nulle.
        assert_relative_eq!(deflection(20.0, 20.0, 1000.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn shear_stress_grows_with_load() {
        let t1 = body_shear_stress(1.2, 100.0, 0.016, 0.002);
        let t2 = body_shear_stress(1.2, 200.0, 0.016, 0.002);
        assert_relative_eq!(t2 / t1, 2.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "tension initiale")]
    fn below_initial_tension_panics() {
        deflection(10.0, 20.0, 1000.0);
    }
}
