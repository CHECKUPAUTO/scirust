//! Hydraulique de puissance — **vitesse de fluide** en conduite circulaire pleine
//! et **dimensionnement** du diamètre à partir d'un débit volumique.
//!
//! ```text
//! vitesse            v = Q / (π/4 · d²)
//! diamètre requis    d = √(4·Q / (π·v))
//! acceptable si      v ≤ v_max
//! ```
//!
//! `Q` débit volumique (m³/s), `d` diamètre intérieur de la conduite (m),
//! `v` vitesse moyenne du fluide (m/s), `v_max` vitesse maximale admissible (m/s).
//!
//! **Convention** : conduite **circulaire pleine**, unités **SI** cohérentes,
//! vitesse **moyenne** de débit (section pleine).
//! **Limite honnête** : régime **établi** en conduite pleine ; les vitesses
//! maximales recommandées (aspiration ≈ 1,5 m/s, refoulement ≈ 5 m/s) dépendent
//! du fluide, de la viscosité et du procédé et sont **fournies par l'appelant** ;
//! aucune valeur « par défaut » n'est inventée ici.

use core::f64::consts::PI;

/// Vitesse moyenne du fluide `v = Q / (π/4 · d²)` (m/s).
///
/// Panique si `flow_rate < 0` ou `pipe_diameter <= 0`.
pub fn hydvel_flow_velocity(flow_rate: f64, pipe_diameter: f64) -> f64 {
    assert!(flow_rate >= 0.0, "le débit Q doit être positif ou nul");
    assert!(
        pipe_diameter > 0.0,
        "le diamètre d doit être strictement positif"
    );
    flow_rate / (PI / 4.0 * pipe_diameter * pipe_diameter)
}

/// Diamètre intérieur requis pour atteindre `target_velocity` :
/// `d = √(4·Q / (π·v))` (m).
///
/// Panique si `flow_rate < 0` ou `target_velocity <= 0`.
pub fn hydvel_pipe_diameter_for_velocity(flow_rate: f64, target_velocity: f64) -> f64 {
    assert!(flow_rate >= 0.0, "le débit Q doit être positif ou nul");
    assert!(
        target_velocity > 0.0,
        "la vitesse cible v doit être strictement positive"
    );
    (4.0_f64 * flow_rate / (PI * target_velocity)).sqrt()
}

/// Vrai si la vitesse respecte le plafond : `v ≤ v_max`.
///
/// Panique si `velocity < 0` ou `max_velocity <= 0`.
pub fn hydvel_is_velocity_acceptable(velocity: f64, max_velocity: f64) -> bool {
    assert!(velocity >= 0.0, "la vitesse v doit être positive ou nulle");
    assert!(
        max_velocity > 0.0,
        "la vitesse maximale v_max doit être strictement positive"
    );
    velocity <= max_velocity
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn velocity_and_diameter_are_reciprocal() {
        // À débit fixé, dimensionner pour v puis recalculer la vitesse
        // sur ce diamètre doit redonner v.
        let flow_rate = 0.01; // m³/s
        let target = 3.0; // m/s
        let d = hydvel_pipe_diameter_for_velocity(flow_rate, target);
        let v = hydvel_flow_velocity(flow_rate, d);
        assert_relative_eq!(v, target, max_relative = 1e-12);
    }

    #[test]
    fn velocity_scales_inverse_square_with_diameter() {
        // v = Q/(π/4·d²) : doubler d divise la vitesse par 4.
        let q = 0.02;
        let v1 = hydvel_flow_velocity(q, 0.05);
        let v2 = hydvel_flow_velocity(q, 0.10);
        assert_relative_eq!(v1 / v2, 4.0, epsilon = 1e-12);
    }

    #[test]
    fn diameter_scales_with_sqrt_of_flow() {
        // d ∝ √Q à vitesse fixée : quadrupler Q double d.
        let v = 5.0;
        let d1 = hydvel_pipe_diameter_for_velocity(0.005, v);
        let d2 = hydvel_pipe_diameter_for_velocity(0.020, v);
        assert_relative_eq!(d2 / d1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn realistic_case_refoulement() {
        // Q = 60 L/min = 1e-3 m³/s dans un tube d = 16 mm.
        // A = π/4·0,016² = 2,0106e-4 m² → v = 1e-3/2,0106e-4 ≈ 4,974 m/s.
        let v = hydvel_flow_velocity(1.0e-3, 0.016);
        assert_relative_eq!(v, 4.97359, max_relative = 1e-4);
        // Vitesse de refoulement plafonnée à 5 m/s (fournie par l'appelant).
        assert!(hydvel_is_velocity_acceptable(v, 5.0));
    }

    #[test]
    fn acceptability_boundary_is_inclusive() {
        // v = v_max doit être accepté (borne inclusive).
        assert!(hydvel_is_velocity_acceptable(1.5, 1.5));
        assert!(!hydvel_is_velocity_acceptable(1.6, 1.5));
    }

    #[test]
    fn zero_flow_gives_zero_velocity() {
        // Débit nul → vitesse nulle (cas limite valide).
        assert_relative_eq!(hydvel_flow_velocity(0.0, 0.02), 0.0, epsilon = 1e-15);
    }

    #[test]
    #[should_panic(expected = "diamètre d doit être strictement positif")]
    fn zero_diameter_panics() {
        hydvel_flow_velocity(0.01, 0.0);
    }
}
