//! Maintenance — indicateurs de **disponibilité** et de maintenabilité :
//! MTBF, MTTR, disponibilité intrinsèque et probabilité de réparation.
//!
//! ```text
//! MTBF             = temps de fonctionnement / n_défaillances
//! MTTR             = temps d'arrêt / n_réparations
//! disponibilité    A = MTBF/(MTBF + MTTR)
//! maintenabilité   M(t) = 1 − e^{−t/MTTR}   (probabilité de réparer en ≤ t)
//! ```
//!
//! `MTBF` temps moyen de bon fonctionnement, `MTTR` temps moyen de réparation,
//! `A` disponibilité intrinsèque (fraction de temps opérationnel), `M(t)`
//! maintenabilité (probabilité qu'une réparation soit achevée avant `t`).
//!
//! **Convention** : temps cohérents. **Limite honnête** : disponibilité
//! **intrinsèque** (n'inclut ni les délais logistiques, ni la maintenance
//! préventive) ; la maintenabilité suppose des durées de réparation
//! **exponentielles**. Se combine avec [`crate::reliability`]. Pour le TRS d'un
//! équipement de production, voir [`crate::oee`].

/// MTBF `= temps de fonctionnement / n_défaillances`.
///
/// Panique si `failures == 0`.
pub fn mtbf(operating_time: f64, failures: u32) -> f64 {
    assert!(failures > 0, "au moins une défaillance est requise");
    operating_time / failures as f64
}

/// MTTR `= temps d'arrêt / n_réparations`.
///
/// Panique si `repairs == 0`.
pub fn mttr(downtime: f64, repairs: u32) -> f64 {
    assert!(repairs > 0, "au moins une réparation est requise");
    downtime / repairs as f64
}

/// Disponibilité intrinsèque `A = MTBF/(MTBF + MTTR)`.
///
/// Panique si `MTBF + MTTR <= 0`.
pub fn inherent_availability(mtbf: f64, mttr: f64) -> f64 {
    let sum = mtbf + mttr;
    assert!(sum > 0.0, "MTBF + MTTR doit être strictement positif");
    mtbf / sum
}

/// Maintenabilité `M(t) = 1 − e^{−t/MTTR}` (probabilité de réparation en ≤ `t`).
///
/// Panique si `mttr <= 0` ou `time < 0`.
pub fn maintainability(time: f64, mttr: f64) -> f64 {
    assert!(mttr > 0.0 && time >= 0.0, "MTTR > 0 et t ≥ 0 requis");
    1.0 - (-time / mttr).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn mtbf_and_mttr_from_records() {
        // 9000 h de marche, 3 pannes → MTBF = 3000 h. 12 h d'arrêt, 3 réparations → MTTR = 4 h.
        assert_relative_eq!(mtbf(9000.0, 3), 3000.0, epsilon = 1e-9);
        assert_relative_eq!(mttr(12.0, 3), 4.0, epsilon = 1e-9);
    }

    #[test]
    fn availability_high_when_mttr_small() {
        // MTBF=3000, MTTR=4 → A = 3000/3004 ≈ 0,9987.
        let a = inherent_availability(3000.0, 4.0);
        assert_relative_eq!(a, 3000.0 / 3004.0, epsilon = 1e-12);
        assert!(a > 0.99);
    }

    #[test]
    fn shorter_repairs_improve_availability() {
        // Réduire le MTTR augmente la disponibilité.
        assert!(inherent_availability(3000.0, 2.0) > inherent_availability(3000.0, 10.0));
    }

    #[test]
    fn maintainability_reaches_certainty() {
        // À t=0 → M=0 ; à t = MTTR → 1−1/e ≈ 0,632 ; t≫MTTR → 1.
        assert_relative_eq!(maintainability(0.0, 4.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            maintainability(4.0, 4.0),
            1.0 - 1.0 / core::f64::consts::E,
            epsilon = 1e-9
        );
        assert!(maintainability(40.0, 4.0) > 0.99);
    }

    #[test]
    #[should_panic(expected = "au moins une défaillance")]
    fn zero_failures_panics() {
        mtbf(9000.0, 0);
    }
}
