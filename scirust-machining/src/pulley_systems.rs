//! Poulies et moufles (palans, block and tackle) — avantage mécanique, effort à
//! fournir, rapport de vitesses et rendement.
//!
//! ```text
//! avantage mécanique idéal   AM_id = n         (n brins portant la charge)
//! rapport de vitesses        VR = n
//! effort réel                F = W/(n·η)
//! avantage mécanique réel    AM = W/F
//! rendement                  η = AM/VR
//! ```
//!
//! `n` nombre de brins de corde supportant la charge (= rapport de vitesses d'un
//! palan idéal), `W` charge (N), `F` effort au brin libre (N), `η` rendement
//! (pertes de poulies). Sans perte, `AM = VR = n` ; avec pertes, `AM < VR`.
//!
//! **Convention** : SI cohérent. **Limite honnête** : palan idéalisé à `n`
//! brins parallèles ; le rendement `η` (pertes aux réas) est une donnée globale
//! fournie par l'appelant, pas calculée poulie par poulie.

/// Avantage mécanique / rapport de vitesses idéal `= n` (nombre de brins).
///
/// Panique si `n == 0`.
pub fn velocity_ratio(supporting_ropes: u32) -> f64 {
    assert!(supporting_ropes > 0, "il faut au moins un brin porteur");
    supporting_ropes as f64
}

/// Effort à fournir au brin libre `F = W/(n·η)` (N).
///
/// Panique si `n == 0` ou `efficiency` hors `]0, 1]`.
pub fn effort_required(load: f64, supporting_ropes: u32, efficiency: f64) -> f64 {
    assert!(supporting_ropes > 0, "il faut au moins un brin porteur");
    assert!(
        efficiency > 0.0 && efficiency <= 1.0,
        "le rendement doit être dans ]0, 1]"
    );
    load / (supporting_ropes as f64 * efficiency)
}

/// Avantage mécanique réel `AM = W/F`.
///
/// Panique si `effort <= 0`.
pub fn actual_mechanical_advantage(load: f64, effort: f64) -> f64 {
    assert!(effort > 0.0, "l'effort doit être strictement positif");
    load / effort
}

/// Rendement du palan `η = AM/VR`.
///
/// Panique si `velocity_ratio <= 0`.
pub fn efficiency(actual_mechanical_advantage: f64, velocity_ratio: f64) -> f64 {
    assert!(
        velocity_ratio > 0.0,
        "le rapport de vitesses doit être strictement positif"
    );
    actual_mechanical_advantage / velocity_ratio
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn ideal_tackle_needs_load_over_n() {
        // Palan idéal (η=1) à 4 brins : F = W/4.
        assert_relative_eq!(effort_required(4000.0, 4, 1.0), 1000.0, epsilon = 1e-9);
        assert_relative_eq!(velocity_ratio(4), 4.0, epsilon = 1e-12);
    }

    #[test]
    fn losses_increase_the_effort() {
        // η=0,8, 4 brins : F = 4000/(4·0,8) = 1250 N > 1000 N idéal.
        let f = effort_required(4000.0, 4, 0.8);
        assert_relative_eq!(f, 1250.0, epsilon = 1e-9);
        assert!(f > effort_required(4000.0, 4, 1.0));
    }

    #[test]
    fn efficiency_recovers_from_advantage_and_ratio() {
        // Avec F=1250, AM = 4000/1250 = 3,2 ; VR=4 → η = 0,8.
        let am = actual_mechanical_advantage(4000.0, 1250.0);
        assert_relative_eq!(am, 3.2, epsilon = 1e-9);
        assert_relative_eq!(efficiency(am, velocity_ratio(4)), 0.8, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "au moins un brin")]
    fn zero_ropes_panics() {
        velocity_ratio(0);
    }
}
