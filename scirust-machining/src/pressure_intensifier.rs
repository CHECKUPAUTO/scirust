//! **Multiplicateur de pression hydraulique** (intensifier) — deux pistons
//! solidaires de sections différentes : la conservation de la force convertit une
//! basse pression sur la grande section en haute pression sur la petite section.
//!
//! ```text
//! rapport de pression   R = A_large / A_small
//! pression de sortie    p_out = p_in · A_large / A_small
//! débit de sortie       Q_out = Q_in · A_small / A_large
//! conservation force    p_in · A_large = p_out · A_small
//! conservation débit    Q_in · A_small = Q_out · A_large   (volumes égaux)
//! ```
//!
//! `A_large` section du piston basse pression (m²), `A_small` section du piston
//! haute pression (m²), `p_in` pression d'entrée sur la grande section (Pa),
//! `p_out` pression de sortie sur la petite section (Pa), `Q_in`/`Q_out` débits
//! volumiques d'entrée/sortie (m³/s), `R` rapport d'intensification (sans unité).
//!
//! **Convention** : unités SI (m², Pa, m³/s), pressions **relatives ou absolues**
//! de façon cohérente entre entrée et sortie.
//! **Limite honnête** : conservation de force **idéale** — frottements des joints,
//! fuites internes et compressibilité du fluide sont **négligés**, ce qui donne la
//! **borne haute** de la pression de sortie (le rendement réel la diminue). Les
//! sections des pistons sont des données géométriques fournies par l'appelant ;
//! aucune valeur « par défaut » de section, de pression ou de débit n'est supposée.

/// Rapport d'intensification `R = A_large / A_small` (sans unité).
///
/// Panique si une section est `<= 0`.
pub fn intensifier_pressure_ratio(large_area: f64, small_area: f64) -> f64 {
    assert!(
        large_area > 0.0 && small_area > 0.0,
        "les sections A_large et A_small doivent être strictement positives"
    );
    large_area / small_area
}

/// Pression de sortie `p_out = p_in · A_large / A_small` (loi de Pascal,
/// conservation de la force `p_in·A_large = p_out·A_small`), en Pa.
///
/// Panique si `input_pressure < 0` ou si une section est `<= 0`.
pub fn intensifier_output_pressure(input_pressure: f64, large_area: f64, small_area: f64) -> f64 {
    assert!(
        input_pressure >= 0.0,
        "la pression d'entrée p_in doit être positive ou nulle"
    );
    assert!(
        large_area > 0.0 && small_area > 0.0,
        "les sections A_large et A_small doivent être strictement positives"
    );
    input_pressure * large_area / small_area
}

/// Débit de sortie `Q_out = Q_in · A_small / A_large` (conservation du volume
/// balayé par les deux pistons solidaires), en m³/s.
///
/// Panique si `input_flow < 0` ou si une section est `<= 0`.
pub fn intensifier_output_flow(input_flow: f64, small_area: f64, large_area: f64) -> f64 {
    assert!(
        input_flow >= 0.0,
        "le débit d'entrée Q_in doit être positif ou nul"
    );
    assert!(
        small_area > 0.0 && large_area > 0.0,
        "les sections A_small et A_large doivent être strictement positives"
    );
    input_flow * small_area / large_area
}

/// Pression d'entrée requise `p_in = p_out · A_small / A_large` pour obtenir une
/// pression de sortie visée (réciproque de [`intensifier_output_pressure`]), en Pa.
///
/// Panique si `output_pressure < 0` ou si une section est `<= 0`.
pub fn intensifier_required_input_pressure(
    output_pressure: f64,
    large_area: f64,
    small_area: f64,
) -> f64 {
    assert!(
        output_pressure >= 0.0,
        "la pression de sortie p_out doit être positive ou nulle"
    );
    assert!(
        large_area > 0.0 && small_area > 0.0,
        "les sections A_large et A_small doivent être strictement positives"
    );
    output_pressure * small_area / large_area
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn ratio_matches_pressure_gain() {
        // Le gain de pression p_out/p_in est exactement le rapport de sections.
        let (a_large, a_small) = (2.0e-3_f64, 5.0e-4_f64);
        let p_in = 100e5_f64;
        let ratio = intensifier_pressure_ratio(a_large, a_small);
        let p_out = intensifier_output_pressure(p_in, a_large, a_small);
        assert_relative_eq!(p_out / p_in, ratio, epsilon = 1e-12);
    }

    #[test]
    fn force_is_conserved() {
        // Conservation de la force : p_in·A_large == p_out·A_small.
        let (a_large, a_small) = (1.5e-3_f64, 3.0e-4_f64);
        let p_in = 63e5_f64;
        let p_out = intensifier_output_pressure(p_in, a_large, a_small);
        assert_relative_eq!(p_in * a_large, p_out * a_small, epsilon = 1e-3);
    }

    #[test]
    fn power_is_conserved_ideally() {
        // Sans pertes, la puissance hydraulique se conserve : p_in·Q_in == p_out·Q_out.
        let (a_large, a_small) = (4.0e-3_f64, 8.0e-4_f64);
        let (p_in, q_in) = (50e5_f64, 1.2e-3_f64);
        let p_out = intensifier_output_pressure(p_in, a_large, a_small);
        let q_out = intensifier_output_flow(q_in, a_small, a_large);
        assert_relative_eq!(p_in * q_in, p_out * q_out, epsilon = 1e-9);
    }

    #[test]
    fn required_input_inverts_output_pressure() {
        // Réciprocité : viser p_out puis calculer p_in redonne bien p_out.
        let (a_large, a_small) = (2.5e-3_f64, 6.25e-4_f64);
        let p_out_target = 400e5_f64;
        let p_in = intensifier_required_input_pressure(p_out_target, a_large, a_small);
        let p_out = intensifier_output_pressure(p_in, a_large, a_small);
        assert_relative_eq!(p_out, p_out_target, epsilon = 1e-6);
    }

    #[test]
    fn realistic_four_to_one_intensifier() {
        // Cas chiffré : rapport 4:1, 60 bar en entrée -> 240 bar en sortie ;
        // débit 2 L/s en entrée -> 0,5 L/s en sortie.
        let (a_large, a_small) = (4.0e-3_f64, 1.0e-3_f64);
        assert_relative_eq!(
            intensifier_pressure_ratio(a_large, a_small),
            4.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            intensifier_output_pressure(60e5_f64, a_large, a_small),
            240e5_f64,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            intensifier_output_flow(2.0e-3_f64, a_small, a_large),
            0.5e-3_f64,
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "strictement positives")]
    fn zero_small_area_panics() {
        let _ = intensifier_pressure_ratio(1.0e-3_f64, 0.0_f64);
    }
}
