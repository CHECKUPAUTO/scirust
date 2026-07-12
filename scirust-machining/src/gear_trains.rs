//! Trains d'engrenages — rapports de transmission des trains simples et composés,
//! vitesse et couple de sortie, effet des roues folles.
//!
//! ```text
//! rapport d'un engrènement  i = Z_menant/Z_mené        (ω_mené/ω_menant)
//! train composé             i_tot = Π (Z_menant/Z_mené)
//! vitesse de sortie         ω_out = ω_in·i_tot
//! couple de sortie          C_out = C_in·(1/i_tot)·η
//! sens                      inversé si nombre d'engrènements externes impair
//! ```
//!
//! `Z` nombres de dents, `i` rapport de vitesses (`ω_sortie/ω_entrée`), `η`
//! rendement global, une **roue folle** (roue intermédiaire) ne modifie pas la
//! magnitude du rapport, seulement le sens. Un rapport `i < 1` est une réduction.
//!
//! **Convention** : vitesses/couples cohérents. **Limite honnête** : cinématique
//! **exacte** de trains à axes fixes (ordinaires) ; les trains **épicycloïdaux**
//! sont dans [`crate::epicyclic`]. Le rendement `η` est fourni par l'appelant.

/// Rapport de vitesses d'un engrènement `i = Z_menant/Z_mené`.
///
/// Panique si `driven_teeth == 0`.
pub fn gear_pair_speed_ratio(driver_teeth: u32, driven_teeth: u32) -> f64 {
    assert!(
        driven_teeth > 0,
        "la roue menée doit avoir au moins une dent"
    );
    driver_teeth as f64 / driven_teeth as f64
}

/// Rapport d'un **train composé** `i_tot = Π(Z_menant/Z_mené)`.
///
/// Panique si les listes diffèrent en longueur ou si une roue menée a `0` dent.
pub fn compound_train_ratio(driver_teeth: &[u32], driven_teeth: &[u32]) -> f64 {
    assert!(
        driver_teeth.len() == driven_teeth.len() && !driver_teeth.is_empty(),
        "menants et menés doivent avoir la même longueur non nulle"
    );
    driver_teeth
        .iter()
        .zip(driven_teeth)
        .map(|(&dr, &dn)| gear_pair_speed_ratio(dr, dn))
        .product()
}

/// Vitesse de sortie `ω_out = ω_in·i_tot`.
pub fn output_speed(input_speed: f64, train_ratio: f64) -> f64 {
    input_speed * train_ratio
}

/// Couple de sortie `C_out = C_in·(1/i_tot)·η`.
///
/// Panique si `train_ratio <= 0` ou `η` hors `]0, 1]`.
pub fn output_torque(input_torque: f64, train_ratio: f64, efficiency: f64) -> f64 {
    assert!(
        train_ratio > 0.0,
        "le rapport de train doit être strictement positif"
    );
    assert!(
        efficiency > 0.0 && efficiency <= 1.0,
        "le rendement doit être dans ]0, 1]"
    );
    input_torque / train_ratio * efficiency
}

/// Vrai si le sens de rotation de sortie est **inversé** (nombre d'engrènements
/// externes impair).
pub fn is_direction_reversed(external_meshes: u32) -> bool {
    external_meshes % 2 == 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn single_pair_reduction() {
        // Pignon 20 dents entraîne roue 60 → i = 1/3 (réduction ×3).
        assert_relative_eq!(gear_pair_speed_ratio(20, 60), 1.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn compound_train_multiplies_ratios() {
        // Deux étages 20/60 et 15/45 → i = (1/3)·(1/3) = 1/9.
        let i = compound_train_ratio(&[20, 15], &[60, 45]);
        assert_relative_eq!(i, 1.0 / 9.0, epsilon = 1e-12);
        // Entrée 900 tr/min → sortie 100 tr/min.
        assert_relative_eq!(output_speed(900.0, i), 100.0, epsilon = 1e-9);
    }

    #[test]
    fn torque_increases_as_speed_decreases() {
        // Réduction ×9 (i=1/9), η=0,95 → couple ×8,55 environ.
        let i = 1.0 / 9.0;
        let c = output_torque(10.0, i, 0.95);
        assert_relative_eq!(c, 10.0 * 9.0 * 0.95, epsilon = 1e-9);
        assert!(c > 10.0);
    }

    #[test]
    fn idler_only_flips_direction() {
        // Roue folle : un engrènement de plus inverse le sens sans changer i.
        assert!(is_direction_reversed(1)); // une paire externe → inversé
        assert!(!is_direction_reversed(2)); // roue folle intercalée → même sens
    }

    #[test]
    #[should_panic(expected = "même longueur")]
    fn mismatched_lists_panic() {
        compound_train_ratio(&[20, 15], &[60]);
    }
}
