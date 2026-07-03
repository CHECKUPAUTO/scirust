//! Règle de Palmgren-Miner (cumul de dommage linéaire) — combine les
//! cycles issus du comptage rainflow ([`super::rainflow`]) avec une
//! courbe S-N pour estimer le dommage de fatigue cumulé.
//!
//! `D = Σ nᵢ/Nᵢ`, `nᵢ` = nombre de cycles subis à la plage de contrainte
//! `i`, `Nᵢ` = nombre de cycles à rupture à cette même plage (lu sur la
//! courbe S-N du matériau). Rupture prédite quand `D >= 1` (Palmgren
//! 1924, Miner 1945 — la règle elle-même est un principe de comptabilité
//! de dommage, pas un résultat expérimental : elle ignore l'ordre de
//! chargement et les effets de séquence, une limite documentée et
//! largement acceptée en pratique pour un premier dimensionnement).
//!
//! **Limite honnête** : ce module ne fournit aucune courbe S-N de
//! matériau réel — [`PowerLawSnCurve`] est un modèle générique en loi de
//! puissance (Basquin) dont l'appelant doit fournir les coefficients
//! (ajustés sur des données d'essai du matériau/alliage/traitement
//! considéré). Prétendre à une courbe S-N "par défaut" serait une
//! affirmation non vérifiable pour un matériau non spécifié.

/// Courbe S-N en loi de puissance (Basquin) : `N(S) = coefficient · S^(-exponent)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowerLawSnCurve {
    pub coefficient: f64,
    pub exponent: f64,
}

impl PowerLawSnCurve {
    /// Nombre de cycles à rupture pour une plage de contrainte `stress_range`.
    pub fn cycles_to_failure(&self, stress_range: f64) -> f64 {
        self.coefficient * stress_range.powf(-self.exponent)
    }
}

/// Dommage cumulé de Miner sur un jeu de `(plage, nombre de cycles)` —
/// typiquement la sortie de
/// [`super::rainflow::aggregate_by_range`]. Rupture prédite quand le
/// résultat atteint ou dépasse `1.0`.
pub fn miner_damage(cycles: &[(f64, f64)], sn_curve: &PowerLawSnCurve) -> f64 {
    cycles
        .iter()
        .map(|&(range, count)| count / sn_curve.cycles_to_failure(range))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn cycles_to_failure_matches_the_power_law_definition() {
        let curve = PowerLawSnCurve {
            coefficient: 1.0e6,
            exponent: 3.0,
        };
        assert_relative_eq!(curve.cycles_to_failure(10.0), 1000.0, epsilon = 1e-9);
        assert_relative_eq!(curve.cycles_to_failure(100.0), 1.0, epsilon = 1e-9);
    }

    #[test]
    fn damage_sums_the_ratio_of_applied_to_allowable_cycles() {
        let curve = PowerLawSnCurve {
            coefficient: 1.0e6,
            exponent: 3.0,
        };
        // N(10)=1000, N(20)=125: 500/1000 + 50/125 = 0.5 + 0.4 = 0.9.
        let cycles = vec![(10.0, 500.0), (20.0, 50.0)];
        assert_relative_eq!(miner_damage(&cycles, &curve), 0.9, epsilon = 1e-9);
    }

    #[test]
    fn damage_of_exactly_one_signals_predicted_failure() {
        let curve = PowerLawSnCurve {
            coefficient: 1.0e6,
            exponent: 3.0,
        };
        let cycles = vec![(10.0, 1000.0)]; // exactly N(10) cycles applied
        assert_relative_eq!(miner_damage(&cycles, &curve), 1.0, epsilon = 1e-9);
    }

    #[test]
    fn no_cycles_means_no_damage() {
        let curve = PowerLawSnCurve {
            coefficient: 1.0e6,
            exponent: 3.0,
        };
        assert_relative_eq!(miner_damage(&[], &curve), 0.0, epsilon = 1e-9);
    }
}
