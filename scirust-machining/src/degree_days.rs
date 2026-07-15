//! **Degrés-jours** — estimation de tendance de la consommation énergétique d'un
//! bâtiment par la méthode quasi-statique des degrés-jours de chauffage/refroidissement.
//!
//! ```text
//! DJ chauffage   HDD = max(0, T_base − T_moy)        (K·jour, sur une journée)
//! DJ refroidiss. CDD = max(0, T_moy − T_base)        (K·jour, sur une journée)
//! énergie        E   = 86400·U·DJ                    (J, sur la période)
//! cumul annuel   DJ_an = Σ DJ_i                       (K·jour)
//! ```
//!
//! `T_base` température de base (seuil de non-chauffage/non-refroidissement, en K
//! ou °C — seule la différence importe, prendre les deux dans la même échelle) ;
//! `T_moy` température moyenne journalière (même échelle) ; `HDD`/`CDD` degrés-jours
//! d'une journée (K·jour) ; `U` coefficient de déperdition du bâtiment (W/K) ;
//! `86400 = 24·3600` nombre de secondes par jour (s/jour) ; `E` énergie sur la
//! période (J) ; `DJ_an` cumul de degrés-jours (K·jour). Un degré-jour vaut 1 K
//! maintenu pendant 1 jour.
//!
//! **Convention** : SI (températures et écarts en K, coefficient de déperdition en
//! W/K, degrés-jours en K·jour, énergie en J). Les températures peuvent être
//! fournies en °C puisque seule leur différence intervient.
//! **Limite honnête** : la méthode des degrés-jours est un **modèle linéaire
//! quasi-statique** de tendance ; elle **ignore** les apports solaires et internes,
//! l'inertie thermique et la dynamique du bâtiment. La température de base et le
//! coefficient de déperdition `U` (W/K) sont **fournis par l'appelant** ; aucune
//! valeur matériau, climatique ou de procédé « par défaut » n'est inventée. Ce
//! n'est **pas** un calcul de charge instantanée.

/// Nombre de secondes dans une journée (s/jour) : `24·3600`.
const SECONDS_PER_DAY: f64 = 24.0 * 3600.0;

/// Degrés-jours de chauffage d'une journée `HDD = max(0, T_base − T_moy)` (K·jour).
///
/// Panique si l'une des températures n'est pas finie (`NaN`/infinie).
pub fn degreeday_heating(base_temperature: f64, mean_daily_temperature: f64) -> f64 {
    assert!(
        base_temperature.is_finite(),
        "la température de base doit être finie"
    );
    assert!(
        mean_daily_temperature.is_finite(),
        "la température moyenne journalière doit être finie"
    );
    (base_temperature - mean_daily_temperature).max(0.0)
}

/// Degrés-jours de refroidissement d'une journée `CDD = max(0, T_moy − T_base)`
/// (K·jour).
///
/// Panique si l'une des températures n'est pas finie (`NaN`/infinie).
pub fn degreeday_cooling(base_temperature: f64, mean_daily_temperature: f64) -> f64 {
    assert!(
        base_temperature.is_finite(),
        "la température de base doit être finie"
    );
    assert!(
        mean_daily_temperature.is_finite(),
        "la température moyenne journalière doit être finie"
    );
    (mean_daily_temperature - base_temperature).max(0.0)
}

/// Énergie déperdie sur la période `E = 86400·U·DJ` (J), avec `U` le coefficient
/// de déperdition (W/K) et `DJ` le cumul de degrés-jours (K·jour).
///
/// Panique si `degree_days < 0` ou si `building_loss_coefficient < 0`.
pub fn degreeday_energy_demand(degree_days: f64, building_loss_coefficient: f64) -> f64 {
    assert!(
        degree_days >= 0.0,
        "le cumul de degrés-jours doit être positif"
    );
    assert!(
        building_loss_coefficient >= 0.0,
        "le coefficient de déperdition du bâtiment doit être positif"
    );
    SECONDS_PER_DAY * building_loss_coefficient * degree_days
}

/// Cumul annuel (ou sur une période) de degrés-jours `DJ_an = Σ DJ_i` (K·jour).
///
/// Panique si la tranche est vide ou si l'un de ses éléments est négatif ou non fini.
pub fn degreeday_annual_from_daily(daily_degree_days: &[f64]) -> f64 {
    assert!(
        !daily_degree_days.is_empty(),
        "la série de degrés-jours journaliers ne doit pas être vide"
    );
    assert!(
        daily_degree_days.iter().all(|&d| d.is_finite() && d >= 0.0),
        "chaque degré-jour journalier doit être fini et positif"
    );
    daily_degree_days.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn heating_and_cooling_are_complementary_and_mutually_exclusive() {
        // Pour toute journée, HDD − CDD = T_base − T_moy et au moins l'un est nul.
        let (base, mean) = (18.0_f64, 5.0_f64);
        let hdd = degreeday_heating(base, mean);
        let cdd = degreeday_cooling(base, mean);
        assert_relative_eq!(hdd - cdd, base - mean, epsilon = 1e-12);
        assert_relative_eq!(hdd * cdd, 0.0, epsilon = 1e-12);
        // Journée froide : chauffage actif (13 K·jour), refroidissement nul.
        assert_relative_eq!(hdd, 13.0, epsilon = 1e-12);
        assert_relative_eq!(cdd, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn cooling_activates_above_base() {
        // T_moy=30, T_base=24 → CDD = 6 K·jour, HDD = 0.
        let (base, mean) = (24.0_f64, 30.0_f64);
        assert_relative_eq!(degreeday_cooling(base, mean), 6.0, epsilon = 1e-12);
        assert_relative_eq!(degreeday_heating(base, mean), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn degree_days_vanish_at_base_temperature() {
        // À T_moy = T_base, chauffage et refroidissement sont tous deux nuls.
        let base = 18.0_f64;
        assert_relative_eq!(degreeday_heating(base, base), 0.0, epsilon = 1e-12);
        assert_relative_eq!(degreeday_cooling(base, base), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn energy_demand_realistic_case() {
        // U = 200 W/K, DJ = 15 K·jour.
        // E = 86400·200·15 = 86400·3000 = 259 200 000 J = 259.2 MJ.
        let energy = degreeday_energy_demand(15.0, 200.0);
        assert_relative_eq!(energy, 259_200_000.0, epsilon = 1e-3);
    }

    #[test]
    fn energy_demand_is_bilinear() {
        // E ∝ U et E ∝ DJ : doubler l'un double l'énergie.
        let base = degreeday_energy_demand(10.0, 150.0);
        assert_relative_eq!(
            degreeday_energy_demand(20.0, 150.0),
            2.0 * base,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            degreeday_energy_demand(10.0, 300.0),
            2.0 * base,
            epsilon = 1e-6
        );
    }

    #[test]
    fn annual_sum_matches_manual_total() {
        // Somme de 5 journées : 13 + 0 + 8.5 + 12 + 4.5 = 38 K·jour.
        let daily = [13.0_f64, 0.0, 8.5, 12.0, 4.5];
        let total = degreeday_annual_from_daily(&daily);
        assert_relative_eq!(total, 38.0, epsilon = 1e-12);
        // Cohérence avec le calcul d'énergie : E = 86400·U·38 pour U = 100 W/K.
        let energy = degreeday_energy_demand(total, 100.0);
        assert_relative_eq!(energy, 86400.0 * 100.0 * 38.0, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(expected = "ne doit pas être vide")]
    fn annual_rejects_empty_series() {
        degreeday_annual_from_daily(&[]);
    }
}
