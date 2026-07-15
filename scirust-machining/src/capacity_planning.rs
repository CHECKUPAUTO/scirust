//! Planification de capacité d'un atelier : capacité disponible en heures-machine,
//! capacité requise par la demande, taux d'utilisation et cadence du goulot
//! d'étranglement.
//!
//! ```text
//! capacité disponible   C_disp = n · h · A
//! capacité requise      C_req  = D · CT
//! taux d'utilisation    U      = C_req / C_disp
//! cadence du goulot     r_min  = min_i r_i
//! ```
//!
//! `C_disp` (`capacity_available_capacity`) capacité disponible [heure-machine],
//! `n` nombre de machines (comptage, sans unité), `h` heures d'ouverture par
//! machine sur la période [h], `A` disponibilité effective de chaque machine
//! (fraction sans dimension dans `[0, 1]`, tenant compte des arrêts). `C_req`
//! (`capacity_required_capacity`) capacité requise [heure-machine], `D` demande à
//! produire sur la période (comptage d'unités, sans unité), `CT` temps de cycle
//! par unité [heure/unité]. `U` (`capacity_utilization`) taux d'utilisation
//! (fraction sans dimension ; `U > 1` signale une surcharge). `r_i` cadence du
//! poste `i` [unité/heure] et `r_min` (`capacity_bottleneck_rate`) cadence du
//! poste le plus lent, qui borne le débit de toute la ligne.
//!
//! **Limite honnête** : ce module n'effectue que l'arithmétique de capacité. Le
//! nombre de machines, les heures d'ouverture, les disponibilités/rendements `A`,
//! les temps de cycle `CT`, la demande `D` et les cadences des postes sont
//! **fournis** par l'appelant (relevés d'atelier, gammes, plan directeur) ; aucune
//! valeur « par défaut », constante de procédé ou hypothèse de matériau n'est
//! inventée ici. Le goulot est simplement défini comme le poste de cadence
//! minimale (postes supposés en série, sans stock tampon).

/// Capacité disponible `C_disp = n · h · A` en heures-machine sur la période :
/// heures d'ouverture cumulées corrigées par la disponibilité effective.
///
/// Panique si `machines`, `hours_per_machine` ou `availability` est négatif, ou si
/// `availability` dépasse `1`.
pub fn capacity_available_capacity(
    machines: f64,
    hours_per_machine: f64,
    availability: f64,
) -> f64 {
    assert!(
        machines >= 0.0,
        "le nombre de machines doit être positif ou nul"
    );
    assert!(
        hours_per_machine >= 0.0,
        "les heures par machine doivent être positives ou nulles"
    );
    assert!(
        (0.0..=1.0).contains(&availability),
        "la disponibilité doit être dans [0, 1]"
    );
    machines * hours_per_machine * availability
}

/// Capacité requise `C_req = D · CT` en heures-machine : temps total nécessaire
/// pour satisfaire la demande au temps de cycle donné.
///
/// Panique si `demand` ou `cycle_time_per_unit` est négatif.
pub fn capacity_required_capacity(demand: f64, cycle_time_per_unit: f64) -> f64 {
    assert!(demand >= 0.0, "la demande doit être positive ou nulle");
    assert!(
        cycle_time_per_unit >= 0.0,
        "le temps de cycle par unité doit être positif ou nul"
    );
    demand * cycle_time_per_unit
}

/// Taux d'utilisation `U = C_req / C_disp` : fraction de la capacité disponible
/// consommée par la charge (une valeur supérieure à `1` indique une surcharge).
///
/// Panique si `required` est négatif ou si `available` n'est pas strictement
/// positif.
pub fn capacity_utilization(required: f64, available: f64) -> f64 {
    assert!(
        required >= 0.0,
        "la capacité requise doit être positive ou nulle"
    );
    assert!(
        available > 0.0,
        "la capacité disponible doit être strictement positive"
    );
    required / available
}

/// Cadence du goulot `r_min = min_i r_i` [unité/heure] : le poste le plus lent
/// borne le débit de la ligne en série.
///
/// Panique si `station_rates` est vide ou si une cadence est négative.
pub fn capacity_bottleneck_rate(station_rates: &[f64]) -> f64 {
    assert!(
        !station_rates.is_empty(),
        "la ligne doit comporter au moins un poste"
    );
    assert!(
        station_rates.iter().all(|&r| r >= 0.0),
        "chaque cadence de poste doit être positive ou nulle"
    );
    station_rates.iter().copied().fold(f64::INFINITY, f64::min)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn available_capacity_realistic_case() {
        // 5 machines, 160 h/mois, disponibilité 0.85 → 680 heures-machine.
        assert_relative_eq!(
            capacity_available_capacity(5.0, 160.0, 0.85),
            680.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn availability_scales_capacity_linearly() {
        // C_disp est proportionnel à A : doubler A double la capacité.
        let base = capacity_available_capacity(3.0, 100.0, 0.4);
        let doubled = capacity_available_capacity(3.0, 100.0, 0.8);
        assert_relative_eq!(doubled, 2.0 * base, max_relative = 1e-12);
        // Disponibilité parfaite → n·h pur.
        assert_relative_eq!(
            capacity_available_capacity(3.0, 100.0, 1.0),
            300.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn utilization_matches_capacity_definitions() {
        // Cas cohérent : D·CT heures requises contre n·h·A disponibles.
        let required = capacity_required_capacity(400.0, 1.5); // 600 h
        let available = capacity_available_capacity(5.0, 160.0, 0.85); // 680 h
        assert_relative_eq!(required, 600.0, max_relative = 1e-12);
        assert_relative_eq!(
            capacity_utilization(required, available),
            600.0_f64 / 680.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn utilization_unity_when_load_equals_capacity() {
        // Charge = capacité → U = 1 exactement (cas limite d'équilibre).
        let cap = capacity_available_capacity(2.0, 50.0, 0.9); // 90 h
        assert_relative_eq!(capacity_utilization(cap, cap), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn bottleneck_is_the_slowest_station() {
        // r_min = min des cadences, indépendant de l'ordre des postes.
        let rates = [12.0, 8.0, 15.0, 10.0];
        assert_relative_eq!(capacity_bottleneck_rate(&rates), 8.0, epsilon = 1e-12);
        let reordered = [10.0, 15.0, 8.0, 12.0];
        assert_relative_eq!(
            capacity_bottleneck_rate(&reordered),
            capacity_bottleneck_rate(&rates),
            epsilon = 1e-12
        );
    }

    #[test]
    fn single_station_rate_is_its_own_bottleneck() {
        // Une ligne à un seul poste : le goulot est ce poste (cas limite).
        assert_relative_eq!(capacity_bottleneck_rate(&[7.5]), 7.5, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "au moins un poste")]
    fn empty_line_panics() {
        capacity_bottleneck_rate(&[]);
    }
}
