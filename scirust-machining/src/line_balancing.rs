//! Équilibrage de ligne d'assemblage — nombre minimal théorique de postes,
//! efficacité d'équilibrage, retard d'équilibrage et indice de lissage à partir
//! de temps de tâche fournis.
//!
//! ```text
//! temps de cycle imposé      CT   = T_dispo / D
//! postes minimaux théoriques N_min = ⌈ ΣtI / CT ⌉
//! efficacité de la ligne      E    = ΣtI / (N · CT)
//! retard d'équilibrage        BD   = 1 − E
//! indice de lissage           SI   = √( Σ (t_max − t_k)² )
//! ```
//!
//! `T_dispo` temps de production disponible par période (s), `D` demande sur la
//! même période (unités), `CT` temps de cycle (s/unité), `ΣtI` temps de travail
//! total de toutes les tâches (s), `N_min` nombre minimal théorique de postes
//! (sans dimension, entier), `N` nombre réel de postes ouverts (sans dimension,
//! entier), `E` efficacité d'équilibrage (fraction sans dimension, `1` = ligne
//! parfaitement équilibrée), `BD` retard d'équilibrage (fraction sans
//! dimension), `t_k` charge de travail du poste `k` (s), `t_max` charge du poste
//! le plus chargé (s), `SI` indice de lissage (s, `0` = charges identiques).
//!
//! **Convention** : unités de temps cohérentes (mêmes secondes partout) et temps
//! de cycle au moins égal à la plus grande charge de poste. **Limite honnête** :
//! les tâches sont supposées indivisibles et les contraintes de préséance ne sont
//! PAS résolues ici — ce module ne fait pas l'affectation des tâches aux postes,
//! il fournit seulement les bornes et indicateurs à partir de temps de tâche et
//! de charges de postes FOURNIS par l'appelant ; aucune valeur « par défaut »
//! n'est inventée.

/// Temps de cycle imposé par la cadence `CT = T_dispo / D`.
///
/// Temps disponible pour traiter une unité afin de satisfaire la demande sur la
/// période considérée.
///
/// Panique si `available_time <= 0` ou `demand_units <= 0`.
pub fn line_cycle_time(available_time: f64, demand_units: f64) -> f64 {
    assert!(
        available_time > 0.0,
        "le temps de production disponible doit être strictement positif"
    );
    assert!(
        demand_units > 0.0,
        "la demande doit être strictement positive"
    );
    available_time / demand_units
}

/// Nombre minimal théorique de postes `N_min = ⌈ ΣtI / CT ⌉`.
///
/// Borne inférieure entière du nombre de postes : on ne peut pas ouvrir moins de
/// postes que le ratio arrondi au supérieur, même avec une affectation parfaite.
///
/// Panique si `total_task_time <= 0` ou `cycle_time <= 0`.
pub fn line_theoretical_min_stations(total_task_time: f64, cycle_time: f64) -> u32 {
    assert!(
        total_task_time > 0.0,
        "le temps de travail total doit être strictement positif"
    );
    assert!(
        cycle_time > 0.0,
        "le temps de cycle doit être strictement positif"
    );
    (total_task_time / cycle_time).ceil() as u32
}

/// Efficacité d'équilibrage `E = ΣtI / (N · CT)`.
///
/// Fraction du temps de main-d'œuvre effectivement utilisée : le dénominateur
/// `N · CT` est le temps total offert par les `N` postes sur un cycle.
///
/// Panique si `total_task_time <= 0`, `actual_stations == 0`, `cycle_time <= 0`
/// ou si la charge dépasse la capacité offerte (`E > 1`, cadence irréalisable).
pub fn line_efficiency(total_task_time: f64, actual_stations: u32, cycle_time: f64) -> f64 {
    assert!(
        total_task_time > 0.0,
        "le temps de travail total doit être strictement positif"
    );
    assert!(
        actual_stations > 0,
        "le nombre de postes doit être strictement positif"
    );
    assert!(
        cycle_time > 0.0,
        "le temps de cycle doit être strictement positif"
    );
    let efficiency = total_task_time / (f64::from(actual_stations) * cycle_time);
    assert!(
        efficiency <= 1.0,
        "l'efficacité ne peut pas dépasser 1 : la charge excède la capacité offerte"
    );
    efficiency
}

/// Retard d'équilibrage `BD = 1 − E`.
///
/// Part du temps de main-d'œuvre perdue en attente (temps mort des postes) ;
/// complément de l'efficacité d'équilibrage.
///
/// Panique si `efficiency` n'est pas dans l'intervalle `]0, 1]`.
pub fn balance_delay(efficiency: f64) -> f64 {
    assert!(
        efficiency > 0.0 && efficiency <= 1.0,
        "l'efficacité doit appartenir à l'intervalle ]0, 1]"
    );
    1.0 - efficiency
}

/// Indice de lissage `SI = √( Σ (t_max − t_k)² )`.
///
/// Écart quadratique des charges de poste par rapport au poste le plus chargé ;
/// vaut `0` lorsque toutes les charges sont identiques (ligne parfaitement
/// lissée) et croît avec le déséquilibre.
///
/// Panique si `station_times` est vide ou contient une charge négative.
pub fn balance_smoothness_index(station_times: &[f64]) -> f64 {
    assert!(
        !station_times.is_empty(),
        "la liste des charges de poste ne doit pas être vide"
    );
    assert!(
        station_times.iter().all(|&t| t >= 0.0),
        "chaque charge de poste doit être positive ou nulle"
    );
    let t_max = station_times
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    station_times
        .iter()
        .map(|&t| (t_max - t).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn realistic_balancing_case() {
        // Ligne : ΣtI = 48 s de travail, CT = 10 s/unité.
        // N_min = ⌈48/10⌉ = ⌈4,8⌉ = 5 postes.
        // Avec N = 5 : E = 48/(5·10) = 0,96 ; BD = 0,04.
        let total = 48.0;
        let ct = 10.0;
        assert_eq!(line_theoretical_min_stations(total, ct), 5);
        let eff = line_efficiency(total, 5, ct);
        assert_relative_eq!(eff, 0.96, epsilon = 1e-9);
        assert_relative_eq!(balance_delay(eff), 0.04, epsilon = 1e-9);
    }

    #[test]
    fn efficiency_and_delay_sum_to_one() {
        // Identité : E + BD = 1 par construction.
        let eff = line_efficiency(72.0, 8, 10.0);
        assert_relative_eq!(eff + balance_delay(eff), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn cycle_time_matches_demand() {
        // Réciprocité cadence : T_dispo = 480 s, D = 96 unités ⇒ CT = 5 s ;
        // traiter D unités au rythme CT consomme exactement T_dispo.
        let available = 480.0;
        let demand = 96.0;
        let ct = line_cycle_time(available, demand);
        assert_relative_eq!(ct, 5.0, epsilon = 1e-12);
        assert_relative_eq!(ct * demand, available, epsilon = 1e-9);
    }

    #[test]
    fn perfect_balance_gives_unit_efficiency_and_zero_smoothness() {
        // Cas limite : 4 postes chargés à 10 s pour CT = 10 s.
        // ΣtI = 40 s ⇒ E = 40/(4·10) = 1 ; BD = 0 ; SI = 0.
        let station_times = [10.0, 10.0, 10.0, 10.0];
        let total: f64 = station_times.iter().sum();
        let eff = line_efficiency(total, station_times.len() as u32, 10.0);
        assert_relative_eq!(eff, 1.0, epsilon = 1e-12);
        assert_relative_eq!(balance_delay(eff), 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            balance_smoothness_index(&station_times),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn efficiency_inversely_proportional_to_station_count() {
        // E ∝ 1/N à charge et CT constants : doubler les postes divise E par deux.
        let total = 60.0;
        let ct = 12.0;
        let e1 = line_efficiency(total, 5, ct);
        let e2 = line_efficiency(total, 10, ct);
        assert_relative_eq!(e2, e1 / 2.0, epsilon = 1e-12);
    }

    #[test]
    fn smoothness_index_known_value() {
        // Charges 8, 6, 5 s : t_max = 8 ; SI = √(0² + 2² + 3²) = √13.
        let si = balance_smoothness_index(&[8.0, 6.0, 5.0]);
        assert_relative_eq!(si, 13.0_f64.sqrt(), epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "l'efficacité ne peut pas dépasser 1")]
    fn overloaded_line_panics() {
        // Charge 120 s pour une capacité offerte de 2·50 = 100 s : irréalisable.
        line_efficiency(120.0, 2, 50.0);
    }
}
