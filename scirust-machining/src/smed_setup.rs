//! SMED (Single-Minute Exchange of Die) — réduction du temps de changement de
//! série : durée totale du changement, temps d'arrêt machine, gain apporté par
//! la conversion de tâches internes en tâches externes, et lot économique lié
//! au temps de réglage.
//!
//! ```text
//! temps de changement total      T_tot = t_int + t_ext
//! temps d'arrêt machine          D_stop = t_int
//! arrêt après conversion         D' = t_int - t_conv
//! taux de réduction d'arrêt      r = (D0 - D1) / D0
//! lot économique lié au réglage  Q* = sqrt(2 · D · (t_setup · c) / H)
//! ```
//!
//! `t_int` temps des tâches internes (s, réalisées machine à l'arrêt), `t_ext`
//! temps des tâches externes (s, réalisées machine en marche), `T_tot` durée
//! totale des opérations de changement (s), `D_stop` temps d'arrêt de la
//! machine (s, égal aux seules tâches internes dans le cas idéal), `t_conv`
//! part des tâches internes converties en externes (s), `D'` temps d'arrêt
//! après conversion (s), `D0`/`D1` temps d'arrêt avant/après amélioration (s),
//! `r` taux de réduction du temps d'arrêt (sans dimension, 0..1), `t_setup`
//! temps de réglage retenu comme base de coût (s), `c` taux de coût de la
//! ressource pendant le réglage (€/s), `D` demande sur la période (unités),
//! `H` coût de possession par unité et par période (€/(unité·période)), `Q*`
//! taille de lot économique (unités).
//!
//! **Convention** : unités de temps cohérentes (mêmes secondes partout) ; pour
//! le lot économique, `D` et `H` partagent la même base de temps et `t_setup`,
//! `c` se combinent en un coût de réglage `S = t_setup · c` (formule de
//! Wilson). **Limite honnête** : distinction interne/externe idéale — on
//! suppose que les tâches externes n'ajoutent aucun temps d'arrêt et que la
//! conversion n'a pas d'effet de bord. Les temps (`t_int`, `t_ext`, `t_conv`),
//! le taux de coût `c`, le coût de possession `H` et la demande `D` sont
//! FOURNIS par l'appelant ; aucune valeur « par défaut » n'est inventée.

/// Durée totale du changement de série `T_tot = t_int + t_ext`.
///
/// Somme des tâches internes (machine arrêtée) et externes (machine en marche) ;
/// c'est le travail total à réaliser, distinct du seul temps d'arrêt.
///
/// Panique si `internal_time < 0` ou `external_time < 0`.
pub fn smed_total_changeover_time(internal_time: f64, external_time: f64) -> f64 {
    assert!(
        internal_time >= 0.0,
        "le temps des tâches internes doit être positif ou nul"
    );
    assert!(
        external_time >= 0.0,
        "le temps des tâches externes doit être positif ou nul"
    );
    internal_time + external_time
}

/// Temps d'arrêt machine imputable au changement `D_stop = t_int`.
///
/// Dans le cas idéal du SMED, seules les tâches internes immobilisent la
/// machine : les tâches externes se font pendant que la machine produit encore
/// (ou déjà). Le temps d'arrêt est donc égal au temps des tâches internes.
///
/// Panique si `internal_time < 0`.
pub fn setup_downtime(internal_time: f64) -> f64 {
    assert!(
        internal_time >= 0.0,
        "le temps des tâches internes doit être positif ou nul"
    );
    internal_time
}

/// Temps d'arrêt après conversion de tâches internes en externes
/// `D' = t_int - t_conv`.
///
/// Cœur de la méthode SMED : convertir des opérations internes (machine à
/// l'arrêt) en opérations externes (machine en marche) réduit d'autant le
/// temps d'arrêt sans changer la nature des tâches.
///
/// Panique si `internal_time < 0`, `converted_to_external < 0` ou
/// `converted_to_external > internal_time`.
pub fn smed_downtime_after_conversion(internal_time: f64, converted_to_external: f64) -> f64 {
    assert!(
        internal_time >= 0.0,
        "le temps des tâches internes doit être positif ou nul"
    );
    assert!(
        converted_to_external >= 0.0,
        "le temps converti en externe doit être positif ou nul"
    );
    assert!(
        converted_to_external <= internal_time,
        "le temps converti ne peut excéder le temps des tâches internes"
    );
    internal_time - converted_to_external
}

/// Taux de réduction du temps d'arrêt `r = (D0 - D1) / D0`.
///
/// Fraction (sans dimension, entre 0 et 1) du temps d'arrêt initial `D0`
/// éliminée par l'amélioration qui l'amène à `D1`.
///
/// Panique si `original_downtime <= 0`, `reduced_downtime < 0` ou
/// `reduced_downtime > original_downtime`.
pub fn smed_downtime_reduction_ratio(original_downtime: f64, reduced_downtime: f64) -> f64 {
    assert!(
        original_downtime > 0.0,
        "le temps d'arrêt initial doit être strictement positif"
    );
    assert!(
        reduced_downtime >= 0.0,
        "le temps d'arrêt réduit doit être positif ou nul"
    );
    assert!(
        reduced_downtime <= original_downtime,
        "le temps d'arrêt réduit ne peut excéder le temps d'arrêt initial"
    );
    (original_downtime - reduced_downtime) / original_downtime
}

/// Taille de lot économique liée au temps de réglage
/// `Q* = sqrt(2 · D · (t_setup · c) / H)` (modèle de Wilson).
///
/// Le coût de réglage `S = t_setup · c` valorise le temps de réglage par un
/// taux de coût `c` : réduire ce temps (objectif du SMED) diminue le lot
/// économique et autorise des séries plus courtes et plus flexibles.
///
/// Panique si `setup_time < 0`, `setup_cost_rate < 0`,
/// `holding_cost_per_unit <= 0` ou `annual_demand < 0`.
pub fn setup_economic_batch_from_setup(
    setup_time: f64,
    setup_cost_rate: f64,
    holding_cost_per_unit: f64,
    annual_demand: f64,
) -> f64 {
    assert!(
        setup_time >= 0.0,
        "le temps de réglage doit être positif ou nul"
    );
    assert!(
        setup_cost_rate >= 0.0,
        "le taux de coût de réglage doit être positif ou nul"
    );
    assert!(
        holding_cost_per_unit > 0.0,
        "le coût de possession doit être strictement positif"
    );
    assert!(
        annual_demand >= 0.0,
        "la demande doit être positive ou nulle"
    );
    (2.0 * annual_demand * (setup_time * setup_cost_rate) / holding_cost_per_unit).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn total_changeover_is_sum_and_symmetric() {
        // T_tot = t_int + t_ext, et l'addition est commutative.
        let (int, ext) = (600.0_f64, 900.0_f64);
        assert_relative_eq!(smed_total_changeover_time(int, ext), 1500.0, epsilon = 1e-9);
        assert_relative_eq!(
            smed_total_changeover_time(int, ext),
            smed_total_changeover_time(ext, int),
            epsilon = 1e-12
        );
    }

    #[test]
    fn downtime_equals_internal_time() {
        // Cas idéal SMED : arrêt machine = tâches internes uniquement.
        assert_relative_eq!(setup_downtime(600.0), 600.0, epsilon = 1e-12);
        // Le temps externe n'apparaît pas dans l'arrêt : T_tot - t_ext = D_stop.
        let (int, ext) = (600.0_f64, 900.0_f64);
        assert_relative_eq!(
            smed_total_changeover_time(int, ext) - ext,
            setup_downtime(int),
            epsilon = 1e-9
        );
    }

    #[test]
    fn conversion_reduces_downtime_linearly() {
        // Convertir t_conv de l'interne vers l'externe retranche t_conv à l'arrêt.
        let int = 600.0_f64;
        assert_relative_eq!(
            smed_downtime_after_conversion(int, 240.0),
            360.0,
            epsilon = 1e-9
        );
        // Conversion nulle : l'arrêt reste égal aux tâches internes.
        assert_relative_eq!(
            smed_downtime_after_conversion(int, 0.0),
            setup_downtime(int),
            epsilon = 1e-12
        );
    }

    #[test]
    fn reduction_ratio_matches_conversion() {
        // Passer de 600 s à 360 s d'arrêt = 40 % de réduction.
        let d0 = 600.0_f64;
        let d1 = smed_downtime_after_conversion(d0, 240.0);
        assert_relative_eq!(smed_downtime_reduction_ratio(d0, d1), 0.4, epsilon = 1e-9);
        // Cas limites : aucun gain ⇒ 0, arrêt supprimé ⇒ 1.
        assert_relative_eq!(smed_downtime_reduction_ratio(d0, d0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(smed_downtime_reduction_ratio(d0, 0.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn economic_batch_realistic_case() {
        // D = 900 u/an, t_setup = 2 h = 7200 s, c = 0,05 €/s ⇒ S = 360 €,
        // H = 4 €/(u·an). Q* = sqrt(2·900·360/4) = sqrt(162000) = 402,49… u.
        let q = setup_economic_batch_from_setup(7200.0, 0.05, 4.0, 900.0);
        assert_relative_eq!(q, 162_000.0_f64.sqrt(), epsilon = 1e-9);
    }

    #[test]
    fn economic_batch_scales_as_sqrt_of_setup_time() {
        // Q* ∝ sqrt(t_setup) : diviser le temps de réglage par quatre
        // (gain SMED) divise le lot économique par deux.
        let q_long = setup_economic_batch_from_setup(7200.0, 0.05, 4.0, 900.0);
        let q_short = setup_economic_batch_from_setup(1800.0, 0.05, 4.0, 900.0);
        assert_relative_eq!(q_short, q_long / 2.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "le temps converti ne peut excéder")]
    fn over_conversion_panics() {
        smed_downtime_after_conversion(600.0, 700.0);
    }
}
