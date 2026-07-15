//! Ordonnancement par la méthode du chemin critique (CPM) : calcul aval/amont
//! d'un maillon et marges (totale, libre) d'une activité déterministe.
//!
//! ```text
//! date de fin au plus tôt   EF   = ES + d
//! date de début au plus tard LS  = LF - d
//! marge totale              TF   = LS - ES        (= LF - EF)
//! marge libre               FF   = ES_succ - EF
//! criticité                 crit ⇔ |TF| ≤ tol     (marge totale nulle)
//! ```
//!
//! `ES` date de début au plus tôt, `EF` date de fin au plus tôt, `LS` date de
//! début au plus tard, `LF` date de fin au plus tard (mêmes unités de temps,
//! p. ex. jour, h ou min) ; `d` durée de l'activité (même unité) ; `ES_succ`
//! date de début au plus tôt du successeur (même unité) ; `TF` marge totale,
//! `FF` marge libre (même unité) ; `tol` tolérance de criticité (même unité).
//!
//! **Convention** : durées déterministes, aucune préemption ; sur le chemin
//! critique la marge totale est nulle.
//! **Limite honnête** : chaque appel traite UN maillon (une passe aval ou
//! amont), les durées d'activité et les dates fournies le sont par l'appelant,
//! qui enchaîne lui-même le réseau (propagation aval puis amont). Aucune
//! contrainte de ressources n'est modélisée et aucune durée « par défaut »
//! n'est inventée ici.
//!
//! Distinct de [`crate::johnson_scheduling`] (flow-shop à deux machines).

/// Tolérance par défaut (en unité de temps) pour juger une marge totale nulle
/// dans [`cpm_is_critical`] : une activité est critique si `|TF| ≤` cette valeur.
pub const CPM_CRITICALITY_TOLERANCE: f64 = 1e-9;

/// Date de fin au plus tôt d'un maillon `EF = ES + d` (passe aval).
///
/// Panique si `early_start` ou `duration` est négatif ou non fini.
pub fn cpm_early_finish(early_start: f64, duration: f64) -> f64 {
    assert!(
        early_start.is_finite() && early_start >= 0.0,
        "la date de début au plus tôt doit être finie et positive ou nulle"
    );
    assert!(
        duration.is_finite() && duration >= 0.0,
        "la durée d'activité doit être finie et positive ou nulle"
    );
    early_start + duration
}

/// Date de début au plus tard d'un maillon `LS = LF - d` (passe amont).
///
/// Panique si `late_finish` ou `duration` est négatif ou non fini, ou si la
/// durée dépasse `late_finish` (ce qui donnerait une date de début négative).
pub fn cpm_late_start(late_finish: f64, duration: f64) -> f64 {
    assert!(
        late_finish.is_finite() && late_finish >= 0.0,
        "la date de fin au plus tard doit être finie et positive ou nulle"
    );
    assert!(
        duration.is_finite() && duration >= 0.0,
        "la durée d'activité doit être finie et positive ou nulle"
    );
    assert!(
        duration <= late_finish,
        "la durée ne peut dépasser la date de fin au plus tard (date de début négative)"
    );
    late_finish - duration
}

/// Marge totale d'une activité `TF = LS - ES` (nulle sur le chemin critique).
///
/// Panique si `late_start` ou `early_start` est négatif ou non fini, ou si la
/// date de début au plus tard précède celle au plus tôt (marge négative).
pub fn cpm_total_float(late_start: f64, early_start: f64) -> f64 {
    assert!(
        late_start.is_finite() && late_start >= 0.0,
        "la date de début au plus tard doit être finie et positive ou nulle"
    );
    assert!(
        early_start.is_finite() && early_start >= 0.0,
        "la date de début au plus tôt doit être finie et positive ou nulle"
    );
    assert!(
        late_start >= early_start,
        "la date au plus tard ne peut précéder la date au plus tôt (marge négative)"
    );
    late_start - early_start
}

/// Marge libre d'une activité `FF = ES_succ - EF`.
///
/// Retard admissible sans décaler la date de début au plus tôt du successeur.
///
/// Panique si `successor_early_start` ou `early_finish` est négatif ou non
/// fini, ou si le successeur démarre avant la fin au plus tôt (marge négative).
pub fn cpm_free_float(successor_early_start: f64, early_finish: f64) -> f64 {
    assert!(
        successor_early_start.is_finite() && successor_early_start >= 0.0,
        "la date de début au plus tôt du successeur doit être finie et positive ou nulle"
    );
    assert!(
        early_finish.is_finite() && early_finish >= 0.0,
        "la date de fin au plus tôt doit être finie et positive ou nulle"
    );
    assert!(
        successor_early_start >= early_finish,
        "le successeur ne peut démarrer avant la fin au plus tôt (marge négative)"
    );
    successor_early_start - early_finish
}

/// Indique si une activité est critique, c.-à-d. de marge totale quasi nulle
/// `|TF| ≤ CPM_CRITICALITY_TOLERANCE`.
///
/// Panique si `total_float` est non fini.
pub fn cpm_is_critical(total_float: f64) -> bool {
    assert!(total_float.is_finite(), "la marge totale doit être finie");
    total_float.abs() <= CPM_CRITICALITY_TOLERANCE
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Réciprocité aval : EF - ES = d, quelle que soit la durée.
    #[test]
    fn early_finish_reciprocity() {
        let early_start = 5.0;
        let duration = 8.0;
        let ef = cpm_early_finish(early_start, duration);
        assert_relative_eq!(ef - early_start, duration, epsilon = 1e-12);
    }

    // Réciprocité amont : LF - LS = d (l'aval et l'amont partagent la durée).
    #[test]
    fn late_start_reciprocity() {
        let late_finish = 20.0;
        let duration = 8.0;
        let ls = cpm_late_start(late_finish, duration);
        assert_relative_eq!(late_finish - ls, duration, epsilon = 1e-12);
    }

    // Cas chiffré réaliste (jours). Maillon : ES=5, d=8 → EF=13 ; LF=20 → LS=12.
    // Marge totale TF = LS - ES = 12 - 5 = 7. Successeur au plus tôt à 15 :
    // marge libre FF = 15 - 13 = 2. Non critique.
    #[test]
    fn worked_case_floats() {
        let early_start = 5.0;
        let duration = 8.0;
        let late_finish = 20.0;
        let successor_early_start = 15.0;

        let ef = cpm_early_finish(early_start, duration);
        let ls = cpm_late_start(late_finish, duration);
        let tf = cpm_total_float(ls, early_start);
        let ff = cpm_free_float(successor_early_start, ef);

        assert_relative_eq!(ef, 13.0, epsilon = 1e-12);
        assert_relative_eq!(ls, 12.0, epsilon = 1e-12);
        assert_relative_eq!(tf, 7.0, epsilon = 1e-12);
        assert_relative_eq!(ff, 2.0, epsilon = 1e-12);
        assert!(!cpm_is_critical(tf));
    }

    // Identité de la marge totale : TF = LS - ES = LF - EF (mêmes durées).
    #[test]
    fn total_float_two_expressions() {
        let early_start = 3.0;
        let duration = 6.0;
        let late_finish = 11.0;
        let ef = cpm_early_finish(early_start, duration);
        let ls = cpm_late_start(late_finish, duration);
        let tf_from_starts = cpm_total_float(ls, early_start);
        let tf_from_finishes = late_finish - ef;
        assert_relative_eq!(tf_from_starts, tf_from_finishes, epsilon = 1e-12);
    }

    // Chemin critique : LS = ES et LF = EF ⇒ marge totale nulle ⇒ critique.
    #[test]
    fn critical_path_zero_float() {
        let early_start = 4.0;
        let duration = 7.0;
        let ef = cpm_early_finish(early_start, duration);
        // Sur le chemin critique, la date au plus tard de fin égale l'aval.
        let late_finish = ef;
        let ls = cpm_late_start(late_finish, duration);
        let tf = cpm_total_float(ls, early_start);
        assert_relative_eq!(ls, early_start, epsilon = 1e-12);
        assert_relative_eq!(tf, 0.0, epsilon = 1e-12);
        assert!(cpm_is_critical(tf));
    }

    #[test]
    #[should_panic(expected = "date de début négative")]
    fn late_start_rejects_duration_exceeding_finish() {
        let _ = cpm_late_start(5.0, 8.0);
    }
}
