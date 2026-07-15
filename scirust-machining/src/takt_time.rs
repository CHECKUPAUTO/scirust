//! Temps takt (Lean) — rythme de production imposé par la demande client :
//! temps takt, cadence de sortie requise et nombre théorique d'opérateurs.
//!
//! ```text
//! temps takt              TT = T / D
//! cadence de sortie       R  = D / T   (= 1 / TT)
//! nombre d'opérateurs     N  = ⌈ Wc / TT ⌉
//! ```
//!
//! `T` temps de production net disponible sur la période (s, hors pauses et
//! arrêts planifiés), `D` demande client sur la même période (unités), `TT`
//! temps takt (s/unité, cadence à laquelle une pièce doit sortir pour suivre la
//! demande), `R` cadence de sortie requise (unités/s), `Wc` contenu de travail
//! total d'une pièce (s/unité, somme des temps manuels de tous les postes),
//! `N` nombre théorique d'opérateurs (arrondi supérieur, sans dimension).
//!
//! **Convention** : unités de temps cohérentes (mêmes secondes partout) ; le
//! temps `T` est déjà NET (pauses, changements de série et arrêts planifiés
//! déduits). **Limite honnête** : le temps disponible net `T`, la demande `D`
//! et le contenu de travail `Wc` sont FOURNIS par l'appelant ; aucune valeur
//! « par défaut » (durée d'équipe, taux de rebut, temps de pause) n'est
//! inventée. Le nombre d'opérateurs suppose une ligne idéalement équilibrée à
//! rendement unitaire (ni pertes d'équilibrage, ni temps morts inter-postes).

/// Temps takt `TT = T / D`.
///
/// Intervalle entre deux sorties de pièce qui synchronise exactement la
/// production sur la demande client.
///
/// Panique si `available_time <= 0` ou `customer_demand <= 0`.
pub fn takt_time(available_time: f64, customer_demand: f64) -> f64 {
    assert!(
        available_time > 0.0,
        "le temps de production disponible net doit être strictement positif"
    );
    assert!(
        customer_demand > 0.0,
        "la demande client doit être strictement positive"
    );
    available_time / customer_demand
}

/// Cadence de sortie requise `R = D / T`.
///
/// Nombre de pièces à produire par unité de temps pour couvrir la demande ;
/// c'est l'inverse du temps takt (`R = 1 / TT`).
///
/// Panique si `customer_demand <= 0` ou `available_time <= 0`.
pub fn takt_required_output_rate(customer_demand: f64, available_time: f64) -> f64 {
    assert!(
        customer_demand > 0.0,
        "la demande client doit être strictement positive"
    );
    assert!(
        available_time > 0.0,
        "le temps de production disponible net doit être strictement positif"
    );
    customer_demand / available_time
}

/// Nombre théorique d'opérateurs `N = ⌈ Wc / TT ⌉`.
///
/// Contenu de travail total d'une pièce divisé par le temps takt, arrondi à
/// l'entier supérieur : effectif minimal d'une ligne idéalement équilibrée qui
/// tient la cadence.
///
/// Panique si `total_work_content < 0` ou `takt_time <= 0`.
pub fn lean_number_of_operators(total_work_content: f64, takt_time: f64) -> u32 {
    assert!(
        total_work_content >= 0.0,
        "le contenu de travail total doit être positif ou nul"
    );
    assert!(
        takt_time > 0.0,
        "le temps takt doit être strictement positif"
    );
    (total_work_content / takt_time).ceil() as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn realistic_takt_case() {
        // Équipe nette T = 27000 s (7 h 30 hors pauses), demande D = 450 unités.
        // TT = 27000/450 = 60 s/unité ; R = 450/27000 = 1/60 unité/s.
        let t = 27_000.0_f64;
        let d = 450.0_f64;
        assert_relative_eq!(takt_time(t, d), 60.0, epsilon = 1e-9);
        assert_relative_eq!(takt_required_output_rate(d, t), 1.0 / 60.0, epsilon = 1e-12);
    }

    #[test]
    fn takt_and_rate_are_reciprocal() {
        // Identité : TT · R = 1 quels que soient T et D.
        let (t, d) = (14_400.0_f64, 320.0_f64);
        let tt = takt_time(t, d);
        let r = takt_required_output_rate(d, t);
        assert_relative_eq!(tt * r, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn takt_inversely_proportional_to_demand() {
        // TT ∝ 1/D : doubler la demande à T fixe divise le temps takt par deux.
        let t = 28_800.0_f64;
        let tt1 = takt_time(t, 400.0);
        let tt2 = takt_time(t, 800.0);
        assert_relative_eq!(tt2, tt1 / 2.0, epsilon = 1e-9);
    }

    #[test]
    fn operators_round_up_from_work_content() {
        // TT = 60 s. Wc = 250 s ⇒ 250/60 = 4.16… ⇒ 5 opérateurs.
        assert_eq!(lean_number_of_operators(250.0, 60.0), 5);
        // Contenu exactement multiple : Wc = 240 s ⇒ 4 postes pleins, pas d'arrondi.
        assert_eq!(lean_number_of_operators(240.0, 60.0), 4);
    }

    #[test]
    fn operators_balanced_line_matches_work_content() {
        // Ligne parfaitement équilibrée : Wc = N · TT retombe sur N pile.
        let tt = takt_time(27_000.0, 450.0); // 60 s
        let wc = 6.0 * tt; // six postes pleins
        assert_eq!(lean_number_of_operators(wc, tt), 6);
    }

    #[test]
    fn zero_work_content_needs_no_operator() {
        // Cas limite : aucun travail manuel ⇒ 0 opérateur.
        assert_eq!(lean_number_of_operators(0.0, 60.0), 0);
    }

    #[test]
    #[should_panic(expected = "la demande client doit être strictement positive")]
    fn zero_demand_panics() {
        takt_time(27_000.0, 0.0);
    }
}
