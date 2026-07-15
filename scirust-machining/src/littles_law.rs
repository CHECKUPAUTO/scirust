//! Loi de Little pour le flux d'atelier — relie en-cours, débit et temps de
//! passage d'un système de production en régime permanent.
//!
//! ```text
//! en-cours       WIP = TH · CT
//! temps de cycle CT  = WIP / TH
//! débit          TH  = WIP / CT
//! ```
//!
//! `WIP` en-cours moyen (Work In Process, nombre d'unités présentes dans le
//! système, sans dimension), `TH` débit moyen (throughput, unités par unité de
//! temps, p. ex. pièces/h), `CT` temps de passage moyen (cycle time, temps
//! qu'une unité passe dans le système, même unité de temps que celle du débit).
//!
//! **Convention** : unités de temps cohérentes entre `TH` et `CT` (si `TH` est en
//! pièces/h alors `CT` est en heures). **Limite honnête** : la loi de Little
//! n'est valable qu'en moyenne sur un système STABLE en régime permanent (entrées
//! = sorties en moyenne, en-cours borné) ; elle ne dit RIEN de la variabilité, de
//! la distribution des temps de passage, ni des régimes transitoires. Les valeurs
//! d'en-cours, de débit et de temps de passage sont FOURNIES par l'appelant à
//! partir de mesures ou d'objectifs — aucune valeur « par défaut » n'est inventée.

/// En-cours moyen `WIP = TH · CT` (loi de Little).
///
/// Nombre moyen d'unités présentes dans le système, produit du débit par le
/// temps de passage moyen.
///
/// Panique si `throughput <= 0` ou `cycle_time <= 0`.
pub fn little_wip(throughput: f64, cycle_time: f64) -> f64 {
    assert!(throughput > 0.0, "le débit doit être strictement positif");
    assert!(
        cycle_time > 0.0,
        "le temps de passage doit être strictement positif"
    );
    throughput * cycle_time
}

/// Temps de passage moyen `CT = WIP / TH` (loi de Little).
///
/// Durée moyenne qu'une unité passe dans le système, en-cours rapporté au débit.
///
/// Panique si `wip <= 0` ou `throughput <= 0`.
pub fn little_cycle_time(wip: f64, throughput: f64) -> f64 {
    assert!(wip > 0.0, "l'en-cours doit être strictement positif");
    assert!(throughput > 0.0, "le débit doit être strictement positif");
    wip / throughput
}

/// Débit moyen `TH = WIP / CT` (loi de Little).
///
/// Cadence moyenne de sortie du système, en-cours rapporté au temps de passage.
///
/// Panique si `wip <= 0` ou `cycle_time <= 0`.
pub fn little_throughput(wip: f64, cycle_time: f64) -> f64 {
    assert!(wip > 0.0, "l'en-cours doit être strictement positif");
    assert!(
        cycle_time > 0.0,
        "le temps de passage doit être strictement positif"
    );
    wip / cycle_time
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn realistic_workshop_case() {
        // Atelier : débit TH = 20 pièces/h, temps de passage CT = 2,5 h.
        // WIP = 20 · 2,5 = 50 pièces en-cours.
        let th = 20.0;
        let ct = 2.5;
        assert_relative_eq!(little_wip(th, ct), 50.0, epsilon = 1e-9);
    }

    #[test]
    fn wip_then_cycle_time_is_reciprocal() {
        // Réciprocité : partir de (TH, CT), calculer WIP, en re-déduire CT.
        let th = 12.0;
        let ct = 3.5;
        let wip = little_wip(th, ct);
        assert_relative_eq!(little_cycle_time(wip, th), ct, epsilon = 1e-12);
    }

    #[test]
    fn wip_then_throughput_is_reciprocal() {
        // Réciprocité : partir de (TH, CT), calculer WIP, en re-déduire TH.
        let th = 8.0;
        let ct = 4.0;
        let wip = little_wip(th, ct);
        assert_relative_eq!(little_throughput(wip, ct), th, epsilon = 1e-12);
    }

    #[test]
    fn three_relations_are_consistent() {
        // Identité croisée : TH · CT = WIP pour un triplet cohérent.
        let wip = 60.0;
        let th = 15.0;
        let ct = little_cycle_time(wip, th);
        assert_relative_eq!(little_throughput(wip, ct), th, epsilon = 1e-12);
        assert_relative_eq!(little_wip(th, ct), wip, epsilon = 1e-9);
    }

    #[test]
    fn wip_proportional_to_cycle_time() {
        // WIP ∝ CT à débit constant : doubler le temps de passage double l'en-cours.
        let th = 10.0;
        let w1 = little_wip(th, 2.0);
        let w2 = little_wip(th, 4.0);
        assert_relative_eq!(w2, 2.0 * w1, epsilon = 1e-12);
    }

    #[test]
    fn cycle_time_inversely_proportional_to_throughput() {
        // CT ∝ 1/TH à en-cours constant : doubler le débit divise le temps de
        // passage par deux.
        let wip = 100.0;
        let c1 = little_cycle_time(wip, 5.0);
        let c2 = little_cycle_time(wip, 10.0);
        assert_relative_eq!(c2, c1 / 2.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "le débit doit être strictement positif")]
    fn non_positive_throughput_panics() {
        little_wip(0.0, 2.5);
    }
}
