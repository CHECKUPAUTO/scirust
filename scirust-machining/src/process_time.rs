//! Temps de gamme (productique) — temps d'une opération série, temps alloué par
//! pièce et cadence de production.
//!
//! ```text
//! temps de série     T_série = T_préparation + N·T_unitaire
//! temps par pièce    T_pièce = T_préparation/N + T_unitaire
//! cadence (pièces/h) = 60 / T_pièce           (T_pièce en min)
//! nb de postes       = ⌈ temps requis / temps disponible ⌉
//! ```
//!
//! `T_préparation` temps de réglage/montage amorti sur le lot (min),
//! `T_unitaire` temps opératoire par pièce (min), `N` taille du lot,
//! `T_pièce` temps alloué par pièce, cadence en pièces par heure.
//!
//! **Convention** : temps en **minutes**. **Limite honnête** : bilan de temps
//! **déterministe** d'une opération (préparation amortie + temps unitaire) ; ne
//! modélise ni les aléas, ni les temps d'attente entre postes, ni l'équilibrage
//! de ligne. Les temps élémentaires sont fournis par l'appelant (chronométrage,
//! MTM, gamme).

/// Temps de série `T_série = T_préparation + N·T_unitaire` (min).
///
/// Panique si `batch_size == 0`.
pub fn batch_time(setup_time_min: f64, batch_size: u32, unit_time_min: f64) -> f64 {
    assert!(batch_size > 0, "la taille du lot doit être au moins 1");
    setup_time_min + batch_size as f64 * unit_time_min
}

/// Temps alloué **par pièce** `T_pièce = T_préparation/N + T_unitaire` (min).
///
/// Panique si `batch_size == 0`.
pub fn time_per_piece(setup_time_min: f64, batch_size: u32, unit_time_min: f64) -> f64 {
    assert!(batch_size > 0, "la taille du lot doit être au moins 1");
    setup_time_min / batch_size as f64 + unit_time_min
}

/// Cadence de production `= 60/T_pièce` (pièces/heure).
///
/// Panique si `time_per_piece_min <= 0`.
pub fn throughput_per_hour(time_per_piece_min: f64) -> f64 {
    assert!(
        time_per_piece_min > 0.0,
        "le temps par pièce doit être strictement positif"
    );
    60.0 / time_per_piece_min
}

/// Nombre de postes nécessaires `= ⌈temps requis / temps disponible⌉`.
///
/// Panique si `available_time <= 0`.
pub fn stations_required(required_time: f64, available_time_per_station: f64) -> u32 {
    assert!(
        available_time_per_station > 0.0,
        "le temps disponible par poste doit être strictement positif"
    );
    (required_time / available_time_per_station).ceil() as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn batch_time_amortises_setup() {
        // Réglage 30 min, 100 pièces, 2 min/pièce → 30 + 200 = 230 min.
        assert_relative_eq!(batch_time(30.0, 100, 2.0), 230.0, epsilon = 1e-9);
    }

    #[test]
    fn per_piece_time_decreases_with_batch_size() {
        // Réglage amorti : plus le lot est grand, plus T_pièce baisse.
        let small = time_per_piece(30.0, 10, 2.0); // 3 + 2 = 5 min
        let large = time_per_piece(30.0, 100, 2.0); // 0,3 + 2 = 2,3 min
        assert_relative_eq!(small, 5.0, epsilon = 1e-9);
        assert_relative_eq!(large, 2.3, epsilon = 1e-9);
        assert!(large < small);
    }

    #[test]
    fn throughput_from_cycle() {
        // T_pièce = 2,3 min → 60/2,3 ≈ 26,1 pièces/h.
        assert_relative_eq!(throughput_per_hour(2.3), 60.0 / 2.3, epsilon = 1e-9);
    }

    #[test]
    fn stations_round_up() {
        // 250 min requis, 480 min/poste dispo → 1 poste ; 700 min → 2 postes.
        assert_eq!(stations_required(250.0, 480.0), 1);
        assert_eq!(stations_required(700.0, 480.0), 2);
    }

    #[test]
    #[should_panic(expected = "taille du lot")]
    fn zero_batch_panics() {
        time_per_piece(30.0, 0, 2.0);
    }
}
