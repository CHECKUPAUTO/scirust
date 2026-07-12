//! Fonderie — **système de remplissage** (gating) : vitesse en sortie de descente
//! (**Torricelli**), section d'étranglement, temps de coulée et profil de descente
//! anti-aspiration.
//!
//! ```text
//! vitesse de descente v = √(2·g·h)              (Torricelli)
//! temps de coulée     t = V/(A·v)               (A = section d'étranglement)
//! section requise     A = V/(v·t)
//! descente conique    A_haut/A_bas = √(h_bas/h_haut)   (évite l'aspiration)
//! ```
//!
//! `g` pesanteur (m/s²), `h` hauteur métallostatique effective (m), `v` vitesse
//! (m/s), `V` volume de métal (m³), `A` section d'étranglement (m²), `t` temps de
//! coulée (s). Le rétrécissement conique de la descente maintient le métal en
//! charge et évite l'aspiration d'air/gaz.
//!
//! **Convention** : SI cohérent. **Limite honnête** : hydraulique idéale
//! (Torricelli, pertes négligées) ; le rapport de gating (descente:chenal:attaques)
//! et le régime d'écoulement (turbulence) sont à choisir par l'appelant. Ne
//! modélise ni les pertes de charge, ni l'oxydation.

/// Vitesse en bas de descente `v = √(2·g·h)` (m/s).
///
/// Panique si `g·h < 0`.
pub fn sprue_exit_velocity(g: f64, effective_head: f64) -> f64 {
    assert!(g * effective_head >= 0.0, "g·h doit être positif");
    (2.0 * g * effective_head).sqrt()
}

/// Temps de coulée `t = V/(A·v)` (s).
///
/// Panique si `choke_area·velocity <= 0`.
pub fn pouring_time(volume: f64, choke_area: f64, velocity: f64) -> f64 {
    assert!(
        choke_area * velocity > 0.0,
        "A·v doit être strictement positif"
    );
    volume / (choke_area * velocity)
}

/// Section d'étranglement requise pour un temps donné `A = V/(v·t)` (m²).
///
/// Panique si `velocity·time <= 0`.
pub fn choke_area(volume: f64, velocity: f64, pouring_time: f64) -> f64 {
    assert!(
        velocity * pouring_time > 0.0,
        "v·t doit être strictement positif"
    );
    volume / (velocity * pouring_time)
}

/// Rapport de sections d'une descente conique anti-aspiration
/// `A_haut/A_bas = √(h_bas/h_haut)`.
///
/// Panique si `top_head <= 0`.
pub fn sprue_taper_ratio(top_head: f64, bottom_head: f64) -> f64 {
    assert!(
        top_head > 0.0,
        "la charge en haut de descente doit être strictement positive"
    );
    (bottom_head / top_head).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn torricelli_velocity() {
        // h=0,2 m, g=9,81 → v = √(2·9,81·0,2) ≈ 1,98 m/s.
        assert_relative_eq!(
            sprue_exit_velocity(9.81, 0.2),
            (2.0f64 * 9.81 * 0.2).sqrt(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn pouring_time_and_choke_area_are_inverse() {
        // t = V/(A·v) et A = V/(v·t) doivent se composer.
        let (v, vel) = (1e-3, 2.0);
        let t = pouring_time(v, 5e-4, vel);
        assert_relative_eq!(choke_area(v, vel, t), 5e-4, max_relative = 1e-9);
    }

    #[test]
    fn larger_choke_fills_faster() {
        // Une section d'étranglement plus grande réduit le temps de coulée.
        assert!(pouring_time(1e-3, 1e-3, 2.0) < pouring_time(1e-3, 5e-4, 2.0));
    }

    #[test]
    fn tapered_sprue_narrows_downward() {
        // La descente se rétrécit vers le bas (charge plus grande) : A_haut/A_bas > 1.
        // h_bas > h_haut → ratio > 1 (section du haut plus grande).
        assert!(sprue_taper_ratio(0.05, 0.20) > 1.0);
        assert_relative_eq!(
            sprue_taper_ratio(0.05, 0.20),
            (0.20f64 / 0.05).sqrt(),
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "A·v")]
    fn zero_choke_pouring_time_panics() {
        pouring_time(1e-3, 0.0, 2.0);
    }
}
