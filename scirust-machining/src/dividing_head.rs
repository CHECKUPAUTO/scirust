//! Appareil **diviseur** de fraiseuse — **division simple** : tours de manivelle
//! par division, nombre de trous à parcourir sur un cercle, et division angulaire.
//!
//! ```text
//! division simple      n = R / N            (tours de manivelle par division)
//! trous sur un cercle  h = f · C            (f = partie fractionnaire de n, C trous)
//! division angulaire   n_θ = R · θ / 360    (tours de manivelle pour θ degrés)
//! ```
//!
//! `R` rapport de la vis sans fin de l'appareil (sans dimension, 40:1 usuel donc
//! `R = 40`), `N` nombre de divisions voulues, `f` partie fractionnaire du nombre
//! de tours, `C` nombre de trous du cercle choisi, `θ` angle en **degrés**,
//! `n`/`n_θ` en tours de manivelle, `h` en trous.
//!
//! **Convention** : angles en degrés (usage atelier des appareils diviseurs) ;
//! `R`, `N`, `C` sans dimension. **Limite honnête** : traite uniquement la
//! division **SIMPLE**, le rapport `R` de la vis sans fin étant **fourni** par
//! l'appelant (aucune valeur « par défaut » comme 40:1 n'est supposée). Les
//! divisions **différentielle** et **composée** (roues de train, décalage du
//! plateau) sont à la charge de l'appelant.

/// Division simple : nombre de tours de manivelle par division `n = R/N`.
///
/// `R` rapport de la vis sans fin (40 usuel), `N` nombre de divisions.
///
/// Panique si `worm_ratio <= 0` ou si `divisions == 0`.
pub fn dividing_simple_turns(worm_ratio: f64, divisions: u32) -> f64 {
    assert!(
        worm_ratio > 0.0,
        "le rapport de la vis sans fin doit être strictement positif"
    );
    assert!(divisions > 0, "le nombre de divisions doit être au moins 1");
    worm_ratio / divisions as f64
}

/// Nombre de trous à parcourir sur le cercle choisi `h = f·C`.
///
/// `fractional_turns` = partie fractionnaire du nombre de tours de manivelle,
/// `hole_circle` = nombre de trous `C` du cercle utilisé.
///
/// Panique si `fractional_turns < 0` ou si `hole_circle == 0`.
pub fn dividing_hole_count(fractional_turns: f64, hole_circle: u32) -> f64 {
    assert!(
        fractional_turns >= 0.0,
        "la partie fractionnaire de tours doit être positive ou nulle"
    );
    assert!(hole_circle > 0, "le cercle doit comporter au moins un trou");
    fractional_turns * hole_circle as f64
}

/// Division angulaire : tours de manivelle pour un angle `θ` en degrés
/// `n_θ = R·θ/360`.
///
/// `R` rapport de la vis sans fin, `degrees` angle `θ` en degrés.
///
/// Panique si `worm_ratio <= 0`.
pub fn dividing_angular_turns(worm_ratio: f64, degrees: f64) -> f64 {
    assert!(
        worm_ratio > 0.0,
        "le rapport de la vis sans fin doit être strictement positif"
    );
    worm_ratio * degrees / 360.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn simple_turns_times_divisions_recovers_ratio() {
        // Identité de réciprocité : n·N = R quelle que soit la division.
        for n in [2u32, 17, 40, 127]
        {
            let turns = dividing_simple_turns(40.0, n);
            assert_relative_eq!(turns * n as f64, 40.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn simple_turns_realistic_case() {
        // Appareil 40:1, N = 17 divisions → 40/17 = 2 + 6/17 tours.
        let turns = dividing_simple_turns(40.0, 17);
        assert_relative_eq!(turns, 40.0 / 17.0, epsilon = 1e-12);
        let frac = turns - turns.floor();
        assert_relative_eq!(frac, 6.0 / 17.0, epsilon = 1e-12);
        // Sur un cercle de 17 trous, la partie 6/17 correspond à exactement 6 trous.
        assert_relative_eq!(dividing_hole_count(frac, 17), 6.0, epsilon = 1e-12);
    }

    #[test]
    fn hole_count_is_linear_in_the_circle() {
        // h = f·C : doubler le nombre de trous double h à f constant.
        let f = 0.25;
        assert_relative_eq!(dividing_hole_count(f, 20), 5.0, epsilon = 1e-12);
        assert_relative_eq!(dividing_hole_count(f, 40), 10.0, epsilon = 1e-12);
    }

    #[test]
    fn angular_full_turn_recovers_worm_ratio() {
        // 360° correspond à une division complète → R tours de manivelle.
        assert_relative_eq!(dividing_angular_turns(40.0, 360.0), 40.0, epsilon = 1e-12);
        // Linéarité : la moitié de l'angle donne la moitié des tours.
        assert_relative_eq!(dividing_angular_turns(40.0, 180.0), 20.0, epsilon = 1e-12);
    }

    #[test]
    fn angular_equals_simple_division_at_matching_angle() {
        // L'angle d'une division vaut 360/N ; la division angulaire doit alors
        // redonner exactement la division simple : R·(360/N)/360 = R/N.
        let n = 8u32;
        let degrees_per_division = 360.0 / n as f64;
        assert_relative_eq!(
            dividing_angular_turns(40.0, degrees_per_division),
            dividing_simple_turns(40.0, n),
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "au moins 1")]
    fn zero_divisions_panics() {
        dividing_simple_turns(40.0, 0);
    }
}
