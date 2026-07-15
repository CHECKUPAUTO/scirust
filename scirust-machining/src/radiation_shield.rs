//! Écrans anti-rayonnement — réduction du flux radiatif entre deux grandes
//! plaques parallèles grises par insertion d'un ou plusieurs écrans réfléchissants.
//!
//! ```text
//! flux sans écran      q0 = σ·(T1⁴ − T2⁴) / (1/e1 + 1/e2 − 1)
//! N écrans (ε égales)  qN = q0 / (N + 1)
//! facteur de réduction f  = 1 / (N + 1)
//! un écran (ep, es)    q1 = σ·(T1⁴ − T2⁴) / [(1/ep + 1/es − 1) + (1/es + 1/ep − 1)]
//!                         = σ·(T1⁴ − T2⁴) / (2/ep + 2/es − 2)
//! ```
//!
//! `σ` constante de Stefan-Boltzmann (W/(m²·K⁴)), `e1`, `e2`, `ep` émissivités des
//! plaques et `es` émissivité de l'écran (sans dimension, `]0, 1]`), `T1`, `T2`
//! températures **absolues** des plaques (K), `q` flux surfacique net (W/m²), `N`
//! nombre d'écrans, `f` facteur de réduction (sans dimension). L'écran est supposé
//! avoir la même émissivité `es` sur ses deux faces ; ses deux résistances
//! radiatives (plaque 1 → écran, écran → plaque 2) s'ajoutent en série.
//!
//! **Limite honnête** : grandes plaques parallèles grises et diffuses (facteur de
//! forme unité), régime permanent, écran mince à température uniforme. Le facteur
//! `1/(N+1)` suppose que **toutes** les émissivités (plaques et écrans) sont égales.
//! Les émissivités, la constante `σ` et les températures sont **fournies** par
//! l'appelant : aucune valeur de matériau ou de procédé n'est supposée par défaut.
//! Complète [`crate::radiation_network`] pour les enceintes à N surfaces.

/// Facteur de réduction du flux radiatif pour `N` écrans de **même émissivité**
/// que les plaques `f = 1/(N + 1)` (sans dimension).
///
/// Panique si `number_of_shields < 0`.
pub fn radshield_reduction_factor_equal_emissivity(number_of_shields: f64) -> f64 {
    assert!(
        number_of_shields >= 0.0,
        "le nombre d'écrans doit être positif ou nul"
    );
    1.0 / (number_of_shields + 1.0)
}

/// Flux radiatif net en présence de `N` écrans de même émissivité que les plaques
/// `qN = q0/(N + 1)` (W/m²), où `q0` est le flux sans écran.
///
/// Panique si `number_of_shields < 0`.
pub fn radshield_flux_with_shields(flux_without_shields: f64, number_of_shields: f64) -> f64 {
    assert!(
        number_of_shields >= 0.0,
        "le nombre d'écrans doit être positif ou nul"
    );
    flux_without_shields / (number_of_shields + 1.0)
}

/// Flux radiatif net **sans écran** entre deux grandes plaques parallèles grises
/// `q0 = σ·(T1⁴ − T2⁴) / (1/e1 + 1/e2 − 1)` (W/m²).
///
/// Panique si `stefan_boltzmann <= 0`, si `emissivity1` ou `emissivity2` hors
/// `]0, 1]`, ou si une température est négative.
pub fn radshield_two_plate_flux(
    stefan_boltzmann: f64,
    emissivity1: f64,
    emissivity2: f64,
    temperature1: f64,
    temperature2: f64,
) -> f64 {
    assert!(
        stefan_boltzmann > 0.0,
        "la constante de Stefan-Boltzmann doit être strictement positive"
    );
    assert!(
        (0.0..=1.0).contains(&emissivity1) && emissivity1 > 0.0,
        "l'émissivité de la plaque 1 doit être dans ]0, 1]"
    );
    assert!(
        (0.0..=1.0).contains(&emissivity2) && emissivity2 > 0.0,
        "l'émissivité de la plaque 2 doit être dans ]0, 1]"
    );
    assert!(
        temperature1 >= 0.0 && temperature2 >= 0.0,
        "les températures absolues doivent être positives"
    );
    stefan_boltzmann * (temperature1.powi(4) - temperature2.powi(4))
        / (1.0 / emissivity1 + 1.0 / emissivity2 - 1.0)
}

/// Flux radiatif net avec **un écran** d'émissivité `es` (sur ses deux faces)
/// inséré entre deux plaques d'émissivité `ep`
/// `q1 = σ·(T1⁴ − T2⁴) / (2/ep + 2/es − 2)` (W/m²).
///
/// Les deux résistances radiatives en série valent chacune `1/ep + 1/es − 1`,
/// d'où le dénominateur `2·(1/ep + 1/es − 1) = 2/ep + 2/es − 2`. Lorsque
/// `es == ep`, ce flux vaut exactement la moitié de [`radshield_two_plate_flux`].
///
/// Panique si `stefan_boltzmann <= 0`, si `emissivity_plate` ou
/// `emissivity_shield` hors `]0, 1]`, ou si une température est négative.
pub fn radshield_flux_one_shield(
    stefan_boltzmann: f64,
    emissivity_plate: f64,
    emissivity_shield: f64,
    temperature1: f64,
    temperature2: f64,
) -> f64 {
    assert!(
        stefan_boltzmann > 0.0,
        "la constante de Stefan-Boltzmann doit être strictement positive"
    );
    assert!(
        (0.0..=1.0).contains(&emissivity_plate) && emissivity_plate > 0.0,
        "l'émissivité des plaques doit être dans ]0, 1]"
    );
    assert!(
        (0.0..=1.0).contains(&emissivity_shield) && emissivity_shield > 0.0,
        "l'émissivité de l'écran doit être dans ]0, 1]"
    );
    assert!(
        temperature1 >= 0.0 && temperature2 >= 0.0,
        "les températures absolues doivent être positives"
    );
    stefan_boltzmann * (temperature1.powi(4) - temperature2.powi(4))
        / (2.0 / emissivity_plate + 2.0 / emissivity_shield - 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn reduction_factor_known_values() {
        // Sans écran (N=0) le flux est intact ; un écran le divise par deux.
        assert_relative_eq!(
            radshield_reduction_factor_equal_emissivity(0.0),
            1.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            radshield_reduction_factor_equal_emissivity(1.0),
            0.5,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            radshield_reduction_factor_equal_emissivity(3.0),
            0.25,
            epsilon = 1e-12
        );
    }

    #[test]
    fn flux_with_shields_matches_reduction_factor() {
        // qN = q0·f(N) : cohérence entre le flux réduit et le facteur de réduction.
        let (q0, n) = (12_500.0_f64, 4.0_f64);
        assert_relative_eq!(
            radshield_flux_with_shields(q0, n),
            q0 * radshield_reduction_factor_equal_emissivity(n),
            max_relative = 1e-12
        );
    }

    #[test]
    fn two_plate_flux_vanishes_at_thermal_equilibrium() {
        // T1 = T2 → flux net nul quelles que soient les émissivités.
        assert_relative_eq!(
            radshield_two_plate_flux(5.670_374_419e-8, 0.7, 0.4, 500.0, 500.0),
            0.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn two_plate_flux_black_plates_numeric() {
        // Plaques noires (e1=e2=1) → dénominateur = 1, donc q0 = σ·T1⁴.
        // σ·1000⁴ = 5,670374419e-8 · 1e12 = 56703,74419 W/m².
        let sigma = 5.670_374_419e-8_f64;
        assert_relative_eq!(
            radshield_two_plate_flux(sigma, 1.0, 1.0, 1000.0, 0.0),
            56_703.744_19,
            epsilon = 1e-3
        );
    }

    #[test]
    fn one_shield_halves_flux_when_emissivities_equal() {
        // es = ep → un écran divise exactement le flux par deux (cas N=1).
        let (sigma, ep, t1, t2) = (5.670_374_419e-8_f64, 0.8_f64, 800.0_f64, 300.0_f64);
        let q0 = radshield_two_plate_flux(sigma, ep, ep, t1, t2);
        assert_relative_eq!(
            radshield_flux_one_shield(sigma, ep, ep, t1, t2),
            q0 / 2.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn one_shield_lower_emissivity_reduces_flux_more() {
        // Un écran plus réfléchissant (es plus faible) réduit davantage le flux.
        let (sigma, ep, t1, t2) = (5.670_374_419e-8_f64, 0.8_f64, 800.0_f64, 300.0_f64);
        let q_bright = radshield_flux_one_shield(sigma, ep, 0.05, t1, t2);
        let q_dull = radshield_flux_one_shield(sigma, ep, 0.8, t1, t2);
        assert!(q_bright < q_dull);
    }

    #[test]
    #[should_panic(expected = "émissivité de l'écran")]
    fn one_shield_zero_shield_emissivity_panics() {
        radshield_flux_one_shield(5.670_374_419e-8, 0.8, 0.0, 800.0, 300.0);
    }
}
