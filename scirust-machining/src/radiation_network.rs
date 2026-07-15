//! Rayonnement thermique par la **méthode du réseau de résistances** (analogie
//! électrique) entre surfaces grises diffuses.
//!
//! ```text
//! résistance de surface  Rs = (1 − ε)/(ε·A)
//! résistance d'espace    R12 = 1/(A·F12)
//! échange deux surfaces  Q = σ·(T1⁴ − T2⁴)/(Rs1 + R12 + Rs2)
//! grandes plaques //     q = σ·(T1⁴ − T2⁴)/(1/ε1 + 1/ε2 − 1)
//! ```
//!
//! `ε` émissivité (`0`–`1`), `A` aire de la surface (m²), `F12` facteur de forme
//! (`0`–`1`), `σ` constante de Stefan-Boltzmann (W/(m²·K⁴)), `T` températures
//! **absolues** (K). `Rs` et `R12` sont des résistances radiatives (1/m²) ; le
//! « potentiel » du réseau est la puissance émissive de corps noir `σ·T⁴`. `Q`
//! est un flux net **total** (W) ; `q` de `radnet_two_gray_parallel_plates` est
//! une **densité** de flux nette (W/m²).
//!
//! **Convention** : températures en **kelvin**, résistances en série. **Limite
//! honnête** : surfaces **grises diffuses opaques** en régime **permanent** ;
//! les émissivités, aires et facteurs de forme sont **fournis par l'appelant**
//! (aucune valeur de matériau ou géométrique par défaut n'est inventée), de même
//! que `σ`. Complète [`crate::view_factor`] (facteurs de forme géométriques) et
//! [`crate::radiation`] (loi de Stefan-Boltzmann).

/// Résistance radiative de **surface** d'un corps gris `Rs = (1 − ε)/(ε·A)`
/// (1/m²).
///
/// Panique si `emissivity` hors `]0, 1]` ou si `area <= 0`.
pub fn radnet_surface_resistance(emissivity: f64, area: f64) -> f64 {
    assert!(
        emissivity > 0.0 && emissivity <= 1.0,
        "l'émissivité doit être dans ]0, 1]"
    );
    assert!(area > 0.0, "l'aire doit être strictement positive");
    (1.0 - emissivity) / (emissivity * area)
}

/// Résistance radiative d'**espace** entre deux surfaces `R12 = 1/(A·F12)`
/// (1/m²).
///
/// Panique si `area <= 0` ou si `view_factor` hors `]0, 1]`.
pub fn radnet_space_resistance(area: f64, view_factor: f64) -> f64 {
    assert!(area > 0.0, "l'aire doit être strictement positive");
    assert!(
        view_factor > 0.0 && view_factor <= 1.0,
        "le facteur de forme doit être dans ]0, 1]"
    );
    1.0 / (area * view_factor)
}

/// Échange radiatif net entre **deux surfaces grises** via les trois résistances
/// en série `Q = σ·(T1⁴ − T2⁴)/(Rs1 + R12 + Rs2)` (W).
///
/// Panique si `stefan_boltzmann <= 0`, si une température est négative ou si une
/// résistance est négative (ou toutes nulles).
pub fn radnet_two_gray_surface_exchange(
    stefan_boltzmann: f64,
    temperature1: f64,
    temperature2: f64,
    surface_resistance1: f64,
    space_resistance: f64,
    surface_resistance2: f64,
) -> f64 {
    assert!(
        stefan_boltzmann > 0.0,
        "la constante de Stefan-Boltzmann doit être strictement positive"
    );
    assert!(
        temperature1 >= 0.0 && temperature2 >= 0.0,
        "les températures absolues doivent être positives"
    );
    assert!(
        surface_resistance1 >= 0.0 && space_resistance >= 0.0 && surface_resistance2 >= 0.0,
        "les résistances radiatives doivent être positives"
    );
    let total = surface_resistance1 + space_resistance + surface_resistance2;
    assert!(
        total > 0.0,
        "la résistance totale du réseau doit être strictement positive"
    );
    stefan_boltzmann * (temperature1.powi(4) - temperature2.powi(4)) / total
}

/// Densité de flux radiatif net entre **grandes plaques parallèles** grises
/// `q = σ·(T1⁴ − T2⁴)/(1/ε1 + 1/ε2 − 1)` (W/m²).
///
/// Panique si `stefan_boltzmann <= 0`, si une émissivité est hors `]0, 1]` ou si
/// une température est négative.
pub fn radnet_two_gray_parallel_plates(
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
        emissivity1 > 0.0 && emissivity1 <= 1.0,
        "l'émissivité 1 doit être dans ]0, 1]"
    );
    assert!(
        emissivity2 > 0.0 && emissivity2 <= 1.0,
        "l'émissivité 2 doit être dans ]0, 1]"
    );
    assert!(
        temperature1 >= 0.0 && temperature2 >= 0.0,
        "les températures absolues doivent être positives"
    );
    stefan_boltzmann * (temperature1.powi(4) - temperature2.powi(4))
        / (1.0 / emissivity1 + 1.0 / emissivity2 - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// σ de référence (W/(m²·K⁴)) fournie pour les tests.
    const SIGMA: f64 = 5.670_374_419e-8;

    #[test]
    fn surface_resistance_vanishes_for_black_body() {
        // ε = 1 (corps noir) → aucune résistance de surface.
        assert_relative_eq!(radnet_surface_resistance(1.0, 2.5), 0.0, epsilon = 1e-15);
        // Résistance connue : (1 − 0,5)/(0,5·4) = 0,25.
        assert_relative_eq!(radnet_surface_resistance(0.5, 4.0), 0.25, epsilon = 1e-12);
    }

    #[test]
    fn space_resistance_reciprocal_of_conductance() {
        // R12·(A·F12) = 1 par définition.
        let (area, view_factor) = (3.0_f64, 0.4_f64);
        let r12 = radnet_space_resistance(area, view_factor);
        assert_relative_eq!(r12 * area * view_factor, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn surface_exchange_vanishes_at_thermal_equilibrium() {
        // T1 = T2 → flux net nul quelles que soient les résistances.
        let q = radnet_two_gray_surface_exchange(SIGMA, 600.0, 600.0, 0.1, 0.5, 0.2);
        assert_relative_eq!(q, 0.0, epsilon = 1e-12);
        // T1 > T2 → flux sortant positif.
        assert!(radnet_two_gray_surface_exchange(SIGMA, 700.0, 500.0, 0.1, 0.5, 0.2) > 0.0);
    }

    #[test]
    fn parallel_plates_equal_network_of_series_resistances() {
        // Pour deux plaques d'aire A et F12 = 1, le réseau série redonne
        // exactement q·A : Rs1 + R12 + Rs2 = (1/ε1 + 1/ε2 − 1)/A.
        let (eps1, eps2) = (0.8_f64, 0.6_f64);
        let (t1, t2) = (900.0_f64, 400.0_f64);
        let area = 2.0_f64;
        let rs1 = radnet_surface_resistance(eps1, area);
        let r12 = radnet_space_resistance(area, 1.0);
        let rs2 = radnet_surface_resistance(eps2, area);
        let q_network = radnet_two_gray_surface_exchange(SIGMA, t1, t2, rs1, r12, rs2);
        let q_plates = area * radnet_two_gray_parallel_plates(SIGMA, eps1, eps2, t1, t2);
        assert_relative_eq!(q_network, q_plates, max_relative = 1e-12);
    }

    #[test]
    fn parallel_plates_known_numeric_case() {
        // ε1 = ε2 = 0,8, T1 = 800 K, T2 = 500 K :
        // dénominateur 1/0,8 + 1/0,8 − 1 = 1,5.
        // T1⁴ − T2⁴ = 4,096e11 − 6,25e10 = 3,471e11.
        // q = 5,670374419e-8·3,471e11/1,5 = 13121,2465 W/m².
        let q = radnet_two_gray_parallel_plates(SIGMA, 0.8, 0.8, 800.0, 500.0);
        assert_relative_eq!(q, 13_121.246_4, max_relative = 1e-6);
    }

    #[test]
    fn parallel_plates_antisymmetric_in_temperature() {
        // Inverser T1 et T2 change le signe du flux (T1⁴ − T2⁴).
        let up = radnet_two_gray_parallel_plates(SIGMA, 0.9, 0.7, 750.0, 300.0);
        let down = radnet_two_gray_parallel_plates(SIGMA, 0.9, 0.7, 300.0, 750.0);
        assert_relative_eq!(up, -down, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "émissivité")]
    fn zero_emissivity_surface_resistance_panics() {
        radnet_surface_resistance(0.0, 1.0);
    }
}
