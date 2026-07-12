//! Fatigue — **contrainte moyenne** et diagrammes de tenue : décomposition
//! amplitude/moyenne, et coefficients de sécurité de **Goodman**, **Soderberg**
//! et **Gerber** (diagramme de Haigh).
//!
//! ```text
//! amplitude      σa = (σmax − σmin)/2       moyenne σm = (σmax + σmin)/2
//! rapport        R = σmin/σmax
//! Goodman        σa/Se + σm/Su = 1/n
//! Soderberg      σa/Se + σm/Sy = 1/n
//! Gerber         n·σa/Se + (n·σm/Su)² = 1
//! ```
//!
//! `σa` amplitude, `σm` moyenne, `Se` limite d'endurance (corrigée), `Su`
//! résistance à la rupture, `Sy` limite élastique, `n` coefficient de sécurité.
//! Goodman est la référence prudente usuelle en traction ; Soderberg (borné à
//! `Sy`) est le plus conservateur ; Gerber (parabole) colle mieux aux essais.
//!
//! **Convention** : contraintes cohérentes de l'appelant (MPa ou Pa). **Limite
//! honnête** : critères de **tenue infinie** à contrainte moyenne positive
//! (traction) ; le comptage de cycles et le cumul de dommage relèvent de la
//! crate `scirust-fatigue` (rainflow, Palmgren-Miner). `Se`, `Su`, `Sy` sont
//! fournis par l'appelant.

/// Amplitude de contrainte `σa = (σmax − σmin)/2`.
pub fn stress_amplitude(s_max: f64, s_min: f64) -> f64 {
    (s_max - s_min) / 2.0
}

/// Contrainte moyenne `σm = (σmax + σmin)/2`.
pub fn mean_stress(s_max: f64, s_min: f64) -> f64 {
    (s_max + s_min) / 2.0
}

/// Rapport de charge `R = σmin/σmax`.
///
/// Panique si `s_max == 0`.
pub fn stress_ratio(s_max: f64, s_min: f64) -> f64 {
    assert!(s_max != 0.0, "σmax ne doit pas être nul");
    s_min / s_max
}

/// Coefficient de sécurité de **Goodman** `1/n = σa/Se + σm/Su`.
///
/// Panique si `Se <= 0` ou `Su <= 0`.
pub fn goodman_safety_factor(sa: f64, sm: f64, se: f64, su: f64) -> f64 {
    assert!(se > 0.0 && su > 0.0, "Se > 0 et Su > 0 requis");
    1.0 / (sa / se + sm / su)
}

/// Coefficient de sécurité de **Soderberg** `1/n = σa/Se + σm/Sy`.
///
/// Panique si `Se <= 0` ou `Sy <= 0`.
pub fn soderberg_safety_factor(sa: f64, sm: f64, se: f64, sy: f64) -> f64 {
    assert!(se > 0.0 && sy > 0.0, "Se > 0 et Sy > 0 requis");
    1.0 / (sa / se + sm / sy)
}

/// Coefficient de sécurité de **Gerber** (parabole) `n·σa/Se + (n·σm/Su)² = 1`.
///
/// Panique si `Se <= 0` ou `Su <= 0`.
pub fn gerber_safety_factor(sa: f64, sm: f64, se: f64, su: f64) -> f64 {
    assert!(se > 0.0 && su > 0.0, "Se > 0 et Su > 0 requis");
    let a = sa / se;
    let b = (sm / su) * (sm / su);
    if b == 0.0
    {
        return 1.0 / a;
    }
    // b·n² + a·n − 1 = 0 → racine positive.
    (-a + (a * a + 4.0 * b).sqrt()) / (2.0 * b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn amplitude_and_mean_from_extremes() {
        // σmax=200, σmin=−50 → σa=125, σm=75, R=−0,25.
        assert_relative_eq!(stress_amplitude(200.0, -50.0), 125.0, epsilon = 1e-9);
        assert_relative_eq!(mean_stress(200.0, -50.0), 75.0, epsilon = 1e-9);
        assert_relative_eq!(stress_ratio(200.0, -50.0), -0.25, epsilon = 1e-9);
    }

    #[test]
    fn fully_reversed_gives_endurance_safety() {
        // σm=0 : Goodman/Soderberg/Gerber → n = Se/σa tous égaux.
        let (sa, se, su, sy) = (100.0, 200.0, 500.0, 350.0);
        assert_relative_eq!(goodman_safety_factor(sa, 0.0, se, su), 2.0, epsilon = 1e-9);
        assert_relative_eq!(
            soderberg_safety_factor(sa, 0.0, se, sy),
            2.0,
            epsilon = 1e-9
        );
        assert_relative_eq!(gerber_safety_factor(sa, 0.0, se, su), 2.0, epsilon = 1e-9);
    }

    #[test]
    fn conservatism_ordering() {
        // À contrainte moyenne positive : Soderberg ≤ Goodman ≤ Gerber (sécurité).
        let (sa, sm, se, su, sy) = (80.0, 150.0, 200.0, 500.0, 350.0);
        let sod = soderberg_safety_factor(sa, sm, se, sy);
        let good = goodman_safety_factor(sa, sm, se, su);
        let ger = gerber_safety_factor(sa, sm, se, su);
        assert!(sod <= good);
        assert!(good <= ger);
    }

    #[test]
    fn goodman_on_the_line_gives_unity() {
        // Point (σa, σm) exactement sur la droite de Goodman → n = 1.
        let (se, su) = (200.0, 500.0);
        let sm = 250.0;
        let sa = se * (1.0 - sm / su); // = 200·0,5 = 100
        assert_relative_eq!(goodman_safety_factor(sa, sm, se, su), 1.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "Se > 0")]
    fn zero_endurance_panics() {
        goodman_safety_factor(100.0, 50.0, 0.0, 500.0);
    }
}
