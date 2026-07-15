//! **Soudage par points par résistance** — chaleur de **Joule** dissipée au
//! point de contact et densité de chaleur déposée dans le noyau fondu (nugget).
//!
//! ```text
//! chaleur de Joule       Q = I²·R·t                       (J)
//! intensité requise      I = √( Q / (R·t) )               (A)   [réciproque]
//! densité de chaleur     q = Q / V                        (J/m³)
//! ```
//!
//! `I` intensité de soudage (A), `R` résistance électrique au point de contact
//! (Ω), `t` temps de passage du courant (s), `Q` chaleur de Joule totale
//! dissipée (J), `V` volume du noyau fondu (m³), `q` densité volumique de
//! chaleur déposée (J/m³).
//!
//! **Convention** : SI. **Limite honnête** : modèle de **chaleur de Joule
//! totale** `Q = I²·R·t` ; les pertes de conduction vers les électrodes et les
//! tôles adjacentes ainsi que le rayonnement sont **négligés**, de sorte que
//! `Q` majore la chaleur réellement disponible pour la fusion. La **résistance
//! de contact** `R` (dynamique, dépendant de l'état de surface, de la pression
//! d'électrode et de la température) est **fournie par l'appelant** ; aucune
//! valeur « par défaut » n'est inventée. La force d'électrode n'admet pas de
//! formule simple et n'est pas modélisée ici. Voir [`crate::weld_heat_input`]
//! (soudage à l'arc) et [`crate::thermal`].

/// Chaleur de Joule totale `Q = I²·R·t` (J).
///
/// Panique si `current < 0`, si `resistance < 0` ou si `time < 0`.
pub fn joule_heat(current: f64, resistance: f64, time: f64) -> f64 {
    assert!(current >= 0.0, "l'intensité I ≥ 0 requise");
    assert!(resistance >= 0.0, "la résistance R ≥ 0 requise");
    assert!(time >= 0.0, "le temps t ≥ 0 requis");
    current.powi(2) * resistance * time
}

/// Intensité de soudage requise `I = √( Q / (R·t) )` (A).
///
/// Réciproque de [`joule_heat`] : intensité nécessaire pour dissiper la chaleur
/// `Q` dans une résistance `R` pendant un temps `t`.
///
/// Panique si `heat < 0`, si `resistance <= 0` ou si `time <= 0`.
pub fn spot_current_from_heat(heat: f64, resistance: f64, time: f64) -> f64 {
    assert!(heat >= 0.0, "la chaleur Q ≥ 0 requise");
    assert!(
        resistance > 0.0,
        "la résistance R doit être strictement positive"
    );
    assert!(time > 0.0, "le temps t doit être strictement positif");
    (heat / (resistance * time)).sqrt()
}

/// Densité volumique de chaleur déposée dans le noyau `q = Q / V` (J/m³).
///
/// Panique si `heat < 0` ou si `nugget_volume <= 0`.
pub fn nugget_heat_density(heat: f64, nugget_volume: f64) -> f64 {
    assert!(heat >= 0.0, "la chaleur Q ≥ 0 requise");
    assert!(
        nugget_volume > 0.0,
        "le volume du noyau V doit être strictement positif"
    );
    heat / nugget_volume
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn joule_heat_realistic_case() {
        // I=10 kA, R=100 µΩ, t=0,2 s → Q = (1e4)²·1e-4·0,2 = 2000 J.
        let q = joule_heat(1.0e4, 1.0e-4, 0.2);
        assert_relative_eq!(q, 2000.0, epsilon = 1e-6);
    }

    #[test]
    fn current_from_heat_is_reciprocal_of_joule_heat() {
        // I → Q → I doit reboucler sur l'intensité de départ.
        let (i, r, t) = (8.0e3_f64, 1.5e-4_f64, 0.25_f64);
        let q = joule_heat(i, r, t);
        assert_relative_eq!(spot_current_from_heat(q, r, t), i, epsilon = 1e-6);
    }

    #[test]
    fn joule_heat_quadratic_in_current() {
        // Doubler l'intensité quadruple la chaleur (∝ I²).
        let base = joule_heat(5.0e3, 2.0e-4, 0.15);
        let doubled = joule_heat(1.0e4, 2.0e-4, 0.15);
        assert_relative_eq!(doubled, 4.0 * base, epsilon = 1e-6);
    }

    #[test]
    fn joule_heat_linear_in_time() {
        // La chaleur est proportionnelle au temps de passage du courant.
        let short = joule_heat(9.0e3, 1.2e-4, 0.10);
        let long = joule_heat(9.0e3, 1.2e-4, 0.30);
        assert_relative_eq!(long, 3.0 * short, epsilon = 1e-6);
    }

    #[test]
    fn nugget_heat_density_realistic_case() {
        // Q=2000 J déposés dans un noyau V=20 mm³=2e-8 m³ → q=1e11 J/m³.
        let q = nugget_heat_density(2000.0, 2.0e-8);
        assert_relative_eq!(q, 1.0e11, epsilon = 1.0);
    }

    #[test]
    fn nugget_heat_density_inverse_in_volume() {
        // À chaleur fixée, doubler le volume divise la densité par deux.
        let small = nugget_heat_density(1500.0, 1.0e-8);
        let large = nugget_heat_density(1500.0, 2.0e-8);
        assert_relative_eq!(large, small / 2.0, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(expected = "le temps t doit être strictement positif")]
    fn current_from_heat_zero_time_panics() {
        spot_current_from_heat(2000.0, 1.0e-4, 0.0);
    }
}
