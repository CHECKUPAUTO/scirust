//! **Courroie plate (Euler-Eytelwein)** — capacité de transmission au glissement
//! imminent : rapport des tensions, puissance transmise et tension centrifuge.
//!
//! ```text
//! rapport limite      T1/T2 = exp(mu·theta)          (glissement imminent)
//! puissance           P = (T1 − T2)·v
//! brin mou            T2 = T1/exp(mu·theta)
//! tension centrifuge  Tc = m'·v²
//! ```
//!
//! `T1`, `T2` tensions des brins tendu et mou (N), `mu` coefficient de frottement
//! courroie/poulie (sans dimension), `theta` angle d'enroulement sur la poulie
//! (rad), `P` puissance transmise (W), `v` vitesse linéaire de la courroie (m/s),
//! `m'` masse linéique de la courroie (kg/m), `Tc` tension centrifuge (N).
//!
//! **Convention** : SI cohérent. **Limite honnête** : formules valables au
//! **glissement imminent** (capacité maximale de la poulie) ; le coefficient de
//! frottement `mu`, la masse linéique `m'` et les tensions sont **fournis par
//! l'appelant** — aucune constante de matériau ou de procédé n'est inventée ;
//! courroie supposée **inextensible**. La tension centrifuge `Tc` diminue la
//! capacité utile à grande vitesse. Complète [`crate::belt_slip`].

/// Rapport limite des tensions au glissement imminent `T1/T2 = exp(mu·theta)`.
///
/// Panique si `friction_coefficient < 0` ou `wrap_angle_rad < 0`.
pub fn flatbelt_tension_ratio(friction_coefficient: f64, wrap_angle_rad: f64) -> f64 {
    assert!(friction_coefficient >= 0.0, "mu ≥ 0 requis");
    assert!(wrap_angle_rad >= 0.0, "theta ≥ 0 requis (rad)");
    (friction_coefficient * wrap_angle_rad).exp()
}

/// Puissance transmise `P = (T1 − T2)·v`, différence des tensions des brins
/// multipliée par la vitesse linéaire de la courroie.
///
/// Panique si `slack_tension < 0`, `tight_tension < slack_tension`, ou
/// `belt_velocity < 0`.
pub fn flatbelt_power(tight_tension: f64, slack_tension: f64, belt_velocity: f64) -> f64 {
    assert!(
        slack_tension >= 0.0 && tight_tension >= slack_tension,
        "T1 ≥ T2 ≥ 0 requis (brin tendu ≥ brin mou)"
    );
    assert!(belt_velocity >= 0.0, "v ≥ 0 requis (m/s)");
    (tight_tension - slack_tension) * belt_velocity
}

/// Tension du brin mou au glissement imminent `T2 = T1/exp(mu·theta)`, déduite de
/// la tension du brin tendu.
///
/// Panique si `tight_tension < 0`, `friction_coefficient < 0`, ou
/// `wrap_angle_rad < 0`.
pub fn flatbelt_slack_from_ratio(
    tight_tension: f64,
    friction_coefficient: f64,
    wrap_angle_rad: f64,
) -> f64 {
    assert!(tight_tension >= 0.0, "T1 ≥ 0 requis");
    assert!(friction_coefficient >= 0.0, "mu ≥ 0 requis");
    assert!(wrap_angle_rad >= 0.0, "theta ≥ 0 requis (rad)");
    tight_tension / (friction_coefficient * wrap_angle_rad).exp()
}

/// Tension centrifuge `Tc = m'·v²` induite par la mise en rotation de la masse de
/// courroie ; elle réduit la capacité utile de transmission à grande vitesse.
///
/// Panique si `mass_per_length < 0` ou `belt_velocity < 0`.
pub fn flatbelt_centrifugal_tension(mass_per_length: f64, belt_velocity: f64) -> f64 {
    assert!(mass_per_length >= 0.0, "m' ≥ 0 requis (kg/m)");
    assert!(belt_velocity >= 0.0, "v ≥ 0 requis (m/s)");
    mass_per_length * belt_velocity * belt_velocity
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn tension_ratio_unity_without_wrap() {
        // theta = 0 (ou mu = 0) → aucun frottement mobilisable → T1/T2 = 1.
        assert_relative_eq!(flatbelt_tension_ratio(0.3, 0.0), 1.0, epsilon = 1e-12);
        assert_relative_eq!(
            flatbelt_tension_ratio(0.0, core::f64::consts::PI),
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn ratio_and_slack_are_reciprocal() {
        // T2 = T1/exp(mu·theta) ⇒ T1/T2 = exp(mu·theta) = rapport limite.
        let mu = 0.35_f64;
        let theta = 2.5_f64;
        let t1 = 1200.0_f64;
        let t2 = flatbelt_slack_from_ratio(t1, mu, theta);
        assert_relative_eq!(t1 / t2, flatbelt_tension_ratio(mu, theta), epsilon = 1e-12);
    }

    #[test]
    fn tension_ratio_realistic_value() {
        // mu = 0.3, theta = pi (enroulement 180°) → exp(0.3·pi) ≈ 2.566332.
        assert_relative_eq!(
            flatbelt_tension_ratio(0.3, core::f64::consts::PI),
            2.566_332,
            epsilon = 1e-5
        );
    }

    #[test]
    fn power_is_linear_in_velocity() {
        // P ∝ v : doubler la vitesse double la puissance transmise.
        let base = flatbelt_power(1000.0, 400.0, 10.0);
        let twice = flatbelt_power(1000.0, 400.0, 20.0);
        assert_relative_eq!(twice, 2.0 * base, epsilon = 1e-12);
        // Cas chiffré : (1000 − 400)·10 = 6000 W.
        assert_relative_eq!(base, 6000.0, epsilon = 1e-9);
    }

    #[test]
    fn centrifugal_tension_is_quadratic() {
        // Tc ∝ v² : doubler la vitesse quadruple la tension centrifuge.
        let base = flatbelt_centrifugal_tension(0.5, 20.0);
        let twice = flatbelt_centrifugal_tension(0.5, 40.0);
        assert_relative_eq!(twice, 4.0 * base, epsilon = 1e-12);
        // Cas chiffré : 0.5·20² = 200 N.
        assert_relative_eq!(base, 200.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "T1 ≥ T2 ≥ 0 requis")]
    fn power_rejects_slack_above_tight() {
        flatbelt_power(300.0, 500.0, 10.0);
    }
}
