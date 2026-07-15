//! **Actionneur à bobine mobile** (voice coil) — force de Lorentz linéaire d'un
//! moteur linéaire à aimant permanent et bobine dans l'entrefer.
//!
//! ```text
//! constante de force  BL = B·ℓ                          (induction × longueur de fil)
//! force de Lorentz    F  = BL·I
//! f.é.m. contre-élec. e  = BL·v
//! pertes Joule        P  = I²·R
//! constante de moteur Km = BL/√R                         (force par racine de perte)
//! ```
//!
//! `B` induction dans l'entrefer (T), `ℓ` longueur de fil plongée dans le champ
//! (m), `BL` constante de force (N/A, identique à la constante de f.é.m. en
//! V·s/m), `I` courant (A), `F` force axiale (N), `v` vitesse de la bobine
//! (m/s), `e` f.é.m. contre-électromotrice (V), `R` résistance de bobine (Ω),
//! `P` pertes Joule (W), `Km` constante de moteur (N/√W).
//!
//! **Convention** : SI (T, m, A, N, m/s, V, Ω, W).
//! **Limite honnête** : force de Lorentz **linéaire**, produit `BL` supposé
//! **constant** sur toute la course — l'induction `B` et la longueur `ℓ` de fil
//! dans le champ sont des données **fournies** par l'appelant, jamais une valeur
//! « par défaut » inventée. On néglige la chute de flux en bout de course
//! (fringing) et l'inductance de la bobine (régime établi). Distinct de
//! [`crate::solenoid_actuator`] (force de réluctance, non linéaire en position
//! et en courant).

/// Constante de force `BL = B·ℓ` (N/A), aussi constante de f.é.m. (V·s/m).
///
/// Panique si `flux_density <= 0` ou `wire_length_in_field <= 0`.
pub fn voicecoil_force_constant(flux_density: f64, wire_length_in_field: f64) -> f64 {
    assert!(
        flux_density > 0.0,
        "l'induction dans l'entrefer doit être strictement positive"
    );
    assert!(
        wire_length_in_field > 0.0,
        "la longueur de fil dans le champ doit être strictement positive"
    );
    flux_density * wire_length_in_field
}

/// Force de Lorentz `F = BL·I` (N). Le signe suit celui du courant.
///
/// Panique si `force_constant` ou `current` n'est pas fini.
pub fn voicecoil_force(force_constant: f64, current: f64) -> f64 {
    assert!(
        force_constant.is_finite(),
        "la constante de force doit être finie"
    );
    assert!(current.is_finite(), "le courant doit être fini");
    force_constant * current
}

/// F.é.m. contre-électromotrice `e = BL·v` (V). Le signe suit celui de la vitesse.
///
/// Panique si `force_constant` ou `velocity` n'est pas fini.
pub fn voicecoil_back_emf(force_constant: f64, velocity: f64) -> f64 {
    assert!(
        force_constant.is_finite(),
        "la constante de force doit être finie"
    );
    assert!(velocity.is_finite(), "la vitesse doit être finie");
    force_constant * velocity
}

/// Pertes Joule `P = I²·R` (W).
///
/// Panique si `coil_resistance <= 0` ou `current` n'est pas fini.
pub fn voicecoil_power_dissipation(current: f64, coil_resistance: f64) -> f64 {
    assert!(current.is_finite(), "le courant doit être fini");
    assert!(
        coil_resistance > 0.0,
        "la résistance de bobine doit être strictement positive"
    );
    current * current * coil_resistance
}

/// Constante de moteur `Km = BL/√R` (N/√W), efficacité force par racine de perte.
///
/// Panique si `force_constant < 0` ou `coil_resistance <= 0`.
pub fn voicecoil_motor_constant(force_constant: f64, coil_resistance: f64) -> f64 {
    assert!(
        force_constant >= 0.0,
        "la constante de force doit être positive ou nulle"
    );
    assert!(
        coil_resistance > 0.0,
        "la résistance de bobine doit être strictement positive"
    );
    force_constant / coil_resistance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn force_constant_is_product_b_times_l() {
        // B = 1,0 T, ℓ = 10 m de fil dans le champ → BL = 10 N/A.
        assert_relative_eq!(voicecoil_force_constant(1.0, 10.0), 10.0, epsilon = 1e-12);
    }

    #[test]
    fn force_and_back_emf_share_the_same_constant() {
        // La même constante BL relie force/courant et f.é.m./vitesse :
        // F/I = e/v = BL.
        let bl = 10.0_f64;
        let force = voicecoil_force(bl, 3.0);
        let emf = voicecoil_back_emf(bl, 3.0);
        assert_relative_eq!(force, emf, epsilon = 1e-12);
        assert_relative_eq!(force / 3.0, bl, epsilon = 1e-12);
    }

    #[test]
    fn force_composes_with_the_force_constant() {
        // F(B·ℓ, I) = B·ℓ·I : 1,0 T · 10 m · 2 A = 20 N.
        let bl = voicecoil_force_constant(1.0, 10.0);
        assert_relative_eq!(voicecoil_force(bl, 2.0), 20.0, epsilon = 1e-12);
    }

    #[test]
    fn power_is_quadratic_in_current() {
        // Doubler le courant quadruple les pertes Joule.
        let p1 = voicecoil_power_dissipation(2.0, 5.0);
        let p2 = voicecoil_power_dissipation(4.0, 5.0);
        assert_relative_eq!(p1, 20.0, epsilon = 1e-12);
        assert_relative_eq!(p2, 4.0 * p1, epsilon = 1e-12);
    }

    #[test]
    fn motor_constant_realistic_case() {
        // BL = 10 N/A, R = 5 Ω → Km = 10/√5 = 2√5 ≈ 4,472135955 N/√W.
        let km = voicecoil_motor_constant(10.0, 5.0);
        assert_relative_eq!(km, 4.472_135_955, max_relative = 1e-3);
        // Vérification : Km·√R redonne BL.
        assert_relative_eq!(km * 5.0_f64.sqrt(), 10.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "la résistance de bobine doit être strictement positive")]
    fn power_rejects_nonpositive_resistance() {
        let _ = voicecoil_power_dissipation(3.0, 0.0);
    }
}
