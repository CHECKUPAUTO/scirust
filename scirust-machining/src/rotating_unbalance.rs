//! Balourd tournant d'un système à 1 degré de liberté — force excitatrice
//! centrifuge, amplitude de la vibration forcée en régime permanent, force
//! transmise au bâti et excentricité spécifique admissible selon **ISO 1940**.
//!
//! ```text
//! force centrifuge      F = m·e·ω²
//! amplitude forcée      X = (m·e/M)·r² / √((1-r²)² + (2ζr)²)
//! transmissibilité      TR = √(1 + (2ζr)²) / √((1-r²)² + (2ζr)²)
//! force transmise       F_t = F·TR
//! rapport de fréquences r = ω/ω_n
//! excentricité admise   e_per = G/ω          (ISO 1940, e·ω = G)
//! ```
//!
//! `m` masse du balourd (kg), `e` excentricité du balourd (m), `ω` vitesse de
//! rotation (rad/s), `M` masse totale mobile du système (kg), `r` rapport de la
//! vitesse de rotation à la pulsation propre `ω_n` (sans dimension), `ζ` taux
//! d'amortissement réduit (sans dimension), `F` force en N, `G` classe de
//! qualité d'équilibrage ISO 1940 (`e·ω`, exprimée en mm/s), `e_per` en mm.
//!
//! **Limite honnête** : modèle à **1 ddl** à excitation par balourd tournant,
//! rotor **rigide** ; la masse du balourd, l'excentricité, la masse mobile et le
//! taux d'amortissement `ζ` sont **fournis** par l'appelant, de même que la
//! classe de qualité `G` (ISO 1940) — aucune valeur « par défaut » n'est
//! inventée ici. Complète [`crate::balancing`].

/// Force centrifuge excitatrice d'un balourd tournant `F = m·e·ω²` (N).
///
/// `unbalance_mass` en kg, `eccentricity` en m, `angular_speed` en rad/s.
///
/// Panique si `unbalance_mass < 0` ou `eccentricity < 0`.
pub fn unbalance_centrifugal_force(
    unbalance_mass: f64,
    eccentricity: f64,
    angular_speed: f64,
) -> f64 {
    assert!(
        unbalance_mass >= 0.0,
        "la masse du balourd doit être positive ou nulle"
    );
    assert!(
        eccentricity >= 0.0,
        "l'excentricité doit être positive ou nulle"
    );
    unbalance_mass * eccentricity * angular_speed * angular_speed
}

/// Amplitude de la vibration forcée en régime permanent (m)
/// `X = (m·e/M)·r² / √((1-r²)² + (2ζr)²)`, `r = ω/ω_n`.
///
/// `unbalance_mass` et `total_mass` en kg, `eccentricity` en m,
/// `frequency_ratio` et `damping_ratio` sans dimension.
///
/// Panique si `total_mass <= 0`, `unbalance_mass < 0`, `eccentricity < 0`,
/// `frequency_ratio < 0`, `damping_ratio < 0`, ou si le dénominateur s'annule
/// (`frequency_ratio == 1` avec `damping_ratio == 0`, résonance non amortie).
pub fn unbalance_forced_amplitude(
    unbalance_mass: f64,
    eccentricity: f64,
    total_mass: f64,
    frequency_ratio: f64,
    damping_ratio: f64,
) -> f64 {
    assert!(
        total_mass > 0.0,
        "la masse mobile doit être strictement positive"
    );
    assert!(
        unbalance_mass >= 0.0,
        "la masse du balourd doit être positive ou nulle"
    );
    assert!(
        eccentricity >= 0.0,
        "l'excentricité doit être positive ou nulle"
    );
    assert!(
        frequency_ratio >= 0.0,
        "le rapport de fréquences doit être positif ou nul"
    );
    assert!(
        damping_ratio >= 0.0,
        "le taux d'amortissement doit être positif ou nul"
    );
    let r2 = frequency_ratio * frequency_ratio;
    let denom = (1.0 - r2) * (1.0 - r2) + (2.0 * damping_ratio * frequency_ratio).powi(2);
    assert!(
        denom > 0.0,
        "résonance non amortie : dénominateur nul (r == 1 et ζ == 0)"
    );
    (unbalance_mass * eccentricity / total_mass) * r2 / denom.sqrt()
}

/// Force transmise au bâti par un balourd `F_t = F·TR` (N),
/// avec `TR = √(1 + (2ζr)²) / √((1-r²)² + (2ζr)²)`.
///
/// `unbalance_force` en N (typiquement issue de [`unbalance_centrifugal_force`]),
/// `frequency_ratio` et `damping_ratio` sans dimension.
///
/// Panique si `frequency_ratio < 0`, `damping_ratio < 0`, ou si le dénominateur
/// s'annule (`frequency_ratio == 1` avec `damping_ratio == 0`).
pub fn unbalance_transmitted_force(
    unbalance_force: f64,
    frequency_ratio: f64,
    damping_ratio: f64,
) -> f64 {
    assert!(
        frequency_ratio >= 0.0,
        "le rapport de fréquences doit être positif ou nul"
    );
    assert!(
        damping_ratio >= 0.0,
        "le taux d'amortissement doit être positif ou nul"
    );
    let r2 = frequency_ratio * frequency_ratio;
    let cross = (2.0 * damping_ratio * frequency_ratio).powi(2);
    let denom = (1.0 - r2) * (1.0 - r2) + cross;
    assert!(
        denom > 0.0,
        "résonance non amortie : dénominateur nul (r == 1 et ζ == 0)"
    );
    let transmissibility = ((1.0 + cross) / denom).sqrt();
    unbalance_force * transmissibility
}

/// Excentricité spécifique **admissible** `e_per = G/ω` (mm) selon **ISO 1940**,
/// où la classe de qualité vérifie `e·ω = G`.
///
/// `balance_grade` en mm/s (classe `G`, p. ex. `6.3`), `angular_speed` en rad/s ;
/// le résultat est une excentricité admissible en mm.
///
/// Panique si `angular_speed <= 0` ou `balance_grade < 0`.
pub fn unbalance_permissible_from_grade(balance_grade: f64, angular_speed: f64) -> f64 {
    assert!(
        angular_speed > 0.0,
        "la vitesse de rotation doit être strictement positive"
    );
    assert!(
        balance_grade >= 0.0,
        "la classe de qualité G doit être positive ou nulle"
    );
    balance_grade / angular_speed
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::PI;

    #[test]
    fn centrifugal_force_scales_with_square_of_speed() {
        // m=0,02 kg, e=0,001 m, ω=100 rad/s → F = 0,02·0,001·10000 = 0,2 N.
        assert_relative_eq!(
            unbalance_centrifugal_force(0.02, 0.001, 100.0),
            0.2,
            epsilon = 1e-12
        );
        // Doubler ω quadruple la force.
        let f1 = unbalance_centrifugal_force(0.02, 0.001, 100.0);
        let f2 = unbalance_centrifugal_force(0.02, 0.001, 200.0);
        assert_relative_eq!(f2 / f1, 4.0, epsilon = 1e-12);
    }

    #[test]
    fn forced_amplitude_at_resonance() {
        // À r=1 : X = (m·e/M)/(2ζ). Avec m=0,5 kg, e=0,002 m, M=50 kg, ζ=0,05 :
        // m·e/M = 2e-5 m, /(2·0,05)=0,1 → X = 2e-4 m.
        let x = unbalance_forced_amplitude(0.5, 0.002, 50.0, 1.0, 0.05);
        assert_relative_eq!(x, 2.0e-4, epsilon = 1e-12);
    }

    #[test]
    fn forced_amplitude_tends_to_me_over_m_at_high_ratio() {
        // Quand r → ∞, X → m·e/M (la masse suit le balourd).
        let limit = 0.5 * 0.002 / 50.0; // = 2e-5 m
        let x = unbalance_forced_amplitude(0.5, 0.002, 50.0, 1.0e4, 0.05);
        assert_relative_eq!(x, limit, epsilon = 1e-10);
    }

    #[test]
    fn transmissibility_is_unity_at_ratio_sqrt2() {
        // Identité : TR = 1 pour r = √2 quel que soit l'amortissement.
        let force = 123.0;
        let r = 2.0_f64.sqrt();
        assert_relative_eq!(
            unbalance_transmitted_force(force, r, 0.05),
            force,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            unbalance_transmitted_force(force, r, 0.30),
            force,
            epsilon = 1e-12
        );
    }

    #[test]
    fn permissible_eccentricity_reciprocity() {
        // Par définition e_per·ω = G (réciprocité ISO 1940).
        let omega = 100.0 * PI;
        let e_per = unbalance_permissible_from_grade(6.3, omega);
        assert_relative_eq!(e_per * omega, 6.3, epsilon = 1e-12);
        // G6.3 à ω=100π rad/s → e_per = 6,3/(100π) ≈ 0,020053 mm.
        assert_relative_eq!(e_per, 6.3 / (100.0 * PI), epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "la vitesse de rotation doit être strictement positive")]
    fn permissible_panics_on_nonpositive_speed() {
        let _ = unbalance_permissible_from_grade(6.3, 0.0);
    }
}
