//! Serrage d'un boulon — relation empirique **couple-tension** `T = K·d·F` et
//! grandeurs associées (précharge cible au proof, perte de serrage par tassement).
//!
//! ```text
//! couple de serrage      T  = K·d·F
//! précharge au couple    F  = T/(K·d)
//! précharge cible (proof) Fi = frac·Sp·At
//! serrage après tassement Fr = F0·(1 − r)
//! ```
//!
//! `T` couple de serrage (N·m), `K` coefficient de serrage (*nut factor*, sans
//! dimension, ~0,2 usuel), `d` diamètre nominal (m), `F`/`F0` précharge (N), `frac`
//! fraction du proof visée (sans dimension, ~0,75 usuel), `Sp` limite au proof du
//! boulon (Pa), `At` section résistante en traction (m²), `Fi` précharge cible (N),
//! `r` fraction de précharge perdue au tassement (sans dimension), `Fr` serrage
//! résiduel (N).
//!
//! **Convention** : SI cohérent (N, m, Pa, N·m), efforts de traction positifs.
//!
//! **Limite honnête** : relation **empirique** `T = K·d·F` ; le coefficient de
//! serrage `K` (*nut factor*) dépend fortement de la lubrification, de l'état de
//! surface et du procédé — il est très dispersé et **FOURNI par l'appelant**, ce
//! module n'invente aucune valeur « par défaut » de `K`, de fraction du proof, de
//! `Sp`, de `At` ni de taux de tassement. Modèle **élastique** qui néglige la
//! torsion résiduelle dans la vis et toute plastification locale.

/// Couple de serrage `T = K·d·F` (N·m).
///
/// `nut_factor` = `K` (sans dimension), `nominal_diameter` = `d` (m),
/// `preload` = `F` (N).
///
/// Panique si `nut_factor <= 0`, si `nominal_diameter <= 0` ou si `preload < 0`.
pub fn bolt_tightening_torque(nut_factor: f64, nominal_diameter: f64, preload: f64) -> f64 {
    assert!(
        nut_factor > 0.0,
        "le coefficient de serrage K doit être strictement positif"
    );
    assert!(
        nominal_diameter > 0.0,
        "le diamètre nominal doit être strictement positif"
    );
    assert!(preload >= 0.0, "la précharge doit être positive ou nulle");
    nut_factor * nominal_diameter * preload
}

/// Précharge induite par le couple `F = T/(K·d)` (N).
///
/// `torque` = `T` (N·m), `nut_factor` = `K` (sans dimension),
/// `nominal_diameter` = `d` (m).
///
/// Panique si `torque < 0`, si `nut_factor <= 0` ou si `nominal_diameter <= 0`.
pub fn bolt_preload_from_torque(torque: f64, nut_factor: f64, nominal_diameter: f64) -> f64 {
    assert!(
        torque >= 0.0,
        "le couple de serrage doit être positif ou nul"
    );
    assert!(
        nut_factor > 0.0,
        "le coefficient de serrage K doit être strictement positif"
    );
    assert!(
        nominal_diameter > 0.0,
        "le diamètre nominal doit être strictement positif"
    );
    torque / (nut_factor * nominal_diameter)
}

/// Précharge cible tirée du proof `Fi = frac·Sp·At` (N).
///
/// `proof_strength` = `Sp` (Pa), `tensile_stress_area` = `At` (m²),
/// `preload_fraction` = `frac` (sans dimension, ~0,75, fraction du proof visée).
///
/// Panique si `proof_strength < 0`, si `tensile_stress_area < 0` ou si
/// `preload_fraction` n'est pas dans `[0, 1]`.
pub fn bolt_preload_from_yield(
    proof_strength: f64,
    tensile_stress_area: f64,
    preload_fraction: f64,
) -> f64 {
    assert!(
        proof_strength >= 0.0,
        "la limite au proof doit être positive ou nulle"
    );
    assert!(
        tensile_stress_area >= 0.0,
        "la section résistante doit être positive ou nulle"
    );
    assert!(
        (0.0..=1.0).contains(&preload_fraction),
        "la fraction du proof doit être comprise entre 0 et 1"
    );
    preload_fraction * proof_strength * tensile_stress_area
}

/// Serrage résiduel après tassement `Fr = F0·(1 − r)` (N).
///
/// `initial_preload` = `F0` (N), `relaxation_fraction` = `r` (sans dimension,
/// fraction de précharge perdue au tassement).
///
/// Panique si `initial_preload < 0` ou si `relaxation_fraction` n'est pas dans
/// `[0, 1]`.
pub fn bolt_clamp_force_after_relaxation(initial_preload: f64, relaxation_fraction: f64) -> f64 {
    assert!(
        initial_preload >= 0.0,
        "la précharge initiale doit être positive ou nulle"
    );
    assert!(
        (0.0..=1.0).contains(&relaxation_fraction),
        "la fraction de tassement doit être comprise entre 0 et 1"
    );
    initial_preload * (1.0 - relaxation_fraction)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn tightening_torque_reference_case() {
        // M12 (d=0,012 m), K=0,2, F=15 000 N → T = 0,2·0,012·15 000 = 36 N·m.
        assert_relative_eq!(
            bolt_tightening_torque(0.2, 0.012, 15_000.0),
            36.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn torque_and_preload_are_reciprocal() {
        // Réciprocité : F → T = K·d·F → F retrouvé par T/(K·d).
        let k = 0.18_f64;
        let d = 0.016_f64;
        let f = 22_500.0_f64;
        let t = bolt_tightening_torque(k, d, f);
        assert_relative_eq!(bolt_preload_from_torque(t, k, d), f, epsilon = 1e-6);
    }

    #[test]
    fn torque_scales_linearly_with_preload() {
        // Proportionnalité : doubler la précharge double le couple (K, d fixés).
        let t1 = bolt_tightening_torque(0.2, 0.010, 10_000.0);
        let t2 = bolt_tightening_torque(0.2, 0.010, 20_000.0);
        assert_relative_eq!(t2, 2.0 * t1, epsilon = 1e-9);
    }

    #[test]
    fn preload_from_yield_reference_case() {
        // frac=0,75, Sp=600e6 Pa, At=100e-6 m² → Fi = 0,75·600e6·100e-6 = 45 000 N.
        assert_relative_eq!(
            bolt_preload_from_yield(600e6, 100e-6, 0.75),
            45_000.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn relaxation_limits_are_consistent() {
        // r=0 : aucun tassement, le serrage vaut la précharge initiale.
        assert_relative_eq!(
            bolt_clamp_force_after_relaxation(45_000.0, 0.0),
            45_000.0,
            epsilon = 1e-9
        );
        // r=1 : perte totale, serrage nul.
        assert_relative_eq!(
            bolt_clamp_force_after_relaxation(45_000.0, 1.0),
            0.0,
            epsilon = 1e-9
        );
        // r=0,1 : Fr = 45 000·0,9 = 40 500 N.
        assert_relative_eq!(
            bolt_clamp_force_after_relaxation(45_000.0, 0.1),
            40_500.0,
            epsilon = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "fraction du proof doit être comprise entre 0 et 1")]
    fn yield_fraction_above_one_panics() {
        bolt_preload_from_yield(600e6, 100e-6, 1.5);
    }
}
