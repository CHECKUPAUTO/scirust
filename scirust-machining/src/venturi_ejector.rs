//! Éjecteur / trompe à **Venturi** : entraînement d'un fluide aspiré par la
//! détente d'un fluide moteur, produisant un vide au col (effet Venturi).
//!
//! ```text
//! rapport d'entraînement  ω = ṁ_asp / ṁ_mot
//! vitesse au col          v_col = √( 2·(p_mot − p_aval) / ρ_mot )   (Bernoulli, moteur incompressible)
//! pression de vide        p_abs = p_atm·(1 − f_vide)
//! débit massique mélangé  ṁ_mix = ṁ_mot + ṁ_asp
//! ```
//!
//! `ṁ_mot` débit massique moteur (kg/s), `ṁ_asp` débit massique aspiré (kg/s),
//! `ω` rapport d'entraînement (sans dimension), `p_mot` pression motrice amont
//! (Pa), `p_aval` pression aval au col (Pa), `ρ_mot` masse volumique du fluide
//! moteur (kg/m³), `v_col` vitesse au col (m/s), `p_atm` pression atmosphérique
//! absolue (Pa), `f_vide` fraction de vide (sans dimension, `[0, 1]`), `p_abs`
//! pression absolue générée (Pa).
//!
//! **Convention** : SI cohérent (kg/s, Pa, kg/m³, m/s). **Limite honnête** :
//! fluide moteur **incompressible** (Bernoulli au col) ; le rapport
//! d'entraînement `ω` et le niveau de vide `f_vide` dépendent de la **géométrie**
//! et du **régime** et sont **fournis** par l'appelant — jamais des valeurs
//! inventées. Mélange **idéal** : ni récupération de pression du diffuseur, ni
//! écoulement compressible ou sonique ne sont modélisés.

/// Rapport d'entraînement `ω = ṁ_asp / ṁ_mot` (sans dimension).
///
/// Panique si `motive_mass_flow <= 0` ou `suction_mass_flow < 0`.
pub fn ejector_entrainment_ratio(suction_mass_flow: f64, motive_mass_flow: f64) -> f64 {
    assert!(
        motive_mass_flow > 0.0,
        "le débit massique moteur doit être > 0"
    );
    assert!(
        suction_mass_flow >= 0.0,
        "le débit massique aspiré doit être ≥ 0"
    );
    suction_mass_flow / motive_mass_flow
}

/// Vitesse au col `v_col = √( 2·(p_mot − p_aval) / ρ_mot )` (m/s) — Bernoulli, fluide moteur incompressible.
///
/// Panique si `motive_density <= 0` ou si `motive_pressure < downstream_pressure`.
pub fn ejector_throat_velocity(
    motive_pressure: f64,
    downstream_pressure: f64,
    motive_density: f64,
) -> f64 {
    assert!(
        motive_density > 0.0,
        "la masse volumique du fluide moteur doit être > 0"
    );
    assert!(
        motive_pressure >= downstream_pressure,
        "la pression motrice doit être ≥ la pression aval"
    );
    (2.0_f64 * (motive_pressure - downstream_pressure) / motive_density).sqrt()
}

/// Pression absolue générée `p_abs = p_atm·(1 − f_vide)` (Pa).
///
/// Panique si `atmospheric_pressure < 0` ou si `vacuum_level_fraction` hors `[0, 1]`.
pub fn ejector_vacuum_pressure(atmospheric_pressure: f64, vacuum_level_fraction: f64) -> f64 {
    assert!(
        atmospheric_pressure >= 0.0,
        "la pression atmosphérique doit être ≥ 0"
    );
    assert!(
        (0.0..=1.0).contains(&vacuum_level_fraction),
        "la fraction de vide doit être dans [0, 1]"
    );
    atmospheric_pressure * (1.0 - vacuum_level_fraction)
}

/// Débit massique du mélange `ṁ_mix = ṁ_mot + ṁ_asp` (kg/s) — conservation de la masse.
///
/// Panique si `motive_mass_flow < 0` ou `suction_mass_flow < 0`.
pub fn ejector_mixed_mass_flow(motive_mass_flow: f64, suction_mass_flow: f64) -> f64 {
    assert!(
        motive_mass_flow >= 0.0,
        "le débit massique moteur doit être ≥ 0"
    );
    assert!(
        suction_mass_flow >= 0.0,
        "le débit massique aspiré doit être ≥ 0"
    );
    motive_mass_flow + suction_mass_flow
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn entrainment_ratio_is_the_flow_quotient() {
        // ṁ_asp = 0,4 kg/s, ṁ_mot = 2,0 kg/s → ω = 0,2.
        assert_relative_eq!(ejector_entrainment_ratio(0.4, 2.0), 0.2, epsilon = 1e-12);
    }

    #[test]
    fn zero_suction_gives_zero_entrainment() {
        // Sans aspiration, ω = 0 quel que soit le débit moteur.
        assert_relative_eq!(ejector_entrainment_ratio(0.0, 3.5), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn throat_velocity_matches_bernoulli() {
        // p_mot = 500 000 Pa, p_aval = 100 000 Pa, ρ_mot = 1000 kg/m³.
        //   v_col = √(2·(500000 − 100000)/1000) = √800 = 28,284271247461902 m/s
        let v = ejector_throat_velocity(500_000.0, 100_000.0, 1000.0);
        assert_relative_eq!(v, 28.284_271_247_461_902, max_relative = 1e-12);
    }

    #[test]
    fn throat_velocity_scales_with_root_pressure_gap() {
        // v_col ∝ √Δp : quadrupler l'écart de pression double la vitesse.
        let v1 = ejector_throat_velocity(200_000.0, 100_000.0, 998.0);
        let v2 = ejector_throat_velocity(500_000.0, 100_000.0, 998.0);
        assert_relative_eq!(v2 / v1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn vacuum_pressure_of_ninety_percent() {
        // p_atm = 101 325 Pa, f_vide = 0,9 → p_abs = 101325·0,1 = 10132,5 Pa.
        assert_relative_eq!(
            ejector_vacuum_pressure(101_325.0, 0.9),
            10_132.5,
            max_relative = 1e-12
        );
    }

    #[test]
    fn mixed_flow_conserves_mass() {
        // ṁ_mix = ṁ_mot + ṁ_asp, cohérent avec le rapport d'entraînement :
        // ṁ_mix = ṁ_mot·(1 + ω).
        let (motive, suction) = (2.0, 0.4);
        let omega = ejector_entrainment_ratio(suction, motive);
        assert_relative_eq!(
            ejector_mixed_mass_flow(motive, suction),
            motive * (1.0 + omega),
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "la pression motrice doit être ≥ la pression aval")]
    fn throat_velocity_panics_when_downstream_exceeds_motive() {
        ejector_throat_velocity(100_000.0, 150_000.0, 1000.0);
    }
}
