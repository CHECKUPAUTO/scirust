//! Acoustique — **niveaux de pression sonore** exprimés en décibels (dB), pour
//! le bruit des machines et postes d'usinage.
//!
//! ```text
//! niveau               L    = 20·log10(p / p_ref)        [dB]
//! pression (inverse)   p    = p_ref · 10^(L / 20)        [Pa]
//! somme énergétique    L₊   = 10·log10(10^(L₁/10) + 10^(L₂/10))
//! atténuation champ    L(d) = L_ref − 20·log10(d / d_ref)
//!  libre (source ponctuelle)
//! ```
//!
//! `p` pression acoustique efficace (RMS) [Pa], `p_ref` pression de référence
//! [Pa] (20 µPa dans l'air), `L` niveau de pression sonore [dB], `L₁`/`L₂`
//! niveaux de deux sources [dB], `L₊` niveau résultant [dB], `L_ref` niveau
//! mesuré à la distance `d_ref` [dB], `d`/`d_ref` distances à la source [m].
//!
//! **Limite honnête** : la pression de référence `p_ref` (usuellement 20 µPa
//! dans l'air, mais 1 µPa dans l'eau) est **fournie par l'appelant** ; aucune
//! valeur n'est supposée par défaut. La somme énergétique suppose des sources
//! **incohérentes** (pas de relation de phase — on additionne les énergies, pas
//! les pressions). L'atténuation est celle du **champ libre** pour une **source
//! ponctuelle** (−6 dB par doublement de distance), sans absorption
//! atmosphérique, réflexions ni directivité, tous fournis à part par l'appelant.

/// Niveau de pression sonore `L = 20·log10(p / p_ref)` [dB].
///
/// `rms_pressure` et `reference_pressure` en pascals (Pa) ; le niveau est en dB.
///
/// Panique si `rms_pressure <= 0` ou `reference_pressure <= 0`.
pub fn spl_from_pressure(rms_pressure: f64, reference_pressure: f64) -> f64 {
    assert!(
        rms_pressure > 0.0,
        "la pression efficace doit être strictement positive (Pa)"
    );
    assert!(
        reference_pressure > 0.0,
        "la pression de référence doit être strictement positive (Pa)"
    );
    20.0 * (rms_pressure / reference_pressure).log10()
}

/// Pression acoustique efficace `p = p_ref · 10^(L / 20)` [Pa] (inverse de
/// [`spl_from_pressure`]).
///
/// `spl` en dB, `reference_pressure` en Pa ; la pression rendue est en Pa.
///
/// Panique si `reference_pressure <= 0` ou si `spl` n'est pas fini.
pub fn spl_to_pressure(spl: f64, reference_pressure: f64) -> f64 {
    assert!(
        reference_pressure > 0.0,
        "la pression de référence doit être strictement positive (Pa)"
    );
    assert!(spl.is_finite(), "le niveau doit être fini (dB)");
    reference_pressure * 10.0_f64.powf(spl / 20.0)
}

/// Addition **énergétique** de deux sources incohérentes
/// `L₊ = 10·log10(10^(L₁/10) + 10^(L₂/10))` [dB].
///
/// `spl1`, `spl2` et le résultat sont en dB.
///
/// Panique si `spl1` ou `spl2` n'est pas fini.
pub fn spl_sum_two_sources(spl1: f64, spl2: f64) -> f64 {
    assert!(spl1.is_finite(), "L₁ doit être fini (dB)");
    assert!(spl2.is_finite(), "L₂ doit être fini (dB)");
    10.0 * (10.0_f64.powf(spl1 / 10.0) + 10.0_f64.powf(spl2 / 10.0)).log10()
}

/// Atténuation en champ libre d'une source ponctuelle
/// `L(d) = L_ref − 20·log10(d / d_ref)` [dB].
///
/// `spl_reference` en dB à la distance `distance_ref` ; `distance` et
/// `distance_ref` en mètres ; le résultat est en dB.
///
/// Panique si `distance <= 0`, `distance_ref <= 0` ou `spl_reference` non fini.
pub fn spl_distance_attenuation(spl_reference: f64, distance_ref: f64, distance: f64) -> f64 {
    assert!(
        distance > 0.0,
        "la distance doit être strictement positive (m)"
    );
    assert!(
        distance_ref > 0.0,
        "la distance de référence doit être strictement positive (m)"
    );
    assert!(
        spl_reference.is_finite(),
        "le niveau de référence doit être fini (dB)"
    );
    spl_reference - 20.0 * (distance / distance_ref).log10()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const P_REF_AIR: f64 = 20e-6; // 20 µPa, référence de l'air.

    #[test]
    fn reference_pressure_gives_zero_db() {
        // p = p_ref  ⇒  L = 20·log10(1) = 0 dB.
        assert_relative_eq!(
            spl_from_pressure(P_REF_AIR, P_REF_AIR),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn known_reference_one_pascal() {
        // 1 Pa RMS ⇒ 20·log10(1/20e-6) = 20·log10(50000) ≈ 93,979 dB.
        assert_relative_eq!(
            spl_from_pressure(1.0, P_REF_AIR),
            93.979_400_09,
            epsilon = 1e-6
        );
        // 2 Pa RMS ⇒ 20·log10(100000) = 100 dB exactement.
        assert_relative_eq!(spl_from_pressure(2.0, P_REF_AIR), 100.0, epsilon = 1e-9);
    }

    #[test]
    fn pressure_and_level_are_reciprocal() {
        // spl_to_pressure ∘ spl_from_pressure = identité.
        let p = 0.6; // Pa
        let level = spl_from_pressure(p, P_REF_AIR);
        assert_relative_eq!(spl_to_pressure(level, P_REF_AIR), p, epsilon = 1e-12);
    }

    #[test]
    fn two_equal_sources_add_three_db() {
        // Deux sources identiques : L₊ = L + 10·log10(2) ≈ L + 3,0103 dB.
        let l = 80.0;
        let expected = l + 10.0 * 2.0_f64.log10();
        assert_relative_eq!(spl_sum_two_sources(l, l), expected, epsilon = 1e-12);
    }

    #[test]
    fn doubling_distance_drops_six_db() {
        // Champ libre : chaque doublement de distance retire 20·log10(2) ≈ 6,0206 dB.
        let l_ref = 94.0;
        let drop = l_ref - spl_distance_attenuation(l_ref, 1.0, 2.0);
        assert_relative_eq!(drop, 20.0 * 2.0_f64.log10(), epsilon = 1e-12);
        // Cas chiffré : 94 dB à 1 m ⇒ 74 dB à 10 m (−20 dB par décade).
        assert_relative_eq!(
            spl_distance_attenuation(94.0, 1.0, 10.0),
            74.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn distant_source_much_quieter_than_dominant() {
        // Une source 20 dB sous l'autre n'ajoute presque rien (≈ +0,04 dB).
        let sum = spl_sum_two_sources(90.0, 70.0);
        assert!(sum > 90.0 && sum < 90.05);
    }

    #[test]
    #[should_panic(expected = "strictement positive")]
    fn zero_pressure_panics() {
        spl_from_pressure(0.0, P_REF_AIR);
    }
}
