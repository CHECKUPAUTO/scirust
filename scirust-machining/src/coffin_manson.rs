//! Fatigue oligocyclique — relation de **Coffin-Manson** (déformation-durée) :
//! amplitudes de déformation plastique et élastique (**Basquin**), courbe totale
//! déformation-durée et durée de transition élastique/plastique.
//!
//! ```text
//! plastique    Δεp/2 = εf'·(2N)^c
//! élastique    Δεe/2 = (σf'/E)·(2N)^b        (Basquin)
//! totale       Δε/2  = (σf'/E)·(2N)^b + εf'·(2N)^c
//! transition   2Nt   = (εf'·E/σf')^(1/(b−c))
//! ```
//!
//! `σf'` coefficient de résistance en fatigue (Pa), `b` exposant de résistance en
//! fatigue (< 0, sans dimension), `εf'` coefficient de ductilité en fatigue (sans
//! dimension), `c` exposant de ductilité en fatigue (< 0, sans dimension), `E`
//! module d'Young (Pa), `2N` nombre de **renversements** à rupture (sans
//! dimension), amplitudes de déformation sans dimension. À `2Nt` les amplitudes
//! élastique et plastique s'égalisent : les cycles courts (`2N < 2Nt`) sont
//! dominés par le plastique, les cycles longs par l'élastique.
//!
//! **Convention** : unités SI cohérentes (`σf'` et `E` dans la même unité de
//! contrainte). **Limite honnête** : loi déformation-durée (Coffin-Manson +
//! Basquin) ; les coefficients et exposants de fatigue (`σf'`, `b`, `εf'`, `c`)
//! et le module `E` sont **fournis par le matériau/l'appelant**, aucune valeur
//! n'est inventée. Approche en **déformation**, distincte de
//! [`crate::fatigue_mean_stress`] (approche en **contrainte**).

/// Amplitude de déformation **plastique** `Δεp/2 = εf'·(2N)^c`.
///
/// Panique si `reversals_to_failure <= 0`.
pub fn coffin_plastic_strain_amplitude(
    fatigue_ductility_coefficient: f64,
    reversals_to_failure: f64,
    fatigue_ductility_exponent: f64,
) -> f64 {
    assert!(
        reversals_to_failure > 0.0,
        "le nombre de renversements 2N doit être strictement positif"
    );
    fatigue_ductility_coefficient * reversals_to_failure.powf(fatigue_ductility_exponent)
}

/// Amplitude de déformation **élastique** (Basquin) `Δεe/2 = (σf'/E)·(2N)^b`.
///
/// Panique si `youngs_modulus <= 0` ou `reversals_to_failure <= 0`.
pub fn coffin_elastic_strain_amplitude(
    fatigue_strength_coefficient: f64,
    youngs_modulus: f64,
    reversals_to_failure: f64,
    fatigue_strength_exponent: f64,
) -> f64 {
    assert!(
        youngs_modulus > 0.0,
        "le module d'Young E doit être strictement positif"
    );
    assert!(
        reversals_to_failure > 0.0,
        "le nombre de renversements 2N doit être strictement positif"
    );
    (fatigue_strength_coefficient / youngs_modulus)
        * reversals_to_failure.powf(fatigue_strength_exponent)
}

/// Amplitude de déformation **totale** `Δε/2 = (σf'/E)·(2N)^b + εf'·(2N)^c`.
///
/// Somme des termes élastique (Basquin) et plastique (Coffin-Manson).
///
/// Panique si `youngs_modulus <= 0` ou `reversals_to_failure <= 0`.
pub fn coffin_total_strain_amplitude(
    fatigue_strength_coefficient: f64,
    youngs_modulus: f64,
    fatigue_strength_exponent: f64,
    fatigue_ductility_coefficient: f64,
    fatigue_ductility_exponent: f64,
    reversals_to_failure: f64,
) -> f64 {
    coffin_elastic_strain_amplitude(
        fatigue_strength_coefficient,
        youngs_modulus,
        reversals_to_failure,
        fatigue_strength_exponent,
    ) + coffin_plastic_strain_amplitude(
        fatigue_ductility_coefficient,
        reversals_to_failure,
        fatigue_ductility_exponent,
    )
}

/// Durée de **transition** `2Nt = (εf'·E/σf')^(1/(b−c))` (renversements où les
/// amplitudes élastique et plastique sont égales).
///
/// Panique si `σf' <= 0`, `youngs_modulus <= 0`, `εf' <= 0` ou `b == c`.
pub fn coffin_transition_reversals(
    fatigue_strength_coefficient: f64,
    youngs_modulus: f64,
    fatigue_strength_exponent: f64,
    fatigue_ductility_coefficient: f64,
    fatigue_ductility_exponent: f64,
) -> f64 {
    assert!(
        fatigue_strength_coefficient > 0.0,
        "le coefficient de résistance σf' doit être strictement positif"
    );
    assert!(
        youngs_modulus > 0.0,
        "le module d'Young E doit être strictement positif"
    );
    assert!(
        fatigue_ductility_coefficient > 0.0,
        "le coefficient de ductilité εf' doit être strictement positif"
    );
    assert!(
        fatigue_strength_exponent != fatigue_ductility_exponent,
        "les exposants b et c doivent différer (b ≠ c)"
    );
    let base = fatigue_ductility_coefficient * youngs_modulus / fatigue_strength_coefficient;
    base.powf(1.0 / (fatigue_strength_exponent - fatigue_ductility_exponent))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn plastic_amplitude_clean_case() {
        // εf'=0,5 ; c=−0,5 ; 2N=100 → 0,5·100^(−1/2) = 0,5/10 = 0,05.
        let (ef_prime, c, two_n) = (0.5_f64, -0.5_f64, 100.0_f64);
        assert_relative_eq!(
            coffin_plastic_strain_amplitude(ef_prime, two_n, c),
            0.05,
            epsilon = 1e-12
        );
    }

    #[test]
    fn elastic_amplitude_clean_case() {
        // σf'=2000 ; E=200000 ; b=−0,5 ; 2N=100 → (2000/200000)·100^(−1/2)
        //  = 0,01·0,1 = 0,001.
        let (sf_prime, e_mod, two_n, b) = (2000.0_f64, 200_000.0_f64, 100.0_f64, -0.5_f64);
        assert_relative_eq!(
            coffin_elastic_strain_amplitude(sf_prime, e_mod, two_n, b),
            0.001,
            epsilon = 1e-12
        );
    }

    #[test]
    fn total_is_sum_of_elastic_and_plastic() {
        // La totale doit valoir exactement élastique + plastique.
        let (sf_prime, e_mod, b) = (1000.0_f64, 200_000.0_f64, -0.09_f64);
        let (ef_prime, c) = (0.5_f64, -0.6_f64);
        let two_n = 5000.0_f64;
        let el = coffin_elastic_strain_amplitude(sf_prime, e_mod, two_n, b);
        let pl = coffin_plastic_strain_amplitude(ef_prime, two_n, c);
        assert_relative_eq!(
            coffin_total_strain_amplitude(sf_prime, e_mod, b, ef_prime, c, two_n),
            el + pl,
            epsilon = 1e-15
        );
    }

    #[test]
    fn at_transition_elastic_equals_plastic() {
        // Par construction, à 2Nt les deux amplitudes coïncident.
        let (sf_prime, e_mod, b) = (1000.0_f64, 200_000.0_f64, -0.09_f64);
        let (ef_prime, c) = (0.5_f64, -0.6_f64);
        let two_nt = coffin_transition_reversals(sf_prime, e_mod, b, ef_prime, c);
        let el = coffin_elastic_strain_amplitude(sf_prime, e_mod, two_nt, b);
        let pl = coffin_plastic_strain_amplitude(ef_prime, two_nt, c);
        assert_relative_eq!(el, pl, epsilon = 1e-10);
    }

    #[test]
    fn total_at_transition_is_twice_elastic() {
        // À 2Nt : totale = élastique + plastique = 2·élastique.
        let (sf_prime, e_mod, b) = (1200.0_f64, 210_000.0_f64, -0.10_f64);
        let (ef_prime, c) = (0.4_f64, -0.55_f64);
        let two_nt = coffin_transition_reversals(sf_prime, e_mod, b, ef_prime, c);
        let el = coffin_elastic_strain_amplitude(sf_prime, e_mod, two_nt, b);
        let total = coffin_total_strain_amplitude(sf_prime, e_mod, b, ef_prime, c, two_nt);
        assert_relative_eq!(total, 2.0 * el, epsilon = 1e-10);
    }

    #[test]
    fn plastic_amplitude_scales_with_ductility_coefficient() {
        // Δεp/2 est linéaire en εf' : doubler εf' double l'amplitude.
        let (c, two_n) = (-0.6_f64, 2000.0_f64);
        let base = coffin_plastic_strain_amplitude(0.5, two_n, c);
        let doubled = coffin_plastic_strain_amplitude(1.0, two_n, c);
        assert_relative_eq!(doubled, 2.0 * base, epsilon = 1e-15);
    }

    #[test]
    #[should_panic(expected = "2N doit être strictement positif")]
    fn nonpositive_reversals_panics() {
        coffin_plastic_strain_amplitude(0.5, 0.0, -0.6);
    }
}
