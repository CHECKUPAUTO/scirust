//! Réacteur **piston** (PFR) — dimensionnement par intégration de la loi de
//! vitesse le long de l'écoulement, pour des réactions d'ordre 1 et 2 à densité
//! constante.
//!
//! ```text
//! conversion ordre 1  X  = 1 − exp(−k·τ)                 [sans dimension]
//! τ requis ordre 1    τ  = −ln(1 − X) / k                [s]
//! volume ordre 1      V  = v̇₀·τ = v̇₀·(−ln(1 − X)/k)      [m³]
//! τ requis ordre 2    τ  = X / (k·C_A0·(1 − X))          [s]
//! ```
//!
//! `X` taux de conversion de A [sans dimension, dans `[0, 1[`], `k` constante
//! cinétique — du premier ordre [1/s], du second ordre [m³/(mol·s)] —, `τ` temps
//! de passage (space time) [s], `v̇₀` débit volumétrique d'alimentation [m³/s],
//! `V` volume utile du réacteur [m³], `C_A0` concentration d'entrée du réactif A
//! [mol/m³]. Le groupe `k·τ` est le nombre de Damköhler `Da` [sans dimension]
//! pour l'ordre 1.
//!
//! **Limite honnête** : l'écoulement est supposé **piston** (pas de dispersion
//! axiale, profil radial de vitesse plat, mélange radial parfait mais aucun
//! mélange axial), en **régime permanent** et à **densité constante**
//! (`v̇` invariant, pas de variation du nombre de moles ni de la température). La
//! constante de vitesse `k`, la concentration d'entrée `C_A0`, ainsi que les
//! enthalpies, volatilités, coefficients de partage et diffusivités éventuels
//! sont **fournis par l'appelant** (loi de vitesse mesurée, corrélation, table) :
//! aucune valeur « par défaut » n'est inventée. Ces fonctions ne traitent **ni
//! les réactions multiples ni le non-isotherme**. À conversion et cinétique
//! d'ordre positif identiques, un PFR est plus **compact** qu'un CSTR : la
//! comparaison des deux volumes est laissée à l'appelant.

/// Conversion d'une réaction d'**ordre 1** à densité constante dans un PFR
/// `X = 1 − exp(−k·τ)` [sans dimension].
///
/// `rate_constant` `k` constante cinétique du premier ordre [1/s], `space_time`
/// `τ` temps de passage [s]. Le produit `k·τ = Da` est le nombre de Damköhler ;
/// `X → 1` quand `Da → ∞`.
///
/// Panique si `rate_constant` ou `space_time` est négatif ou non fini.
pub fn pfr_first_order_conversion(rate_constant: f64, space_time: f64) -> f64 {
    assert!(
        rate_constant.is_finite() && rate_constant >= 0.0,
        "la constante cinétique doit être finie et positive ou nulle (1/s)"
    );
    assert!(
        space_time.is_finite() && space_time >= 0.0,
        "le temps de passage doit être fini et positif ou nul (s)"
    );
    1.0 - (-rate_constant * space_time).exp()
}

/// Temps de passage **requis** pour atteindre une conversion donnée en ordre 1
/// `τ = −ln(1 − X) / k` [s] — réciproque de [`pfr_first_order_conversion`].
///
/// `conversion` `X` conversion visée [sans dimension, dans `[0, 1[`],
/// `rate_constant` `k` constante cinétique du premier ordre [1/s]. `τ → ∞` quand
/// `X → 1` : la conversion complète exige un réacteur infini.
///
/// Panique si `conversion` sort de `[0, 1[` (la borne 1 fait diverger le
/// logarithme), ou si `rate_constant` n'est pas strictement positif (division).
pub fn pfr_space_time_first_order(conversion: f64, rate_constant: f64) -> f64 {
    assert!(
        conversion.is_finite() && (0.0..1.0).contains(&conversion),
        "la conversion doit être comprise dans [0, 1[ (strictement inférieure à 1)"
    );
    assert!(
        rate_constant > 0.0,
        "la constante cinétique doit être strictement positive (1/s)"
    );
    -(1.0 - conversion).ln() / rate_constant
}

/// Volume utile d'un PFR pour une réaction d'**ordre 1** à densité constante
/// `V = v̇₀·τ = v̇₀·(−ln(1 − X)/k)` [m³].
///
/// `volumetric_flow` `v̇₀` débit volumétrique d'alimentation [m³/s],
/// `conversion` `X` conversion visée [sans dimension, dans `[0, 1[`],
/// `rate_constant` `k` constante cinétique du premier ordre [1/s]. Le volume est
/// proportionnel au débit volumétrique à conversion et cinétique fixées.
///
/// Panique si `volumetric_flow` est négatif ou non fini, si `conversion` sort de
/// `[0, 1[`, ou si `rate_constant` n'est pas strictement positif (via
/// [`pfr_space_time_first_order`]).
pub fn pfr_volume_first_order(volumetric_flow: f64, conversion: f64, rate_constant: f64) -> f64 {
    assert!(
        volumetric_flow.is_finite() && volumetric_flow >= 0.0,
        "le débit volumétrique doit être fini et positif ou nul (m³/s)"
    );
    volumetric_flow * pfr_space_time_first_order(conversion, rate_constant)
}

/// Temps de passage **requis** pour une réaction d'**ordre 2** à densité
/// constante `τ = X / (k·C_A0·(1 − X))` [s].
///
/// `conversion` `X` conversion visée [sans dimension, dans `[0, 1[`],
/// `rate_constant` `k` constante cinétique du second ordre [m³/(mol·s)],
/// `inlet_concentration` `C_A0` concentration d'entrée du réactif A [mol/m³]. À
/// `k` et `C_A0` fixés, `τ` croît comme `X/(1 − X)` et diverge quand `X → 1`.
///
/// Panique si `conversion` sort de `[0, 1[` (la borne 1 fait diverger la
/// division), si `rate_constant` n'est pas strictement positif, ou si
/// `inlet_concentration` n'est pas strictement positive.
pub fn pfr_second_order_space_time(
    conversion: f64,
    rate_constant: f64,
    inlet_concentration: f64,
) -> f64 {
    assert!(
        conversion.is_finite() && (0.0..1.0).contains(&conversion),
        "la conversion doit être comprise dans [0, 1[ (strictement inférieure à 1)"
    );
    assert!(
        rate_constant > 0.0,
        "la constante cinétique doit être strictement positive (m³/(mol·s))"
    );
    assert!(
        inlet_concentration > 0.0,
        "la concentration d'entrée doit être strictement positive (mol/m³)"
    );
    conversion / (rate_constant * inlet_concentration * (1.0 - conversion))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::LN_2;

    #[test]
    fn first_order_conversion_known_case() {
        // k = ln2 1/s, τ = 1 s → Da = ln2 → X = 1 − exp(−ln2) = 1 − 0,5 = 0,5.
        assert_relative_eq!(pfr_first_order_conversion(LN_2, 1.0), 0.5, epsilon = 1e-12);
    }

    #[test]
    fn zero_damkohler_gives_zero_conversion() {
        // Cas limite : k = 0 (pas de réaction) → Da = 0 → X = 1 − exp(0) = 0.
        assert_relative_eq!(pfr_first_order_conversion(0.0, 10.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn space_time_first_order_known_case() {
        // X = 0,5, k = ln2 1/s → τ = −ln(0,5)/ln2 = ln2/ln2 = 1 s.
        assert_relative_eq!(pfr_space_time_first_order(0.5, LN_2), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn conversion_and_space_time_are_reciprocal() {
        // τ → X → τ doit revenir au point de départ (réciprocité exacte).
        let k = 0.5_f64;
        let tau = 4.0_f64;
        let x = pfr_first_order_conversion(k, tau);
        let tau_back = pfr_space_time_first_order(x, k);
        assert_relative_eq!(tau_back, tau, epsilon = 1e-9);
    }

    #[test]
    fn volume_is_flow_times_space_time() {
        // V = v̇₀·τ : à conversion et k fixés, V est proportionnel au débit.
        // v̇₀ = 2 m³/s, X = 0,5, k = ln2 → τ = 1 s → V = 2 m³.
        let tau = pfr_space_time_first_order(0.5, LN_2);
        assert_relative_eq!(
            pfr_volume_first_order(2.0, 0.5, LN_2),
            2.0 * tau,
            epsilon = 1e-12
        );
        assert_relative_eq!(pfr_volume_first_order(2.0, 0.5, LN_2), 2.0, epsilon = 1e-12);
    }

    #[test]
    fn second_order_space_time_known_case() {
        // X = 0,5, k = 2 m³/(mol·s), C_A0 = 1 mol/m³
        // → τ = 0,5/(2·1·(1 − 0,5)) = 0,5/1 = 0,5 s.
        assert_relative_eq!(
            pfr_second_order_space_time(0.5, 2.0, 1.0),
            0.5,
            epsilon = 1e-12
        );
    }

    #[test]
    fn second_order_space_time_scales_with_conversion_group() {
        // À k et C_A0 fixés, τ croît comme X/(1 − X) : passer de X = 0,5
        // (groupe 1) à X = 0,75 (groupe 3) triple τ.
        let tau_half = pfr_second_order_space_time(0.5, 2.0, 1.0);
        let tau_three_quarters = pfr_second_order_space_time(0.75, 2.0, 1.0);
        assert_relative_eq!(tau_three_quarters, 3.0 * tau_half, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "strictement inférieure à 1")]
    fn full_conversion_space_time_panics() {
        // X = 1 fait diverger τ = −ln(1 − X)/k : entrée rejetée.
        pfr_space_time_first_order(1.0, 0.5);
    }
}
