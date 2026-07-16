//! Réacteur **parfaitement agité continu** (CSTR) — dimensionnement par le
//! bilan matière en régime permanent et conversion d'une réaction d'ordre 1 à
//! densité constante.
//!
//! ```text
//! volume            V  = F_A0·X / (−r_A)            [m³]
//! temps de passage  τ  = V / v̇₀                     [s]
//! conversion ordre 1 X = k·τ / (1 + k·τ)            [sans dimension]
//! τ requis (ordre 1) τ = X / (k·(1 − X))            [s]
//! ```
//!
//! `V` volume utile du réacteur [m³], `F_A0` débit molaire d'alimentation du
//! réactif A [mol/s], `X` taux de conversion de A [sans dimension, dans `[0, 1]`],
//! `−r_A` vitesse de disparition de A **évaluée à la concentration de sortie**
//! [mol/(m³·s)], `v̇₀` débit volumétrique d'alimentation [m³/s], `τ` temps de
//! passage (space time) [s], `k` constante cinétique du premier ordre [1/s].
//! Le groupe `k·τ` est le nombre de Damköhler `Da` [sans dimension].
//!
//! **Limite honnête** : le CSTR est supposé **parfaitement agité** (composition
//! uniforme, donc égale à celle de la sortie), en **régime permanent** ; les
//! formules d'ordre 1 supposent en outre une **densité constante** (`v̇` invariant).
//! La vitesse `−r_A`, la constante cinétique `k`, les enthalpies, volatilités,
//! coefficients de partage et diffusivités sont **fournis par l'appelant** (loi de
//! vitesse mesurée, corrélation, table) : aucune valeur « par défaut » n'est
//! inventée, et `−r_A` doit être **évaluée à la concentration de sortie**. Ces
//! fonctions ne traitent **ni les réactions multiples ni le non-isotherme** ; le
//! bilan thermique (élévation de température, retrait de chaleur) est séparé.

/// Volume utile d'un CSTR par l'équation de dimensionnement
/// `V = F_A0·X / (−r_A)` [m³].
///
/// `molar_feed_rate` `F_A0` débit molaire d'alimentation du réactif [mol/s],
/// `conversion` `X` taux de conversion visé [sans dimension, dans `[0, 1]`],
/// `reaction_rate` `−r_A` vitesse de disparition **à la sortie** [mol/(m³·s)].
///
/// Panique si `molar_feed_rate` est négatif ou non fini, si `conversion` sort de
/// `[0, 1]`, ou si `reaction_rate` n'est pas strictement positif (division).
pub fn cstr_volume(molar_feed_rate: f64, conversion: f64, reaction_rate: f64) -> f64 {
    assert!(
        molar_feed_rate.is_finite() && molar_feed_rate >= 0.0,
        "le débit molaire d'alimentation doit être fini et positif ou nul (mol/s)"
    );
    assert!(
        (0.0..=1.0).contains(&conversion),
        "la conversion doit être comprise dans [0, 1] (sans dimension)"
    );
    assert!(
        reaction_rate > 0.0,
        "la vitesse de réaction −r_A doit être strictement positive (mol/(m³·s))"
    );
    molar_feed_rate * conversion / reaction_rate
}

/// Temps de passage (space time) `τ = V / v̇₀` [s].
///
/// `volume` `V` volume utile du réacteur [m³], `volumetric_flow` `v̇₀` débit
/// volumétrique d'alimentation [m³/s]. `τ` est le temps moyen nominal de séjour
/// à densité constante.
///
/// Panique si `volume` est négatif ou non fini, ou si `volumetric_flow` n'est pas
/// strictement positif (division).
pub fn cstr_space_time(volume: f64, volumetric_flow: f64) -> f64 {
    assert!(
        volume.is_finite() && volume >= 0.0,
        "le volume doit être fini et positif ou nul (m³)"
    );
    assert!(
        volumetric_flow > 0.0,
        "le débit volumétrique doit être strictement positif (m³/s)"
    );
    volume / volumetric_flow
}

/// Conversion d'une réaction d'**ordre 1** à densité constante dans un CSTR
/// `X = k·τ / (1 + k·τ)` [sans dimension].
///
/// `rate_constant` `k` constante cinétique du premier ordre [1/s], `space_time`
/// `τ` temps de passage [s]. Le produit `k·τ = Da` est le nombre de Damköhler ;
/// `X → 1` quand `Da → ∞`.
///
/// Panique si `rate_constant` ou `space_time` est négatif ou non fini.
pub fn cstr_first_order_conversion(rate_constant: f64, space_time: f64) -> f64 {
    assert!(
        rate_constant.is_finite() && rate_constant >= 0.0,
        "la constante cinétique doit être finie et positive ou nulle (1/s)"
    );
    assert!(
        space_time.is_finite() && space_time >= 0.0,
        "le temps de passage doit être fini et positif ou nul (s)"
    );
    let damkohler = rate_constant * space_time;
    damkohler / (1.0 + damkohler)
}

/// Temps de passage **requis** pour atteindre une conversion donnée en ordre 1
/// `τ = X / (k·(1 − X))` [s] — réciproque de [`cstr_first_order_conversion`].
///
/// `conversion` `X` conversion visée [sans dimension, dans `[0, 1[`],
/// `rate_constant` `k` constante cinétique du premier ordre [1/s]. `τ → ∞` quand
/// `X → 1` : la conversion complète exige un réacteur infini.
///
/// Panique si `conversion` sort de `[0, 1[` (la borne 1 fait diverger la
/// division), ou si `rate_constant` n'est pas strictement positif.
pub fn cstr_required_space_time_first_order(conversion: f64, rate_constant: f64) -> f64 {
    assert!(
        conversion.is_finite() && (0.0..1.0).contains(&conversion),
        "la conversion doit être comprise dans [0, 1[ (strictement inférieure à 1)"
    );
    assert!(
        rate_constant > 0.0,
        "la constante cinétique doit être strictement positive (1/s)"
    );
    conversion / (rate_constant * (1.0 - conversion))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn volume_known_case() {
        // F_A0 = 5 mol/s, X = 0,8, −r_A = 2 mol/(m³·s)
        // → V = 5·0,8/2 = 2 m³.
        assert_relative_eq!(cstr_volume(5.0, 0.8, 2.0), 2.0, epsilon = 1e-12);
    }

    #[test]
    fn volume_is_proportional_to_conversion() {
        // À F_A0 et −r_A fixés, V est linéaire en X : doubler X double V.
        let v1 = cstr_volume(3.0, 0.2, 4.0);
        let v2 = cstr_volume(3.0, 0.4, 4.0);
        assert_relative_eq!(v2, 2.0 * v1, epsilon = 1e-12);
    }

    #[test]
    fn space_time_known_case() {
        // V = 10 m³, v̇₀ = 2 m³/s → τ = 5 s.
        assert_relative_eq!(cstr_space_time(10.0, 2.0), 5.0, epsilon = 1e-12);
    }

    #[test]
    fn first_order_conversion_known_case() {
        // k = 0,5 1/s, τ = 4 s → Da = 2 → X = 2/(1+2) = 2/3.
        assert_relative_eq!(
            cstr_first_order_conversion(0.5, 4.0),
            2.0 / 3.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn conversion_and_required_space_time_are_reciprocal() {
        // τ → X → τ doit revenir au point de départ (réciprocité exacte).
        let k = 0.5_f64;
        let tau = 4.0_f64;
        let x = cstr_first_order_conversion(k, tau);
        let tau_back = cstr_required_space_time_first_order(x, k);
        assert_relative_eq!(tau_back, tau, epsilon = 1e-9);
        // Contrôle chiffré direct : X = 2/3, k = 0,5 → τ = (2/3)/(0,5·(1/3)) = 4.
        assert_relative_eq!(
            cstr_required_space_time_first_order(2.0 / 3.0, 0.5),
            4.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn zero_damkohler_gives_zero_conversion() {
        // Cas limite : k = 0 (pas de réaction) → Da = 0 → X = 0.
        assert_relative_eq!(cstr_first_order_conversion(0.0, 10.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "strictement inférieure à 1")]
    fn full_conversion_required_space_time_panics() {
        // X = 1 fait diverger τ = X/(k(1−X)) : entrée rejetée.
        cstr_required_space_time_first_order(1.0, 0.5);
    }
}
