//! Réacteur **discontinu** (batch) — temps requis pour atteindre une conversion
//! donnée selon l'ordre de réaction (0, 1 ou 2) et temps de cycle incluant le
//! temps mort de chargement/nettoyage.
//!
//! ```text
//! ordre 0   t = C₀·X / k                                          [s]
//! ordre 1   t = −ln(1 − X) / k                                    [s]
//! ordre 2   t = X / (k·C₀·(1 − X))         (réactif unique)       [s]
//! cycle     t_c = t_réaction + t_mort                             [s]
//! ```
//!
//! `t` temps de réaction pour atteindre la conversion `X` [s], `X` taux de
//! conversion du réactif limitant [sans dimension, 0 ≤ X < 1], `k` constante de
//! vitesse [unité dépendant de l'ordre : s⁻¹ à l'ordre 1, mol⁻¹·m³·s⁻¹ à l'ordre 2,
//! mol·m⁻³·s⁻¹ à l'ordre 0], `C₀` concentration initiale du réactif
//! [mol·m⁻³, cohérente avec `k`], `t_c` temps de cycle [s], `t_mort` temps mort
//! (chargement, vidange, nettoyage) [s].
//!
//! **Limite honnête** : ces relations décrivent un réacteur **discontinu à volume
//! constant** et **isotherme**, avec un **ordre de réaction fourni** (0, 1 ou 2 ;
//! aucun mécanisme n'est déduit). La constante de vitesse `k`, la concentration
//! initiale `C₀` et le temps mort de chargement/nettoyage sont **fournis par
//! l'appelant** (mesures, tables, régression, gamme) ; aucune valeur « par défaut »
//! n'est inventée. Les **unités de `C₀` doivent être cohérentes avec `k`**. Le
//! **semi-batch** (alimentation en cours de marche), la non-isothermie et les
//! réactions multiples ne sont **pas** traités.

/// Temps requis en réacteur batch pour une réaction d'**ordre 0** :
/// `t = C₀·X / k` [s].
///
/// `conversion` `X` taux de conversion [sans dimension, 0 ≤ X < 1],
/// `rate_constant` `k` constante de vitesse d'ordre 0 [mol·m⁻³·s⁻¹],
/// `initial_concentration` `C₀` concentration initiale du réactif [mol·m⁻³].
///
/// Panique si `conversion` n'est pas dans `[0, 1[`, ou si `rate_constant` /
/// `initial_concentration` n'est pas fini ou n'est pas strictement positif.
pub fn batch_time_zero_order(
    conversion: f64,
    rate_constant: f64,
    initial_concentration: f64,
) -> f64 {
    assert!(
        conversion.is_finite() && (0.0..1.0).contains(&conversion),
        "la conversion doit être dans l'intervalle [0, 1["
    );
    assert!(
        rate_constant.is_finite() && rate_constant > 0.0,
        "la constante de vitesse doit être finie et strictement positive"
    );
    assert!(
        initial_concentration.is_finite() && initial_concentration > 0.0,
        "la concentration initiale doit être finie et strictement positive (mol·m⁻³)"
    );
    initial_concentration * conversion / rate_constant
}

/// Temps requis en réacteur batch pour une réaction d'**ordre 1** :
/// `t = −ln(1 − X) / k` [s].
///
/// `conversion` `X` taux de conversion [sans dimension, 0 ≤ X < 1],
/// `rate_constant` `k` constante de vitesse d'ordre 1 [s⁻¹].
///
/// Panique si `conversion` n'est pas dans `[0, 1[`, ou si `rate_constant`
/// n'est pas fini ou n'est pas strictement positif.
pub fn batch_time_first_order(conversion: f64, rate_constant: f64) -> f64 {
    assert!(
        conversion.is_finite() && (0.0..1.0).contains(&conversion),
        "la conversion doit être dans l'intervalle [0, 1["
    );
    assert!(
        rate_constant.is_finite() && rate_constant > 0.0,
        "la constante de vitesse doit être finie et strictement positive"
    );
    -(1.0 - conversion).ln() / rate_constant
}

/// Temps requis en réacteur batch pour une réaction d'**ordre 2** (réactif
/// unique) : `t = X / (k·C₀·(1 − X))` [s].
///
/// `conversion` `X` taux de conversion [sans dimension, 0 ≤ X < 1],
/// `rate_constant` `k` constante de vitesse d'ordre 2 [mol⁻¹·m³·s⁻¹],
/// `initial_concentration` `C₀` concentration initiale du réactif [mol·m⁻³].
///
/// Panique si `conversion` n'est pas dans `[0, 1[`, ou si `rate_constant` /
/// `initial_concentration` n'est pas fini ou n'est pas strictement positif.
pub fn batch_time_second_order(
    conversion: f64,
    rate_constant: f64,
    initial_concentration: f64,
) -> f64 {
    assert!(
        conversion.is_finite() && (0.0..1.0).contains(&conversion),
        "la conversion doit être dans l'intervalle [0, 1["
    );
    assert!(
        rate_constant.is_finite() && rate_constant > 0.0,
        "la constante de vitesse doit être finie et strictement positive"
    );
    assert!(
        initial_concentration.is_finite() && initial_concentration > 0.0,
        "la concentration initiale doit être finie et strictement positive (mol·m⁻³)"
    );
    conversion / (rate_constant * initial_concentration * (1.0 - conversion))
}

/// Temps de **cycle** d'un réacteur batch : `t_c = t_réaction + t_mort` [s].
///
/// `reaction_time` `t_réaction` durée de réaction utile [s], `downtime`
/// `t_mort` temps mort de chargement, vidange et nettoyage [s].
///
/// Panique si `reaction_time` ou `downtime` n'est pas fini ou est négatif.
pub fn batch_cycle_time(reaction_time: f64, downtime: f64) -> f64 {
    assert!(
        reaction_time.is_finite() && reaction_time >= 0.0,
        "le temps de réaction doit être fini et positif ou nul"
    );
    assert!(
        downtime.is_finite() && downtime >= 0.0,
        "le temps mort doit être fini et positif ou nul"
    );
    reaction_time + downtime
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn zero_order_is_proportional_to_conversion() {
        // Ordre 0 : t = C₀·X/k. À C₀ et k fixés, t ∝ X ; doubler X double t.
        let t_half = batch_time_zero_order(0.25, 2.0, 4.0);
        let t_full = batch_time_zero_order(0.50, 2.0, 4.0);
        // t = 4·0,25/2 = 0,5 s ; t = 4·0,5/2 = 1,0 s.
        assert_relative_eq!(t_half, 0.5, epsilon = 1e-9);
        assert_relative_eq!(t_full, 1.0, epsilon = 1e-9);
        assert_relative_eq!(t_full, 2.0 * t_half, epsilon = 1e-9);
    }

    #[test]
    fn first_order_half_life_identity() {
        // Ordre 1 : atteindre X = 0,5 demande t = ln 2 / k, indépendant de C₀.
        let k = 0.1_f64;
        let t = batch_time_first_order(0.5, k);
        assert_relative_eq!(t, core::f64::consts::LN_2 / k, epsilon = 1e-12);
        // À conversion nulle, le temps requis est nul.
        assert_relative_eq!(batch_time_first_order(0.0, k), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn second_order_realistic_case() {
        // k = 0,05 mol⁻¹·m³·s⁻¹, C₀ = 2 mol·m⁻³, X = 0,8.
        // t = 0,8 / (0,05·2·(1 − 0,8)) = 0,8 / (0,05·2·0,2) = 0,8 / 0,02 = 40 s.
        let t = batch_time_second_order(0.8, 0.05, 2.0);
        assert_relative_eq!(t, 40.0, epsilon = 1e-9);
    }

    #[test]
    fn second_order_scales_inversely_with_initial_concentration() {
        // Ordre 2 : t ∝ 1/C₀ à k et X fixés ; doubler C₀ divise t par deux.
        let t_low = batch_time_second_order(0.6, 0.05, 1.0);
        let t_high = batch_time_second_order(0.6, 0.05, 2.0);
        assert_relative_eq!(t_high, t_low / 2.0, epsilon = 1e-9);
    }

    #[test]
    fn cycle_time_adds_downtime() {
        // Temps de cycle = réaction + temps mort ; identité additive.
        assert_relative_eq!(batch_cycle_time(40.0, 900.0), 940.0, epsilon = 1e-9);
        // Temps mort nul : le cycle se réduit au temps de réaction.
        assert_relative_eq!(batch_cycle_time(40.0, 0.0), 40.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "conversion doit être dans l'intervalle")]
    fn conversion_at_one_panics() {
        // X = 1 (conversion totale) donnerait un temps infini à l'ordre 1 ou 2.
        batch_time_first_order(1.0, 0.1);
    }
}
