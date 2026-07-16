//! **Distribution des temps de séjour** (DTS) d'un traceur en régime permanent :
//! temps de séjour moyen, fonctions `E(t)` et `F(t)` du CSTR idéal, et nombre de
//! réacteurs en série équivalent caractérisant le mélange.
//!
//! ```text
//! temps de séjour moyen τ = V / v̇                         [s]
//! âge de sortie (CSTR)  E(t) = exp(−t/τ) / τ              [1/s]
//! cumulée (CSTR)        F(t) = 1 − exp(−t/τ)              [sans dimension]
//! réacteurs en série    N = τ² / σ²                       [sans dimension]
//! ```
//!
//! `V` volume utile [m³], `v̇` débit volumétrique en régime permanent [m³/s], `τ`
//! temps de séjour moyen [s], `t` temps écoulé depuis l'injection du traceur [s],
//! `E(t)` fonction d'âge à la sortie (densité de probabilité, d'intégrale unité)
//! [1/s], `F(t)` fraction cumulée du traceur déjà sortie [sans dimension, dans
//! `[0, 1]`], `σ²` variance de la DTS mesurée [s²], `N` nombre de cuves parfaitement
//! agitées en série équivalent [sans dimension].
//!
//! **Limite honnête** : ces relations décrivent la DTS d'un **traceur** en **régime
//! permanent** à débit constant. `E(t)` et `F(t)` sont données pour le **CSTR
//! idéal** (parfaitement agité) ; l'écart au piston se **mesure expérimentalement**
//! (courbe de traceur). Le modèle des réacteurs en série résume le mélange par un
//! seul paramètre `N` (`N = 1` : CSTR ; `N → ∞` : écoulement piston) et se calcule à
//! partir de la **variance FOURNIE/mesurée** `σ²`. Les propriétés (enthalpies,
//! volatilités, coefficients de partage, constantes cinétiques, diffusivités…) sont
//! **fournies par l'appelant** : aucune valeur « par défaut » n'est inventée.

/// Temps de séjour moyen `τ = V / v̇` [s].
///
/// `volume` `V` volume utile [m³], `volumetric_flow` `v̇` débit volumétrique en
/// régime permanent [m³/s]. `τ` est l'aire sous `1 − F(t)`, soit le premier moment
/// de la DTS.
///
/// Panique si `volume` est négatif ou non fini, ou si `volumetric_flow` n'est pas
/// strictement positif (division).
pub fn rtd_mean_residence_time(volume: f64, volumetric_flow: f64) -> f64 {
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

/// Fonction d'âge à la sortie d'un **CSTR idéal** `E(t) = exp(−t/τ) / τ` [1/s].
///
/// `time` `t` temps depuis l'injection [s], `mean_residence_time` `τ` temps de
/// séjour moyen [s]. `E(t)` est une densité de probabilité : son intégrale de `0`
/// à `∞` vaut `1`, et `E(0) = 1/τ` est sa valeur maximale (décroissance
/// exponentielle).
///
/// Panique si `time` est négatif ou non fini, ou si `mean_residence_time` n'est
/// pas strictement positif (division).
pub fn rtd_exit_age_cstr(time: f64, mean_residence_time: f64) -> f64 {
    assert!(
        time.is_finite() && time >= 0.0,
        "le temps doit être fini et positif ou nul (s)"
    );
    assert!(
        mean_residence_time > 0.0,
        "le temps de séjour moyen doit être strictement positif (s)"
    );
    (-time / mean_residence_time).exp() / mean_residence_time
}

/// Fonction cumulée de sortie d'un **CSTR idéal** `F(t) = 1 − exp(−t/τ)`
/// [sans dimension] — fraction du traceur déjà sortie au temps `t`.
///
/// `time` `t` temps depuis l'injection [s], `mean_residence_time` `τ` temps de
/// séjour moyen [s]. `F(t)` croît de `0` (à `t = 0`) vers `1` (à `t → ∞`) et vérifie
/// `F(t) = 1 − τ·E(t)` : c'est la primitive de [`rtd_exit_age_cstr`].
///
/// Panique si `time` est négatif ou non fini, ou si `mean_residence_time` n'est
/// pas strictement positif (division).
pub fn rtd_cumulative_cstr(time: f64, mean_residence_time: f64) -> f64 {
    assert!(
        time.is_finite() && time >= 0.0,
        "le temps doit être fini et positif ou nul (s)"
    );
    assert!(
        mean_residence_time > 0.0,
        "le temps de séjour moyen doit être strictement positif (s)"
    );
    1.0 - (-time / mean_residence_time).exp()
}

/// Nombre de réacteurs parfaitement agités **en série équivalent**
/// `N = τ² / σ²` [sans dimension].
///
/// `mean_residence_time` `τ` temps de séjour moyen [s], `variance` `σ²` variance de
/// la DTS mesurée [s²]. `N = 1` correspond à un CSTR unique (`σ² = τ²`) et `N → ∞`
/// à l'écoulement piston (`σ² → 0`) : `N` mesure ainsi la proximité du piston.
///
/// Panique si `mean_residence_time` est négatif ou non fini, ou si `variance`
/// n'est pas strictement positif (division).
pub fn rtd_tanks_in_series_number(mean_residence_time: f64, variance: f64) -> f64 {
    assert!(
        mean_residence_time.is_finite() && mean_residence_time >= 0.0,
        "le temps de séjour moyen doit être fini et positif ou nul (s)"
    );
    assert!(
        variance > 0.0,
        "la variance doit être strictement positive (s²)"
    );
    mean_residence_time * mean_residence_time / variance
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn mean_residence_time_known_case() {
        // V = 10 m³, v̇ = 2 m³/s → τ = 5 s.
        assert_relative_eq!(rtd_mean_residence_time(10.0, 2.0), 5.0, epsilon = 1e-12);
    }

    #[test]
    fn exit_age_at_origin_is_inverse_tau() {
        // E(0) = exp(0)/τ = 1/τ : valeur maximale de la densité.
        let tau = 4.0_f64;
        assert_relative_eq!(rtd_exit_age_cstr(0.0, tau), 1.0 / tau, epsilon = 1e-12);
    }

    #[test]
    fn cumulative_and_exit_age_are_consistent() {
        // Identité du CSTR idéal : F(t) = 1 − τ·E(t) pour tout t.
        let tau = 2.5_f64;
        let t = 3.0_f64;
        let e = rtd_exit_age_cstr(t, tau);
        let f = rtd_cumulative_cstr(t, tau);
        assert_relative_eq!(f, 1.0 - tau * e, epsilon = 1e-12);
    }

    #[test]
    fn cumulative_at_one_tau_is_known() {
        // F(τ) = 1 − exp(−1) = 0,632120558828557… (indépendant de τ).
        let tau = 7.0_f64;
        assert_relative_eq!(
            rtd_cumulative_cstr(tau, tau),
            1.0 - core::f64::consts::E.recip(),
            epsilon = 1e-12
        );
        // Contrôle chiffré direct du littéral.
        assert_relative_eq!(
            rtd_cumulative_cstr(tau, tau),
            0.632_120_558_828_557_7,
            epsilon = 1e-3
        );
    }

    #[test]
    fn single_cstr_gives_unit_tank_number() {
        // Pour un CSTR, σ² = τ² donc N = τ²/σ² = 1 (limite mélange parfait).
        let tau = 6.0_f64;
        let variance = tau * tau;
        assert_relative_eq!(
            rtd_tanks_in_series_number(tau, variance),
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn tank_number_scales_inversely_with_variance() {
        // À τ fixé, N ∝ 1/σ² : diviser la variance par 4 multiplie N par 4
        // (l'écoulement se rapproche du piston).
        let tau = 5.0_f64;
        let n1 = rtd_tanks_in_series_number(tau, 8.0);
        let n2 = rtd_tanks_in_series_number(tau, 2.0);
        assert_relative_eq!(n2, 4.0 * n1, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "strictement positive")]
    fn zero_variance_panics() {
        // σ² = 0 (piston idéal) fait diverger N = τ²/σ² : entrée rejetée.
        rtd_tanks_in_series_number(5.0, 0.0);
    }
}
