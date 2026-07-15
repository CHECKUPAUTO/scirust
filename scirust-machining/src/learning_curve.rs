//! Courbe d'apprentissage de **Wright** (modèle log-linéaire) : exposant
//! d'apprentissage, temps du n-ième exemplaire, temps cumulé et temps moyen.
//!
//! ```text
//! exposant d'apprentissage   e   = ln(b) / ln(2)
//! temps du n-ième exemplaire  Tn  = T1 · n^e
//! temps cumulé (N exemplaires) Tc = Σ_{i=1..N} T1 · i^e
//! temps moyen unitaire        Ta  = Tc / N
//! ```
//!
//! `b` taux d'apprentissage (fraction sans dimension, `]0 ; 1]` ; `b = 0.8`
//! signifie que le temps unitaire est réduit à 80 % à chaque doublement de la
//! quantité produite), `e` exposant d'apprentissage (sans dimension, négatif ou
//! nul), `T1` temps du premier exemplaire (unité de temps cohérente, p. ex. h),
//! `n` rang de l'exemplaire (sans dimension, `≥ 1`), `N` quantité totale
//! produite (exemplaires), `Tn` temps du n-ième exemplaire (même unité que
//! `T1`), `Tc` temps cumulé de la série (même unité que `T1`), `Ta` temps moyen
//! par exemplaire (même unité que `T1`).
//!
//! **Convention** : unité de temps cohérente (le même « h » ou « min » partout).
//! **Limite honnête** : modèle unitaire log-linéaire de Wright (temps par
//! exemplaire strictement décroissant en loi de puissance) ; il ignore l'oubli,
//! les paliers, le régime permanent et l'effet de la taille de lot. Le taux
//! d'apprentissage `b` et le temps de référence `T1` sont FOURNIS par
//! l'appelant ; aucune valeur « par défaut » n'est inventée.

/// Exposant d'apprentissage `e = ln(b) / ln(2)`.
///
/// Pente de la droite en coordonnées log-log ; `e ≤ 0`, avec `e = 0` pour
/// `b = 1` (aucun apprentissage) et `e = -1` pour `b = 0.5`.
///
/// Panique si `learning_rate <= 0` ou `learning_rate > 1`.
pub fn learning_curve_exponent(learning_rate: f64) -> f64 {
    assert!(
        learning_rate > 0.0 && learning_rate <= 1.0,
        "le taux d'apprentissage doit appartenir à l'intervalle ]0 ; 1]"
    );
    learning_rate.ln() / 2.0_f64.ln()
}

/// Temps du n-ième exemplaire `Tn = T1 · n^e`, avec `e = ln(b)/ln(2)`.
///
/// Temps unitaire prédit pour l'exemplaire de rang `n` ; à chaque doublement du
/// rang le temps est multiplié par `b`.
///
/// Panique si `first_unit_time < 0`, `unit_number < 1`, `learning_rate <= 0`
/// ou `learning_rate > 1`.
pub fn learning_curve_unit_time(first_unit_time: f64, unit_number: f64, learning_rate: f64) -> f64 {
    assert!(
        first_unit_time >= 0.0,
        "le temps du premier exemplaire doit être positif ou nul"
    );
    assert!(
        unit_number >= 1.0,
        "le rang de l'exemplaire doit être supérieur ou égal à 1"
    );
    let exponent = learning_curve_exponent(learning_rate);
    first_unit_time * unit_number.powf(exponent)
}

/// Temps cumulé de la série `Tc = Σ_{i=1..N} T1 · i^e` (somme exacte du modèle
/// unitaire).
///
/// Somme des temps unitaires des `N` premiers exemplaires produits.
///
/// Panique si `first_unit_time < 0`, `total_units == 0`, `learning_rate <= 0`
/// ou `learning_rate > 1`.
pub fn learning_curve_cumulative_time(
    first_unit_time: f64,
    total_units: u32,
    learning_rate: f64,
) -> f64 {
    assert!(
        first_unit_time >= 0.0,
        "le temps du premier exemplaire doit être positif ou nul"
    );
    assert!(
        total_units >= 1,
        "la quantité totale produite doit être au moins égale à 1"
    );
    let exponent = learning_curve_exponent(learning_rate);
    let mut cumulative = 0.0_f64;
    for i in 1..=total_units
    {
        cumulative += first_unit_time * (i as f64).powf(exponent);
    }
    cumulative
}

/// Temps moyen par exemplaire `Ta = Tc / N`.
///
/// Temps unitaire moyen sur la série, obtenu en répartissant le temps cumulé
/// sur les `N` exemplaires produits.
///
/// Panique si `cumulative_time < 0` ou `units <= 0`.
pub fn learning_curve_average_time(cumulative_time: f64, units: f64) -> f64 {
    assert!(
        cumulative_time >= 0.0,
        "le temps cumulé doit être positif ou nul"
    );
    assert!(
        units > 0.0,
        "la quantité produite doit être strictement positive"
    );
    cumulative_time / units
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn exponent_reference_values() {
        // b = 1 ⇒ e = 0 (aucun apprentissage) ; b = 0.5 ⇒ e = -1 exactement.
        assert_relative_eq!(learning_curve_exponent(1.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(learning_curve_exponent(0.5), -1.0, epsilon = 1e-12);
    }

    #[test]
    fn unit_time_doubling_multiplies_by_learning_rate() {
        // Identité de Wright : Tn(2n) = b · Tn(n), car (2n)^e = 2^e · n^e = b · n^e.
        let (t1, b) = (100.0, 0.8);
        for &n in &[1.0_f64, 3.0, 7.5, 50.0]
        {
            let tn = learning_curve_unit_time(t1, n, b);
            let t2n = learning_curve_unit_time(t1, 2.0 * n, b);
            assert_relative_eq!(t2n, b * tn, epsilon = 1e-9);
        }
    }

    #[test]
    fn unit_time_realistic_case() {
        // b = 0.8, T1 = 100 h : T1 = 100, T2 = 80, T4 = 64, T8 = 51.2.
        let (t1, b) = (100.0, 0.8);
        assert_relative_eq!(learning_curve_unit_time(t1, 1.0, b), 100.0, epsilon = 1e-9);
        assert_relative_eq!(learning_curve_unit_time(t1, 2.0, b), 80.0, epsilon = 1e-9);
        assert_relative_eq!(learning_curve_unit_time(t1, 4.0, b), 64.0, epsilon = 1e-9);
        assert_relative_eq!(learning_curve_unit_time(t1, 8.0, b), 51.2, epsilon = 1e-9);
    }

    #[test]
    fn unit_time_proportional_to_first_unit_time() {
        // Tn ∝ T1 : doubler T1 double le temps de chaque exemplaire.
        let b = 0.85;
        let a = learning_curve_unit_time(120.0, 10.0, b);
        let c = learning_curve_unit_time(240.0, 10.0, b);
        assert_relative_eq!(c, 2.0 * a, epsilon = 1e-9);
    }

    #[test]
    fn cumulative_matches_sum_of_first_units() {
        // Tc(2) = T1 + T1·b ; average = Tc/N. b = 0.8, T1 = 100 ⇒ Tc = 180, Ta = 90.
        let (t1, b) = (100.0, 0.8);
        let tc2 = learning_curve_cumulative_time(t1, 2, b);
        assert_relative_eq!(tc2, 180.0, epsilon = 1e-9);
        assert_relative_eq!(learning_curve_average_time(tc2, 2.0), 90.0, epsilon = 1e-9);
        // Cohérence somme/unitaire pour N = 4.
        let manual = learning_curve_unit_time(t1, 1.0, b)
            + learning_curve_unit_time(t1, 2.0, b)
            + learning_curve_unit_time(t1, 3.0, b)
            + learning_curve_unit_time(t1, 4.0, b);
        assert_relative_eq!(
            learning_curve_cumulative_time(t1, 4, b),
            manual,
            epsilon = 1e-9
        );
    }

    #[test]
    fn no_learning_keeps_time_constant() {
        // b = 1 ⇒ e = 0 : tout exemplaire coûte T1, donc Tc = N·T1 et Ta = T1.
        let t1 = 42.0;
        assert_relative_eq!(learning_curve_unit_time(t1, 99.0, 1.0), t1, epsilon = 1e-9);
        let tc = learning_curve_cumulative_time(t1, 5, 1.0);
        assert_relative_eq!(tc, 5.0 * t1, epsilon = 1e-9);
        assert_relative_eq!(learning_curve_average_time(tc, 5.0), t1, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "taux d'apprentissage doit appartenir à l'intervalle")]
    fn out_of_range_learning_rate_panics() {
        learning_curve_exponent(1.5);
    }
}
