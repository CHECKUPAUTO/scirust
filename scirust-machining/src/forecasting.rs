//! Prévision de demande — lissage exponentiel simple et mesures d'erreur
//! (moyenne mobile, écart absolu moyen, signal de suivi du biais).
//!
//! ```text
//! lissage exponentiel simple  F   = F_prev + alpha·(A - F_prev)
//! moyenne mobile              MA  = (Σ x_i) / n
//! écart absolu moyen          MAD = (Σ |e_i|) / n
//! signal de suivi             TS  = E_cum / MAD
//! ```
//!
//! `F_prev` prévision précédente (unités), `A` demande réalisée (unités), `F`
//! nouvelle prévision (unités), `alpha` coefficient de lissage (sans dimension,
//! `alpha ∈ [0, 1]`), `x_i` valeurs récentes de la série (unités), `n` nombre
//! d'observations, `e_i` erreurs de prévision `A_i - F_i` (unités), `E_cum`
//! erreur cumulée signée (unités), `MAD` écart absolu moyen (unités), `TS`
//! signal de suivi (sans dimension, nombre de MAD que représente le biais).
//!
//! **Limite honnête** : modèles de série temporelle sans tendance ni
//! saisonnalité (lissage exponentiel simple) ; le coefficient `alpha` est
//! FOURNI par la politique de gestion (compromis réactivité/stabilité) et n'est
//! jamais inventé ici. Le signal de suivi détecte un biais systématique (dérive
//! persistante) mais ne corrige pas le modèle par lui-même. Toutes les valeurs
//! d'entrée (prévisions, demandes, erreurs) sont fournies par l'appelant.

/// Lissage exponentiel simple `F = F_prev + alpha·(A - F_prev)`.
///
/// Combinaison convexe entre la prévision précédente et la demande réalisée :
/// `F = (1 - alpha)·F_prev + alpha·A`. Un `alpha` proche de `1` réagit vite,
/// proche de `0` lisse fortement.
///
/// Panique si `smoothing_alpha` n'est pas dans `[0, 1]`.
pub fn forecast_exponential_smoothing(
    previous_forecast: f64,
    actual_demand: f64,
    smoothing_alpha: f64,
) -> f64 {
    assert!(
        (0.0..=1.0).contains(&smoothing_alpha),
        "le coefficient de lissage alpha doit appartenir à [0, 1]"
    );
    previous_forecast + smoothing_alpha * (actual_demand - previous_forecast)
}

/// Moyenne mobile `MA = (Σ x_i) / n`.
///
/// Moyenne arithmétique des `n` valeurs récentes, utilisée comme prévision de
/// la période suivante en l'absence de tendance.
///
/// Panique si `recent_values` est vide.
pub fn forecast_moving_average(recent_values: &[f64]) -> f64 {
    assert!(
        !recent_values.is_empty(),
        "la fenêtre de valeurs récentes ne doit pas être vide"
    );
    let sum: f64 = recent_values.iter().sum();
    sum / recent_values.len() as f64
}

/// Écart absolu moyen `MAD = (Σ |e_i|) / n`.
///
/// Moyenne des valeurs absolues des erreurs de prévision ; mesure de dispersion
/// insensible au signe des écarts.
///
/// Panique si `errors` est vide.
pub fn forecast_mean_absolute_deviation(errors: &[f64]) -> f64 {
    assert!(
        !errors.is_empty(),
        "la liste d'erreurs ne doit pas être vide"
    );
    let sum_abs: f64 = errors.iter().map(|e| e.abs()).sum();
    sum_abs / errors.len() as f64
}

/// Signal de suivi `TS = E_cum / MAD`.
///
/// Rapport de l'erreur cumulée signée à l'écart absolu moyen : un `|TS|` élevé
/// (usuellement au-delà de 4 selon la politique) révèle un biais systématique
/// du modèle ; le signe indique le sens de la dérive.
///
/// Panique si `mean_absolute_deviation <= 0` (division impossible).
pub fn forecast_tracking_signal(cumulative_error: f64, mean_absolute_deviation: f64) -> f64 {
    assert!(
        mean_absolute_deviation > 0.0,
        "l'écart absolu moyen doit être strictement positif"
    );
    cumulative_error / mean_absolute_deviation
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn exponential_smoothing_realistic_case() {
        // F_prev = 100, A = 120, alpha = 0.3.
        // F = 100 + 0.3·(120 - 100) = 100 + 6 = 106.
        let (previous, actual, alpha) = (100.0_f64, 120.0_f64, 0.3_f64);
        assert_relative_eq!(
            forecast_exponential_smoothing(previous, actual, alpha),
            106.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn exponential_smoothing_alpha_extremes() {
        // alpha = 0 : la prévision ne bouge pas (F = F_prev).
        // alpha = 1 : la prévision suit exactement la demande (F = A).
        let (previous, actual) = (80.0_f64, 130.0_f64);
        assert_relative_eq!(
            forecast_exponential_smoothing(previous, actual, 0.0),
            previous,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            forecast_exponential_smoothing(previous, actual, 1.0),
            actual,
            epsilon = 1e-12
        );
    }

    #[test]
    fn exponential_smoothing_is_convex_combination() {
        // Identité : F = (1 - alpha)·F_prev + alpha·A.
        let (previous, actual, alpha) = (95.0_f64, 140.0_f64, 0.25_f64);
        let expected = (1.0 - alpha) * previous + alpha * actual;
        assert_relative_eq!(
            forecast_exponential_smoothing(previous, actual, alpha),
            expected,
            epsilon = 1e-9
        );
    }

    #[test]
    fn moving_average_constant_window() {
        // Fenêtre constante : la moyenne mobile vaut la constante elle-même.
        assert_relative_eq!(
            forecast_moving_average(&[42.0, 42.0, 42.0, 42.0]),
            42.0,
            epsilon = 1e-12
        );
        // Cas chiffré : moyenne de 10, 20, 30, 40 = 100/4 = 25.
        assert_relative_eq!(
            forecast_moving_average(&[10.0, 20.0, 30.0, 40.0]),
            25.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn mad_ignores_sign_of_errors() {
        // MAD symétrique : {3, -5, 4, -2, 1} et {-3, 5, -4, 2, -1} donnent le
        // même MAD = (3 + 5 + 4 + 2 + 1)/5 = 15/5 = 3.
        let a = [3.0, -5.0, 4.0, -2.0, 1.0];
        let b = [-3.0, 5.0, -4.0, 2.0, -1.0];
        assert_relative_eq!(forecast_mean_absolute_deviation(&a), 3.0, epsilon = 1e-9);
        assert_relative_eq!(
            forecast_mean_absolute_deviation(&a),
            forecast_mean_absolute_deviation(&b),
            epsilon = 1e-12
        );
    }

    #[test]
    fn tracking_signal_counts_mad_units() {
        // E_cum = 12, MAD = 3 ⇒ TS = 4 (biais de 4 MAD).
        // Le signe est conservé : E_cum = -12 ⇒ TS = -4.
        assert_relative_eq!(forecast_tracking_signal(12.0, 3.0), 4.0, epsilon = 1e-9);
        assert_relative_eq!(forecast_tracking_signal(-12.0, 3.0), -4.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "alpha doit appartenir à [0, 1]")]
    fn alpha_out_of_range_panics() {
        forecast_exponential_smoothing(100.0, 120.0, 1.5);
    }
}
