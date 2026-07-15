//! Vibrations transmises à l'**ensemble du corps** (WBV, ISO 2631) : valeur
//! d'exposition journalière normalisée `A(8)`, axe dominant, somme vectorielle et
//! durée avant d'atteindre une valeur d'action ou limite.
//!
//! ```text
//! exposition journalière   A(8) = a_w·√(T/8)
//! axe dominant             a_dom = max(a_x, a_y, a_z)
//! somme vectorielle        a_v  = √(a_x² + a_y² + a_z²)
//! durée avant limite       T_L  = 8·(a_L/a_w)²
//! ```
//!
//! `a_w` valeur d'accélération pondérée en fréquence (m/s²), `a_x, a_y, a_z`
//! accélérations pondérées par axe (pondérations directionnelles `k_x = k_y = 1,4`
//! et `k_z = 1,0` **déjà appliquées** aux entrées, m/s²), `T` durée d'exposition
//! effective (h), `A(8)` valeur d'exposition journalière normalisée sur 8 h
//! (m/s²), `a_dom` accélération de l'axe le plus sollicité (m/s²), `a_v` somme
//! vectorielle des trois axes (m/s²), `a_L` valeur d'action ou limite fournie
//! (m/s²), `T_L` durée d'exposition à `a_w` avant d'atteindre `a_L` (h).
//!
//! **Convention** : accélérations en m/s², durées en heures (le `8` du
//! dénominateur est la période de référence de 8 h). **Limite honnête** : les
//! accélérations sont **déjà pondérées en fréquence** (filtres ISO 2631) et
//! **fournies** par la mesure sur le poste réel ; les pondérations directionnelles
//! `k_x, k_y, k_z` sont supposées déjà appliquées par l'appelant. La normalisation
//! est purement énergétique sur 8 h et ne modélise ni la posture ni le siège. Les
//! valeurs d'action et limite usuelles (0,5 et 1,15 m/s²) sont **fournies** par la
//! réglementation applicable — jamais présumées ici. Distinct de
//! [`crate::hand_arm_vibration`].

/// Valeur d'exposition journalière normalisée `A(8) = a_w·√(T/8)` (m/s²).
///
/// Panique si `weighted_acceleration < 0` ou `exposure_duration_hours < 0`.
pub fn wbv_daily_exposure_a8(weighted_acceleration: f64, exposure_duration_hours: f64) -> f64 {
    assert!(
        weighted_acceleration >= 0.0 && exposure_duration_hours >= 0.0,
        "a_w ≥ 0 et T ≥ 0 requis"
    );
    weighted_acceleration * (exposure_duration_hours / 8.0_f64).sqrt()
}

/// Accélération de l'axe dominant `a_dom = max(a_x, a_y, a_z)` (m/s²), les
/// pondérations directionnelles étant déjà appliquées aux entrées.
///
/// Panique si l'une des accélérations est `< 0`.
pub fn wbv_dominant_axis_acceleration(ax: f64, ay: f64, az: f64) -> f64 {
    assert!(
        ax >= 0.0 && ay >= 0.0 && az >= 0.0,
        "a_x, a_y, a_z ≥ 0 requis"
    );
    ax.max(ay).max(az)
}

/// Somme vectorielle des accélérations pondérées
/// `a_v = √(a_x² + a_y² + a_z²)` (m/s²).
///
/// Panique si l'une des accélérations est `< 0`.
pub fn wbv_vector_sum(ax: f64, ay: f64, az: f64) -> f64 {
    assert!(
        ax >= 0.0 && ay >= 0.0 && az >= 0.0,
        "a_x, a_y, a_z ≥ 0 requis"
    );
    (ax * ax + ay * ay + az * az).sqrt()
}

/// Durée d'exposition à `a_w` avant d'atteindre la valeur limite fournie
/// `T_L = 8·(a_L/a_w)²` (h).
///
/// Panique si `weighted_acceleration <= 0` ou `limit_value < 0`.
pub fn wbv_time_to_limit(weighted_acceleration: f64, limit_value: f64) -> f64 {
    assert!(
        weighted_acceleration > 0.0 && limit_value >= 0.0,
        "a_w > 0 et a_L ≥ 0 requis"
    );
    let ratio = limit_value / weighted_acceleration;
    8.0 * ratio * ratio
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn a8_over_full_reference_period_equals_magnitude() {
        // T = 8 h ⇒ A(8) = a_w·√1 = a_w (cas limite de la période de référence).
        assert_relative_eq!(wbv_daily_exposure_a8(0.5, 8.0), 0.5, max_relative = 1e-12);
    }

    #[test]
    fn a8_realistic_case() {
        // a_w = 0,8 m/s² pendant 2 h : A(8) = 0,8·√(2/8) = 0,8·0,5 = 0,4 m/s².
        assert_relative_eq!(wbv_daily_exposure_a8(0.8, 2.0), 0.4, max_relative = 1e-12);
    }

    #[test]
    fn a8_proportional_to_magnitude() {
        // Pour une durée fixée, A(8) ∝ a_w : doubler a_w double A(8).
        let single = wbv_daily_exposure_a8(0.3, 4.0);
        let doubled = wbv_daily_exposure_a8(0.6, 4.0);
        assert_relative_eq!(doubled / single, 2.0, max_relative = 1e-12);
    }

    #[test]
    fn dominant_axis_returns_largest() {
        // max(0,6 ; 0,9 ; 1,2) = 1,2 (axe le plus sollicité).
        assert_relative_eq!(
            wbv_dominant_axis_acceleration(0.6, 0.9, 1.2),
            1.2,
            max_relative = 1e-12
        );
    }

    #[test]
    fn vector_sum_pythagorean_triple() {
        // √(0,3² + 0,4² + 0²) = √0,25 = 0,5 (triplet pythagoricien 3-4-5).
        assert_relative_eq!(wbv_vector_sum(0.3, 0.4, 0.0), 0.5, max_relative = 1e-12);
    }

    #[test]
    fn vector_sum_dominates_each_axis() {
        // La somme vectorielle est toujours ≥ à l'axe dominant (a_v² = Σ a_i²).
        let (ax, ay, az) = (0.5_f64, 0.3_f64, 0.2_f64);
        let dom = wbv_dominant_axis_acceleration(ax, ay, az);
        let v = wbv_vector_sum(ax, ay, az);
        assert!(v >= dom);
        // Sur un seul axe non nul, la somme vectorielle égale cet axe.
        assert_relative_eq!(wbv_vector_sum(0.7, 0.0, 0.0), 0.7, max_relative = 1e-12);
    }

    #[test]
    fn time_to_limit_inverts_daily_exposure() {
        // À T = T_L, l'exposition journalière vaut exactement la limite (réciprocité).
        let (a_w, limit) = (0.8, 0.5);
        let t_l = wbv_time_to_limit(a_w, limit); // 8·(0,625)² = 3,125 h
        assert_relative_eq!(t_l, 3.125, max_relative = 1e-12);
        assert_relative_eq!(wbv_daily_exposure_a8(a_w, t_l), limit, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "a_w > 0")]
    fn zero_magnitude_time_to_limit_panics() {
        wbv_time_to_limit(0.0, 0.5);
    }
}
