//! Fonction de perte de **Taguchi** — coût quadratique de l'écart à la cible.
//!
//! ```text
//! nominal = meilleur   L(y) = k·(y − y0)²
//! coefficient de perte k    = L_tol / Δ²             (identifié à la tolérance)
//! perte moyenne d'un lot L̄  = k·((μ − y0)² + σ²)     (biais² + variance)
//! plus petit = meilleur L(y) = k·y²                   (cible y0 = 0)
//! ```
//!
//! `L` perte (unité monétaire, p. ex. €), `y` valeur mesurée, `y0` valeur cible
//! (`target_value`), `k` coefficient de perte (€·unité⁻²), `Δ` demi-intervalle de
//! tolérance (`tolerance`), `L_tol` perte encourue à la limite de tolérance (€),
//! `μ` moyenne du lot, `σ²` variance du lot (unité²). L'unité de `y`, `y0`, `Δ`
//! est libre mais **commune** (mm, µm, …) et fixe l'unité de `k`.
//!
//! **Limite honnête** : fonction de perte **quadratique** de Taguchi, qui approxime
//! le coût sociétal de la variabilité autour de la cible. Le coefficient `k` est
//! **fourni** par l'appelant — identifié à la limite de tolérance
//! ([`taguchi_loss_coefficient`]) ou par le coût de non-qualité — de même que la
//! cible et la tolérance ; aucune valeur « par défaut » n'est inventée ici.

/// Perte « nominal = meilleur » `L = k·(y − y0)²`.
///
/// Panique si `loss_coefficient < 0`.
pub fn taguchi_loss_nominal_best(
    measured_value: f64,
    target_value: f64,
    loss_coefficient: f64,
) -> f64 {
    assert!(loss_coefficient >= 0.0, "k ≥ 0 requis");
    let deviation = measured_value - target_value;
    loss_coefficient * deviation * deviation
}

/// Coefficient de perte identifié à la tolérance `k = L_tol / Δ²`.
///
/// Panique si `tolerance <= 0` ou `loss_at_tolerance < 0`.
pub fn taguchi_loss_coefficient(tolerance: f64, loss_at_tolerance: f64) -> f64 {
    assert!(tolerance > 0.0, "Δ > 0 requis");
    assert!(loss_at_tolerance >= 0.0, "L_tol ≥ 0 requis");
    loss_at_tolerance / (tolerance * tolerance)
}

/// Perte moyenne d'un lot `L̄ = k·((μ − y0)² + σ²)`.
///
/// Panique si `loss_coefficient < 0` ou `variance < 0`.
pub fn taguchi_average_loss(
    mean_value: f64,
    target_value: f64,
    variance: f64,
    loss_coefficient: f64,
) -> f64 {
    assert!(loss_coefficient >= 0.0, "k ≥ 0 requis");
    assert!(variance >= 0.0, "σ² ≥ 0 requis");
    let bias = mean_value - target_value;
    loss_coefficient * (bias * bias + variance)
}

/// Perte « plus petit = meilleur » `L = k·y²` (cible implicite `y0 = 0`).
///
/// Panique si `measured_value < 0` ou `loss_coefficient < 0`.
pub fn taguchi_loss_smaller_better(measured_value: f64, loss_coefficient: f64) -> f64 {
    assert!(
        measured_value >= 0.0,
        "y ≥ 0 requis (caractéristique positive)"
    );
    assert!(loss_coefficient >= 0.0, "k ≥ 0 requis");
    loss_coefficient * measured_value * measured_value
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn coefficient_and_nominal_best_are_reciprocal() {
        // k identifié à la tolérance, puis évalué EXACTEMENT à la limite y = y0 + Δ
        // redonne la perte de tolérance L_tol.
        let (target, tolerance, loss_at_tolerance) = (50.0, 0.05, 8.0);
        let k = taguchi_loss_coefficient(tolerance, loss_at_tolerance);
        let loss = taguchi_loss_nominal_best(target + tolerance, target, k);
        assert_relative_eq!(loss, loss_at_tolerance, max_relative = 1e-12);
    }

    #[test]
    fn nominal_best_is_quadratic_in_deviation() {
        // L ∝ (y − y0)² : doubler l'écart quadruple la perte.
        let (target, k) = (50.0, 3200.0);
        let l1 = taguchi_loss_nominal_best(target + 0.02, target, k);
        let l2 = taguchi_loss_nominal_best(target + 0.04, target, k);
        assert_relative_eq!(l2 / l1, 4.0, epsilon = 1e-9);
    }

    #[test]
    fn average_loss_reduces_to_nominal_best_when_variance_zero() {
        // σ² = 0 ⇒ L̄ = k·(μ − y0)² = perte ponctuelle en μ.
        let (mean, target, k) = (50.01, 50.0, 3200.0);
        assert_relative_eq!(
            taguchi_average_loss(mean, target, 0.0, k),
            taguchi_loss_nominal_best(mean, target, k),
            max_relative = 1e-12
        );
    }

    #[test]
    fn average_loss_is_k_times_variance_when_centered() {
        // Lot centré (μ = y0) ⇒ L̄ = k·σ² : le biais² s'annule.
        let (target, variance, k) = (50.0, 4.0e-4, 3200.0);
        assert_relative_eq!(
            taguchi_average_loss(target, target, variance, k),
            k * variance,
            max_relative = 1e-12
        );
    }

    #[test]
    fn realistic_batch_case() {
        // Arbre Ø50 mm, tolérance ±0,05 mm, rebut à la limite = 8,00 €.
        // k = 8,00 / 0,05² = 3200 €/mm².
        // Lot : biais μ − y0 = 0,01 mm, variance σ² = 4·10⁻⁴ mm².
        // L̄ = 3200·(0,01² + 4·10⁻⁴) = 3200·(1·10⁻⁴ + 4·10⁻⁴) = 3200·5·10⁻⁴ = 1,60 €.
        let k = taguchi_loss_coefficient(0.05, 8.0);
        assert_relative_eq!(k, 3200.0, max_relative = 1e-12);
        let average = taguchi_average_loss(50.01, 50.0, 4.0e-4, k);
        assert_relative_eq!(average, 1.60, max_relative = 1e-12);
    }

    #[test]
    fn smaller_better_is_quadratic() {
        // L = k·y² : doubler y quadruple la perte.
        let k = 500.0;
        let l1 = taguchi_loss_smaller_better(0.8, k);
        let l2 = taguchi_loss_smaller_better(1.6, k);
        assert_relative_eq!(l2 / l1, 4.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "Δ > 0")]
    fn zero_tolerance_panics() {
        taguchi_loss_coefficient(0.0, 8.0);
    }
}
