//! Taguchi loss function and the cost of non-quality.
//!
//! A pass/fail specification says a part just inside the limit is perfect and one
//! just outside is worthless — a discontinuity nature does not honour. Taguchi
//! replaces it with a **continuous quadratic loss** measured from the *target*:
//! any deviation costs society something, growing with its square,
//!
//! ```text
//! L(y) = k·(y − T)²    (nominal-the-best) .
//! ```
//!
//! The coefficient is fixed by one known point — the loss `A` at the tolerance
//! limit `Δ` gives `k = A/Δ²` ([`loss_coefficient`]). Averaged over a batch the
//! expected loss is
//!
//! ```text
//! E[L] = k·E[(y − T)²] = k·(σ² + δ²) = k·I² ,
//! ```
//!
//! the loss coefficient times the **inertia squared** — the exact reason
//! inertial tolerancing ([`crate::inertia`]) minimises Taguchi loss directly,
//! and the sense in which `I` *is* `√(E[loss]/k)`. [`expected_loss`] takes the
//! inertia, [`expected_loss_from_moments`] the raw `(μ, σ, T)`.
//!
//! ## Economic tolerance
//!
//! Balancing the customer's functional loss against the producer's rework cost
//! gives Taguchi's economic manufacturing tolerance: if a deviation `Δ₀` at the
//! functional limit costs `A₀`, and reworking in the factory costs `A`, it pays
//! to rework once the loss reaches `A`, i.e. at
//!
//! ```text
//! Δ = Δ₀·√(A / A₀)     ([`economic_tolerance`]) .
//! ```

/// Nominal-the-best quadratic loss `L = k·(y − target)²`.
pub fn quadratic_loss(k: f64, y: f64, target: f64) -> f64 {
    let d = y - target;
    k * d * d
}

/// Loss coefficient `k = cost_at_limit / half_tolerance²`, fixing the quadratic
/// loss so that a part at the tolerance limit costs `cost_at_limit`. Returns 0
/// for a non-positive tolerance.
pub fn loss_coefficient(cost_at_limit: f64, half_tolerance: f64) -> f64 {
    if half_tolerance <= 0.0
    {
        return 0.0;
    }
    cost_at_limit / (half_tolerance * half_tolerance)
}

/// Expected nominal-the-best loss from the batch **inertia**, `E[L] = k·I²`.
pub fn expected_loss(k: f64, inertia: f64) -> f64 {
    k * inertia * inertia
}

/// Expected nominal-the-best loss from raw moments, `E[L] = k·(σ² + (μ − T)²)`.
pub fn expected_loss_from_moments(k: f64, mean: f64, sd: f64, target: f64) -> f64 {
    let delta = mean - target;
    k * (sd * sd + delta * delta)
}

/// Expected **smaller-the-better** loss `E[k·y²] = k·(μ² + σ²)` (target 0, e.g.
/// wear, shrinkage, impurity).
pub fn smaller_the_better_loss(k: f64, mean: f64, sd: f64) -> f64 {
    k * (mean * mean + sd * sd)
}

/// Expected **larger-the-better** loss `E[k/y²] ≈ (k/μ²)(1 + 3(σ/μ)²)` (target
/// ∞, e.g. strength, life). The second-order approximation needs `μ > 0`;
/// returns `+∞` otherwise.
pub fn larger_the_better_loss(k: f64, mean: f64, sd: f64) -> f64 {
    if mean <= 0.0
    {
        return f64::INFINITY;
    }
    let cv2 = (sd / mean).powi(2);
    (k / (mean * mean)) * (1.0 + 3.0 * cv2)
}

/// Taguchi's economic manufacturing tolerance `Δ = Δ₀·√(A/A₀)`: the deviation at
/// which the functional quality loss reaches the rework cost, so tightening
/// further no longer pays. `functional_loss` `A₀` is the loss at the functional
/// half-tolerance `functional_half_tol` `Δ₀`; `rework_cost` is `A`. Returns 0 for
/// a non-positive functional loss or tolerance.
pub fn economic_tolerance(functional_loss: f64, functional_half_tol: f64, rework_cost: f64) -> f64 {
    if functional_loss <= 0.0 || functional_half_tol <= 0.0 || rework_cost < 0.0
    {
        return 0.0;
    }
    functional_half_tol * (rework_cost / functional_loss).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn coefficient_hits_the_cost_at_the_limit() {
        // k = A/Δ²; at y = T + Δ the loss is exactly A.
        let (cost, half_tol, target) = (12.0, 0.5, 10.0);
        let k = loss_coefficient(cost, half_tol);
        assert_relative_eq!(
            quadratic_loss(k, target + half_tol, target),
            cost,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            quadratic_loss(k, target - half_tol, target),
            cost,
            epsilon = 1e-12
        );
    }

    #[test]
    fn expected_loss_is_k_times_inertia_squared() {
        // E[L] via inertia == via moments, for I = √(δ²+σ²).
        let (k, mean, sd, target): (f64, f64, f64, f64) = (48.0, 10.1, 0.08, 10.0);
        let delta = mean - target;
        let inertia = (delta * delta + sd * sd).sqrt();
        assert_relative_eq!(
            expected_loss(k, inertia),
            expected_loss_from_moments(k, mean, sd, target),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            expected_loss(k, inertia),
            k * inertia * inertia,
            epsilon = 1e-12
        );
    }

    #[test]
    fn economic_tolerance_balances_loss_and_rework() {
        // At Δ the functional loss k₀·Δ² equals the rework cost A.
        let (a0, delta0, a) = (20.0, 1.0, 5.0);
        let delta = economic_tolerance(a0, delta0, a);
        let k0 = loss_coefficient(a0, delta0);
        assert_relative_eq!(k0 * delta * delta, a, epsilon = 1e-12);
        // Cheaper rework ⇒ tighter economic tolerance.
        assert!(economic_tolerance(a0, delta0, 2.0) < delta);
    }

    #[test]
    fn smaller_and_larger_the_better() {
        // Smaller-the-better: pure second moment about 0.
        assert_relative_eq!(
            smaller_the_better_loss(3.0, 2.0, 0.5),
            3.0 * (4.0 + 0.25),
            epsilon = 1e-12
        );
        // Larger-the-better decreases as the mean grows.
        let a = larger_the_better_loss(1.0, 10.0, 1.0);
        let b = larger_the_better_loss(1.0, 20.0, 1.0);
        assert!(b < a);
        assert_eq!(larger_the_better_loss(1.0, 0.0, 1.0), f64::INFINITY);
    }
}
