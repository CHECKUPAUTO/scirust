//! Core of inertial tolerancing (*tolérancement inertiel*, M. Pillet et al.).
//!
//! Traditional tolerancing accepts a characteristic when every part lies in a
//! `[LSL, USL]` interval — it judges *distance to a limit*. Inertial
//! tolerancing instead limits the **inertia**
//!
//! ```text
//! I = √(δ² + σ²),   δ = μ − Target   (off-centering),   σ = std-dev
//! ```
//!
//! the root-mean-square deviation of the characteristic **from its target**.
//! Because
//!
//! ```text
//! E[(X − Target)²] = (μ − Target)² + σ² = δ² + σ² = I²,
//! ```
//!
//! the inertia is exactly the square root of the mean Taguchi quadratic loss:
//! a batch that is slightly off-centre with small spread and one that is
//! centred with larger spread are judged *equivalent* when they carry the same
//! expected loss. A single scalar `I_max` replaces the `± tolerance`, and the
//! acceptance region in the `(δ, σ)` plane becomes a **half-disc** of radius
//! `I_max` (the *inertia cone*) rather than the `Cpk` rectangle.
//!
//! Reference: Adragna, Pillet, Formosa, Samper, *Inertial tolerancing and
//! capability indices in an assembly production* (arXiv:1002.0270).

use serde::{Deserialize, Serialize};

/// A characteristic described by its off-centering and dispersion, the two
/// numbers inertial tolerancing works with.
///
/// `off_centering` is the signed distance of the mean from the target
/// (`δ = μ − Target`); `sigma` is the standard deviation (`σ ≥ 0`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Inertia {
    /// Off-centering `δ = μ − Target` (signed).
    pub off_centering: f64,
    /// Standard deviation `σ` (non-negative).
    pub sigma: f64,
}

impl Inertia {
    /// Build from an off-centering `δ` and dispersion `σ`.
    pub fn new(off_centering: f64, sigma: f64) -> Self {
        Self {
            off_centering,
            sigma: sigma.abs(),
        }
    }

    /// Build from a process mean, dispersion and target value.
    pub fn from_moments(mean: f64, sigma: f64, target: f64) -> Self {
        Self::new(mean - target, sigma)
    }

    /// Estimate the inertia of a batch from a raw sample and its target.
    ///
    /// Uses the **population** dispersion `σ̂² = (1/n) Σ(xᵢ − x̄)²`, which makes
    /// the squared inertia `Î² = δ̂² + σ̂²` an *unbiased* estimator of the true
    /// `I² = δ² + σ²`: with `δ̂ = x̄ − T`,
    ///
    /// ```text
    /// E[δ̂²] = δ² + σ²/n,   E[σ̂²] = σ²·(n−1)/n   ⇒   E[Î²] = δ² + σ² = I².
    /// ```
    ///
    /// Equivalently `Î² = (1/n) Σ(xᵢ − T)²`, the second moment about target.
    /// Returns a zero inertia for an empty sample.
    pub fn from_sample(data: &[f64], target: f64) -> Self {
        let n = data.len();
        if n == 0
        {
            return Self::new(0.0, 0.0);
        }
        let nf = n as f64;
        let mean = data.iter().sum::<f64>() / nf;
        let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / nf;
        Self::new(mean - target, var.sqrt())
    }

    /// The inertia value `I = √(δ² + σ²)`.
    pub fn value(&self) -> f64 {
        (self.off_centering * self.off_centering + self.sigma * self.sigma).sqrt()
    }

    /// The expected Taguchi quadratic loss coefficient `I² = δ² + σ²` — the
    /// mean squared deviation from target, before scaling by a cost `k`.
    pub fn mean_squared_deviation(&self) -> f64 {
        self.off_centering * self.off_centering + self.sigma * self.sigma
    }

    /// Fraction of the squared inertia attributable to off-centering,
    /// `δ² / I²` (in `[0, 1]`). A value near 1 means the batch is dominated by
    /// a centering error (re-centre it); near 0 means dispersion dominates.
    /// Returns 0 for a null inertia.
    pub fn off_centering_ratio(&self) -> f64 {
        let i2 = self.mean_squared_deviation();
        if i2 <= 0.0
        {
            0.0
        }
        else
        {
            self.off_centering * self.off_centering / i2
        }
    }

    /// Variance of the squared-inertia estimator `Î² = (1/n) Σ(xᵢ − T)²` for a
    /// sample of size `n` drawn from this (normal) characteristic:
    ///
    /// ```text
    /// Var(Î²) = 2σ²(2δ² + σ²) / n .
    /// ```
    ///
    /// Its square root is a standard error for `Î²`, useful for putting an
    /// uncertainty band on an estimated inertia. Returns 0 for `n = 0`.
    pub fn variance_of_squared_estimate(&self, n: usize) -> f64 {
        if n == 0
        {
            return 0.0;
        }
        let (d2, s2) = (
            self.off_centering * self.off_centering,
            self.sigma * self.sigma,
        );
        2.0 * s2 * (2.0 * d2 + s2) / n as f64
    }
}

/// Inertia of a mixture of lots (sub-batches), the property that makes
/// inertial tolerancing robust to **lot mixing**: the squared inertia of the
/// pooled lot is the proportion-weighted mean of the sub-lot squared inertias,
///
/// ```text
/// I_c² = Σ pᵢ Iᵢ² ,   δ_c = Σ pᵢ δᵢ ,   σ_c² = I_c² − δ_c² ,
/// ```
///
/// because `E[(X − T)²]` over the mixture is the mixture of the per-lot
/// `E[(X − T)²] = Iᵢ²`. `lots` are `(proportion, inertia)` pairs; the
/// proportions are normalised to sum to 1. Returns a null inertia for an empty
/// or zero-weight input.
pub fn mix_lots(lots: &[(f64, Inertia)]) -> Inertia {
    let total: f64 = lots.iter().map(|(p, _)| p.abs()).sum();
    if total <= 0.0
    {
        return Inertia::new(0.0, 0.0);
    }
    let mut i2 = 0.0;
    let mut delta = 0.0;
    for (p, inertia) in lots
    {
        let w = p.abs() / total;
        i2 += w * inertia.mean_squared_deviation();
        delta += w * inertia.off_centering;
    }
    // σ_c² = I_c² − δ_c² is non-negative by Jensen (mixing never reduces spread
    // below the between-lot mean shift), but guard against rounding.
    let sigma = (i2 - delta * delta).max(0.0).sqrt();
    Inertia::new(delta, sigma)
}

/// Resultant inertia of a multi-degree-of-freedom (e.g. 3D location +
/// orientation) deviation whose per-DOF inertias combine statistically:
/// `I = √(Σ Iⱼ²)`, the Euclidean norm of the component inertias.
///
/// This is the isotropic combination; for a functional requirement that
/// weights the DOFs unequally, use [`crate::chain`] with the DOFs as
/// contributors and their influence coefficients (the general 3D inertial
/// tolerancing of arXiv:1002.0253 is the geometric reference).
pub fn vector_inertia(components: &[Inertia]) -> f64 {
    components
        .iter()
        .map(|c| c.mean_squared_deviation())
        .sum::<f64>()
        .sqrt()
}

/// Correct an observed inertia for measurement dispersion. A measurement
/// system with standard uncertainty `u` inflates the observed variance
/// (`σ_obs² = σ_true² + u²`) but not the off-centering, so the true inertia is
///
/// ```text
/// I_true = √(I_obs² − u²).
/// ```
///
/// Returns `None` when `u` exceeds the observed dispersion `σ_obs` (so
/// `σ_true² = σ_obs² − u²` would be negative — the measurement noise alone
/// exceeds the observed spread and the correction is undefined). Pairs with
/// `scirust-metrology` for `u` from a GUM budget.
pub fn correct_for_measurement(observed: &Inertia, u: f64) -> Option<Inertia> {
    let corrected_var = observed.sigma * observed.sigma - u * u;
    if corrected_var < 0.0
    {
        None
    }
    else
    {
        Some(Inertia::new(observed.off_centering, corrected_var.sqrt()))
    }
}

/// Expected quadratic (Taguchi) loss of a characteristic with the given
/// inertia, `E[L] = k · I²`, where `k` is the loss coefficient (cost per unit
/// squared deviation from target).
pub fn expected_taguchi_loss(inertia: &Inertia, k: f64) -> f64 {
    k * inertia.mean_squared_deviation()
}

/// The Taguchi loss coefficient `k = A₀ / Δ₀²` implied by a loss `A₀`
/// incurred when the characteristic sits at a distance `Δ₀` from target
/// (typically `A₀` = scrap/rework cost at the tolerance limit `Δ₀`).
pub fn taguchi_k(loss_at_limit: f64, distance_at_limit: f64) -> f64 {
    if distance_at_limit == 0.0
    {
        return 0.0;
    }
    loss_at_limit / (distance_at_limit * distance_at_limit)
}

/// Maximum inertia for a bilateral tolerance interval `it = USL − LSL`, sized
/// so a batch sitting on the cone boundary reaches capability `target_cp`
/// when perfectly centred.
///
/// A centred batch has `I = σ`, so requiring `Cp = it/(6σ) ≥ target_cp` gives
/// `I_max = it / (6 · target_cp)`. The common conventions are
/// `target_cp = 1` (`I_max = it/6`, `Cpm = 1` on the boundary) and
/// `target_cp = 2` (`I_max = it/12`, a "6σ" target).
pub fn i_max_from_tolerance(it: f64, target_cp: f64) -> f64 {
    if target_cp <= 0.0
    {
        return f64::INFINITY;
    }
    it / (6.0 * target_cp)
}

/// The `Cp = 1` maximum inertia `I_max = it/6` for a bilateral tolerance
/// interval `it = USL − LSL` (shorthand for [`i_max_from_tolerance`] with
/// `target_cp = 1`).
pub fn i_max_cp1(it: f64) -> f64 {
    it / 6.0
}

/// The acceptance region of inertial tolerancing: the half-disc
/// `{ (δ, σ) : δ² + σ² ≤ I_max², σ ≥ 0 }` in the off-centering/dispersion
/// plane — the *inertia cone*.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct InertiaCone {
    /// Radius of the half-disc: the maximum admissible inertia.
    pub i_max: f64,
}

impl InertiaCone {
    /// A cone of radius `i_max`.
    pub fn new(i_max: f64) -> Self {
        Self { i_max: i_max.abs() }
    }

    /// Cone whose radius is the `Cp = target_cp` inertia of a bilateral
    /// tolerance interval `it` (see [`i_max_from_tolerance`]).
    pub fn from_tolerance(it: f64, target_cp: f64) -> Self {
        Self::new(i_max_from_tolerance(it, target_cp))
    }

    /// Whether a characteristic lies inside the acceptance cone
    /// (`I ≤ I_max`).
    pub fn accepts(&self, inertia: &Inertia) -> bool {
        inertia.value() <= self.i_max
    }

    /// Signed inertial margin `I_max − I`: positive inside the cone, negative
    /// outside, zero on the boundary.
    pub fn margin(&self, inertia: &Inertia) -> f64 {
        self.i_max - inertia.value()
    }

    /// The largest dispersion `σ` still admissible at a given off-centering
    /// `δ` (`√(I_max² − δ²)`), or `None` when `|δ| > I_max` (no admissible
    /// spread — the batch is off-target beyond the whole inertia budget).
    pub fn max_sigma_at(&self, off_centering: f64) -> Option<f64> {
        let rem = self.i_max * self.i_max - off_centering * off_centering;
        if rem < 0.0 { None } else { Some(rem.sqrt()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn inertia_is_rms_deviation_from_target() {
        let i = Inertia::new(3.0, 4.0);
        assert_relative_eq!(i.value(), 5.0, epsilon = 1e-12);
        assert_relative_eq!(i.mean_squared_deviation(), 25.0, epsilon = 1e-12);
    }

    #[test]
    fn from_moments_takes_target_into_account() {
        let i = Inertia::from_moments(10.5, 0.2, 10.0);
        assert_relative_eq!(i.off_centering, 0.5, epsilon = 1e-12);
        assert_relative_eq!(i.value(), (0.25f64 + 0.04).sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn squared_sample_inertia_is_unbiased_for_i_squared() {
        // Î² = (1/n) Σ (xᵢ − T)² exactly.
        let data = [9.8, 10.1, 10.0, 10.3, 9.9];
        let t = 10.0;
        let i = Inertia::from_sample(&data, t);
        let want_i2 = data.iter().map(|&x| (x - t).powi(2)).sum::<f64>() / data.len() as f64;
        assert_relative_eq!(i.mean_squared_deviation(), want_i2, epsilon = 1e-12);
    }

    #[test]
    fn taguchi_loss_scales_with_inertia_squared() {
        let i = Inertia::new(0.3, 0.4); // I² = 0.25
        let k = taguchi_k(100.0, 0.5); // loss 100 at distance 0.5 ⇒ k = 400
        assert_relative_eq!(k, 400.0, epsilon = 1e-12);
        assert_relative_eq!(expected_taguchi_loss(&i, k), 400.0 * 0.25, epsilon = 1e-12);
    }

    #[test]
    fn i_max_conventions() {
        assert_relative_eq!(i_max_from_tolerance(1.0, 1.0), 1.0 / 6.0, epsilon = 1e-12);
        assert_relative_eq!(i_max_from_tolerance(1.0, 2.0), 1.0 / 12.0, epsilon = 1e-12);
        assert_relative_eq!(i_max_cp1(0.6), 0.1, epsilon = 1e-12);
    }

    #[test]
    fn cone_accepts_inside_and_rejects_outside() {
        let cone = InertiaCone::new(0.1);
        // Off-centre but low spread — accepted if within the disc.
        assert!(cone.accepts(&Inertia::new(0.06, 0.08))); // I = 0.1 exactly
        assert!(!cone.accepts(&Inertia::new(0.09, 0.08))); // I > 0.1
        assert_relative_eq!(cone.margin(&Inertia::new(0.0, 0.06)), 0.04, epsilon = 1e-12);
    }

    #[test]
    fn cone_max_sigma_shrinks_with_off_centering() {
        let cone = InertiaCone::new(0.1);
        assert_relative_eq!(cone.max_sigma_at(0.0).unwrap(), 0.1, epsilon = 1e-12);
        assert_relative_eq!(cone.max_sigma_at(0.06).unwrap(), 0.08, epsilon = 1e-12);
        assert!(cone.max_sigma_at(0.2).is_none());
    }

    #[test]
    fn mixing_lots_averages_squared_inertia() {
        // Two equal lots, opposite off-centering ±0.1, σ = 0. Pooled δ = 0 but
        // pooled I² = mean of Iᵢ² = 0.01 ⇒ pooled inertia = 0.1 (all now spread).
        let a = Inertia::new(0.1, 0.0);
        let b = Inertia::new(-0.1, 0.0);
        let mixed = mix_lots(&[(1.0, a), (1.0, b)]);
        assert_relative_eq!(mixed.off_centering, 0.0, epsilon = 1e-12);
        assert_relative_eq!(mixed.value(), 0.1, epsilon = 1e-12);
        // I_c² is always the weighted mean of the component I².
        assert_relative_eq!(
            mixed.mean_squared_deviation(),
            0.5 * a.mean_squared_deviation() + 0.5 * b.mean_squared_deviation(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn mixing_a_single_lot_is_identity() {
        let a = Inertia::new(0.03, 0.05);
        let mixed = mix_lots(&[(3.0, a)]);
        assert_relative_eq!(mixed.value(), a.value(), epsilon = 1e-12);
    }

    #[test]
    fn vector_inertia_is_euclidean_norm() {
        let v = vector_inertia(&[Inertia::new(0.03, 0.04), Inertia::new(0.0, 0.12)]);
        // √(0.05² + 0.12²) = 0.13.
        assert_relative_eq!(v, 0.13, epsilon = 1e-12);
    }

    #[test]
    fn measurement_correction_removes_gauge_variance() {
        // Observed σ = 0.05 includes u = 0.03 ⇒ true σ = 0.04.
        let obs = Inertia::new(0.02, 0.05);
        let corr = correct_for_measurement(&obs, 0.03).unwrap();
        assert_relative_eq!(corr.sigma, 0.04, epsilon = 1e-12);
        assert_relative_eq!(corr.off_centering, 0.02, epsilon = 1e-12);
        // Gauge noise larger than observed spread ⇒ undefined.
        assert!(correct_for_measurement(&obs, 0.1).is_none());
    }

    #[test]
    fn squared_estimate_variance_matches_closed_form() {
        let i = Inertia::new(0.3, 0.4);
        // 2·σ²·(2δ² + σ²)/n = 2·0.16·(0.18 + 0.16)/8 = 2·0.16·0.34/8 = 0.0136.
        assert_relative_eq!(i.variance_of_squared_estimate(8), 0.0136, epsilon = 1e-12);
    }
}
