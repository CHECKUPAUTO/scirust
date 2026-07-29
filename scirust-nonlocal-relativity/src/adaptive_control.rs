//! Shared adaptive step-size-control primitives used by *both* adaptive
//! worldline integrators ([`crate::simulate_nonlocal_worldline_adaptive_with_policy`]'s
//! embedded Heun-Euler controller and
//! [`crate::simulate_nonlocal_worldline_adaptive_with_stepper_policy`]'s
//! step-doubling controller).
//!
//! Before this module, each controller computed its local-error estimate as
//! the *unscaled* sum of the coordinate and velocity L2 differences,
//! `||x_high - x_low||_2 + ||u_high - u_low||_2`, and compared it against a
//! single absolute tolerance. That conflated two problems:
//!
//! - coordinates and velocities generally have different numerical scales, and
//!   different coordinate components (a radius, an angle, a time) have
//!   different scales *among themselves*, so one absolute tolerance cannot be
//!   appropriate for all of them at once;
//! - the acceptance decision then depended on the chosen coordinate chart and
//!   its units far more than on any intrinsic accuracy requirement — rescaling
//!   an angular coordinate, say, would shift the accept/reject boundary even
//!   though nothing numerically relevant changed.
//!
//! [`scaled_local_error_norm`] replaces that with the standard componentwise
//! scaled root-mean-square local-error norm used by production adaptive
//! Runge-Kutta codes (Hairer, Nørsett & Wanner, *Solving Ordinary
//! Differential Equations I*, 2nd ed., §II.4): for each state component `i`,
//!
//! ```text
//! scale_i = abs_tol_i + rel_tol * max(|y_low_i|, |y_high_i|)
//! ratio_i = (y_high_i - y_low_i) / scale_i
//! norm    = sqrt( mean_i ( ratio_i^2 ) )
//! ```
//!
//! with separate absolute tolerances for coordinate and velocity components
//! (see [`AdaptiveTolerance`]). A step is accepted when `norm <= 1`. Because
//! each component is divided by its own scale, the norm is dimensionless and
//! far less sensitive to the coordinate chart than the previous unscaled sum.
//!
//! This improves *scaling* robustness. It does **not** make the norm
//! geometrically invariant: it is still evaluated componentwise in the
//! supplied chart and says nothing about the indefinite spacetime metric, so
//! it must never be described as establishing coordinate covariance. A
//! metric-aware research diagnostic is a separate, optional concern.

use crate::{NonlocalRelativityError, NonlocalResult, WorldlineState, validate_scalar};

/// Componentwise scaled local-error tolerance for adaptive stepping.
///
/// The three fields feed the standard scaled error norm implemented by
/// [`scaled_local_error_norm`]:
///
/// - `relative` scales with the local magnitude of the state and is shared by
///   every component (it is dimensionless);
/// - `coordinate_absolute` is the absolute floor for coordinate components,
///   dominating the scale where a coordinate passes through zero;
/// - `velocity_absolute` is the absolute floor for velocity components, kept
///   separate because velocities and coordinates generally differ in scale.
///
/// All three must be finite and strictly positive; zero, negative, NaN, and
/// infinite values are rejected with a typed
/// [`NonlocalRelativityError::InvalidAdaptiveConfiguration`]. Keeping the
/// absolute tolerances strictly positive guarantees every component scale is
/// strictly positive, so the norm never divides by zero even when both state
/// estimates vanish in some component.
///
/// The three-scalar shape is deliberately the minimal form that separates the
/// coordinate and velocity scales; a future extension to fully per-component
/// tolerances can add a variant without changing [`scaled_local_error_norm`]'s
/// call sites, because the norm already asks this type for a per-component
/// scale rather than reading the fields directly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveTolerance {
    relative: f64,
    coordinate_absolute: f64,
    velocity_absolute: f64,
}

impl AdaptiveTolerance {
    /// Validate and construct a componentwise tolerance.
    ///
    /// Every field must be finite and strictly positive. The offending field
    /// name and value are reported in
    /// [`NonlocalRelativityError::InvalidAdaptiveConfiguration`] on rejection.
    pub fn new(
        relative: f64,
        coordinate_absolute: f64,
        velocity_absolute: f64,
    ) -> NonlocalResult<Self> {
        for (field, value) in [
            ("relative", relative),
            ("coordinate_absolute", coordinate_absolute),
            ("velocity_absolute", velocity_absolute),
        ]
        {
            if !value.is_finite() || value <= 0.0
            {
                return Err(NonlocalRelativityError::InvalidAdaptiveConfiguration { field, value });
            }
        }

        Ok(Self {
            relative,
            coordinate_absolute,
            velocity_absolute,
        })
    }

    /// Construct a tolerance that uses one magnitude for all three fields.
    ///
    /// This is the compatibility shape for the legacy single-scalar
    /// `error_tolerance` argument: it seeds the relative tolerance and both
    /// absolute tolerances with the same positive magnitude. Prefer
    /// [`AdaptiveTolerance::new`] to set the coordinate and velocity absolute
    /// scales independently.
    pub fn uniform(magnitude: f64) -> NonlocalResult<Self> {
        Self::new(magnitude, magnitude, magnitude)
    }

    /// Return the dimensionless relative tolerance.
    #[must_use]
    pub const fn relative(self) -> f64 {
        self.relative
    }

    /// Return the absolute tolerance floor for coordinate components.
    #[must_use]
    pub const fn coordinate_absolute(self) -> f64 {
        self.coordinate_absolute
    }

    /// Return the absolute tolerance floor for velocity components.
    #[must_use]
    pub const fn velocity_absolute(self) -> f64 {
        self.velocity_absolute
    }

    /// Componentwise scale `abs_tol + rel_tol * max(|low|, |high|)` for a
    /// coordinate component.
    #[must_use]
    fn coordinate_scale(self, low: f64, high: f64) -> f64 {
        self.coordinate_absolute + self.relative * low.abs().max(high.abs())
    }

    /// Componentwise scale `abs_tol + rel_tol * max(|low|, |high|)` for a
    /// velocity component.
    #[must_use]
    fn velocity_scale(self, low: f64, high: f64) -> f64 {
        self.velocity_absolute + self.relative * low.abs().max(high.abs())
    }
}

/// Deterministic componentwise-scaled root-mean-square local-error norm
/// between a lower-order and a higher-order state estimate of the same
/// accepted step.
///
/// See this module's documentation for the formula. The two estimates are the
/// two members of the step's error pair — for the embedded Heun-Euler
/// controller, the Euler predictor (`lower`) and the Heun corrector
/// (`higher`); for the step-doubling controller, one full step (`lower`) and
/// two half steps (`higher`). The scale uses `max(|low|, |high|)` per
/// component, so the result is symmetric in the two arguments; the
/// `lower`/`higher` naming reflects intent, not a requirement.
///
/// A returned value `<= 1.0` means the step meets `tolerance`. The reduction
/// is evaluated in a fixed component order (all `D` coordinate components,
/// then all `D` velocity components) so the floating-point result is
/// bit-for-bit reproducible for identical inputs.
///
/// This improves robustness to differing component scales and to the choice of
/// coordinate units, but it is evaluated componentwise in the supplied chart
/// and is **not** a geometrically invariant error measure: it must never be
/// described as establishing coordinate covariance.
pub fn scaled_local_error_norm<const D: usize>(
    lower: &WorldlineState<D>,
    higher: &WorldlineState<D>,
    tolerance: AdaptiveTolerance,
) -> NonlocalResult<f64> {
    debug_assert!(D > 0, "worldline dimension must be positive");

    let mut sum_of_squares = 0.0_f64;

    for component in 0..D
    {
        let low = lower.coordinates[component];
        let high = higher.coordinates[component];
        let scale = tolerance.coordinate_scale(low, high);
        let ratio = (high - low) / scale;
        validate_scalar("adaptive_coordinate_error_ratio", ratio, component)?;
        sum_of_squares += ratio * ratio;
        validate_scalar(
            "adaptive_error_ratio_accumulator",
            sum_of_squares,
            component,
        )?;
    }

    for component in 0..D
    {
        let low = lower.velocity[component];
        let high = higher.velocity[component];
        let scale = tolerance.velocity_scale(low, high);
        let ratio = (high - low) / scale;
        validate_scalar("adaptive_velocity_error_ratio", ratio, D + component)?;
        sum_of_squares += ratio * ratio;
        validate_scalar(
            "adaptive_error_ratio_accumulator",
            sum_of_squares,
            D + component,
        )?;
    }

    let component_count = (2 * D) as f64;
    let norm = (sum_of_squares / component_count).sqrt();
    validate_scalar("adaptive_scaled_error_norm", norm, 2 * D)?;
    Ok(norm)
}

/// Safety factor applied to every proposed step size, shared by both adaptive
/// controllers.
pub(crate) const STEP_SAFETY_FACTOR: f64 = 0.9;
/// Maximum per-step growth factor for an accepted step.
pub(crate) const STEP_GROWTH_CAP: f64 = 4.0;
/// Minimum per-step shrink factor for a rejected step (bounds a single
/// rejection's shrinkage so a huge error estimate cannot collapse the step in
/// one move).
pub(crate) const STEP_SHRINK_FLOOR: f64 = 0.1;
/// Step-control exponent `1 / (p_low + 1)` for a first-order lower method
/// (`p_low = 1`). Both controllers pair a first-order estimate with a
/// second-order one (embedded Euler/Heun) or use a first-order method with
/// step-doubling, so both use this exponent.
pub(crate) const FIRST_ORDER_STEP_EXPONENT: f64 = 0.5;

/// Grow an accepted step toward `max_step`, scaled by how far below tolerance
/// the scaled error `normalized_error` was, then clamped to `[min_step,
/// max_step]`. A zero error (identical estimates) grows at the cap.
#[must_use]
pub(crate) fn grow_accepted_step(
    step: f64,
    normalized_error: f64,
    min_step: f64,
    max_step: f64,
) -> f64 {
    let growth = if normalized_error > 0.0
    {
        STEP_SAFETY_FACTOR * normalized_error.powf(-FIRST_ORDER_STEP_EXPONENT)
    }
    else
    {
        STEP_GROWTH_CAP
    };
    (step * growth.min(STEP_GROWTH_CAP)).clamp(min_step, max_step)
}

/// Shrink a rejected step by the same control law, bounded below by
/// `STEP_SHRINK_FLOOR` per rejection. The result is *not* clamped to
/// `min_step`; the caller decides whether crossing `min_step` is an error.
#[must_use]
pub(crate) fn shrink_rejected_step(step: f64, normalized_error: f64) -> f64 {
    let shrink = STEP_SAFETY_FACTOR * normalized_error.powf(-FIRST_ORDER_STEP_EXPONENT);
    step * shrink.max(STEP_SHRINK_FLOOR)
}

/// Clamp a proposed step so an accepted step cannot overshoot the target
/// affine parameter.
#[must_use]
pub(crate) fn clamp_step_to_target(proposed_step: f64, current_parameter: f64, target: f64) -> f64 {
    let remaining = target - current_parameter;
    proposed_step.min(remaining)
}

/// Outcome of the shared accept/reject step-size decision for one trial.
pub(crate) enum StepControl {
    /// The trial met tolerance; propose `next_step` for the following step.
    Accept {
        /// Proposed size for the next accepted step.
        next_step: f64,
    },
    /// The trial was rejected but the rejection budget and minimum step both
    /// still allow another attempt at `next_step`.
    Retry {
        /// Smaller size to retry the current step at.
        next_step: f64,
    },
}

/// The single shared step-size controller used by *both* adaptive worldline
/// integrators.
///
/// Given a trial's scaled error, it decides acceptance, enforces
/// `max_rejections_per_step`, and enforces `min_step`, returning the two
/// distinct typed errors
/// ([`NonlocalRelativityError::AdaptiveRejectionBudgetExhausted`] and
/// [`NonlocalRelativityError::AdaptiveMinimumStepExhausted`]) consistently. The
/// caller owns the `rejection_count`, which this function increments on each
/// rejection; the caller resets it (by declaring a fresh counter) for every new
/// accepted step.
pub(crate) fn control_step(
    normalized_error: f64,
    step: f64,
    rejection_count: &mut usize,
    accepted_step: usize,
    min_step: f64,
    max_step: f64,
    max_rejections_per_step: usize,
) -> NonlocalResult<StepControl> {
    if normalized_error <= 1.0
    {
        return Ok(StepControl::Accept {
            next_step: grow_accepted_step(step, normalized_error, min_step, max_step),
        });
    }

    *rejection_count += 1;
    if *rejection_count >= max_rejections_per_step
    {
        return Err(NonlocalRelativityError::AdaptiveRejectionBudgetExhausted {
            accepted_step,
            rejections: *rejection_count,
            attempted_step: step,
            error_estimate: normalized_error,
        });
    }

    let shrunk_step = shrink_rejected_step(step, normalized_error);
    if shrunk_step < min_step
    {
        return Err(NonlocalRelativityError::AdaptiveMinimumStepExhausted {
            accepted_step,
            attempted_step: step,
            proposed_step: shrunk_step,
            min_step,
            error_estimate: normalized_error,
        });
    }

    Ok(StepControl::Retry {
        next_step: shrunk_step,
    })
}

#[cfg(test)]
mod tests {
    use super::{AdaptiveTolerance, scaled_local_error_norm};
    use crate::{NonlocalRelativityError, WorldlineState};

    fn tol() -> AdaptiveTolerance {
        AdaptiveTolerance::new(1.0e-6, 1.0e-9, 1.0e-9).unwrap()
    }

    #[test]
    fn identical_states_have_exactly_zero_error() {
        let state = WorldlineState::new([1.0, -2.0, 3.0, -4.0], [2.0, 0.25, -0.5, 0.75]);
        let norm = scaled_local_error_norm(&state, &state, tol()).unwrap();
        assert_eq!(norm.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn single_component_perturbation_matches_closed_form() {
        let tolerance = AdaptiveTolerance::new(0.0625, 1.0, 1.0).unwrap();
        let low = WorldlineState::new([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]);
        // Perturb exactly one coordinate component. Its scale is
        // abs=1 + rel*max(|0|,|delta|). With delta = 4 and rel = 0.0625 the
        // scale is 1 + 0.25 = 1.25, so ratio = 4 / 1.25 = 3.2, and the RMS
        // over the 8 components is sqrt(3.2^2 / 8) = 3.2 / sqrt(8).
        let mut high = low;
        high.coordinates[1] = 4.0;
        let norm = scaled_local_error_norm(&low, &high, tolerance).unwrap();
        let expected = 3.2 / 8.0_f64.sqrt();
        assert!(
            (norm - expected).abs() < 1.0e-15,
            "norm={norm}, expected={expected}"
        );
    }

    #[test]
    fn norm_is_symmetric_in_its_arguments() {
        let low = WorldlineState::new([1.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4]);
        let high = WorldlineState::new([1.1, 1.9, 3.2, 3.7], [0.15, 0.18, 0.33, 0.44]);
        let forward = scaled_local_error_norm(&low, &high, tol()).unwrap();
        let backward = scaled_local_error_norm(&high, &low, tol()).unwrap();
        assert_eq!(forward.to_bits(), backward.to_bits());
    }

    #[test]
    fn relative_dominated_norm_is_invariant_under_simultaneous_scaling() {
        // With a negligible absolute tolerance, scaling the state and its
        // error by the same factor leaves every ratio (and hence the norm)
        // unchanged, because scale_i scales with the state magnitude.
        let relative = 1.0e-3;
        let tolerance = AdaptiveTolerance::new(relative, 1.0e-30, 1.0e-30).unwrap();
        let low = WorldlineState::new([10.0, -20.0, 30.0, 40.0], [1.0, -2.0, 3.0, 4.0]);
        let mut high = low;
        for component in 0..4
        {
            high.coordinates[component] *= 1.0 + 1.0e-4;
            high.velocity[component] *= 1.0 - 2.0e-4;
        }
        let base = scaled_local_error_norm(&low, &high, tolerance).unwrap();

        for factor in [1.0e-3, 1.0e3, 1.0e6]
        {
            let mut low_scaled = low;
            let mut high_scaled = high;
            for component in 0..4
            {
                low_scaled.coordinates[component] *= factor;
                low_scaled.velocity[component] *= factor;
                high_scaled.coordinates[component] *= factor;
                high_scaled.velocity[component] *= factor;
            }
            let scaled = scaled_local_error_norm(&low_scaled, &high_scaled, tolerance).unwrap();
            let relative_change = (scaled - base).abs() / base;
            assert!(
                relative_change < 1.0e-9,
                "factor={factor}: base={base}, scaled={scaled}, change={relative_change}"
            );
        }
    }

    #[test]
    fn absolute_tolerance_governs_the_scale_near_zero() {
        // Near zero the relative term vanishes, so the scale is the absolute
        // tolerance and the ratio is error / abs_tol.
        let tolerance = AdaptiveTolerance::new(1.0e-3, 2.0e-6, 5.0e-9).unwrap();
        let low = WorldlineState::new([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]);
        let mut high = low;
        high.coordinates[0] = 2.0e-6; // ratio ~= 1 against coordinate_absolute
        high.velocity[0] = 5.0e-9; // ratio ~= 1 against velocity_absolute
        let norm = scaled_local_error_norm(&low, &high, tolerance).unwrap();
        // Two components each with ratio ~= 1, over 8 components: sqrt(2/8).
        let expected = (2.0_f64 / 8.0).sqrt();
        let relative_change = (norm - expected).abs() / expected;
        assert!(relative_change < 1.0e-3, "norm={norm}, expected={expected}");
    }

    #[test]
    fn norm_is_bit_for_bit_repeatable() {
        let low = WorldlineState::new([1.0, -2.5, 3.25, -4.125], [2.0, 0.25, -0.5, 0.75]);
        let high = WorldlineState::new([1.01, -2.4, 3.30, -4.0], [2.1, 0.2, -0.55, 0.8]);
        let first = scaled_local_error_norm(&low, &high, tol()).unwrap();
        let second = scaled_local_error_norm(&low, &high, tol()).unwrap();
        assert_eq!(first.to_bits(), second.to_bits());
    }

    #[test]
    fn constructor_rejects_non_positive_and_non_finite_fields() {
        for bad in [0.0, -1.0e-6, f64::NAN, f64::INFINITY]
        {
            assert!(matches!(
                AdaptiveTolerance::new(bad, 1.0e-9, 1.0e-9),
                Err(NonlocalRelativityError::InvalidAdaptiveConfiguration {
                    field: "relative",
                    ..
                })
            ));
            assert!(matches!(
                AdaptiveTolerance::new(1.0e-6, bad, 1.0e-9),
                Err(NonlocalRelativityError::InvalidAdaptiveConfiguration {
                    field: "coordinate_absolute",
                    ..
                })
            ));
            assert!(matches!(
                AdaptiveTolerance::new(1.0e-6, 1.0e-9, bad),
                Err(NonlocalRelativityError::InvalidAdaptiveConfiguration {
                    field: "velocity_absolute",
                    ..
                })
            ));
        }
        assert!(AdaptiveTolerance::new(1.0e-6, 1.0e-9, 1.0e-9).is_ok());
        assert!(AdaptiveTolerance::uniform(1.0e-8).is_ok());
        assert!(matches!(
            AdaptiveTolerance::uniform(0.0),
            Err(NonlocalRelativityError::InvalidAdaptiveConfiguration {
                field: "relative",
                ..
            })
        ));
    }
}
