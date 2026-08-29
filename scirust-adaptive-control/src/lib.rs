//! Deterministic adaptive-step control primitives shared across SciRust.
//!
//! This crate owns only the scalar I-controller law that turns a normalized
//! local-error estimate into an accept/reject decision and a bounded scale
//! factor. It deliberately does not own an integrator, an error norm, a step
//! minimum, or any domain-specific state.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use core::fmt;

/// Accept/reject decision for one normalized local-error estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StepDecision {
    /// The normalized error is at or below the unit acceptance threshold.
    Accept,
    /// The normalized error exceeds the unit acceptance threshold.
    Reject,
}

/// Validated parameters for the elementary I-controller
/// `safety * error^(-exponent)` with bounded scaling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IController {
    safety: f64,
    min_scale: f64,
    max_scale: f64,
    exponent: f64,
}

impl IController {
    /// Validate and construct an I-controller.
    ///
    /// All parameters must be finite and strictly positive, and
    /// `min_scale <= max_scale`.
    pub fn new(
        safety: f64,
        min_scale: f64,
        max_scale: f64,
        exponent: f64,
    ) -> Result<Self, AdaptiveControlError> {
        for (field, value) in [
            ("safety", safety),
            ("min_scale", min_scale),
            ("max_scale", max_scale),
            ("exponent", exponent),
        ]
        {
            if !value.is_finite() || value <= 0.0
            {
                return Err(AdaptiveControlError::InvalidParameter { field, value });
            }
        }
        if min_scale > max_scale
        {
            return Err(AdaptiveControlError::InvalidScaleBounds {
                min_scale,
                max_scale,
            });
        }
        Ok(Self {
            safety,
            min_scale,
            max_scale,
            exponent,
        })
    }

    /// Return the configured safety multiplier.
    #[must_use]
    pub const fn safety(self) -> f64 {
        self.safety
    }

    /// Return the minimum allowed scale factor.
    #[must_use]
    pub const fn min_scale(self) -> f64 {
        self.min_scale
    }

    /// Return the maximum allowed scale factor.
    #[must_use]
    pub const fn max_scale(self) -> f64 {
        self.max_scale
    }

    /// Return the controller exponent.
    #[must_use]
    pub const fn exponent(self) -> f64 {
        self.exponent
    }

    /// Evaluate one non-negative normalized error.
    ///
    /// Zero error maps exactly to `max_scale`; otherwise the scale is
    /// `(safety * error^(-exponent)).clamp(min_scale, max_scale)`. Acceptance
    /// is defined solely by the conventional normalized threshold `error <= 1`.
    pub fn evaluate(self, normalized_error: f64) -> Result<ControlAction, AdaptiveControlError> {
        if !normalized_error.is_finite() || normalized_error < 0.0
        {
            return Err(AdaptiveControlError::InvalidNormalizedError(
                normalized_error,
            ));
        }

        let scale = if normalized_error == 0.0
        {
            self.max_scale
        }
        else
        {
            (self.safety * normalized_error.powf(-self.exponent))
                .clamp(self.min_scale, self.max_scale)
        };
        let decision = if normalized_error <= 1.0
        {
            StepDecision::Accept
        }
        else
        {
            StepDecision::Reject
        };

        Ok(ControlAction { decision, scale })
    }
}

/// Result of evaluating one normalized error with an [`IController`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControlAction {
    decision: StepDecision,
    scale: f64,
}

impl ControlAction {
    /// Return the accept/reject decision.
    #[must_use]
    pub const fn decision(self) -> StepDecision {
        self.decision
    }

    /// Return the bounded multiplicative step-size scale.
    #[must_use]
    pub const fn scale(self) -> f64 {
        self.scale
    }
}

/// Validation failures for generic adaptive control.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdaptiveControlError {
    /// A controller parameter was non-finite or non-positive.
    InvalidParameter {
        /// Stable parameter name.
        field: &'static str,
        /// Rejected value.
        value: f64,
    },
    /// The configured scale interval was reversed.
    InvalidScaleBounds {
        /// Proposed lower bound.
        min_scale: f64,
        /// Proposed upper bound.
        max_scale: f64,
    },
    /// A normalized error was negative or non-finite.
    InvalidNormalizedError(f64),
}

impl fmt::Display for AdaptiveControlError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self
        {
            Self::InvalidParameter { field, value } =>
            {
                write!(
                    formatter,
                    "adaptive-control parameter {field} must be finite and positive, got {value}"
                )
            },
            Self::InvalidScaleBounds {
                min_scale,
                max_scale,
            } => write!(
                formatter,
                "adaptive-control scale bounds must satisfy min <= max, got {min_scale} > {max_scale}"
            ),
            Self::InvalidNormalizedError(value) => write!(
                formatter,
                "normalized adaptive error must be finite and non-negative, got {value}"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AdaptiveControlError, IController, StepDecision};

    #[test]
    fn zero_error_uses_exact_growth_cap() {
        let controller = IController::new(0.9, 0.1, 4.0, 0.5).unwrap();
        let action = controller.evaluate(0.0).unwrap();
        assert_eq!(action.decision(), StepDecision::Accept);
        assert_eq!(action.scale().to_bits(), 4.0_f64.to_bits());
    }

    #[test]
    fn nonlocal_parameters_match_historical_formula() {
        let controller = IController::new(0.9, 0.1, 4.0, 0.5).unwrap();
        for error in [0.25_f64, 1.0, 4.0, 100.0]
        {
            let expected = (0.9 * error.powf(-0.5)).clamp(0.1, 4.0);
            let actual = controller.evaluate(error).unwrap().scale();
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn dopri_parameters_match_historical_formula() {
        let controller = IController::new(0.9, 0.2, 5.0, 0.2).unwrap();
        for error in [0.01_f64, 0.5, 1.0, 2.0, 1000.0]
        {
            let expected = (0.9 * error.powf(-0.2)).clamp(0.2, 5.0);
            let actual = controller.evaluate(error).unwrap().scale();
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn unit_threshold_controls_acceptance() {
        let controller = IController::new(0.9, 0.2, 5.0, 0.2).unwrap();
        assert_eq!(
            controller.evaluate(1.0).unwrap().decision(),
            StepDecision::Accept
        );
        assert_eq!(
            controller
                .evaluate(f64::from_bits(1.0_f64.to_bits() + 1))
                .unwrap()
                .decision(),
            StepDecision::Reject
        );
    }

    #[test]
    fn invalid_inputs_fail_closed() {
        assert!(matches!(
            IController::new(0.0, 0.1, 4.0, 0.5),
            Err(AdaptiveControlError::InvalidParameter {
                field: "safety",
                ..
            })
        ));
        assert!(matches!(
            IController::new(0.9, 5.0, 4.0, 0.5),
            Err(AdaptiveControlError::InvalidScaleBounds { .. })
        ));
        let controller = IController::new(0.9, 0.1, 4.0, 0.5).unwrap();
        assert!(matches!(
            controller.evaluate(f64::NAN),
            Err(AdaptiveControlError::InvalidNormalizedError(_))
        ));
        assert!(matches!(
            controller.evaluate(-1.0),
            Err(AdaptiveControlError::InvalidNormalizedError(_))
        ));
    }
}
