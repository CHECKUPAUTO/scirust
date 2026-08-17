//! Checked public boundaries for the legacy IBP verifier.
//!
//! The historical verifier predates `scirust-core`'s fallible public-API
//! convention and therefore contains assertion-based dimension checks. This
//! module adds structured validation without changing the numerical formulas or
//! breaking the existing API.

use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::error::{Result, SciRustError};
use crate::nn::ibp::{IbpLinear, IbpMlp, Interval, certified_robust};

fn invalid(message: impl Into<String>) -> SciRustError {
    SciRustError::InvalidConfig(message.into())
}

impl Interval {
    /// Build a finite, well-formed axis-aligned box.
    pub fn try_new(lo: Vec<f32>, hi: Vec<f32>) -> Result<Self> {
        let interval = Self { lo, hi };
        interval.validate()?;
        Ok(interval)
    }

    /// Validate the invariants required by every sound interval certificate.
    pub fn validate(&self) -> Result<()> {
        if self.lo.len() != self.hi.len()
        {
            return Err(invalid(format!(
                "IBP interval endpoint lengths differ: lo={}, hi={}",
                self.lo.len(),
                self.hi.len()
            )));
        }
        for (index, (&lo, &hi)) in self.lo.iter().zip(&self.hi).enumerate()
        {
            if !lo.is_finite() || !hi.is_finite()
            {
                return Err(invalid(format!(
                    "IBP interval endpoint {index} must be finite, got [{lo}, {hi}]"
                )));
            }
            if lo > hi
            {
                return Err(invalid(format!(
                    "IBP interval endpoint {index} is reversed: lo={lo} > hi={hi}"
                )));
            }
        }
        Ok(())
    }

    /// Checked point constructor rejecting non-finite coordinates.
    pub fn try_point(x: &[f32]) -> Result<Self> {
        Self::try_new(x.to_vec(), x.to_vec())
    }

    /// Checked L∞-ball constructor.
    ///
    /// `eps` must be finite and non-negative and every centre coordinate must be
    /// finite. Overflow in `x ± eps` is rejected by the resulting interval
    /// validation.
    pub fn try_around(x: &[f32], eps: f32) -> Result<Self> {
        if !eps.is_finite() || eps < 0.0
        {
            return Err(invalid(format!(
                "IBP radius must be finite and non-negative, got {eps}"
            )));
        }
        if let Some((index, value)) = x.iter().enumerate().find(|(_, value)| !value.is_finite())
        {
            return Err(invalid(format!(
                "IBP centre coordinate {index} must be finite, got {value}"
            )));
        }
        Self::try_new(
            x.iter().map(|&value| value - eps).collect(),
            x.iter().map(|&value| value + eps).collect(),
        )
    }

    /// Checked maximum radius.
    pub fn try_max_radius(&self) -> Result<f32> {
        self.validate()?;
        Ok(self
            .lo
            .iter()
            .zip(&self.hi)
            .map(|(&lo, &hi)| 0.5 * (hi - lo))
            .fold(0.0, f32::max))
    }
}

fn map_dimension_panic<T>(op: &'static str, f: impl FnOnce() -> T) -> Result<T> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|_| {
        invalid(format!(
            "{op}: input dimensions do not match the verifier/model contract"
        ))
    })
}

impl IbpLinear {
    /// Fallible interval propagation.
    ///
    /// Rejects malformed boxes before invoking the historical affine rule and
    /// converts its legacy dimension assertion into a structured error.
    pub fn try_forward_interval(&self, input: &Interval) -> Result<Interval> {
        input.validate()?;
        let output = map_dimension_panic("IbpLinear::forward_interval", || {
            self.forward_interval(input)
        })?;
        output.validate()?;
        Ok(output)
    }
}

impl IbpMlp {
    /// Fallible IBP certification boundary.
    pub fn try_certify(&self, input: &Interval) -> Result<Interval> {
        input.validate()?;
        let output = map_dimension_panic("IbpMlp::certify", || self.certify(input))?;
        output.validate()?;
        Ok(output)
    }

    /// Fallible zonotope certification boundary.
    pub fn try_certify_zonotope(&self, input: &Interval) -> Result<Interval> {
        input.validate()?;
        let output = map_dimension_panic("IbpMlp::certify_zonotope", || {
            self.certify_zonotope(input)
        })?;
        output.validate()?;
        Ok(output)
    }

    /// Fallible DeepPoly certification boundary.
    pub fn try_certify_deeppoly(&self, input: &Interval) -> Result<Interval> {
        input.validate()?;
        let output = map_dimension_panic("IbpMlp::certify_deeppoly", || {
            self.certify_deeppoly(input)
        })?;
        output.validate()?;
        Ok(output)
    }
}

/// Checked robustness verdict for a certified output interval.
///
/// Rejects malformed output boxes and target classes outside the output range
/// before delegating to the historical predicate.
pub fn try_certified_robust(out: &Interval, target: usize) -> Result<bool> {
    out.validate()?;
    if target >= out.lo.len()
    {
        return Err(invalid(format!(
            "certified_robust target {target} is outside {} output classes",
            out.lo.len()
        )));
    }
    Ok(certified_robust(out, target))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_rejects_mismatched_reversed_and_non_finite_bounds() {
        assert!(Interval::try_new(vec![0.0], vec![]).is_err());
        assert!(Interval::try_new(vec![1.0], vec![0.0]).is_err());
        assert!(Interval::try_new(vec![f32::NAN], vec![1.0]).is_err());
        assert!(Interval::try_new(vec![0.0], vec![f32::INFINITY]).is_err());
    }

    #[test]
    fn around_rejects_invalid_radius_and_centre() {
        assert!(Interval::try_around(&[0.0], -0.1).is_err());
        assert!(Interval::try_around(&[0.0], f32::NAN).is_err());
        assert!(Interval::try_around(&[f32::INFINITY], 0.1).is_err());
    }

    #[test]
    fn checked_linear_converts_dimension_assertion_to_error() {
        let linear = IbpLinear::new(vec![1.0, 2.0], vec![0.0], 2, 1);
        let wrong_width = Interval::try_new(vec![0.0], vec![1.0]).unwrap();
        assert!(linear.try_forward_interval(&wrong_width).is_err());
    }

    #[test]
    fn checked_mlp_rejects_malformed_or_wrong_width_input() {
        let linear = IbpLinear::new(vec![1.0, 2.0], vec![0.0], 2, 1);
        let mlp = IbpMlp::new(vec![linear]);
        let wrong_width = Interval::try_new(vec![0.0], vec![1.0]).unwrap();
        assert!(mlp.try_certify(&wrong_width).is_err());

        let malformed = Interval {
            lo: vec![0.0, 1.0],
            hi: vec![1.0],
        };
        assert!(mlp.try_certify(&malformed).is_err());
    }

    #[test]
    fn checked_verdict_rejects_bad_target() {
        let out = Interval::try_new(vec![2.0, 0.0], vec![3.0, 1.0]).unwrap();
        assert_eq!(try_certified_robust(&out, 0).unwrap(), true);
        assert!(try_certified_robust(&out, 2).is_err());
    }
}
