//! Domain-neutral history transformation and contextual weighting contracts.
//!
//! These contracts are deliberately independent from [`crate::HistoryKernel`]:
//! transforming a retained value into the representation needed "now" and
//! assigning a contextual scalar weight answer different questions from the
//! kernel that aggregates history over its logical positions.

use core::convert::Infallible;
use core::fmt;

use crate::HistoryEntry;

/// Transform one retained entry into a representation suitable for a current context.
///
/// `Context` is an explicit type parameter so a specialization that needs source
/// metadata cannot obtain it from ambient state. If required metadata may be absent,
/// the specialization must represent that possibility in `Context` and return a typed
/// error rather than inventing a source point or other default.
pub trait HistoryTransform<Value, Position, Context> {
    /// Transformed retained value.
    type Output;

    /// Typed transformation failure.
    type Error;

    /// Transform `entry` for `context` without mutating retained storage.
    fn transform(
        &self,
        entry: &HistoryEntry<Value, Position>,
        context: &Context,
    ) -> Result<Self::Output, Self::Error>;
}

/// Exact identity transform.
///
/// The implementation performs no arithmetic and returns `Value::clone()`.
/// For ordinary scalar/array values this preserves the stored representation,
/// including floating-point bit patterns.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct IdentityHistoryTransform;

impl<Value, Position, Context> HistoryTransform<Value, Position, Context>
    for IdentityHistoryTransform
where
    Value: Clone,
{
    type Output = Value;
    type Error = Infallible;

    fn transform(
        &self,
        entry: &HistoryEntry<Value, Position>,
        _context: &Context,
    ) -> Result<Self::Output, Self::Error> {
        Ok(entry.value().clone())
    }
}

/// Validation failure for a generic scalar history weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HistoryWeightError {
    /// The proposed contextual weight is NaN or infinite.
    NonFinite(f64),
}

impl fmt::Display for HistoryWeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::NonFinite(weight) => write!(f, "history weight must be finite, got {weight}"),
        }
    }
}

/// A scalar history weight whose finiteness has been validated.
///
/// The inner value is private so downstream implementations of [`HistoryWeight`]
/// cannot manufacture an unchecked `NaN` or infinity. Signed and zero weights
/// remain valid: sign policy belongs to the downstream specialization, while
/// this generic contract enforces only numerical finiteness.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct FiniteHistoryWeight(f64);

impl FiniteHistoryWeight {
    /// Exact multiplicative identity.
    pub const ONE: Self = Self(1.0);

    /// Validate and construct a finite history weight.
    ///
    /// # Errors
    ///
    /// Returns [`HistoryWeightError::NonFinite`] for NaN or either infinity.
    pub fn new(weight: f64) -> Result<Self, HistoryWeightError> {
        if weight.is_finite()
        {
            Ok(Self(weight))
        }
        else
        {
            Err(HistoryWeightError::NonFinite(weight))
        }
    }

    /// Return the validated scalar value.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

/// Contextual scalar weighting of one retained entry.
///
/// Weighting is separate from age/position kernels: implementations may inspect
/// the retained entry and an explicit context, but they do not own retention or
/// history aggregation. The validated return type makes the finite-weight
/// invariant mandatory for every implementation rather than an optional caller
/// convention.
pub trait HistoryWeight<Value, Position, Context> {
    /// Typed weighting failure.
    type Error;

    /// Return the contextual finite scalar weight for `entry`.
    fn weight(
        &self,
        entry: &HistoryEntry<Value, Position>,
        context: &Context,
    ) -> Result<FiniteHistoryWeight, Self::Error>;
}

/// Identity contextual weighting, returning exactly `1.0`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct IdentityHistoryWeight;

impl<Value, Position, Context> HistoryWeight<Value, Position, Context> for IdentityHistoryWeight {
    type Error = Infallible;

    fn weight(
        &self,
        _entry: &HistoryEntry<Value, Position>,
        _context: &Context,
    ) -> Result<FiniteHistoryWeight, Self::Error> {
        Ok(FiniteHistoryWeight::ONE)
    }
}

/// Validate a generic scalar history weight without imposing a sign policy.
///
/// This convenience constructor is equivalent to [`FiniteHistoryWeight::new`].
/// Some downstream kernels may legitimately use signed weights; positivity is
/// therefore a specialization-level policy. The generic contract only rejects
/// non-finite values, which cannot safely participate in deterministic numeric
/// aggregation.
pub fn validate_history_weight(weight: f64) -> Result<FiniteHistoryWeight, HistoryWeightError> {
    FiniteHistoryWeight::new(weight)
}

#[cfg(test)]
mod tests {
    use super::{
        FiniteHistoryWeight, HistoryTransform, HistoryWeight, HistoryWeightError,
        IdentityHistoryTransform, IdentityHistoryWeight, validate_history_weight,
    };
    use crate::HistoryEntry;

    #[test]
    fn identity_transform_preserves_floating_point_bits() {
        let value = [
            f64::from_bits(1),
            -0.0,
            7.25,
            f64::from_bits(0x7fe0_0000_0000_0001),
        ];
        let entry = HistoryEntry::new(value, 0.375_f64);
        let transformed = IdentityHistoryTransform.transform(&entry, &()).unwrap();

        for (actual, expected) in transformed.iter().zip(value.iter())
        {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn identity_weight_is_exact_multiplicative_identity() {
        let entry = HistoryEntry::new(42_u64, 3_u64);
        let weight = IdentityHistoryWeight.weight(&entry, &()).unwrap();
        assert_eq!(weight.get().to_bits(), 1.0_f64.to_bits());
        assert_eq!(weight, FiniteHistoryWeight::ONE);
    }

    #[test]
    fn non_finite_weights_fail_closed_by_construction() {
        for weight in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY]
        {
            assert!(matches!(
                validate_history_weight(weight),
                Err(HistoryWeightError::NonFinite(actual)) if actual.to_bits() == weight.to_bits()
            ));
            assert!(matches!(
                FiniteHistoryWeight::new(weight),
                Err(HistoryWeightError::NonFinite(actual)) if actual.to_bits() == weight.to_bits()
            ));
        }

        assert_eq!(
            validate_history_weight(-2.5).unwrap().get().to_bits(),
            (-2.5_f64).to_bits()
        );
        assert_eq!(
            validate_history_weight(0.0).unwrap().get().to_bits(),
            0.0_f64.to_bits()
        );
    }

    #[test]
    fn custom_weight_cannot_bypass_finite_return_type() {
        struct SignedScale;

        impl HistoryWeight<f64, f64, f64> for SignedScale {
            type Error = HistoryWeightError;

            fn weight(
                &self,
                _entry: &HistoryEntry<f64, f64>,
                scale: &f64,
            ) -> Result<FiniteHistoryWeight, Self::Error> {
                FiniteHistoryWeight::new(*scale)
            }
        }

        let entry = HistoryEntry::new(3.0_f64, 1.0_f64);
        assert_eq!(
            SignedScale.weight(&entry, &-2.0).unwrap().get().to_bits(),
            (-2.0_f64).to_bits()
        );
        assert!(matches!(
            SignedScale.weight(&entry, &f64::NAN),
            Err(HistoryWeightError::NonFinite(weight)) if weight.is_nan()
        ));
    }

    #[test]
    fn required_metadata_is_explicit_in_transform_context() {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum MetadataError {
            MissingSource,
        }

        struct SourceScale;

        impl HistoryTransform<f64, f64, Option<f64>> for SourceScale {
            type Output = f64;
            type Error = MetadataError;

            fn transform(
                &self,
                entry: &HistoryEntry<f64, f64>,
                source_scale: &Option<f64>,
            ) -> Result<Self::Output, Self::Error> {
                let scale = source_scale.ok_or(MetadataError::MissingSource)?;
                Ok(*entry.value() * scale)
            }
        }

        let entry = HistoryEntry::new(3.0_f64, 1.0_f64);
        assert_eq!(
            SourceScale.transform(&entry, &None),
            Err(MetadataError::MissingSource)
        );
        assert_eq!(SourceScale.transform(&entry, &Some(2.0)).unwrap(), 6.0);
    }
}
