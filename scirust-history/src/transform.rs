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

/// Contextual scalar weighting of one retained entry.
///
/// Weighting is separate from age/position kernels: implementations may inspect
/// the retained entry and an explicit context, but they do not own retention or
/// history aggregation.
pub trait HistoryWeight<Value, Position, Context> {
    /// Typed weighting failure.
    type Error;

    /// Return the contextual scalar weight for `entry`.
    fn weight(
        &self,
        entry: &HistoryEntry<Value, Position>,
        context: &Context,
    ) -> Result<f64, Self::Error>;
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
    ) -> Result<f64, Self::Error> {
        Ok(1.0)
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

/// Validate a generic scalar history weight without imposing a sign policy.
///
/// Some downstream kernels may legitimately use signed weights; positivity is
/// therefore a specialization-level policy. The generic contract only rejects
/// non-finite values, which cannot safely participate in deterministic numeric
/// aggregation.
pub fn validate_history_weight(weight: f64) -> Result<f64, HistoryWeightError> {
    if weight.is_finite()
    {
        Ok(weight)
    }
    else
    {
        Err(HistoryWeightError::NonFinite(weight))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        HistoryTransform, HistoryWeight, HistoryWeightError, IdentityHistoryTransform,
        IdentityHistoryWeight, validate_history_weight,
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
        assert_eq!(weight.to_bits(), 1.0_f64.to_bits());
    }

    #[test]
    fn non_finite_weights_fail_closed() {
        for weight in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY]
        {
            assert!(matches!(
                validate_history_weight(weight),
                Err(HistoryWeightError::NonFinite(actual)) if actual.to_bits() == weight.to_bits()
            ));
        }

        assert_eq!(
            validate_history_weight(-2.5).unwrap().to_bits(),
            (-2.5_f64).to_bits()
        );
        assert_eq!(
            validate_history_weight(0.0).unwrap().to_bits(),
            0.0_f64.to_bits()
        );
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
