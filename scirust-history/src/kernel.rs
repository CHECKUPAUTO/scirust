//! Domain-neutral retained-history evaluation contract.
//!
//! A history kernel consumes an immutable [`crate::HistoryView`] and produces
//! one derived output. The contract intentionally says nothing about the
//! mathematics used by the kernel: fractional calculus, convolution, learned
//! recurrence, or any other rule remains owned by the implementing crate.

use crate::HistoryView;

/// Deterministic evaluation over an immutable retained-history view.
///
/// `Value` and `Position` are deliberately generic. Implementations decide
/// whether positions participate in the evaluation and define their own output
/// and typed error. Storage and retention remain owned by [`crate::HistoryBackend`].
pub trait HistoryKernel<Value, Position> {
    /// Derived value produced by one evaluation.
    type Output;

    /// Typed failure reported by the kernel implementation.
    type Error;

    /// Evaluate the retained history without mutating its storage or policy.
    fn evaluate(
        &self,
        history: &HistoryView<'_, Value, Position>,
    ) -> Result<Self::Output, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::HistoryKernel;
    use crate::{CompleteHistory, HistoryBackend, HistoryEntry, HistoryView};

    struct LastValue;

    impl HistoryKernel<u64, u64> for LastValue {
        type Output = Option<u64>;
        type Error = core::convert::Infallible;

        fn evaluate(
            &self,
            history: &HistoryView<'_, u64, u64>,
        ) -> Result<Self::Output, Self::Error> {
            Ok(history
                .get(history.retained_samples().saturating_sub(1))
                .map(|entry| *entry.value()))
        }
    }

    #[test]
    fn kernel_consumes_view_without_mutating_history() {
        let mut history = CompleteHistory::new();
        history.push(HistoryEntry::new(11_u64, 1_u64)).unwrap();
        history.push(HistoryEntry::new(29_u64, 4_u64)).unwrap();

        let before = history.clone();
        let value = LastValue.evaluate(&history.view()).unwrap();

        assert_eq!(value, Some(29));
        assert_eq!(history, before);
    }
}
