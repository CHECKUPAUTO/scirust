//! Caputo L1 adapter for the domain-neutral `scirust-history` kernel contract.
//!
//! The numerical implementation remains [`crate::caputo_l1_nonuniform`]. This
//! module only assembles scalar samples and their true retained positions from
//! a [`scirust_history::HistoryView`] before delegating to that operator.

use scirust_history::{HistoryKernel, HistoryView};

use crate::{FractionalError, FractionalOrder, caputo_l1_nonuniform};

/// Left-sided non-uniform Caputo L1 evaluation over scalar retained history.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CaputoL1HistoryKernel {
    order: FractionalOrder,
}

impl CaputoL1HistoryKernel {
    /// Construct a history kernel from an already validated fractional order.
    #[must_use]
    pub const fn new(order: FractionalOrder) -> Self {
        Self { order }
    }

    /// Return the configured fractional order.
    #[must_use]
    pub const fn order(self) -> FractionalOrder {
        self.order
    }
}

impl HistoryKernel<f64, f64> for CaputoL1HistoryKernel {
    type Output = f64;
    type Error = FractionalError;

    fn evaluate(&self, history: &HistoryView<'_, f64, f64>) -> Result<Self::Output, Self::Error> {
        let retained = history.retained_samples();
        let mut samples = Vec::with_capacity(retained);
        let mut positions = Vec::with_capacity(retained);

        for entry in history.iter()
        {
            samples.push(*entry.value());
            positions.push(*entry.position());
        }

        caputo_l1_nonuniform(&samples, &positions, self.order)
    }
}
