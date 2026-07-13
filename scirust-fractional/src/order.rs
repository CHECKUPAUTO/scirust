//! Validated fractional orders.

use crate::FractionalError;

/// A validated fractional order in the open interval `0 < alpha < 1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FractionalOrder(f64);

impl FractionalOrder {
    /// Validate and construct a fractional order.
    pub fn new(alpha: f64) -> Result<Self, FractionalError> {
        if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0
        {
            return Err(FractionalError::InvalidOrder(alpha));
        }

        Ok(Self(alpha))
    }

    /// Return the scalar order `alpha`.
    #[must_use]
    pub const fn value(self) -> f64 {
        self.0
    }
}
