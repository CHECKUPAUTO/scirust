//! Exact group and point orders for the toy domain.

use scirust_modalg::numtheory::factor;

use crate::curve::{CurveError, ToyCurve, ToyPoint};

impl ToyCurve {
    /// Returns the group cardinality from canonical exhaustive enumeration.
    pub fn group_order(self) -> u64 {
        self.enumerate_points().len() as u64
    }

    /// Computes the exact additive order of a point.
    ///
    /// It starts from the group cardinality and removes prime factors only when
    /// the resulting scalar multiple is the group identity.
    pub fn point_order(self, point: ToyPoint) -> Result<u64, CurveError> {
        self.validate_point(point)?;

        if point.is_infinity()
        {
            return Ok(1);
        }

        let mut order = self.group_order();
        for (prime, exponent) in factor(order)
        {
            for _ in 0..exponent
            {
                let reduced_order = order / prime;
                if self.scalar_mul(point, reduced_order)?.is_infinity()
                {
                    order = reduced_order;
                }
                else
                {
                    break;
                }
            }
        }
        Ok(order)
    }

    /// Checks the Hasse interval exactly after exhaustive enumeration.
    pub fn satisfies_hasse_bound(self) -> bool {
        let group_order = self.group_order() as i128;
        let centre = self.prime().value() as i128 + 1;
        let deviation = group_order - centre;
        deviation * deviation <= 4 * self.prime().value() as i128
    }
}
