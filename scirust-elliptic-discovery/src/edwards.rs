//! Exact twisted-Edwards curves over the bounded toy prime-field domain.
//!
//! This module is deliberately affine and explicit. Generic twisted-Edwards
//! parameters do not guarantee a complete addition law, so any zero addition
//! denominator is reported instead of fabricating a point or silently changing
//! coordinate systems.

use std::error::Error;
use std::fmt;

use crate::field::{FieldError, Fp, ToyPrime};

/// A nonsingular twisted-Edwards curve
/// `a*x^2 + y^2 = 1 + d*x^2*y^2` over one [`ToyPrime`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TwistedEdwardsCurve {
    prime: ToyPrime,
    a: Fp,
    d: Fp,
}

/// One validated affine point on a [`TwistedEdwardsCurve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TwistedEdwardsPoint {
    x: Fp,
    y: Fp,
}

/// Failure while constructing or operating on a toy twisted-Edwards curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TwistedEdwardsError {
    /// The curve is singular because `a == 0`, `d == 0`, or `a == d`.
    SingularCurve,
    /// A supplied point does not satisfy the curve equation.
    PointNotOnCurve,
    /// The generic affine addition law encountered a zero denominator.
    ExceptionalDenominator,
    /// Arithmetic mixed residues from distinct prime fields.
    Field(FieldError),
}

impl fmt::Display for TwistedEdwardsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SingularCurve => write!(
                formatter,
                "twisted-Edwards parameters must satisfy a != 0, d != 0 and a != d"
            ),
            Self::PointNotOnCurve => write!(formatter, "point is not on this twisted-Edwards curve"),
            Self::ExceptionalDenominator => write!(
                formatter,
                "twisted-Edwards affine addition encountered a zero denominator"
            ),
            Self::Field(error) => write!(formatter, "{error}"),
        }
    }
}

impl Error for TwistedEdwardsError {}

impl From<FieldError> for TwistedEdwardsError {
    fn from(error: FieldError) -> Self {
        Self::Field(error)
    }
}

impl TwistedEdwardsCurve {
    /// Construct a nonsingular twisted-Edwards curve from canonical toy-field
    /// parameters. `a` and `d` are reduced modulo `prime`.
    pub fn new(prime: ToyPrime, a: u64, d: u64) -> Result<Self, TwistedEdwardsError> {
        let a = Fp::new(prime, a);
        let d = Fp::new(prime, d);
        if a.is_zero() || d.is_zero() || a == d {
            return Err(TwistedEdwardsError::SingularCurve);
        }
        Ok(Self { prime, a, d })
    }

    /// Prime field of the curve.
    #[must_use]
    pub const fn prime(self) -> ToyPrime {
        self.prime
    }

    /// Twisted coefficient `a`.
    #[must_use]
    pub const fn a(self) -> Fp {
        self.a
    }

    /// Twisted coefficient `d`.
    #[must_use]
    pub const fn d(self) -> Fp {
        self.d
    }

    /// Neutral element `(0, 1)`.
    #[must_use]
    pub const fn identity(self) -> TwistedEdwardsPoint {
        TwistedEdwardsPoint {
            x: Fp::new(self.prime, 0),
            y: Fp::new(self.prime, 1),
        }
    }

    /// Validate and construct one affine point.
    pub fn point(self, x: u64, y: u64) -> Result<TwistedEdwardsPoint, TwistedEdwardsError> {
        let point = TwistedEdwardsPoint {
            x: Fp::new(self.prime, x),
            y: Fp::new(self.prime, y),
        };
        self.validate_point(point)?;
        Ok(point)
    }

    /// Returns whether the exact affine coordinates satisfy this curve.
    #[must_use]
    pub fn contains(self, point: TwistedEdwardsPoint) -> bool {
        self.contains_checked(point).unwrap_or(false)
    }

    /// Add two validated affine points using the exact generic
    /// twisted-Edwards law.
    pub fn add(
        self,
        left: TwistedEdwardsPoint,
        right: TwistedEdwardsPoint,
    ) -> Result<TwistedEdwardsPoint, TwistedEdwardsError> {
        self.validate_point(left)?;
        self.validate_point(right)?;

        let x1x2 = left.x.checked_mul(right.x)?;
        let y1y2 = left.y.checked_mul(right.y)?;
        let cross = x1x2.checked_mul(y1y2)?;
        let d_cross = self.d.checked_mul(cross)?;
        let one = Fp::new(self.prime, 1);
        let x_denominator = one.checked_add(d_cross)?;
        let y_denominator = one.checked_sub(d_cross)?;
        if x_denominator.is_zero() || y_denominator.is_zero() {
            return Err(TwistedEdwardsError::ExceptionalDenominator);
        }

        let x_numerator = left
            .x
            .checked_mul(right.y)?
            .checked_add(left.y.checked_mul(right.x)?)?;
        let y_numerator = y1y2.checked_sub(self.a.checked_mul(x1x2)?)?;
        let result = TwistedEdwardsPoint {
            x: x_numerator.checked_div(x_denominator)?,
            y: y_numerator.checked_div(y_denominator)?,
        };
        self.validate_point(result)?;
        Ok(result)
    }

    /// Exact doubling through the same checked affine law.
    pub fn double(
        self,
        point: TwistedEdwardsPoint,
    ) -> Result<TwistedEdwardsPoint, TwistedEdwardsError> {
        self.add(point, point)
    }

    /// Additive inverse `(-x, y)`.
    pub fn neg(
        self,
        point: TwistedEdwardsPoint,
    ) -> Result<TwistedEdwardsPoint, TwistedEdwardsError> {
        self.validate_point(point)?;
        Ok(TwistedEdwardsPoint {
            x: point.x.neg(),
            y: point.y,
        })
    }

    /// Deterministic double-and-add scalar multiplication. The generic affine
    /// law remains fail-closed if an exceptional denominator is encountered.
    pub fn scalar_mul(
        self,
        mut scalar: u64,
        point: TwistedEdwardsPoint,
    ) -> Result<TwistedEdwardsPoint, TwistedEdwardsError> {
        self.validate_point(point)?;
        let mut accumulator = self.identity();
        let mut addend = point;
        while scalar != 0 {
            if scalar & 1 == 1 {
                accumulator = self.add(accumulator, addend)?;
            }
            scalar >>= 1;
            if scalar != 0 {
                addend = self.double(addend)?;
            }
        }
        Ok(accumulator)
    }

    /// Enumerate every affine point in canonical `(x, y)` order. The bounded
    /// toy field makes the exhaustive `p^2` scan intentional and reproducible.
    #[must_use]
    pub fn enumerate(self) -> Vec<TwistedEdwardsPoint> {
        let p = self.prime.value();
        let mut points = Vec::new();
        for x in 0..p {
            for y in 0..p {
                let point = TwistedEdwardsPoint {
                    x: Fp::new(self.prime, x),
                    y: Fp::new(self.prime, y),
                };
                if self.contains(point) {
                    points.push(point);
                }
            }
        }
        points
    }

    fn validate_point(self, point: TwistedEdwardsPoint) -> Result<(), TwistedEdwardsError> {
        if self.contains_checked(point)? {
            Ok(())
        } else {
            Err(TwistedEdwardsError::PointNotOnCurve)
        }
    }

    fn contains_checked(self, point: TwistedEdwardsPoint) -> Result<bool, FieldError> {
        if point.x.prime() != self.prime || point.y.prime() != self.prime {
            return Err(FieldError::DifferentPrimes);
        }
        let x2 = point.x.checked_mul(point.x)?;
        let y2 = point.y.checked_mul(point.y)?;
        let left = self.a.checked_mul(x2)?.checked_add(y2)?;
        let right = Fp::new(self.prime, 1).checked_add(self.d.checked_mul(x2.checked_mul(y2)?)?)?;
        Ok(left == right)
    }
}

impl TwistedEdwardsPoint {
    /// Affine x-coordinate.
    #[must_use]
    pub const fn x(self) -> Fp {
        self.x
    }

    /// Affine y-coordinate.
    #[must_use]
    pub const fn y(self) -> Fp {
        self.y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve() -> TwistedEdwardsCurve {
        TwistedEdwardsCurve::new(ToyPrime::new(17).unwrap(), 1, 3).unwrap()
    }

    #[test]
    fn rejects_singular_parameter_sets() {
        let prime = ToyPrime::new(17).unwrap();
        assert_eq!(
            TwistedEdwardsCurve::new(prime, 0, 3),
            Err(TwistedEdwardsError::SingularCurve)
        );
        assert_eq!(
            TwistedEdwardsCurve::new(prime, 3, 3),
            Err(TwistedEdwardsError::SingularCurve)
        );
    }

    #[test]
    fn validates_identity_and_rejects_off_curve_point() {
        let curve = curve();
        assert!(curve.contains(curve.identity()));
        assert_eq!(curve.point(0, 0), Err(TwistedEdwardsError::PointNotOnCurve));
    }

    #[test]
    fn exact_addition_identity_inverse_and_doubling_hold() {
        let curve = curve();
        let point = curve.point(2, 5).unwrap();
        assert_eq!(curve.add(point, curve.identity()).unwrap(), point);
        assert_eq!(
            curve.add(point, curve.neg(point).unwrap()).unwrap(),
            curve.identity()
        );
        assert_eq!(curve.double(point).unwrap(), curve.point(13, 3).unwrap());
    }

    #[test]
    fn scalar_multiplication_matches_repeated_addition() {
        let curve = curve();
        let point = curve.point(3, 4).unwrap();
        let mut repeated = curve.identity();
        for _ in 0..7 {
            repeated = curve.add(repeated, point).unwrap();
        }
        assert_eq!(curve.scalar_mul(7, point).unwrap(), repeated);
        assert_eq!(curve.scalar_mul(0, point).unwrap(), curve.identity());
    }

    #[test]
    fn enumeration_is_exact_ordered_and_closed() {
        let curve = curve();
        let points = curve.enumerate();
        assert_eq!(points.len(), 24);
        assert_eq!(points[0], curve.point(0, 1).unwrap());
        for point in points {
            assert!(curve.contains(point));
        }
    }
}
