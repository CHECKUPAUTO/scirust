//! Exact Montgomery curves over the bounded toy prime-field domain.
//!
//! The affine model is `B*y^2 = x^3 + A*x^2 + x`. The neutral element is
//! represented explicitly as the point at infinity; all arithmetic stays inside
//! the verified toy prime field and uses checked affine formulas.

use std::error::Error;
use std::fmt;

use crate::field::{FieldError, Fp, ToyPrime};

/// A nonsingular Montgomery curve `B*y^2 = x^3 + A*x^2 + x`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MontgomeryCurve {
    prime: ToyPrime,
    a: Fp,
    b: Fp,
}

/// One validated point on a [`MontgomeryCurve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MontgomeryPoint {
    /// Neutral element of the group.
    Infinity,
    /// Affine point with canonical prime-field coordinates.
    Affine {
        /// Affine x-coordinate.
        x: Fp,
        /// Affine y-coordinate.
        y: Fp,
    },
}

/// Failure while constructing or operating on a toy Montgomery curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MontgomeryError {
    /// The model is singular: `B == 0` or `A^2 == 4`.
    SingularCurve,
    /// A supplied affine point does not satisfy the curve equation.
    PointNotOnCurve,
    /// Arithmetic mixed residues from distinct prime fields.
    Field(FieldError),
}

impl fmt::Display for MontgomeryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::SingularCurve => write!(
                formatter,
                "Montgomery parameters must satisfy B != 0 and A^2 != 4"
            ),
            Self::PointNotOnCurve => write!(formatter, "point is not on this Montgomery curve"),
            Self::Field(error) => write!(formatter, "{error}"),
        }
    }
}

impl Error for MontgomeryError {}

impl From<FieldError> for MontgomeryError {
    fn from(error: FieldError) -> Self {
        Self::Field(error)
    }
}

impl MontgomeryCurve {
    /// Construct a nonsingular Montgomery curve. `A` and `B` are reduced modulo
    /// `prime` before the nonsingularity test.
    pub fn new(prime: ToyPrime, a: u64, b: u64) -> Result<Self, MontgomeryError> {
        let a = Fp::new(prime, a);
        let b = Fp::new(prime, b);
        let four = Fp::new(prime, 4);
        if b.is_zero() || a.checked_mul(a)? == four
        {
            return Err(MontgomeryError::SingularCurve);
        }
        Ok(Self { prime, a, b })
    }

    /// Prime field of the curve.
    #[must_use]
    pub const fn prime(self) -> ToyPrime {
        self.prime
    }

    /// Montgomery coefficient `A`.
    #[must_use]
    pub const fn a(self) -> Fp {
        self.a
    }

    /// Montgomery coefficient `B`.
    #[must_use]
    pub const fn b(self) -> Fp {
        self.b
    }

    /// Neutral point at infinity.
    #[must_use]
    pub const fn identity(self) -> MontgomeryPoint {
        MontgomeryPoint::Infinity
    }

    /// Validate and construct one affine point.
    pub fn point(self, x: u64, y: u64) -> Result<MontgomeryPoint, MontgomeryError> {
        let point = MontgomeryPoint::Affine {
            x: Fp::new(self.prime, x),
            y: Fp::new(self.prime, y),
        };
        self.validate_point(point)?;
        Ok(point)
    }

    /// Return whether a point belongs to this curve.
    #[must_use]
    pub fn contains(self, point: MontgomeryPoint) -> bool {
        self.contains_checked(point).unwrap_or(false)
    }

    /// Add two validated points using exact affine Montgomery formulas.
    pub fn add(
        self,
        left: MontgomeryPoint,
        right: MontgomeryPoint,
    ) -> Result<MontgomeryPoint, MontgomeryError> {
        self.validate_point(left)?;
        self.validate_point(right)?;
        match (left, right)
        {
            (MontgomeryPoint::Infinity, point) | (point, MontgomeryPoint::Infinity) => Ok(point),
            (
                MontgomeryPoint::Affine { x: x1, y: y1 },
                MontgomeryPoint::Affine { x: x2, y: y2 },
            ) =>
            {
                if x1 == x2
                {
                    if y1.checked_add(y2)?.is_zero()
                    {
                        return Ok(MontgomeryPoint::Infinity);
                    }
                    return self.double(left);
                }

                let slope = y2.checked_sub(y1)?.checked_div(x2.checked_sub(x1)?)?;
                self.finish_addition(x1, y1, x2, slope)
            },
        }
    }

    /// Double one validated point.
    pub fn double(self, point: MontgomeryPoint) -> Result<MontgomeryPoint, MontgomeryError> {
        self.validate_point(point)?;
        let MontgomeryPoint::Affine { x, y } = point
        else
        {
            return Ok(MontgomeryPoint::Infinity);
        };
        if y.is_zero()
        {
            return Ok(MontgomeryPoint::Infinity);
        }

        let three = Fp::new(self.prime, 3);
        let two = Fp::new(self.prime, 2);
        let one = Fp::new(self.prime, 1);
        let numerator = three
            .checked_mul(x.checked_mul(x)?)?
            .checked_add(two.checked_mul(self.a)?.checked_mul(x)?)?
            .checked_add(one)?;
        let denominator = two.checked_mul(self.b)?.checked_mul(y)?;
        let slope = numerator.checked_div(denominator)?;
        self.finish_addition(x, y, x, slope)
    }

    /// Additive inverse `(x, -y)`; infinity is self-inverse.
    pub fn neg(self, point: MontgomeryPoint) -> Result<MontgomeryPoint, MontgomeryError> {
        self.validate_point(point)?;
        Ok(match point
        {
            MontgomeryPoint::Infinity => MontgomeryPoint::Infinity,
            MontgomeryPoint::Affine { x, y } => MontgomeryPoint::Affine { x, y: y.neg() },
        })
    }

    /// Deterministic double-and-add scalar multiplication.
    pub fn scalar_mul(
        self,
        mut scalar: u64,
        point: MontgomeryPoint,
    ) -> Result<MontgomeryPoint, MontgomeryError> {
        self.validate_point(point)?;
        let mut accumulator = self.identity();
        let mut addend = point;
        while scalar != 0
        {
            if scalar & 1 == 1
            {
                accumulator = self.add(accumulator, addend)?;
            }
            scalar >>= 1;
            if scalar != 0
            {
                addend = self.double(addend)?;
            }
        }
        Ok(accumulator)
    }

    /// Enumerate infinity followed by every affine point in canonical `(x,y)`
    /// order. The deliberate `p <= 4093` toy boundary makes exhaustive scanning
    /// reproducible and bounded.
    #[must_use]
    pub fn enumerate(self) -> Vec<MontgomeryPoint> {
        let p = self.prime.value();
        let mut points = vec![MontgomeryPoint::Infinity];
        for x in 0..p
        {
            for y in 0..p
            {
                let point = MontgomeryPoint::Affine {
                    x: Fp::new(self.prime, x),
                    y: Fp::new(self.prime, y),
                };
                if self.contains(point)
                {
                    points.push(point);
                }
            }
        }
        points
    }

    fn finish_addition(
        self,
        x1: Fp,
        y1: Fp,
        x2: Fp,
        slope: Fp,
    ) -> Result<MontgomeryPoint, MontgomeryError> {
        let x3 = self
            .b
            .checked_mul(slope.checked_mul(slope)?)?
            .checked_sub(self.a)?
            .checked_sub(x1)?
            .checked_sub(x2)?;
        let y3 = slope.checked_mul(x1.checked_sub(x3)?)?.checked_sub(y1)?;
        let result = MontgomeryPoint::Affine { x: x3, y: y3 };
        self.validate_point(result)?;
        Ok(result)
    }

    fn validate_point(self, point: MontgomeryPoint) -> Result<(), MontgomeryError> {
        if self.contains_checked(point)?
        {
            Ok(())
        }
        else
        {
            Err(MontgomeryError::PointNotOnCurve)
        }
    }

    fn contains_checked(self, point: MontgomeryPoint) -> Result<bool, FieldError> {
        let MontgomeryPoint::Affine { x, y } = point
        else
        {
            return Ok(true);
        };
        if x.prime() != self.prime || y.prime() != self.prime
        {
            return Err(FieldError::DifferentPrimes);
        }
        let x2 = x.checked_mul(x)?;
        let x3 = x2.checked_mul(x)?;
        let left = self.b.checked_mul(y.checked_mul(y)?)?;
        let right = x3.checked_add(self.a.checked_mul(x2)?)?.checked_add(x)?;
        Ok(left == right)
    }
}

impl MontgomeryPoint {
    /// Affine coordinates, or `None` for the point at infinity.
    #[must_use]
    pub const fn affine(self) -> Option<(Fp, Fp)> {
        match self
        {
            Self::Infinity => None,
            Self::Affine { x, y } => Some((x, y)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve() -> MontgomeryCurve {
        MontgomeryCurve::new(ToyPrime::new(17).unwrap(), 3, 1).unwrap()
    }

    #[test]
    fn rejects_singular_parameter_sets() {
        let prime = ToyPrime::new(17).unwrap();
        assert_eq!(
            MontgomeryCurve::new(prime, 2, 0),
            Err(MontgomeryError::SingularCurve)
        );
        assert_eq!(MontgomeryCurve::new(prime, 3, 1), Ok(curve()));
        assert_eq!(
            MontgomeryCurve::new(prime, 15, 1),
            Err(MontgomeryError::SingularCurve)
        );
    }

    #[test]
    fn validates_identity_and_rejects_off_curve_point() {
        let curve = curve();
        assert!(curve.contains(curve.identity()));
        assert_eq!(curve.point(0, 1), Err(MontgomeryError::PointNotOnCurve));
    }

    #[test]
    fn exact_group_identities_hold() {
        let curve = curve();
        let point = curve.point(16, 1).unwrap();
        assert_eq!(curve.add(point, curve.identity()).unwrap(), point);
        assert_eq!(
            curve.add(point, curve.neg(point).unwrap()).unwrap(),
            curve.identity()
        );
        assert_eq!(curve.double(point).unwrap(), curve.point(0, 0).unwrap());
        assert_eq!(curve.scalar_mul(4, point).unwrap(), curve.identity());
    }

    #[test]
    fn scalar_multiplication_matches_repeated_addition() {
        let curve = curve();
        let point = curve.point(5, 1).unwrap();
        let mut repeated = curve.identity();
        for _ in 0..7
        {
            repeated = curve.add(repeated, point).unwrap();
        }
        assert_eq!(curve.scalar_mul(7, point).unwrap(), repeated);
        assert_eq!(curve.scalar_mul(0, point).unwrap(), curve.identity());
    }

    #[test]
    fn enumeration_is_exact_ordered_and_closed() {
        let curve = curve();
        let points = curve.enumerate();
        assert_eq!(points.len(), 16);
        assert_eq!(points[0], MontgomeryPoint::Infinity);
        assert_eq!(points[1], curve.point(0, 0).unwrap());
        for point in points
        {
            assert!(curve.contains(point));
        }
    }
}
