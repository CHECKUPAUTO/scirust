//! Short-Weierstrass toy curves and exact group operations.

use std::error::Error;
use std::fmt;

use crate::field::{Fp, ToyPrime};

/// A nonsingular short-Weierstrass curve over a bounded toy prime field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ToyCurve {
    prime: ToyPrime,
    a: Fp,
    b: Fp,
}

impl ToyCurve {
    /// Creates y squared equals x cubed plus ax plus b after reducing coefficients.
    ///
    /// The curve is rejected when 4a cubed plus 27b squared is zero modulo p.
    pub fn new(prime: ToyPrime, a: u64, b: u64) -> Result<Self, CurveError> {
        let a = Fp::new(prime, a);
        let b = Fp::new(prime, b);
        let discriminant_core = a
            .square()
            .mul_same(a)
            .mul_same(Fp::new(prime, 4))
            .add_same(b.square().mul_same(Fp::new(prime, 27)));
        if discriminant_core.is_zero()
        {
            return Err(CurveError::Singular);
        }
        Ok(Self { prime, a, b })
    }

    /// Returns the curve's prime field.
    pub const fn prime(self) -> ToyPrime {
        self.prime
    }

    /// Returns the canonical a coefficient.
    pub const fn a(self) -> u64 {
        self.a.value()
    }

    /// Returns the canonical b coefficient.
    pub const fn b(self) -> u64 {
        self.b.value()
    }

    /// Returns the group identity for this curve.
    pub const fn identity(self) -> ToyPoint {
        ToyPoint {
            curve: self,
            coordinates: Coordinates::Infinity,
        }
    }

    /// Creates an affine point from canonical local field residues.
    ///
    /// Both coordinates must already be below the toy-field prime and must
    /// satisfy the curve equation. This API intentionally does not parse byte
    /// encodings or reduce unbounded external integers.
    pub fn point_from_local_residues(self, x: u64, y: u64) -> Result<ToyPoint, CurveError> {
        if x >= self.prime.value() || y >= self.prime.value()
        {
            return Err(CurveError::CoordinateOutsideField);
        }
        let point = ToyPoint {
            curve: self,
            coordinates: Coordinates::Affine {
                x: Fp::new(self.prime, x),
                y: Fp::new(self.prime, y),
            },
        };
        if self.is_on_curve(&point)
        {
            Ok(point)
        }
        else
        {
            Err(CurveError::PointNotOnCurve)
        }
    }

    /// Returns whether point belongs to this curve and satisfies its equation.
    pub fn is_on_curve(self, point: &ToyPoint) -> bool {
        if point.curve != self
        {
            return false;
        }
        match point.coordinates
        {
            Coordinates::Infinity => true,
            Coordinates::Affine { x, y } =>
            {
                let left = y.square();
                let right = x
                    .square()
                    .mul_same(x)
                    .add_same(self.a.mul_same(x))
                    .add_same(self.b);
                left == right
            },
        }
    }

    /// Returns the inverse of point.
    pub fn negate(self, point: ToyPoint) -> Result<ToyPoint, CurveError> {
        self.validate_point(point)?;
        match point.coordinates
        {
            Coordinates::Infinity => Ok(point),
            Coordinates::Affine { x, y } => Ok(self.affine_unchecked(x, y.neg())),
        }
    }

    /// Adds two points exactly using the short-Weierstrass group law.
    pub fn add(self, left: ToyPoint, right: ToyPoint) -> Result<ToyPoint, CurveError> {
        self.validate_point(left)?;
        self.validate_point(right)?;

        match (left.coordinates, right.coordinates)
        {
            (Coordinates::Infinity, _) => Ok(right),
            (_, Coordinates::Infinity) => Ok(left),
            (Coordinates::Affine { x: x1, y: y1 }, Coordinates::Affine { x: x2, y: y2 }) =>
            {
                if x1 == x2 && (y1 != y2 || y1.is_zero())
                {
                    return Ok(self.identity());
                }

                let slope = if x1 == x2
                {
                    let numerator = x1
                        .square()
                        .mul_same(Fp::new(self.prime, 3))
                        .add_same(self.a);
                    let denominator = y1.add_same(y1);
                    numerator
                        .checked_div(denominator)
                        .map_err(|_| CurveError::NonInvertibleDenominator)?
                }
                else
                {
                    y2.sub_same(y1)
                        .checked_div(x2.sub_same(x1))
                        .map_err(|_| CurveError::NonInvertibleDenominator)?
                };

                let x3 = slope.square().sub_same(x1).sub_same(x2);
                let y3 = slope.mul_same(x1.sub_same(x3)).sub_same(y1);
                Ok(self.affine_unchecked(x3, y3))
            },
        }
    }

    /// Computes a scalar multiple by exact double-and-add.
    pub fn scalar_mul(self, point: ToyPoint, mut scalar: u64) -> Result<ToyPoint, CurveError> {
        self.validate_point(point)?;
        let mut result = self.identity();
        let mut addend = point;

        while scalar != 0
        {
            if scalar & 1 == 1
            {
                result = self.add(result, addend)?;
            }
            scalar >>= 1;
            if scalar != 0
            {
                addend = self.add(addend, addend)?;
            }
        }

        Ok(result)
    }

    pub(crate) const fn affine_unchecked(self, x: Fp, y: Fp) -> ToyPoint {
        ToyPoint {
            curve: self,
            coordinates: Coordinates::Affine { x, y },
        }
    }

    pub(crate) fn validate_point(self, point: ToyPoint) -> Result<(), CurveError> {
        if point.curve != self
        {
            return Err(CurveError::PointFromAnotherCurve);
        }
        if !self.is_on_curve(&point)
        {
            return Err(CurveError::PointNotOnCurve);
        }
        Ok(())
    }
}

/// A point tied to the curve that created it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ToyPoint {
    curve: ToyCurve,
    coordinates: Coordinates,
}

impl ToyPoint {
    /// Returns whether this is the group identity.
    pub const fn is_infinity(self) -> bool {
        matches!(self.coordinates, Coordinates::Infinity)
    }

    /// Returns local affine residues, or none for the group identity.
    pub const fn affine_coordinates(self) -> Option<(u64, u64)> {
        match self.coordinates
        {
            Coordinates::Infinity => None,
            Coordinates::Affine { x, y } => Some((x.value(), y.value())),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Coordinates {
    Infinity,
    Affine { x: Fp, y: Fp },
}

/// An invalid curve or point operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurveError {
    /// The curve has zero discriminant.
    Singular,
    /// A point belongs to a different curve.
    PointFromAnotherCurve,
    /// A coordinate pair does not satisfy the curve equation.
    PointNotOnCurve,
    /// A local point coordinate is not a canonical residue of this toy field.
    CoordinateOutsideField,
    /// A group-law denominator unexpectedly had no inverse.
    NonInvertibleDenominator,
}

impl fmt::Display for CurveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::Singular => write!(formatter, "short-Weierstrass curve is singular"),
            Self::PointFromAnotherCurve => write!(formatter, "point belongs to another toy curve"),
            Self::PointNotOnCurve => write!(formatter, "point does not satisfy the curve equation"),
            Self::CoordinateOutsideField =>
            {
                write!(formatter, "point coordinate is outside the toy prime field")
            },
            Self::NonInvertibleDenominator =>
            {
                write!(formatter, "group-law denominator is not invertible")
            },
        }
    }
}

impl Error for CurveError {}
