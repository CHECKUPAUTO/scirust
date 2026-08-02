#![forbid(unsafe_code)]
#![cfg(feature = "portable-simd")]
//! Hypercomplex Elliptic Curves over Cayley-Dickson Octonions and Sedenions.
//! Optimized for SIMD vector instructions and strictly zero-allocation.

use scirust_simd::hypercomplex::{OctonionSimd, SedenionSimd};

/// An affine point on an Octonionic Elliptic Curve, or the Point at Infinity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OctonionPoint {
    Infinity,
    Affine { x: OctonionSimd, y: OctonionSimd },
}

/// An Octonionic Elliptic Curve defined by Y^2 = X^3 + A*X + B.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OctonionCurve {
    pub a: OctonionSimd,
    pub b: OctonionSimd,
}

impl OctonionCurve {
    #[inline(always)]
    pub fn new(a: OctonionSimd, b: OctonionSimd) -> Self {
        Self { a, b }
    }

    /// Checks if a point lies on the curve.
    #[inline(always)]
    pub fn is_on_curve(&self, point: &OctonionPoint) -> bool {
        match point
        {
            OctonionPoint::Infinity => true,
            OctonionPoint::Affine { x, y } =>
            {
                let left = *y * *y;
                let right = (*x * *x * *x) + (self.a * *x) + self.b;
                // Since floats have precision limits, we check within a small epsilon tolerance.
                (left - right).norm() < 1e-4
            },
        }
    }

    /// Adds two points on the octonionic curve.
    #[inline(always)]
    pub fn add(&self, p1: OctonionPoint, p2: OctonionPoint) -> OctonionPoint {
        match (p1, p2)
        {
            (OctonionPoint::Infinity, _) => p2,
            (_, OctonionPoint::Infinity) => p1,
            (OctonionPoint::Affine { x: x1, y: y1 }, OctonionPoint::Affine { x: x2, y: y2 }) =>
            {
                let x_diff = x2 - x1;
                let y_diff = y2 - y1;

                if x_diff.norm() < 1e-6
                {
                    let y_sum = y1 + y2;
                    if y_sum.norm() < 1e-6 || y1.norm() < 1e-6
                    {
                        OctonionPoint::Infinity
                    }
                    else
                    {
                        // Point doubling: lambda = (3 * x1^2 + a) * (2 * y1)^-1
                        let three = OctonionSimd::ONE.scale(3.0);
                        let two = OctonionSimd::ONE.scale(2.0);
                        let num = (three * x1 * x1) + self.a;
                        let den = two * y1;
                        if den.norm_sqr() < 1e-12
                        {
                            OctonionPoint::Infinity
                        }
                        else
                        {
                            let lambda = num * den.inverse();
                            let x3 = (lambda * lambda) - x1 - x2;
                            let y3 = (lambda * (x1 - x3)) - y1;
                            OctonionPoint::Affine { x: x3, y: y3 }
                        }
                    }
                }
                else
                {
                    // Point addition: lambda = (y2 - y1) * (x2 - x1)^-1
                    if x_diff.norm_sqr() < 1e-12
                    {
                        OctonionPoint::Infinity
                    }
                    else
                    {
                        let lambda = y_diff * x_diff.inverse();
                        let x3 = (lambda * lambda) - x1 - x2;
                        let y3 = (lambda * (x1 - x3)) - y1;
                        OctonionPoint::Affine { x: x3, y: y3 }
                    }
                }
            },
        }
    }

    /// Exact double-and-add scalar multiplication using a stack-allocated/register-resident loop.
    /// Guarantees zero heap allocation.
    #[inline(always)]
    pub fn scalar_mul(&self, point: OctonionPoint, mut scalar: u32) -> OctonionPoint {
        let mut result = OctonionPoint::Infinity;
        let mut addend = point;

        while scalar > 0
        {
            if scalar & 1 == 1
            {
                result = self.add(result, addend);
            }
            addend = self.add(addend, addend);
            scalar >>= 1;
        }

        result
    }

    /// Batched, high-performance scalar multiplication on a pre-allocated buffer.
    /// Zero heap allocations during execution.
    #[inline(always)]
    pub fn scalar_mul_batch(
        &self,
        points: &[OctonionPoint],
        scalars: &[u32],
        out: &mut [OctonionPoint],
    ) {
        let len = points.len().min(scalars.len()).min(out.len());
        for i in 0..len
        {
            out[i] = self.scalar_mul(points[i], scalars[i]);
        }
    }
}

/// An affine point on a Sedenionic Elliptic Curve, or the Point at Infinity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SedenionPoint {
    Infinity,
    Affine { x: SedenionSimd, y: SedenionSimd },
}

/// A Sedenionic Elliptic Curve defined by Y^2 = X^3 + A*X + B.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SedenionCurve {
    pub a: SedenionSimd,
    pub b: SedenionSimd,
}

impl SedenionCurve {
    #[inline(always)]
    pub fn new(a: SedenionSimd, b: SedenionSimd) -> Self {
        Self { a, b }
    }

    /// Checks if a point lies on the curve.
    #[inline(always)]
    pub fn is_on_curve(&self, point: &SedenionPoint) -> bool {
        match point
        {
            SedenionPoint::Infinity => true,
            SedenionPoint::Affine { x, y } =>
            {
                let left = *y * *y;
                let right = (*x * *x * *x) + (self.a * *x) + self.b;
                // Epsilon check for float tolerance
                (left - right).norm() < 1e-4
            },
        }
    }

    /// Adds two points on the sedenionic curve.
    #[inline(always)]
    pub fn add(&self, p1: SedenionPoint, p2: SedenionPoint) -> SedenionPoint {
        match (p1, p2)
        {
            (SedenionPoint::Infinity, _) => p2,
            (_, SedenionPoint::Infinity) => p1,
            (SedenionPoint::Affine { x: x1, y: y1 }, SedenionPoint::Affine { x: x2, y: y2 }) =>
            {
                let x_diff = x2 - x1;
                let y_diff = y2 - y1;

                if x_diff.norm() < 1e-6
                {
                    let y_sum = y1 + y2;
                    if y_sum.norm() < 1e-6 || y1.norm() < 1e-6
                    {
                        SedenionPoint::Infinity
                    }
                    else
                    {
                        // Point doubling: lambda = (3 * x1^2 + a) * (2 * y1)^-1
                        let three = SedenionSimd::ONE.scale(3.0);
                        let two = SedenionSimd::ONE.scale(2.0);
                        let num = (three * x1 * x1) + self.a;
                        let den = two * y1;
                        if den.norm_sqr() < 1e-12
                        {
                            SedenionPoint::Infinity
                        }
                        else
                        {
                            let lambda = num * den.inverse();
                            let x3 = (lambda * lambda) - x1 - x2;
                            let y3 = (lambda * (x1 - x3)) - y1;
                            SedenionPoint::Affine { x: x3, y: y3 }
                        }
                    }
                }
                else
                {
                    // Point addition: lambda = (y2 - y1) * (x2 - x1)^-1
                    if x_diff.norm_sqr() < 1e-12
                    {
                        SedenionPoint::Infinity
                    }
                    else
                    {
                        let lambda = y_diff * x_diff.inverse();
                        let x3 = (lambda * lambda) - x1 - x2;
                        let y3 = (lambda * (x1 - x3)) - y1;
                        SedenionPoint::Affine { x: x3, y: y3 }
                    }
                }
            },
        }
    }

    /// Exact double-and-add scalar multiplication using stack-allocated/register-resident loop.
    /// Guarantees zero heap allocation.
    #[inline(always)]
    pub fn scalar_mul(&self, point: SedenionPoint, mut scalar: u32) -> SedenionPoint {
        let mut result = SedenionPoint::Infinity;
        let mut addend = point;

        while scalar > 0
        {
            if scalar & 1 == 1
            {
                result = self.add(result, addend);
            }
            addend = self.add(addend, addend);
            scalar >>= 1;
        }

        result
    }

    /// Batched, high-performance scalar multiplication on a pre-allocated buffer.
    /// Zero heap allocations during execution.
    #[inline(always)]
    pub fn scalar_mul_batch(
        &self,
        points: &[SedenionPoint],
        scalars: &[u32],
        out: &mut [SedenionPoint],
    ) {
        let len = points.len().min(scalars.len()).min(out.len());
        for i in 0..len
        {
            out[i] = self.scalar_mul(points[i], scalars[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_octonion_curve_operations() {
        let a = OctonionSimd::ONE;
        let b = OctonionSimd::ONE.scale(-1.0);
        let curve = OctonionCurve::new(a, b);

        let p1 = OctonionPoint::Infinity;
        let p2 = OctonionPoint::Affine {
            x: OctonionSimd::ONE,
            y: OctonionSimd::ONE,
        };

        assert!(curve.is_on_curve(&p2));
        assert_eq!(curve.add(p1, p2), p2);
        assert_eq!(curve.add(p2, p1), p2);

        let p3 = curve.add(p2, p2);
        assert!(curve.is_on_curve(&p3));

        let mul = curve.scalar_mul(p2, 5);
        assert!(curve.is_on_curve(&mul));

        let mut out = [OctonionPoint::Infinity; 2];
        curve.scalar_mul_batch(&[p2, p2], &[3, 4], &mut out);
        assert!(curve.is_on_curve(&out[0]));
        assert!(curve.is_on_curve(&out[1]));
    }

    #[test]
    fn test_sedenion_curve_operations() {
        let a = SedenionSimd::ONE;
        let b = SedenionSimd::ONE.scale(-1.0);
        let curve = SedenionCurve::new(a, b);

        let p1 = SedenionPoint::Infinity;
        let p2 = SedenionPoint::Affine {
            x: SedenionSimd::ONE,
            y: SedenionSimd::ONE,
        };

        assert!(curve.is_on_curve(&p2));
        assert_eq!(curve.add(p1, p2), p2);
        assert_eq!(curve.add(p2, p1), p2);

        let p3 = curve.add(p2, p2);
        assert!(curve.is_on_curve(&p3));

        let mul = curve.scalar_mul(p2, 3);
        assert!(curve.is_on_curve(&mul));
    }
}
