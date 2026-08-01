//! Canonical, exact point enumeration.

use std::collections::BTreeMap;

use crate::curve::{ToyCurve, ToyPoint};
use crate::field::Fp;

impl ToyCurve {
    /// Enumerates every point over the curve's prime field exactly.
    ///
    /// The group identity appears first. Affine points then follow ascending
    /// coordinate order. The implementation constructs a square-root table
    /// first, using only exact modular multiplication and ordered containers.
    pub fn enumerate_points(self) -> Vec<ToyPoint> {
        let prime = self.prime();
        let modulus = prime.value();
        let mut roots: BTreeMap<u64, Vec<u64>> = BTreeMap::new();

        for y in 0..modulus {
            let y = Fp::new(prime, y);
            roots.entry(y.square().value()).or_default().push(y.value());
        }

        let mut points = vec![self.identity()];
        let a = Fp::new(prime, self.a());
        let b = Fp::new(prime, self.b());

        for x in 0..modulus {
            let x = Fp::new(prime, x);
            let right = x
                .square()
                .mul_same(x)
                .add_same(a.mul_same(x))
                .add_same(b);
            if let Some(ys) = roots.get(&right.value()) {
                for &y in ys {
                    points.push(self.affine_unchecked(x, Fp::new(prime, y)));
                }
            }
        }

        points
    }
}
