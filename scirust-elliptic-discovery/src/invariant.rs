//! Exact curve invariants used by catalog rules and candidate evaluation.

use crate::{Fp, ToyCurve};

/// Exact short-Weierstrass invariants represented in the curve's prime field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CurveInvariants {
    discriminant: Fp,
    j_invariant: Fp,
}

impl CurveInvariants {
    /// Computes the discriminant and j-invariant without floating point.
    pub fn compute(curve: ToyCurve) -> Self {
        let prime = curve.prime();
        let a = Fp::new(prime, curve.a());
        let b = Fp::new(prime, curve.b());
        let a_cubed = a.pow(3);
        let b_squared = b.pow(2);
        let four_a_cubed = a_cubed
            .checked_mul(Fp::new(prime, 4))
            .expect("values use the same prime");
        let core = four_a_cubed
            .checked_add(
                b_squared
                    .checked_mul(Fp::new(prime, 27))
                    .expect("values use the same prime"),
            )
            .expect("values use the same prime");
        let discriminant = core
            .checked_mul(Fp::new(prime, 16).neg())
            .expect("values use the same prime");
        let numerator = four_a_cubed
            .checked_mul(Fp::new(prime, 1728))
            .expect("values use the same prime");
        let j_invariant = numerator
            .checked_div(core)
            .expect("nonsingular curve has nonzero discriminant core");
        Self {
            discriminant,
            j_invariant,
        }
    }

    /// Canonical discriminant residue.
    pub const fn discriminant(self) -> u64 {
        self.discriminant.value()
    }

    /// Canonical j-invariant residue.
    pub const fn j_invariant(self) -> u64 {
        self.j_invariant.value()
    }

    /// Whether this curve belongs to the exceptional j=0 family.
    pub const fn is_j_zero(self) -> bool {
        self.j_invariant.is_zero()
    }

    /// Whether this curve belongs to the exceptional j=1728 family.
    pub fn is_j_1728(self) -> bool {
        self.j_invariant == Fp::new(self.j_invariant.prime(), 1728)
    }
}

/// Returns nontrivial cube roots of unity in canonical residue order.
pub fn nontrivial_cube_roots(curve: ToyCurve) -> Vec<u64> {
    let prime = curve.prime();
    (2..prime.value())
        .filter(|&value| Fp::new(prime, value).pow(3).value() == 1)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToyPrime;

    #[test]
    fn exceptional_families_have_expected_j_invariant() {
        let prime = ToyPrime::new(13).expect("13 is prime");
        let j_zero = ToyCurve::new(prime, 0, 2).expect("nonsingular");
        let j_1728 = ToyCurve::new(prime, 2, 0).expect("nonsingular");
        assert!(CurveInvariants::compute(j_zero).is_j_zero());
        assert!(CurveInvariants::compute(j_1728).is_j_1728());
    }

    #[test]
    fn cube_roots_are_exact_and_nontrivial() {
        let curve = ToyCurve::new(ToyPrime::new(13).expect("prime"), 0, 2)
            .expect("nonsingular");
        assert_eq!(nontrivial_cube_roots(curve), vec![3, 9]);
    }
}
