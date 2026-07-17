//! Deliberately-weakened **control** variants (spec Phase-1 controls A–D).
//!
//! Controls exist so the analysis tooling can prove it detects known weakness.
//! A structural break in a control validates the harness; it does NOT kill v0.1.
//! If the tools fail to break Controls A or B, Phase 1 is INCONCLUSIVE because
//! the harness is untrustworthy.

use crate::algebra::word::Word;
use crate::algebra::{Oct, Quat};
use crate::fixtures::RoundMaterial;
use crate::permutation::round::f_round;

/// Octonion round-function variant selector.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Variant {
    /// Exact v0.1 `F-PROG` (spec §12.2).
    V01,
    /// Control A — linear-only (no MUL): `F_A(R) = PERM_π(R ⊞ K0)`.
    /// Ring-affine over `Z/2^k`; the additive-affinity detector must recover it.
    ControlA,
    /// Control B — ring-linear multiplication only: `F_B(R) = (K1 ⊗ R) ⊗ K2`.
    /// `Z/2^k`-linear; the ring-matrix detector must recover its `8×8` matrix.
    ControlB,
    /// Control C — one-multiply round: `F_C(R) = PERM_π(ROT_λ(K1 ⊗ R)) ⊕ RC`.
    ControlC,
}

impl Variant {
    /// Stable label for reports/CLI.
    pub fn label(self) -> &'static str {
        match self
        {
            Variant::V01 => "v0.1",
            Variant::ControlA => "control-A-linear-only",
            Variant::ControlB => "control-B-ring-linear",
            Variant::ControlC => "control-C-one-multiply",
        }
    }
    /// Parse from a CLI token.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str()
        {
            "v01" | "v0.1" | "v1" => Some(Variant::V01),
            "a" | "control-a" | "controla" => Some(Variant::ControlA),
            "b" | "control-b" | "controlb" => Some(Variant::ControlB),
            "c" | "control-c" | "controlc" => Some(Variant::ControlC),
            _ => None,
        }
    }
    /// Evaluate this variant's round function.
    pub fn round<W: Word>(self, r: Oct<W>, m: &RoundMaterial<W>) -> Oct<W> {
        match self
        {
            Variant::V01 => f_round(r, m),
            Variant::ControlA =>
            {
                // linear-only: add key, permute slots. No MUL, no ROT, no XOR.
                r.add(m.k0).perm_pi()
            },
            Variant::ControlB =>
            {
                // ring-linear: (K1 ⊗ R) ⊗ K2, nothing else. No offset.
                m.k1.mul(r).mul(m.k2)
            },
            Variant::ControlC =>
            {
                // one MUL + public layers, no K0 add, no second MUL.
                let p = m.k1.mul(r);
                p.rot_lambda().perm_pi().xor(m.rc)
            },
        }
    }
}

/// Per-lane rotation amounts for the quaternion control shell (first 4 of `λ`).
pub const QUAT_LAMBDA: [u32; 4] = [7, 19, 31, 47];
/// Coefficient-slot permutation for the quaternion control (a 4-derangement).
pub const QUAT_PI: [usize; 4] = [1, 3, 0, 2];

/// Control D — associative quaternion round with a comparable shell
/// (add, two muls, 4-lane rotation, 4-slot permutation, XOR). Used ONLY to
/// compare structure against the octonion; differences do not prove security.
pub fn f_round_quat<W: Word>(
    r: Quat<W>,
    k0: Quat<W>,
    k1: Quat<W>,
    k2: Quat<W>,
    rc: Quat<W>,
) -> Quat<W> {
    let a = r.add(k0);
    let p = k1.mul(a);
    let q = p.mul(k2);
    // 4-lane rotation
    let mut u = q;
    for j in 0..4
    {
        u.c[j] = u.c[j].rotl(QUAT_LAMBDA[j]);
    }
    // 4-slot permutation
    let w = Quat {
        c: std::array::from_fn(|i| u.c[QUAT_PI[i]]),
    };
    w.xor(rc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::word::W8;
    use crate::fixtures::{Fixture, FixtureId};

    #[test]
    fn control_b_is_pure_multiplication() {
        let m = Fixture::new(FixtureId::OddNorm).round_material::<W8>(0);
        let x = Oct::<W8>::from_u64s([3, 1, 4, 1, 5, 9, 2, 6]);
        assert_eq!(Variant::ControlB.round(x, &m), m.k1.mul(x).mul(m.k2));
    }

    #[test]
    fn control_a_has_no_multiplication_effect() {
        // Control A output depends only on add+perm; scaling K1/K2 must not matter.
        let f = Fixture::new(FixtureId::PseudoRandom(5));
        let mut m = f.round_material::<W8>(0);
        let x = Oct::<W8>::from_u64s([1, 2, 3, 4, 5, 6, 7, 8]);
        let before = Variant::ControlA.round(x, &m);
        m.k1 = Oct::zero();
        m.k2 = Oct::zero();
        assert_eq!(Variant::ControlA.round(x, &m), before);
    }
}
