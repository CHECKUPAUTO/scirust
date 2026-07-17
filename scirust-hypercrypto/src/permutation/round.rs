//! The exact v0.1 round function `F-PROG` (spec §12.2).
//!
//! ```text
//! a = R ⊞ K0 ; p = K1 ⊗ a ; q = p ⊗ K2 ; u = ROT_λ(q) ; w = PERM_π(u) ; F = w ⊕ RC
//! ```
//!
//! The bracketing `(K1 ⊗ a) ⊗ K2` and the K1-left / K2-right ordering are fixed
//! and semantically significant (non-associative, non-commutative).

use crate::algebra::Oct;
use crate::algebra::word::Word;
use crate::fixtures::RoundMaterial;

/// Intermediate outputs of one round, for per-layer probing (spec §Experiment 2/5).
#[derive(Copy, Clone, Debug)]
pub struct RoundLayers<W: Word> {
    /// `a = R ⊞ K0`.
    pub a: Oct<W>,
    /// `p = K1 ⊗ a`.
    pub p: Oct<W>,
    /// `q = (K1 ⊗ a) ⊗ K2`.
    pub q: Oct<W>,
    /// `u = ROT_λ(q)`.
    pub u: Oct<W>,
    /// `w = PERM_π(u)`.
    pub w: Oct<W>,
    /// `F = w ⊕ RC`.
    pub f: Oct<W>,
}

/// Exact v0.1 round function `F_r(R)` (spec §12.2).
pub fn f_round<W: Word>(r: Oct<W>, m: &RoundMaterial<W>) -> Oct<W> {
    let a = r.add(m.k0);
    let p = m.k1.mul(a);
    let q = p.mul(m.k2);
    let u = q.rot_lambda();
    let w = u.perm_pi();
    w.xor(m.rc)
}

/// Traced variant returning every intermediate layer.
pub fn f_round_traced<W: Word>(r: Oct<W>, m: &RoundMaterial<W>) -> RoundLayers<W> {
    let a = r.add(m.k0);
    let p = m.k1.mul(a);
    let q = p.mul(m.k2);
    let u = q.rot_lambda();
    let w = u.perm_pi();
    let f = w.xor(m.rc);
    RoundLayers { a, p, q, u, w, f }
}

/// The pre-rotation map `G(x) = (K1 ⊗ (x ⊞ K0)) ⊗ K2` (spec §Experiment 2).
/// Isolated because it is exactly ring-affine over `Z/2^k`.
pub fn g_pre_rotation<W: Word>(x: Oct<W>, m: &RoundMaterial<W>) -> Oct<W> {
    let a = x.add(m.k0);
    let p = m.k1.mul(a);
    p.mul(m.k2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::word::W8;
    use crate::fixtures::{Fixture, FixtureId};

    #[test]
    fn traced_matches_direct() {
        let m = Fixture::new(FixtureId::PseudoRandom(1)).round_material::<W8>(0);
        let x = Oct::<W8>::from_u64s([1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(f_round(x, &m), f_round_traced(x, &m).f);
    }

    #[test]
    fn g_is_the_pre_rotation_stage() {
        let m = Fixture::new(FixtureId::PseudoRandom(9)).round_material::<W8>(2);
        let x = Oct::<W8>::from_u64s([9, 8, 7, 6, 5, 4, 3, 2]);
        assert_eq!(g_pre_rotation(x, &m), f_round_traced(x, &m).q);
    }
}
