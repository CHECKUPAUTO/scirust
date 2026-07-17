//! Experiment 4 — algebraic degree via exact ANF (spec §Experiment 4).
//!
//! For a Boolean vector function on `n ≤ 18` input bits we build each output
//! bit's truth table, apply the Möbius transform to obtain the Algebraic Normal
//! Form, and read off the maximum monomial degree. This is an **exact** ANF
//! computation, never a sampled estimate; functions with `n > 18` input bits
//! are reported as out of exact range (no `2^64` truth table, per the spec).

use crate::algebra::Oct;
use crate::algebra::word::Word;

// The exact ANF / Möbius machinery is now the shared, general-purpose utility in
// `scirust_modalg::boolean`; only the octonion-specific wrappers live here.
pub use scirust_modalg::boolean::{DegreeResult, bitfn_degree};

fn oct_to_bits<W: Word>(o: Oct<W>) -> u64 {
    let bits = W::BITS;
    let mut acc = 0u64;
    for i in 0..8
    {
        acc |= o.c[i].to_u64() << (i as u32 * bits);
    }
    acc
}

fn bits_to_oct<W: Word>(v: u64) -> Oct<W> {
    let bits = W::BITS;
    let mask = if bits >= 64
    {
        u64::MAX
    }
    else
    {
        (1u64 << bits) - 1
    };
    let mut c = [W::ZERO; 8];
    for (i, slot) in c.iter_mut().enumerate()
    {
        *slot = W::from_u64((v >> (i as u32 * bits)) & mask);
    }
    Oct::from_coeffs(c)
}

/// Exact ANF degree of a map `f: Oct<W> → Oct<W>`, treating the octonion as a
/// bit-vector of width `8k`. Returns `None` when `8k` exceeds the exact-ANF
/// range of [`scirust_modalg::boolean::bitfn_degree`].
pub fn octfn_degree<W: Word>(f: impl Fn(Oct<W>) -> Oct<W>) -> Option<DegreeResult> {
    let n = 8 * W::BITS;
    bitfn_degree(move |x| oct_to_bits::<W>(f(bits_to_oct::<W>(x))), n, n)
}

/// Build a closure computing "the R branch after `rounds` Feistel rounds, with
/// the L branch initialised to zero", as a function of the initial R branch.
/// This gives a single-octonion (`8k`-bit) input suitable for exact ANF, and
/// lets us measure degree growth by round (spec §Experiment 4).
pub fn feistel_branch_after<W: Word>(
    fixture: &crate::fixtures::Fixture,
    variant: crate::permutation::Variant,
    rounds: u32,
) -> impl Fn(Oct<W>) -> Oct<W> + '_ {
    move |r0: Oct<W>| {
        let mut s = crate::permutation::State::new(Oct::<W>::zero(), r0);
        for r in 0..rounds
        {
            let m = fixture.round_material::<W>(r);
            let fout = variant.round(s.r, &m);
            s = crate::permutation::State::new(s.r, s.l.xor(fout));
        }
        s.r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::OctLayers;
    use crate::algebra::word::W2;
    use crate::fixtures::{Fixture, FixtureId};
    use crate::permutation::Variant;
    use crate::permutation::round::f_round;

    #[test]
    fn linear_map_has_degree_one() {
        // Control B is ring-linear; each output bit... note: ring-linear is NOT
        // GF(2)-linear (carries in the wrapping multiply), so degree may exceed 1.
        // Use a genuinely GF(2)-linear map (bit permutation) to pin degree 1.
        let f = |x: Oct<W2>| x.perm_pi();
        let d = octfn_degree::<W2>(f).unwrap();
        assert_eq!(
            d.max_degree, 1,
            "a bit permutation is GF(2)-affine, degree 1"
        );
    }

    #[test]
    fn full_round_has_degree_above_one() {
        let m = Fixture::new(FixtureId::PseudoRandom(7)).round_material::<W2>(0);
        let f = |x: Oct<W2>| f_round(x, &m);
        let d = octfn_degree::<W2>(f).unwrap();
        assert!(d.max_degree > 1, "F should be nonlinear over GF(2)");
    }

    #[test]
    fn degree_grows_with_rounds() {
        let fx = Fixture::new(FixtureId::PseudoRandom(11));
        let d1 = octfn_degree::<W2>(feistel_branch_after::<W2>(&fx, Variant::V01, 1))
            .unwrap()
            .max_degree;
        let d3 = octfn_degree::<W2>(feistel_branch_after::<W2>(&fx, Variant::V01, 3))
            .unwrap()
            .max_degree;
        assert!(d3 >= d1, "degree should not decrease with more rounds");
    }
}
