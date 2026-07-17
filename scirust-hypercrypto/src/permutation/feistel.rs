//! Balanced two-branch Feistel shell (spec §12.3, §13.2), parameterized by
//! round count, variant (v0.1 or a control), and whether whitening is applied.
//!
//! Invertibility is structural (the XOR combiner is an involution); the round
//! function is never inverted (spec §13.1). Forward/inverse round-trip is an
//! implementation-correctness check, not a security property.

use crate::algebra::Oct;
use crate::algebra::word::Word;
use crate::fixtures::Fixture;
use crate::permutation::controls::Variant;

/// Two-branch Feistel state `(L, R)`, each a full octonion (spec §12.3).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct State<W: Word> {
    /// Left branch.
    pub l: Oct<W>,
    /// Right branch.
    pub r: Oct<W>,
}

impl<W: Word> State<W> {
    /// Construct a state.
    pub fn new(l: Oct<W>, r: Oct<W>) -> Self {
        State { l, r }
    }
}

/// Forward permutation `P_K` (spec §12.3). `whiten` toggles the Even–Mansour
/// input/output whitening; the exact v0.1 path uses `whiten = true`.
pub fn forward<W: Word>(
    mut s: State<W>,
    fixture: &Fixture,
    variant: Variant,
    rounds: u32,
    whiten: bool,
) -> State<W> {
    if whiten
    {
        let w = fixture.whitening::<W>();
        s.l = s.l.xor(w.in_l);
        s.r = s.r.xor(w.in_r);
    }
    for r in 0..rounds
    {
        let m = fixture.round_material::<W>(r);
        let fout = variant.round(s.r, &m);
        s = State {
            l: s.r,
            r: s.l.xor(fout),
        };
    }
    if whiten
    {
        let w = fixture.whitening::<W>();
        s.l = s.l.xor(w.out_l);
        s.r = s.r.xor(w.out_r);
    }
    s
}

/// Inverse permutation `P_K^{-1}` (spec §13.2). Reuses the round function; the
/// only inverted operation is the XOR combiner.
pub fn inverse<W: Word>(
    mut s: State<W>,
    fixture: &Fixture,
    variant: Variant,
    rounds: u32,
    whiten: bool,
) -> State<W> {
    if whiten
    {
        let w = fixture.whitening::<W>();
        s.l = s.l.xor(w.out_l);
        s.r = s.r.xor(w.out_r);
    }
    for r in (0..rounds).rev()
    {
        let m = fixture.round_material::<W>(r);
        // current (L,R) = ρ_r(Lprev, Rprev): Lprev = R ⊕ F_r(L), Rprev = L
        let lprev = s.r.xor(variant.round(s.l, &m));
        let rprev = s.l;
        s = State { l: lprev, r: rprev };
    }
    if whiten
    {
        let w = fixture.whitening::<W>();
        s.l = s.l.xor(w.in_l);
        s.r = s.r.xor(w.in_r);
    }
    s
}

/// Forward evaluation recording the state after each round (research tracing).
pub fn forward_traced<W: Word>(
    mut s: State<W>,
    fixture: &Fixture,
    variant: Variant,
    rounds: u32,
    whiten: bool,
) -> (State<W>, Vec<State<W>>) {
    let mut trace = Vec::with_capacity(rounds as usize + 1);
    if whiten
    {
        let w = fixture.whitening::<W>();
        s.l = s.l.xor(w.in_l);
        s.r = s.r.xor(w.in_r);
    }
    trace.push(s);
    for r in 0..rounds
    {
        let m = fixture.round_material::<W>(r);
        let fout = variant.round(s.r, &m);
        s = State {
            l: s.r,
            r: s.l.xor(fout),
        };
        trace.push(s);
    }
    if whiten
    {
        let w = fixture.whitening::<W>();
        s.l = s.l.xor(w.out_l);
        s.r = s.r.xor(w.out_r);
    }
    (s, trace)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::word::{W2, W4, W8};
    use crate::fixtures::{Fixture, FixtureId};

    fn roundtrip_at<W: Word>(variant: Variant, whiten: bool) {
        let f = Fixture::new(FixtureId::PseudoRandom(0xabcdef));
        let mut m = crate::fixtures::SplitMix64::new(1);
        for _ in 0..64
        {
            let l = Oct::<W>::from_u64s(std::array::from_fn(|_| m.next_u64()));
            let r = Oct::<W>::from_u64s(std::array::from_fn(|_| m.next_u64()));
            let s = State::new(l, r);
            let enc = forward(s, &f, variant, 6, whiten);
            let dec = inverse(enc, &f, variant, 6, whiten);
            assert_eq!(dec, s, "roundtrip failed variant={:?}", variant);
        }
    }

    #[test]
    fn roundtrip_all_widths_and_variants() {
        for v in [
            Variant::V01,
            Variant::ControlA,
            Variant::ControlB,
            Variant::ControlC,
        ]
        {
            for whiten in [false, true]
            {
                roundtrip_at::<W2>(v, whiten);
                roundtrip_at::<W4>(v, whiten);
                roundtrip_at::<W8>(v, whiten);
            }
        }
    }
}
