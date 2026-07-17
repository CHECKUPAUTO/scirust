//! Forward/inverse round-trip tests (spec §13). These establish IMPLEMENTATION
//! correctness (the permutation is a bijection and the inverse is exact); they
//! are NOT a cryptographic security property.

use scirust_hypercrypto::algebra::Oct;
use scirust_hypercrypto::algebra::word::{W2, W4, W8, W16, W64, Word};
use scirust_hypercrypto::fixtures::{Fixture, FixtureId, SplitMix64};
use scirust_hypercrypto::permutation::{State, Variant, feistel};

fn roundtrip_sampled<W: Word>(variant: Variant, rounds: u32, whiten: bool, n: usize) {
    for fid in [
        FixtureId::Zero,
        FixtureId::Incrementing,
        FixtureId::HighBit,
        FixtureId::OddNorm,
        FixtureId::EvenNormZeroDiv,
        FixtureId::PseudoRandom(0xC0FFEE),
    ]
    {
        let f = Fixture::new(fid);
        let mut rng = SplitMix64::new(0x51ED ^ fid.label().len() as u64);
        for _ in 0..n
        {
            let l = Oct::<W>::from_u64s(std::array::from_fn(|_| rng.next_u64()));
            let r = Oct::<W>::from_u64s(std::array::from_fn(|_| rng.next_u64()));
            let s = State::new(l, r);
            let enc = feistel::forward(s, &f, variant, rounds, whiten);
            let dec = feistel::inverse(enc, &f, variant, rounds, whiten);
            assert_eq!(
                dec, s,
                "roundtrip failed: {:?} {fid:?} whiten={whiten}",
                variant
            );
        }
    }
}

#[test]
fn roundtrip_all_widths_v01() {
    roundtrip_sampled::<W2>(Variant::V01, 24, true, 200);
    roundtrip_sampled::<W4>(Variant::V01, 24, true, 200);
    roundtrip_sampled::<W8>(Variant::V01, 24, true, 200);
    roundtrip_sampled::<W16>(Variant::V01, 24, true, 200);
    roundtrip_sampled::<W64>(Variant::V01, 24, true, 200);
}

#[test]
fn roundtrip_all_variants() {
    for v in [
        Variant::V01,
        Variant::ControlA,
        Variant::ControlB,
        Variant::ControlC,
    ]
    {
        for whiten in [false, true]
        {
            roundtrip_sampled::<W8>(v, 12, whiten, 100);
        }
    }
}

#[test]
fn exhaustive_small_projection_roundtrip_w2() {
    // Exhaustive over a small NANO-2 projection: fix L=0, sweep the 16-bit R
    // branch through 4 rounds; every state must round-trip. (2^16 states.)
    let f = Fixture::new(FixtureId::PseudoRandom(7));
    for code in 0u32..65536
    {
        let r = Oct::<W2>::from_u64s(std::array::from_fn(|i| ((code >> (2 * i)) & 3) as u64));
        let s = State::new(Oct::<W2>::zero(), r);
        let enc = feistel::forward(s, &f, Variant::V01, 4, true);
        let dec = feistel::inverse(enc, &f, Variant::V01, 4, true);
        assert_eq!(dec, s, "roundtrip failed at code {code}");
    }
}

#[test]
fn double_inverse_is_identity() {
    // P^{-1}(P(x)) == x and P(P^{-1}(x)) == x
    let f = Fixture::new(FixtureId::OddNorm);
    let mut rng = SplitMix64::new(99);
    for _ in 0..500
    {
        let l = Oct::<W64>::from_u64s(std::array::from_fn(|_| rng.next_u64()));
        let r = Oct::<W64>::from_u64s(std::array::from_fn(|_| rng.next_u64()));
        let s = State::new(l, r);
        let fwd = feistel::forward(s, &f, Variant::V01, 24, true);
        assert_eq!(feistel::inverse(fwd, &f, Variant::V01, 24, true), s);
        let inv = feistel::inverse(s, &f, Variant::V01, 24, true);
        assert_eq!(feistel::forward(inv, &f, Variant::V01, 24, true), s);
    }
}
