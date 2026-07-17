//! Known-answer tests for the exact octonion algebra (spec §8.3–§8.4).
//!
//! Every value here is transcribed from, or derivable from, the authoritative
//! worked examples in the merged specification.

use scirust_hypercrypto::algebra::Oct;
use scirust_hypercrypto::algebra::word::{W8, W64, Word};

const MAX: u64 = u64::MAX; // -1 mod 2^64 = ê_k coefficient

fn o64(c: [u64; 8]) -> Oct<W64> {
    Oct::from_u64s(c)
}

// ---- The five specification multiplication KATs (spec §8.4) -----------------

#[test]
fn kat1_non_commutativity_e1_e2() {
    let e1 = Oct::<W64>::e(1);
    let e2 = Oct::<W64>::e(2);
    assert_eq!(e1.mul(e2).to_u64s(), [0, 0, 0, 0, 1, 0, 0, 0]); // e1⊗e2 = e4
    assert_eq!(e2.mul(e1).to_u64s(), [0, 0, 0, 0, MAX, 0, 0, 0]); // e2⊗e1 = -e4
}

#[test]
fn kat2_associative_fano_line_124() {
    let (e1, e2, e4) = (Oct::<W64>::e(1), Oct::<W64>::e(2), Oct::<W64>::e(4));
    let left = e1.mul(e2).mul(e4); // (e1⊗e2)⊗e4
    let right = e1.mul(e2.mul(e4)); // e1⊗(e2⊗e4)
    assert_eq!(left.to_u64s(), [MAX, 0, 0, 0, 0, 0, 0, 0]); // -e0
    assert_eq!(right.to_u64s(), [MAX, 0, 0, 0, 0, 0, 0, 0]); // -e0
    assert_eq!(left, right, "association holds on the Fano line {{1,2,4}}");
}

#[test]
fn kat3_non_associativity_123() {
    let (e1, e2, e3) = (Oct::<W64>::e(1), Oct::<W64>::e(2), Oct::<W64>::e(3));
    let left = e1.mul(e2).mul(e3); // (e1⊗e2)⊗e3 = -e6
    let right = e1.mul(e2.mul(e3)); // e1⊗(e2⊗e3) = e6
    assert_eq!(left.to_u64s(), [0, 0, 0, 0, 0, 0, MAX, 0]);
    assert_eq!(right.to_u64s(), [0, 0, 0, 0, 0, 0, 1, 0]);
    assert_ne!(left, right, "non-associative off the Fano lines");
    // associator = left - right = -2 e6
    assert_eq!(left.sub(right).to_u64s(), [0, 0, 0, 0, 0, 0, MAX - 1, 0]);
}

#[test]
fn kat4_general_product() {
    // x = e1 + e2 ; y = e2 + e3 ; x⊗y = -e0 + e4 + e5 + e7
    let x = o64([0, 1, 1, 0, 0, 0, 0, 0]);
    let y = o64([0, 0, 1, 1, 0, 0, 0, 0]);
    assert_eq!(x.mul(y).to_u64s(), [MAX, 0, 0, 0, 1, 1, 0, 1]);
}

#[test]
fn kat5_reverse_product() {
    // y⊗x = -e0 - e4 - e5 - e7
    let x = o64([0, 1, 1, 0, 0, 0, 0, 0]);
    let y = o64([0, 0, 1, 1, 0, 0, 0, 0]);
    assert_eq!(y.mul(x).to_u64s(), [MAX, 0, 0, 0, MAX, MAX, 0, MAX]);
}

// ---- Structural properties (spec §8.3) --------------------------------------

#[test]
fn all_64_basis_products_match_two_oracles() {
    for i in 0..8
    {
        for j in 0..8
        {
            let a = Oct::<W64>::e(i);
            let b = Oct::<W64>::e(j);
            // authoritative oracle equals the independent triple oracle
            assert_eq!(a.mul(b), a.mul_via_triples(b), "e{i}*e{j}");
        }
    }
}

#[test]
fn imaginary_units_antisymmetric() {
    for i in 1..8
    {
        for j in 1..8
        {
            if i == j
            {
                continue;
            }
            let a = Oct::<W64>::e(i).mul(Oct::<W64>::e(j));
            let b = Oct::<W64>::e(j).mul(Oct::<W64>::e(i));
            assert_eq!(a, b.neg(), "e{i}·e{j} = -(e{j}·e{i})");
        }
    }
}

#[test]
fn imaginary_units_square_to_minus_one() {
    for i in 1..8
    {
        assert_eq!(
            Oct::<W64>::e(i).mul(Oct::<W64>::e(i)).to_u64s(),
            [MAX, 0, 0, 0, 0, 0, 0, 0]
        );
    }
}

#[test]
fn identity_and_conjugation() {
    let mut rng = 0x1234_5678u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    for _ in 0..500
    {
        let x = Oct::<W64>::from_u64s(std::array::from_fn(|_| next()));
        assert_eq!(x.mul(Oct::one()), x);
        assert_eq!(Oct::one().mul(x), x);
        // conj(conj(x)) == x ; x + conj(x) == 2*x0
        assert_eq!(x.conj().conj(), x);
        let trace = x.add(x.conj());
        assert_eq!(trace.c[0].to_u64(), x.c[0].to_u64().wrapping_mul(2));
        for i in 1..8
        {
            assert_eq!(trace.c[i].to_u64(), 0);
        }
        // norm: x ⊗ conj(x) == N(x) * e0
        let nx = x.norm().to_u64();
        let prod = x.mul(x.conj());
        assert_eq!(prod.c[0].to_u64(), nx);
        for i in 1..8
        {
            assert_eq!(prod.c[i].to_u64(), 0, "x⊗x̄ must be a scalar");
        }
    }
}

#[test]
fn octonions_are_alternative() {
    // left/right alternative laws: (x⊗x)⊗y = x⊗(x⊗y) and (y⊗x)⊗x = y⊗(x⊗x).
    let mut rng = 0x9e37_79b9u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    for _ in 0..1000
    {
        let x = Oct::<W64>::from_u64s(std::array::from_fn(|_| next()));
        let y = Oct::<W64>::from_u64s(std::array::from_fn(|_| next()));
        assert_eq!(x.mul(x).mul(y), x.mul(x.mul(y)), "left alternative");
        assert_eq!(y.mul(x).mul(x), y.mul(x.mul(x)), "right alternative");
        // flexibility: (x⊗y)⊗x == x⊗(y⊗x)
        assert_eq!(x.mul(y).mul(x), x.mul(y.mul(x)), "flexible");
    }
}

#[test]
fn explicit_non_associativity_witness() {
    // A concrete non-Fano-line triple witnesses non-associativity (spec §8.4).
    let (a, b, c) = (Oct::<W8>::e(1), Oct::<W8>::e(2), Oct::<W8>::e(3));
    assert_ne!(a.mul(b).mul(c), a.mul(b.mul(c)));
}

#[test]
fn norm_multiplicative_degen_identity() {
    // N(x⊗y) == N(x)·N(y) for random octonions (Degen eight-square identity).
    let mut rng = 0xdead_beefu64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    for _ in 0..2000
    {
        let x = Oct::<W64>::from_u64s(std::array::from_fn(|_| next()));
        let y = Oct::<W64>::from_u64s(std::array::from_fn(|_| next()));
        assert_eq!(x.mul(y).norm(), x.norm().wmul(y.norm()));
    }
}
