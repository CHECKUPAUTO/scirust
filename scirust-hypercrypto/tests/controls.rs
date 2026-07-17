//! Control-detection tests (spec Phase-1 controls). The harness MUST break the
//! deliberately-weakened Controls A and B; if it cannot, Phase 1 is
//! inconclusive because the tooling is untrustworthy. Also includes negative
//! sanity checks (the full round must NOT be recovered as affine).

use scirust_hypercrypto::algebra::Oct;
use scirust_hypercrypto::algebra::word::{W2, W8};
use scirust_hypercrypto::analysis::linearity::{gf2_affine_test, ring_affine_recover};
use scirust_hypercrypto::fixtures::{Fixture, FixtureId};
use scirust_hypercrypto::permutation::Variant;
use scirust_hypercrypto::permutation::round::f_round;

#[test]
fn control_a_recovered_as_ring_affine() {
    let m = Fixture::new(FixtureId::PseudoRandom(0xA)).round_material::<W2>(0);
    let (res, _a, _b) = ring_affine_recover::<W2>(move |x| Variant::ControlA.round(x, &m), 1, 0);
    assert!(
        res.holds,
        "Control A (linear-only) must be recovered as ring-affine"
    );
}

#[test]
fn control_b_recovered_as_ring_linear() {
    let m = Fixture::new(FixtureId::OddNorm).round_material::<W2>(0);
    let (res, mat, off) = ring_affine_recover::<W2>(move |x| Variant::ControlB.round(x, &m), 1, 0);
    assert!(res.holds, "Control B (ring-linear) must be recovered");
    assert_eq!(off, Oct::<W2>::zero(), "pure linear => zero offset");
    // odd-norm K1,K2 => the recovered ring matrix is invertible
    assert!(
        mat.is_unit(),
        "odd-norm multipliers => invertible ring matrix"
    );
}

#[test]
fn full_round_is_not_recovered_as_affine() {
    // Negative sanity: the real round function must NOT be captured by either
    // simple model (else it would be a structural break).
    let m = Fixture::new(FixtureId::PseudoRandom(5)).round_material::<W2>(0);
    let (ring, _, _) = ring_affine_recover::<W2>(move |x| f_round(x, &m), 1, 0);
    assert!(!ring.holds, "full round must not be ring-affine");
    let gf2 = gf2_affine_test::<W2>(move |x| f_round(x, &m), 1, 60_000);
    assert!(!gf2.holds, "full round must not be GF(2)-affine");
}

#[test]
fn control_detection_at_mini8() {
    // Controls must also be detected at a larger sampled width.
    let ma = Fixture::new(FixtureId::PseudoRandom(0xA)).round_material::<W8>(0);
    let (a, _, _) = ring_affine_recover::<W8>(move |x| Variant::ControlA.round(x, &ma), 3, 50_000);
    assert!(a.holds, "Control A must be recovered at MINI-8");
    let mb = Fixture::new(FixtureId::OddNorm).round_material::<W8>(0);
    let (b, _, off) =
        ring_affine_recover::<W8>(move |x| Variant::ControlB.round(x, &mb), 3, 50_000);
    assert!(
        b.holds && off == Oct::<W8>::zero(),
        "Control B must be recovered at MINI-8"
    );
}
