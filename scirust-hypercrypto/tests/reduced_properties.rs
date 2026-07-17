//! Reduced-model property tests (spec §16, §17) and determinism of the
//! machine-readable output. These validate the harness's core claims on
//! exhaustively-analysable variants.

use scirust_hypercrypto::algebra::word::{W2, W8};
use scirust_hypercrypto::analysis::battery::{
    Flags, Verdict, analyze_v01, decide, validate_controls,
};
use scirust_hypercrypto::analysis::degree::{feistel_branch_after, octfn_degree};
use scirust_hypercrypto::analysis::invariants::norm_functional_through;
use scirust_hypercrypto::analysis::report::sha256_hex;
use scirust_hypercrypto::fixtures::{Fixture, FixtureId};
use scirust_hypercrypto::permutation::Variant;
use scirust_hypercrypto::permutation::round::f_round;

#[test]
fn machine_output_is_deterministic() {
    // Two independent runs of the battery must produce byte-identical JSON and
    // hence identical fingerprints (no OS entropy, no wall-clock).
    let mut fa = Flags {
        roundtrip_ok: true,
        ..Default::default()
    };
    let mut fb = Flags {
        roundtrip_ok: true,
        ..Default::default()
    };
    let ja = analyze_v01::<W2>(0xABCD, 20_000, &mut fa).to_pretty();
    let jb = analyze_v01::<W2>(0xABCD, 20_000, &mut fb).to_pretty();
    assert_eq!(
        sha256_hex(&ja),
        sha256_hex(&jb),
        "battery output not deterministic"
    );
}

#[test]
fn nano2_verdict_is_continue() {
    let (a, b, _) = validate_controls::<W2>(1, 0);
    let mut flags = Flags {
        roundtrip_ok: true,
        control_a_broken: a,
        control_b_broken: b,
        ..Default::default()
    };
    let _ = analyze_v01::<W2>(1, 40_000, &mut flags);
    let (verdict, reasons) = decide(&flags);
    assert_eq!(
        verdict,
        Verdict::Continue,
        "unexpected verdict; reasons: {reasons:?}"
    );
}

#[test]
fn algebraic_degree_grows_with_rounds() {
    // exact ANF at NANO-2: branch degree must be non-decreasing and exceed 1.
    let fx = Fixture::new(FixtureId::PseudoRandom(0xD0E));
    let mut last = 0u32;
    for r in [1u32, 2, 3, 4]
    {
        let d = octfn_degree::<W2>(feistel_branch_after::<W2>(&fx, Variant::V01, r))
            .unwrap()
            .max_degree;
        assert!(d >= last, "degree decreased at r={r}");
        last = d;
    }
    assert!(last > 1, "multi-round degree should exceed 1 (nonlinear)");
}

#[test]
fn no_full_round_norm_invariant() {
    // Kill-criterion probe: N(x) must NOT determine N(F(x)) at MINI-8.
    let m = Fixture::new(FixtureId::PseudoRandom(3)).round_material::<W8>(0);
    let r = norm_functional_through::<W8>(move |x| f_round(x, &m), 5, 60_000);
    assert!(
        !r.norm_determines_output_norm,
        "a surviving norm invariant would be a KILL criterion"
    );
    assert!(
        r.witness.is_some(),
        "expected an explicit refutation witness"
    );
}

#[test]
fn control_degree_and_key_dependence() {
    // Control A (linear-only) = PERM_π(R ⊞ K0). At NANO-2 a 2-bit constant add
    // is GF(2)-affine, so Control A is degree 1 for EVERY key — this validates
    // that the degree tool identifies a linear construction.
    //
    // FINDING (documented, not a kill): at the tiny NANO-2 width the v0.1
    // round's GF(2) degree is KEY-DEPENDENT — some round keys make the two
    // ring-linear multiplies also GF(2)-linear, yielding a degree-1 single
    // round; other keys give degree 2. This is a small-width weak-key artifact:
    // it does not persist across the multi-round construction (degree grows
    // 2→3→6→8 over 4 rounds) and it is not affine for all keys.
    let mut any_nonlinear = false;
    for seed in [0xDE6u64, 1, 2, 3, 4, 5, 0x5C1_0001, 0xBEEF]
    {
        let m = Fixture::new(FixtureId::PseudoRandom(seed)).round_material::<W2>(0);
        let da = octfn_degree::<W2>(move |x| Variant::ControlA.round(x, &m))
            .unwrap()
            .max_degree;
        assert_eq!(
            da, 1,
            "Control A must be degree 1 for every key (seed=0x{seed:x})"
        );
        let d1 = octfn_degree::<W2>(move |x| f_round(x, &m))
            .unwrap()
            .max_degree;
        if d1 > 1
        {
            any_nonlinear = true;
        }
    }
    assert!(
        any_nonlinear,
        "at least one key must yield a nonlinear (degree>1) round"
    );
}

#[test]
fn weak_key_gf2_affine_class_is_documented() {
    // FINDING (Phase-1, documented, NOT a construction break): a class of weak
    // keys linearizes the round over GF(2). Concretely, high-bit-only multiplier
    // octonions make the multiply GF(2)-linear (128·x ≡ 128·(x mod 2) mod 256),
    // and zero multipliers make the round constant. Realistic pseudo-random,
    // odd-norm, even-norm, and incrementing keys keep the round NONLINEAR, so it
    // is not affine for all keys (the verdict gate requires all-key affinity).
    // This is a lead for Phase-2 weak-key / related-key analysis and for the
    // eventual key schedule (which must avoid such degenerate multipliers).
    use scirust_hypercrypto::algebra::word::W8;
    use scirust_hypercrypto::analysis::linearity::gf2_affine_test;

    let affine = |fid: FixtureId| {
        let m = Fixture::new(fid).round_material::<W8>(0);
        gf2_affine_test::<W8>(move |x| f_round(x, &m), 5, 60_000).holds
    };

    // weak keys => GF(2)-affine round
    assert!(
        affine(FixtureId::HighBit),
        "high-bit multipliers linearize the round"
    );
    assert!(
        affine(FixtureId::Zero),
        "zero multipliers give a constant/affine round"
    );

    // realistic keys => nonlinear round
    for fid in [
        FixtureId::PseudoRandom(0x5C1_0001),
        FixtureId::OddNorm,
        FixtureId::EvenNormZeroDiv,
        FixtureId::Incrementing,
    ]
    {
        assert!(
            !affine(fid),
            "general key {fid:?} must give a nonlinear round"
        );
    }
}
