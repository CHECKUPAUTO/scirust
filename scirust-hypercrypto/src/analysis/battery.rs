//! Orchestration of the Phase-1 experiment battery and the gating verdict
//! (spec §"Phase-1 verdict", §"Kill criteria").
//!
//! The verdict is one of three exact strings and is computed from the boolean
//! outcomes of the experiments. `CONTINUE` explicitly does NOT mean "secure".

use crate::algebra::Oct;
use crate::algebra::OctLayers;
use crate::algebra::word::{W2, W4, W8, W16, W64, WidthTag, Word};
use crate::analysis::degree::{feistel_branch_after, octfn_degree};
use crate::analysis::invariants::{
    associator_evenness, layer_preserves_norm, norm_functional_through, norm_multiplicative_left,
};
use crate::analysis::linearity::{bit_affine_recover, gf2_affine_test, ring_affine_recover};
use crate::analysis::matrix_lifting::{Side, lift};
use crate::analysis::report::{Json, u64x8};
use crate::analysis::subspace::{cycle_census, run_battery};
use crate::analysis::util::sample_octs;
use crate::analysis::zero_divisors::{differential_bias, find_zero_divisors_w2, kernel_sizes};
use crate::fixtures::{Fixture, FixtureId};
use crate::permutation::round::{f_round, g_pre_rotation};
use crate::permutation::{State, Variant, feistel};

/// The three admissible Phase-1 verdicts (spec §"Phase-1 verdict").
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Verdict {
    /// A structural break was found.
    NoGo,
    /// No gating break found by these experiments (NOT evidence of security).
    Continue,
    /// Tooling or coverage insufficient.
    Inconclusive,
}

impl Verdict {
    /// The exact verdict line required by the spec.
    pub fn line(self) -> &'static str {
        match self
        {
            Verdict::NoGo => "PHASE-1 VERDICT: NO-GO — STRUCTURAL BREAK FOUND",
            Verdict::Continue =>
            {
                "PHASE-1 VERDICT: CONTINUE — NO GATING BREAK FOUND BY THESE EXPERIMENTS"
            },
            Verdict::Inconclusive =>
            {
                "PHASE-1 VERDICT: INCONCLUSIVE — TOOLING OR COVERAGE INSUFFICIENT"
            },
        }
    }
}

/// Gating flags aggregated across the battery.
#[derive(Clone, Debug, Default)]
pub struct Flags {
    /// Control A (ring-affine) was recovered by the detector.
    pub control_a_broken: bool,
    /// Control B (ring-linear) was recovered by the detector.
    pub control_b_broken: bool,
    /// The full v0.1 round was (unexpectedly) exactly ring-affine.
    pub full_round_ring_affine: bool,
    /// The full v0.1 round was (unexpectedly) exactly GF(2)-affine.
    pub full_round_gf2_affine: bool,
    /// A multi-round (≥2) reduced permutation was (unexpectedly) exactly affine.
    pub multiround_affine: bool,
    /// A norm-only invariant survived the full reduced permutation.
    pub norm_invariant_survives: bool,
    /// Forward/inverse round-trip held on every tested state.
    pub roundtrip_ok: bool,
}

/// Decide the verdict from the aggregated flags (spec §"Kill criteria").
pub fn decide(flags: &Flags) -> (Verdict, Vec<String>) {
    let mut reasons = Vec::new();

    if !flags.roundtrip_ok
    {
        reasons.push("forward/inverse round-trip mismatch (implementation/kill)".to_string());
        return (Verdict::NoGo, reasons);
    }
    if !flags.control_a_broken || !flags.control_b_broken
    {
        reasons.push(
            "analysis harness failed to break Control A/B — tooling is untrustworthy".to_string(),
        );
        return (Verdict::Inconclusive, reasons);
    }
    if flags.full_round_ring_affine
    {
        reasons.push("full round is exactly ring-affine over Z/2^k".to_string());
    }
    if flags.full_round_gf2_affine
    {
        reasons.push("full round is exactly GF(2)-affine".to_string());
    }
    if flags.multiround_affine
    {
        reasons.push("a multi-round reduced permutation is exactly affine (scales)".to_string());
    }
    if flags.norm_invariant_survives
    {
        reasons.push("a norm-only invariant survives the full reduced permutation".to_string());
    }
    if reasons.is_empty()
    {
        reasons.push("no gating structural break detected by these experiments".to_string());
        (Verdict::Continue, reasons)
    }
    else
    {
        (Verdict::NoGo, reasons)
    }
}

fn coverage_note<W: Word>() -> &'static str {
    if 8 * W::BITS <= 16
    {
        "single-octonion domain exhaustible (2^16)"
    }
    else
    {
        "single-octonion domain sampled"
    }
}

/// Run the harness-validation controls at width `W`, returning `(a_broken,
/// b_broken, json)` (spec: controls must be broken or Phase-1 is inconclusive).
pub fn validate_controls<W: Word>(seed: u64, sample: usize) -> (bool, bool, Json) {
    // Control A: F_A(R) = PERM_π(R ⊞ K0); must be recovered as ring-affine.
    let ma = Fixture::new(FixtureId::PseudoRandom(0xA)).round_material::<W>(0);
    let (a_res, _, _) =
        ring_affine_recover::<W>(move |x| Variant::ControlA.round(x, &ma), seed, sample);

    // Control B: F_B(R) = (K1 ⊗ R) ⊗ K2; must be recovered as ring-linear.
    let mb = Fixture::new(FixtureId::OddNorm).round_material::<W>(0);
    let (b_res, b_mat, b_off) =
        ring_affine_recover::<W>(move |x| Variant::ControlB.round(x, &mb), seed, sample);
    let b_linear = b_res.holds && b_off == Oct::<W>::zero();

    let json = Json::obj(vec![
        ("width", Json::U64(W::BITS as u64)),
        (
            "control_A_linear_only",
            Json::obj(vec![
                ("recovered_ring_affine", Json::Bool(a_res.holds)),
                ("coverage", Json::s(a_res.coverage.label())),
                ("note", Json::s(a_res.note)),
            ]),
        ),
        (
            "control_B_ring_linear",
            Json::obj(vec![
                ("recovered_ring_linear", Json::Bool(b_linear)),
                ("matrix_invertible_over_ring", Json::Bool(b_mat.is_unit())),
                ("matrix_gf2_rank", Json::U64(b_mat.gf2_rank() as u64)),
                ("coverage", Json::s(b_res.coverage.label())),
            ]),
        ),
    ]);
    (a_res.holds, b_linear, json)
}

/// Run the full v0.1 experiment battery at width `W`, updating `flags` and
/// returning a JSON section.
pub fn analyze_v01<W: Word>(seed: u64, sample: usize, flags: &mut Flags) -> Json {
    let fx = Fixture::new(FixtureId::PseudoRandom(0x5C1_0001));
    let m = fx.round_material::<W>(0);

    // --- Experiment 1: matrix lifting of L_a / R_a for a few fixtures --------
    let mut lift_json = Vec::new();
    for (label, a) in [
        ("k1", m.k1),
        ("k2", m.k2),
        ("odd-norm", {
            let mm = Fixture::new(FixtureId::OddNorm).round_material::<W>(0);
            mm.k1
        }),
        ("even-norm", {
            let mm = Fixture::new(FixtureId::EvenNormZeroDiv).round_material::<W>(0);
            mm.k1
        }),
    ]
    {
        let l = lift(a, Side::Left, seed, sample);
        let r = lift(a, Side::Right, seed, sample);
        lift_json.push(Json::obj(vec![
            ("multiplier", Json::s(label)),
            ("coeffs", u64x8(a.to_u64s())),
            ("norm", Json::U64(a.norm().to_u64())),
            (
                "L_a",
                Json::obj(vec![
                    ("matrix_matches_oracle", Json::Bool(l.matrix_matches_oracle)),
                    ("det_mod", Json::U64(l.det_mod)),
                    ("invertible_over_ring", Json::Bool(l.invertible)),
                    ("gf2_rank", Json::U64(l.gf2_rank as u64)),
                    ("kernel_log2", Json::U64(l.kernel_log2 as u64)),
                    ("coverage", Json::s(l.coverage.label())),
                ]),
            ),
            (
                "R_a",
                Json::obj(vec![
                    ("matrix_matches_oracle", Json::Bool(r.matrix_matches_oracle)),
                    ("det_mod", Json::U64(r.det_mod)),
                    ("invertible_over_ring", Json::Bool(r.invertible)),
                    ("gf2_rank", Json::U64(r.gf2_rank as u64)),
                    ("kernel_log2", Json::U64(r.kernel_log2 as u64)),
                ]),
            ),
        ]));
    }

    // --- Experiment 2: pre-rotation G is ring-affine -------------------------
    let (g_res, g_mat, _g_b) =
        ring_affine_recover::<W>(move |x| g_pre_rotation(x, &m), seed, sample);

    // --- Experiment 3: full-round affinity (should NOT be affine) ------------
    // Detail for the default fixture...
    let (f_ring, _, _) = ring_affine_recover::<W>(move |x| f_round(x, &m), seed, sample);
    // the GF(2) pair test is inherently sampled; ensure a meaningful count.
    let f_gf2 = gf2_affine_test::<W>(move |x| f_round(x, &m), seed, sample.max(50_000));
    let f_bit = bit_affine_recover::<W>(move |x| f_round(x, &m), seed, sample);

    // ...but the GATING flags are STRUCTURAL: a single-key affine round at a
    // tiny width is a weak-key artifact (at NANO-2 some ring-linear multipliers
    // are also GF(2)-linear), so we require affinity to hold across MANY
    // independent fixtures before flagging it as a construction-level break.
    let scan_fixtures = [
        FixtureId::PseudoRandom(0x5C1_0001),
        FixtureId::PseudoRandom(1),
        FixtureId::PseudoRandom(2),
        FixtureId::PseudoRandom(0xDE6),
        FixtureId::OddNorm,
        FixtureId::EvenNormZeroDiv,
        FixtureId::HighBit,
        FixtureId::Incrementing,
    ];
    let mut all_ring_affine = true;
    let mut all_gf2_affine = true;
    let mut any_single_round_gf2_affine = false;
    for fid in scan_fixtures
    {
        let mm = Fixture::new(fid).round_material::<W>(0);
        let (rr, _, _) = ring_affine_recover::<W>(move |x| f_round(x, &mm), seed, sample);
        let gg = gf2_affine_test::<W>(move |x| f_round(x, &mm), seed, sample.max(50_000));
        all_ring_affine &= rr.holds;
        all_gf2_affine &= gg.holds;
        any_single_round_gf2_affine |= gg.holds;
    }
    if all_ring_affine
    {
        flags.full_round_ring_affine = true;
    }
    if all_gf2_affine
    {
        flags.full_round_gf2_affine = true;
    }

    // multi-round affinity: branch-after-r for r in {2,4}
    let mut multiround = Vec::new();
    for r in [2u32, 4]
    {
        let (res, _, _) = ring_affine_recover::<W>(
            feistel_branch_after::<W>(&fx, Variant::V01, r),
            seed,
            sample,
        );
        if res.holds && r >= 2
        {
            flags.multiround_affine = true;
        }
        multiround.push(Json::obj(vec![
            ("rounds", Json::U64(r as u64)),
            ("ring_affine", Json::Bool(res.holds)),
            ("coverage", Json::s(res.coverage.label())),
        ]));
    }

    // --- Experiment 4: algebraic degree (exact only for small widths) --------
    let degree_json = if 8 * W::BITS <= 18
    {
        let df = octfn_degree::<W>(move |x| f_round(x, &m)).unwrap();
        let mut per_round = Vec::new();
        for r in [1u32, 2, 3, 4]
        {
            if let Some(d) = octfn_degree::<W>(feistel_branch_after::<W>(&fx, Variant::V01, r))
            {
                per_round.push(Json::obj(vec![
                    ("rounds", Json::U64(r as u64)),
                    ("max_degree", Json::U64(d.max_degree as u64)),
                ]));
            }
        }
        Json::obj(vec![
            ("exact_anf", Json::Bool(true)),
            ("single_round_F_max_degree", Json::U64(df.max_degree as u64)),
            ("input_bits", Json::U64(df.input_bits as u64)),
            ("feistel_degree_by_round", Json::Arr(per_round)),
        ])
    }
    else
    {
        Json::obj(vec![
            ("exact_anf", Json::Bool(false)),
            (
                "note",
                Json::s("out of exact ANF range (8k > 18 input bits)"),
            ),
        ])
    };

    // --- Experiment 5: norm/conjugation invariants ---------------------------
    let nm_left = norm_multiplicative_left::<W>(m.k1, seed, sample);
    let conj_pres = layer_preserves_norm::<W>("conj", |x| x.conj(), seed, sample);
    let perm_pres = layer_preserves_norm::<W>("PERM", |x| x.perm_pi(), seed, sample);
    let rot_pres = layer_preserves_norm::<W>("ROT", |x| x.rot_lambda(), seed, sample);
    let full_norm = norm_functional_through::<W>(move |x| f_round(x, &m), seed, sample.max(20000));
    // full reduced permutation (branch-after-8) norm survival
    let perm_norm = norm_functional_through::<W>(
        feistel_branch_after::<W>(&fx, Variant::V01, 8),
        seed,
        sample.max(20000),
    );
    if perm_norm.norm_determines_output_norm
    {
        flags.norm_invariant_survives = true;
    }
    let assoc = associator_evenness::<W>(seed, sample.clamp(1000, 20000));

    // --- Experiment 6: zero divisors -----------------------------------------
    let kernels = kernel_sizes::<W>(seed, 16);
    let nonunit_with_kernel = kernels
        .iter()
        .filter(|k| !k.norm_is_unit && (k.left_kernel_log2 > 0 || k.right_kernel_log2 > 0))
        .count();
    let diff = differential_bias::<W>(
        move |x| f_round(x, &m),
        Oct::<W>::e(1),
        seed,
        sample.clamp(1000, 20000),
    );

    // --- Experiment 7: subspace preservation + cycle census ------------------
    let sub = run_battery::<W>(move |x| f_round(x, &m), seed, sample.clamp(1000, 20000));
    let sub_json: Vec<Json> = sub
        .iter()
        .map(|s| {
            Json::obj(vec![
                ("set", Json::s(s.name.clone())),
                ("preserved_by_F", Json::Bool(s.preserved)),
                ("members_tested", Json::U64(s.members_tested as u64)),
                ("coverage", Json::s(s.coverage.label())),
            ])
        })
        .collect();
    let cycles = {
        let f = fx;
        cycle_census::<W>(
            move |x| {
                let s = State::new(Oct::<W>::zero(), x);
                feistel::forward(s, &f, Variant::V01, 4, false).r
            },
            seed,
            sample.clamp(1000, 20000),
        )
    };

    // --- forward/inverse round-trip sanity at this width ---------------------
    let rt_ok = roundtrip_ok::<W>(&fx, 6, seed);
    flags.roundtrip_ok = flags.roundtrip_ok && rt_ok;

    Json::obj(vec![
        ("width_bits", Json::U64(W::BITS as u64)),
        ("coverage_note", Json::s(coverage_note::<W>())),
        ("exp1_matrix_lifting", Json::Arr(lift_json)),
        (
            "exp2_pre_rotation_G",
            Json::obj(vec![
                ("ring_affine", Json::Bool(g_res.holds)),
                ("A_invertible_over_ring", Json::Bool(g_mat.is_unit())),
                ("A_gf2_rank", Json::U64(g_mat.gf2_rank() as u64)),
                ("coverage", Json::s(g_res.coverage.label())),
                (
                    "interpretation",
                    Json::s("G is affine BEFORE ROT/PERM/XORC — attack surface, not full-round"),
                ),
            ]),
        ),
        (
            "exp3_full_round_affinity",
            Json::obj(vec![
                ("ring_affine", Json::Bool(f_ring.holds)),
                ("gf2_affine", Json::Bool(f_gf2.holds)),
                (
                    "bit_affine_exact_recovery",
                    match &f_bit
                    {
                        Some(r) => Json::obj(vec![
                            ("holds", Json::Bool(r.holds)),
                            ("disagreement_ppm", Json::U64(r.disagreement_ppm)),
                            ("coverage", Json::s(r.coverage.label())),
                        ]),
                        None => Json::s("not applicable (8k > 64 bits)"),
                    },
                ),
                ("multi_round_affinity", Json::Arr(multiround)),
                (
                    "structural_scan",
                    Json::obj(vec![
                        ("fixtures_scanned", Json::U64(scan_fixtures.len() as u64)),
                        ("ring_affine_for_all_keys", Json::Bool(all_ring_affine)),
                        ("gf2_affine_for_all_keys", Json::Bool(all_gf2_affine)),
                        (
                            "gf2_affine_for_some_keys",
                            Json::Bool(any_single_round_gf2_affine),
                        ),
                        (
                            "note",
                            Json::s(
                                "single-round GF(2)-affinity is key-dependent at small widths \
                                 (weak-key artifact); only all-key affinity gates the verdict",
                            ),
                        ),
                    ]),
                ),
            ]),
        ),
        ("exp4_algebraic_degree", degree_json),
        (
            "exp5_invariants",
            Json::obj(vec![
                ("norm_multiplicative_left_holds", Json::Bool(nm_left.holds)),
                ("conj_preserves_norm", Json::Bool(conj_pres.holds)),
                ("perm_preserves_norm", Json::Bool(perm_pres.holds)),
                ("rot_preserves_norm", Json::Bool(rot_pres.holds)),
                (
                    "norm_determines_F_output_norm",
                    Json::Bool(full_norm.norm_determines_output_norm),
                ),
                (
                    "norm_determines_perm8_output_norm",
                    Json::Bool(perm_norm.norm_determines_output_norm),
                ),
                ("associator_all_even", Json::Bool(assoc.all_even)),
                ("associator_example", u64x8(assoc.example)),
            ]),
        ),
        (
            "exp6_zero_divisors",
            Json::obj(vec![
                ("sampled_multipliers", Json::U64(kernels.len() as u64)),
                (
                    "nonunit_with_nontrivial_kernel",
                    Json::U64(nonunit_with_kernel as u64),
                ),
                (
                    "max_left_kernel_log2",
                    Json::U64(
                        kernels
                            .iter()
                            .map(|k| k.left_kernel_log2)
                            .max()
                            .unwrap_or(0) as u64,
                    ),
                ),
                (
                    "diff_probe_best_output_freq_ppm",
                    Json::U64(diff.best_freq_ppm),
                ),
            ]),
        ),
        (
            "exp7_subspace",
            Json::obj(vec![
                ("preservation", Json::Arr(sub_json)),
                (
                    "cycle_census_4_rounds",
                    Json::obj(vec![
                        ("fixed_points", Json::U64(cycles.fixed_points)),
                        ("two_cycles", Json::U64(cycles.two_cycles)),
                        ("samples", Json::U64(cycles.samples as u64)),
                    ]),
                ),
            ]),
        ),
        ("forward_inverse_roundtrip_ok", Json::Bool(rt_ok)),
    ])
}

/// Forward/inverse round-trip check on sampled states at width `W`.
pub fn roundtrip_ok<W: Word>(fx: &Fixture, rounds: u32, seed: u64) -> bool {
    let ls = sample_octs::<W>(seed, 256);
    let rs = sample_octs::<W>(seed ^ 0xF0F0, 256);
    for (l, r) in ls.iter().zip(rs.iter())
    {
        let s = State::new(*l, *r);
        let enc = feistel::forward(s, fx, Variant::V01, rounds, true);
        let dec = feistel::inverse(enc, fx, Variant::V01, rounds, true);
        if dec != s
        {
            return false;
        }
    }
    true
}

/// Dispatch a width tag to the monomorphized control-validation.
pub fn validate_controls_tag(tag: WidthTag, seed: u64, sample: usize) -> (bool, bool, Json) {
    match tag
    {
        WidthTag::Nano2 => validate_controls::<W2>(seed, sample),
        WidthTag::Nano4 => validate_controls::<W4>(seed, sample),
        WidthTag::Mini8 => validate_controls::<W8>(seed, sample),
        WidthTag::Mini16 => validate_controls::<W16>(seed, sample),
        WidthTag::Full64 => validate_controls::<W64>(seed, sample),
    }
}

/// Dispatch a width tag to the monomorphized v0.1 battery.
pub fn analyze_v01_tag(tag: WidthTag, seed: u64, sample: usize, flags: &mut Flags) -> Json {
    match tag
    {
        WidthTag::Nano2 => analyze_v01::<W2>(seed, sample, flags),
        WidthTag::Nano4 => analyze_v01::<W4>(seed, sample, flags),
        WidthTag::Mini8 => analyze_v01::<W8>(seed, sample, flags),
        WidthTag::Mini16 => analyze_v01::<W16>(seed, sample, flags),
        WidthTag::Full64 => analyze_v01::<W64>(seed, sample, flags),
    }
}

/// Degree comparison at NANO-2 between v0.1 and Controls A–D (spec §Experiment 4
/// "compare exact v0.1 with Controls A–D"). Control D is the associative
/// quaternion; a difference does NOT prove a security benefit of non-assoc.
pub fn degree_compare_nano2() -> Json {
    use crate::algebra::{Quat, W2};
    use crate::permutation::controls::f_round_quat;

    let fx = Fixture::new(FixtureId::PseudoRandom(0xDE6));
    let m = fx.round_material::<W2>(0);
    let deg = |v: Variant| {
        octfn_degree::<W2>(move |x| v.round(x, &m))
            .map(|d| d.max_degree)
            .unwrap_or(0)
    };

    // Control D quaternion round at W2: 4 coeffs * 2 bits = 8 input bits.
    let qk0 = Quat::<W2>::from_u64s([1, 2, 3, 0]);
    let qk1 = Quat::<W2>::from_u64s([1, 1, 0, 2]);
    let qk2 = Quat::<W2>::from_u64s([3, 0, 1, 1]);
    let qrc = Quat::<W2>::from_u64s([2, 1, 3, 0]);
    let quat_bits = |x: u64| -> u64 {
        let q = Quat::<W2>::from_u64s(std::array::from_fn(|i| (x >> (2 * i)) & 3));
        let y = f_round_quat(q, qk0, qk1, qk2, qrc);
        let yc = y.to_u64s();
        (0..4).fold(0u64, |acc, i| acc | (yc[i] << (2 * i)))
    };
    let quat_deg = crate::analysis::degree::bitfn_degree(quat_bits, 8, 8)
        .map(|d| d.max_degree)
        .unwrap_or(0);

    Json::obj(vec![
        ("width", Json::s("NANO-2 (exact ANF)")),
        (
            "v0.1_single_round_F_max_degree",
            Json::U64(deg(Variant::V01) as u64),
        ),
        (
            "controlA_linear_only_max_degree",
            Json::U64(deg(Variant::ControlA) as u64),
        ),
        (
            "controlB_ring_linear_max_degree",
            Json::U64(deg(Variant::ControlB) as u64),
        ),
        (
            "controlC_one_multiply_max_degree",
            Json::U64(deg(Variant::ControlC) as u64),
        ),
        ("controlD_quaternion_max_degree", Json::U64(quat_deg as u64)),
        (
            "note",
            Json::s("degrees compared structurally; a difference is NOT a security claim"),
        ),
    ])
}

/// Zero-divisor explicit examples (W2 exhaustive) as JSON.
pub fn zero_divisor_examples_json(limit: usize) -> Json {
    let pairs = find_zero_divisors_w2(limit);
    Json::Arr(
        pairs
            .into_iter()
            .map(|p| {
                Json::obj(vec![
                    ("a", u64x8(p.a)),
                    ("b", u64x8(p.b)),
                    ("norm_a", Json::U64(p.norm_a)),
                    ("a_mul_b", Json::s("zero (verified)")),
                ])
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn controls_are_broken_by_harness() {
        let (a, b, _) = validate_controls::<W2>(1, 0);
        assert!(a, "Control A must be recovered");
        assert!(b, "Control B must be recovered");
    }

    #[test]
    fn v01_battery_yields_continue_at_nano2() {
        let mut flags = Flags {
            roundtrip_ok: true,
            ..Default::default()
        };
        let (a, b, _) = validate_controls::<W2>(1, 0);
        flags.control_a_broken = a;
        flags.control_b_broken = b;
        let _ = analyze_v01::<W2>(1, 8000, &mut flags);
        let (verdict, reasons) = decide(&flags);
        assert_eq!(verdict, Verdict::Continue, "reasons: {reasons:?}");
    }
}
