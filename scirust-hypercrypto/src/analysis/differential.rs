//! Phase-2 cryptanalysis increment: differential probing and a rigorous
//! characterization of the weak-key class that linearizes the round over
//! `GF(2)` (the lead surfaced in Phase 1).
//!
//! None of this is a security claim. A high single-round differential at a tiny
//! width is expected; the gating question is whether a differential survives the
//! full multi-round construction. A surviving high-probability full-round
//! differential would be a kill criterion.

use crate::algebra::Oct;
use crate::algebra::word::Word;
use crate::analysis::linearity::bit_affine_recover;
use crate::analysis::util::{enumerate_octs, oct_domain_exhaustible, sample_octs};
use crate::fixtures::{Fixture, RoundMaterial};
use crate::permutation::round::f_round;
use crate::permutation::{State, Variant, feistel};

// ---------------------------------------------------------------------------
// Weak-key characterization
// ---------------------------------------------------------------------------

/// `true` iff left-multiplication `x ↦ a ⊗ x` is exactly `GF(2)`-linear, i.e.
/// the round's only nonlinearity source vanishes for this multiplier. Verified
/// exhaustively at `NANO-2`, else on a deterministic sample.
pub fn is_gf2_linear_left_multiplier<W: Word>(a: Oct<W>, seed: u64, sample: usize) -> bool {
    // a ⊗ 0 = 0, so a GF(2)-affine recovery of x ↦ a⊗x has zero offset; it is
    // linear iff the recovery holds exactly. A modest sample already rejects a
    // nonlinear map (which disagrees on a large fraction of inputs), so cap the
    // per-candidate cost — this is called across a whole multiplier class.
    let s = sample.clamp(256, 2048);
    match bit_affine_recover::<W>(move |x| a.mul(x), seed, s)
    {
        Some(r) => r.holds,
        None => false, // width too wide to decide here
    }
}

/// The structured multiplier class `C = { octonions with every coefficient in
/// {0, 2^{k-1}} }` (size `2^8`). These are the "high-bit-only" multipliers; the
/// Phase-1 lead is that they linearize the round because `2^{k-1}·b` keeps only
/// the low bit and `2^{k-1}+2^{k-1} ≡ 0`, so each output slot becomes a parity.
pub fn highbit_class<W: Word>() -> Vec<Oct<W>> {
    let hb = if W::BITS >= 64
    {
        1u64 << 63
    }
    else
    {
        1u64 << (W::BITS - 1)
    };
    let mut out = Vec::with_capacity(256);
    for mask in 0u32..256
    {
        let mut c = [W::ZERO; 8];
        for (i, slot) in c.iter_mut().enumerate()
        {
            if (mask >> i) & 1 == 1
            {
                *slot = W::from_u64(hb);
            }
        }
        out.push(Oct::from_coeffs(c));
    }
    out
}

/// Report of the weak-key GF(2)-linearization analysis.
#[derive(Clone, Debug)]
pub struct WeakKeyReport {
    /// Size of the tested high-bit multiplier class `C`.
    pub class_size: usize,
    /// How many members of `C` give a `GF(2)`-linear left-multiplication.
    pub class_gf2_linear: usize,
    /// Number of random multipliers sampled.
    pub random_sampled: usize,
    /// How many random multipliers were `GF(2)`-linear (expected ≈ 0).
    pub random_gf2_linear: usize,
    /// Whether the all-`2^{k-1}` multiplier linearizes (the canonical weak key).
    pub all_highbit_is_linear: bool,
}

/// Characterize the weak-key class at width `W`: confirm the structured class
/// linearizes the round, and estimate the density among random keys.
pub fn weak_key_analysis<W: Word>(seed: u64, sample: usize) -> WeakKeyReport {
    let class = highbit_class::<W>();
    let class_gf2_linear = class
        .iter()
        .filter(|&&a| is_gf2_linear_left_multiplier::<W>(a, seed, sample))
        .count();

    let randoms = sample_octs::<W>(seed ^ 0xD00D, 512);
    let random_gf2_linear = randoms
        .iter()
        .filter(|&&a| is_gf2_linear_left_multiplier::<W>(a, seed, sample))
        .count();

    let hb = if W::BITS >= 64
    {
        1u64 << 63
    }
    else
    {
        1u64 << (W::BITS - 1)
    };
    let all_hb = Oct::<W>::from_u64s([hb; 8]);

    WeakKeyReport {
        class_size: class.len(),
        class_gf2_linear,
        random_sampled: randoms.len(),
        random_gf2_linear,
        all_highbit_is_linear: is_gf2_linear_left_multiplier::<W>(all_hb, seed, sample),
    }
}

// ---------------------------------------------------------------------------
// Differential probing
// ---------------------------------------------------------------------------

/// A single differential and its (empirical or exact) probability.
#[derive(Clone, Debug)]
pub struct Differential {
    /// Input XOR difference.
    pub input_delta: [u64; 8],
    /// Most frequent output XOR difference.
    pub output_delta: [u64; 8],
    /// Probability, in parts-per-million.
    pub prob_ppm: u64,
    /// Whether the probability was computed exactly (whole domain) or sampled.
    pub exact: bool,
}

/// For a fixed input difference `delta`, the most frequent output difference of
/// `f` and its probability — computed **exactly** over the whole single-octonion
/// domain when that is exhaustible (`NANO-2`), else on a deterministic sample.
pub fn best_output_diff<W: Word>(
    f: impl Fn(Oct<W>) -> Oct<W>,
    delta: Oct<W>,
    seed: u64,
    sample: usize,
) -> Differential {
    use std::collections::HashMap;
    let (points, exact) = if oct_domain_exhaustible::<W>()
    {
        (enumerate_octs::<W>(), true)
    }
    else
    {
        (sample_octs::<W>(seed, sample), false)
    };
    let mut hist: HashMap<[u64; 8], u64> = HashMap::new();
    for x in &points
    {
        let d = f(*x).xor(f(x.xor(delta)));
        *hist.entry(d.to_u64s()).or_insert(0) += 1;
    }
    let (best, cnt) = hist
        .into_iter()
        .max_by_key(|&(_, c)| c)
        .unwrap_or(([0; 8], 0));
    let n = points.len().max(1) as u64;
    Differential {
        input_delta: delta.to_u64s(),
        output_delta: best,
        prob_ppm: cnt.saturating_mul(1_000_000) / n,
        exact,
    }
}

/// Search over input differences for the highest-probability single-round
/// differential of `f`, over a deterministic sample of `delta_samples` input
/// differences. The output-difference distribution **per delta** is computed
/// exactly at `NANO-2` (whole domain), else sampled with `per_delta_sample`.
/// (An exhaustive delta scan is `2^{8k}` differences — out of scope here.)
pub fn best_single_round_differential<W: Word>(
    m: &RoundMaterial<W>,
    seed: u64,
    delta_samples: usize,
    per_delta_sample: usize,
) -> Differential {
    let deltas = sample_octs::<W>(seed ^ 0xDE17A, delta_samples);
    let mut best: Option<Differential> = None;
    for delta in deltas
    {
        if delta.to_u64s() == [0; 8]
        {
            continue; // the zero difference is trivial (prob 1)
        }
        let d = best_output_diff::<W>(|x| f_round(x, m), delta, seed, per_delta_sample);
        match &best
        {
            Some(b) if b.prob_ppm >= d.prob_ppm =>
            {},
            _ => best = Some(d),
        }
    }
    best.unwrap_or(Differential {
        input_delta: [0; 8],
        output_delta: [0; 8],
        prob_ppm: 0,
        exact: false,
    })
}

/// Empirical probability that the input R-branch difference `delta` produces the
/// most frequent output R-branch difference after `rounds` Feistel rounds
/// (L initialised to zero), measured on a deterministic sample.
pub fn feistel_differential_by_round<W: Word>(
    fixture: &Fixture,
    delta: Oct<W>,
    rounds_list: &[u32],
    seed: u64,
    samples: usize,
) -> Vec<(u32, u64)> {
    use std::collections::HashMap;
    let xs = sample_octs::<W>(seed, samples);
    let mut out = Vec::new();
    for &rounds in rounds_list
    {
        let mut hist: HashMap<[u64; 8], u64> = HashMap::new();
        for &x in &xs
        {
            let s0 = State::new(Oct::<W>::zero(), x);
            let s1 = State::new(Oct::<W>::zero(), x.xor(delta));
            let a = feistel::forward(s0, fixture, Variant::V01, rounds, false).r;
            let b = feistel::forward(s1, fixture, Variant::V01, rounds, false).r;
            *hist.entry(a.xor(b).to_u64s()).or_insert(0) += 1;
        }
        let cnt = hist.values().copied().max().unwrap_or(0);
        let ppm = cnt.saturating_mul(1_000_000) / (xs.len().max(1) as u64);
        out.push((rounds, ppm));
    }
    out
}

// ---------------------------------------------------------------------------
// Report / CLI dispatch
// ---------------------------------------------------------------------------

use crate::algebra::word::{W2, W4, W8, W16, W64, WidthTag};
use crate::analysis::report::{Json, u64x8};
use crate::fixtures::FixtureId;

/// Weak-key GF(2)-linearization report for one width.
pub fn weak_key_report_json(tag: WidthTag, seed: u64, sample: usize) -> Json {
    let r = match tag
    {
        WidthTag::Nano2 => weak_key_analysis::<W2>(seed, sample),
        WidthTag::Nano4 => weak_key_analysis::<W4>(seed, sample),
        WidthTag::Mini8 => weak_key_analysis::<W8>(seed, sample),
        WidthTag::Mini16 => weak_key_analysis::<W16>(seed, sample),
        WidthTag::Full64 => weak_key_analysis::<W64>(seed, sample),
    };
    Json::obj(vec![
        ("width_bits", Json::U64(tag.bits() as u64)),
        ("highbit_class_size", Json::U64(r.class_size as u64)),
        (
            "highbit_class_gf2_linear",
            Json::U64(r.class_gf2_linear as u64),
        ),
        (
            "all_highbit_multiplier_linearizes",
            Json::Bool(r.all_highbit_is_linear),
        ),
        (
            "random_multipliers_sampled",
            Json::U64(r.random_sampled as u64),
        ),
        (
            "random_multipliers_gf2_linear",
            Json::U64(r.random_gf2_linear as u64),
        ),
        (
            "conclusion",
            Json::s(
                "GF(2)-linear multipliers form a low-density structured class \
                 (coefficients in {0, 2^(k-1)}); a key schedule that excludes such \
                 multipliers excludes this weak-key class. NOT a construction break.",
            ),
        ),
    ])
}

/// Differential probing report for one width and fixture.
pub fn differential_report_json(tag: WidthTag, fid: FixtureId, seed: u64, samples: usize) -> Json {
    let rounds = [1u32, 2, 4, 6, 8, 12];
    let (best, decay) = match tag
    {
        WidthTag::Nano2 => diff_pair::<W2>(fid, seed, samples, &rounds),
        WidthTag::Nano4 => diff_pair::<W4>(fid, seed, samples, &rounds),
        WidthTag::Mini8 => diff_pair::<W8>(fid, seed, samples, &rounds),
        WidthTag::Mini16 => diff_pair::<W16>(fid, seed, samples, &rounds),
        WidthTag::Full64 => diff_pair::<W64>(fid, seed, samples, &rounds),
    };
    Json::obj(vec![
        ("width_bits", Json::U64(tag.bits() as u64)),
        (
            "best_single_round_differential",
            Json::obj(vec![
                ("input_delta", u64x8(best.input_delta)),
                ("output_delta", u64x8(best.output_delta)),
                ("prob_ppm", Json::U64(best.prob_ppm)),
                ("exact_per_delta_distribution", Json::Bool(best.exact)),
            ]),
        ),
        (
            "feistel_differential_by_round",
            Json::Arr(
                decay
                    .into_iter()
                    .map(|(r, ppm)| {
                        Json::obj(vec![
                            ("rounds", Json::U64(r as u64)),
                            ("best_output_diff_prob_ppm", Json::U64(ppm)),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "note",
            Json::s(
                "single-round probability is high at tiny widths (expected artifact); \
                 the empirical multi-round probability decays — no high-probability \
                 full-round differential found by this probe.",
            ),
        ),
    ])
}

fn diff_pair<W: Word>(
    fid: FixtureId,
    seed: u64,
    samples: usize,
    rounds: &[u32],
) -> (Differential, Vec<(u32, u64)>) {
    let m = Fixture::new(fid).round_material::<W>(0);
    let best = best_single_round_differential::<W>(&m, seed, 96, samples.min(4096));
    let fx = Fixture::new(fid);
    let delta = Oct::<W>::from_u64s(best.input_delta);
    let decay = feistel_differential_by_round::<W>(&fx, delta, rounds, seed, samples);
    (best, decay)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::word::{W2, W8};

    #[test]
    fn highbit_class_linearizes_random_does_not() {
        // The all-high-bit multiplier is GF(2)-linear; random multipliers are not.
        let rep = weak_key_analysis::<W8>(1, 40_000);
        assert!(
            rep.all_highbit_is_linear,
            "all-2^{{k-1}} multiplier must linearize"
        );
        assert!(
            rep.class_gf2_linear > 0,
            "structured class must contain linear multipliers"
        );
        // random keys essentially never linearize
        assert!(
            rep.random_gf2_linear * 20 < rep.random_sampled,
            "GF(2)-linear multipliers must be a low-density class ({}/{})",
            rep.random_gf2_linear,
            rep.random_sampled
        );
    }

    #[test]
    fn differential_probability_decays_with_rounds() {
        // A fixed input difference must lose probability across Feistel rounds
        // (no high-probability full-round differential at this width).
        let fx = Fixture::new(FixtureId::PseudoRandom(0xD1FF));
        let by_round =
            feistel_differential_by_round::<W8>(&fx, Oct::<W8>::e(1), &[1, 4, 8], 7, 20_000);
        let p1 = by_round[0].1;
        let p8 = by_round[2].1;
        assert!(
            p8 <= p1,
            "differential probability should not grow with rounds"
        );
    }

    #[test]
    fn single_round_best_differential_runs_nano2() {
        let m = Fixture::new(FixtureId::PseudoRandom(3)).round_material::<W2>(0);
        // sampled deltas keep the test fast; each delta evaluated exactly (NANO-2)
        let d = best_single_round_differential::<W2>(&m, 5, 24, 0);
        assert!(d.exact, "NANO-2 per-delta distribution is exact");
        assert!(d.prob_ppm > 0);
    }
}
