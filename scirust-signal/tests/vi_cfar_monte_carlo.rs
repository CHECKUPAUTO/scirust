//! Monte Carlo false-alarm-rate validation for `radar::vi_cfar`.
//!
//! Each detector mode's threshold factor is *calibrated* (see
//! `radar::vi_cfar`'s module docs, "Threshold calibration") against the
//! i.i.d. unit-mean-exponential reference-cell model — CA and RobustTrimmed
//! analytically (exact up to bisection precision), GO/SO by a deterministic
//! seeded Monte-Carlo bisection internal to the crate. This file is the
//! *independent* empirical check: it generates the same clutter model with a
//! **different** RNG (`scirust-stats`'s `SplitMix64` + `Exponential`, not
//! `radar::vi_cfar`'s own private calibration generator) and verifies the
//! observed false-alarm rate is statistically compatible with the design
//! `P_fa`.
//!
//! # Confidence interval
//!
//! For `n` independent Bernoulli(`p_fa`) trials, the empirical proportion's
//! standard error is `SE = sqrt(p_fa (1-p_fa) / n)` (normal approximation to
//! the binomial, valid here since `n * p_fa` and `n * (1-p_fa)` are both
//! large for every `(n, p_fa)` used below). Tests assert
//! `|observed - p_fa| < z * SE` with `z = 4`: under the null hypothesis
//! (calibration is correct) a false failure occurs with probability
//! `~6.3e-5` per test (two-sided normal tail at `z=4`), loose enough to avoid
//! CI flakiness but tight enough to catch a materially wrong calibration
//! (a `2x` error in `alpha`, for instance, moves `p_fa` by far more than
//! `z * SE` at these sample sizes). `n` is chosen so `z * SE` is a small
//! fraction of `p_fa` itself (documented per test) — never a brittle bound
//! around an unrealistically tiny `p_fa` with too few trials.
//!
//! # Fast vs. long-running
//!
//! The tests below (~30,000 samples each) run as part of the normal test
//! suite. A separate, larger-`n`, tighter-bound `#[ignore]`d test is provided
//! for manual/dedicated statistical confirmation — run it with:
//!
//! ```text
//! cargo test -p scirust-signal --test vi_cfar_monte_carlo -- --ignored --nocapture
//! ```
//!
//! Per this repository's documented policy for `#[ignore]`d tests ("sondes de
//! mesure, jamais des gates" — measurement probes, never gates,
//! `docs/REFERENCE.md`), the long-running variant is a deeper measurement,
//! not a substitute correctness gate: every mode is already checked by a
//! non-ignored test above it.

use scirust_signal::radar::vi_cfar::{
    CfarConfig, CfarMode, DetectorPolicy, EdgePolicy, InputValidationPolicy, RobustNoiseEstimator,
    SwitchingThresholds, evaluate_slice,
};
use scirust_stats::{Distribution, Exponential, SplitMix64, chi_square_gof};

/// Generates `n` i.i.d. unit-mean-`Exponential` power samples — the clutter
/// model every calibration in `radar::vi_cfar` targets — via inverse-CDF
/// sampling from `scirust-stats`'s independently-tested `SplitMix64` and
/// `Exponential`.
fn exponential_clutter(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = SplitMix64::new(seed);
    let dist = Exponential::new(1.0);
    (0..n).map(|_| dist.quantile(rng.next_f64())).collect()
}

/// `z * SE` for a binomial proportion `p` over `n` trials.
fn z_bound(p: f64, n: usize, z: f64) -> f64 {
    z * (p * (1.0 - p) / n as f64).sqrt()
}

fn assert_pfa_compatible(observed: f64, target: f64, n: usize, label: &str) {
    let bound = z_bound(target, n, 4.0);
    assert!(
        (observed - target).abs() < bound,
        "{label}: observed P_fa {observed:.5} vs target {target:.5} (n={n}, \
         4*SE bound {bound:.5})"
    );
}

fn base_config(reference_cells: usize, guard_cells: usize, pfa: f64) -> CfarConfig {
    CfarConfig {
        reference_cells,
        guard_cells,
        pfa,
        edge_policy: EdgePolicy::Exclude,
        input_validation: InputValidationPolicy::RejectNegative,
        detector: DetectorPolicy::Ca,
        robust_estimator: RobustNoiseEstimator::TrimmedMean {
            trim_low: 2,
            trim_high: 2,
        },
    }
}

#[test]
fn ca_holds_its_design_pfa() {
    let (reference_cells, guard_cells, pfa) = (16, 2, 0.05);
    let mut config = base_config(reference_cells, guard_cells, pfa);
    config.detector = DetectorPolicy::Ca;
    let power = exponential_clutter(30_000, 0x5647_4152_5f43_4131);
    let decisions = evaluate_slice(&power, &config).unwrap();
    let n = decisions.len();
    let observed = decisions.iter().filter(|d| d.detected).count() as f64 / n as f64;
    assert_pfa_compatible(observed, pfa, n, "CA");
}

#[test]
fn go_holds_its_design_pfa() {
    let (reference_cells, guard_cells, pfa) = (16, 2, 0.05);
    let mut config = base_config(reference_cells, guard_cells, pfa);
    config.detector = DetectorPolicy::Go;
    let power = exponential_clutter(30_000, 0x5647_4152_5f47_4f31);
    let decisions = evaluate_slice(&power, &config).unwrap();
    let n = decisions.len();
    let observed = decisions.iter().filter(|d| d.detected).count() as f64 / n as f64;
    assert_pfa_compatible(observed, pfa, n, "GO (numerically calibrated)");
}

#[test]
fn so_holds_its_design_pfa() {
    let (reference_cells, guard_cells, pfa) = (16, 2, 0.05);
    let mut config = base_config(reference_cells, guard_cells, pfa);
    config.detector = DetectorPolicy::So;
    let power = exponential_clutter(30_000, 0x5647_4152_5f53_4f31);
    let decisions = evaluate_slice(&power, &config).unwrap();
    let n = decisions.len();
    let observed = decisions.iter().filter(|d| d.detected).count() as f64 / n as f64;
    assert_pfa_compatible(observed, pfa, n, "SO (numerically calibrated)");
}

#[test]
fn robust_trimmed_holds_its_design_pfa() {
    let (reference_cells, guard_cells, pfa) = (16, 2, 0.05);
    let mut config = base_config(reference_cells, guard_cells, pfa);
    config.detector = DetectorPolicy::AlwaysRobust;
    config.robust_estimator = RobustNoiseEstimator::TrimmedMean {
        trim_low: 4,
        trim_high: 4,
    };
    let power = exponential_clutter(30_000, 0x5647_4152_5f52_4f31);
    let decisions = evaluate_slice(&power, &config).unwrap();
    let n = decisions.len();
    let observed = decisions.iter().filter(|d| d.detected).count() as f64 / n as f64;
    assert_pfa_compatible(observed, pfa, n, "RobustTrimmed");
}

#[test]
fn robust_censored_holds_its_design_pfa() {
    let (reference_cells, guard_cells, pfa) = (16, 2, 0.05);
    let mut config = base_config(reference_cells, guard_cells, pfa);
    config.detector = DetectorPolicy::AlwaysRobust;
    config.robust_estimator = RobustNoiseEstimator::CensoredMean {
        trim_low: 4,
        trim_high: 4,
    };
    let power = exponential_clutter(30_000, 0x5647_4152_5f52_4332);
    let decisions = evaluate_slice(&power, &config).unwrap();
    let n = decisions.len();
    let observed = decisions.iter().filter(|d| d.detected).count() as f64 / n as f64;
    assert_pfa_compatible(observed, pfa, n, "RobustCensored");
}

#[test]
fn classical_switch_under_pure_homogeneous_clutter_tracks_ca_pfa() {
    // Under pure i.i.d. exponential clutter every CUT *should* classify as
    // homogeneous/homogeneous with MR close to 1 (CA), so the composite
    // switch's empirical P_fa should track CA's own calibrated P_fa closely
    // here. This is deliberately a narrower claim than "the composite
    // detector is exactly CFAR" (not established — see the module docs):
    // it only checks the switch's behavior in the one regime where it is
    // expected to reduce to CA almost everywhere.
    let (reference_cells, guard_cells, pfa) = (16, 2, 0.05);
    let mut config = base_config(reference_cells, guard_cells, pfa);
    config.detector =
        DetectorPolicy::ClassicalViCfar(scirust_signal::radar::vi_cfar::SwitchingThresholds {
            k_vi: 6.0,
            k_mr: 2.0,
        });
    let power = exponential_clutter(30_000, 0x5647_4152_5f56_4931);
    let decisions = evaluate_slice(&power, &config).unwrap();
    let n = decisions.len();
    let ca_fraction = decisions
        .iter()
        .filter(|d| d.mode == scirust_signal::radar::vi_cfar::CfarMode::Ca)
        .count() as f64
        / n as f64;
    assert!(
        ca_fraction > 0.95,
        "expected CA to dominate under pure homogeneous clutter: {ca_fraction}"
    );
    let observed = decisions.iter().filter(|d| d.detected).count() as f64 / n as f64;
    assert_pfa_compatible(
        observed,
        pfa,
        n,
        "ClassicalViCfar under homogeneous clutter",
    );
}

/// Draws `n` independent samples where cell `i`'s clutter power is
/// `levels[i] * Exponential(1)` — still genuine i.i.d.-*shape* exponential
/// clutter at every cell, just with a deterministic, per-index power *scale*
/// (`levels`) in place of [`exponential_clutter`]'s uniform scale. No
/// synthetic target is ever added by any caller below: every "detected"
/// outcome remains, by construction, a genuine false alarm against whatever
/// local clutter power happens to be in effect at that cell.
fn scaled_exponential_clutter(levels: &[f64], seed: u64) -> Vec<f64> {
    let mut rng = SplitMix64::new(seed);
    let dist = Exponential::new(1.0);
    levels
        .iter()
        .map(|&level| level * dist.quantile(rng.next_f64()))
        .collect()
}

/// A permanent step in clutter power every `period` samples (low for the
/// first half of each period, `high` for the second) — the classical
/// "clutter edge" that pushes CA-CFAR's pooled estimate into the wrong
/// regime and motivates the switch's greatest-of branch.
fn clutter_edge_levels(n: usize, period: usize, high: f64) -> Vec<f64> {
    (0..n)
        .map(|i| if i % period < period / 2 { 1.0 } else { high })
        .collect()
}

/// Unit-power clutter with a single strong point-clutter cell every `period`
/// samples (at a fixed `offset` within each period) — non-homogeneous in
/// exactly one reference half-window for CUTs near that cell, the other half
/// staying clean. Models one interferer, not a permanent level shift.
fn single_interferer_levels(n: usize, period: usize, offset: usize, high: f64) -> Vec<f64> {
    let mut levels = vec![1.0; n];
    let mut i = offset;
    while i < n
    {
        levels[i] = high;
        i += period;
    }
    levels
}

/// Unit-power clutter with a *pair* of strong point-clutter cells `gap`
/// samples apart, repeating every `period` samples — spaced so that, as the
/// CUT sweeps continuously across the array, a genuine range of CUT
/// positions ends up with an elevated cell in *both* reference half-windows
/// at once (double contamination — the case classical VI-CFAR's four-mode
/// switch has no branch for; see `radar::vi_cfar`'s module docs).
fn double_interferer_levels(
    n: usize,
    period: usize,
    offset: usize,
    gap: usize,
    high: f64,
) -> Vec<f64> {
    let mut levels = vec![1.0; n];
    let mut i = offset;
    while i + gap < n
    {
        levels[i] = high;
        levels[i + gap] = high;
        i += period;
    }
    levels
}

#[test]
fn classical_switch_pfa_across_non_homogeneous_scenarios_gof() {
    // Strengthens the pure-homogeneous-clutter check above with
    // `scirust_stats::chi_square_gof`: three scenarios that force the
    // composite switch through its CA (row 1), SO (rows 3/4 — one
    // interferer) and Robust (row 5 — an interferer in *both* halves)
    // branches are pooled into a single joint statistic. All three are
    // constructed so the switch's *design* P_fa claim is expected to hold
    // essentially exactly, not just approximately, even though the
    // population is non-homogeneous:
    //   - CA's own branch is unaffected by the other scenarios' interferers.
    //   - a single very large interferer (`high = 50`) in one half makes
    //     that half's mean overwhelmingly larger than the clean half's with
    //     probability ~1, so smallest-of's exact calibration (min of two
    //     true i.i.d. Gamma(n,1) half-sums) is realized almost surely.
    //   - the injected interferer count per half (1) is well under the
    //     robust estimator's trim count (4 below), so the trimmed mean's
    //     exact calibration (immune to contamination up to the trim count —
    //     see the module docs, "Robust double-contamination strategy")
    //     applies unchanged.
    // A genuine clutter *edge* (a permanent level shift, not a point
    // interferer) is deliberately excluded from this two-sided combination:
    // GO's calibration there is honestly one-sided (P_fa at or below
    // target, never an exact two-sided match — see
    // `clutter_switch_edge_branch_does_not_exceed_its_design_pfa` below), so
    // folding it into a two-sided chi-square statistic would be the wrong
    // tool for that specific claim.
    let (reference_cells, guard_cells, pfa) = (16, 2, 0.05);
    let n = 40_000;
    let mut config = base_config(reference_cells, guard_cells, pfa);
    config.detector = DetectorPolicy::ClassicalViCfar(SwitchingThresholds {
        k_vi: 6.0,
        k_mr: 2.0,
    });
    config.robust_estimator = RobustNoiseEstimator::TrimmedMean {
        trim_low: 4,
        trim_high: 4,
    };

    let scenarios: [(&str, Vec<f64>, u64); 3] = [
        ("homogeneous", vec![1.0; n], 0x4753_4f46_5f48_4f4d),
        (
            "single interferer (SO branch)",
            single_interferer_levels(n, 125, 10, 50.0),
            0x4753_4f46_5f53_4f31,
        ),
        (
            "double interferer (robust branch)",
            double_interferer_levels(n, 200, 10, 20, 50.0),
            0x4753_4f46_5f52_4f31,
        ),
    ];

    let mut observed = Vec::with_capacity(scenarios.len() * 2);
    let mut expected = Vec::with_capacity(scenarios.len() * 2);
    for (label, levels, seed) in &scenarios
    {
        let power = scaled_exponential_clutter(levels, *seed);
        let decisions = evaluate_slice(&power, &config).unwrap();
        let n_valid = decisions.len() as f64;
        let detected = decisions.iter().filter(|d| d.detected).count() as f64;
        println!("{label}: {detected}/{n_valid} detected (target rate {pfa})");
        observed.push(detected);
        observed.push(n_valid - detected);
        expected.push(n_valid * pfa);
        expected.push(n_valid * (1.0 - pfa));
    }

    let result = chi_square_gof(&observed, &expected, 0).unwrap();
    assert!(
        result.p_value > 1.0e-4,
        "composite switch's P_fa deviates jointly across scenarios: chi2={:.3} df={:.0} \
         p={:.6} observed={observed:?} expected={expected:?}",
        result.statistic,
        result.df,
        result.p_value
    );
}

#[test]
fn clutter_switch_edge_branch_does_not_exceed_its_design_pfa() {
    // Classical CFAR literature's claim for greatest-of at a clutter edge is
    // one-sided: P_fa stays *at or below* the design target (GO always uses
    // the larger of the two half-window estimates, which is never lower than
    // the correctly-matching half's own estimate) — not that it hits the
    // target exactly. This checks that documented, one-sided safety property
    // directly against the CUTs where the switch itself selected
    // `CfarMode::Go`, rather than folding it into a two-sided test it was
    // never expected to pass exactly.
    let (reference_cells, guard_cells, pfa) = (16, 2, 0.05);
    let n = 48_000;
    let mut config = base_config(reference_cells, guard_cells, pfa);
    config.detector = DetectorPolicy::ClassicalViCfar(SwitchingThresholds {
        k_vi: 6.0,
        k_mr: 2.0,
    });
    let power =
        scaled_exponential_clutter(&clutter_edge_levels(n, 400, 15.0), 0x4753_4f46_5f45_4447);
    let decisions = evaluate_slice(&power, &config).unwrap();
    let go_decisions: Vec<_> = decisions
        .iter()
        .filter(|d| d.mode == CfarMode::Go)
        .collect();
    let n_go = go_decisions.len();
    assert!(
        n_go > 500,
        "expected the edge to trip the GO branch often enough to measure: n_go={n_go}"
    );
    let observed = go_decisions.iter().filter(|d| d.detected).count() as f64 / n_go as f64;
    let bound = pfa + z_bound(pfa, n_go, 4.0);
    assert!(
        observed <= bound,
        "GO branch at a clutter edge should control P_fa at or below target: \
         observed={observed:.5} target={pfa:.5} bound={bound:.5} n_go={n_go}"
    );
}

/// Larger-`n`, tighter-bound confirmation of the same four claims above.
/// Measurement probe, not a gate (see the module docs) — run manually with
/// `cargo test -p scirust-signal --test vi_cfar_monte_carlo -- --ignored --nocapture`.
#[test]
#[ignore]
fn long_running_pfa_confirmation() {
    let (reference_cells, guard_cells, pfa) = (16, 2, 0.02);
    let n_samples = 400_000;
    let cases: [(DetectorPolicy, Option<RobustNoiseEstimator>, u64, &str); 5] = [
        (DetectorPolicy::Ca, None, 0x4c4f_4e47_5f43_4131, "CA"),
        (DetectorPolicy::Go, None, 0x4c4f_4e47_5f47_4f31, "GO"),
        (DetectorPolicy::So, None, 0x4c4f_4e47_5f53_4f31, "SO"),
        (
            DetectorPolicy::AlwaysRobust,
            Some(RobustNoiseEstimator::TrimmedMean {
                trim_low: 4,
                trim_high: 4,
            }),
            0x4c4f_4e47_5f52_4f31,
            "RobustTrimmed",
        ),
        (
            DetectorPolicy::AlwaysRobust,
            Some(RobustNoiseEstimator::CensoredMean {
                trim_low: 4,
                trim_high: 4,
            }),
            0x4c4f_4e47_5f52_4332,
            "RobustCensored",
        ),
    ];
    for (detector, robust_estimator, seed, label) in cases
    {
        let mut config = base_config(reference_cells, guard_cells, pfa);
        config.detector = detector;
        if let Some(robust_estimator) = robust_estimator
        {
            config.robust_estimator = robust_estimator;
        }
        let power = exponential_clutter(n_samples, seed);
        let decisions = evaluate_slice(&power, &config).unwrap();
        let n = decisions.len();
        let observed = decisions.iter().filter(|d| d.detected).count() as f64 / n as f64;
        println!("{label}: observed P_fa={observed:.5} target={pfa:.5} n={n}");
        assert_pfa_compatible(observed, pfa, n, label);
    }
}
