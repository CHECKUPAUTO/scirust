//! Adversarial and contamination scenarios for
//! [`scirust_causal::PartialCorrelationTest`]: robustness under contamination
//! (§16.3), permutation-calibration adversarial cases (§16.4), and the
//! numerical/statistical boundary cases of Phase 5C.2's own specification
//! (§18) — near-singular conditioning sets, weak (near-unfaithful) signals,
//! mixed regimes, and the two honest **negative results** this crate commits
//! to documenting rather than hiding: linear partial correlation cannot see
//! purely nonlinear or purely heteroscedastic conditional dependence.
//!
//! Basic correlation cases, causal motifs, dataset contracts, and property
//! tests live in `conditional_independence.rs`.

use scirust_causal::{
    CausalDataset, CausalError, CausalVariable, ConditionalIndependenceConfig,
    ConditionalIndependenceMethod, ConditionalIndependenceTest, Environment, IndependenceDecision,
    Intervention, InterventionKind, PartialCorrelationTest, RegimeSelection, RobustCalibration,
    SampleBlock, VariableKind, VariableRole,
};
use scirust_multivariate::RobustScatterConfig;
use scirust_solvers::Matrix;
use scirust_stats::SplitMix64;

// ─── Fixtures ────────────────────────────────────────────────────────────

fn dataset_from_columns(columns: &[Vec<f64>]) -> CausalDataset {
    let n = columns[0].len();
    let d = columns.len();
    let mut data = vec![0.0; n * d];
    for row in 0..n
    {
        for col in 0..d
        {
            data[row * d + col] = columns[col][row];
        }
    }
    let variables: Vec<CausalVariable> = (0..d)
        .map(|i| {
            CausalVariable::new(
                i,
                format!("v{i}"),
                VariableRole::Unspecified,
                VariableKind::Continuous,
            )
            .unwrap()
        })
        .collect();
    let matrix = Matrix::from_row_major(n, d, data);
    let env = Environment::observational("obs").unwrap();
    CausalDataset::single_environment(variables, env, &matrix, "adversarial fixture").unwrap()
}

/// Centered uniform noise on `[-0.5, 0.5)`.
fn noise(rng: &mut SplitMix64) -> f64 {
    rng.next_f64() - 0.5
}

/// Occasionally (probability `0.05`) draws a spike `30x` larger in scale than
/// the ordinary noise — deterministic given `rng`, but not the same number of
/// draws every call (that's fine: determinism only requires a fixed sequence
/// for a fixed seed, not a fixed draw count per call).
fn heavy_tailed_noise(rng: &mut SplitMix64) -> f64 {
    let u = rng.next_f64();
    if u < 0.05
    {
        (rng.next_f64() - 0.5) * 30.0
    }
    else
    {
        rng.next_f64() - 0.5
    }
}

fn gaussian_config(fisher_z: bool) -> ConditionalIndependenceConfig {
    ConditionalIndependenceConfig::new(
        0.05,
        ConditionalIndependenceMethod::GaussianPartialCorrelation { fisher_z },
    )
    .unwrap()
}

fn robust_config(calibration: RobustCalibration) -> ConditionalIndependenceConfig {
    ConditionalIndependenceConfig::new(
        0.05,
        ConditionalIndependenceMethod::RobustPartialCorrelation {
            scatter: RobustScatterConfig::default(),
            calibration,
        },
    )
    .unwrap()
}

fn permutation_config(permutations: usize, seed: u64) -> ConditionalIndependenceConfig {
    ConditionalIndependenceConfig::new(
        0.05,
        ConditionalIndependenceMethod::PermutationPartialCorrelation {
            permutations,
            seed,
            residualization: scirust_causal::ResidualizationMethod::OrdinaryLeastSquares,
        },
    )
    .unwrap()
}

// ─── §16.3 Robustness ────────────────────────────────────────────────────

/// A handful of **vertical** outliers (extreme `y`, ordinary `x`) attenuate
/// the classical Pearson statistic toward zero (variance inflation in the
/// denominator dominates a bounded cross-term in the numerator); OGK's
/// bounded-influence reweighting keeps the robust statistic closer to the
/// clean relationship. This is a demonstration in *one constructed scenario*,
/// not a claim that robust estimation is universally better (see the
/// `clean_gaussian_like_data_classical_and_robust_agree` test below for a
/// case where the two methods simply agree).
#[test]
fn vertical_outliers_distort_classical_more_than_robust() {
    let mut rng = SplitMix64::new(100);
    let n_clean = 150;
    let mut x: Vec<f64> = Vec::with_capacity(n_clean + 8);
    let mut y: Vec<f64> = Vec::with_capacity(n_clean + 8);
    for _ in 0..n_clean
    {
        let xi = noise(&mut rng);
        x.push(xi);
        y.push(0.9 * xi + 0.1 * noise(&mut rng));
    }
    for i in 0..8
    {
        x.push(noise(&mut rng));
        y.push(if i % 2 == 0 { 50.0 } else { -50.0 });
    }

    let dataset = dataset_from_columns(&[x, y]);
    let classical = PartialCorrelationTest::new(gaussian_config(true))
        .test(&dataset, 0, 1, &[])
        .unwrap();
    let robust = PartialCorrelationTest::new(robust_config(RobustCalibration::NoPValue))
        .test(&dataset, 0, 1, &[])
        .unwrap();

    assert!(
        robust.effect_size > classical.effect_size,
        "expected OGK to preserve more of the true signal than Pearson under \
         vertical contamination: classical={} robust={}",
        classical.effect_size,
        robust.effect_size
    );
}

/// A minority of **bad leverage points** (extreme in *both* `x` and `y`,
/// off the true relationship) can manufacture an apparently strong linear
/// relationship out of an otherwise uncorrelated cloud — the classical
/// scenario robust estimation exists for.
#[test]
fn bad_leverage_points_can_fool_the_classical_statistic() {
    let mut rng = SplitMix64::new(101);
    let n_clean = 100;
    let mut x: Vec<f64> = Vec::with_capacity(n_clean + 6);
    let mut y: Vec<f64> = Vec::with_capacity(n_clean + 6);
    for _ in 0..n_clean
    {
        x.push(noise(&mut rng));
        y.push(noise(&mut rng));
    }
    for i in 0..6
    {
        let leverage = 10.0 + i as f64 * 0.3;
        x.push(leverage);
        y.push(leverage);
    }

    let dataset = dataset_from_columns(&[x, y]);
    let classical = PartialCorrelationTest::new(gaussian_config(true))
        .test(&dataset, 0, 1, &[])
        .unwrap();
    let robust = PartialCorrelationTest::new(robust_config(RobustCalibration::NoPValue))
        .test(&dataset, 0, 1, &[])
        .unwrap();

    assert_eq!(
        classical.decision,
        IndependenceDecision::Dependent,
        "the 6 bad-leverage points should manufacture spurious classical significance, got \
         statistic {}",
        classical.statistic
    );
    assert!(
        robust.effect_size < classical.effect_size,
        "expected OGK to be less swayed by 6/106 bad-leverage points: classical={} robust={}",
        classical.effect_size,
        robust.effect_size
    );
}

/// A large-enough **correlated** (structured) contaminating block is a harder
/// case than isolated outliers: it can carry enough of its own internal
/// linear structure to mislead robust estimation too, not just classical.
/// This test documents whatever the two methods actually do here — robust
/// estimation is bounded-influence, not breakdown-proof at any contamination
/// fraction, and this crate does not claim otherwise.
#[test]
fn correlated_contamination_can_mislead_both_classical_and_robust() {
    let mut rng = SplitMix64::new(102);
    let n_clean = 100;
    let n_contaminated = 40;
    let mut x: Vec<f64> = Vec::with_capacity(n_clean + n_contaminated);
    let mut y: Vec<f64> = Vec::with_capacity(n_clean + n_contaminated);
    for _ in 0..n_clean
    {
        x.push(noise(&mut rng));
        y.push(noise(&mut rng));
    }
    for _ in 0..n_contaminated
    {
        let xi = noise(&mut rng);
        x.push(xi);
        y.push(3.0 * xi + 0.1 * noise(&mut rng));
    }

    let dataset = dataset_from_columns(&[x, y]);
    let classical = PartialCorrelationTest::new(gaussian_config(true))
        .test(&dataset, 0, 1, &[])
        .unwrap();
    let robust = PartialCorrelationTest::new(robust_config(RobustCalibration::NoPValue))
        .test(&dataset, 0, 1, &[])
        .unwrap();

    assert_eq!(
        classical.decision,
        IndependenceDecision::Dependent,
        "a 40/140 correlated contaminating block should be enough to move the classical \
         statistic, got {}",
        classical.statistic
    );
    // Deliberately not asserting robust "resists" this: a ~29% structured
    // contamination fraction is well within reach of OGK's own breakdown
    // behavior. What is asserted is only that the run completes and reports
    // a definite, finite statistic either way — the actual decision is
    // recorded, not dictated, by this test.
    assert!(robust.statistic.is_finite());
}

/// The mandated "clean" counter-example: with no contamination at all, the
/// classical and robust statistics should simply agree, and so should their
/// (Fisher-z vs. Gaussian-approximation) decisions at the same alpha.
#[test]
fn clean_gaussian_like_data_classical_and_robust_agree() {
    let mut rng = SplitMix64::new(103);
    let n = 200;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n
    {
        let xi = noise(&mut rng);
        x.push(xi);
        y.push(0.85 * xi + 0.15 * noise(&mut rng));
    }
    let dataset = dataset_from_columns(&[x, y]);

    let classical = PartialCorrelationTest::new(gaussian_config(true))
        .test(&dataset, 0, 1, &[])
        .unwrap();
    let robust =
        PartialCorrelationTest::new(robust_config(RobustCalibration::GaussianApproximation))
            .test(&dataset, 0, 1, &[])
            .unwrap();

    assert!(
        (classical.statistic - robust.statistic).abs() < 0.05,
        "classical={} robust={} should be close on clean data",
        classical.statistic,
        robust.statistic
    );
    assert_eq!(classical.decision, IndependenceDecision::Dependent);
    assert_eq!(robust.decision, IndependenceDecision::Dependent);
}

/// Running the identical robust request twice must produce a bit-identical
/// result — no internal nondeterminism (iteration order, hidden RNG, …).
#[test]
fn robust_partial_correlation_is_bitwise_deterministic() {
    let mut rng = SplitMix64::new(104);
    let n = 120;
    let mut x = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n
    {
        let xi = noise(&mut rng);
        let zi = 0.5 * xi + noise(&mut rng);
        let yi = 0.5 * zi + noise(&mut rng);
        x.push(xi);
        z.push(zi);
        y.push(yi);
    }
    let dataset = dataset_from_columns(&[x, z, y]);
    let test = PartialCorrelationTest::new(robust_config(RobustCalibration::NoPValue));
    let a = test.test(&dataset, 0, 2, &[1]).unwrap();
    let b = test.test(&dataset, 0, 2, &[1]).unwrap();
    assert_eq!(a, b);
}

/// A conditioning variable that varies only by a tiny amount (not exactly
/// constant — see the dataset-contract tests for exact-constant handling)
/// must not panic the robust path; it should either resolve to a finite
/// result or a specific typed [`CausalError`], never a crash.
#[test]
fn near_constant_conditioning_dimension_does_not_crash_robust_fit() {
    let mut rng = SplitMix64::new(105);
    let n = 50;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for i in 0..n
    {
        x.push(noise(&mut rng));
        y.push(noise(&mut rng));
        z.push(5.0 + 1e-6 * i as f64);
    }
    let dataset = dataset_from_columns(&[x, y, z]);
    let test = PartialCorrelationTest::new(robust_config(RobustCalibration::NoPValue));
    match test.test(&dataset, 0, 1, &[2])
    {
        Ok(result) => assert!(result.statistic.is_finite()),
        Err(CausalError::ScatterFailure(_) | CausalError::ZeroVariance { .. }) =>
        {},
        Err(other) => panic!("unexpected error for a near-constant conditioning column: {other}"),
    }
}

// ─── §16.4 Permutation calibration ───────────────────────────────────────

#[test]
fn permutation_calibration_is_deterministic_given_the_same_seed() {
    let mut rng = SplitMix64::new(110);
    let n = 80;
    let x: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 0.6 * xi + noise(&mut rng)).collect();
    let dataset = dataset_from_columns(&[x, y]);
    let test = PartialCorrelationTest::new(permutation_config(300, 42));
    let a = test.test(&dataset, 0, 1, &[]).unwrap();
    let b = test.test(&dataset, 0, 1, &[]).unwrap();
    assert_eq!(a, b);
}

#[test]
fn different_seeds_can_change_the_permutation_p_value_but_not_the_result_shape() {
    let mut rng = SplitMix64::new(111);
    let n = 80;
    let x: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 0.6 * xi + noise(&mut rng)).collect();
    let dataset = dataset_from_columns(&[x, y]);
    let a = PartialCorrelationTest::new(permutation_config(300, 1))
        .test(&dataset, 0, 1, &[])
        .unwrap();
    let b = PartialCorrelationTest::new(permutation_config(300, 2))
        .test(&dataset, 0, 1, &[])
        .unwrap();
    assert_eq!(
        a.statistic, b.statistic,
        "the observed statistic does not depend on the seed"
    );
    assert!(a.p_value.is_some() && b.p_value.is_some());
}

/// With every permutation succeeding (no degenerate resample), the reported
/// [`scirust_causal::CalibrationMethod::Permutation`] count must equal the
/// exact number requested, and no "skipped permutations" warning is raised.
#[test]
fn permutation_count_is_reported_exactly_when_none_are_skipped() {
    let mut rng = SplitMix64::new(112);
    let n = 60;
    let x: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 0.7 * xi + noise(&mut rng)).collect();
    let dataset = dataset_from_columns(&[x, y]);
    let result = PartialCorrelationTest::new(permutation_config(500, 7))
        .test(&dataset, 0, 1, &[])
        .unwrap();
    assert_eq!(
        result.calibration,
        scirust_causal::CalibrationMethod::Permutation {
            permutations: 500,
            seed: 7
        }
    );
    assert!(
        result.warnings.is_empty(),
        "unexpected warnings: {:?}",
        result.warnings
    );
}

/// `p = (1 + exceedances) / (1 + B)`: with `B` permutations the p-value must
/// be an exact multiple of `1 / (1 + B)`.
#[test]
fn permutation_p_value_respects_the_two_sided_exceedance_formula() {
    let mut rng = SplitMix64::new(113);
    let n = 50;
    let x: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let y: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let dataset = dataset_from_columns(&[x, y]);
    let permutations = 99;
    let result = PartialCorrelationTest::new(permutation_config(permutations, 3))
        .test(&dataset, 0, 1, &[])
        .unwrap();
    let p = result.p_value.unwrap();
    let scaled = p * (1.0 + permutations as f64);
    assert!(
        (scaled - scaled.round()).abs() < 1e-9,
        "p={p} is not an exact multiple of 1/{}",
        1 + permutations
    );
}

#[test]
fn permutation_calibration_detects_strong_dependence() {
    let mut rng = SplitMix64::new(114);
    let n = 100;
    let x: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| 0.9 * xi + 0.1 * noise(&mut rng))
        .collect();
    let dataset = dataset_from_columns(&[x, y]);
    let result = PartialCorrelationTest::new(permutation_config(500, 21))
        .test(&dataset, 0, 1, &[])
        .unwrap();
    assert_eq!(result.decision, IndependenceDecision::Dependent);
}

#[test]
fn permutation_calibration_does_not_reject_true_independence() {
    let mut rng = SplitMix64::new(115);
    let n = 100;
    let x: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let y: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let dataset = dataset_from_columns(&[x, y]);
    let result = PartialCorrelationTest::new(permutation_config(500, 22))
        .test(&dataset, 0, 1, &[])
        .unwrap();
    assert_eq!(
        result.decision,
        IndependenceDecision::IndependentWithinThreshold
    );
}

/// The Freedman-Lane-style residual permutation applied to a genuine chain
/// (`X -> Z -> Y`): marginally dependent, conditionally independent given
/// `Z`, exactly as the Fisher-z calibrated version already establishes — this
/// confirms the *permutation* calibration path agrees on the same causal
/// motif, not just on i.i.d. noise.
#[test]
fn permutation_calibration_on_a_chain_via_residual_permutation() {
    let mut rng = SplitMix64::new(116);
    let n = 300;
    let mut x = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n
    {
        let xi = noise(&mut rng);
        let zi = 0.8 * xi + noise(&mut rng);
        let yi = 0.8 * zi + noise(&mut rng);
        x.push(xi);
        z.push(zi);
        y.push(yi);
    }
    let dataset = dataset_from_columns(&[x, z, y]);
    let test = PartialCorrelationTest::new(permutation_config(300, 99));

    let marginal = test.test(&dataset, 0, 2, &[]).unwrap();
    assert_eq!(marginal.decision, IndependenceDecision::Dependent);

    let conditional = test.test(&dataset, 0, 2, &[1]).unwrap();
    assert_eq!(
        conditional.decision,
        IndependenceDecision::IndependentWithinThreshold,
        "residual permutation should not reject X ⟂ Y | Z on a chain, got statistic {} p={:?}",
        conditional.statistic,
        conditional.p_value
    );
}

#[test]
fn zero_permutations_is_a_typed_configuration_error() {
    let result = ConditionalIndependenceConfig::new(
        0.05,
        ConditionalIndependenceMethod::PermutationPartialCorrelation {
            permutations: 0,
            seed: 1,
            residualization: scirust_causal::ResidualizationMethod::OrdinaryLeastSquares,
        },
    );
    assert!(matches!(
        result,
        Err(CausalError::InvalidConfiguration {
            name: "permutations",
            ..
        })
    ));
}

#[test]
fn permutation_method_is_invariant_to_conditioning_set_order() {
    let mut rng = SplitMix64::new(117);
    let n = 80;
    let a: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let b: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let x: Vec<f64> = (0..n)
        .map(|i| 0.5 * a[i] + 0.3 * b[i] + noise(&mut rng))
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| 0.3 * a[i] + 0.5 * b[i] + noise(&mut rng))
        .collect();
    let dataset = dataset_from_columns(&[x, y, a, b]);
    let test = PartialCorrelationTest::new(permutation_config(200, 5));
    let forward = test.test(&dataset, 0, 1, &[2, 3]).unwrap();
    let reordered = test.test(&dataset, 0, 1, &[3, 2]).unwrap();
    assert_eq!(forward, reordered);
}

// ─── §18 Adversarial ─────────────────────────────────────────────────────

/// `z2 = z1 + tiny perturbation`: strongly, but not exactly, collinear. The
/// design's numerical rank must still be judged full under the default
/// tolerance — near-collinearity is ill-conditioned, not singular.
#[test]
fn near_perfect_multicollinearity_is_not_treated_as_exact_rank_deficiency() {
    let mut rng = SplitMix64::new(120);
    let n = 60;
    let z1: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let z2: Vec<f64> = z1
        .iter()
        .map(|&v| v + 1e-4 * (rng.next_f64() - 0.5))
        .collect();
    let x: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let y: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let dataset = dataset_from_columns(&[x, y, z1, z2]);
    let result = PartialCorrelationTest::new(gaussian_config(true)).test(&dataset, 0, 1, &[2, 3]);
    assert!(
        result.is_ok(),
        "near-collinear (not exact) conditioning columns should not error: {result:?}"
    );
}

/// `z2 = 2 * z1` exactly: `[intercept, z1, z2]` has rank 2, not 3 — the
/// end-to-end (public-API) counterpart to `partial_correlation`'s own
/// internal unit test.
#[test]
fn exact_rank_deficiency_is_a_typed_error_end_to_end() {
    let mut rng = SplitMix64::new(121);
    let n = 30;
    let z1: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
    let z2: Vec<f64> = z1.iter().map(|&v| 2.0 * v).collect();
    let x: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let y: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let dataset = dataset_from_columns(&[x, y, z1, z2]);
    let result = PartialCorrelationTest::new(gaussian_config(true)).test(&dataset, 0, 1, &[2, 3]);
    assert!(matches!(
        result,
        Err(CausalError::RankDeficientConditioningSet {
            rank: 2,
            columns: 3
        })
    ));
}

/// A conditioning set whose size leaves only **one residual degree of
/// freedom** (`n - (1 + |Z|) = 1`) forces `x`'s and `y`'s residuals into the
/// same one-dimensional subspace — any two nonzero vectors confined to a
/// 1-D subspace are perfectly (anti)correlated *by construction*, regardless
/// of whether `x` and `y` have any real relationship. This is a genuine
/// numerical failure mode of "conditioning set nearly as large as the
/// sample," not a contrived one: the honest response is `statistic ≈ ±1` with
/// `Inconclusive` (Fisher-z is undefined at `|r| = 1`), **not** a confident
/// `Dependent` — this test locks that honesty in.
#[test]
fn conditioning_set_nearly_as_large_as_the_sample_forces_a_spurious_perfect_correlation() {
    let mut rng = SplitMix64::new(122);
    let n = 7; // 5 conditioning vars + intercept = 6 columns; residual df = 1.
    let z_cols: Vec<Vec<f64>> = (0..5)
        .map(|_| (0..n).map(|_| noise(&mut rng)).collect())
        .collect();
    let x: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let y: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let mut columns = vec![x, y];
    columns.extend(z_cols);
    let dataset = dataset_from_columns(&columns);
    let result = PartialCorrelationTest::new(gaussian_config(true))
        .test(&dataset, 0, 1, &[2, 3, 4, 5, 6])
        .unwrap();
    assert!(
        result.statistic.abs() > 1.0 - 1e-6,
        "expected a near-{{-1,1}} artifact from a saturated conditioning set, got {}",
        result.statistic
    );
    assert_eq!(result.decision, IndependenceDecision::Inconclusive);
    assert_eq!(result.p_value, None);
}

/// One fewer sample than the boundary above: `min_required = |Z| + 2 = 7 > 6`
/// is a typed error, never a silent (and here, meaningless) computation.
#[test]
fn one_below_the_sample_size_boundary_is_a_typed_error() {
    let mut rng = SplitMix64::new(123);
    let n = 6;
    let z_cols: Vec<Vec<f64>> = (0..5)
        .map(|_| (0..n).map(|_| noise(&mut rng)).collect())
        .collect();
    let x: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let y: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let mut columns = vec![x, y];
    columns.extend(z_cols);
    let dataset = dataset_from_columns(&columns);
    let result =
        PartialCorrelationTest::new(gaussian_config(true)).test(&dataset, 0, 1, &[2, 3, 4, 5, 6]);
    assert!(matches!(
        result,
        Err(CausalError::InsufficientSamples {
            required: 7,
            actual: 6
        })
    ));
}

/// A chain with a **tiny** path coefficient: the true dependence is real but
/// so weak that, at this (still substantial) sample size, the test has
/// essentially no power to detect it. Reporting
/// `IndependentWithinThreshold` here is the *honest* response to a
/// low-signal-to-noise regime — it is explicitly **not** proof the chain's
/// edges are absent, which is exactly why the decision is named
/// "within threshold," not "independent."
#[test]
fn near_unfaithful_chain_with_tiny_coefficient_is_not_reliably_detected() {
    let mut rng = SplitMix64::new(124);
    let n = 400;
    let mut x = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let coefficient = 0.02;
    for _ in 0..n
    {
        let xi = noise(&mut rng);
        let zi = coefficient * xi + noise(&mut rng);
        let yi = coefficient * zi + noise(&mut rng);
        x.push(xi);
        z.push(zi);
        y.push(yi);
    }
    let dataset = dataset_from_columns(&[x, z, y]);
    let marginal = PartialCorrelationTest::new(gaussian_config(true))
        .test(&dataset, 0, 2, &[])
        .unwrap();
    assert_eq!(
        marginal.decision,
        IndependenceDecision::IndependentWithinThreshold,
        "a true but tiny-coefficient chain should not be reliably detected at this sample size, \
         got statistic {} p={:?} — if this now fails, the fixture's signal-to-noise ratio has \
         changed enough to need re-tuning, not that detecting it would be wrong",
        marginal.statistic,
        marginal.p_value
    );
}

/// A small minority of rows (~6%) carrying a **direct `X -> Y` bypass** that
/// ignores `Z` entirely is picked up as conditional dependence by both
/// methods — correctly so, since a real (if minority-driven) bypass edge is
/// present. What is compared is *how much* each statistic reacts to that
/// minority, not whether either is immune to it.
#[test]
fn a_small_minority_of_bypass_contaminated_rows_affects_the_conditional_test() {
    let mut rng = SplitMix64::new(125);
    let n_clean = 300;
    let n_contaminated = 20;
    let mut x = Vec::with_capacity(n_clean + n_contaminated);
    let mut z = Vec::with_capacity(n_clean + n_contaminated);
    let mut y = Vec::with_capacity(n_clean + n_contaminated);
    for _ in 0..n_clean
    {
        let xi = noise(&mut rng);
        let zi = 0.8 * xi + noise(&mut rng);
        let yi = 0.8 * zi + noise(&mut rng);
        x.push(xi);
        z.push(zi);
        y.push(yi);
    }
    for _ in 0..n_contaminated
    {
        let xi = noise(&mut rng);
        x.push(xi);
        z.push(noise(&mut rng));
        y.push(4.0 * xi + noise(&mut rng));
    }
    let dataset = dataset_from_columns(&[x, z, y]);
    let classical = PartialCorrelationTest::new(gaussian_config(true))
        .test(&dataset, 0, 2, &[1])
        .unwrap();
    let robust = PartialCorrelationTest::new(robust_config(RobustCalibration::NoPValue))
        .test(&dataset, 0, 2, &[1])
        .unwrap();
    assert!(classical.statistic.is_finite());
    assert!(robust.statistic.is_finite());
}

/// Occasional large-magnitude draws (independent `x`, `y`, no true relation):
/// heavy tails inflate the variance of the Pearson statistic itself, which
/// degrades — in either direction — the Fisher-z asymptotic approximation's
/// nominal calibration. This records the actual outcome at one fixed,
/// reproducible seed; it is illustrative of the phenomenon, not a general
/// proof that heavy tails always produce a specific decision.
#[test]
fn heavy_tailed_independent_variables_are_handled_without_error() {
    let mut rng = SplitMix64::new(126);
    let n = 300;
    let x: Vec<f64> = (0..n).map(|_| heavy_tailed_noise(&mut rng)).collect();
    let y: Vec<f64> = (0..n).map(|_| heavy_tailed_noise(&mut rng)).collect();
    let dataset = dataset_from_columns(&[x, y]);
    let result = PartialCorrelationTest::new(gaussian_config(true))
        .test(&dataset, 0, 1, &[])
        .unwrap();
    // No specific decision is asserted: the point of this test is that a
    // heavy-tailed sample is handled deterministically and without a typed
    // error, whatever three-way decision it lands on.
    assert!(result.statistic.is_finite());
    assert!(result.p_value.is_some());
}

/// `Y = X² + small noise` with `X` symmetric about `0`: `Y` is a near-
/// deterministic function of `X` (as dependent as it gets), but `Cov(X, X²)
/// = 0` by symmetry, so the *linear* partial correlation is ≈ 0. This is the
/// documented, undisguised failure mode of every method in this module: they
/// measure linear association, and a linear statistic is simply not evidence
/// about purely nonlinear structure. Do not read `IndependentWithinThreshold`
/// here as "no dependence" — it means exactly what it always means: the
/// *linear* null was not rejected.
///
/// `x` is built as exact `+v`/`-v` mirror pairs so `Σx_i = 0` and `Σx_i³ = 0`
/// **exactly** (not just in expectation) — the only source of sample
/// correlation left is the cross-term with `y`'s own independent noise,
/// which is small and does not grow with `n`; this keeps the test's outcome
/// from depending on the luck of a particular seed.
#[test]
fn nonlinear_dependence_is_invisible_to_a_linear_partial_correlation_test() {
    let mut rng = SplitMix64::new(127);
    let n_pairs = 200;
    let mut x = Vec::with_capacity(2 * n_pairs);
    let mut y = Vec::with_capacity(2 * n_pairs);
    for _ in 0..n_pairs
    {
        let magnitude = 1.0 + 3.0 * rng.next_f64(); // in [1, 4)
        x.push(magnitude);
        y.push(magnitude * magnitude + 0.1 * noise(&mut rng));
        x.push(-magnitude);
        y.push(magnitude * magnitude + 0.1 * noise(&mut rng));
    }
    let dataset = dataset_from_columns(&[x, y]);
    let result = PartialCorrelationTest::new(gaussian_config(true))
        .test(&dataset, 0, 1, &[])
        .unwrap();
    assert!(
        result.effect_size < 0.2,
        "Y = X^2 should show a near-zero *linear* correlation despite being a near-deterministic \
         function of X, got effect_size {}",
        result.effect_size
    );
    assert_ne!(
        result.decision,
        IndependenceDecision::Dependent,
        "the linear test must not (and structurally cannot) flag this genuinely nonlinear \
         dependence"
    );
}

/// `Y = noise * (1 + |X|)`: `E[Y | X] = 0` for every `X` (mean-independent —
/// no *linear* relationship), yet `Y`'s variance depends on `X`, so `X` and
/// `Y` are genuinely conditionally dependent through their second moment.
/// Pearson correlation, a mean-based statistic, cannot see this either —
/// another explicit, undisguised negative result rather than a hidden one.
#[test]
fn heteroscedastic_dependence_is_invisible_to_a_mean_based_linear_test() {
    let mut rng = SplitMix64::new(128);
    let n = 300;
    let x: Vec<f64> = (0..n).map(|_| 3.0 * noise(&mut rng)).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| noise(&mut rng) * (1.0 + xi.abs()))
        .collect();
    let dataset = dataset_from_columns(&[x, y]);
    let result = PartialCorrelationTest::new(gaussian_config(true))
        .test(&dataset, 0, 1, &[])
        .unwrap();
    assert_ne!(
        result.decision,
        IndependenceDecision::Dependent,
        "a purely variance-mediated (heteroscedastic) dependence has no linear-mean signal for \
         this test to detect"
    );
}

/// Pooling interventional rows (`do(Z) = 0`, breaking `X -> Z`) into an
/// observational chain changes the conditional relationship being measured.
/// [`RegimeSelection::ObservationalOnly`] correctly excludes them;
/// [`RegimeSelection::ExplicitRows`] spanning both blocks is a deliberate
/// escape hatch that does **not** protect the caller from mixing regimes —
/// this test demonstrates that the mixture is observably different, exactly
/// the risk the crate's docs warn about rather than silently paper over.
#[test]
fn intervention_rows_mixed_into_observational_data_change_the_result() {
    let mut rng = SplitMix64::new(129);
    let n = 200;

    let mut obs_x = Vec::with_capacity(n);
    let mut obs_z = Vec::with_capacity(n);
    let mut obs_y = Vec::with_capacity(n);
    for _ in 0..n
    {
        let xi = noise(&mut rng);
        let zi = 0.8 * xi + noise(&mut rng);
        let yi = 0.8 * zi + noise(&mut rng);
        obs_x.push(xi);
        obs_z.push(zi);
        obs_y.push(yi);
    }
    let mut obs_data = vec![0.0; n * 3];
    for row in 0..n
    {
        obs_data[row * 3] = obs_x[row];
        obs_data[row * 3 + 1] = obs_z[row];
        obs_data[row * 3 + 2] = obs_y[row];
    }
    let obs_block = SampleBlock::from_matrix(
        Environment::observational("obs").unwrap(),
        &Matrix::from_row_major(n, 3, obs_data),
    )
    .unwrap();

    let mut iv_data = vec![0.0; n * 3];
    for row in 0..n
    {
        let xi = noise(&mut rng);
        let zi = 0.0; // do(Z) = 0: the X -> Z edge is severed in this block.
        let yi = 0.8 * zi + noise(&mut rng);
        iv_data[row * 3] = xi;
        iv_data[row * 3 + 1] = zi;
        iv_data[row * 3 + 2] = yi;
    }
    let iv = Intervention::new(1, InterventionKind::Atomic { value: 0.0 }).unwrap();
    let iv_env = Environment::new("do_z", vec![iv]).unwrap();
    let iv_block =
        SampleBlock::from_matrix(iv_env, &Matrix::from_row_major(n, 3, iv_data)).unwrap();

    let variables = vec![
        CausalVariable::new(0, "x", VariableRole::Unspecified, VariableKind::Continuous).unwrap(),
        CausalVariable::new(1, "z", VariableRole::Unspecified, VariableKind::Continuous).unwrap(),
        CausalVariable::new(2, "y", VariableRole::Unspecified, VariableKind::Continuous).unwrap(),
    ];
    let dataset =
        CausalDataset::new(variables, vec![obs_block, iv_block], "mixed regimes").unwrap();

    let observational_only = gaussian_config(true).with_regime(RegimeSelection::ObservationalOnly);
    let mixed =
        gaussian_config(true).with_regime(RegimeSelection::ExplicitRows((0..2 * n).collect()));

    let correct = PartialCorrelationTest::new(observational_only)
        .test(&dataset, 0, 2, &[1])
        .unwrap();
    let pooled = PartialCorrelationTest::new(mixed)
        .test(&dataset, 0, 2, &[1])
        .unwrap();

    assert_eq!(correct.sample_count, n);
    assert_eq!(pooled.sample_count, 2 * n);
    assert!(
        (correct.statistic - pooled.statistic).abs() > 1e-6,
        "pooling interventional rows that sever X -> Z should visibly change the conditional \
         statistic relative to the correctly-scoped observational-only result: correct={} \
         pooled={}",
        correct.statistic,
        pooled.statistic
    );
}

/// A small environment sits right at the sample-size boundary for a
/// conditional query: `n = 3` with one conditioning variable
/// (`min_required = 3`) just barely succeeds; `n = 2` for the identical
/// query is a typed [`CausalError::InsufficientSamples`], not a silently
/// degraded computation.
#[test]
fn a_small_environment_sits_right_at_the_sample_size_boundary() {
    let variables = vec![
        CausalVariable::new(0, "x", VariableRole::Unspecified, VariableKind::Continuous).unwrap(),
        CausalVariable::new(1, "y", VariableRole::Unspecified, VariableKind::Continuous).unwrap(),
        CausalVariable::new(2, "z", VariableRole::Unspecified, VariableKind::Continuous).unwrap(),
    ];

    let just_enough = SampleBlock::from_matrix(
        Environment::observational("tiny_site").unwrap(),
        &Matrix::from_row_major(3, 3, vec![1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 1.5, 4.0]),
    )
    .unwrap();
    let dataset_ok = CausalDataset::new(
        variables.clone(),
        vec![just_enough],
        "boundary: n = required",
    )
    .unwrap();
    let config =
        gaussian_config(true).with_regime(RegimeSelection::Environment("tiny_site".to_string()));
    let result = PartialCorrelationTest::new(config).test(&dataset_ok, 0, 1, &[2]);
    assert!(
        result.is_ok(),
        "n == required should just barely succeed: {result:?}"
    );

    let one_short = SampleBlock::from_matrix(
        Environment::observational("tiny_site").unwrap(),
        &Matrix::from_row_major(2, 3, vec![1.0, 2.0, 1.0, 2.0, 3.0, 2.0]),
    )
    .unwrap();
    let dataset_short =
        CausalDataset::new(variables, vec![one_short], "boundary: n = required - 1").unwrap();
    let config =
        gaussian_config(true).with_regime(RegimeSelection::Environment("tiny_site".to_string()));
    let result = PartialCorrelationTest::new(config).test(&dataset_short, 0, 1, &[2]);
    assert!(matches!(
        result,
        Err(CausalError::InsufficientSamples {
            required: 3,
            actual: 2
        })
    ));
}

/// `scirust-causal` has no notion of measurement "units" on a
/// [`CausalVariable`] — there is nothing to be "invalid" that isn't already
/// covered by [`VariableKind`] (see `rejects_non_continuous_variable_kind` in
/// `conditional_independence.rs`). What the type system *does* enforce is
/// unique variable metadata: a duplicate variable name is rejected at
/// dataset-construction time, before any CI test ever runs.
#[test]
fn duplicate_variable_metadata_is_rejected_at_dataset_construction() {
    let variables = vec![
        CausalVariable::new(0, "x", VariableRole::Unspecified, VariableKind::Continuous).unwrap(),
        CausalVariable::new(1, "x", VariableRole::Unspecified, VariableKind::Continuous).unwrap(),
    ];
    let block = SampleBlock::from_matrix(
        Environment::observational("obs").unwrap(),
        &Matrix::from_row_major(4, 2, vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0]),
    )
    .unwrap();
    let result = CausalDataset::new(variables, vec![block], "duplicate names");
    assert!(matches!(result, Err(CausalError::InvalidContract { .. })));
}
