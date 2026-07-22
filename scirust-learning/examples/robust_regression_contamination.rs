//! Deterministic contamination benchmark for robust regression.
//!
//! One fixed affine ground truth (`y = 5 + 2·x₀ − 3·x₁` over a 60-row design
//! with deterministic normal noise), replayed under increasing front-loaded
//! target contamination (a gross `+100` shift). Four estimators are compared
//! at every level:
//!
//! - ordinary least squares (0 % breakdown baseline);
//! - Huber IRLS (`δ = 1.345`);
//! - trimmed least squares (`h = 0.7`, tolerates at most 30 % contamination);
//! - median-of-means (5 seeded blocks: the guarantee needs the outliers to
//!   touch fewer than half the blocks, so it degrades once the contamination
//!   count exceeds two blocks' worth).
//!
//! Reported per (level, method): parameter error (L2 over coefficients and
//! intercept), RMSE and median absolute error of the fitted model against the
//! *clean* targets, worst-case clean-target error, effective sample count,
//! iterations, and convergence. **Unfavourable regimes are part of the
//! output**: OLS degrades from the first outlier; the trimmed fit breaks past
//! 30 %; median-of-means breaks once a majority of blocks are touched. No
//! estimator here resists arbitrary majority corruption.
//!
//! Output is deterministic CSV on stdout (`{:.17e}`); run twice and compare
//! byte-for-byte (`cmp` / SHA-256). No timestamps, no timings.

use scirust_learning::{
    RegressionDataset, RobustLoss, RobustRegressionConfig, RobustRegressionError,
    RobustRegressionMethod, fit_robust_regression,
};
use scirust_solvers::linalg::Matrix;
use scirust_stats::{Distribution, Normal, SplitMix64};

const SAMPLE_COUNT: usize = 60;
const NOISE_SEED: u64 = 0x5EED_0724;
const NOISE_SCALE: f64 = 0.05;
const OUTLIER_SHIFT: f64 = 100.0;
const TRUE_COEFFICIENTS: [f64; 2] = [2.0, -3.0];
const TRUE_INTERCEPT: f64 = 5.0;
const CONTAMINATION_LEVELS: [f64; 6] = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4];

/// Fixed design plus clean targets with deterministic normal noise.
fn base_dataset() -> (Matrix, Vec<f64>) {
    let standard = Normal::standard();
    let mut rng = SplitMix64::new(NOISE_SEED);
    let mut rows = Vec::with_capacity(SAMPLE_COUNT * 2);
    let mut clean = Vec::with_capacity(SAMPLE_COUNT);

    for i in 0..SAMPLE_COUNT
    {
        let x0 = (i % 6) as f64 + 0.25 * ((i % 7) as f64);
        let x1 = ((i / 6) as f64) * 0.5 - 2.0;
        let u = 1.0e-6 + rng.next_f64() * (1.0 - 2.0e-6);
        let noise = NOISE_SCALE * standard.quantile(u);

        rows.push(x0);
        rows.push(x1);
        clean.push(TRUE_INTERCEPT + TRUE_COEFFICIENTS[0] * x0 + TRUE_COEFFICIENTS[1] * x1 + noise);
    }

    (Matrix::from_row_major(SAMPLE_COUNT, 2, rows), clean)
}

fn method_configs() -> Vec<(&'static str, RobustRegressionConfig)> {
    let base = RobustRegressionConfig::default();

    vec![
        (
            "ols",
            RobustRegressionConfig {
                method: RobustRegressionMethod::OrdinaryLeastSquares,
                ..base
            },
        ),
        (
            "huber_irls",
            RobustRegressionConfig {
                method: RobustRegressionMethod::IterativelyReweightedLeastSquares,
                loss: RobustLoss::Huber { delta: 1.345 },
                ..base
            },
        ),
        (
            "trimmed_0_7",
            RobustRegressionConfig {
                method: RobustRegressionMethod::TrimmedLeastSquares {
                    retained_fraction: 0.7,
                },
                ..base
            },
        ),
        (
            "median_of_means_5",
            RobustRegressionConfig {
                method: RobustRegressionMethod::MedianOfMeans {
                    block_count: 5,
                    seed: 0x0B10_C55,
                },
                ..base
            },
        ),
    ]
}

fn main() -> Result<(), RobustRegressionError> {
    let (features, clean_targets) = base_dataset();

    println!("# robust_regression_contamination deterministic benchmark");
    println!(
        "# n={SAMPLE_COUNT} noise_scale={NOISE_SCALE} outlier_shift={OUTLIER_SHIFT} \
truth=[{},{}]+{}",
        TRUE_COEFFICIENTS[0], TRUE_COEFFICIENTS[1], TRUE_INTERCEPT
    );
    println!(
        "# columns: fraction,outliers,method,parameter_error,clean_rmse,clean_median_abs_error,\
clean_worst_error,effective_samples,iterations,converged"
    );

    for &fraction in &CONTAMINATION_LEVELS
    {
        let outliers = ((SAMPLE_COUNT as f64) * fraction).floor() as usize;

        let mut targets = clean_targets.clone();

        for target in targets.iter_mut().take(outliers)
        {
            *target += OUTLIER_SHIFT;
        }

        let dataset = RegressionDataset {
            features: features.clone(),
            targets: Matrix::from_row_major(SAMPLE_COUNT, 1, targets),
            sample_weights: None,
        };

        for (name, configuration) in method_configs()
        {
            let report = fit_robust_regression(&dataset, configuration)?;

            let parameter_error = ((report.model.coefficients[(0, 0)] - TRUE_COEFFICIENTS[0])
                .powi(2)
                + (report.model.coefficients[(1, 0)] - TRUE_COEFFICIENTS[1]).powi(2)
                + (report.model.intercept[0] - TRUE_INTERCEPT).powi(2))
            .sqrt();

            let predictions = report.model.predict(&features)?;

            let mut errors: Vec<f64> = (0..SAMPLE_COUNT)
                .map(|row| (predictions[(row, 0)] - clean_targets[row]).abs())
                .collect();

            let rmse = (errors.iter().map(|e| e * e).sum::<f64>() / SAMPLE_COUNT as f64).sqrt();

            errors.sort_by(f64::total_cmp);

            let median_abs = (errors[SAMPLE_COUNT / 2 - 1] + errors[SAMPLE_COUNT / 2]) / 2.0;
            let worst = errors[SAMPLE_COUNT - 1];

            println!(
                "{fraction:.17e},{outliers},{name},{parameter_error:.17e},{rmse:.17e},\
{median_abs:.17e},{worst:.17e},{},{},{}",
                report.effective_sample_count, report.iterations, report.converged,
            );
        }
    }

    Ok(())
}
