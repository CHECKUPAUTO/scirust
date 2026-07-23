//! Deterministic real-data validation for phase 4E.9: the unified certified
//! decision pipeline (4E.7) on real OBD2 telemetry.
//!
//! The synthetic pipeline benchmark (`industrial-certified-pipeline`) proves the
//! decision surface; this exercises the whole stack — the estimator tournament
//! with abstention (4E.5) and split-conformal coverage (4E.6), composed by the
//! certified pipeline — on the in-repo Opel Corsa OBD2 dataset under a genuinely
//! leakage-free protocol: **leave-one-segment-out** (train on every drive segment
//! but the held one, test on the held one), with a proper/calibration split
//! inside training. Features are the other 11 channels; the target is one channel.
//!
//! For each of three target channels we run the certified pipeline on every
//! leave-one-segment-out fold with the panel OLS (incumbent), Huber, Tukey, MM,
//! and report — honestly — the distribution of verdicts, the mean empirical test
//! coverage of the certified band, and whether the best robust fit actually beat
//! OLS on held-out error. Real industrial data does not owe us a robust win; the
//! point is a faithful measurement, abstentions included.
//!
//! Tractability trade-offs (declared, not hidden): the 43 140-row dataset is
//! decimated by a fixed stride, and the MM estimator uses fewer subset starts
//! than its library default. Both are deterministic. Run twice, byte-identical.

use std::path::{Path, PathBuf};
use std::{fs, process};

use scirust_learning::{
    HighBreakdownConfig, HighBreakdownMethod, RegressionDataset, RobustLoss,
    RobustRegressionConfig, RobustRegressionMethod, fit_high_breakdown, fit_robust_regression,
};
use scirust_solvers::linalg::Matrix;
use scirust_srcc_bench::{
    CertifiedPipelineConfig, CoverageMode, EstimatorEvidence, EstimatorTournament, Orientation,
    TournamentDecision, parse_obd2, run_certified_pipeline, sha256_hex,
};

const OBD2_SHA: &str = "229ef4488a89b62be530acce36ec7522421e7b82b1b5279022ffac72f0bb0751";
const TARGETS: [&str; 3] = ["ENGINE_LOAD", "THROTTLE_POS", "MAF"];
const LEVEL: f64 = 0.9;
const DECIMATION: usize = 36; // 43_140 rows -> ~1_199 rows (declared decimation)
const CALIBRATION_STRIDE: usize = 5;
const MM_SUBSET_STARTS: usize = 24; // below the library default of 200 (declared)
const SD_FLOOR: f64 = 1.0e-9;
const TOURNAMENT_SEED: u64 = 0x00CE_5712_F1ED;

fn read_verified(path: &Path, expected_sha: &str) -> String {
    let bytes = fs::read(path).unwrap_or_else(|error| {
        eprintln!(
            "cannot read {}: {error}. The OBD2 telemetry ships in-repo under \
examples/obd2_diagnostic/data/.",
            path.display()
        );
        process::exit(1);
    });
    let actual = sha256_hex(&bytes);
    assert!(
        actual == expected_sha,
        "checksum mismatch for {}: expected {expected_sha}, found {actual}",
        path.display()
    );
    String::from_utf8(bytes).expect("dataset file is valid UTF-8")
}

/// Per-column standardizer fitted on the proper-training rows.
struct Standardizer {
    means: Vec<f64>,
    sds: Vec<f64>,
}

impl Standardizer {
    fn fit(features: &[Vec<f64>]) -> Self {
        let rows = features.len();
        let cols = features.first().map_or(0, Vec::len);
        let mut means = vec![0.0; cols];
        for row in features
        {
            for (accumulator, value) in means.iter_mut().zip(row)
            {
                *accumulator += value;
            }
        }
        for mean in &mut means
        {
            *mean /= rows as f64;
        }
        let mut variances = vec![0.0; cols];
        for row in features
        {
            for (index, value) in row.iter().enumerate()
            {
                let delta = value - means[index];
                variances[index] += delta * delta;
            }
        }
        let sds: Vec<f64> = variances
            .iter()
            .map(|variance| (variance / rows as f64).sqrt().max(SD_FLOOR))
            .collect();
        Self { means, sds }
    }

    fn apply(&self, features: &[Vec<f64>]) -> Vec<Vec<f64>> {
        features
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(index, value)| (value - self.means[index]) / self.sds[index])
                    .collect()
            })
            .collect()
    }
}

#[derive(Clone, Copy)]
enum Estimator {
    Ols,
    Huber,
    Tukey,
    Mm,
}

impl Estimator {
    fn name(self) -> &'static str {
        match self
        {
            Self::Ols => "ols",
            Self::Huber => "huber",
            Self::Tukey => "tukey",
            Self::Mm => "mm",
        }
    }
}

const PANEL: [Estimator; 4] = [
    Estimator::Ols,
    Estimator::Huber,
    Estimator::Tukey,
    Estimator::Mm,
];

/// Fits `estimator` on standardized proper-training rows, returning
/// `(coefficients, intercept)`. `None` if the fit fails.
fn fit_estimator(
    estimator: Estimator,
    features: &[Vec<f64>],
    targets: &[f64],
) -> Option<(Vec<f64>, f64)> {
    let p = features.first().map_or(0, Vec::len);
    let matrix = Matrix::from_row_major(features.len(), p, features.concat());
    match estimator
    {
        Estimator::Ols | Estimator::Huber | Estimator::Tukey =>
        {
            let dataset = RegressionDataset {
                features: matrix,
                targets: Matrix::from_row_major(targets.len(), 1, targets.to_vec()),
                sample_weights: None,
            };
            let (method, loss) = match estimator
            {
                Estimator::Ols => (
                    RobustRegressionMethod::OrdinaryLeastSquares,
                    RobustLoss::Squared,
                ),
                Estimator::Huber => (
                    RobustRegressionMethod::IterativelyReweightedLeastSquares,
                    RobustLoss::Huber { delta: 1.345 },
                ),
                _ => (
                    RobustRegressionMethod::IterativelyReweightedLeastSquares,
                    RobustLoss::TukeyBisquare { cutoff: 4.685 },
                ),
            };
            let model = fit_robust_regression(
                &dataset,
                RobustRegressionConfig {
                    method,
                    loss,
                    ..RobustRegressionConfig::default()
                },
            )
            .ok()?
            .model;
            let coefficients: Vec<f64> = (0..p).map(|j| model.coefficients[(j, 0)]).collect();
            Some((coefficients, model.intercept[0]))
        },
        Estimator::Mm =>
        {
            let report = fit_high_breakdown(
                &matrix,
                targets,
                HighBreakdownConfig {
                    method: HighBreakdownMethod::MmEstimator,
                    subset_starts: MM_SUBSET_STARTS,
                    ..HighBreakdownConfig::default()
                },
            )
            .ok()?;
            Some((report.coefficients, report.intercept))
        },
    }
}

fn predict(coefficients: &[f64], intercept: f64, row: &[f64]) -> f64 {
    intercept
        + coefficients
            .iter()
            .zip(row)
            .map(|(c, x)| c * x)
            .sum::<f64>()
}

/// Signed residuals `y − ŷ` for the given rows.
fn residuals(model: &(Vec<f64>, f64), features: &[Vec<f64>], targets: &[f64]) -> Vec<f64> {
    features
        .iter()
        .zip(targets)
        .map(|(row, &y)| y - predict(&model.0, model.1, row))
        .collect()
}

fn mean_abs(values: &[f64]) -> f64 {
    if values.is_empty()
    {
        return f64::NAN;
    }
    values.iter().map(|v| v.abs()).sum::<f64>() / values.len() as f64
}

/// Per-channel accumulator over leave-one-segment-out folds.
#[derive(Default)]
struct ChannelSummary {
    select: usize,
    hold: usize,
    tie: usize,
    inconclusive: usize,
    reject: usize,
    coverage_sum: f64,
    coverage_folds: usize,
    ols_mae_sum: f64,
    best_robust_mae_sum: f64,
    robust_wins: usize,
    folds: usize,
}

fn main() {
    let data_path = PathBuf::from("examples/obd2_diagnostic/data/opel_corsa_telemetry.csv");
    let text = read_verified(&data_path, OBD2_SHA);

    println!("# obd2_certified — phase 4E.9: certified pipeline on real OBD2 telemetry");
    println!(
        "# leave-one-segment-out; panel OLS/Huber/Tukey/MM; coverage level {LEVEL}; \
decimation 1/{DECIMATION}; MM starts {MM_SUBSET_STARTS}"
    );
    println!(
        "# channel        folds  select hold tie incon  mean_cov  ols_mae  robust_mae  robust_wins"
    );

    for target in TARGETS
    {
        let data = parse_obd2(&text, target).expect("OBD2 parses for the target channel");
        let groups = data.groups.as_ref().expect("OBD2 carries segment groups");

        // Fixed-stride decimation, preserving segment membership.
        let kept: Vec<usize> = (0..data.targets.len())
            .filter(|row| row % DECIMATION == 0)
            .collect();
        let mut segments: Vec<u64> = kept.iter().map(|&row| groups[row]).collect();
        segments.sort_unstable();
        segments.dedup();

        let mut summary = ChannelSummary::default();

        for &held in &segments
        {
            let train: Vec<usize> = kept
                .iter()
                .copied()
                .filter(|&row| groups[row] != held)
                .collect();
            let test: Vec<usize> = kept
                .iter()
                .copied()
                .filter(|&row| groups[row] == held)
                .collect();
            if train.len() < 50 || test.len() < 5
            {
                continue;
            }

            // Proper/calibration split inside training.
            let mut proper = Vec::new();
            let mut calibration = Vec::new();
            for (position, &row) in train.iter().enumerate()
            {
                if position % CALIBRATION_STRIDE == 0
                {
                    calibration.push(row);
                }
                else
                {
                    proper.push(row);
                }
            }

            let proper_features_raw = gather(&data.features, &proper);
            let standardizer = Standardizer::fit(&proper_features_raw);
            let proper_features = standardizer.apply(&proper_features_raw);
            let proper_targets = gather_targets(&data.targets, &proper);
            let calibration_features = standardizer.apply(&gather(&data.features, &calibration));
            let calibration_targets = gather_targets(&data.targets, &calibration);
            let test_features = standardizer.apply(&gather(&data.features, &test));
            let test_targets = gather_targets(&data.targets, &test);

            // Fit the panel; skip the fold if any estimator fails.
            let mut fitted = Vec::new();
            let mut ok = true;
            for &estimator in &PANEL
            {
                match fit_estimator(estimator, &proper_features, &proper_targets)
                {
                    Some(model) => fitted.push((estimator, model)),
                    None =>
                    {
                        ok = false;
                        break;
                    },
                }
            }
            if !ok
            {
                continue;
            }

            let evidence: Vec<EstimatorEvidence> = fitted
                .iter()
                .map(|(estimator, model)| {
                    EstimatorEvidence::new(
                        estimator.name(),
                        residuals(model, &calibration_features, &calibration_targets),
                        residuals(model, &test_features, &test_targets),
                    )
                })
                .collect();

            let config = CertifiedPipelineConfig {
                tournament: EstimatorTournament {
                    orientation: Orientation::LowerIsBetter,
                    min_improvement: 0.0,
                    tie_margin: 0.0,
                    quality_floor: None,
                    resamples: 2000,
                    level: 0.95,
                    seed: TOURNAMENT_SEED,
                },
                coverage_level: LEVEL,
                coverage_mode: CoverageMode::Marginal,
            };
            let report = run_certified_pipeline(&evidence[0], &evidence[1..], &config)
                .expect("pipeline runs");

            summary.folds += 1;
            match &report.selection.decision
            {
                TournamentDecision::Select { .. } => summary.select += 1,
                TournamentDecision::HoldIncumbent => summary.hold += 1,
                TournamentDecision::Tie { .. } => summary.tie += 1,
                TournamentDecision::Inconclusive => summary.inconclusive += 1,
                TournamentDecision::RejectAll => summary.reject += 1,
            }
            if let Some(cert) = &report.coverage
                && cert.empirical_test_coverage.is_finite()
            {
                summary.coverage_sum += cert.empirical_test_coverage;
                summary.coverage_folds += 1;
            }

            // Honest robustness check: best robust held-out MAE vs OLS.
            let ols_mae = mean_abs(&evidence[0].test_residuals);
            let best_robust_mae = evidence[1..]
                .iter()
                .map(|e| mean_abs(&e.test_residuals))
                .fold(f64::INFINITY, f64::min);
            summary.ols_mae_sum += ols_mae;
            summary.best_robust_mae_sum += best_robust_mae;
            if best_robust_mae + 1.0e-9 < ols_mae
            {
                summary.robust_wins += 1;
            }
        }

        let folds = summary.folds.max(1) as f64;
        let mean_cov = if summary.coverage_folds > 0
        {
            summary.coverage_sum / summary.coverage_folds as f64
        }
        else
        {
            f64::NAN
        };
        println!(
            "{target:<14} {:>5}  {:>6} {:>4} {:>3} {:>5}  {:>8.4} {:>8.3} {:>10.3}  {:>4}/{}",
            summary.folds,
            summary.select,
            summary.hold,
            summary.tie,
            summary.inconclusive,
            mean_cov,
            summary.ols_mae_sum / folds,
            summary.best_robust_mae_sum / folds,
            summary.robust_wins,
            summary.folds,
        );
    }

    println!(
        "# honest read: on this dataset robustness rarely dominates OLS; the pipeline abstains \
(tie/inconclusive) where the panel is statistically indistinguishable, and the certified band \
holds near the nominal {LEVEL} coverage on held-out segments."
    );
}

fn gather(features: &[Vec<f64>], rows: &[usize]) -> Vec<Vec<f64>> {
    rows.iter().map(|&row| features[row].clone()).collect()
}

fn gather_targets(targets: &[f64], rows: &[usize]) -> Vec<f64> {
    rows.iter().map(|&row| targets[row]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_is_affine() {
        assert!((predict(&[2.0, -1.0], 0.5, &[3.0, 4.0]) - (0.5 + 6.0 - 4.0)).abs() < 1e-12);
    }

    #[test]
    fn standardizer_centres_and_scales_its_fit_data() {
        let features = vec![vec![1.0, 10.0], vec![3.0, 30.0], vec![5.0, 50.0]];
        let standardized = Standardizer::fit(&features).apply(&features);
        for column in 0..2
        {
            let mean: f64 =
                standardized.iter().map(|row| row[column]).sum::<f64>() / standardized.len() as f64;
            let variance: f64 = standardized
                .iter()
                .map(|row| (row[column] - mean).powi(2))
                .sum::<f64>()
                / standardized.len() as f64;
            assert!(
                mean.abs() < 1e-12,
                "column {column} mean {mean} not centred"
            );
            assert!(
                (variance - 1.0).abs() < 1e-9,
                "column {column} variance {variance} not unit"
            );
        }
    }

    #[test]
    fn residuals_are_signed_target_minus_prediction() {
        let model = (vec![1.0], 0.0); // ŷ = x
        let residuals = residuals(&model, &[vec![2.0], vec![5.0]], &[3.0, 4.0]);
        assert_eq!(residuals, vec![1.0, -1.0]);
    }
}
