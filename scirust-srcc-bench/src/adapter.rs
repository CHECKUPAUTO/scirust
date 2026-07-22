//! The common baseline interface, with declared capabilities.
//!
//! Every method enters the harness through [`BaselineAdapter`], which
//! declares — as data, before anything runs — three things reviewers need:
//!
//! - its [`TaskKind`] (a regression model is never scored on anomaly
//!   metrics, and vice versa);
//! - its [`FittingProtocol`]: **inductive** methods fit on the training
//!   split and are applied to the evaluation split; **transductive** methods
//!   (LOF, DBSCAN as implemented in `scirust-unsupervised`) score the
//!   evaluation split as a whole and ignore the training split — a
//!   legitimate unsupervised protocol, but one that must be *declared*, not
//!   discovered in a footnote;
//! - its [`AdapterOutput`] shape, which decides which metrics exist at all:
//!   score-producing detectors get AUROC, label-only detectors (DBSCAN)
//!   structurally cannot — there is no score to rank. Methods are never
//!   forced into metrics they do not support.
//!
//! Failures of the underlying estimators (singular covariance,
//! non-convergence, degenerate scales) surface as typed
//! [`AdapterError`] values carrying the method's own error text — never a
//! default score.

use core::fmt;

use scirust_learning::{
    RegressionDataset, RobustLoss, RobustRegressionConfig, RobustRegressionMethod,
    fit_robust_regression,
};
use scirust_multivariate::{FittedDistanceMetric, Matrix as MultivariateMatrix};
use scirust_solvers::linalg::Matrix as SolversMatrix;
use scirust_spc::{CusumChart, EwmaChart, HotellingT2};
use scirust_unsupervised::{
    Dbscan, DbscanConfig, IForestConfig, IsolationForest, LocalOutlierFactor, LofConfig,
};

use crate::dataset::TabularDataset;

/// What kind of question a method answers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TaskKind {
    /// Predict a continuous target for each evaluation row.
    Regression,
    /// Rank or flag anomalous evaluation rows.
    AnomalyDetection,
    /// Raise alarms along a univariate evaluation stream.
    StreamAlarm,
}

/// How a method uses the training split.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FittingProtocol {
    /// Fits on the training split, applies to the evaluation split.
    Inductive,
    /// Operates on the evaluation split as a whole; the training split is
    /// ignored. Declared so the records say so.
    Transductive,
}

/// What a method produces for the evaluation split.
#[derive(Clone, Debug, PartialEq)]
pub enum AdapterOutput {
    /// One predicted target per evaluation row.
    Predictions(Vec<f64>),
    /// One anomaly score per evaluation row (higher = more anomalous).
    AnomalyScores(Vec<f64>),
    /// One binary anomaly flag per evaluation row (no scores — AUROC is
    /// structurally impossible and is never fabricated).
    AnomalyLabels(Vec<bool>),
    /// Alarm positions (0-based steps) along the evaluation stream.
    AlarmSteps(Vec<usize>),
}

/// Typed adapter errors.
#[derive(Clone, Debug, PartialEq)]
pub enum AdapterError {
    /// The training split is empty.
    EmptyTrain,
    /// The evaluation split is empty.
    EmptyEvaluation,
    /// Train and evaluation disagree on the feature count.
    FeatureCountMismatch {
        /// Training feature count.
        train: usize,
        /// Evaluation feature count.
        evaluation: usize,
    },
    /// A stream adapter addresses a column the dataset does not have.
    MissingColumn {
        /// Requested column.
        column: usize,
        /// Available columns.
        feature_count: usize,
    },
    /// The underlying estimator failed; the detail is its own error text.
    UnderlyingFailure {
        /// The adapter's stable name.
        method: &'static str,
        /// The underlying error, verbatim.
        detail: String,
    },
    /// The fit is degenerate for this data (singular covariance, zero
    /// scale); stated as such, never smoothed over.
    DegenerateFit {
        /// The adapter's stable name.
        method: &'static str,
        /// What is degenerate.
        detail: &'static str,
    },
}

impl fmt::Display for AdapterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::EmptyTrain => formatter.write_str("training split is empty"),
            Self::EmptyEvaluation => formatter.write_str("evaluation split is empty"),
            Self::FeatureCountMismatch { train, evaluation } => write!(
                formatter,
                "train has {train} features, evaluation has {evaluation}"
            ),
            Self::MissingColumn {
                column,
                feature_count,
            } => write!(
                formatter,
                "stream column {column} out of range for {feature_count} features"
            ),
            Self::UnderlyingFailure { method, detail } =>
            {
                write!(formatter, "{method} failed: {detail}")
            },
            Self::DegenerateFit { method, detail } =>
            {
                write!(formatter, "{method} fit is degenerate: {detail}")
            },
        }
    }
}

impl std::error::Error for AdapterError {}

/// The common interface every benchmarked method implements.
pub trait BaselineAdapter {
    /// Stable method identifier (the `method` key of emitted records).
    fn name(&self) -> &'static str;

    /// The task this method answers.
    fn task(&self) -> TaskKind;

    /// How the training split is used.
    fn protocol(&self) -> FittingProtocol;

    /// Runs the method: fit according to the declared protocol, produce the
    /// declared output for the evaluation split.
    fn run(
        &self,
        train: &TabularDataset,
        evaluation: &TabularDataset,
    ) -> Result<AdapterOutput, AdapterError>;
}

fn check_shapes(train: &TabularDataset, evaluation: &TabularDataset) -> Result<(), AdapterError> {
    if train.sample_count() == 0
    {
        return Err(AdapterError::EmptyTrain);
    }

    if evaluation.sample_count() == 0
    {
        return Err(AdapterError::EmptyEvaluation);
    }

    if train.feature_count() != evaluation.feature_count()
    {
        return Err(AdapterError::FeatureCountMismatch {
            train: train.feature_count(),
            evaluation: evaluation.feature_count(),
        });
    }

    Ok(())
}

fn solvers_matrix(features: &[Vec<f64>]) -> SolversMatrix {
    let rows = features.len();
    let cols = features.first().map_or(0, Vec::len);
    let mut data = Vec::with_capacity(rows * cols);

    for row in features
    {
        data.extend_from_slice(row);
    }

    SolversMatrix::from_row_major(rows, cols, data)
}

fn multivariate_matrix(features: &[Vec<f64>]) -> MultivariateMatrix {
    MultivariateMatrix {
        rows: features.len(),
        cols: features.first().map_or(0, Vec::len),
        data: features.to_vec(),
    }
}

/// Sample mean and (n − 1) standard deviation of one feature column.
fn column_mean_std(dataset: &TabularDataset, column: usize) -> Result<(f64, f64), AdapterError> {
    let feature_count = dataset.feature_count();

    if column >= feature_count
    {
        return Err(AdapterError::MissingColumn {
            column,
            feature_count,
        });
    }

    let n = dataset.sample_count();
    let mean = dataset.features.iter().map(|row| row[column]).sum::<f64>() / n as f64;

    if n < 2
    {
        return Ok((mean, 0.0));
    }

    let variance = dataset
        .features
        .iter()
        .map(|row| (row[column] - mean).powi(2))
        .sum::<f64>()
        / (n - 1) as f64;

    Ok((mean, variance.sqrt()))
}

// ---------------------------------------------------------------------------
// Regression family (scirust-learning)
// ---------------------------------------------------------------------------

/// Robust-regression adapter over `scirust-learning`'s deterministic
/// estimators.
pub struct RobustRegressionAdapter {
    name: &'static str,
    configuration: RobustRegressionConfig,
}

impl RobustRegressionAdapter {
    /// Ordinary least squares (0 % breakdown baseline).
    #[must_use]
    pub fn ordinary_least_squares() -> Self {
        Self {
            name: "ols",
            configuration: RobustRegressionConfig {
                method: RobustRegressionMethod::OrdinaryLeastSquares,
                ..RobustRegressionConfig::default()
            },
        }
    }

    /// Huber IRLS with the given delta.
    #[must_use]
    pub fn huber(delta: f64) -> Self {
        Self {
            name: "huber_irls",
            configuration: RobustRegressionConfig {
                method: RobustRegressionMethod::IterativelyReweightedLeastSquares,
                loss: RobustLoss::Huber { delta },
                ..RobustRegressionConfig::default()
            },
        }
    }

    /// Trimmed least squares retaining the given fraction.
    #[must_use]
    pub fn trimmed(retained_fraction: f64) -> Self {
        Self {
            name: "trimmed_ls",
            configuration: RobustRegressionConfig {
                method: RobustRegressionMethod::TrimmedLeastSquares { retained_fraction },
                ..RobustRegressionConfig::default()
            },
        }
    }

    /// Median-of-means over seeded blocks.
    #[must_use]
    pub fn median_of_means(block_count: usize, seed: u64) -> Self {
        Self {
            name: "median_of_means",
            configuration: RobustRegressionConfig {
                method: RobustRegressionMethod::MedianOfMeans { block_count, seed },
                ..RobustRegressionConfig::default()
            },
        }
    }
}

impl BaselineAdapter for RobustRegressionAdapter {
    fn name(&self) -> &'static str {
        self.name
    }

    fn task(&self) -> TaskKind {
        TaskKind::Regression
    }

    fn protocol(&self) -> FittingProtocol {
        FittingProtocol::Inductive
    }

    fn run(
        &self,
        train: &TabularDataset,
        evaluation: &TabularDataset,
    ) -> Result<AdapterOutput, AdapterError> {
        check_shapes(train, evaluation)?;

        let dataset = RegressionDataset {
            features: solvers_matrix(&train.features),
            targets: SolversMatrix::from_row_major(train.sample_count(), 1, train.targets.clone()),
            sample_weights: None,
        };

        let report = fit_robust_regression(&dataset, self.configuration).map_err(|error| {
            AdapterError::UnderlyingFailure {
                method: self.name,
                detail: error.to_string(),
            }
        })?;

        let predictions = report
            .model
            .predict(&solvers_matrix(&evaluation.features))
            .map_err(|error| AdapterError::UnderlyingFailure {
                method: self.name,
                detail: error.to_string(),
            })?;

        Ok(AdapterOutput::Predictions(
            (0..evaluation.sample_count())
                .map(|row| predictions[(row, 0)])
                .collect(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Anomaly family (scirust-unsupervised, scirust-multivariate, scirust-spc)
// ---------------------------------------------------------------------------

/// Isolation Forest (inductive, seeded, score-producing).
pub struct IsolationForestAdapter {
    /// Forest configuration (the seed lives here).
    pub configuration: IForestConfig,
}

impl BaselineAdapter for IsolationForestAdapter {
    fn name(&self) -> &'static str {
        "isolation_forest"
    }

    fn task(&self) -> TaskKind {
        TaskKind::AnomalyDetection
    }

    fn protocol(&self) -> FittingProtocol {
        FittingProtocol::Inductive
    }

    fn run(
        &self,
        train: &TabularDataset,
        evaluation: &TabularDataset,
    ) -> Result<AdapterOutput, AdapterError> {
        check_shapes(train, evaluation)?;

        let mut forest = IsolationForest::new(self.configuration.clone());
        forest.fit(&train.features);

        Ok(AdapterOutput::AnomalyScores(
            forest.anomaly_scores(&evaluation.features),
        ))
    }
}

/// Local outlier factor (transductive, score-producing).
pub struct LofAdapter {
    /// Neighborhood size.
    pub configuration: LofConfig,
}

impl BaselineAdapter for LofAdapter {
    fn name(&self) -> &'static str {
        "local_outlier_factor"
    }

    fn task(&self) -> TaskKind {
        TaskKind::AnomalyDetection
    }

    fn protocol(&self) -> FittingProtocol {
        FittingProtocol::Transductive
    }

    fn run(
        &self,
        train: &TabularDataset,
        evaluation: &TabularDataset,
    ) -> Result<AdapterOutput, AdapterError> {
        check_shapes(train, evaluation)?;

        let lof = LocalOutlierFactor::new(self.configuration.clone());

        Ok(AdapterOutput::AnomalyScores(
            lof.fit_predict(&evaluation.features),
        ))
    }
}

/// DBSCAN as a label-only anomaly detector (transductive; noise = anomaly).
pub struct DbscanAdapter {
    /// Density parameters.
    pub configuration: DbscanConfig,
}

impl BaselineAdapter for DbscanAdapter {
    fn name(&self) -> &'static str {
        "dbscan_noise"
    }

    fn task(&self) -> TaskKind {
        TaskKind::AnomalyDetection
    }

    fn protocol(&self) -> FittingProtocol {
        FittingProtocol::Transductive
    }

    fn run(
        &self,
        train: &TabularDataset,
        evaluation: &TabularDataset,
    ) -> Result<AdapterOutput, AdapterError> {
        check_shapes(train, evaluation)?;

        let clustering = Dbscan::new(self.configuration.clone()).fit(&evaluation.features);

        Ok(AdapterOutput::AnomalyLabels(
            clustering.labels.iter().map(|&label| label == -1).collect(),
        ))
    }
}

/// Regularized Mahalanobis distance to the training location (inductive,
/// score-producing; `scirust-multivariate`'s fitted metric).
pub struct MahalanobisAdapter {
    /// Ridge added to the scatter before inversion.
    pub ridge: f64,
}

impl BaselineAdapter for MahalanobisAdapter {
    fn name(&self) -> &'static str {
        "regularized_mahalanobis"
    }

    fn task(&self) -> TaskKind {
        TaskKind::AnomalyDetection
    }

    fn protocol(&self) -> FittingProtocol {
        FittingProtocol::Inductive
    }

    fn run(
        &self,
        train: &TabularDataset,
        evaluation: &TabularDataset,
    ) -> Result<AdapterOutput, AdapterError> {
        check_shapes(train, evaluation)?;

        let metric = FittedDistanceMetric::fit_regularized_mahalanobis(
            &multivariate_matrix(&train.features),
            self.ridge,
        )
        .map_err(|error| AdapterError::UnderlyingFailure {
            method: "regularized_mahalanobis",
            detail: error.to_string(),
        })?;

        let FittedDistanceMetric::RegularizedMahalanobis { location, .. } = &metric
        else
        {
            return Err(AdapterError::DegenerateFit {
                method: "regularized_mahalanobis",
                detail: "fit returned an unexpected metric variant",
            });
        };

        let location = location.clone();

        let mut scores = Vec::with_capacity(evaluation.sample_count());

        for row in &evaluation.features
        {
            let score = metric.distance(row, &location).map_err(|error| {
                AdapterError::UnderlyingFailure {
                    method: "regularized_mahalanobis",
                    detail: error.to_string(),
                }
            })?;

            scores.push(score);
        }

        Ok(AdapterOutput::AnomalyScores(scores))
    }
}

/// Hotelling T² (inductive, score-producing; `scirust-spc`).
pub struct HotellingT2Adapter;

impl BaselineAdapter for HotellingT2Adapter {
    fn name(&self) -> &'static str {
        "hotelling_t2"
    }

    fn task(&self) -> TaskKind {
        TaskKind::AnomalyDetection
    }

    fn protocol(&self) -> FittingProtocol {
        FittingProtocol::Inductive
    }

    fn run(
        &self,
        train: &TabularDataset,
        evaluation: &TabularDataset,
    ) -> Result<AdapterOutput, AdapterError> {
        check_shapes(train, evaluation)?;

        let chart = HotellingT2::fit(&train.features).ok_or(AdapterError::DegenerateFit {
            method: "hotelling_t2",
            detail: "training covariance is singular",
        })?;

        Ok(AdapterOutput::AnomalyScores(
            evaluation
                .features
                .iter()
                .map(|row| chart.t2(row))
                .collect(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Stream family (scirust-spc), univariate on a designated column
// ---------------------------------------------------------------------------

/// CUSUM chart on one feature column: target and sigma are estimated from
/// the training column (inductive); the chart is reset after each alarm so
/// every alarm is a fresh detection.
pub struct CusumAdapter {
    /// The monitored feature column.
    pub column: usize,
    /// Reference value `k` in sigma units (half the shift to detect).
    pub k: f64,
    /// Decision interval `h` in sigma units.
    pub h: f64,
}

impl BaselineAdapter for CusumAdapter {
    fn name(&self) -> &'static str {
        "cusum"
    }

    fn task(&self) -> TaskKind {
        TaskKind::StreamAlarm
    }

    fn protocol(&self) -> FittingProtocol {
        FittingProtocol::Inductive
    }

    fn run(
        &self,
        train: &TabularDataset,
        evaluation: &TabularDataset,
    ) -> Result<AdapterOutput, AdapterError> {
        check_shapes(train, evaluation)?;

        let (target, sigma) = column_mean_std(train, self.column)?;

        if sigma == 0.0
        {
            return Err(AdapterError::DegenerateFit {
                method: "cusum",
                detail: "training column has zero standard deviation",
            });
        }

        let mut chart = CusumChart::new(target, sigma, self.k, self.h);
        let mut alarms = Vec::new();

        for (step, row) in evaluation.features.iter().enumerate()
        {
            if chart.update(row[self.column]).is_some()
            {
                alarms.push(step);
                chart.reset();
            }
        }

        Ok(AdapterOutput::AlarmSteps(alarms))
    }
}

/// EWMA chart on one feature column: center and sigma from the training
/// column (inductive); the chart is re-initialized after each alarm.
pub struct EwmaAdapter {
    /// The monitored feature column.
    pub column: usize,
    /// EWMA smoothing weight `lambda` in (0, 1].
    pub lambda: f64,
    /// Control-limit width `l` in sigma units.
    pub l: f64,
}

impl BaselineAdapter for EwmaAdapter {
    fn name(&self) -> &'static str {
        "ewma"
    }

    fn task(&self) -> TaskKind {
        TaskKind::StreamAlarm
    }

    fn protocol(&self) -> FittingProtocol {
        FittingProtocol::Inductive
    }

    fn run(
        &self,
        train: &TabularDataset,
        evaluation: &TabularDataset,
    ) -> Result<AdapterOutput, AdapterError> {
        check_shapes(train, evaluation)?;

        let (center, sigma) = column_mean_std(train, self.column)?;

        if sigma == 0.0
        {
            return Err(AdapterError::DegenerateFit {
                method: "ewma",
                detail: "training column has zero standard deviation",
            });
        }

        let mut chart = EwmaChart::new(center, sigma, self.lambda, self.l);
        let mut alarms = Vec::new();

        for (step, row) in evaluation.features.iter().enumerate()
        {
            if chart.update(row[self.column])
            {
                alarms.push(step);
                chart = EwmaChart::new(center, sigma, self.lambda, self.l);
            }
        }

        Ok(AdapterOutput::AlarmSteps(alarms))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn regression_data() -> (TabularDataset, TabularDataset) {
        // y = 2·x0 − x1 + 1, exactly.
        let make = |rows: core::ops::Range<usize>| -> TabularDataset {
            let features: Vec<Vec<f64>> = rows
                .clone()
                .map(|row| vec![row as f64, (row % 3) as f64])
                .collect();

            let targets = features
                .iter()
                .map(|row| 2.0 * row[0] - row[1] + 1.0)
                .collect();

            TabularDataset {
                features,
                targets,
                groups: None,
                time_index: None,
            }
        };

        (make(0..12), make(12..18))
    }

    fn anomaly_data() -> (TabularDataset, TabularDataset) {
        let train = TabularDataset {
            features: (0..24)
                .map(|row| vec![(row % 5) as f64 * 0.1, 1.0 + (row % 3) as f64 * 0.1])
                .collect(),
            targets: vec![0.0; 24],
            groups: None,
            time_index: None,
        };

        // Evaluation: 8 clean rows near the training cloud, 2 far outliers.
        let mut features: Vec<Vec<f64>> = (0..8)
            .map(|row| vec![(row % 5) as f64 * 0.1, 1.0 + (row % 3) as f64 * 0.1])
            .collect();
        features.push(vec![25.0, -30.0]);
        features.push(vec![30.0, -25.0]);

        let mut labels = vec![0.0; 8];
        labels.extend([1.0, 1.0]);

        let evaluation = TabularDataset {
            features,
            targets: labels,
            groups: None,
            time_index: None,
        };

        (train, evaluation)
    }

    #[test]
    fn regression_adapters_recover_a_noiseless_affine_law() {
        let (train, evaluation) = regression_data();

        for adapter in [
            RobustRegressionAdapter::ordinary_least_squares(),
            RobustRegressionAdapter::huber(1.345),
        ]
        {
            let output = adapter.run(&train, &evaluation).unwrap();

            let AdapterOutput::Predictions(predictions) = output
            else
            {
                panic!("regression adapters produce predictions");
            };

            for (prediction, reference) in predictions.iter().zip(&evaluation.targets)
            {
                assert!(
                    (prediction - reference).abs() < 1.0e-8,
                    "prediction {prediction} vs {reference}",
                );
            }
        }
    }

    #[test]
    fn score_producing_detectors_rank_gross_outliers_last() {
        let (train, evaluation) = anomaly_data();

        let adapters: Vec<Box<dyn BaselineAdapter>> = vec![
            Box::new(IsolationForestAdapter {
                configuration: IForestConfig {
                    n_trees: 50,
                    subsample_size: 16,
                    max_depth: 8,
                    seed: 7,
                },
            }),
            Box::new(MahalanobisAdapter { ridge: 1.0e-6 }),
            Box::new(HotellingT2Adapter),
            Box::new(LofAdapter {
                configuration: LofConfig { k: 3 },
            }),
        ];

        for adapter in adapters
        {
            let AdapterOutput::AnomalyScores(scores) = adapter.run(&train, &evaluation).unwrap()
            else
            {
                panic!("{} must produce scores", adapter.name());
            };

            assert_eq!(scores.len(), 10);

            let clean_max = scores[..8].iter().fold(f64::MIN, |a, &b| a.max(b));

            for &outlier_score in &scores[8..]
            {
                assert!(
                    outlier_score > clean_max,
                    "{}: outlier score {outlier_score} must exceed clean max {clean_max}",
                    adapter.name(),
                );
            }
        }
    }

    #[test]
    fn dbscan_flags_noise_without_scores() {
        let (train, evaluation) = anomaly_data();

        let adapter = DbscanAdapter {
            configuration: DbscanConfig {
                eps: 0.5,
                min_pts: 3,
            },
        };

        assert_eq!(adapter.protocol(), FittingProtocol::Transductive);

        let AdapterOutput::AnomalyLabels(flags) = adapter.run(&train, &evaluation).unwrap()
        else
        {
            panic!("dbscan produces labels only");
        };

        assert!(flags[8] && flags[9], "gross outliers must be noise");
        assert!(
            !flags[..8].iter().any(|&flag| flag),
            "clean rows must not be noise"
        );
    }

    #[test]
    fn stream_adapters_alarm_after_a_level_shift() {
        let train = TabularDataset {
            features: (0..40)
                .map(|row| vec![10.0 + ((row % 7) as f64 - 3.0) * 0.1])
                .collect(),
            targets: vec![0.0; 40],
            groups: None,
            time_index: None,
        };

        // Evaluation stream: 15 in-control steps, then a +2 level shift.
        let evaluation = TabularDataset {
            features: (0..30)
                .map(|step| {
                    let base = 10.0 + ((step % 7) as f64 - 3.0) * 0.1;
                    let shifted = if step >= 15 { base + 2.0 } else { base };
                    vec![shifted]
                })
                .collect(),
            targets: vec![0.0; 30],
            groups: None,
            time_index: None,
        };

        for adapter in [
            Box::new(CusumAdapter {
                column: 0,
                k: 0.5,
                h: 5.0,
            }) as Box<dyn BaselineAdapter>,
            Box::new(EwmaAdapter {
                column: 0,
                lambda: 0.2,
                l: 2.7,
            }),
        ]
        {
            let AdapterOutput::AlarmSteps(alarms) = adapter.run(&train, &evaluation).unwrap()
            else
            {
                panic!("stream adapters produce alarm steps");
            };

            let first_after_onset = alarms.iter().find(|&&step| step >= 15);

            assert!(
                first_after_onset.is_some(),
                "{}: the +2 sigma-scale shift must eventually alarm",
                adapter.name(),
            );

            assert!(
                !alarms.iter().any(|&step| step < 15),
                "{}: no false alarm expected on the in-control prefix",
                adapter.name(),
            );
        }
    }

    #[test]
    fn shape_and_degeneracy_errors_are_typed() {
        let (train, evaluation) = regression_data();

        let empty = TabularDataset {
            features: vec![],
            targets: vec![],
            groups: None,
            time_index: None,
        };

        let adapter = RobustRegressionAdapter::ordinary_least_squares();

        assert_eq!(
            adapter.run(&empty, &evaluation),
            Err(AdapterError::EmptyTrain)
        );
        assert_eq!(
            adapter.run(&train, &empty),
            Err(AdapterError::EmptyEvaluation)
        );

        let mut narrow = evaluation.clone();

        for row in &mut narrow.features
        {
            row.pop();
        }

        assert_eq!(
            adapter.run(&train, &narrow),
            Err(AdapterError::FeatureCountMismatch {
                train: 2,
                evaluation: 1,
            }),
        );

        // Constant column: CUSUM cannot scale.
        let constant = TabularDataset {
            features: vec![vec![5.0], vec![5.0], vec![5.0]],
            targets: vec![0.0; 3],
            groups: None,
            time_index: None,
        };

        let cusum = CusumAdapter {
            column: 0,
            k: 0.5,
            h: 5.0,
        };

        assert_eq!(
            cusum.run(&constant, &constant),
            Err(AdapterError::DegenerateFit {
                method: "cusum",
                detail: "training column has zero standard deviation",
            }),
        );

        let out_of_range = CusumAdapter {
            column: 5,
            k: 0.5,
            h: 5.0,
        };

        assert_eq!(
            out_of_range.run(&train, &evaluation),
            Err(AdapterError::MissingColumn {
                column: 5,
                feature_count: 2,
            }),
        );
    }
}
