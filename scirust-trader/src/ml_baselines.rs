//! Reproducible market-ML baselines.
//!
//! These implementations are deliberately compact reference baselines, not a
//! replacement for specialized ML libraries. Their role in `scirust-trader` is
//! to provide deterministic comparisons under the time-ordered validation
//! harness: ridge-linear/logistic models, CART regression trees, a seeded
//! Random Forest, gradient-boosted regression trees, and a simple autoregressive
//! sequence baseline.

use serde::{Deserialize, Serialize};

pub trait RegressionBaseline {
    fn predict(&self, features: &[f32]) -> f32;

    fn predict_many(&self, features: &[Vec<f32>]) -> Vec<f32> {
        features.iter().map(|row| self.predict(row)).collect()
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct LinearFitConfig {
    pub epochs: usize,
    pub learning_rate: f32,
    pub l2: f32,
}

impl Default for LinearFitConfig {
    fn default() -> Self {
        Self {
            epochs: 1_000,
            learning_rate: 0.03,
            l2: 1e-4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RidgeLinearRegressor {
    pub weights: Vec<f32>,
    pub bias: f32,
    pub feature_mean: Vec<f32>,
    pub feature_scale: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BaselineError {
    EmptyDataset,
    ShapeMismatch,
    NonFiniteData,
    InvalidConfig,
    InvalidLabels,
    InvalidSequence,
}

fn validate_xy(x: &[Vec<f32>], y: &[f32]) -> Result<usize, BaselineError> {
    if x.is_empty() || y.is_empty() || x.len() != y.len() || x[0].is_empty() {
        return Err(BaselineError::EmptyDataset);
    }
    let width = x[0].len();
    if x.iter().any(|row| row.len() != width) {
        return Err(BaselineError::ShapeMismatch);
    }
    if x.iter().flatten().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return Err(BaselineError::NonFiniteData);
    }
    Ok(width)
}

fn standardization(x: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>) {
    let width = x[0].len();
    let n = x.len() as f32;
    let mut mean = vec![0.0f32; width];
    for row in x {
        for j in 0..width {
            mean[j] += row[j];
        }
    }
    for value in &mut mean {
        *value /= n;
    }
    let mut scale = vec![0.0f32; width];
    for row in x {
        for j in 0..width {
            let d = row[j] - mean[j];
            scale[j] += d * d;
        }
    }
    for value in &mut scale {
        *value = (*value / n).sqrt();
        if *value < 1e-8 {
            *value = 1.0;
        }
    }
    (mean, scale)
}

impl RidgeLinearRegressor {
    pub fn fit(
        x: &[Vec<f32>],
        y: &[f32],
        cfg: LinearFitConfig,
    ) -> Result<Self, BaselineError> {
        let width = validate_xy(x, y)?;
        if cfg.epochs == 0
            || !cfg.learning_rate.is_finite()
            || cfg.learning_rate <= 0.0
            || !cfg.l2.is_finite()
            || cfg.l2 < 0.0
        {
            return Err(BaselineError::InvalidConfig);
        }
        let (mean, scale) = standardization(x);
        let mut weights = vec![0.0f32; width];
        let mut bias = y.iter().sum::<f32>() / y.len() as f32;
        let inv_n = 1.0 / y.len() as f32;
        for _ in 0..cfg.epochs {
            let mut grad_w = vec![0.0f32; width];
            let mut grad_b = 0.0f32;
            for (row, &target) in x.iter().zip(y) {
                let mut prediction = bias;
                for j in 0..width {
                    prediction += weights[j] * ((row[j] - mean[j]) / scale[j]);
                }
                let error = prediction - target;
                grad_b += error;
                for j in 0..width {
                    grad_w[j] += error * ((row[j] - mean[j]) / scale[j]);
                }
            }
            bias -= cfg.learning_rate * grad_b * inv_n;
            for j in 0..width {
                let gradient = grad_w[j] * inv_n + cfg.l2 * weights[j];
                weights[j] -= cfg.learning_rate * gradient;
            }
        }
        Ok(Self {
            weights,
            bias,
            feature_mean: mean,
            feature_scale: scale,
        })
    }
}

impl RegressionBaseline for RidgeLinearRegressor {
    fn predict(&self, features: &[f32]) -> f32 {
        if features.len() != self.weights.len() {
            return f32::NAN;
        }
        let mut out = self.bias;
        for (j, &value) in features.iter().enumerate() {
            out += self.weights[j] * ((value - self.feature_mean[j]) / self.feature_scale[j]);
        }
        out
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LogisticClassifier {
    pub weights: Vec<f32>,
    pub bias: f32,
    pub feature_mean: Vec<f32>,
    pub feature_scale: Vec<f32>,
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

impl LogisticClassifier {
    pub fn fit(
        x: &[Vec<f32>],
        labels: &[f32],
        cfg: LinearFitConfig,
    ) -> Result<Self, BaselineError> {
        let width = validate_xy(x, labels)?;
        if labels.iter().any(|v| *v != 0.0 && *v != 1.0) {
            return Err(BaselineError::InvalidLabels);
        }
        if cfg.epochs == 0
            || !cfg.learning_rate.is_finite()
            || cfg.learning_rate <= 0.0
            || !cfg.l2.is_finite()
            || cfg.l2 < 0.0
        {
            return Err(BaselineError::InvalidConfig);
        }
        let (mean, scale) = standardization(x);
        let mut weights = vec![0.0f32; width];
        let mut bias = 0.0f32;
        let inv_n = 1.0 / labels.len() as f32;
        for _ in 0..cfg.epochs {
            let mut grad_w = vec![0.0f32; width];
            let mut grad_b = 0.0f32;
            for (row, &label) in x.iter().zip(labels) {
                let mut logit = bias;
                for j in 0..width {
                    logit += weights[j] * ((row[j] - mean[j]) / scale[j]);
                }
                let error = sigmoid(logit) - label;
                grad_b += error;
                for j in 0..width {
                    grad_w[j] += error * ((row[j] - mean[j]) / scale[j]);
                }
            }
            bias -= cfg.learning_rate * grad_b * inv_n;
            for j in 0..width {
                weights[j] -= cfg.learning_rate * (grad_w[j] * inv_n + cfg.l2 * weights[j]);
            }
        }
        Ok(Self {
            weights,
            bias,
            feature_mean: mean,
            feature_scale: scale,
        })
    }

    pub fn predict_probability(&self, features: &[f32]) -> f32 {
        if features.len() != self.weights.len() {
            return f32::NAN;
        }
        let mut logit = self.bias;
        for (j, &value) in features.iter().enumerate() {
            logit += self.weights[j] * ((value - self.feature_mean[j]) / self.feature_scale[j]);
        }
        sigmoid(logit)
    }

    pub fn predict_class(&self, features: &[f32], threshold: f32) -> bool {
        self.predict_probability(features) >= threshold
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct TreeConfig {
    pub max_depth: usize,
    pub min_leaf: usize,
}

impl Default for TreeConfig {
    fn default() -> Self {
        Self {
            max_depth: 4,
            min_leaf: 5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RegressionTreeNode {
    Leaf {
        value: f32,
    },
    Split {
        feature: usize,
        threshold: f32,
        left: Box<RegressionTreeNode>,
        right: Box<RegressionTreeNode>,
    },
}

impl RegressionTreeNode {
    fn predict(&self, row: &[f32]) -> f32 {
        match self {
            Self::Leaf { value } => *value,
            Self::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                if row[*feature] <= *threshold {
                    left.predict(row)
                } else {
                    right.predict(row)
                }
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DecisionTreeRegressor {
    pub root: RegressionTreeNode,
    pub feature_count: usize,
}

fn target_mean(y: &[f32], indices: &[usize]) -> f32 {
    indices.iter().map(|&i| y[i]).sum::<f32>() / indices.len() as f32
}

fn sse(y: &[f32], indices: &[usize]) -> f32 {
    if indices.is_empty() {
        return 0.0;
    }
    let mean = target_mean(y, indices);
    indices
        .iter()
        .map(|&i| {
            let d = y[i] - mean;
            d * d
        })
        .sum()
}

fn all_features(width: usize) -> Vec<usize> {
    (0..width).collect()
}

fn build_tree(
    x: &[Vec<f32>],
    y: &[f32],
    indices: &[usize],
    depth: usize,
    cfg: TreeConfig,
    candidate_features: &[usize],
    rng: &mut Option<DeterministicRng>,
    mtry: usize,
) -> RegressionTreeNode {
    let leaf_value = target_mean(y, indices);
    if depth >= cfg.max_depth || indices.len() < cfg.min_leaf.saturating_mul(2) || sse(y, indices) <= 1e-12 {
        return RegressionTreeNode::Leaf { value: leaf_value };
    }

    let mut features = candidate_features.to_vec();
    if let Some(generator) = rng.as_mut() {
        for i in (1..features.len()).rev() {
            let j = generator.index(i + 1);
            features.swap(i, j);
        }
        features.truncate(mtry.min(features.len()).max(1));
        features.sort_unstable();
    }

    let mut best: Option<(f32, usize, f32, Vec<usize>, Vec<usize>)> = None;
    for &feature in &features {
        let mut values: Vec<f32> = indices.iter().map(|&i| x[i][feature]).collect();
        values.sort_by(|a, b| a.total_cmp(b));
        values.dedup_by(|a, b| a.to_bits() == b.to_bits());
        if values.len() < 2 {
            continue;
        }
        for pair in values.windows(2) {
            let threshold = pair[0] + (pair[1] - pair[0]) * 0.5;
            let mut left = Vec::new();
            let mut right = Vec::new();
            for &index in indices {
                if x[index][feature] <= threshold {
                    left.push(index);
                } else {
                    right.push(index);
                }
            }
            if left.len() < cfg.min_leaf || right.len() < cfg.min_leaf {
                continue;
            }
            let loss = sse(y, &left) + sse(y, &right);
            let replace = best
                .as_ref()
                .map(|(best_loss, best_feature, best_threshold, _, _)| {
                    loss < *best_loss - 1e-9
                        || ((loss - *best_loss).abs() <= 1e-9
                            && (feature < *best_feature
                                || (feature == *best_feature && threshold < *best_threshold)))
                })
                .unwrap_or(true);
            if replace {
                best = Some((loss, feature, threshold, left, right));
            }
        }
    }

    let Some((_, feature, threshold, left_indices, right_indices)) = best else {
        return RegressionTreeNode::Leaf { value: leaf_value };
    };
    let left = build_tree(
        x,
        y,
        &left_indices,
        depth + 1,
        cfg,
        candidate_features,
        rng,
        mtry,
    );
    let right = build_tree(
        x,
        y,
        &right_indices,
        depth + 1,
        cfg,
        candidate_features,
        rng,
        mtry,
    );
    RegressionTreeNode::Split {
        feature,
        threshold,
        left: Box::new(left),
        right: Box::new(right),
    }
}

impl DecisionTreeRegressor {
    pub fn fit(x: &[Vec<f32>], y: &[f32], cfg: TreeConfig) -> Result<Self, BaselineError> {
        let width = validate_xy(x, y)?;
        if cfg.min_leaf == 0 {
            return Err(BaselineError::InvalidConfig);
        }
        let indices: Vec<usize> = (0..x.len()).collect();
        let mut rng = None;
        let root = build_tree(
            x,
            y,
            &indices,
            0,
            cfg,
            &all_features(width),
            &mut rng,
            width,
        );
        Ok(Self {
            root,
            feature_count: width,
        })
    }
}

impl RegressionBaseline for DecisionTreeRegressor {
    fn predict(&self, features: &[f32]) -> f32 {
        if features.len() != self.feature_count {
            return f32::NAN;
        }
        self.root.predict(features)
    }
}

#[derive(Debug, Clone, Copy)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn index(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct RandomForestConfig {
    pub trees: usize,
    pub tree: TreeConfig,
    pub feature_fraction: f32,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RandomForestRegressor {
    pub trees: Vec<DecisionTreeRegressor>,
    pub feature_count: usize,
}

impl RandomForestRegressor {
    pub fn fit(
        x: &[Vec<f32>],
        y: &[f32],
        cfg: RandomForestConfig,
    ) -> Result<Self, BaselineError> {
        let width = validate_xy(x, y)?;
        if cfg.trees == 0
            || cfg.tree.min_leaf == 0
            || !cfg.feature_fraction.is_finite()
            || cfg.feature_fraction <= 0.0
            || cfg.feature_fraction > 1.0
        {
            return Err(BaselineError::InvalidConfig);
        }
        let mtry = ((width as f32 * cfg.feature_fraction).ceil() as usize).clamp(1, width);
        let mut rng = DeterministicRng::new(cfg.seed);
        let mut trees = Vec::with_capacity(cfg.trees);
        for _ in 0..cfg.trees {
            let bootstrap: Vec<usize> = (0..x.len()).map(|_| rng.index(x.len())).collect();
            let mut tree_rng = Some(rng);
            let root = build_tree(
                x,
                y,
                &bootstrap,
                0,
                cfg.tree,
                &all_features(width),
                &mut tree_rng,
                mtry,
            );
            rng = tree_rng.unwrap();
            trees.push(DecisionTreeRegressor {
                root,
                feature_count: width,
            });
        }
        Ok(Self {
            trees,
            feature_count: width,
        })
    }
}

impl RegressionBaseline for RandomForestRegressor {
    fn predict(&self, features: &[f32]) -> f32 {
        if features.len() != self.feature_count || self.trees.is_empty() {
            return f32::NAN;
        }
        self.trees.iter().map(|tree| tree.predict(features)).sum::<f32>()
            / self.trees.len() as f32
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct GradientBoostConfig {
    pub stages: usize,
    pub learning_rate: f32,
    pub tree: TreeConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GradientBoostedTreesRegressor {
    pub initial_prediction: f32,
    pub learning_rate: f32,
    pub trees: Vec<DecisionTreeRegressor>,
    pub feature_count: usize,
}

impl GradientBoostedTreesRegressor {
    pub fn fit(
        x: &[Vec<f32>],
        y: &[f32],
        cfg: GradientBoostConfig,
    ) -> Result<Self, BaselineError> {
        let width = validate_xy(x, y)?;
        if cfg.stages == 0
            || cfg.tree.min_leaf == 0
            || !cfg.learning_rate.is_finite()
            || cfg.learning_rate <= 0.0
            || cfg.learning_rate > 1.0
        {
            return Err(BaselineError::InvalidConfig);
        }
        let initial = y.iter().sum::<f32>() / y.len() as f32;
        let mut predictions = vec![initial; y.len()];
        let mut trees = Vec::with_capacity(cfg.stages);
        for _ in 0..cfg.stages {
            let residuals: Vec<f32> = y
                .iter()
                .zip(&predictions)
                .map(|(&target, &prediction)| target - prediction)
                .collect();
            let tree = DecisionTreeRegressor::fit(x, &residuals, cfg.tree)?;
            for (prediction, row) in predictions.iter_mut().zip(x) {
                *prediction += cfg.learning_rate * tree.predict(row);
            }
            trees.push(tree);
        }
        Ok(Self {
            initial_prediction: initial,
            learning_rate: cfg.learning_rate,
            trees,
            feature_count: width,
        })
    }
}

impl RegressionBaseline for GradientBoostedTreesRegressor {
    fn predict(&self, features: &[f32]) -> f32 {
        if features.len() != self.feature_count {
            return f32::NAN;
        }
        self.initial_prediction
            + self.learning_rate
                * self
                    .trees
                    .iter()
                    .map(|tree| tree.predict(features))
                    .sum::<f32>()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AutoregressiveRegressor {
    pub lags: usize,
    pub linear: RidgeLinearRegressor,
}

impl AutoregressiveRegressor {
    pub fn fit(
        series: &[f32],
        lags: usize,
        cfg: LinearFitConfig,
    ) -> Result<Self, BaselineError> {
        if lags == 0 || series.len() <= lags || series.iter().any(|v| !v.is_finite()) {
            return Err(BaselineError::InvalidSequence);
        }
        let mut x = Vec::with_capacity(series.len() - lags);
        let mut y = Vec::with_capacity(series.len() - lags);
        for i in lags..series.len() {
            x.push(series[i - lags..i].to_vec());
            y.push(series[i]);
        }
        Ok(Self {
            lags,
            linear: RidgeLinearRegressor::fit(&x, &y, cfg)?,
        })
    }

    pub fn predict_next(&self, history: &[f32]) -> f32 {
        if history.len() < self.lags {
            return f32::NAN;
        }
        self.linear.predict(&history[history.len() - self.lags..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn regression_data() -> (Vec<Vec<f32>>, Vec<f32>) {
        let x: Vec<Vec<f32>> = (0..80)
            .map(|i| {
                let a = i as f32 / 10.0;
                vec![a, (i % 7) as f32]
            })
            .collect();
        let y = x.iter().map(|row| 2.0 * row[0] - 0.5 * row[1] + 1.0).collect();
        (x, y)
    }

    #[test]
    fn ridge_recovers_simple_linear_relation() {
        let (x, y) = regression_data();
        let model = RidgeLinearRegressor::fit(&x, &y, LinearFitConfig::default()).unwrap();
        let prediction = model.predict(&[3.0, 2.0]);
        assert!((prediction - 6.0).abs() < 0.15);
    }

    #[test]
    fn logistic_separates_simple_classes() {
        let x = vec![vec![-2.0], vec![-1.0], vec![1.0], vec![2.0]];
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let model = LogisticClassifier::fit(
            &x,
            &y,
            LinearFitConfig {
                epochs: 1_500,
                learning_rate: 0.05,
                l2: 0.0,
            },
        )
        .unwrap();
        assert!(!model.predict_class(&[-1.5], 0.5));
        assert!(model.predict_class(&[1.5], 0.5));
    }

    #[test]
    fn tree_learns_threshold_structure() {
        let x: Vec<Vec<f32>> = (0..40).map(|i| vec![i as f32]).collect();
        let y: Vec<f32> = (0..40).map(|i| if i < 20 { -1.0 } else { 1.0 }).collect();
        let tree = DecisionTreeRegressor::fit(
            &x,
            &y,
            TreeConfig {
                max_depth: 2,
                min_leaf: 2,
            },
        )
        .unwrap();
        assert!(tree.predict(&[5.0]) < 0.0);
        assert!(tree.predict(&[35.0]) > 0.0);
    }

    #[test]
    fn seeded_random_forest_is_reproducible() {
        let (x, y) = regression_data();
        let cfg = RandomForestConfig {
            trees: 12,
            tree: TreeConfig {
                max_depth: 4,
                min_leaf: 3,
            },
            feature_fraction: 0.7,
            seed: 42,
        };
        let a = RandomForestRegressor::fit(&x, &y, cfg).unwrap();
        let b = RandomForestRegressor::fit(&x, &y, cfg).unwrap();
        assert_eq!(a.predict(&[2.5, 3.0]).to_bits(), b.predict(&[2.5, 3.0]).to_bits());
    }

    #[test]
    fn boosting_improves_over_constant_prediction() {
        let (x, y) = regression_data();
        let model = GradientBoostedTreesRegressor::fit(
            &x,
            &y,
            GradientBoostConfig {
                stages: 20,
                learning_rate: 0.1,
                tree: TreeConfig {
                    max_depth: 2,
                    min_leaf: 3,
                },
            },
        )
        .unwrap();
        let constant_mse = y
            .iter()
            .map(|target| (target - model.initial_prediction).powi(2))
            .sum::<f32>()
            / y.len() as f32;
        let boosted_mse = x
            .iter()
            .zip(&y)
            .map(|(row, target)| (target - model.predict(row)).powi(2))
            .sum::<f32>()
            / y.len() as f32;
        assert!(boosted_mse < constant_mse);
    }

    #[test]
    fn autoregression_is_deterministic() {
        let series: Vec<f32> = (0..80).map(|i| i as f32 * 0.1).collect();
        let cfg = LinearFitConfig::default();
        let a = AutoregressiveRegressor::fit(&series, 3, cfg).unwrap();
        let b = AutoregressiveRegressor::fit(&series, 3, cfg).unwrap();
        assert_eq!(a.predict_next(&series).to_bits(), b.predict_next(&series).to_bits());
    }
}
