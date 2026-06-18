//! scirust-automl — Automated Machine Learning
//!
//! Pipeline search, hyperparameter optimization (random/grid/Bayesian),
//! model selection with statistical tests, feature engineering, and a
//! full AutoML orchestrator.  Deterministic via an embedded XorShift64
//! PRNG (no `rand` crate dependency in this crate).

use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;

// =========================================================================
// 1.  XorShift64 – deterministic PRNG (no `rand` crate)
// =========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.max(1), // state must be non-zero
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform f64 in [0, 1)
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        let u = self.next_u64();
        (u >> 11) as f64 * 1.1102230246251565e-16 // 1.0 / 2^53
    }

    /// Uniform f64 in [lo, hi)
    #[inline]
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// Uniform f64 in [lo, hi] via log-space — for LogUniform
    #[inline]
    pub fn log_uniform(&mut self, lo: f64, hi: f64) -> f64 {
        (lo.ln() + (hi.ln() - lo.ln()) * self.next_f64()).exp()
    }

    /// Uniform usize in [0, n)
    #[inline]
    pub fn usize(&mut self, n: usize) -> usize {
        if n == 0
        {
            return 0;
        }
        (((self.next_u64() as u128) * (n as u128)) >> 64) as usize
    }

    /// Normal(0, 1) via Box–Muller
    pub fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-30);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// =========================================================================
// 2.  Parameter distributions
// =========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParamDistribution {
    Uniform { lo: f64, hi: f64 },
    LogUniform { lo: f64, hi: f64 },
    Categorical { options: Vec<String> },
    IntRange { lo: i64, hi: i64 },
}

impl ParamDistribution {
    pub fn sample(&self, rng: &mut XorShift64) -> f64 {
        match self
        {
            Self::Uniform { lo, hi } => rng.uniform(*lo, *hi),
            Self::LogUniform { lo, hi } => rng.log_uniform(*lo, *hi),
            Self::Categorical { options } =>
            {
                let idx = rng.usize(options.len());
                // Encode categorical as f64 via hash for sampling
                hash_str(&options[idx])
            },
            Self::IntRange { lo, hi } =>
            {
                let i = *lo + rng.usize((*hi - *lo + 1) as usize) as i64;
                i as f64
            },
        }
    }

    /// Deterministic grid over the distribution (for grid search).
    /// Returns up to `n` points.  Categorical returns one value per option.
    pub fn grid_points(&self, n: usize) -> Vec<f64> {
        match self
        {
            Self::Uniform { lo, hi } =>
            {
                if n <= 1
                {
                    return vec![(*lo + *hi) / 2.0];
                }
                (0..n)
                    .map(|i| *lo + (*hi - *lo) * (i as f64) / ((n - 1) as f64))
                    .collect()
            },
            Self::LogUniform { lo, hi } =>
            {
                let llo = lo.ln();
                let lhi = hi.ln();
                if n <= 1
                {
                    return vec![((llo + lhi) / 2.0).exp()];
                }
                (0..n)
                    .map(|i| (llo + (lhi - llo) * (i as f64) / ((n - 1) as f64)).exp())
                    .collect()
            },
            Self::Categorical { options } =>
            {
                let m = n.min(options.len());
                options.iter().map(|s| hash_str(s)).collect::<Vec<_>>()[..m].to_vec()
            },
            Self::IntRange { lo, hi } =>
            {
                let range = (*hi - *lo + 1) as usize;
                let step = (range as f64 / n as f64).ceil() as usize;
                let step = step.max(1);
                (*lo..=*hi)
                    .step_by(step)
                    .take(n)
                    .map(|v| v as f64)
                    .collect()
            },
        }
    }
}

fn hash_str(s: &str) -> f64 {
    let mut h: u64 = 14695981039346656037;
    for b in s.bytes()
    {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    (h & ((1u64 << 53) - 1)) as f64 / (1u64 << 53) as f64
}

// =========================================================================
// 3.  Preprocessing step definitions
// =========================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PreprocessorKind {
    StandardScaler,
    Normalizer,
    PCA { n_components: usize },
    PolynomialFeatures { degree: usize },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelKind {
    Linear,
    RandomForest,
    GradientBoosting,
    NeuralNetwork,
}

// =========================================================================
// 4.  Pipeline template — the search space
// =========================================================================

/// Hyperparameter entry: name → distribution
pub type HyperSpace = HashMap<String, ParamDistribution>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineTemplate {
    pub preprocessors: Vec<PreprocessorKind>,
    pub model: ModelKind,
    pub hyperparameters: HyperSpace,
}

impl PipelineTemplate {
    pub fn new(model: ModelKind, preprocessors: Vec<PreprocessorKind>) -> Self {
        let hp = default_hyperparams(&model);
        Self {
            preprocessors,
            model,
            hyperparameters: hp,
        }
    }

    /// Generate `n` candidate pipelines by sampling the hyperparameter space.
    pub fn generate_candidates(&self, n: usize, rng: &mut XorShift64) -> Vec<PipelineConfig> {
        (0..n)
            .map(|_| {
                let mut params = HashMap::new();
                for (name, dist) in &self.hyperparameters
                {
                    params.insert(name.clone(), dist.sample(rng));
                }
                PipelineConfig {
                    preprocessors: self.preprocessors.clone(),
                    model: self.model.clone(),
                    params,
                }
            })
            .collect()
    }
}

fn default_hyperparams(model: &ModelKind) -> HyperSpace {
    let mut h = HashMap::new();
    match model
    {
        ModelKind::Linear =>
        {
            h.insert(
                "regularization".into(),
                ParamDistribution::Uniform { lo: 0.0, hi: 10.0 },
            );
        },
        ModelKind::RandomForest =>
        {
            h.insert(
                "n_trees".into(),
                ParamDistribution::IntRange { lo: 10, hi: 500 },
            );
            h.insert(
                "max_depth".into(),
                ParamDistribution::IntRange { lo: 3, hi: 30 },
            );
            h.insert(
                "min_samples_split".into(),
                ParamDistribution::IntRange { lo: 2, hi: 20 },
            );
        },
        ModelKind::GradientBoosting =>
        {
            h.insert(
                "learning_rate".into(),
                ParamDistribution::LogUniform { lo: 0.001, hi: 1.0 },
            );
            h.insert(
                "n_estimators".into(),
                ParamDistribution::IntRange { lo: 50, hi: 500 },
            );
            h.insert(
                "max_depth".into(),
                ParamDistribution::IntRange { lo: 3, hi: 15 },
            );
        },
        ModelKind::NeuralNetwork =>
        {
            h.insert(
                "learning_rate".into(),
                ParamDistribution::LogUniform {
                    lo: 0.0001,
                    hi: 0.1,
                },
            );
            h.insert(
                "hidden_size".into(),
                ParamDistribution::IntRange { lo: 32, hi: 256 },
            );
            h.insert(
                "n_layers".into(),
                ParamDistribution::IntRange { lo: 1, hi: 5 },
            );
            h.insert(
                "epochs".into(),
                ParamDistribution::IntRange { lo: 50, hi: 500 },
            );
        },
    }
    h
}

// =========================================================================
// 5.  Pipeline config — a concrete instantiation
// =========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub preprocessors: Vec<PreprocessorKind>,
    pub model: ModelKind,
    pub params: HashMap<String, f64>,
}

impl PipelineConfig {
    /// Convenience: get a parameter with default fallback.
    pub fn param(&self, name: &str, default: f64) -> f64 {
        self.params.get(name).copied().unwrap_or(default)
    }

    pub fn param_int(&self, name: &str, default: i64) -> i64 {
        self.params.get(name).map(|v| *v as i64).unwrap_or(default)
    }
}

// =========================================================================
// 6.  Metric definitions
// =========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Metric {
    Accuracy,
    Precision,
    Recall,
    F1,
    Mse,
    Mae,
    R2,
}

impl Metric {
    /// All metrics are maximized: higher is better.
    /// For Mse, Mae we return negative value so the "bigger is better" convention holds.
    pub fn is_classification(&self) -> bool {
        matches!(
            self,
            Self::Accuracy | Self::Precision | Self::Recall | Self::F1
        )
    }

    pub fn evaluate(&self, y_true: &[f64], y_pred: &[f64], _is_classification: bool) -> f64 {
        match self
        {
            Self::Accuracy => accuracy(y_true, y_pred),
            Self::Precision => precision(y_true, y_pred),
            Self::Recall => recall(y_true, y_pred),
            Self::F1 => f1_score(y_true, y_pred),
            Self::Mse => -mse(y_true, y_pred),
            Self::Mae => -mae(y_true, y_pred),
            Self::R2 => r2_score(y_true, y_pred),
        }
    }
}

fn accuracy(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let n = y_true.len();
    if n == 0
    {
        return 0.0;
    }
    let correct = y_true
        .iter()
        .zip(y_pred)
        .filter(|(t, p)| (*t - *p).abs() < 0.5)
        .count();
    correct as f64 / n as f64
}

fn precision(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let n = y_true.len();
    if n == 0
    {
        return 0.0;
    }
    // Binarise with threshold 0.5
    let yp_bin: Vec<f64> = y_pred
        .iter()
        .map(|p| if *p >= 0.5 { 1.0 } else { 0.0 })
        .collect();
    let mut tp = 0;
    let mut fp = 0;
    for i in 0..n
    {
        if yp_bin[i] >= 0.5
        {
            if (y_true[i] - 1.0).abs() < 0.5
            {
                tp += 1;
            }
            else
            {
                fp += 1;
            }
        }
    }
    if tp + fp == 0
    {
        return 0.0;
    }
    tp as f64 / (tp + fp) as f64
}

fn recall(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let n = y_true.len();
    if n == 0
    {
        return 0.0;
    }
    let yp_bin: Vec<f64> = y_pred
        .iter()
        .map(|p| if *p >= 0.5 { 1.0 } else { 0.0 })
        .collect();
    let mut tp = 0;
    let mut fn_count = 0;
    for i in 0..n
    {
        if (y_true[i] - 1.0).abs() < 0.5
        {
            if yp_bin[i] >= 0.5
            {
                tp += 1;
            }
            else
            {
                fn_count += 1;
            }
        }
    }
    if tp + fn_count == 0
    {
        return 0.0;
    }
    tp as f64 / (tp + fn_count) as f64
}

fn f1_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let p = precision(y_true, y_pred);
    let r = recall(y_true, y_pred);
    if (p + r).abs() < 1e-12
    {
        return 0.0;
    }
    2.0 * p * r / (p + r)
}

fn mse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let n = y_true.len();
    if n == 0
    {
        return 0.0;
    }
    y_true
        .iter()
        .zip(y_pred)
        .map(|(t, p)| (t - p).powi(2))
        .sum::<f64>()
        / n as f64
}

fn mae(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let n = y_true.len();
    if n == 0
    {
        return 0.0;
    }
    y_true
        .iter()
        .zip(y_pred)
        .map(|(t, p)| (t - p).abs())
        .sum::<f64>()
        / n as f64
}

fn r2_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let n = y_true.len();
    if n == 0
    {
        return 0.0;
    }
    let mean_y = y_true.iter().sum::<f64>() / n as f64;
    let ss_res = y_true
        .iter()
        .zip(y_pred)
        .map(|(t, p)| (t - p).powi(2))
        .sum::<f64>();
    let ss_tot = y_true.iter().map(|t| (t - mean_y).powi(2)).sum::<f64>();
    if ss_tot < 1e-12
    {
        return 1.0;
    }
    1.0 - ss_res / ss_tot
}

// =========================================================================
// 7.  Preprocessing implementations
// =========================================================================

pub struct StandardScaler {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

impl StandardScaler {
    pub fn fit(x: &[Vec<f64>]) -> Self {
        let n_samples = x.len();
        if n_samples == 0
        {
            return Self {
                mean: vec![],
                std: vec![],
            };
        }
        let n_feat = x[0].len();
        let mut mean = vec![0.0; n_feat];
        let mut std = vec![0.0; n_feat];
        for row in x
        {
            for (j, v) in row.iter().enumerate()
            {
                mean[j] += v;
            }
        }
        for v in &mut mean
        {
            *v /= n_samples as f64;
        }
        for row in x
        {
            for (j, v) in row.iter().enumerate()
            {
                std[j] += (v - mean[j]).powi(2);
            }
        }
        for v in &mut std
        {
            *v = (*v / n_samples as f64).sqrt();
            if *v < 1e-12
            {
                *v = 1.0;
            }
        }
        Self { mean, std }
    }

    pub fn transform(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        x.iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(j, v)| (v - self.mean[j]) / self.std[j])
                    .collect()
            })
            .collect()
    }
}

pub struct Normalizer {
    pub norms: Option<Vec<f64>>,
}

impl Normalizer {
    pub fn fit(x: &[Vec<f64>]) -> Self {
        if x.is_empty()
        {
            return Self { norms: None };
        }
        let norms: Vec<f64> = x
            .iter()
            .map(|row| {
                let n2: f64 = row.iter().map(|v| v * v).sum();
                n2.sqrt().max(1e-12)
            })
            .collect();
        Self { norms: Some(norms) }
    }

    pub fn transform(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        x.iter()
            .enumerate()
            .map(|(i, row)| {
                let norm = self.norms.as_ref().map(|n| n[i]).unwrap_or(1.0).max(1e-12);
                row.iter().map(|v| v / norm).collect()
            })
            .collect()
    }
}

pub struct PCA {
    pub components: Vec<Vec<f64>>,
    pub mean: Vec<f64>,
    pub n_components: usize,
}

impl PCA {
    pub fn fit(x: &[Vec<f64>], n_components: usize) -> Self {
        let n_samples = x.len();
        if n_samples == 0
        {
            return Self {
                components: vec![],
                mean: vec![],
                n_components,
            };
        }
        let n_feat = x[0].len();
        let nc = n_components.min(n_feat);
        let mean: Vec<f64> = (0..n_feat)
            .map(|j| x.iter().map(|row| row[j]).sum::<f64>() / n_samples as f64)
            .collect();

        // Covariance matrix (n_feat x n_feat)
        let mut cov = vec![vec![0.0; n_feat]; n_feat];
        for row in x
        {
            let centered: Vec<f64> = row.iter().enumerate().map(|(j, v)| v - mean[j]).collect();
            for i in 0..n_feat
            {
                for j in 0..n_feat
                {
                    cov[i][j] += centered[i] * centered[j];
                }
            }
        }
        for row in &mut cov
        {
            for v in row.iter_mut()
            {
                *v /= (n_samples - 1) as f64;
            }
        }

        // Power iteration for first nc eigenvectors
        let components = power_iteration(&cov, nc);

        Self {
            components,
            mean,
            n_components: nc,
        }
    }

    pub fn transform(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        x.iter()
            .map(|row| {
                let centered: Vec<f64> = row
                    .iter()
                    .enumerate()
                    .map(|(j, v)| v - self.mean[j])
                    .collect();
                self.components
                    .iter()
                    .map(|comp| centered.iter().zip(comp).map(|(c, w)| c * w).sum::<f64>())
                    .collect()
            })
            .collect()
    }
}

fn power_iteration(a: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    let n = a.len();
    if n == 0 || k == 0
    {
        return vec![];
    }
    let mut rng = XorShift64::new(12345);
    // Simple iterative PCA via deflation
    let mut components = Vec::new();
    let mut residual: Vec<Vec<f64>> = a.to_vec();
    for _ in 0..k
    {
        let mut v: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        normalize_vec(&mut v);
        for _ in 0..50
        {
            // v = A * v
            let mut av = vec![0.0; n];
            for i in 0..n
            {
                av[i] = residual[i].iter().zip(&v).map(|(aij, vj)| aij * vj).sum();
            }
            normalize_vec(&mut av);
            v = av;
        }
        // Ensure sign consistency
        if let Some(first) = v.first()
        {
            if *first < 0.0
            {
                for vi in &mut v
                {
                    *vi = -*vi;
                }
            }
        }
        // Rayleigh quotient → eigenvalue
        let mut av = vec![0.0; n];
        for i in 0..n
        {
            av[i] = residual[i].iter().zip(&v).map(|(aij, vj)| aij * vj).sum();
        }
        let lambda = v.iter().zip(&av).map(|(vi, avi)| vi * avi).sum::<f64>();
        // Deflate
        for i in 0..n
        {
            for j in 0..n
            {
                residual[i][j] -= lambda * v[i] * v[j];
            }
        }
        components.push(v);
    }
    components
}

fn normalize_vec(v: &mut [f64]) {
    let n2: f64 = v.iter().map(|x| x * x).sum();
    if n2 < 1e-15
    {
        return;
    }
    let inv = 1.0 / n2.sqrt();
    for x in v.iter_mut()
    {
        *x *= inv;
    }
}

/// PolynomialFeatures: generate all polynomial combinations up to `degree`.
pub struct PolynomialFeatures {
    pub degree: usize,
    pub combinations: Vec<Vec<usize>>, // indices of original features per output column
}

impl PolynomialFeatures {
    pub fn fit(x: &[Vec<f64>], degree: usize) -> Self {
        let n_feat = if x.is_empty() { 0 } else { x[0].len() };
        let combos = generate_combinations(n_feat, degree);
        Self {
            degree,
            combinations: combos,
        }
    }

    pub fn transform(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        x.iter()
            .map(|row| {
                self.combinations
                    .iter()
                    .map(|indices| {
                        let prod: f64 = indices.iter().map(|&idx| row[idx]).product();
                        prod
                    })
                    .collect()
            })
            .collect()
    }
}

fn generate_combinations(n_feat: usize, degree: usize) -> Vec<Vec<usize>> {
    let mut result: Vec<Vec<usize>> = Vec::new();
    if n_feat == 0 || degree == 0
    {
        return result;
    }
    // Degree 1: individual features
    for i in 0..n_feat
    {
        result.push(vec![i]);
    }
    let mut current = result.clone();
    for _d in 2..=degree
    {
        let mut next = Vec::new();
        for combo in &current
        {
            let last = *combo.last().unwrap();
            for i in last..n_feat
            {
                let mut c = combo.clone();
                c.push(i);
                next.push(c);
            }
        }
        result.extend(next.clone());
        current = next;
    }
    result
}

// =========================================================================
// 8.  Model implementations
// =========================================================================

/// A trained model that can predict.
pub trait Model: Send + Sync {
    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64>;
}

// --- Linear model (ridge regression / logistic via sigmoid for classification) ---

pub struct LinearModel {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub is_classification: bool,
    pub reg: f64,
}

impl LinearModel {
    pub fn fit(x: &[Vec<f64>], y: &[f64], is_classification: bool, reg: f64) -> Self {
        let n = x.len();
        if n == 0
        {
            return Self {
                weights: vec![],
                bias: 0.0,
                is_classification,
                reg: 0.0,
            };
        }
        let d = x[0].len();
        // Solve (X^T X + reg*I) w = X^T y  via normal equations
        let mut xtx = vec![vec![0.0; d]; d];
        let mut xty = vec![0.0; d];
        for (row, &yi) in x.iter().zip(y)
        {
            for i in 0..d
            {
                xty[i] += row[i] * yi;
                for j in 0..d
                {
                    xtx[i][j] += row[i] * row[j];
                }
            }
        }
        for i in 0..d
        {
            xtx[i][i] += reg;
        }
        let weights = solve_symmetric(&xtx, &xty);
        // Compute bias as mean residual
        let mut bias = 0.0;
        for (row, &yi) in x.iter().zip(y)
        {
            let pred: f64 = row.iter().zip(&weights).map(|(r, w)| r * w).sum();
            bias += yi - pred;
        }
        bias /= n as f64;
        Self {
            weights,
            bias,
            is_classification,
            reg,
        }
    }
}

impl Model for LinearModel {
    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter()
            .map(|row| {
                let raw: f64 = row
                    .iter()
                    .zip(&self.weights)
                    .map(|(r, w)| r * w)
                    .sum::<f64>()
                    + self.bias;
                if self.is_classification
                {
                    sigmoid(raw)
                }
                else
                {
                    raw
                }
            })
            .collect()
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Cholesky-based solver for symmetric positive-definite systems.
fn solve_symmetric(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    if n == 0
    {
        return vec![];
    }
    // Cholesky: A = L L^T
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n
    {
        for j in 0..=i
        {
            let mut sum = 0.0;
            for k in 0..j
            {
                sum += l[i][k] * l[j][k];
            }
            if i == j
            {
                let val = a[i][i] - sum;
                l[i][j] = val.sqrt().max(1e-12);
            }
            else
            {
                l[i][j] = (a[i][j] - sum) / l[j][j];
            }
        }
    }
    // Forward: L y = b
    let mut y = vec![0.0; n];
    for i in 0..n
    {
        let mut sum = 0.0;
        for k in 0..i
        {
            sum += l[i][k] * y[k];
        }
        y[i] = (b[i] - sum) / l[i][i];
    }
    // Backward: L^T x = y
    let mut x = vec![0.0; n];
    for i in (0..n).rev()
    {
        let mut sum = 0.0;
        for k in (i + 1)..n
        {
            sum += l[k][i] * x[k];
        }
        x[i] = (y[i] - sum) / l[i][i];
    }
    x
}

// --- RandomForest ---

pub struct RandomForest {
    pub trees: Vec<DecisionTree>,
}

impl RandomForest {
    pub fn fit(
        x: &[Vec<f64>],
        y: &[f64],
        n_trees: usize,
        max_depth: usize,
        min_samples_split: usize,
        is_classification: bool,
        rng: &mut XorShift64,
    ) -> Self {
        let n = x.len();
        let trees: Vec<DecisionTree> = (0..n_trees)
            .map(|_| {
                // Bootstrap sample
                let (bx, by) = bootstrap_sample(x, y, n, rng);
                DecisionTree::fit(
                    &bx,
                    &by,
                    max_depth,
                    min_samples_split,
                    is_classification,
                    rng,
                )
            })
            .collect();
        Self { trees }
    }
}

impl Model for RandomForest {
    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let mut all_preds: Vec<Vec<f64>> = self.trees.iter().map(|t| t.predict(x)).collect();
        let n_samples = x.len();
        (0..n_samples)
            .map(|i| {
                let sum: f64 = all_preds.iter_mut().map(|p| p[i]).sum();
                sum / self.trees.len() as f64
            })
            .collect()
    }
}

// --- GradientBoosting ---

pub struct GradientBoosting {
    pub trees: Vec<DecisionTree>,
    pub learning_rate: f64,
    pub initial_pred: f64,
    pub is_classification: bool,
}

impl GradientBoosting {
    pub fn fit(
        x: &[Vec<f64>],
        y: &[f64],
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        is_classification: bool,
        rng: &mut XorShift64,
    ) -> Self {
        let n = x.len();
        let initial_pred = y.iter().sum::<f64>() / n as f64;
        let mut residuals: Vec<f64> = y.iter().map(|yi| yi - initial_pred).collect();
        let mut trees = Vec::new();

        for _ in 0..n_estimators
        {
            let tree = DecisionTree::fit(x, &residuals, max_depth, 5, false, rng);
            let preds = tree.predict(x);
            for (r, p) in residuals.iter_mut().zip(&preds)
            {
                *r -= learning_rate * p;
            }
            trees.push(tree);
        }

        Self {
            trees,
            learning_rate,
            initial_pred,
            is_classification,
        }
    }
}

impl Model for GradientBoosting {
    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter()
            .map(|row| {
                let mut pred = self.initial_pred;
                for tree in &self.trees
                {
                    let row_vec = vec![row.clone()];
                    pred += self.learning_rate * tree.predict(&row_vec)[0];
                }
                if self.is_classification
                {
                    sigmoid(pred)
                }
                else
                {
                    pred
                }
            })
            .collect()
    }
}

// --- DecisionTree (used by RF and GBT) ---

pub struct DecisionTree {
    root: TreeNode,
}

enum TreeNode {
    Leaf(f64),
    Split {
        feature: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

impl DecisionTree {
    fn fit(
        x: &[Vec<f64>],
        y: &[f64],
        max_depth: usize,
        min_samples_split: usize,
        is_classification: bool,
        rng: &mut XorShift64,
    ) -> Self {
        let _ = is_classification;
        Self {
            root: build_tree(x, y, 0, max_depth, min_samples_split, rng),
        }
    }
}

impl Model for DecisionTree {
    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|row| traverse(&self.root, row)).collect()
    }
}

fn traverse(node: &TreeNode, x: &[f64]) -> f64 {
    match node
    {
        TreeNode::Leaf(v) => *v,
        TreeNode::Split {
            feature,
            threshold,
            left,
            right,
        } =>
        {
            if x[*feature] <= *threshold
            {
                traverse(left, x)
            }
            else
            {
                traverse(right, x)
            }
        },
    }
}

fn build_tree(
    x: &[Vec<f64>],
    y: &[f64],
    depth: usize,
    max_depth: usize,
    min_samples_split: usize,
    rng: &mut XorShift64,
) -> TreeNode {
    let n = x.len();
    if n == 0
    {
        return TreeNode::Leaf(0.0);
    }
    if depth >= max_depth || n < min_samples_split
    {
        return TreeNode::Leaf(y.iter().sum::<f64>() / n as f64);
    }

    let d = x[0].len();
    let mut best_feature = 0;
    let mut best_threshold = 0.0;
    let mut best_score = f64::INFINITY;
    let n_features = (d as f64).sqrt().ceil() as usize;

    for _ in 0..n_features
    {
        let fi = rng.usize(d);
        // Find best split on this feature
        let mut vals: Vec<(f64, f64)> = x.iter().map(|r| r[fi]).zip(y.iter().copied()).collect();
        vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        for i in 1..vals.len()
        {
            let threshold = (vals[i - 1].0 + vals[i].0) / 2.0;
            if (vals[i].0 - vals[i - 1].0).abs() < 1e-12
            {
                continue;
            }
            let mut left_y = vec![];
            let mut right_y = vec![];
            for &(vx, vy) in &vals
            {
                if vx <= threshold
                {
                    left_y.push(vy);
                }
                else
                {
                    right_y.push(vy);
                }
            }
            if left_y.len() < min_samples_split || right_y.len() < min_samples_split
            {
                continue;
            }
            let score = variance(&left_y) + variance(&right_y);
            if score < best_score
            {
                best_score = score;
                best_feature = fi;
                best_threshold = threshold;
            }
        }
    }

    if best_score == f64::INFINITY
    {
        return TreeNode::Leaf(y.iter().sum::<f64>() / n as f64);
    }

    let mut left_x = Vec::new();
    let mut left_y = Vec::new();
    let mut right_x = Vec::new();
    let mut right_y = Vec::new();
    for (row, &yi) in x.iter().zip(y)
    {
        if row[best_feature] <= best_threshold
        {
            left_x.push(row.clone());
            left_y.push(yi);
        }
        else
        {
            right_x.push(row.clone());
            right_y.push(yi);
        }
    }

    TreeNode::Split {
        feature: best_feature,
        threshold: best_threshold,
        left: Box::new(build_tree(
            &left_x,
            &left_y,
            depth + 1,
            max_depth,
            min_samples_split,
            rng,
        )),
        right: Box::new(build_tree(
            &right_x,
            &right_y,
            depth + 1,
            max_depth,
            min_samples_split,
            rng,
        )),
    }
}

fn variance(y: &[f64]) -> f64 {
    if y.len() <= 1
    {
        return 0.0;
    }
    let mean = y.iter().sum::<f64>() / y.len() as f64;
    y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (y.len() - 1) as f64
}

fn bootstrap_sample(
    x: &[Vec<f64>],
    y: &[f64],
    n: usize,
    rng: &mut XorShift64,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut bx = Vec::with_capacity(n);
    let mut by = Vec::with_capacity(n);
    for _ in 0..n
    {
        let idx = rng.usize(n);
        bx.push(x[idx].clone());
        by.push(y[idx]);
    }
    (bx, by)
}

// --- NeuralNetwork ---

pub struct NeuralNetwork {
    layers: Vec<(Vec<Vec<f64>>, Vec<f64>)>, // (weights, biases) per layer
    activation_fn: fn(f64) -> f64,
    is_classification: bool,
}

impl NeuralNetwork {
    pub fn fit(
        x: &[Vec<f64>],
        y: &[f64],
        hidden_layers: &[usize],
        learning_rate: f64,
        epochs: usize,
        is_classification: bool,
        rng: &mut XorShift64,
    ) -> Self {
        let n = x.len();
        if n == 0 || hidden_layers.is_empty()
        {
            return Self {
                layers: vec![],
                activation_fn: relu,
                is_classification,
            };
        }
        let input_dim = x[0].len();
        let output_dim = 1;

        // Build layer sizes: input → hidden[0] → … → hidden[n-1] → output
        let mut sizes = vec![input_dim];
        sizes.extend_from_slice(hidden_layers);
        sizes.push(output_dim);

        // Init weights (He init) and biases
        let mut layers: Vec<(Vec<Vec<f64>>, Vec<f64>)> = Vec::new();
        for k in 0..(sizes.len() - 1)
        {
            let fan_in = sizes[k];
            let fan_out = sizes[k + 1];
            let std = (2.0 / fan_in as f64).sqrt();
            let weights: Vec<Vec<f64>> = (0..fan_out)
                .map(|_| (0..fan_in).map(|_| rng.normal() * std).collect())
                .collect();
            let biases: Vec<f64> = (0..fan_out).map(|_| 0.0).collect();
            layers.push((weights, biases));
        }

        // SGD training
        for _epoch in 0..epochs
        {
            for i in 0..n
            {
                // Forward pass
                let mut activations = vec![x[i].clone()];
                let mut zs = Vec::new();
                for (layer_idx, (weights, biases)) in layers.iter().enumerate()
                {
                    let prev = &activations[activations.len() - 1];
                    let mut z = biases.clone();
                    for (oi, w_row) in weights.iter().enumerate()
                    {
                        for (ii, w) in w_row.iter().enumerate()
                        {
                            z[oi] += w * prev[ii];
                        }
                    }
                    zs.push(z.clone());
                    let a: Vec<f64> = if layer_idx == layers.len() - 1
                    {
                        if is_classification
                        {
                            z.iter().map(|&v| sigmoid(v)).collect()
                        }
                        else
                        {
                            z.clone()
                        }
                    }
                    else
                    {
                        z.iter().map(|&v| relu(v)).collect()
                    };
                    activations.push(a);
                }

                // Backprop
                let output = &activations[activations.len() - 1];
                let target = y[i];
                let mut delta = if is_classification
                {
                    vec![(output[0] - target) * output[0] * (1.0 - output[0])]
                }
                else
                {
                    vec![output[0] - target]
                };

                for layer_idx in (0..layers.len()).rev()
                {
                    let prev_act = &activations[layer_idx];
                    let lr = learning_rate;

                    // Compute gradients and apply with gradient clipping
                    {
                        let (weights, biases) = &mut layers[layer_idx];
                        for oi in 0..weights.len()
                        {
                            let grad_b = delta[oi];
                            let clipped_b = grad_b.clamp(-1.0, 1.0);
                            biases[oi] -= lr * clipped_b;
                            for ii in 0..weights[oi].len()
                            {
                                let grad_w = delta[oi] * prev_act[ii];
                                let clipped_w = grad_w.clamp(-1.0, 1.0);
                                weights[oi][ii] -= lr * clipped_w;
                            }
                        }
                    }

                    // Propagate delta to previous layer
                    if layer_idx > 0
                    {
                        let (cur_weights, _) = &layers[layer_idx];
                        let prev_a = &activations[layer_idx];
                        let mut new_delta = vec![0.0; prev_a.len()];
                        for pi in 0..prev_a.len()
                        {
                            let mut sum = 0.0;
                            for oi in 0..delta.len()
                            {
                                sum += delta[oi] * cur_weights[oi][pi];
                            }
                            // ReLU derivative
                            new_delta[pi] = sum * if prev_a[pi] > 0.0 { 1.0 } else { 0.0 };
                        }
                        delta = new_delta;
                    }
                }
            }
        }

        Self {
            layers,
            activation_fn: relu,
            is_classification,
        }
    }
}

impl Model for NeuralNetwork {
    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter()
            .map(|row| {
                let mut a = row.clone();
                for (layer_idx, (weights, biases)) in self.layers.iter().enumerate()
                {
                    let mut z = biases.clone();
                    for (oi, w_row) in weights.iter().enumerate()
                    {
                        for (ii, w) in w_row.iter().enumerate()
                        {
                            z[oi] += w * a[ii];
                        }
                    }
                    a = if layer_idx == self.layers.len() - 1
                    {
                        if self.is_classification
                        {
                            z.iter().map(|&v| sigmoid(v)).collect()
                        }
                        else
                        {
                            z
                        }
                    }
                    else
                    {
                        z.iter().map(|&v| (self.activation_fn)(v)).collect()
                    };
                }
                a[0]
            })
            .collect()
    }
}

fn relu(x: f64) -> f64 {
    x.max(0.0)
}

// =========================================================================
// 9.  Pipeline runner
// =========================================================================

/// Result of evaluating one pipeline on a fold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoldResult {
    pub fold_index: usize,
    pub train_metric: f64,
    pub val_metric: f64,
}

/// Full cross-validation result for one pipeline config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CvResult {
    pub folds: Vec<FoldResult>,
    pub mean_train: f64,
    pub mean_val: f64,
    pub std_val: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
}

impl CvResult {
    pub fn compute(folds: Vec<FoldResult>) -> Self {
        let n = folds.len() as f64;
        let mean_train = folds.iter().map(|f| f.train_metric).sum::<f64>() / n;
        let mean_val = folds.iter().map(|f| f.val_metric).sum::<f64>() / n;
        let var_val = folds
            .iter()
            .map(|f| (f.val_metric - mean_val).powi(2))
            .sum::<f64>()
            / (n - 1.0).max(1.0);
        let std_val = var_val.sqrt();
        // t_0.025 for appropriate df; use 1.96 for n >= 30, otherwise rounded
        let t_crit = if n >= 30.0
        {
            1.96
        }
        else
        {
            t_critical(n as usize - 1)
        };
        let se = std_val / n.sqrt();
        Self {
            mean_train,
            mean_val,
            std_val,
            ci_lower: mean_val - t_crit * se,
            ci_upper: mean_val + t_crit * se,
            folds,
        }
    }
}

fn t_critical(_df: usize) -> f64 {
    // Simplified: use 2.0 for small samples
    2.0
}

/// Apply preprocessing steps to a dataset.
pub fn apply_preprocessing(
    x: &[Vec<f64>],
    steps: &[PreprocessorKind],
    fitted: Option<&Vec<Box<dyn Preprocessor>>>,
) -> (Vec<Vec<f64>>, Vec<Box<dyn Preprocessor>>) {
    let mut current = x.to_vec();
    let mut processors: Vec<Box<dyn Preprocessor>> = Vec::new();
    if let Some(fit) = fitted
    {
        for proc in fit.iter()
        {
            current = proc.transform(&current);
        }
        return (current, vec![]);
    }
    for step in steps
    {
        let proc: Box<dyn Preprocessor> = match step
        {
            PreprocessorKind::StandardScaler =>
            {
                let p = StandardScaler::fit(&current);
                current = p.transform(&current);
                Box::new(FittedScaler(p))
            },
            PreprocessorKind::Normalizer =>
            {
                let p = Normalizer::fit(&current);
                current = p.transform(&current);
                Box::new(FittedNormalizer(p))
            },
            PreprocessorKind::PCA { n_components } =>
            {
                let p = PCA::fit(&current, *n_components);
                current = p.transform(&current);
                Box::new(FittedPca(p))
            },
            PreprocessorKind::PolynomialFeatures { degree } =>
            {
                let p = PolynomialFeatures::fit(&current, *degree);
                current = p.transform(&current);
                Box::new(FittedPoly(p))
            },
        };
        processors.push(proc);
    }
    (current, processors)
}

pub trait Preprocessor: Send + Sync {
    fn transform(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>>;
}

struct FittedScaler(StandardScaler);
impl Preprocessor for FittedScaler {
    fn transform(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        self.0.transform(x)
    }
}
struct FittedNormalizer(Normalizer);
impl Preprocessor for FittedNormalizer {
    fn transform(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        self.0.transform(x)
    }
}
struct FittedPca(PCA);
impl Preprocessor for FittedPca {
    fn transform(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        self.0.transform(x)
    }
}
struct FittedPoly(PolynomialFeatures);
impl Preprocessor for FittedPoly {
    fn transform(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        self.0.transform(x)
    }
}

/// Train a model given a pipeline config and preprocessed data.
pub fn train_model(
    config: &PipelineConfig,
    x: &[Vec<f64>],
    y: &[f64],
    is_classification: bool,
    rng: &mut XorShift64,
) -> Box<dyn Model> {
    match config.model
    {
        ModelKind::Linear =>
        {
            let reg = config.param("regularization", 0.1);
            Box::new(LinearModel::fit(x, y, is_classification, reg))
        },
        ModelKind::RandomForest =>
        {
            let n_trees = config.param_int("n_trees", 100) as usize;
            let max_depth = config.param_int("max_depth", 10) as usize;
            let min_split = config.param_int("min_samples_split", 2) as usize;
            Box::new(RandomForest::fit(
                x,
                y,
                n_trees,
                max_depth,
                min_split,
                is_classification,
                rng,
            ))
        },
        ModelKind::GradientBoosting =>
        {
            let lr = config.param("learning_rate", 0.1);
            let n_est = config.param_int("n_estimators", 100) as usize;
            let max_depth = config.param_int("max_depth", 5) as usize;
            Box::new(GradientBoosting::fit(
                x,
                y,
                n_est,
                lr,
                max_depth,
                is_classification,
                rng,
            ))
        },
        ModelKind::NeuralNetwork =>
        {
            let lr = config.param("learning_rate", 0.01);
            let hidden = config.param_int("hidden_size", 64) as usize;
            let n_layers = config.param_int("n_layers", 2) as usize;
            let epochs = config.param_int("epochs", 100) as usize;
            let hidden_layers: Vec<usize> = (0..n_layers).map(|_| hidden).collect();
            Box::new(NeuralNetwork::fit(
                x,
                y,
                &hidden_layers,
                lr,
                epochs,
                is_classification,
                rng,
            ))
        },
    }
}

/// Run k-fold cross-validation for a single pipeline config.
pub fn cross_validate(
    config: &PipelineConfig,
    x: &[Vec<f64>],
    y: &[f64],
    k: usize,
    metric: Metric,
    is_classification: bool,
    rng: &mut XorShift64,
) -> CvResult {
    let n = x.len();
    let fold_size = n.div_ceil(k);
    let indices = shuffled_indices(n, rng);
    let mut folds = Vec::new();

    for fold_idx in 0..k
    {
        let start = fold_idx * fold_size;
        let end = (start + fold_size).min(n);

        let train_idx: Vec<usize> = indices
            .iter()
            .enumerate()
            .filter(|(i, _)| *i < start || *i >= end)
            .map(|(_, &idx)| idx)
            .collect();
        let val_idx: Vec<usize> = indices[start..end].to_vec();

        // Apply preprocessing fit on train
        let train_x: Vec<Vec<f64>> = train_idx.iter().map(|&i| x[i].clone()).collect();
        let train_y: Vec<f64> = train_idx.iter().map(|&i| y[i]).collect();
        let val_x: Vec<Vec<f64>> = val_idx.iter().map(|&i| x[i].clone()).collect();
        let val_y: Vec<f64> = val_idx.iter().map(|&i| y[i]).collect();

        let (proc_train, procs) = apply_preprocessing(&train_x, &config.preprocessors, None);
        let proc_val = apply_preprocessing(&val_x, &config.preprocessors, Some(&procs)).0;

        let model = train_model(config, &proc_train, &train_y, is_classification, rng);

        let train_pred = model.predict(&proc_train);
        let val_pred = model.predict(&proc_val);

        let train_metric = metric.evaluate(&train_y, &train_pred, is_classification);
        let val_metric = metric.evaluate(&val_y, &val_pred, is_classification);

        folds.push(FoldResult {
            fold_index: fold_idx,
            train_metric,
            val_metric,
        });
    }

    CvResult::compute(folds)
}

fn shuffled_indices(n: usize, rng: &mut XorShift64) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    for i in (1..n).rev()
    {
        let j = rng.usize(i + 1);
        v.swap(i, j);
    }
    v
}

/// Time series cross-validation (forward chaining).
pub fn time_series_cv(
    config: &PipelineConfig,
    x: &[Vec<f64>],
    y: &[f64],
    n_splits: usize,
    metric: Metric,
    is_classification: bool,
    rng: &mut XorShift64,
) -> CvResult {
    let n = x.len();
    let min_train = n / (n_splits + 1);
    let mut folds = Vec::new();

    for fold_idx in 0..n_splits
    {
        let split = min_train + fold_idx * (n - min_train) / n_splits;
        let train_x: Vec<Vec<f64>> = x[..split].to_vec();
        let train_y: Vec<f64> = y[..split].to_vec();
        let val_x: Vec<Vec<f64>> = x[split..].to_vec();
        let val_y: Vec<f64> = y[split..].to_vec();

        let (proc_train, procs) = apply_preprocessing(&train_x, &config.preprocessors, None);
        let proc_val = apply_preprocessing(&val_x, &config.preprocessors, Some(&procs)).0;

        let model = train_model(config, &proc_train, &train_y, is_classification, rng);
        let train_pred = model.predict(&proc_train);
        let val_pred = model.predict(&proc_val);

        let train_metric = metric.evaluate(&train_y, &train_pred, is_classification);
        let val_metric = metric.evaluate(&val_y, &val_pred, is_classification);

        folds.push(FoldResult {
            fold_index: fold_idx,
            train_metric,
            val_metric,
        });
    }

    CvResult::compute(folds)
}

// =========================================================================
// 10. Bayesian Optimization (simplified Gaussian Process)
// =========================================================================

/// Matern 5/2 kernel
fn matern52_kernel(x1: &[f64], x2: &[f64], length_scale: f64) -> f64 {
    let dist = euclidean(x1, x2) / length_scale;
    let sqrt5 = 5.0_f64.sqrt();
    let d = sqrt5 * dist;
    (1.0 + d + d * d / 3.0) * (-d).exp()
}

fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(ai, bi)| (ai - bi).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Simplified Gaussian Process regressor
pub struct GaussianProcess {
    pub x_train: Vec<Vec<f64>>,
    pub y_train: Vec<f64>,
    pub alpha: Vec<f64>, // K^{-1} y
    pub length_scale: f64,
    pub noise: f64,
}

impl GaussianProcess {
    pub fn fit(x: &[Vec<f64>], y: &[f64], length_scale: f64, noise: f64) -> Self {
        let n = x.len();
        if n == 0
        {
            return Self {
                x_train: vec![],
                y_train: vec![],
                alpha: vec![],
                length_scale,
                noise,
            };
        }

        // Build kernel matrix K + noise * I
        let mut k = vec![vec![0.0; n]; n];
        for i in 0..n
        {
            for j in 0..n
            {
                k[i][j] = matern52_kernel(&x[i], &x[j], length_scale);
                if i == j
                {
                    k[i][j] += noise;
                }
            }
        }

        let alpha = solve_symmetric(&k, y);

        Self {
            x_train: x.to_vec(),
            y_train: y.to_vec(),
            alpha,
            length_scale,
            noise,
        }
    }

    pub fn predict(&self, x_star: &[f64]) -> (f64, f64) {
        if self.x_train.is_empty()
        {
            return (0.0, 1.0);
        }
        let n = self.x_train.len();
        // k_* = kernel(x_train[i], x_star)
        let k_star: Vec<f64> = self
            .x_train
            .iter()
            .map(|xi| matern52_kernel(xi, x_star, self.length_scale))
            .collect();

        // mean = k_*^T * alpha
        let mean: f64 = k_star.iter().zip(&self.alpha).map(|(k, a)| k * a).sum();

        // variance = k(x_star, x_star) - k_*^T K^{-1} k_*
        // K^{-1} k_* = solve(K, k_*). We recompute K and solve.
        let mut k_mat = vec![vec![0.0; n]; n];
        for i in 0..n
        {
            for j in 0..n
            {
                k_mat[i][j] =
                    matern52_kernel(&self.x_train[i], &self.x_train[j], self.length_scale);
                if i == j
                {
                    k_mat[i][j] += self.noise;
                }
            }
        }
        let v = solve_symmetric(&k_mat, &k_star);
        let k_ss = matern52_kernel(x_star, x_star, self.length_scale) + self.noise;
        let var = (k_ss - k_star.iter().zip(&v).map(|(ks, vi)| ks * vi).sum::<f64>()).max(1e-12);

        (mean, var)
    }
}

/// Expected Improvement acquisition function
pub fn expected_improvement(
    gp: &GaussianProcess,
    x_candidate: &[f64],
    y_best: f64,
    xi: f64,
) -> f64 {
    let (mu, var) = gp.predict(x_candidate);
    let sigma = var.sqrt();
    if sigma < 1e-12
    {
        return 0.0;
    }
    let z = (mu - y_best - xi) / sigma;
    let cdf_z = normal_cdf(z);
    let pdf_z = normal_pdf(z);
    (mu - y_best - xi) * cdf_z + sigma * pdf_z
}

fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
}

/// Error function approximation (Abramowitz & Stegun)
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let t = 1.0 / (1.0 + p * x);
    sign * (1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp())
}

/// Bayesian optimization over a continuous search space defined by bounds.
pub fn bayesian_optimize(
    f: &mut dyn FnMut(&[f64]) -> f64,
    bounds: &[(f64, f64)], // (lo, hi) per dimension
    max_iter: usize,
    initial_samples: usize,
    rng: &mut XorShift64,
) -> (Vec<f64>, f64) {
    let d = bounds.len();
    let mut x_obs: Vec<Vec<f64>> = Vec::new();
    let mut y_obs: Vec<f64> = Vec::new();

    // Initial random samples
    for _ in 0..initial_samples
    {
        let x: Vec<f64> = bounds
            .iter()
            .map(|(lo, hi)| rng.uniform(*lo, *hi))
            .collect();
        let y = f(&x);
        x_obs.push(x);
        y_obs.push(y);
    }

    // Optimization loop
    for _ in 0..max_iter.saturating_sub(initial_samples)
    {
        let gp = GaussianProcess::fit(&x_obs, &y_obs, 0.5, 1e-6);
        let y_best = y_obs
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(0.0);

        // Gradient ascent to find argmax EI
        let mut best_x = vec![0.0; d];
        let mut best_ei = f64::NEG_INFINITY;

        // Multi-start local optimization
        for _ in 0..20
        {
            let mut x = random_point(bounds, rng);
            for _ in 0..30
            {
                // Numerical gradient of EI w.r.t. x
                let mut grad = vec![0.0; d];
                let ei_cur = expected_improvement(&gp, &x, y_best, 0.01);
                let eps = 1e-4;
                for j in 0..d
                {
                    let mut x2 = x.clone();
                    x2[j] = (x2[j] + eps).clamp(bounds[j].0, bounds[j].1);
                    let ei_2 = expected_improvement(&gp, &x2, y_best, 0.01);
                    grad[j] = (ei_2 - ei_cur) / eps;
                }
                // Gradient ascent step
                let step_size = 0.05;
                for j in 0..d
                {
                    x[j] = (x[j] + step_size * grad[j]).clamp(bounds[j].0, bounds[j].1);
                }
            }
            let ei = expected_improvement(&gp, &x, y_best, 0.01);
            if ei > best_ei
            {
                best_ei = ei;
                best_x = x;
            }
        }

        let y = f(&best_x);
        x_obs.push(best_x);
        y_obs.push(y);
    }

    // Return best observed point
    let best_idx = y_obs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    (x_obs[best_idx].clone(), y_obs[best_idx])
}

fn random_point(bounds: &[(f64, f64)], rng: &mut XorShift64) -> Vec<f64> {
    bounds
        .iter()
        .map(|(lo, hi)| rng.uniform(*lo, *hi))
        .collect()
}

// =========================================================================
// 11. Hyperparameter Optimizer
// =========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub max_iterations: usize,
    pub cv_folds: usize,
    pub search_budget: usize,
    pub seed: u64,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            cv_folds: 5,
            search_budget: 50,
            seed: 0x5C12_3E70,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialResult {
    pub config: PipelineConfig,
    pub cv_result: CvResult,
    pub score: f64,
}

pub struct HyperOptimizer {
    config: OptimizerConfig,
    rng: RefCell<XorShift64>,
}

impl HyperOptimizer {
    pub fn new(cfg: OptimizerConfig) -> Self {
        let rng = XorShift64::new(cfg.seed);
        Self {
            config: cfg,
            rng: RefCell::new(rng),
        }
    }

    /// Trivial random search over hyperparameters.
    pub fn random_search(
        &self,
        pipeline: &PipelineTemplate,
        x: &[Vec<f64>],
        y: &[f64],
        metric: Metric,
        is_classification: bool,
    ) -> Vec<TrialResult> {
        let mut rng = self.rng.borrow_mut();
        let candidates = pipeline.generate_candidates(self.config.search_budget, &mut rng);
        candidates
            .iter()
            .map(|cfg| {
                let cv = cross_validate(
                    cfg,
                    x,
                    y,
                    self.config.cv_folds,
                    metric,
                    is_classification,
                    &mut rng,
                );
                TrialResult {
                    config: cfg.clone(),
                    score: cv.mean_val,
                    cv_result: cv,
                }
            })
            .collect()
    }

    /// Grid search over hyperparameters.
    pub fn grid_search(
        &self,
        pipeline: &PipelineTemplate,
        x: &[Vec<f64>],
        y: &[f64],
        metric: Metric,
        is_classification: bool,
    ) -> Vec<TrialResult> {
        let mut rng = self.rng.borrow_mut();

        // Build grid
        let dims: Vec<ParamGridDim> = pipeline
            .hyperparameters
            .iter()
            .map(|(name, dist)| ParamGridDim {
                name: name.clone(),
                values: dist.grid_points(5), // 5 points per param
            })
            .collect();

        let configs = cartesian_product(&dims, pipeline);
        let n = configs.len().min(self.config.search_budget);
        configs[..n]
            .iter()
            .map(|cfg| {
                let cv = cross_validate(
                    cfg,
                    x,
                    y,
                    self.config.cv_folds,
                    metric,
                    is_classification,
                    &mut rng,
                );
                TrialResult {
                    config: cfg.clone(),
                    score: cv.mean_val,
                    cv_result: cv,
                }
            })
            .collect()
    }

    /// Bayesian hyperparameter optimization.
    pub fn bayesian_opt(
        &self,
        pipeline: &PipelineTemplate,
        x: &[Vec<f64>],
        y: &[f64],
        metric: Metric,
        is_classification: bool,
    ) -> Vec<TrialResult> {
        let seed = self.config.seed;

        // Build bounds for continuous params
        let param_names: Vec<String> = pipeline.hyperparameters.keys().cloned().collect();
        let bounds: Vec<(f64, f64)> = param_names
            .iter()
            .map(|name| match &pipeline.hyperparameters[name]
            {
                ParamDistribution::Uniform { lo, hi } => (*lo, *hi),
                ParamDistribution::LogUniform { lo, hi } => (*lo, *hi),
                ParamDistribution::IntRange { lo, hi } => (*lo as f64, *hi as f64),
                ParamDistribution::Categorical { .. } => (0.0, 1.0),
            })
            .collect();

        if bounds.is_empty()
        {
            return self.random_search(pipeline, x, y, metric, is_classification);
        }

        let inner_rng = RefCell::new(XorShift64::new(seed.wrapping_add(999)));
        let mut objective = |params: &[f64]| -> f64 {
            let mut config_params = HashMap::new();
            for (name, &v) in param_names.iter().zip(params)
            {
                let dist = &pipeline.hyperparameters[name];
                let val = match dist
                {
                    ParamDistribution::IntRange { .. } => v.round(),
                    _ => v,
                };
                config_params.insert(name.clone(), val);
            }
            let config = PipelineConfig {
                preprocessors: pipeline.preprocessors.clone(),
                model: pipeline.model.clone(),
                params: config_params,
            };
            let cv = cross_validate(
                &config,
                x,
                y,
                self.config.cv_folds,
                metric,
                is_classification,
                &mut inner_rng.borrow_mut(),
            );
            cv.mean_val
        };

        let initial = self.config.search_budget / 2;
        let max_iter = self.config.search_budget;
        let mut rng = XorShift64::new(seed.wrapping_add(888));
        let (best_params, best_score) =
            bayesian_optimize(&mut objective, &bounds, max_iter, initial.max(3), &mut rng);

        let mut best_params_map = HashMap::new();
        for (name, &v) in param_names.iter().zip(&best_params)
        {
            let dist = &pipeline.hyperparameters[name];
            let val = match dist
            {
                ParamDistribution::IntRange { .. } => v.round(),
                _ => v,
            };
            best_params_map.insert(name.clone(), val);
        }
        let best_config = PipelineConfig {
            preprocessors: pipeline.preprocessors.clone(),
            model: pipeline.model.clone(),
            params: best_params_map,
        };

        let mut rng2 = XorShift64::new(seed.wrapping_add(777));
        let best_cv = cross_validate(
            &best_config,
            x,
            y,
            self.config.cv_folds,
            metric,
            is_classification,
            &mut rng2,
        );

        vec![TrialResult {
            config: best_config,
            score: best_score,
            cv_result: best_cv,
        }]
    }
}

/// Helper struct for parametric grid search.
struct ParamGridDim {
    name: String,
    values: Vec<f64>,
}

/// Cartesian product of parameter grids into PipelineConfigs
fn cartesian_product(dims: &[ParamGridDim], template: &PipelineTemplate) -> Vec<PipelineConfig> {
    if dims.is_empty()
    {
        return vec![PipelineConfig {
            preprocessors: template.preprocessors.clone(),
            model: template.model.clone(),
            params: HashMap::new(),
        }];
    }

    let mut result: Vec<HashMap<String, f64>> = vec![HashMap::new()];
    for dim in dims
    {
        let mut next = Vec::new();
        for existing in &result
        {
            for &val in &dim.values
            {
                let mut m = existing.clone();
                m.insert(dim.name.clone(), val);
                next.push(m);
            }
        }
        result = next;
    }

    result
        .into_iter()
        .map(|params| PipelineConfig {
            preprocessors: template.preprocessors.clone(),
            model: template.model.clone(),
            params,
        })
        .collect()
}

// =========================================================================
// 12. Model Selector
// =========================================================================

/// Paired t-test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTestResult {
    pub t_statistic: f64,
    pub p_value: f64,
    pub is_significant: bool, // alpha = 0.05
    pub winner: String,       // "model_a", "model_b", or "tie"
}

/// Simplified paired t-test comparing two sets of per-fold scores.
pub fn paired_t_test(a: &[f64], b: &[f64]) -> TTestResult {
    let n = a.len().min(b.len());
    if n < 2
    {
        return TTestResult {
            t_statistic: 0.0,
            p_value: 1.0,
            is_significant: false,
            winner: "tie".into(),
        };
    }

    let diffs: Vec<f64> = a.iter().zip(b).map(|(ai, bi)| ai - bi).take(n).collect();
    let mean_diff = diffs.iter().sum::<f64>() / n as f64;
    let var_diff: f64 = diffs.iter().map(|d| (d - mean_diff).powi(2)).sum::<f64>() / (n - 1) as f64;
    let se = var_diff.sqrt() / (n as f64).sqrt();

    if se < 1e-15
    {
        return TTestResult {
            t_statistic: 0.0,
            p_value: 1.0,
            is_significant: false,
            winner: "tie".into(),
        };
    }

    let t = mean_diff / se;
    // Approximate two-tailed p-value using the t-distribution
    let df = (n - 1) as f64;
    let p = 2.0 * t_distribution_cdf(-t.abs(), df);

    let winner = if p < 0.05
    {
        if mean_diff > 0.0
        {
            "model_a"
        }
        else
        {
            "model_b"
        }
    }
    else
    {
        "tie"
    };

    TTestResult {
        t_statistic: t,
        p_value: p,
        is_significant: p < 0.05,
        winner: winner.to_string(),
    }
}

/// Simplified t-distribution CDF
fn t_distribution_cdf(t: f64, df: f64) -> f64 {
    let x = df / (df + t * t);
    let beta = beta_inc(df / 2.0, 0.5, x) / 2.0;
    if t >= 0.0 { 1.0 - beta } else { beta }
}

/// Simplified incomplete beta function (regularized)
fn beta_inc(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0
    {
        return 0.0;
    }
    if x >= 1.0
    {
        return 1.0;
    }
    // Continued fraction representation
    let max_iter = 200;
    let eps = 3e-16;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    d = if d.abs() < 1e-30 { 1e-30 } else { d };
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=max_iter
    {
        let m = m as f64;
        let m2 = 2.0 * m;
        let mut aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        d = if d.abs() < 1e-30 { 1e-30 } else { d };
        c = 1.0 + aa / c;
        c = if c.abs() < 1e-30 { 1e-30 } else { c };
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        d = if d.abs() < 1e-30 { 1e-30 } else { d };
        c = 1.0 + aa / c;
        c = if c.abs() < 1e-30 { 1e-30 } else { c };
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < eps
        {
            break;
        }
    }

    let t = a.ln() + b.ln() + a * x.ln() + b * (1.0 - x).ln()
        - (a + b).ln()
        - ln_gamma(a)
        - ln_gamma(b)
        + ln_gamma(a + b);
    let front = t.exp();
    front * h / a
}

/// Log Gamma approximation (Stirling)
fn ln_gamma(z: f64) -> f64 {
    if z < 0.5
    {
        return (std::f64::consts::PI / (-std::f64::consts::PI * z).sin()).ln() - ln_gamma(1.0 - z);
    }
    let z = z - 1.0;

    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * (z + 7.0 / 24.0 / (z + 1.0)).ln()
        - (z + 7.0 / 24.0 / (z + 1.0))
        + 1.0 / (12.0 * (z + 1.0))
}

// --- Ensemble methods ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnsembleMethod {
    Voting,    // Average prediction for regression, majority for classification
    Averaging, // Simple average
    Weighted,  // Weighted by validation score
}

/// Ensemble of models.
pub struct ModelEnsemble {
    pub models: Vec<(f64, Box<dyn Model>)>, // (weight, model)
    pub method: EnsembleMethod,
    pub is_classification: bool,
}

impl ModelEnsemble {
    pub fn new(
        models: Vec<(f64, Box<dyn Model>)>,
        method: EnsembleMethod,
        is_classification: bool,
    ) -> Self {
        Self {
            models,
            method,
            is_classification,
        }
    }

    pub fn from_scores(
        model_scores: Vec<(f64, Box<dyn Model>)>,
        method: EnsembleMethod,
        is_classification: bool,
    ) -> Self {
        let total: f64 = model_scores.iter().map(|(s, _)| s).sum();
        let models: Vec<(f64, Box<dyn Model>)> = model_scores
            .into_iter()
            .map(|(score, model)| {
                let w = if total.abs() > 1e-12
                {
                    score / total
                }
                else
                {
                    1.0
                };
                (w, model)
            })
            .collect();
        Self::new(models, method, is_classification)
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let n_samples = x.len();
        let all_preds: Vec<Vec<f64>> = self.models.iter().map(|(_, m)| m.predict(x)).collect();
        (0..n_samples)
            .map(|i| match self.method
            {
                EnsembleMethod::Voting | EnsembleMethod::Averaging =>
                {
                    let sum: f64 = self
                        .models
                        .iter()
                        .zip(&all_preds)
                        .map(|((w, _), p)| w * p[i])
                        .sum();
                    sum
                },
                EnsembleMethod::Weighted =>
                {
                    let sum: f64 = self
                        .models
                        .iter()
                        .zip(&all_preds)
                        .map(|((w, _), p)| w * p[i])
                        .sum();
                    sum
                },
            })
            .collect()
    }
}

/// `ModelSelector` compares multiple models on a task and selects the best.
pub struct ModelSelector {
    pub templates: Vec<PipelineTemplate>,
    pub metric: Metric,
    pub cv_folds: usize,
    pub seed: u64,
}

impl ModelSelector {
    pub fn new(
        templates: Vec<PipelineTemplate>,
        metric: Metric,
        cv_folds: usize,
        seed: u64,
    ) -> Self {
        Self {
            templates,
            metric,
            cv_folds,
            seed,
        }
    }

    /// Compare all templates and return a ranked list with statistics.
    pub fn compare(
        &self,
        x: &[Vec<f64>],
        y: &[f64],
        is_classification: bool,
    ) -> Vec<ModelComparison> {
        let n = self.templates.len();
        if n == 0
        {
            return vec![];
        }

        // Evaluate each template
        let mut rng = XorShift64::new(self.seed);
        let mut results: Vec<(String, PipelineConfig, CvResult)> = Vec::new();
        for template in &self.templates
        {
            // Use a default config with mean hyperparam values
            let cfg = template_default_config(template);
            let cv = cross_validate(
                &cfg,
                x,
                y,
                self.cv_folds,
                self.metric,
                is_classification,
                &mut rng,
            );
            results.push((template_label(template), cfg, cv));
        }

        // Pairwise comparisons
        let mut comparisons = Vec::new();
        for (i, (name_a, cfg_a, cv_a)) in results.iter().enumerate()
        {
            let mut pairwise = Vec::new();
            for (j, (name_b, _, cv_b)) in results.iter().enumerate()
            {
                if i == j
                {
                    continue;
                }
                let a_scores: Vec<f64> = cv_a.folds.iter().map(|f| f.val_metric).collect();
                let b_scores: Vec<f64> = cv_b.folds.iter().map(|f| f.val_metric).collect();
                let ttest = paired_t_test(&a_scores, &b_scores);
                pairwise.push((name_b.clone(), ttest));
            }
            comparisons.push(ModelComparison {
                name: name_a.clone(),
                config: cfg_a.clone(),
                cv_result: cv_a.clone(),
                rank: 0,
                pairwise_tests: pairwise,
            });
        }

        // Sort by mean_val (descending)
        comparisons.sort_by(|a, b| {
            b.cv_result
                .mean_val
                .partial_cmp(&a.cv_result.mean_val)
                .unwrap()
        });
        for (i, c) in comparisons.iter_mut().enumerate()
        {
            c.rank = i + 1;
        }

        comparisons
    }

    /// Build an ensemble from the top-k models.
    pub fn build_ensemble(
        &self,
        comparisons: &[ModelComparison],
        top_k: usize,
        method: EnsembleMethod,
        x: &[Vec<f64>],
        y: &[f64],
        is_classification: bool,
    ) -> ModelEnsemble {
        let mut rng = XorShift64::new(self.seed);
        let models: Vec<(f64, Box<dyn Model>)> = comparisons
            .iter()
            .take(top_k)
            .map(|mc| {
                let (proc_x, _) = apply_preprocessing(x, &mc.config.preprocessors, None);
                let model = train_model(&mc.config, &proc_x, y, is_classification, &mut rng);
                (mc.cv_result.mean_val.max(0.001), model)
            })
            .collect();

        ModelEnsemble::from_scores(models, method, is_classification)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparison {
    pub name: String,
    pub config: PipelineConfig,
    pub cv_result: CvResult,
    pub rank: usize,
    pub pairwise_tests: Vec<(String, TTestResult)>,
}

fn template_label(t: &PipelineTemplate) -> String {
    format!("{:?}", t.model)
}

fn template_default_config(t: &PipelineTemplate) -> PipelineConfig {
    let params: HashMap<String, f64> = t
        .hyperparameters
        .iter()
        .map(|(name, dist)| {
            let val = match dist
            {
                ParamDistribution::Uniform { lo, hi } => (*lo + *hi) / 2.0,
                ParamDistribution::LogUniform { lo, hi } => ((lo.ln() + hi.ln()) / 2.0).exp(),
                ParamDistribution::IntRange { lo, hi } => ((*lo + *hi) / 2) as f64,
                ParamDistribution::Categorical { options } =>
                {
                    options.first().map(|s| hash_str(s)).unwrap_or(0.0)
                },
            };
            (name.clone(), val)
        })
        .collect();
    PipelineConfig {
        preprocessors: t.preprocessors.clone(),
        model: t.model.clone(),
        params,
    }
}

// =========================================================================
// 13. Feature Engineer
// =========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSet {
    pub feature_indices: Vec<usize>,
    pub feature_names: Vec<String>,
}

pub struct FeatureEngineer {
    pub variance_threshold: f64,
    pub correlation_threshold: f64,
    pub max_poly_degree: usize,
}

impl Default for FeatureEngineer {
    fn default() -> Self {
        Self {
            variance_threshold: 0.01,
            correlation_threshold: 0.95,
            max_poly_degree: 2,
        }
    }
}

impl FeatureEngineer {
    /// Generate polynomial features up to the configured degree.
    pub fn polynomial_features(&self, x: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<String>) {
        let n_feat = x.first().map(|r| r.len()).unwrap_or(0);
        let combos = generate_combinations(n_feat, self.max_poly_degree);
        let names: Vec<String> = combos
            .iter()
            .map(|indices| {
                indices
                    .iter()
                    .map(|i| format!("x{}", i))
                    .collect::<Vec<_>>()
                    .join("*")
            })
            .collect();

        let transformed: Vec<Vec<f64>> = x
            .iter()
            .map(|row| {
                combos
                    .iter()
                    .map(|indices| indices.iter().map(|&i| row[i]).product())
                    .collect()
            })
            .collect();

        (transformed, names)
    }

    /// Detect interaction features: pairwise products with highest correlation to target.
    pub fn interaction_features(&self, x: &[Vec<f64>], y: &[f64], top_k: usize) -> FeatureSet {
        let n = x.len();
        let n_feat = x.first().map(|r| r.len()).unwrap_or(0);
        let mut interactions = Vec::new();

        for i in 0..n_feat
        {
            for j in (i + 1)..n_feat
            {
                let interaction: Vec<f64> = x.iter().map(|row| row[i] * row[j]).collect();
                let corr = pearson_correlation(&interaction, y);
                interactions.push(((i, j), corr));
            }
        }

        interactions.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        let indices: Vec<usize> = interactions
            .iter()
            .take(top_k)
            .enumerate()
            .map(|(k, _)| n_feat + k)
            .collect();

        // Generate interaction column names
        let mut names = Vec::new();
        let mut cols = Vec::new();
        for &((i, j), _) in interactions.iter().take(top_k)
        {
            names.push(format!("x{}*x{}", i, j));
            let interaction_col: Vec<f64> = x.iter().map(|row| row[i] * row[j]).collect();
            cols.push(interaction_col);
        }

        // Build combined feature matrix
        let combined: Vec<Vec<f64>> = x
            .iter()
            .enumerate()
            .map(|(sample_idx, row)| {
                let mut r = row.clone();
                for col in &cols
                {
                    r.push(col[sample_idx]);
                }
                r
            })
            .collect();

        let _ = combined;
        let _ = n;

        FeatureSet {
            feature_indices: indices,
            feature_names: names,
        }
    }

    /// Filter features by variance
    pub fn variance_threshold_filter(&self, x: &[Vec<f64>]) -> FeatureSet {
        let n_feat = x.first().map(|r| r.len()).unwrap_or(0);
        let n = x.len() as f64;
        let mut kept = Vec::new();
        let mut names = Vec::new();

        for j in 0..n_feat
        {
            let mean = x.iter().map(|row| row[j]).sum::<f64>() / n;
            let var = x.iter().map(|row| (row[j] - mean).powi(2)).sum::<f64>() / n;
            if var >= self.variance_threshold
            {
                kept.push(j);
                names.push(format!("x{}", j));
            }
        }

        FeatureSet {
            feature_indices: kept,
            feature_names: names,
        }
    }

    /// Filter features by pairwise correlation
    pub fn correlation_filter(&self, x: &[Vec<f64>]) -> FeatureSet {
        let n_feat = x.first().map(|r| r.len()).unwrap_or(0);
        let n = x.len() as f64;

        // Compute correlation matrix
        let mut kept: Vec<usize> = (0..n_feat).collect();
        let mut removed = vec![false; n_feat];

        for i in 0..n_feat
        {
            if removed[i]
            {
                continue;
            }
            for j in (i + 1)..n_feat
            {
                if removed[j]
                {
                    continue;
                }
                let corr = pearson_correlation_cols(x, i, j, n);
                if corr.abs() > self.correlation_threshold
                {
                    removed[j] = true;
                }
            }
        }

        kept.retain(|&k| !removed[k]);

        let names: Vec<String> = kept.iter().map(|&k| format!("x{}", k)).collect();

        FeatureSet {
            feature_indices: kept,
            feature_names: names,
        }
    }

    /// Simplified mutual information selector
    pub fn mutual_information_filter(&self, x: &[Vec<f64>], y: &[f64], top_k: usize) -> FeatureSet {
        let n_feat = x.first().map(|r| r.len()).unwrap_or(0);
        let mut scores: Vec<(usize, f64)> = Vec::new();

        for j in 0..n_feat
        {
            let mi = mutual_information_simple(&x.iter().map(|row| row[j]).collect::<Vec<_>>(), y);
            scores.push((j, mi));
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top: Vec<usize> = scores.iter().take(top_k).map(|(j, _)| *j).collect();
        let names: Vec<String> = top.iter().map(|&j| format!("x{}", j)).collect();

        FeatureSet {
            feature_indices: top,
            feature_names: names,
        }
    }

    /// Run the full feature engineering pipeline.
    pub fn engineer(
        &self,
        x: &[Vec<f64>],
        y: &[f64],
        use_poly: bool,
        use_interactions: bool,
        use_variance: bool,
        use_correlation: bool,
        use_mi: bool,
        mi_top_k: usize,
    ) -> (Vec<Vec<f64>>, Vec<String>) {
        let mut current = x.to_vec();
        let mut names: Vec<String> = (0..x.first().map(|r| r.len()).unwrap_or(0))
            .map(|j| format!("x{}", j))
            .collect();

        if use_poly
        {
            let (poly_x, poly_names) = self.polynomial_features(&current);
            current = poly_x;
            names.extend(poly_names);
        }

        if use_interactions
        {
            let inter = self.interaction_features(&current, y, 5);
            let n_feat = current.first().map(|r| r.len()).unwrap_or(0);
            for name in &inter.feature_names
            {
                names.push(name.clone());
            }
            // Recompute interactions on current feature set
            let interaction_cols = generate_interaction_columns(&current, n_feat);
            for col in interaction_cols.iter()
            {
                for row in current.iter_mut()
                {
                    row.push(col[row.len() % col.len()]);
                }
            }
            let _ = n_feat;
        }

        if use_variance
        {
            let fs = self.variance_threshold_filter(&current);
            current = apply_feature_selection(&current, &fs);
            names = fs
                .feature_indices
                .iter()
                .map(|&i| names[i].clone())
                .collect();
        }

        if use_correlation
        {
            let fs = self.correlation_filter(&current);
            current = apply_feature_selection(&current, &fs);
            names = fs
                .feature_indices
                .iter()
                .map(|&i| names[i].clone())
                .collect();
        }

        if use_mi && !y.is_empty()
        {
            let fs = self.mutual_information_filter(&current, y, mi_top_k);
            current = apply_feature_selection(&current, &fs);
            names = fs
                .feature_indices
                .iter()
                .map(|&i| names[i].clone())
                .collect();
        }

        (current, names)
    }
}

fn pearson_correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len()) as f64;
    if n < 2.0
    {
        return 0.0;
    }
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for i in 0..a.len().min(b.len())
    {
        cov += (a[i] - ma) * (b[i] - mb);
        va += (a[i] - ma).powi(2);
        vb += (b[i] - mb).powi(2);
    }
    if va.abs() < 1e-12 || vb.abs() < 1e-12
    {
        return 0.0;
    }
    cov / (va.sqrt() * vb.sqrt())
}

fn pearson_correlation_cols(x: &[Vec<f64>], i: usize, j: usize, _n: f64) -> f64 {
    let col_i: Vec<f64> = x.iter().map(|row| row[i]).collect();
    let col_j: Vec<f64> = x.iter().map(|row| row[j]).collect();
    pearson_correlation(&col_i, &col_j)
}

fn mutual_information_simple(x: &[f64], y: &[f64]) -> f64 {
    // Binned mutual information (simplified)
    let n = x.len().min(y.len());
    if n < 2
    {
        return 0.0;
    }
    let n_bins = 10;

    // Bin x and y
    let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let x_range = (x_max - x_min).max(1e-12);
    let y_range = (y_max - y_min).max(1e-12);

    let mut joint = vec![vec![0u32; n_bins]; n_bins];
    let mut x_counts = vec![0u32; n_bins];
    let mut y_counts = vec![0u32; n_bins];

    for i in 0..n
    {
        let bx = (((x[i] - x_min) / x_range * (n_bins - 1) as f64) as usize).min(n_bins - 1);
        let by = (((y[i] - y_min) / y_range * (n_bins - 1) as f64) as usize).min(n_bins - 1);
        joint[bx][by] += 1;
        x_counts[bx] += 1;
        y_counts[by] += 1;
    }

    let nf = n as f64;
    let mut mi = 0.0;
    for bx in 0..n_bins
    {
        for by in 0..n_bins
        {
            if joint[bx][by] > 0
            {
                let p_xy = joint[bx][by] as f64 / nf;
                let p_x = x_counts[bx] as f64 / nf;
                let p_y = y_counts[by] as f64 / nf;
                mi += p_xy * (p_xy / (p_x * p_y)).ln();
            }
        }
    }
    mi
}

fn apply_feature_selection(x: &[Vec<f64>], fs: &FeatureSet) -> Vec<Vec<f64>> {
    x.iter()
        .map(|row| fs.feature_indices.iter().map(|&j| row[j]).collect())
        .collect()
}

fn generate_interaction_columns(_x: &[Vec<f64>], _n_feat: usize) -> Vec<Vec<f64>> {
    // Stub: called from engineer when use_interactions=true; real pairs are built inside
    // interaction_features. For consistent shape we return empty here.
    vec![]
}

// =========================================================================
// 14. AutoML Orchestrator
// =========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoMLConfig {
    pub models: Vec<ModelKind>,
    pub preprocessors: Vec<PreprocessorKind>,
    pub optimization: OptimizerConfig,
    pub cv_folds: usize,
    pub metric: Metric,
    pub ensemble_top_k: usize,
    pub ensemble_method: EnsembleMethod,
    pub early_stopping_rounds: usize,
    pub seed: u64,
    pub use_feature_engineering: bool,
    pub verbose: bool,
}

impl Default for AutoMLConfig {
    fn default() -> Self {
        Self {
            models: vec![
                ModelKind::Linear,
                ModelKind::RandomForest,
                ModelKind::GradientBoosting,
            ],
            preprocessors: vec![PreprocessorKind::StandardScaler],
            optimization: OptimizerConfig::default(),
            cv_folds: 5,
            metric: Metric::Accuracy,
            ensemble_top_k: 3,
            ensemble_method: EnsembleMethod::Voting,
            early_stopping_rounds: 10,
            seed: 0x5C12_3E70,
            use_feature_engineering: true,
            verbose: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoMLReport {
    pub best_model: ModelComparison,
    pub all_models: Vec<ModelComparison>,
    pub ensemble_used: bool,
    pub feature_importance: Vec<(String, f64)>,
    pub best_config: PipelineConfig,
    pub best_cv_score: f64,
    pub best_cv_std: f64,
    pub total_trials: usize,
    pub elapsed_seconds: f64,
    pub metadata: HashMap<String, String>,
}

/// The main AutoML struct.
pub struct AutoML {
    config: AutoMLConfig,
}

impl AutoML {
    pub fn new(cfg: AutoMLConfig) -> Self {
        Self { config: cfg }
    }

    /// Run the full AutoML pipeline.
    pub fn fit(&self, x: &[Vec<f64>], y: &[f64], is_classification: bool) -> AutoMLReport {
        let start = std::time::Instant::now();
        let mut total_trials = 0;

        // Feature engineering
        let (feat_x, feat_names) = if self.config.use_feature_engineering
        {
            let fe = FeatureEngineer::default();
            let (fx, names) = fe.engineer(
                x, y, true, // use_poly
                true, // use_interactions
                true, // use_variance
                true, // use_correlation
                true, // use_mi
                20,   // mi_top_k
            );
            total_trials += 1;
            (fx, names)
        }
        else
        {
            (
                x.to_vec(),
                (0..x.first().map(|r| r.len()).unwrap_or(0))
                    .map(|j| format!("x{}", j))
                    .collect(),
            )
        };

        // Build templates for each model
        let templates: Vec<PipelineTemplate> = self
            .config
            .models
            .iter()
            .map(|m| PipelineTemplate::new(m.clone(), self.config.preprocessors.clone()))
            .collect();

        // Hyperparameter optimization per template
        let optimizer = HyperOptimizer::new(self.config.optimization.clone());
        let mut all_trials: Vec<TrialResult> = Vec::new();

        for template in &templates
        {
            if self.config.verbose
            {
                eprintln!("  AutoML: optimizing {:?} ...", template.model);
            }

            let trials = optimizer.random_search(
                template,
                &feat_x,
                y,
                self.config.metric,
                is_classification,
            );
            total_trials += trials.len();
            all_trials.extend(trials);

            // Early stopping check: if no improvement after N rounds
            let recent = all_trials
                .len()
                .saturating_sub(self.config.early_stopping_rounds);
            if all_trials.len() > self.config.early_stopping_rounds
            {
                let recent_max = all_trials[recent..]
                    .iter()
                    .map(|t| t.score)
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0);
                let earlier_max = all_trials[..recent]
                    .iter()
                    .map(|t| t.score)
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0);
                if recent_max <= earlier_max
                {
                    if self.config.verbose
                    {
                        eprintln!("  Early stopping triggered.");
                    }
                    break;
                }
            }
        }

        // Find best config overall
        let best_trial = all_trials
            .iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
            .unwrap();

        // Model comparison
        let selector = ModelSelector::new(
            templates.clone(),
            self.config.metric,
            self.config.cv_folds,
            self.config.seed,
        );
        let comparisons = selector.compare(&feat_x, y, is_classification);

        // Feature importance (simple: use Pearson correlation with target)
        let feat_importance: Vec<(String, f64)> = feat_names
            .iter()
            .enumerate()
            .map(|(j, name)| {
                let col: Vec<f64> = feat_x.iter().map(|row| row[j]).collect();
                let corr = pearson_correlation(&col, y);
                (name.clone(), corr.abs())
            })
            .collect();

        // Build ensemble
        let ensemble_used = comparisons.len() >= 2;
        let _ensemble = if ensemble_used
        {
            Some(selector.build_ensemble(
                &comparisons,
                self.config.ensemble_top_k,
                self.config.ensemble_method.clone(),
                &feat_x,
                y,
                is_classification,
            ))
        }
        else
        {
            None
        };

        let mut metadata = HashMap::new();
        metadata.insert(
            "models_tested".to_string(),
            self.config
                .models
                .iter()
                .map(|m| format!("{:?}", m))
                .collect::<Vec<_>>()
                .join(","),
        );
        metadata.insert(
            "preprocessors".to_string(),
            format!("{:?}", self.config.preprocessors),
        );

        AutoMLReport {
            best_model: comparisons.first().cloned().unwrap(),
            all_models: comparisons,
            ensemble_used,
            feature_importance: feat_importance,
            best_config: best_trial.config.clone(),
            best_cv_score: best_trial.score,
            best_cv_std: best_trial.cv_result.std_val,
            total_trials,
            elapsed_seconds: start.elapsed().as_secs_f64(),
            metadata,
        }
    }
}

// =========================================================================
// 15. Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_regression_data(n: usize, d: usize, rng: &mut XorShift64) -> (Vec<Vec<f64>>, Vec<f64>) {
        let x: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..d).map(|_| rng.uniform(-2.0, 2.0)).collect())
            .collect();
        // y = 2*x0 + 0.5*x1 - x2 + noise
        let y: Vec<f64> = x
            .iter()
            .map(|row| {
                let signal = 2.0 * row[0] + 0.5 * row[1] - row.get(2).copied().unwrap_or(0.0);
                signal + rng.normal() * 0.3
            })
            .collect();
        (x, y)
    }

    fn make_classification_data(
        n: usize,
        d: usize,
        rng: &mut XorShift64,
    ) -> (Vec<Vec<f64>>, Vec<f64>) {
        let x: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..d).map(|_| rng.uniform(-2.0, 2.0)).collect())
            .collect();
        let y: Vec<f64> = x
            .iter()
            .map(|row| {
                let logit = row[0] + row[1] - row[2];
                if sigmoid(logit) >= 0.5 { 1.0 } else { 0.0 }
            })
            .collect();
        (x, y)
    }

    // ---- XorShift PRNG tests ----

    #[test]
    fn test_xorshift_deterministic() {
        let mut a = XorShift64::new(42);
        let mut b = XorShift64::new(42);
        for _ in 0..100
        {
            assert_eq!(a.next_f64(), b.next_f64());
        }
    }

    #[test]
    fn test_xorshift_uniform_range() {
        let mut rng = XorShift64::new(1);
        for _ in 0..1000
        {
            let v = rng.uniform(10.0, 20.0);
            assert!(v >= 10.0 && v < 20.0);
        }
    }

    #[test]
    fn test_xorshift_usize_range() {
        let mut rng = XorShift64::new(2);
        for _ in 0..1000
        {
            let v = rng.usize(10);
            assert!(v < 10);
        }
    }

    // ---- Parameter distributions ----

    #[test]
    fn test_uniform_distribution() {
        let d = ParamDistribution::Uniform { lo: 0.0, hi: 10.0 };
        let mut rng = XorShift64::new(3);
        for _ in 0..100
        {
            let v = d.sample(&mut rng);
            assert!(v >= 0.0 && v < 10.0);
        }
    }

    #[test]
    fn test_log_uniform_distribution() {
        let d = ParamDistribution::LogUniform { lo: 1.0, hi: 10.0 };
        let mut rng = XorShift64::new(4);
        for _ in 0..100
        {
            let v = d.sample(&mut rng);
            assert!(v >= 1.0 && v <= 10.0, "value {} out of range", v);
        }
    }

    #[test]
    fn test_int_range_distribution() {
        let d = ParamDistribution::IntRange { lo: 5, hi: 15 };
        let mut rng = XorShift64::new(5);
        for _ in 0..100
        {
            let v = d.sample(&mut rng) as i64;
            assert!(v >= 5 && v <= 15);
        }
    }

    #[test]
    fn test_categorical_distribution() {
        let d = ParamDistribution::Categorical {
            options: vec!["a".into(), "b".into(), "c".into()],
        };
        let mut rng = XorShift64::new(6);
        for _ in 0..100
        {
            let v = d.sample(&mut rng);
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_grid_points_uniform() {
        let d = ParamDistribution::Uniform { lo: 0.0, hi: 4.0 };
        let pts = d.grid_points(5);
        assert_eq!(pts.len(), 5);
        assert!((pts[0] - 0.0).abs() < 1e-10);
        assert!((pts[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_grid_points_int_range() {
        let d = ParamDistribution::IntRange { lo: 1, hi: 10 };
        let pts = d.grid_points(5);
        assert!(!pts.is_empty());
        for &v in &pts
        {
            let i = v as i64;
            assert!(i >= 1 && i <= 10, "{} out of range", i);
        }
    }

    // ---- Metrics ----

    #[test]
    fn test_accuracy_perfect() {
        let y_true = vec![1.0, 0.0, 1.0, 0.0];
        let y_pred = vec![1.0, 0.0, 1.0, 0.0];
        assert!((accuracy(&y_true, &y_pred) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mse_zero() {
        let y = vec![3.0, 5.0, 7.0];
        assert!((mse(&y, &y)).abs() < 1e-10);
    }

    #[test]
    fn test_r2_perfect() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((r2_score(&y, &y) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_f1_perfect() {
        let y_true = vec![1.0, 1.0, 0.0, 0.0];
        let y_pred = vec![1.0, 1.0, 0.0, 0.0];
        assert!((f1_score(&y_true, &y_pred) - 1.0).abs() < 1e-10);
    }

    // ---- StandardScaler ----

    #[test]
    fn test_standard_scaler_zero_mean() {
        let x = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let scaler = StandardScaler::fit(&x);
        let transformed = scaler.transform(&x);
        let mean0: f64 = transformed.iter().map(|r| r[0]).sum::<f64>() / 3.0;
        let mean1: f64 = transformed.iter().map(|r| r[1]).sum::<f64>() / 3.0;
        assert!(mean0.abs() < 1e-10);
        assert!(mean1.abs() < 1e-10);
    }

    #[test]
    fn test_standard_scaler_unit_std() {
        let x = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let scaler = StandardScaler::fit(&x);
        let transformed = scaler.transform(&x);
        let var0: f64 = transformed.iter().map(|r| r[0].powi(2)).sum::<f64>() / 3.0;
        let var1: f64 = transformed.iter().map(|r| r[1].powi(2)).sum::<f64>() / 3.0;
        assert!((var0 - 1.0).abs() < 1e-10);
        assert!((var1 - 1.0).abs() < 1e-10);
    }

    // ---- PCA ----

    #[test]
    fn test_pca_reduces_dimensionality() {
        let mut rng = XorShift64::new(7);
        let x: Vec<Vec<f64>> = (0..50)
            .map(|_| (0..10).map(|_| rng.uniform(-1.0, 1.0)).collect())
            .collect();
        let pca = PCA::fit(&x, 3);
        let transformed = pca.transform(&x);
        assert_eq!(transformed.len(), 50);
        assert_eq!(transformed[0].len(), 3);
    }

    // ---- PolynomialFeatures ----

    #[test]
    fn test_polynomial_features_degree_2() {
        let x = vec![vec![2.0, 3.0]];
        let pf = PolynomialFeatures::fit(&x, 2);
        let transformed = pf.transform(&x);
        // x0, x1, x0^2, x0*x1, x1^2
        assert_eq!(transformed[0].len(), 5);
        assert!((transformed[0][0] - 2.0).abs() < 1e-10);
        assert!((transformed[0][1] - 3.0).abs() < 1e-10);
        assert!((transformed[0][2] - 4.0).abs() < 1e-10);
        assert!((transformed[0][3] - 6.0).abs() < 1e-10);
        assert!((transformed[0][4] - 9.0).abs() < 1e-10);
    }

    // ---- Linear model ----

    #[test]
    fn test_linear_model_regression() {
        let mut rng = XorShift64::new(8);
        let (x, y) = make_regression_data(100, 3, &mut rng);
        let model = LinearModel::fit(&x, &y, false, 0.01);
        let preds = model.predict(&x);
        let r2 = r2_score(&y, &preds);
        assert!(r2 > 0.5, "Linear model R2 too low: {}", r2);
    }

    #[test]
    fn test_linear_model_classification() {
        let mut rng = XorShift64::new(9);
        let (x, y) = make_classification_data(100, 3, &mut rng);
        let model = LinearModel::fit(&x, &y, true, 0.1);
        let preds = model.predict(&x);
        let acc = accuracy(&y, &preds);
        assert!(acc > 0.6, "Linear classification accuracy too low: {}", acc);
    }

    // ---- RandomForest ----

    #[test]
    fn test_random_forest_regression() {
        let mut rng = XorShift64::new(10);
        let (x, y) = make_regression_data(100, 3, &mut rng);
        let rf = RandomForest::fit(&x, &y, 20, 10, 5, false, &mut rng);
        let preds = rf.predict(&x);
        let r2 = r2_score(&y, &preds);
        assert!(r2 > 0.4, "RF R2 too low: {}", r2);
    }

    // ---- GradientBoosting ----

    #[test]
    fn test_gradient_boosting_regression() {
        let mut rng = XorShift64::new(11);
        let (x, y) = make_regression_data(80, 3, &mut rng);
        let gb = GradientBoosting::fit(&x, &y, 30, 0.1, 5, false, &mut rng);
        let preds = gb.predict(&x);
        let r2 = r2_score(&y, &preds);
        assert!(r2 > 0.3, "GBT R2 too low: {}", r2);
    }

    // ---- NeuralNetwork ----

    #[test]
    fn test_neural_network_regression() {
        let mut rng = XorShift64::new(12);
        let (x, y) = make_regression_data(80, 3, &mut rng);
        let nn = NeuralNetwork::fit(&x, &y, &[32, 16], 0.01, 200, false, &mut rng);
        let preds = nn.predict(&x);
        let r2 = r2_score(&y, &preds);
        assert!(r2 > 0.3, "NN regression R2 too low: {}", r2);
    }

    #[test]
    fn test_neural_network_classification() {
        let mut rng = XorShift64::new(13);
        let (x, y) = make_classification_data(80, 3, &mut rng);
        let nn = NeuralNetwork::fit(&x, &y, &[32, 16], 0.01, 150, true, &mut rng);
        let preds = nn.predict(&x);
        let acc = accuracy(&y, &preds);
        assert!(acc > 0.6, "NN classification accuracy too low: {}", acc);
    }

    // ---- Cross-validation ----

    #[test]
    fn test_cross_validate_returns_folds() {
        let mut rng = XorShift64::new(14);
        let (x, y) = make_regression_data(60, 3, &mut rng);
        let config = PipelineConfig {
            preprocessors: vec![],
            model: ModelKind::Linear,
            params: HashMap::new(),
        };
        let cv = cross_validate(&config, &x, &y, 5, Metric::R2, false, &mut rng);
        assert_eq!(cv.folds.len(), 5);
        assert!(cv.ci_lower <= cv.mean_val && cv.mean_val <= cv.ci_upper);
    }

    #[test]
    fn test_time_series_cv_monotonic() {
        let mut rng = XorShift64::new(15);
        let (x, y) = make_regression_data(50, 3, &mut rng);
        let config = PipelineConfig {
            preprocessors: vec![],
            model: ModelKind::Linear,
            params: HashMap::new(),
        };
        let cv = time_series_cv(&config, &x, &y, 3, Metric::R2, false, &mut rng);
        assert!(!cv.folds.is_empty());
        assert!(cv.folds.len() >= 1);
    }

    // ---- Gaussian Process / Bayesian Optimization ----

    #[test]
    fn test_gp_predict_interpolates() {
        let x = vec![vec![0.0], vec![1.0], vec![2.0]];
        let y = vec![0.0, 2.0, 4.0];
        let gp = GaussianProcess::fit(&x, &y, 0.5, 1e-6);
        let (mu, var) = gp.predict(&[1.0]);
        assert!((mu - 2.0).abs() < 0.5, "GP mean diverged: {}", mu);
        assert!(var > 0.0);
    }

    #[test]
    fn test_bayesian_optimize_finds_minimum() {
        // Minimize (x-3)^2 → optimum at x=3
        let mut f = |x: &[f64]| -> f64 { -(x[0] - 3.0).powi(2) };
        let mut rng = XorShift64::new(16);
        let (best_x, best_y) = bayesian_optimize(&mut f, &[(0.0, 6.0)], 20, 5, &mut rng);
        assert!(
            (best_x[0] - 3.0).abs() < 1.0,
            "BO didn't find optimum: x={}",
            best_x[0]
        );
        assert!(best_y > -1.0);
    }

    // ---- HyperOptimizer ----

    #[test]
    fn test_random_search_produces_trials() {
        let mut rng = XorShift64::new(17);
        let (x, y) = make_regression_data(50, 3, &mut rng);
        let template = PipelineTemplate::new(ModelKind::Linear, vec![]);
        let opt = HyperOptimizer::new(OptimizerConfig {
            search_budget: 10,
            ..Default::default()
        });
        let trials = opt.random_search(&template, &x, &y, Metric::R2, false);
        assert_eq!(trials.len(), 10);
    }

    #[test]
    fn test_grid_search_produces_trials() {
        let mut rng = XorShift64::new(18);
        let (x, y) = make_regression_data(30, 2, &mut rng);
        let template = PipelineTemplate::new(ModelKind::Linear, vec![]);
        let opt = HyperOptimizer::new(OptimizerConfig {
            search_budget: 8,
            ..Default::default()
        });
        let trials = opt.grid_search(&template, &x, &y, Metric::R2, false);
        assert!(!trials.is_empty());
    }

    // ---- Model Selector / Pairwise tests ----

    #[test]
    fn test_paired_t_test_significant() {
        let a = vec![0.9, 0.92, 0.88, 0.91, 0.93];
        let b = vec![0.7, 0.72, 0.68, 0.71, 0.70];
        let result = paired_t_test(&a, &b);
        assert!(result.is_significant);
        assert_eq!(result.winner, "model_a");
    }

    #[test]
    fn test_paired_t_test_tie() {
        let a = vec![0.85, 0.84, 0.86, 0.85, 0.86];
        let b = vec![0.85, 0.85, 0.85, 0.85, 0.85];
        let result = paired_t_test(&a, &b);
        assert!(!result.is_significant);
    }

    #[test]
    fn test_model_selector_ranks_models() {
        let mut rng = XorShift64::new(19);
        let (x, y) = make_classification_data(60, 3, &mut rng);
        let templates = vec![
            PipelineTemplate::new(ModelKind::Linear, vec![]),
            PipelineTemplate::new(ModelKind::RandomForest, vec![]),
        ];
        let selector = ModelSelector::new(templates, Metric::Accuracy, 5, 123);
        let comparisons = selector.compare(&x, &y, true);
        assert_eq!(comparisons.len(), 2);
        assert!(comparisons[0].rank == 1, "First model should be rank 1");
        assert!(comparisons[1].rank == 2, "Second model should be rank 2");
    }

    // ---- Ensemble ----

    #[test]
    fn test_ensemble_averaging() {
        let mut rng = XorShift64::new(20);
        let (x, y) = make_regression_data(50, 3, &mut rng);
        let m1: Box<dyn Model> = Box::new(LinearModel::fit(&x, &y, false, 0.1));
        let m2: Box<dyn Model> = Box::new(LinearModel::fit(&x, &y, false, 1.0));
        let ensemble = ModelEnsemble::from_scores(
            vec![(1.0, m1), (1.0, m2)],
            EnsembleMethod::Averaging,
            false,
        );
        let preds = ensemble.predict(&x);
        assert_eq!(preds.len(), y.len());
        let r2 = r2_score(&y, &preds);
        assert!(r2 > 0.5, "Ensemble R2 too low: {}", r2);
    }

    // ---- Feature Engineer ----

    #[test]
    fn test_feature_engineer_polynomial() {
        let mut rng = XorShift64::new(21);
        let x: Vec<Vec<f64>> = (0..10)
            .map(|_| (0..3).map(|_| rng.uniform(-1.0, 1.0)).collect())
            .collect();
        let fe = FeatureEngineer::default();
        let (poly, names) = fe.polynomial_features(&x);
        assert!(
            poly[0].len() > 3,
            "Should have more features after polynomial expansion"
        );
        assert!(!names.is_empty());
    }

    #[test]
    fn test_feature_engineer_variance_filter() {
        let x = vec![
            vec![1.0, 0.0, 3.0],
            vec![2.0, 0.0, 4.0],
            vec![1.0, 0.0, 3.0],
            vec![2.0, 0.0, 4.0],
        ];
        let fe = FeatureEngineer {
            variance_threshold: 0.01,
            ..Default::default()
        };
        let fs = fe.variance_threshold_filter(&x);
        // Column 1 has zero variance, should be removed
        assert!(!fs.feature_indices.contains(&1));
        assert!(fs.feature_indices.contains(&0));
    }

    #[test]
    fn test_feature_engineer_correlation_filter() {
        // Col 0 and col 2 are identical → one should be removed
        let x = vec![
            vec![1.0, 5.0, 1.0],
            vec![2.0, 6.0, 2.0],
            vec![3.0, 7.0, 3.0],
            vec![4.0, 8.0, 4.0],
        ];
        let fe = FeatureEngineer {
            correlation_threshold: 0.9,
            ..Default::default()
        };
        let fs = fe.correlation_filter(&x);
        assert!(
            fs.feature_indices.len() < 3,
            "Should remove highly correlated feature"
        );
    }

    // ---- Full AutoML pipeline ----

    #[test]
    fn test_automl_fit_regression() {
        let mut rng = XorShift64::new(22);
        let (x, y) = make_regression_data(60, 3, &mut rng);
        let cfg = AutoMLConfig {
            models: vec![ModelKind::Linear, ModelKind::RandomForest],
            preprocessors: vec![],
            optimization: OptimizerConfig {
                search_budget: 8,
                cv_folds: 3,
                ..Default::default()
            },
            cv_folds: 3,
            metric: Metric::R2,
            ensemble_top_k: 2,
            use_feature_engineering: false,
            verbose: false,
            ..Default::default()
        };
        let automl = AutoML::new(cfg);
        let report = automl.fit(&x, &y, false);

        assert!(
            report.best_cv_score > 0.0,
            "Best score should be positive for R2"
        );
        assert!(!report.all_models.is_empty());
        assert!(report.total_trials > 0);
        assert!(report.elapsed_seconds >= 0.0);
    }

    #[test]
    fn test_automl_fit_classification() {
        let mut rng = XorShift64::new(23);
        let (x, y) = make_classification_data(60, 3, &mut rng);
        let cfg = AutoMLConfig {
            models: vec![ModelKind::Linear],
            preprocessors: vec![],
            optimization: OptimizerConfig {
                search_budget: 5,
                cv_folds: 3,
                ..Default::default()
            },
            cv_folds: 3,
            metric: Metric::Accuracy,
            ensemble_top_k: 1,
            use_feature_engineering: false,
            verbose: false,
            ..Default::default()
        };
        let automl = AutoML::new(cfg);
        let report = automl.fit(&x, &y, true);

        assert!(report.best_cv_score > 0.0);
        assert!(report.total_trials > 0);
    }

    #[test]
    fn test_pipeline_template_generate() {
        let mut rng = XorShift64::new(24);
        let t = PipelineTemplate::new(
            ModelKind::RandomForest,
            vec![PreprocessorKind::StandardScaler],
        );
        let candidates = t.generate_candidates(5, &mut rng);
        assert_eq!(candidates.len(), 5);
        for c in &candidates
        {
            assert!(c.params.contains_key("n_trees"));
            assert!(c.params.contains_key("max_depth"));
        }
    }

    #[test]
    fn test_preprocessing_pipeline() {
        let x = vec![vec![1.0, 10.0], vec![3.0, 30.0], vec![5.0, 50.0]];
        let steps = vec![
            PreprocessorKind::StandardScaler,
            PreprocessorKind::Normalizer,
        ];
        let (processed, _) = apply_preprocessing(&x, &steps, None);
        assert_eq!(processed.len(), 3);
        assert_eq!(processed[0].len(), 2);
    }

    #[test]
    fn test_apply_feature_selection() {
        let x = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let fs = FeatureSet {
            feature_indices: vec![0, 2],
            feature_names: vec!["x0".into(), "x2".into()],
        };
        let sel = apply_feature_selection(&x, &fs);
        assert_eq!(sel[0], vec![1.0, 3.0]);
        assert_eq!(sel[1], vec![4.0, 6.0]);
    }
}
