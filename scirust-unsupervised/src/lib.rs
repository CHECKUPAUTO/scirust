//! Unsupervised pattern detection algorithms.
//!
//! This crate provides implementations of common unsupervised learning algorithms:
//! - Autoencoder for anomaly detection via reconstruction error
//! - Isolation Forest for efficient anomaly detection
//! - DBSCAN for density-based clustering
//! - Local Outlier Factor (LOF) for local outlier detection
//! - Gaussian Mixture Models (GMM) for soft clustering
//! - One-Class SVM for one-class classification

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
fn norm2(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).clamp(-100.0, 100.0).exp())
}

// Simple pseudo-random number generator (xorshift64) – deterministic, no `rand` dep.
#[derive(Clone)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn gen_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    fn gen_range(&mut self, lo: usize, hi: usize) -> usize {
        // Degenerate/empty range: return the lower bound instead of dividing by
        // zero. Callers treating [lo, hi) as empty must guard the result anyway.
        if hi <= lo
        {
            return lo;
        }
        lo + (self.next_u64() as usize) % (hi - lo)
    }

    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev()
        {
            let j = self.gen_range(0, i + 1);
            v.swap(i, j);
        }
    }
}

// ===========================================================================
// 1. Autoencoder
// ===========================================================================

/// Configuration for the autoencoder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoencoderConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub learning_rate: f64,
    pub epochs: usize,
}

impl Default for AutoencoderConfig {
    fn default() -> Self {
        Self {
            input_dim: 10,
            hidden_dim: 4,
            learning_rate: 0.01,
            epochs: 100,
        }
    }
}

/// A simple single-hidden-layer autoencoder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Autoencoder {
    /// Encoder weights: hidden_dim × input_dim
    pub encoder_w: Vec<Vec<f64>>,
    /// Encoder bias: hidden_dim
    pub encoder_b: Vec<f64>,
    /// Decoder weights: input_dim × hidden_dim
    pub decoder_w: Vec<Vec<f64>>,
    /// Decoder bias: input_dim
    pub decoder_b: Vec<f64>,
    config: AutoencoderConfig,
}

impl Autoencoder {
    pub fn new(config: AutoencoderConfig) -> Self {
        let mut rng = Rng::new(42);
        let encoder_w: Vec<Vec<f64>> = (0..config.hidden_dim)
            .map(|_| {
                (0..config.input_dim)
                    .map(|_| (rng.gen_f64() - 0.5) * 2.0 / (config.input_dim as f64).sqrt())
                    .collect()
            })
            .collect();
        let encoder_b = vec![0.0; config.hidden_dim];
        let decoder_w: Vec<Vec<f64>> = (0..config.input_dim)
            .map(|_| {
                (0..config.hidden_dim)
                    .map(|_| (rng.gen_f64() - 0.5) * 2.0 / (config.hidden_dim as f64).sqrt())
                    .collect()
            })
            .collect();
        let decoder_b = vec![0.0; config.input_dim];
        Self {
            encoder_w,
            encoder_b,
            decoder_w,
            decoder_b,
            config,
        }
    }

    /// Encode input → hidden representation (sigmoid activation).
    pub fn encode(&self, input: &[f64]) -> Vec<f64> {
        self.encoder_w
            .iter()
            .zip(self.encoder_b.iter())
            .map(|(w, &b)| sigmoid(dot(w, input) + b))
            .collect()
    }

    /// Decode hidden → reconstruction (sigmoid activation).
    pub fn decode(&self, hidden: &[f64]) -> Vec<f64> {
        self.decoder_w
            .iter()
            .zip(self.decoder_b.iter())
            .map(|(w, &b)| sigmoid(dot(w, hidden) + b))
            .collect()
    }

    /// Forward pass: input → reconstruction.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let hidden = self.encode(input);
        self.decode(&hidden)
    }

    /// Reconstruction error (mean squared error).
    pub fn reconstruction_error(&self, input: &[f64]) -> f64 {
        let recon = self.forward(input);
        input
            .iter()
            .zip(recon.iter())
            .map(|(x, r)| (x - r).powi(2))
            .sum::<f64>()
            / input.len() as f64
    }

    /// Train the autoencoder on a dataset via backpropagation.
    #[allow(clippy::needless_range_loop)]
    pub fn train(&mut self, data: &[Vec<f64>]) {
        let lr = self.config.learning_rate;
        let input_dim = self.config.input_dim;
        let hidden_dim = self.config.hidden_dim;

        for _ in 0..self.config.epochs
        {
            for sample in data
            {
                // Forward
                let mut hidden_raw = vec![0.0; hidden_dim];
                for h in 0..hidden_dim
                {
                    hidden_raw[h] = dot(&self.encoder_w[h], sample) + self.encoder_b[h];
                }
                let hidden: Vec<f64> = hidden_raw.iter().map(|&z| sigmoid(z)).collect();

                let mut output_raw = vec![0.0; input_dim];
                for o in 0..input_dim
                {
                    output_raw[o] = dot(&self.decoder_w[o], &hidden) + self.decoder_b[o];
                }
                let output: Vec<f64> = output_raw.iter().map(|&z| sigmoid(z)).collect();

                // Output error
                let mut d_output = vec![0.0; input_dim];
                for o in 0..input_dim
                {
                    let sig_deriv = output[o] * (1.0 - output[o]);
                    d_output[o] = 2.0 * (output[o] - sample[o]) * sig_deriv / input_dim as f64;
                }

                // Decoder gradients
                for o in 0..input_dim
                {
                    for h in 0..hidden_dim
                    {
                        self.decoder_w[o][h] -= lr * d_output[o] * hidden[h];
                    }
                    self.decoder_b[o] -= lr * d_output[o];
                }

                // Hidden error
                let mut d_hidden = vec![0.0; hidden_dim];
                for h in 0..hidden_dim
                {
                    for o in 0..input_dim
                    {
                        d_hidden[h] += self.decoder_w[o][h] * d_output[o];
                    }
                    let sig_deriv = hidden[h] * (1.0 - hidden[h]);
                    d_hidden[h] *= sig_deriv;
                }

                // Encoder gradients
                for h in 0..hidden_dim
                {
                    for i in 0..input_dim
                    {
                        self.encoder_w[h][i] -= lr * d_hidden[h] * sample[i];
                    }
                    self.encoder_b[h] -= lr * d_hidden[h];
                }
            }
        }
    }

    /// Detect anomalies: returns indices of samples exceeding threshold.
    pub fn detect_anomalies(&self, data: &[Vec<f64>], threshold: f64) -> Vec<usize> {
        data.iter()
            .enumerate()
            .filter(|(_, sample)| self.reconstruction_error(sample) > threshold)
            .map(|(i, _)| i)
            .collect()
    }

    /// Compute anomaly scores for all samples.
    pub fn anomaly_scores(&self, data: &[Vec<f64>]) -> Vec<f64> {
        data.iter()
            .map(|sample| self.reconstruction_error(sample))
            .collect()
    }
}

// ===========================================================================
// 2. Isolation Forest
// ===========================================================================

/// Configuration for the Isolation Forest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IForestConfig {
    pub n_trees: usize,
    pub subsample_size: usize,
    pub max_depth: usize,
    pub seed: u64,
}

impl Default for IForestConfig {
    fn default() -> Self {
        Self {
            n_trees: 100,
            subsample_size: 256,
            max_depth: 10,
            seed: 42,
        }
    }
}

/// A node in an isolation tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum IsolNode {
    Internal {
        feature: usize,
        threshold: f64,
        left: Box<IsolNode>,
        right: Box<IsolNode>,
    },
    Leaf {
        size: usize,
    },
}

/// Isolation Forest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsolationForest {
    trees: Vec<IsolNode>,
    config: IForestConfig,
    /// Average path length of unsuccessful search in a BST (c(n)).
    avg_path: f64,
}

impl IsolationForest {
    pub fn new(config: IForestConfig) -> Self {
        Self {
            trees: Vec::new(),
            config,
            avg_path: 0.0,
        }
    }

    fn build_tree(data: &[Vec<f64>], max_depth: usize, rng: &mut Rng) -> IsolNode {
        if data.len() <= 1 || max_depth == 0
        {
            return IsolNode::Leaf { size: data.len() };
        }

        let n_features = data[0].len();
        if n_features == 0
        {
            // Zero-dimension rows carry no split information.
            return IsolNode::Leaf { size: data.len() };
        }
        let feature = rng.gen_range(0, n_features);

        // Find min and max for the selected feature
        let min_val = data
            .iter()
            .map(|r| r[feature])
            .fold(f64::INFINITY, f64::min);
        let max_val = data
            .iter()
            .map(|r| r[feature])
            .fold(f64::NEG_INFINITY, f64::max);

        if (max_val - min_val).abs() < 1e-10
        {
            return IsolNode::Leaf { size: data.len() };
        }

        let threshold = min_val + rng.gen_f64() * (max_val - min_val);

        let (left, right): (Vec<_>, Vec<_>) =
            data.iter().cloned().partition(|r| r[feature] < threshold);

        if left.is_empty() || right.is_empty()
        {
            return IsolNode::Leaf { size: data.len() };
        }

        IsolNode::Internal {
            feature,
            threshold,
            left: Box::new(Self::build_tree(&left, max_depth - 1, rng)),
            right: Box::new(Self::build_tree(&right, max_depth - 1, rng)),
        }
    }

    fn path_length(node: &IsolNode, sample: &[f64], current_depth: usize) -> f64 {
        match node
        {
            IsolNode::Leaf { size } => current_depth as f64 + Self::c(*size),
            IsolNode::Internal {
                feature,
                threshold,
                left,
                right,
            } =>
            {
                if sample[*feature] < *threshold
                {
                    Self::path_length(left, sample, current_depth + 1)
                }
                else
                {
                    Self::path_length(right, sample, current_depth + 1)
                }
            },
        }
    }

    /// Average path length of unsuccessful search in a BST with `n` elements.
    fn c(n: usize) -> f64 {
        if n <= 1
        {
            return 0.0;
        }
        let n_f = n as f64;
        2.0 * ((n_f - 1.0).ln() + 0.577_215_664_9) - (2.0 * (n_f - 1.0) / n_f)
    }

    /// Fit the isolation forest on the dataset.
    pub fn fit(&mut self, data: &[Vec<f64>]) {
        let mut rng = Rng::new(self.config.seed);
        let n = data.len();

        // Normalize by the average path length for the *actual* subsample size.
        // Trees are built on at most `n` rows, so when the dataset is smaller
        // than the configured subsample_size the effective size is `n`; using the
        // configured size here would miscalibrate the 2^(-avg/c) score scale.
        let effective_subsample = self.config.subsample_size.min(n);
        self.avg_path = Self::c(effective_subsample);

        self.trees.clear();
        for _ in 0..self.config.n_trees
        {
            let mut indices: Vec<usize> = (0..n).collect();
            rng.shuffle(&mut indices);
            let subsample: Vec<Vec<f64>> = indices
                .iter()
                .take(self.config.subsample_size)
                .map(|&i| data[i].clone())
                .collect();
            let tree = Self::build_tree(&subsample, self.config.max_depth, &mut rng);
            self.trees.push(tree);
        }
    }

    /// Compute the anomaly score for a single sample (lower = more normal).
    pub fn score(&self, sample: &[f64]) -> f64 {
        if self.avg_path == 0.0
        {
            return 0.0;
        }
        let avg = self
            .trees
            .iter()
            .map(|t| Self::path_length(t, sample, 0))
            .sum::<f64>()
            / self.trees.len() as f64;

        // Anomaly score: 2^(-avg / c(n))
        (-avg / self.avg_path * std::f64::consts::LN_2).exp()
    }

    /// Detect anomalies: returns indices with score above threshold.
    pub fn detect_anomalies(&self, data: &[Vec<f64>], threshold: f64) -> Vec<usize> {
        data.iter()
            .enumerate()
            .filter(|(_, s)| self.score(s) > threshold)
            .map(|(i, _)| i)
            .collect()
    }

    /// Compute anomaly scores for all samples.
    pub fn anomaly_scores(&self, data: &[Vec<f64>]) -> Vec<f64> {
        data.iter().map(|s| self.score(s)).collect()
    }
}

// ===========================================================================
// 3. DBSCAN Clustering
// ===========================================================================

/// Configuration for DBSCAN.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbscanConfig {
    pub eps: f64,
    pub min_pts: usize,
}

impl Default for DbscanConfig {
    fn default() -> Self {
        Self {
            eps: 0.5,
            min_pts: 5,
        }
    }
}

/// DBSCAN clustering result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbscanResult {
    /// Cluster label for each point (-1 = noise).
    pub labels: Vec<i32>,
    /// Number of clusters found (does not count noise).
    pub n_clusters: usize,
}

/// DBSCAN: Density-Based Spatial Clustering of Applications with Noise.
pub struct Dbscan {
    config: DbscanConfig,
}

impl Dbscan {
    pub fn new(config: DbscanConfig) -> Self {
        Self { config }
    }

    fn region_query(data: &[Vec<f64>], point_idx: usize, eps: f64) -> Vec<usize> {
        data.iter()
            .enumerate()
            .filter(|(i, _)| *i != point_idx)
            .filter(|(_, p)| euclidean_distance(&data[point_idx], p) <= eps)
            .map(|(i, _)| i)
            .collect()
    }

    fn expand_cluster(
        data: &[Vec<f64>],
        labels: &mut [i32],
        point_idx: usize,
        neighbors: &[usize],
        cluster_id: i32,
        eps: f64,
        min_pts: usize,
    ) {
        labels[point_idx] = cluster_id;
        let mut seeds: Vec<usize> = neighbors.to_vec();
        let mut i = 0;
        while i < seeds.len()
        {
            let q = seeds[i];
            if labels[q] == -1
            {
                labels[q] = cluster_id;
            }
            else if labels[q] == 0
            {
                labels[q] = cluster_id;
                let q_neighbors = Self::region_query(data, q, eps);
                if q_neighbors.len() + 1 >= min_pts
                {
                    for &n in &q_neighbors
                    {
                        if !seeds.contains(&n)
                        {
                            seeds.push(n);
                        }
                    }
                }
            }
            i += 1;
        }
    }

    /// Run DBSCAN on the dataset.
    pub fn fit(&self, data: &[Vec<f64>]) -> DbscanResult {
        let n = data.len();
        let mut labels = vec![0_i32; n]; // 0 = unvisited
        let mut cluster_id = 0_i32;
        let eps = self.config.eps;
        let min_pts = self.config.min_pts;

        for i in 0..n
        {
            if labels[i] != 0
            {
                continue;
            }
            let neighbors = Self::region_query(data, i, eps);
            // Core point check: neighbors + 1 (self) >= min_pts
            if neighbors.len() + 1 < min_pts
            {
                labels[i] = -1; // noise
            }
            else
            {
                cluster_id += 1;
                Self::expand_cluster(data, &mut labels, i, &neighbors, cluster_id, eps, min_pts);
            }
        }

        DbscanResult {
            labels,
            n_clusters: cluster_id as usize,
        }
    }
}

// ===========================================================================
// 4. Local Outlier Factor (LOF)
// ===========================================================================

/// Configuration for LOF.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LofConfig {
    pub k: usize,
}

impl Default for LofConfig {
    fn default() -> Self {
        Self { k: 5 }
    }
}

/// Local Outlier Factor detector.
pub struct LocalOutlierFactor {
    config: LofConfig,
}

impl LocalOutlierFactor {
    pub fn new(config: LofConfig) -> Self {
        Self { config }
    }

    /// Compute k-distance and reachability-distance for all points.
    fn compute_distances(&self, data: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<usize>>, Vec<Vec<f64>>) {
        let n = data.len();
        let k = self.config.k.min(n.saturating_sub(1));

        // Pairwise distances
        let mut dists: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
        for i in 0..n
        {
            for j in (i + 1)..n
            {
                let d = euclidean_distance(&data[i], &data[j]);
                dists[i][j] = d;
                dists[j][i] = d;
            }
        }

        // Pass 1: for each point, its k nearest neighbors and its k-distance.
        // k-distances of *all* points must be known before reachability can be
        // computed, since reachability of i w.r.t. j depends on k_distances[j].
        let mut k_distances = vec![0.0_f64; n];
        let mut k_neighbors: Vec<Vec<usize>> = Vec::with_capacity(n);

        for i in 0..n
        {
            let mut idxs: Vec<usize> = (0..n).filter(|&j| j != i).collect();
            idxs.sort_by(|&a, &b| dists[i][a].partial_cmp(&dists[i][b]).unwrap());
            let kn = idxs[..k.min(idxs.len())].to_vec();
            let kd = if kn.is_empty()
            {
                0.0
            }
            else
            {
                dists[i][kn[kn.len() - 1]]
            };
            k_distances[i] = kd;
            k_neighbors.push(kn);
        }

        // Pass 2: reachability distance of i w.r.t. each neighbor j in its
        // neighborhood, reach_dist(i, j) = max(dist(i, j), k_distance(j)).
        let mut reachability: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n
        {
            let kn = &k_neighbors[i];
            let mut rd = Vec::with_capacity(kn.len());
            for &j in kn
            {
                let r = dists[i][j].max(k_distances[j]);
                rd.push(r);
            }
            reachability.push(rd);
        }

        (k_distances, k_neighbors, reachability)
    }

    /// Local reachability density of point i.
    fn lrd(reachability: &[Vec<f64>], idx: usize) -> f64 {
        let rd = &reachability[idx];
        if rd.is_empty()
        {
            return 0.0;
        }
        let sum: f64 = rd.iter().sum();
        if sum == 0.0
        {
            return 0.0;
        }
        rd.len() as f64 / sum
    }

    /// Compute LOF scores for all points.
    #[allow(clippy::needless_range_loop)]
    pub fn fit_predict(&self, data: &[Vec<f64>]) -> Vec<f64> {
        let n = data.len();
        if n == 0
        {
            return Vec::new();
        }
        let (_k_distances, k_neighbors, reachability) = self.compute_distances(data);

        let mut lrds = vec![0.0_f64; n];
        for i in 0..n
        {
            lrds[i] = Self::lrd(&reachability, i);
        }

        let mut lof_scores = vec![1.0_f64; n];
        for i in 0..n
        {
            if k_neighbors[i].is_empty()
            {
                continue;
            }
            let sum: f64 = k_neighbors[i].iter().map(|&j| lrds[j]).sum();
            let avg = sum / k_neighbors[i].len() as f64;
            if lrds[i] > 0.0
            {
                lof_scores[i] = avg / lrds[i];
            }
        }

        lof_scores
    }

    /// Detect outliers: indices with LOF score above threshold.
    pub fn detect_outliers(&self, data: &[Vec<f64>], threshold: f64) -> Vec<usize> {
        self.fit_predict(data)
            .iter()
            .enumerate()
            .filter(|(_, &s)| s > threshold)
            .map(|(i, _)| i)
            .collect()
    }
}

// ===========================================================================
// 5. Gaussian Mixture Model (GMM)
// ===========================================================================

/// Configuration for GMM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GmmConfig {
    pub n_components: usize,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub seed: u64,
}

impl Default for GmmConfig {
    fn default() -> Self {
        Self {
            n_components: 3,
            max_iterations: 100,
            tolerance: 1e-6,
            seed: 42,
        }
    }
}

/// A single Gaussian component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianComponent {
    pub mean: Vec<f64>,
    /// Diagonal covariance (simplified).
    pub variance: Vec<f64>,
    pub weight: f64,
}

/// Gaussian Mixture Model using EM algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianMixtureModel {
    pub components: Vec<GaussianComponent>,
    config: GmmConfig,
    log_likelihood_history: Vec<f64>,
    /// Per-dimension variance floor, fixed at fit time as a small fraction of
    /// the training data's own per-dimension variance. An absolute floor
    /// (e.g. a bare `1e-10`) silently over-smooths any dataset whose natural
    /// scale is itself far below that — inflating fitted standard deviations
    /// past the true cluster separation and collapsing distinct clusters
    /// into one. Falls back to a tiny absolute value only for a
    /// (near-)degenerate dimension with essentially zero spread.
    variance_floor: Vec<f64>,
}

impl GaussianMixtureModel {
    pub fn new(config: GmmConfig) -> Self {
        Self {
            components: Vec::new(),
            config,
            log_likelihood_history: Vec::new(),
            variance_floor: Vec::new(),
        }
    }

    /// Population variance per dimension across `data`.
    fn data_variance(data: &[Vec<f64>], dim: usize) -> Vec<f64> {
        let n = data.len() as f64;
        let mut mean = vec![0.0; dim];
        for x in data
        {
            for d in 0..dim
            {
                mean[d] += x[d];
            }
        }
        for m in mean.iter_mut()
        {
            *m /= n;
        }
        let mut var = vec![0.0; dim];
        for x in data
        {
            for d in 0..dim
            {
                let diff = x[d] - mean[d];
                var[d] += diff * diff;
            }
        }
        for v in var.iter_mut()
        {
            *v /= n;
        }
        var
    }

    fn gaussian_pdf(x: &[f64], mean: &[f64], variance: &[f64], floor: &[f64]) -> f64 {
        let dim = x.len();
        let mut log_prob = 0.0;
        for d in 0..dim
        {
            let diff = x[d] - mean[d];
            let var = variance[d].max(floor[d]);
            log_prob += -0.5 * (diff * diff / var + var.ln() + (2.0 * std::f64::consts::PI).ln());
        }
        log_prob.exp()
    }

    #[allow(clippy::needless_range_loop)]
    fn responsibilities(
        data: &[Vec<f64>],
        components: &[GaussianComponent],
        floor: &[f64],
    ) -> Vec<Vec<f64>> {
        let n = data.len();
        let k = components.len();
        let mut resp = vec![vec![0.0; k]; n];

        for i in 0..n
        {
            let mut max_log = f64::NEG_INFINITY;
            let mut log_probs = Vec::with_capacity(k);
            for c in components
            {
                let lp = Self::gaussian_pdf(&data[i], &c.mean, &c.variance, floor)
                    .max(1e-300)
                    .ln()
                    + c.weight.ln();
                log_probs.push(lp);
                if lp > max_log
                {
                    max_log = lp;
                }
            }
            let mut sum = 0.0;
            for (j, &lp) in log_probs.iter().enumerate()
            {
                resp[i][j] = (lp - max_log).exp();
                sum += resp[i][j];
            }
            for j in 0..k
            {
                resp[i][j] /= sum;
            }
        }
        resp
    }

    fn log_likelihood(data: &[Vec<f64>], components: &[GaussianComponent], floor: &[f64]) -> f64 {
        data.iter()
            .map(|x| {
                let mut total = 0.0;
                for c in components
                {
                    total += c.weight * Self::gaussian_pdf(x, &c.mean, &c.variance, floor);
                }
                total.max(1e-300).ln()
            })
            .sum()
    }

    /// Fit the GMM using the EM algorithm.
    #[allow(clippy::needless_range_loop)]
    pub fn fit(&mut self, data: &[Vec<f64>]) {
        let n = data.len();
        // Nothing to fit on empty data: leave the model with no components rather
        // than indexing `data[0]`.
        if n == 0
        {
            self.components.clear();
            self.log_likelihood_history.clear();
            return;
        }
        let dim = data[0].len();
        // Each component is seeded from a distinct sample, so we cannot have more
        // components than samples. Clamp to avoid indexing `indices` out of bounds
        // when n_components > n_samples.
        let k = self.config.n_components.min(n);
        let mut rng = Rng::new(self.config.seed);

        // The data's own per-dimension variance anchors both the initial
        // component variance and the variance floor to the data's actual
        // scale, rather than to hardcoded values that only happen to suit
        // O(1)-scale data.
        let data_var = Self::data_variance(data, dim);
        self.variance_floor = data_var.iter().map(|&v| (v * 1e-9).max(1e-300)).collect();

        // Initialize components via random sampling
        let mut indices: Vec<usize> = (0..n).collect();
        rng.shuffle(&mut indices);

        self.components = (0..k)
            .map(|j| {
                let sample = &data[indices[j]];
                GaussianComponent {
                    mean: sample.clone(),
                    variance: data_var.clone(),
                    weight: 1.0 / k as f64,
                }
            })
            .collect();

        self.log_likelihood_history.clear();

        for _ in 0..self.config.max_iterations
        {
            // E-step
            let resp = Self::responsibilities(data, &self.components, &self.variance_floor);

            // M-step
            for j in 0..k
            {
                let n_j: f64 = resp.iter().map(|r| r[j]).sum();
                if n_j < 1e-10
                {
                    continue;
                }

                // Update mean
                let mut new_mean = vec![0.0; dim];
                for (i, x) in data.iter().enumerate()
                {
                    for d in 0..dim
                    {
                        new_mean[d] += resp[i][j] * x[d];
                    }
                }
                for d in 0..dim
                {
                    new_mean[d] /= n_j;
                }

                // Update variance
                let mut new_var = vec![0.0; dim];
                for (i, x) in data.iter().enumerate()
                {
                    for d in 0..dim
                    {
                        new_var[d] += resp[i][j] * (x[d] - new_mean[d]).powi(2);
                    }
                }
                for d in 0..dim
                {
                    new_var[d] = (new_var[d] / n_j).max(self.variance_floor[d]);
                }

                // Update weight
                let new_weight = n_j / n as f64;

                self.components[j] = GaussianComponent {
                    mean: new_mean,
                    variance: new_var,
                    weight: new_weight,
                };
            }

            // Check convergence
            let ll = Self::log_likelihood(data, &self.components, &self.variance_floor);
            self.log_likelihood_history.push(ll);

            if self.log_likelihood_history.len() >= 2
            {
                let prev = self.log_likelihood_history[self.log_likelihood_history.len() - 2];
                if (ll - prev).abs() < self.config.tolerance
                {
                    break;
                }
            }
        }
    }

    /// Predict the most likely component for each sample.
    pub fn predict(&self, data: &[Vec<f64>]) -> Vec<usize> {
        let resp = Self::responsibilities(data, &self.components, &self.variance_floor);
        resp.iter()
            .map(|r| {
                r.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Predict responsibilities (soft assignment) for each sample.
    pub fn predict_proba(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        Self::responsibilities(data, &self.components, &self.variance_floor)
    }

    /// Compute log-likelihood of the data.
    pub fn score(&self, data: &[Vec<f64>]) -> f64 {
        Self::log_likelihood(data, &self.components, &self.variance_floor)
    }

    /// Compute BIC (Bayesian Information Criterion). Lower is better.
    pub fn bic(&self, data: &[Vec<f64>]) -> f64 {
        let n = data.len() as f64;
        let k = self.components.len();
        let dim = self.components.first().map_or(0, |c| c.mean.len());
        // params: each component has mean (dim) + variance (dim) + weight (1)
        let n_params = k as f64 * (2.0 * dim as f64 + 1.0) - 1.0; // weights sum to 1
        let ll = self.score(data);
        -2.0 * ll + n_params * n.ln()
    }

    /// Compute AIC (Akaike Information Criterion). Lower is better.
    pub fn aic(&self, data: &[Vec<f64>]) -> f64 {
        let dim = self.components.first().map_or(0, |c| c.mean.len());
        let k = self.components.len();
        let n_params = k as f64 * (2.0 * dim as f64 + 1.0) - 1.0;
        let ll = self.score(data);
        -2.0 * ll + 2.0 * n_params
    }
}

// ===========================================================================
// 6. One-Class SVM
// ===========================================================================

/// Configuration for One-Class SVM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OneClassSvmConfig {
    pub kernel_bandwidth: f64,
    pub nu: f64,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub seed: u64,
}

impl Default for OneClassSvmConfig {
    fn default() -> Self {
        Self {
            kernel_bandwidth: 1.0,
            nu: 0.1,
            max_iterations: 1000,
            tolerance: 1e-6,
            seed: 42,
        }
    }
}

/// Simplified One-Class SVM using the simplified SMO-like approach.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OneClassSvm {
    /// Support vectors.
    pub support_vectors: Vec<Vec<f64>>,
    /// Dual coefficients (alpha values).
    pub alphas: Vec<f64>,
    /// Threshold (rho).
    pub rho: f64,
    /// Kernel bandwidth.
    gamma: f64,
    config: OneClassSvmConfig,
}

impl OneClassSvm {
    pub fn new(config: OneClassSvmConfig) -> Self {
        let gamma = 1.0 / (config.kernel_bandwidth.powi(2));
        Self {
            support_vectors: Vec::new(),
            alphas: Vec::new(),
            rho: 0.0,
            gamma,
            config,
        }
    }

    /// RBF kernel between two vectors.
    fn rbf_kernel(&self, x: &[f64], y: &[f64]) -> f64 {
        let d = euclidean_distance(x, y);
        (-self.gamma * d * d).exp()
    }

    /// Compute the kernel matrix.
    fn kernel_matrix(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = data.len();
        let mut k = vec![vec![0.0; n]; n];
        for i in 0..n
        {
            k[i][i] = self.rbf_kernel(&data[i], &data[i]);
            for j in (i + 1)..n
            {
                let val = self.rbf_kernel(&data[i], &data[j]);
                k[i][j] = val;
                k[j][i] = val;
            }
        }
        k
    }

    /// Simplified training using sequential minimal optimization heuristic.
    #[allow(clippy::needless_range_loop)]
    pub fn fit(&mut self, data: &[Vec<f64>]) {
        let n = data.len();
        let nu = self.config.nu;
        let mut rng = Rng::new(self.config.seed);

        let k = self.kernel_matrix(data);
        let mut alphas = vec![0.0_f64; n];

        // Box-constraint upper bound for the nu-OCSVM dual: 0 <= alpha_i <= 1/(nu*n).
        let upper = 1.0 / (nu * n as f64);

        // Initialize: set a fraction to 1/nu*n to represent support vectors
        let mut indices: Vec<usize> = (0..n).collect();
        rng.shuffle(&mut indices);
        let n_sv = ((nu * n as f64) as usize).max(1).min(n);
        for i in indices.iter().take(n_sv)
        {
            alphas[*i] = 1.0 / (n_sv as f64);
        }

        // Iterate: adjust alphas to maximize margin
        for _ in 0..self.config.max_iterations
        {
            let mut max_change = 0.0_f64;

            for i in 0..n
            {
                // Decision value for sample i
                let f_i: f64 = alphas.iter().enumerate().map(|(j, &a)| a * k[i][j]).sum();

                let error = f_i - self.rho;

                if (alphas[i] > 1e-8 && error < -self.config.tolerance)
                    || (alphas[i] < upper + 1e-8 && error > self.config.tolerance)
                {
                    // Find j with maximum |error_i - error_j|
                    let mut best_j = (i + 1) % n;
                    let mut best_improve = 0.0_f64;

                    for j in 0..n
                    {
                        if j == i
                        {
                            continue;
                        }
                        let f_j: f64 = alphas.iter().enumerate().map(|(l, &a)| a * k[j][l]).sum();
                        let err_j = f_j - self.rho;
                        let improve = (error - err_j).abs();
                        if improve > best_improve
                        {
                            best_improve = improve;
                            best_j = j;
                        }
                    }

                    // Joint optimization of i and j
                    let f_j: f64 = alphas
                        .iter()
                        .enumerate()
                        .map(|(l, &a)| a * k[best_j][l])
                        .sum();
                    let err_j = f_j - self.rho;

                    let eta = k[i][i] + k[best_j][best_j] - 2.0 * k[i][best_j];
                    if eta < 1e-10
                    {
                        continue;
                    }

                    let mut new_alpha_i = alphas[i] + (error - err_j) / eta;
                    let mut new_alpha_j = alphas[best_j] - (error - err_j) / eta;

                    // Clip to [0, 1/(nu*n)] for one-class.
                    new_alpha_i = new_alpha_i.max(0.0).min(upper);
                    new_alpha_j = new_alpha_j.max(0.0).min(upper);

                    let change =
                        (new_alpha_i - alphas[i]).abs() + (new_alpha_j - alphas[best_j]).abs();
                    if change > max_change
                    {
                        max_change = change;
                    }

                    alphas[i] = new_alpha_i;
                    alphas[best_j] = new_alpha_j;
                }
            }

            // Enforce the dual equality constraint sum(alpha) = 1. Box clipping
            // during the joint update can perturb the sum, so renormalize.
            let alpha_sum: f64 = alphas.iter().sum();
            if alpha_sum > 1e-12
            {
                for a in alphas.iter_mut()
                {
                    *a /= alpha_sum;
                }
            }

            // Update rho
            let mut rho_sum = 0.0;
            let mut rho_count = 0;
            for i in 0..n
            {
                if alphas[i] > 1e-8 && alphas[i] < upper - 1e-8
                {
                    let f_i: f64 = alphas.iter().enumerate().map(|(j, &a)| a * k[i][j]).sum();
                    rho_sum += f_i;
                    rho_count += 1;
                }
            }
            if rho_count > 0
            {
                self.rho = rho_sum / rho_count as f64;
            }

            if max_change < self.config.tolerance
            {
                break;
            }
        }

        // Extract support vectors
        self.support_vectors.clear();
        self.alphas.clear();
        for i in 0..n
        {
            if alphas[i] > 1e-8
            {
                self.support_vectors.push(data[i].clone());
                self.alphas.push(alphas[i]);
            }
        }
    }

    /// Decision function: returns the signed distance from the decision boundary.
    pub fn decision_function(&self, x: &[f64]) -> f64 {
        let val: f64 = self
            .support_vectors
            .iter()
            .zip(self.alphas.iter())
            .map(|(sv, &a)| a * self.rbf_kernel(x, sv))
            .sum();
        val - self.rho
    }

    /// Predict: returns true if inlier, false if outlier.
    pub fn predict(&self, data: &[Vec<f64>]) -> Vec<bool> {
        data.iter()
            .map(|x| self.decision_function(x) >= 0.0)
            .collect()
    }

    /// Detect outliers: returns indices classified as outliers.
    pub fn detect_outliers(&self, data: &[Vec<f64>]) -> Vec<usize> {
        data.iter()
            .enumerate()
            .filter(|(_, x)| self.decision_function(x) < 0.0)
            .map(|(i, _)| i)
            .collect()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_2d_data() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 1.0],
            vec![1.1, 1.2],
            vec![0.9, 0.8],
            vec![1.2, 1.1],
            vec![0.8, 0.9],
            vec![1.0, 1.0],
            vec![5.0, 5.0], // outlier
            vec![5.1, 4.9],
            vec![4.9, 5.1],
            vec![5.2, 5.0],
        ]
    }

    // ---- Autoencoder tests ----

    #[test]
    fn test_autoencoder_forward() {
        let config = AutoencoderConfig {
            input_dim: 4,
            hidden_dim: 2,
            learning_rate: 0.1,
            epochs: 10,
        };
        let ae = Autoencoder::new(config);
        let input = vec![0.5, 0.3, 0.8, 0.1];
        let output = ae.forward(&input);
        assert_eq!(output.len(), 4);
        for &v in &output
        {
            assert!((0.0..=1.0).contains(&v), "output out of [0,1] range");
        }
    }

    #[test]
    fn test_autoencoder_reconstruction_error() {
        let config = AutoencoderConfig {
            input_dim: 3,
            hidden_dim: 2,
            learning_rate: 0.1,
            epochs: 10,
        };
        let ae = Autoencoder::new(config);
        let input = vec![0.5, 0.5, 0.5];
        let err = ae.reconstruction_error(&input);
        assert!(err >= 0.0);
    }

    #[test]
    fn test_autoencoder_train_reduces_error() {
        let config = AutoencoderConfig {
            input_dim: 3,
            hidden_dim: 2,
            learning_rate: 0.5,
            epochs: 200,
        };
        let mut ae = Autoencoder::new(config);
        let data: Vec<Vec<f64>> = (0..50)
            .map(|i| {
                let x = i as f64 / 50.0;
                vec![x, x * 0.5, x * 0.2]
            })
            .collect();

        let err_before: f64 =
            data.iter().map(|s| ae.reconstruction_error(s)).sum::<f64>() / data.len() as f64;
        ae.train(&data);
        let err_after: f64 =
            data.iter().map(|s| ae.reconstruction_error(s)).sum::<f64>() / data.len() as f64;
        assert!(err_after < err_before, "training should reduce error");
    }

    #[test]
    fn test_autoencoder_anomaly_detection() {
        let config = AutoencoderConfig {
            input_dim: 2,
            hidden_dim: 1,
            learning_rate: 0.5,
            epochs: 300,
        };
        let mut ae = Autoencoder::new(config);
        let data = vec![
            vec![1.0, 1.0],
            vec![1.1, 1.2],
            vec![0.9, 0.8],
            vec![1.2, 1.1],
            vec![0.8, 0.9],
            vec![1.0, 1.0],
        ];
        ae.train(&data);

        let scores = ae.anomaly_scores(&data);
        assert_eq!(scores.len(), 6);
        for s in &scores
        {
            assert!(*s >= 0.0);
        }
    }

    // ---- Isolation Forest tests ----

    #[test]
    fn test_iforest_fit_and_score() {
        let config = IForestConfig {
            n_trees: 10,
            subsample_size: 8,
            max_depth: 5,
            seed: 42,
        };
        let mut iforest = IsolationForest::new(config);
        let data = simple_2d_data();
        iforest.fit(&data);

        let scores = iforest.anomaly_scores(&data);
        assert_eq!(scores.len(), data.len());
        for s in &scores
        {
            assert!(*s >= 0.0 && *s <= 1.0, "score out of [0,1] range: {}", s);
        }
    }

    #[test]
    fn test_iforest_outlier_detection() {
        let config = IForestConfig {
            n_trees: 50,
            subsample_size: 8,
            max_depth: 10,
            seed: 42,
        };
        let mut iforest = IsolationForest::new(config);
        let data = simple_2d_data();
        iforest.fit(&data);

        // The (5,5) cluster (indices 6..10) is the planted outlier group; the
        // dense (1,1) cluster (indices 0..6) is the inlier group. Anomaly scores
        // are normalized as 2^(-avg_path / c(n)), so the planted outliers must
        // score strictly above the inliers. We assert this relative ordering
        // rather than a fixed threshold (which depends on the exact c(n) scale).
        let scores = iforest.anomaly_scores(&data);
        let max_inlier = scores[0..6].iter().cloned().fold(f64::MIN, f64::max);
        let mean_outlier = scores[6..10].iter().sum::<f64>() / 4.0;
        assert!(
            mean_outlier > max_inlier,
            "planted outliers should score above inliers: mean_outlier={} max_inlier={}",
            mean_outlier,
            max_inlier
        );

        // Detection with a threshold midway between the two groups must flag at
        // least one of the planted outliers.
        let threshold = (max_inlier + mean_outlier) / 2.0;
        let anomalies = iforest.detect_anomalies(&data, threshold);
        assert!(!anomalies.is_empty(), "should detect at least one anomaly");
        assert!(
            anomalies.iter().any(|&i| (6..10).contains(&i)),
            "a detected anomaly should be in the planted outlier group"
        );
    }

    #[test]
    fn test_iforest_score_range() {
        let config = IForestConfig {
            n_trees: 10,
            subsample_size: 5,
            max_depth: 5,
            seed: 1,
        };
        let mut iforest = IsolationForest::new(config);
        let data: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64, i as f64]).collect();
        iforest.fit(&data);

        for sample in &data
        {
            let score = iforest.score(sample);
            assert!((0.0..=1.0).contains(&score));
        }
    }

    // ---- DBSCAN tests ----

    #[test]
    fn test_dbscan_basic() {
        let config = DbscanConfig {
            eps: 0.15,
            min_pts: 2,
        };
        let dbscan = Dbscan::new(config);
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.2, 0.2],
            vec![5.0, 5.0],
            vec![5.1, 5.1],
        ];
        let result = dbscan.fit(&data);
        assert!(
            result.n_clusters >= 2,
            "should find at least 2 clusters, got {} labels={:?}",
            result.n_clusters,
            result.labels
        );
        assert!(result.labels.contains(&1));
        assert!(result.labels.contains(&2));
    }

    #[test]
    fn test_dbscan_noise() {
        let config = DbscanConfig {
            eps: 0.3,
            min_pts: 3,
        };
        let dbscan = Dbscan::new(config);
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.2, 0.2],
            vec![10.0, 10.0],
        ];
        let result = dbscan.fit(&data);
        assert!(
            result.labels.contains(&-1),
            "isolated point should be noise"
        );
    }

    #[test]
    fn test_dbscan_single_cluster() {
        let config = DbscanConfig {
            eps: 2.0,
            min_pts: 2,
        };
        let dbscan = Dbscan::new(config);
        let data = vec![
            vec![0.0, 0.0],
            vec![0.5, 0.5],
            vec![1.0, 1.0],
            vec![1.5, 1.5],
        ];
        let result = dbscan.fit(&data);
        assert_eq!(result.n_clusters, 1, "all points in one cluster");
        assert!(result.labels.iter().all(|&l| l == 1));
    }

    // ---- LOF tests ----

    #[test]
    fn test_lof_scores() {
        let config = LofConfig { k: 3 };
        let lof = LocalOutlierFactor::new(config);
        let data = simple_2d_data();
        let scores = lof.fit_predict(&data);
        assert_eq!(scores.len(), data.len());
        for s in &scores
        {
            assert!(*s >= 0.0);
        }
    }

    #[test]
    fn test_lof_outlier_detection() {
        let config = LofConfig { k: 3 };
        let lof = LocalOutlierFactor::new(config);
        // Outlier has very different location from the main cluster
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![50.0, 50.0],
        ];
        let scores = lof.fit_predict(&data);
        // The outlier (50,50) should have a higher LOF than cluster points
        assert!(
            scores[4] > scores[0],
            "outlier LOF {} should exceed inlier LOF {}",
            scores[4],
            scores[0]
        );
    }

    #[test]
    fn test_lof_normal_points() {
        let config = LofConfig { k: 2 };
        let lof = LocalOutlierFactor::new(config);
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
        ];
        let scores = lof.fit_predict(&data);
        // All points in same cluster should have LOF close to 1
        for s in &scores
        {
            assert!(*s < 2.0, "tight cluster points should have low LOF");
        }
    }

    #[test]
    fn test_lof_symmetric_all_ones() {
        // Regression test for the reachability bug: reachability of point i
        // w.r.t. neighbor j used k_distances[j] before it was filled for j > i,
        // silently collapsing reach_dist(i, j) to the raw distance. On a
        // perfectly symmetric configuration every point has identical local
        // structure, so the correct LOF is exactly 1.0 for all points. The
        // buggy code produced order-dependent values (~0.863, ~1.008, ~1.153).
        let config = LofConfig { k: 3 };
        let lof = LocalOutlierFactor::new(config);
        let data = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let scores = lof.fit_predict(&data);
        assert_eq!(scores.len(), data.len());
        for (i, s) in scores.iter().enumerate()
        {
            assert!(
                (s - 1.0).abs() < 1e-9,
                "symmetric point {} should have LOF exactly 1.0, got {}",
                i,
                s
            );
        }
    }

    // ---- GMM tests ----

    #[test]
    fn test_gmm_fit() {
        let config = GmmConfig {
            n_components: 2,
            max_iterations: 50,
            tolerance: 1e-6,
            seed: 42,
        };
        let mut gmm = GaussianMixtureModel::new(config);
        let data: Vec<Vec<f64>> = (0..100)
            .map(|i| {
                if i < 50
                {
                    vec![i as f64 / 50.0, i as f64 / 50.0]
                }
                else
                {
                    vec![
                        (i as f64 - 50.0) / 50.0 + 5.0,
                        (i as f64 - 50.0) / 50.0 + 5.0,
                    ]
                }
            })
            .collect();
        gmm.fit(&data);
        assert_eq!(gmm.components.len(), 2);
    }

    #[test]
    fn test_gmm_predict() {
        let config = GmmConfig {
            n_components: 2,
            max_iterations: 50,
            tolerance: 1e-6,
            seed: 42,
        };
        let mut gmm = GaussianMixtureModel::new(config);
        let data: Vec<Vec<f64>> = (0..100)
            .map(|i| {
                if i < 50
                {
                    vec![i as f64 / 50.0, i as f64 / 50.0]
                }
                else
                {
                    vec![
                        (i as f64 - 50.0) / 50.0 + 5.0,
                        (i as f64 - 50.0) / 50.0 + 5.0,
                    ]
                }
            })
            .collect();
        gmm.fit(&data);
        let labels = gmm.predict(&data);
        assert_eq!(labels.len(), 100);
        // All labels should be 0 or 1
        for &l in &labels
        {
            assert!(l < 2);
        }
    }

    #[test]
    fn test_gmm_predict_proba() {
        let config = GmmConfig {
            n_components: 3,
            max_iterations: 30,
            tolerance: 1e-6,
            seed: 42,
        };
        let mut gmm = GaussianMixtureModel::new(config);
        let data: Vec<Vec<f64>> = (0..60)
            .map(|i| vec![i as f64 / 20.0, (i % 3) as f64])
            .collect();
        gmm.fit(&data);
        let proba = gmm.predict_proba(&data);
        assert_eq!(proba.len(), 60);
        for p in &proba
        {
            assert_eq!(p.len(), 3);
            let sum: f64 = p.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "probabilities should sum to 1");
        }
    }

    #[test]
    fn test_gmm_log_likelihood_increases() {
        let config = GmmConfig {
            n_components: 2,
            max_iterations: 50,
            tolerance: 1e-10,
            seed: 42,
        };
        let mut gmm = GaussianMixtureModel::new(config);
        let data: Vec<Vec<f64>> = (0..50)
            .map(|i| vec![i as f64 / 10.0, (i as f64).sin()])
            .collect();
        gmm.fit(&data);
        let history = &gmm.log_likelihood_history;
        assert!(history.len() >= 2);
        // Log-likelihood should be non-decreasing (mostly)
        for i in 1..history.len()
        {
            assert!(
                history[i] >= history[i - 1] - 1e-4,
                "log-likelihood should not decrease significantly: {} -> {}",
                history[i - 1],
                history[i]
            );
        }
    }

    #[test]
    fn test_gmm_bic_aic() {
        let config = GmmConfig {
            n_components: 2,
            max_iterations: 30,
            tolerance: 1e-6,
            seed: 42,
        };
        let mut gmm = GaussianMixtureModel::new(config);
        let data: Vec<Vec<f64>> = (0..40)
            .map(|i| vec![i as f64 / 10.0, (i as f64).cos()])
            .collect();
        gmm.fit(&data);
        let bic = gmm.bic(&data);
        let aic = gmm.aic(&data);
        assert!(bic.is_finite());
        assert!(aic.is_finite());
        // BIC should be >= AIC (BIC has stronger penalty)
        assert!(bic >= aic);
    }

    #[test]
    fn test_gmm_separates_tiny_scale_clusters() {
        // Two clusters separated by 1e-6, each jittered by only ~1e-8 — well
        // separated relative to their own true variance (~1e-16), but far
        // smaller than a hardcoded initial variance of 1.0 or an absolute
        // variance floor of 1e-10 (whose implied std of 1e-5 alone dwarfs the
        // 1e-6 separation). A GMM whose initialization and variance floor
        // scale with the data should still recover the two clusters cleanly.
        let scale = 1e-6;
        let n = 100;
        let half = n / 2;
        let data: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                if i < half
                {
                    let j = i as f64;
                    vec![scale * 0.01 * (j / half as f64 - 0.5)]
                }
                else
                {
                    let j = (i - half) as f64;
                    vec![scale + scale * 0.01 * (j / half as f64 - 0.5)]
                }
            })
            .collect();

        let config = GmmConfig {
            n_components: 2,
            max_iterations: 100,
            tolerance: 1e-15,
            seed: 42,
        };
        let mut gmm = GaussianMixtureModel::new(config);
        gmm.fit(&data);
        let labels = gmm.predict(&data);

        // Each half should be assigned overwhelmingly to a single, shared
        // label, and the two halves' labels should differ.
        let count_0_first_half = labels[..half].iter().filter(|&&l| l == 0).count();
        let count_0_second_half = labels[half..].iter().filter(|&&l| l == 0).count();
        let first_half_purity = count_0_first_half.max(half - count_0_first_half);
        let second_half_purity = count_0_second_half.max(half - count_0_second_half);
        assert!(
            first_half_purity >= half - half / 20,
            "first half should be >=95% one label, got {count_0_first_half}/{half} label-0"
        );
        assert!(
            second_half_purity >= half - half / 20,
            "second half should be >=95% one label, got {count_0_second_half}/{half} label-0"
        );
        // The dominant label of each half must differ (else it's one big blob).
        let first_half_label = count_0_first_half > half / 2;
        let second_half_label = count_0_second_half > half / 2;
        assert_ne!(
            first_half_label, second_half_label,
            "the two clusters should be assigned different labels"
        );
    }

    // ---- One-Class SVM tests ----

    #[test]
    fn test_one_class_svm_fit() {
        let config = OneClassSvmConfig {
            kernel_bandwidth: 1.0,
            nu: 0.2,
            max_iterations: 100,
            tolerance: 1e-4,
            seed: 42,
        };
        let mut svm = OneClassSvm::new(config);
        let data: Vec<Vec<f64>> = (0..30)
            .map(|i| vec![i as f64 / 10.0, (i as f64 / 10.0).sin()])
            .collect();
        svm.fit(&data);
        assert!(!svm.support_vectors.is_empty());
    }

    #[test]
    fn test_one_class_svm_predict() {
        let config = OneClassSvmConfig {
            kernel_bandwidth: 2.0,
            nu: 0.1,
            max_iterations: 200,
            tolerance: 1e-4,
            seed: 42,
        };
        let mut svm = OneClassSvm::new(config);
        let data: Vec<Vec<f64>> = (0..50)
            .map(|i| {
                let x = i as f64 / 10.0;
                vec![x, x * 0.5]
            })
            .collect();
        svm.fit(&data);

        let predictions = svm.predict(&data);
        assert_eq!(predictions.len(), 50);
        // Most training points should be inliers
        let inliers = predictions.iter().filter(|&&p| p).count();
        assert!(inliers > 0, "at least some points should be inliers");
    }

    #[test]
    fn test_one_class_svm_decision_function() {
        let config = OneClassSvmConfig {
            kernel_bandwidth: 1.0,
            nu: 0.1,
            max_iterations: 100,
            tolerance: 1e-4,
            seed: 42,
        };
        let mut svm = OneClassSvm::new(config);
        let data: Vec<Vec<f64>> = (0..30)
            .map(|i| vec![i as f64 / 10.0, i as f64 / 10.0])
            .collect();
        svm.fit(&data);

        // Point near the data should have positive decision value
        let near = vec![1.5, 1.5];
        let far = vec![100.0, 100.0];

        let df_near = svm.decision_function(&near);
        let df_far = svm.decision_function(&far);

        // The far point should have lower decision value
        assert!(df_far < df_near, "far point should be more anomalous");
    }

    #[test]
    fn test_one_class_svm_detect_outliers() {
        let config = OneClassSvmConfig {
            kernel_bandwidth: 1.0,
            nu: 0.1,
            max_iterations: 200,
            tolerance: 1e-4,
            seed: 42,
        };
        let mut svm = OneClassSvm::new(config);
        let mut data: Vec<Vec<f64>> = (0..30)
            .map(|i| vec![i as f64 / 10.0, i as f64 / 10.0])
            .collect();
        data.push(vec![50.0, 50.0]);
        data.push(vec![100.0, 100.0]);
        svm.fit(&data);

        let _outliers = svm.detect_outliers(&data);
        let df_inlier = svm.decision_function(&data[0]);
        let df_outlier = svm.decision_function(&data[31]);
        assert!(
            df_outlier < df_inlier,
            "outlier decision {} should be less than inlier decision {}",
            df_outlier,
            df_inlier
        );
    }

    // ---- Edge case tests ----

    #[test]
    fn test_empty_data_dbscan() {
        let config = DbscanConfig {
            eps: 0.5,
            min_pts: 2,
        };
        let dbscan = Dbscan::new(config);
        let result = dbscan.fit(&[]);
        assert_eq!(result.n_clusters, 0);
        assert!(result.labels.is_empty());
    }

    #[test]
    fn test_single_point_dbscan() {
        let config = DbscanConfig {
            eps: 0.5,
            min_pts: 2,
        };
        let dbscan = Dbscan::new(config);
        let result = dbscan.fit(&[vec![1.0, 1.0]]);
        assert_eq!(result.labels, vec![-1]); // noise (not enough neighbors)
    }

    #[test]
    fn test_empty_data_lof() {
        let config = LofConfig { k: 3 };
        let lof = LocalOutlierFactor::new(config);
        let scores = lof.fit_predict(&[]);
        assert!(scores.is_empty());
    }

    #[test]
    fn test_two_points_lof() {
        let config = LofConfig { k: 1 };
        let lof = LocalOutlierFactor::new(config);
        let data = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let scores = lof.fit_predict(&data);
        assert_eq!(scores.len(), 2);
    }

    #[test]
    fn test_utility_functions() {
        assert!((euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]) - 5.0).abs() < 1e-10);
        assert!((dot(&[1.0, 2.0], &[3.0, 4.0]) - 11.0).abs() < 1e-10);
        assert!((norm2(&[3.0, 4.0]) - 5.0).abs() < 1e-10);
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(100.0) > 0.99);
        assert!(sigmoid(-100.0) < 0.01);
    }

    #[test]
    fn test_rng_determinism() {
        let mut rng1 = Rng::new(42);
        let mut rng2 = Rng::new(42);
        for _ in 0..100
        {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    // ---- Oracle tests (closed-form expected values) ----

    #[test]
    fn test_autoencoder_encode_of_zero_is_half() {
        // encode returns sigmoid(dot(w,x)+b). For x = all-zeros, dot(w,0)=0 and
        // the encoder bias is initialized to zeros, so each pre-activation is
        // exactly 0 and sigmoid(0)=0.5 regardless of the random weights.
        let config = AutoencoderConfig {
            input_dim: 3,
            hidden_dim: 2,
            learning_rate: 0.1,
            epochs: 0,
        };
        let ae = Autoencoder::new(config);
        let hidden = ae.encode(&[0.0, 0.0, 0.0]);
        assert_eq!(hidden.len(), 2);
        for &h in &hidden
        {
            assert!((h - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn test_gmm_single_component_fits_sample_mean_and_variance() {
        // With k=1 every responsibility is exactly 1, so the M-step yields the
        // population mean [1,1] and population variance [1,1]. The single-Gaussian
        // log-likelihood of the four points is -11.351508265637381.
        let config = GmmConfig {
            n_components: 1,
            max_iterations: 10,
            tolerance: 1e-12,
            seed: 42,
        };
        let mut gmm = GaussianMixtureModel::new(config);
        let data = vec![
            vec![0.0, 0.0],
            vec![2.0, 0.0],
            vec![0.0, 2.0],
            vec![2.0, 2.0],
        ];
        gmm.fit(&data);
        assert_eq!(gmm.components.len(), 1);
        let comp = &gmm.components[0];
        assert!((comp.mean[0] - 1.0).abs() < 1e-9);
        assert!((comp.mean[1] - 1.0).abs() < 1e-9);
        assert!((comp.variance[0] - 1.0).abs() < 1e-9);
        assert!((comp.variance[1] - 1.0).abs() < 1e-9);
        assert!((comp.weight - 1.0).abs() < 1e-12);
        assert!((gmm.score(&data) - (-11.351508265637381)).abs() < 1e-6);
    }

    #[test]
    fn test_iforest_c_normalization_value() {
        // c(n) = 2*(ln(n-1)+gamma) - 2*(n-1)/n. These are the canonical
        // isolation-forest normalization constants.
        assert!((IsolationForest::c(256) - 10.244770920116851).abs() < 1e-3);
        assert!((IsolationForest::c(8) - 3.2962516279106264).abs() < 1e-3);
        assert!((IsolationForest::c(5) - 2.327020052039781).abs() < 1e-3);
    }

    #[test]
    fn test_dbscan_two_separated_pairs_exact_labels() {
        // Points 0,1 are 0.1 apart (<= eps) and form cluster 1; points 2,3 are
        // 0.1 apart and form cluster 2. The two pairs are ~7.07 apart so they
        // never merge. Processing order 0..3 assigns cluster ids 1 then 2.
        let config = DbscanConfig {
            eps: 0.5,
            min_pts: 2,
        };
        let dbscan = Dbscan::new(config);
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![5.0, 5.0],
            vec![5.0, 5.1],
        ];
        let result = dbscan.fit(&data);
        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.labels, vec![1, 1, 2, 2]);
    }

    #[test]
    fn test_one_class_svm_decision_at_support_vector() {
        // decision_function(x) = sum_i alpha_i * rbf(x, sv_i) - rho. With a single
        // support vector sv=[0,0], alpha=1, gamma=1 (bandwidth=1) and rho=0:
        //   x=[0,0]: rbf=exp(0)=1 => decision = 1.0
        //   x=[1,0]: rbf=exp(-1)=0.36787944117144233 => decision = 0.36787944117144233
        let config = OneClassSvmConfig {
            kernel_bandwidth: 1.0,
            nu: 0.1,
            max_iterations: 0,
            tolerance: 1e-6,
            seed: 42,
        };
        let mut svm = OneClassSvm::new(config);
        svm.support_vectors = vec![vec![0.0, 0.0]];
        svm.alphas = vec![1.0];
        svm.rho = 0.0;
        assert!((svm.decision_function(&[0.0, 0.0]) - 1.0).abs() < 1e-12);
        assert!((svm.decision_function(&[1.0, 0.0]) - 0.36787944117144233).abs() < 1e-12);
    }

    // ---- Regression tests for degenerate-input fixes ----

    #[test]
    fn test_gmm_fit_empty_data_no_panic() {
        // Regression: fit() read data[0] unconditionally and panicked on empty
        // input. It should now leave the model empty instead.
        let config = GmmConfig {
            n_components: 3,
            max_iterations: 10,
            tolerance: 1e-6,
            seed: 42,
        };
        let mut gmm = GaussianMixtureModel::new(config);
        gmm.fit(&[]);
        assert!(gmm.components.is_empty());
        assert!(gmm.log_likelihood_history.is_empty());
    }

    #[test]
    fn test_gmm_fit_more_components_than_samples_no_panic() {
        // Regression: with n_components > n_samples the initializer indexed
        // `indices[j]` out of bounds and panicked. The component count must be
        // clamped to the number of samples, and weights must still sum to 1.
        let config = GmmConfig {
            n_components: 5,
            max_iterations: 20,
            tolerance: 1e-9,
            seed: 42,
        };
        let mut gmm = GaussianMixtureModel::new(config);
        let data = vec![vec![0.0, 0.0], vec![5.0, 5.0]];
        gmm.fit(&data);
        assert_eq!(gmm.components.len(), 2, "clamped to n_samples");
        let weight_sum: f64 = gmm.components.iter().map(|c| c.weight).sum();
        assert!(
            (weight_sum - 1.0).abs() < 1e-9,
            "mixture weights should sum to 1, got {}",
            weight_sum
        );
        // Model is usable afterwards.
        let labels = gmm.predict(&data);
        assert_eq!(labels.len(), 2);
    }

    #[test]
    fn test_rng_gen_range_empty_range_no_panic() {
        // Regression: gen_range(lo, hi) computed `% (hi - lo)` and panicked with a
        // remainder-by-zero when lo == hi. It should now return lo.
        let mut rng = Rng::new(7);
        assert_eq!(rng.gen_range(0, 0), 0);
        assert_eq!(rng.gen_range(4, 4), 4);
    }

    #[test]
    fn test_iforest_fit_zero_dimension_rows_no_panic() {
        // Regression: building a tree selected a feature via gen_range(0, 0) on
        // zero-dimension rows (remainder-by-zero) and, once that returned 0,
        // indexed an empty row out of bounds. Fit must complete without panic and
        // produce well-defined scores in [0, 1]. Zero-dim rows carry no split
        // information, so every tree is a single Leaf of the subsample size; with
        // the effective-subsample normalizer the path c(size) equals the
        // normalizer c(size), giving score exactly 0.5 for all rows.
        let config = IForestConfig {
            n_trees: 5,
            subsample_size: 4,
            max_depth: 5,
            seed: 42,
        };
        let mut iforest = IsolationForest::new(config);
        let data: Vec<Vec<f64>> = vec![Vec::new(); 6];
        iforest.fit(&data);
        let scores = iforest.anomaly_scores(&data);
        assert_eq!(scores.len(), 6);
        for s in &scores
        {
            assert!((0.0..=1.0).contains(s), "score out of range: {}", s);
            assert!(
                (s - 0.5).abs() < 1e-12,
                "root-leaf score should be 0.5: {}",
                s
            );
        }
    }

    #[test]
    fn test_iforest_score_calibrated_to_effective_subsample() {
        // Regression: when the dataset is smaller than the configured
        // subsample_size, fit() normalized by c(subsample_size) instead of
        // c(n). With n < subsample_size the effective subsample is n, so the
        // normalizer must equal c(n). We verify this via a point that isolates
        // at depth ~c(n): a lone far-away outlier in a small dataset should
        // score close to 1 (very anomalous). With the buggy larger normalizer
        // the score would be depressed well below the correctly-calibrated one.
        let n = 8usize;
        let mut data: Vec<Vec<f64>> = (0..n - 1).map(|i| vec![i as f64 * 0.01]).collect();
        data.push(vec![1000.0]); // clear outlier

        let make = |subsample_size: usize| {
            let cfg = IForestConfig {
                n_trees: 100,
                subsample_size,
                max_depth: 10,
                seed: 123,
            };
            let mut f = IsolationForest::new(cfg);
            f.fit(&data);
            f
        };

        // subsample_size larger than n: effective subsample is n, so avg_path
        // must be c(n), NOT c(subsample_size).
        let big = make(256);
        assert!(
            (big.avg_path - IsolationForest::c(n)).abs() < 1e-12,
            "normalizer must be c(n)={} not c(256)={}, got {}",
            IsolationForest::c(n),
            IsolationForest::c(256),
            big.avg_path
        );

        // Correctly calibrated, the outlier scores strictly higher than under
        // the (buggy) inflated normalizer would allow: compare against a forest
        // whose subsample equals n exactly — the scores must match.
        let exact = make(n);
        assert!((big.avg_path - exact.avg_path).abs() < 1e-12);
        let s_big = big.score(&data[n - 1]);
        let s_exact = exact.score(&data[n - 1]);
        assert!((s_big - s_exact).abs() < 1e-12);
        // Sanity: the lone outlier is highly anomalous.
        assert!(s_big > 0.5, "outlier should score high, got {}", s_big);
    }
}
