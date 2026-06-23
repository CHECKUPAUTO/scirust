//! Predictive model — a small MLP trained on the autodiff tape.
//!
//! The model takes a **feature vector** (indicators flattened into a fixed-size
//! window) and outputs a single scalar: the expected next-bar return.
//!
//! Training is deterministic (seeded PCG) and uses Adam. The trained weights
//! are serialisable so they can be sealed into a proof file alongside the
//! market snapshot.

use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::autodiff::reverse::{Tape, Tensor, Var};
use scirust_core::nn::activation::ReLU;
use scirust_core::nn::init::{KaimingNormal, Zeros};
use scirust_core::nn::linear::Linear;
use scirust_core::nn::module::Module;
use scirust_core::nn::rng::PcgEngine;
use scirust_core::nn::sequential::Sequential;
use serde::{Deserialize, Serialize};

/// A lightweight predictor: input features → scalar return prediction.
pub struct PricePredictor {
    pub net: Sequential,
    pub input_dim: usize,
}

/// Serialised weights — can be hashed for the proof file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelWeights {
    pub layers: Vec<Vec<f32>>,
    pub input_dim: usize,
    pub fingerprint: String,
}

impl PricePredictor {
    /// Build a fresh model with given hidden sizes (e.g. `[16, 8]`).
    pub fn new(input_dim: usize, hidden: &[usize], seed: u64) -> Self {
        let mut rng = PcgEngine::new(seed);
        let mut net = Sequential::new();
        let mut prev = input_dim;
        for &h in hidden
        {
            net = net.add(Linear::new(prev, h, &KaimingNormal, &Zeros, &mut rng));
            net = net.add(ReLU::new());
            prev = h;
        }
        net = net.add(Linear::new(prev, 1, &KaimingNormal, &Zeros, &mut rng));
        Self { net, input_dim }
    }

    /// Forward pass on a given tape — returns the scalar output `Var`.
    pub fn forward<'t>(&mut self, tape: &'t Tape, x: Var<'t>) -> Var<'t> {
        self.net.forward(tape, x)
    }

    /// Train one step: forward → MSE loss → backward → Adam step.
    /// Returns the loss value for logging.
    pub fn train_step(&mut self, features: &[f32], target: f32, lr: f32) -> f32 {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(features.to_vec(), 1, features.len()));
        let pred = self.forward(&tape, x);
        let pred_val = tape.value(pred.idx()).data[0];
        let loss_val = (pred_val - target).powi(2);
        let y = tape.input(Tensor::from_vec(vec![target], 1, 1));
        let diff = pred.try_sub(y).unwrap();
        let loss = diff.mul(diff);
        tape.backward(loss.idx());
        let mut opt = Adam::new(lr);
        opt.step(&self.net.parameter_indices(), &tape);
        self.net.sync(&tape);
        loss_val
    }

    /// Predict the next-bar return (deterministic forward, no tape backward).
    pub fn predict(&mut self, features: &[f32]) -> f32 {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(features.to_vec(), 1, features.len()));
        let pred = self.forward(&tape, x);
        tape.value(pred.idx()).data[0]
    }

    /// Export weights as a flat vector (for the proof file).
    /// Each layer's weight + bias are concatenated into one entry.
    pub fn export_weights(&self) -> ModelWeights {
        let sd = self.net.state_dict();
        let mut keys: Vec<&String> = sd.keys().collect();
        keys.sort();
        let mut layers = Vec::new();
        let mut current = Vec::new();
        let mut last_prefix = String::new();
        for key in keys
        {
            let tensor = sd.get(key).unwrap();
            let prefix = key.rsplit('.').nth(1).unwrap_or(key);
            if prefix != last_prefix && !current.is_empty()
            {
                layers.push(std::mem::take(&mut current));
            }
            current.extend_from_slice(&tensor.data);
            last_prefix = prefix.to_string();
        }
        if !current.is_empty()
        {
            layers.push(current);
        }
        let fingerprint = weights_fingerprint(&layers);
        ModelWeights {
            layers,
            input_dim: self.input_dim,
            fingerprint,
        }
    }
}

/// Compute a SHA-256 fingerprint of the weights (for the proof file).
fn weights_fingerprint(layers: &[Vec<f32>]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for layer in layers
    {
        let bytes: Vec<u8> = layer.iter().flat_map(|f| f.to_le_bytes()).collect();
        hasher.update(&bytes);
    }
    format!("{:x}", hasher.finalize())
}

/// Build the feature vector from a snapshot and indicator set.
///
/// The feature vector is a fixed-size slice that combines:
/// - last N normalized closes
/// - last RSI value
/// - last MACD histogram
/// - last ATR (normalized)
///
/// Normalization is done by dividing by the last close, so the features are
/// scale-invariant — the model learns *returns*, not absolute prices.
pub fn build_features(
    closes: &[f32],
    rsi_series: &[f32],
    macd_hist: &[f32],
    atr_series: &[f32],
    lookback: usize,
) -> Vec<f32> {
    let n = closes.len();
    if n == 0
    {
        return Vec::new();
    }
    let last_close = closes[n - 1].max(1e-6);
    let mut features = Vec::with_capacity(lookback + 3);
    let start = n.saturating_sub(lookback);
    for &c in &closes[start..n]
    {
        features.push(c / last_close - 1.0);
    }
    while features.len() < lookback
    {
        features.insert(0, 0.0);
    }
    features.push(rsi_series.last().copied().unwrap_or(50.0) / 100.0);
    features.push(macd_hist.last().copied().unwrap_or(0.0) / last_close);
    features.push(atr_series.last().copied().unwrap_or(0.0) / last_close);
    features
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_initializes_and_predicts() {
        let mut model = PricePredictor::new(10, &[16, 8], 42);
        let features = vec![0.0; 10];
        let pred = model.predict(&features);
        assert!(pred.is_finite());
    }

    #[test]
    fn training_reduces_loss() {
        let mut model = PricePredictor::new(5, &[16, 8], 42);
        let features = vec![0.1, -0.05, 0.2, 0.0, 0.15];
        let target = 1.0f32;
        let initial_loss = model.train_step(&features, target, 0.01);
        let mut last_loss = initial_loss;
        for _ in 0..20
        {
            last_loss = model.train_step(&features, target, 0.01);
        }
        assert!(
            last_loss < initial_loss,
            "loss should decrease: {} < {}",
            last_loss,
            initial_loss
        );
    }

    #[test]
    fn weights_export_has_fingerprint() {
        let model = PricePredictor::new(5, &[8], 42);
        let w = model.export_weights();
        assert!(!w.layers.is_empty());
        assert_eq!(w.fingerprint.len(), 64);
    }

    #[test]
    fn feature_vector_has_fixed_size() {
        let closes = vec![100.0; 20];
        let rsi = vec![50.0; 20];
        let macd = vec![0.0; 20];
        let atr = vec![1.0; 20];
        let features = build_features(&closes, &rsi, &macd, &atr, 10);
        assert_eq!(features.len(), 13);
    }

    #[test]
    fn same_seed_same_weights() {
        let m1 = PricePredictor::new(5, &[8], 42);
        let m2 = PricePredictor::new(5, &[8], 42);
        let w1 = m1.export_weights();
        let w2 = m2.export_weights();
        assert_eq!(w1.fingerprint, w2.fingerprint);
    }
}
