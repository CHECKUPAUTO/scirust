//! Certified bounds — lightweight Interval Bound Propagation (IBP) for the
//! trading model.
//!
//! Given a perturbation radius `eps` around the input feature vector, we
//! propagate interval bounds through the network and obtain a provable
//! output interval `[lo, hi]`. The LLM is then **hard-constrained** to
//! announce a prediction within this interval.
//!
//! This is a simplified IBP: for a linear layer `y = Wx + b`, if `x ∈ [x_lo, x_hi]`
//! then `y_j ∈ [sum(W_ji * x_lo_i if W_ji>0 else W_ji * x_hi_i) + b_j,
//!              sum(W_ji * x_hi_i if W_ji>0 else W_ji * x_lo_i) + b_j]`.
//! For ReLU, `[max(0, lo), max(0, hi)]`.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::model::ModelWeights;

/// An interval `[lo, hi]`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Interval {
    pub lo: f32,
    pub hi: f32,
}

impl Interval {
    pub fn new(lo: f32, hi: f32) -> Self {
        Self { lo, hi }
    }

    pub fn point(x: f32) -> Self {
        Self { lo: x, hi: x }
    }

    pub fn width(&self) -> f32 {
        self.hi - self.lo
    }

    pub fn midpoint(&self) -> f32 {
        (self.lo + self.hi) / 2.0
    }
}

/// Certified output of the model on an input box of radius `eps`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertifiedBounds {
    /// Input perturbation radius (L∞ ball).
    pub eps: f32,
    /// Output interval for the predicted return.
    pub output: Interval,
    /// Midpoint prediction.
    pub midpoint: f32,
    /// Half-width of the output interval (uncertainty).
    pub uncertainty: f32,
    /// Hash of the weights used (for the proof file).
    pub weights_fingerprint: String,
}

impl CertifiedBounds {
    /// Serialize to JSON for the proof file.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

/// Propagate interval bounds through the network's weights.
///
/// The weights are expected as a list of flat vectors, alternating:
/// `weights[0]` = layer-0 weight matrix (row-major, `in * out`),
/// `weights[1]` = layer-0 bias (`out`),
/// `weights[2]` = layer-1 weight matrix, etc.
///
/// `input` is the feature vector; `eps` is the perturbation radius.
pub fn certify(weights: &ModelWeights, input: &[f32], eps: f32) -> CertifiedBounds {
    let mut bounds: Vec<Interval> = input
        .iter()
        .map(|&x| Interval::new(x - eps, x + eps))
        .collect();

    let mut i = 0;
    while i < weights.layers.len()
    {
        let w_flat = &weights.layers[i];
        if i + 1 >= weights.layers.len()
        {
            break;
        }
        let b_flat = &weights.layers[i + 1];
        let in_dim = bounds.len();
        let out_dim = b_flat.len();
        if w_flat.len() < in_dim * out_dim
        {
            break;
        }
        let mut out_bounds = Vec::with_capacity(out_dim);
        for j in 0..out_dim
        {
            let mut lo = b_flat[j];
            let mut hi = b_flat[j];
            for k in 0..in_dim
            {
                let w = w_flat[k * out_dim + j];
                let (x_lo, x_hi) = (bounds[k].lo, bounds[k].hi);
                if w >= 0.0
                {
                    lo += w * x_lo;
                    hi += w * x_hi;
                }
                else
                {
                    lo += w * x_hi;
                    hi += w * x_lo;
                }
            }
            out_bounds.push(Interval::new(lo, hi));
        }
        bounds = out_bounds;
        i += 2;
        if i < weights.layers.len()
        {
            // ReLU activation: [max(0, lo), max(0, hi)]
            for b in &mut bounds
            {
                b.lo = b.lo.max(0.0);
                b.hi = b.hi.max(0.0);
            }
        }
    }
    let out = if bounds.is_empty()
    {
        Interval::new(0.0, 0.0)
    }
    else
    {
        bounds[0]
    };
    let midpoint = out.midpoint();
    let uncertainty = out.width() / 2.0;
    CertifiedBounds {
        eps,
        output: out,
        midpoint,
        uncertainty,
        weights_fingerprint: weights.fingerprint.clone(),
    }
}

/// Feature attribution placeholder — uses the indicator values themselves
/// as a simple proxy. In a full implementation, this would call
/// `scirust_core::xai::integrated_gradients`.
pub fn feature_attribution(features: &[f32], names: &[String]) -> BTreeMap<String, f32> {
    let total: f32 = features.iter().map(|f| f.abs()).sum::<f32>().max(1e-6);
    features
        .iter()
        .enumerate()
        .map(|(i, &f)| {
            let name = names.get(i).cloned().unwrap_or_else(|| format!("f{}", i));
            (name, f.abs() / total)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::PricePredictor;

    #[test]
    fn interval_basic() {
        let i = Interval::new(1.0, 3.0);
        assert_eq!(i.width(), 2.0);
        assert_eq!(i.midpoint(), 2.0);
    }

    #[test]
    fn certify_returns_finite_bounds() {
        let mut model = PricePredictor::new(5, &[8], 42);
        let features = vec![0.1, -0.05, 0.2, 0.0, 0.15];
        let _ = model.predict(&features);
        let weights = model.export_weights();
        let bounds = certify(&weights, &features, 0.01);
        assert!(bounds.output.lo.is_finite());
        assert!(bounds.output.hi.is_finite());
        assert!(bounds.output.lo <= bounds.output.hi);
        assert_eq!(bounds.eps, 0.01);
        assert_eq!(bounds.weights_fingerprint, weights.fingerprint);
    }

    #[test]
    fn larger_eps_widens_output() {
        let mut model = PricePredictor::new(5, &[8], 42);
        let features = vec![0.1, -0.05, 0.2, 0.0, 0.15];
        let _ = model.predict(&features);
        let weights = model.export_weights();
        let small = certify(&weights, &features, 0.001);
        let large = certify(&weights, &features, 0.1);
        assert!(large.uncertainty >= small.uncertainty);
    }

    #[test]
    fn attribution_sums_to_one() {
        let features = vec![0.5, -0.3, 0.2];
        let names = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let attr = feature_attribution(&features, &names);
        let total: f32 = attr.values().sum();
        assert!((total - 1.0).abs() < 1e-5);
    }
}
