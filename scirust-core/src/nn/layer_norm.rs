// scirust-core/src/nn/layer_norm.rs
//
// LayerNorm — normalisation par ligne/token (Pre-LN convention).
//
// Calcule mean et variance par ligne, puis normalise : (x - mean) / sqrt(var + eps)
// puis scale par gamma et shift par beta.

use std::collections::HashMap;
use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::module::Module;
use crate::nn::init::Initializer;
use crate::nn::rng::PcgEngine;

pub struct LayerNorm {
    pub name:      String,
    pub gamma:     Tensor,
    pub beta:      Tensor,
    pub eps:       f32,
    last_g_idx:    Option<usize>,
    last_b_idx:    Option<usize>,
}

impl LayerNorm {
    pub fn new(d_model: usize, eps: f32, init: &dyn Initializer, rng: &mut PcgEngine) -> Self {
        let mut gamma = Tensor::zeros(1, d_model);
        let mut beta  = Tensor::zeros(1, d_model);
        init.fill(&mut gamma, 1, d_model, rng);
        init.fill(&mut beta, 1, d_model, rng);
        Self {
            name: "layer_norm".into(),
            gamma, beta, eps,
            last_g_idx: None, last_b_idx: None,
        }
    }

    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.into();
        self
    }
}

impl Clone for LayerNorm {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            gamma: self.gamma.clone(),
            beta:  self.beta.clone(),
            eps:   self.eps,
            last_g_idx: None, last_b_idx: None,
        }
    }
}

impl Module for LayerNorm {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        let g = tape.input(self.gamma.clone());
        let b = tape.input(self.beta.clone());
        self.last_g_idx = Some(g.idx());
        self.last_b_idx = Some(b.idx());
        input.layer_norm(g, b, self.eps)
    }

    fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::new();
        if let Some(i) = self.last_g_idx { v.push(i); }
        if let Some(i) = self.last_b_idx { v.push(i); }
        v
    }

    fn sync(&mut self, tape: &Tape) {
        if let Some(i) = self.last_g_idx { self.gamma = tape.value(i); }
        if let Some(i) = self.last_b_idx { self.beta  = tape.value(i); }
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.insert(format!("{}/gamma", self.name), self.gamma.clone());
        map.insert(format!("{}/beta", self.name), self.beta.clone());
        map
    }

    fn load_state_dict(&mut self, sd: &HashMap<String, Tensor>) -> std::result::Result<(), String> {
        let g = sd.get(&format!("{}/gamma", self.name))
            .ok_or_else(|| format!("missing key: {}/gamma", self.name))?;
        let b = sd.get(&format!("{}/beta", self.name))
            .ok_or_else(|| format!("missing key: {}/beta", self.name))?;
        if g.shape() != (1, self.gamma.cols) {
            return Err(format!("gamma shape mismatch"));
        }
        self.gamma = g.clone();
        self.beta = b.clone();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::init::Zeros;

    #[test]
    fn layer_norm_forward_shape_preserved() {
        let mut rng = PcgEngine::new(1);
        let mut ln = LayerNorm::new(4, 1e-5, &Zeros, &mut rng);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 1, 4));
        let y = ln.forward(&tape, x);
        assert_eq!(y.shape(), (1, 4));
    }

    #[test]
    fn layer_norm_zero_init_identity_without_gamma() {
        let mut rng = PcgEngine::new(1);
        let mut ln = LayerNorm::new(2, 1e-5, &Zeros, &mut rng);
        ln.gamma = Tensor::from_vec(vec![1.0, 1.0], 1, 2);
        ln.beta = Tensor::zeros(1, 2);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0, 3.0], 1, 2));
        let y = ln.forward(&tape, x);
        let v = tape.value(y.idx());
        // mean = 2, var = 1, std = 1
        // (1-2)/1 = -1, (3-2)/1 = 1
        assert!((v.data[0] - (-1.0)).abs() < 1e-4);
        assert!((v.data[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn layer_norm_gradient_flows() {
        let mut rng = PcgEngine::new(1);
        let mut ln = LayerNorm::new(3, 1e-5, &Zeros, &mut rng);
        ln.gamma = Tensor::from_vec(vec![1.0; 3], 1, 3);
        ln.beta = Tensor::zeros(1, 3);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3));
        let x_idx = x.idx();
        let y = ln.forward(&tape, x).sum();
        y.backward();
        let g = tape.grad(x_idx);
        // Gradient existe et n'est pas nul partout
        assert!(g.data.iter().any(|&v| v.abs() > 1e-6));
    }
}
