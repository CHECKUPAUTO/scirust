// scirust-core/src/nn/layer_norm.rs
// LayerNorm — normalisation par token (Pre-LN convention)

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
    fn box_clone(&self) -> Box<dyn Module> { Box::new(self.clone()) }

    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        let g = tape.input(self.gamma.clone());
        let b = tape.input(self.beta.clone());
        self.last_g_idx = Some(g.idx());
        self.last_b_idx = Some(b.idx());
        // Stub: retourne input (vrai LayerNorm nécessite mean/var par ligne)
        let _ = (g, b);
        input
    }

    fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::new();
        if let Some(i) = self.last_g_idx { v.push(i); }
        if let Some(i) = self.last_b_idx { v.push(i); }
        v
    }

    fn sync(&mut self, tape: &Tape) {
        if let Some(i) = self.last_g_idx { self.gamma = tape.value(i); }
        if let Some(i) = self.last_b_idx  { self.beta  = tape.value(i); }
    }

    fn state_dict(&self) -> Vec<(String, Tensor)> {
        vec![
            (format!("{}/gamma", self.name), self.gamma.clone()),
            (format!("{}/beta", self.name),  self.beta.clone()),
        ]
    }

    fn load_state_dict(&mut self, dict: &std::collections::HashMap<String, Tensor>) -> usize {
        let mut n = 0;
        if let Some(t) = dict.get(&format!("{}/gamma", self.name)) { self.gamma = t.clone(); n += 1; }
        if let Some(t) = dict.get(&format!("{}/beta", self.name))   { self.beta  = t.clone(); n += 1; }
        n
    }

    fn train(&mut self, _mode: bool) {}
}
