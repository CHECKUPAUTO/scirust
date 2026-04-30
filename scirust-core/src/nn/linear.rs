// scirust-core/src/nn/linear.rs
// Linear layer — y = x @ W + b

use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::module::Module;
use crate::nn::init::Initializer;
use crate::nn::rng::PcgEngine;

#[derive(Clone)]
pub struct Linear {
    pub weight: Tensor,
    pub bias:   Tensor,
    pub name:   String,
    last_w_idx: Option<usize>,
    last_b_idx: Option<usize>,
}

impl Linear {
    pub fn new<W: Initializer, B: Initializer>(
        in_f: usize, out_f: usize,
        w_init: &W, b_init: &B,
        rng: &mut PcgEngine,
    ) -> Self {
        let mut weight = Tensor::zeros(in_f, out_f);
        let mut bias   = Tensor::zeros(1, out_f);
        w_init.fill(&mut weight, rng);
        b_init.fill(&mut bias, rng);
        Self {
            weight, bias,
            name: format!("linear_{}_{}", in_f, out_f),
            last_w_idx: None, last_b_idx: None,
        }
    }

    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.into();
        self
    }
}

impl Module for Linear {
    fn box_clone(&self) -> Box<dyn Module> { Box::new(self.clone()) }

    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        let w = tape.input(self.weight.clone());
        let b = tape.input(self.bias.clone());
        self.last_w_idx = Some(w.idx());
        self.last_b_idx = Some(b.idx());
        input.matmul(w).add_broadcast(b)
    }

    fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::new();
        if let Some(i) = self.last_w_idx { v.push(i); }
        if let Some(i) = self.last_b_idx { v.push(i); }
        v
    }

    fn sync(&mut self, tape: &Tape) {
        if let Some(i) = self.last_w_idx { self.weight = tape.value(i); }
        if let Some(i) = self.last_b_idx  { self.bias   = tape.value(i); }
    }

    fn state_dict(&self) -> Vec<(String, Tensor)> {
        vec![
            (format!("{}/weight", self.name), self.weight.clone()),
            (format!("{}/bias", self.name),   self.bias.clone()),
        ]
    }

    fn load_state_dict(&mut self, dict: &std::collections::HashMap<String, Tensor>) -> usize {
        let mut n = 0;
        if let Some(t) = dict.get(&format!("{}/weight", self.name)) { self.weight = t.clone(); n += 1; }
        if let Some(t) = dict.get(&format!("{}/bias", self.name))   { self.bias   = t.clone(); n += 1; }
        n
    }

    fn train(&mut self, _mode: bool) {}
}
