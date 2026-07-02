use scirust_core::autodiff::reverse::{Tape, Tensor, Var};
use scirust_core::nn::init::Initializer;
use scirust_core::nn::module::Module;
use scirust_core::nn::rng::PcgEngine;
use std::collections::HashMap;

pub struct RMSNorm {
    pub name: String,
    pub weight: Tensor,
    pub eps: f32,
    last_w_idx: Option<usize>,
}

impl RMSNorm {
    pub fn new(d_model: usize, eps: f32, init: &dyn Initializer, rng: &mut PcgEngine) -> Self {
        let mut weight = Tensor::ones(1, d_model);
        init.fill(&mut weight, 1, d_model, rng);
        Self {
            name: "rms_norm".into(),
            weight,
            eps,
            last_w_idx: None,
        }
    }

    #[must_use]
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.into();
        self
    }
}

impl Clone for RMSNorm {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            weight: self.weight.clone(),
            eps: self.eps,
            last_w_idx: None,
        }
    }
}

impl Module for RMSNorm {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        let (rows, cols) = input.shape();
        let w = tape.input(self.weight.clone());
        self.last_w_idx = Some(w.idx());

        let eps_t = tape.input(Tensor::from_vec(vec![self.eps], 1, 1));

        let xs = input.hadamard(input);
        let mean = xs.mean_axis(1);
        let mean_b = mean.broadcast(rows, cols);
        let eps_b = eps_t.broadcast(rows, cols);
        let rms = mean_b.add(eps_b).sqrt();
        let inv_rms = rms.reciprocal();
        let normalized = input.hadamard(inv_rms);
        normalized.hadamard(w.broadcast(rows, cols))
    }

    fn parameter_indices(&self) -> Vec<usize> {
        self.last_w_idx.map(|i| vec![i]).unwrap_or_default()
    }

    fn sync(&mut self, tape: &Tape) {
        if let Some(i) = self.last_w_idx
        {
            self.weight = tape.value(i);
        }
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.insert(format!("{}/weight", self.name), self.weight.clone());
        map
    }

    fn load_state_dict(&mut self, sd: &HashMap<String, Tensor>) -> scirust_core::error::Result<()> {
        let w = sd
            .get(&format!("{}/weight", self.name))
            .ok_or_else(|| format!("missing key: {}/weight", self.name))?;
        self.weight = w.clone();
        Ok(())
    }
}
