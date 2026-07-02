use scirust_core::autodiff::reverse::Tensor;
use scirust_core::autodiff::reverse::{Tape, Var};
use scirust_core::nn::init::Initializer;
use scirust_core::nn::linear::Linear;
use scirust_core::nn::module::Module;
use scirust_core::nn::rng::PcgEngine;
use std::collections::HashMap;

pub struct SwiGLUFFN {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
    pub name: String,
}

impl SwiGLUFFN {
    pub fn new<I: Initializer>(d_model: usize, d_ff: usize, init: &I, rng: &mut PcgEngine) -> Self {
        let z = scirust_core::nn::init::Zeros;
        Self {
            gate: Linear::new(d_model, d_ff, init, &z, rng),
            up: Linear::new(d_model, d_ff, init, &z, rng),
            down: Linear::new(d_ff, d_model, init, &z, rng),
            name: format!("swiglu_d{d_model}_ff{d_ff}"),
        }
    }

    #[must_use]
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.into();
        self
    }

    pub fn forward<'t>(&mut self, tape: &'t Tape, x: Var<'t>) -> Var<'t> {
        let g = self.gate.forward(tape, x);
        let u = self.up.forward(tape, x);
        let s = silu(g);
        let h = s.hadamard(u);
        self.down.forward(tape, h)
    }

    pub fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::new();
        v.extend(self.gate.parameter_indices());
        v.extend(self.up.parameter_indices());
        v.extend(self.down.parameter_indices());
        v
    }

    pub fn sync(&mut self, tape: &Tape) {
        self.gate.sync(tape);
        self.up.sync(tape);
        self.down.sync(tape);
    }

    pub fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        let p = &self.name;
        map.insert(format!("{p}.gate.weight"), self.gate.weight.clone());
        map.insert(format!("{p}.up.weight"), self.up.weight.clone());
        map.insert(format!("{p}.down.weight"), self.down.weight.clone());
        map
    }

    pub fn load_state_dict(
        &mut self,
        sd: &HashMap<String, Tensor>,
    ) -> scirust_core::error::Result<()> {
        let p = &self.name;
        self.gate.weight = sd
            .get(&format!("{p}.gate.weight"))
            .ok_or_else(|| format!("missing {p}.gate.weight"))?
            .clone();
        self.up.weight = sd
            .get(&format!("{p}.up.weight"))
            .ok_or_else(|| format!("missing {p}.up.weight"))?
            .clone();
        self.down.weight = sd
            .get(&format!("{p}.down.weight"))
            .ok_or_else(|| format!("missing {p}.down.weight"))?
            .clone();
        Ok(())
    }
}

impl Clone for SwiGLUFFN {
    fn clone(&self) -> Self {
        Self {
            gate: self.gate.clone(),
            up: self.up.clone(),
            down: self.down.clone(),
            name: self.name.clone(),
        }
    }
}

fn silu<'t>(x: Var<'t>) -> Var<'t> {
    let s = x.sigmoid();
    x.hadamard(s)
}
