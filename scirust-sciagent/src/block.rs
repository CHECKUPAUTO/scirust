use scirust_core::autodiff::reverse::{Tape, Tensor, Var};
use scirust_core::nn::init::Initializer;
use scirust_core::nn::module::Module;
use scirust_core::nn::rng::PcgEngine;
use std::collections::HashMap;

use crate::attention::GQAAttention;
use crate::norm::RMSNorm;
use crate::swiglu::SwiGLUFFN;

pub struct SciAgentBlock {
    pub rms_attn: RMSNorm,
    pub attn: GQAAttention,
    pub rms_ffn: RMSNorm,
    pub ffn: SwiGLUFFN,
    pub name: String,
}

impl SciAgentBlock {
    pub fn new<I: Initializer>(
        d_model: usize,
        n_heads: usize,
        n_kv_heads: usize,
        d_ff: usize,
        rope_theta: f32,
        eps: f32,
        init: &I,
        rng: &mut PcgEngine,
        name: &str,
    ) -> Self {
        Self {
            rms_attn: RMSNorm::new(d_model, eps, init, rng).with_name(&format!("{name}.rms_attn")),
            attn: GQAAttention::new(d_model, n_heads, n_kv_heads, rope_theta, init, rng)
                .with_name(&format!("{name}.attn")),
            rms_ffn: RMSNorm::new(d_model, eps, init, rng).with_name(&format!("{name}.rms_ffn")),
            ffn: SwiGLUFFN::new(d_model, d_ff, init, rng).with_name(&format!("{name}.ffn")),
            name: name.to_string(),
        }
    }

    pub fn forward<'t>(&mut self, tape: &'t Tape, x: Var<'t>, seq_len: usize) -> Var<'t> {
        let a = self.rms_attn.forward(tape, x);
        let a = self.attn.forward(tape, a, seq_len);
        let x = x.add(a);

        let b = self.rms_ffn.forward(tape, x);
        let b = self.ffn.forward(tape, b);
        x.add(b)
    }

    pub fn infer_step<'t>(&mut self, tape: &'t Tape, x: Var<'t>, pos: usize) -> Var<'t> {
        let a = self.rms_attn.forward(tape, x);
        let a = self.attn.infer_step(tape, a, pos);
        let x = x.add(a);

        let b = self.rms_ffn.forward(tape, x);
        let b = self.ffn.forward(tape, b);
        x.add(b)
    }

    pub fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::new();
        v.extend(self.rms_attn.parameter_indices());
        v.extend(self.attn.parameter_indices());
        v.extend(self.rms_ffn.parameter_indices());
        v.extend(self.ffn.parameter_indices());
        v
    }

    pub fn sync(&mut self, tape: &Tape) {
        self.rms_attn.sync(tape);
        self.attn.sync(tape);
        self.rms_ffn.sync(tape);
        self.ffn.sync(tape);
    }

    pub fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.extend(self.rms_attn.state_dict());
        map.extend(self.attn.state_dict());
        map.extend(self.rms_ffn.state_dict());
        map.extend(self.ffn.state_dict());
        map
    }

    pub fn load_state_dict(&mut self, sd: &HashMap<String, Tensor>) -> scirust_core::error::Result<()> {
        self.rms_attn.load_state_dict(sd)?;
        self.attn.load_state_dict(sd)?;
        self.rms_ffn.load_state_dict(sd)?;
        self.ffn.load_state_dict(sd)?;
        Ok(())
    }
}

impl Clone for SciAgentBlock {
    fn clone(&self) -> Self {
        Self {
            rms_attn: self.rms_attn.clone(),
            attn: self.attn.clone(),
            rms_ffn: self.rms_ffn.clone(),
            ffn: self.ffn.clone(),
            name: self.name.clone(),
        }
    }
}
