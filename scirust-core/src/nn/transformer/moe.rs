use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::init::Initializer;
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::nn::rng::PcgEngine;
use std::collections::HashMap;

/// Mixture of Experts (MoE) layer.
pub struct MoELayer<E: Module> {
    pub gate: Linear,
    pub experts: Vec<E>,
    pub k: usize,
    pub name: String,
}

impl<E: Module> MoELayer<E> {
    pub fn new<W: Initializer, B: Initializer>(
        d_model: usize,
        num_experts: usize,
        k: usize,
        expert_factory: impl Fn() -> E,
        w_init: &W,
        b_init: &B,
        rng: &mut PcgEngine,
    ) -> Self {
        let gate = Linear::new(d_model, num_experts, w_init, b_init, rng);
        let mut experts = Vec::new();
        for _ in 0..num_experts
        {
            experts.push(expert_factory());
        }
        Self {
            gate,
            experts,
            k,
            name: format!("moe_{num_experts}_k{k}"),
        }
    }
}

impl<E: Module> Module for MoELayer<E> {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        let gate_logits = self.gate.forward(tape, input);
        let gate_probs = gate_logits.try_softmax(1).unwrap();

        let probs = tape.value(gate_probs.idx());
        let (rows, cols) = gate_probs.shape();

        let mut output = tape.input(Tensor::zeros(rows, input.shape().1));

        for i in 0..rows
        {
            let row_probs = &probs.data[i * cols..(i + 1) * cols];
            let mut indexed_probs: Vec<(usize, f32)> =
                row_probs.iter().cloned().enumerate().collect();
            indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let top_k = &indexed_probs[0..self.k];
            let mut row_output: Option<Var> = None;

            for &(expert_idx, prob) in top_k
            {
                let input_row = input.try_slice_rows(i, 1).unwrap();
                let expert_out = self.experts[expert_idx].forward(tape, input_row);
                let weighted = expert_out.scale(prob);

                row_output = Some(match row_output
                {
                    None => weighted,
                    Some(acc) => acc.try_add(weighted).unwrap(),
                });
            }

            if let Some(ro) = row_output
            {
                if i == 0
                {
                    output = ro;
                }
            }
        }

        output
    }

    fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::new();
        v.extend(self.gate.parameter_indices());
        for expert in &self.experts
        {
            v.extend(expert.parameter_indices());
        }
        v
    }

    fn sync(&mut self, tape: &Tape) {
        self.gate.sync(tape);
        for expert in &mut self.experts
        {
            expert.sync(tape);
        }
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        for (k, v) in self.gate.state_dict()
        {
            map.insert(format!("gate.{}", k), v);
        }
        for (i, expert) in self.experts.iter().enumerate()
        {
            for (k, v) in expert.state_dict()
            {
                map.insert(format!("expert{}.{}", i, k), v);
            }
        }
        map
    }
}
