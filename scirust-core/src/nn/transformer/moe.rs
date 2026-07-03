use crate::autodiff::reverse::{Tape, Tensor, Var, concat_rows};
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

        let out_cols = input.shape().1;
        let mut row_outputs: Vec<Var<'t>> = Vec::with_capacity(rows);

        for i in 0..rows
        {
            let row_probs = &probs.data[i * cols..(i + 1) * cols];
            let mut indexed_probs: Vec<(usize, f32)> =
                row_probs.iter().cloned().enumerate().collect();
            // total_cmp is a total order over f32, so a NaN gate probability
            // (e.g. from a NaN activation upstream) sorts deterministically
            // instead of panicking the way partial_cmp().unwrap() did.
            indexed_probs.sort_by(|a, b| b.1.total_cmp(&a.1));

            let top_k = &indexed_probs[0..self.k.min(indexed_probs.len())];
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

            // Every row contributes its own mixed-expert output. (Previously only
            // row 0 was kept, silently dropping all other batch rows and returning
            // a (1, out) tensor instead of (rows, out).)
            row_outputs.push(row_output.unwrap_or_else(|| tape.input(Tensor::zeros(1, out_cols))));
        }

        if row_outputs.is_empty()
        {
            return tape.input(Tensor::zeros(rows, out_cols));
        }
        concat_rows(tape, &row_outputs)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::init::{KaimingNormal, Zeros};

    #[test]
    fn moe_forward_processes_every_row() {
        let mut rng = PcgEngine::new(7);
        let mut moe = MoELayer::new(
            4,
            2,
            1,
            || Linear::new(4, 4, &KaimingNormal, &Zeros, &mut PcgEngine::new(99)),
            &KaimingNormal,
            &Zeros,
            &mut rng,
        );
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(
            vec![
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0,
            ],
            3,
            4,
        ));
        let out = moe.forward(&tape, x);
        // Regression: the full (rows, out) tensor — pre-fix this collapsed to (1, 4).
        assert_eq!(tape.value(out.idx()).shape(), (3, 4));
        // Gradients must flow through every row without panicking.
        let loss = out.sum();
        tape.backward(loss.idx());
        assert!(tape.value(out.idx()).data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn moe_forward_does_not_panic_on_nan_gate() {
        // A NaN activation propagates to the gate probabilities. The top-k sort
        // must not panic — total_cmp handles NaN, whereas the old
        // partial_cmp().unwrap() panicked on it.
        let mut rng = PcgEngine::new(7);
        let mut moe = MoELayer::new(
            4,
            2,
            1,
            || Linear::new(4, 4, &KaimingNormal, &Zeros, &mut PcgEngine::new(99)),
            &KaimingNormal,
            &Zeros,
            &mut rng,
        );
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![f32::NAN; 8], 2, 4));
        let out = moe.forward(&tape, x);
        assert_eq!(tape.value(out.idx()).shape(), (2, 4));
    }
}
