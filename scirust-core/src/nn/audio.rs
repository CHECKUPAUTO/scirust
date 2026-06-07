use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::init::Initializer;
use crate::nn::conv2d::Conv2d;
use crate::nn::conv_utils::Padding;
use crate::nn::module::Module;
use crate::nn::rng::PcgEngine;

/// Connectionist Temporal Classification (CTC) Loss.
pub struct CTCLoss;

impl CTCLoss {
    /// Representative implementation of CTC loss.
    /// Returns the negative log-likelihood of the target sequence.
    pub fn forward<'t>(&self, tape: &'t Tape, logits: Var<'t>, targets: Var<'t>) -> Var<'t> {
        // logits: (T, vocab_size), targets: (S)
        // Simplified: log-sum-exp across time to simulate alignment cost
        let log_probs = logits.log_softmax(1);
        let (t_steps, vocab_size) = log_probs.shape();

        let mut total_log_p = tape.input(Tensor::zeros(1, 1));
        let target_vals = tape.value(targets.idx());

        for t in 0..t_steps {
            // For each time step, pick the target token's log probability if valid
            let target_idx = if t < target_vals.data.len() {
                target_vals.data[t] as usize
            } else {
                vocab_size - 1 // blank/padding
            };

            let lp = log_probs.slice_rows(t, 1).slice_cols(target_idx % vocab_size, 1);
            total_log_p = total_log_p.add(lp);
        }

        total_log_p.scale(-1.0)
    }
}

/// Basic Audio Encoder (CNN-based).
pub struct AudioEncoder {
    pub conv1: Conv2d,
    pub conv2: Conv2d,
}

impl AudioEncoder {
    pub fn new<W: Initializer, B: Initializer>(
        in_channels: usize,
        hidden_channels: usize,
        out_channels: usize,
        w_init: &W,
        b_init: &B,
        rng: &mut PcgEngine,
    ) -> Self {
        let conv1 = Conv2d::new(in_channels, hidden_channels, 3, 2, Padding::Same, w_init, Some(b_init), rng);
        let conv2 = Conv2d::new(hidden_channels, out_channels, 3, 2, Padding::Same, w_init, Some(b_init), rng);
        Self { conv1, conv2 }
    }
}

impl Module for AudioEncoder {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        let x = self.conv1.forward(tape, input);
        let x = x.relu();
        self.conv2.forward(tape, x).relu()
    }

    fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::new();
        v.extend(self.conv1.parameter_indices());
        v.extend(self.conv2.parameter_indices());
        v
    }

    fn sync(&mut self, tape: &Tape) {
        self.conv1.sync(tape);
        self.conv2.sync(tape);
    }
}
