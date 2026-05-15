// scirust-core/src/nn/lstm.rs
//
// Couche LSTM standard (Long Short-Term Memory).
//
// Implémente une LSTM vanilla avec 4 portes (input, forget, cell, output)
// et mémoire cellulaire. Supporte séquences arbitraires.
//
// Shapes :
//   - input  : (seq_len * batch, input_size)
//   - w_ih   : (4 * hidden_size, input_size)
//   - w_hh   : (4 * hidden_size, hidden_size)
//   - b_ih   : (1, 4 * hidden_size) — broadcast row-wise
//   - b_hh   : (1, 4 * hidden_size) — broadcast row-wise
//   - output : (seq_len * batch, hidden_size)

use crate::autodiff::reverse::{concat_rows, Tape, Tensor, Var};
use crate::nn::rng::PcgEngine;

pub struct LSTM {
    pub input_size: usize,
    pub hidden_size: usize,
    pub w_ih: Tensor,
    pub w_hh: Tensor,
    pub b_ih: Option<Tensor>,
    pub b_hh: Option<Tensor>,
    pub has_bias: bool,
    last_w_ih: Option<usize>,
    last_w_hh: Option<usize>,
    last_b_ih: Option<usize>,
    last_b_hh: Option<usize>,
}

impl LSTM {
    /// Crée une nouvelle couche LSTM.
    ///
    /// Les poids sont initialisés avec une distribution uniforme sur
    /// [-scale, scale] où scale = sqrt(2 / (4 * hidden_size)).
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        bias: bool,
        rng: &mut PcgEngine,
    ) -> Self {
        let scale = (2.0 / (4.0 * hidden_size as f32)).sqrt();
        let mut w_ih = Tensor::zeros(4 * hidden_size, input_size);
        let mut w_hh = Tensor::zeros(4 * hidden_size, hidden_size);
        for x in w_ih.data.iter_mut() {
            *x = rng.float_signed() * scale;
        }
        for x in w_hh.data.iter_mut() {
            *x = rng.float_signed() * scale;
        }
        let (b_ih, b_hh) = if bias {
            (
                Some(Tensor::zeros(1, 4 * hidden_size)),
                Some(Tensor::zeros(1, 4 * hidden_size)),
            )
        } else {
            (None, None)
        };
        Self {
            input_size,
            hidden_size,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            has_bias: bias,
            last_w_ih: None,
            last_w_hh: None,
            last_b_ih: None,
            last_b_hh: None,
        }
    }

    /// Forward pass séquentiel à travers `seq_len` pas de temps.
    ///
    /// input shape : (seq_len * batch, input_size) — tous les pas de
    /// temps concaténés verticalement.
    ///
    /// Retourne un tenseur (seq_len * batch, hidden_size) où les
    /// sorties de chaque pas sont concaténées dans l'ordre temporel.
    pub fn forward_sequence<'t>(
        &mut self,
        tape: &'t Tape,
        input: Var<'t>,
        seq_len: usize,
        batch_size: usize,
    ) -> Var<'t> {
        let w_ih = tape.input(self.w_ih.clone());
        let w_hh = tape.input(self.w_hh.clone());
        self.last_w_ih = Some(w_ih.idx());
        self.last_w_hh = Some(w_hh.idx());

        let b_ih = self.b_ih.as_ref().map(|b| {
            let v = tape.input(b.clone());
            self.last_b_ih = Some(v.idx());
            v
        });
        let b_hh = self.b_hh.as_ref().map(|b| {
            let v = tape.input(b.clone());
            self.last_b_hh = Some(v.idx());
            v
        });

        let mut h = tape.input(Tensor::zeros(batch_size, self.hidden_size));
        let mut c = tape.input(Tensor::zeros(batch_size, self.hidden_size));

        let w_ih_t = w_ih.transpose();
        let w_hh_t = w_hh.transpose();

        let mut outputs: Vec<Var<'t>> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let x_t = input
                .clone()
                .slice_rows(t * batch_size, (t + 1) * batch_size);

            // gates = x_t @ W_ih^T + h @ W_hh^T + b_ih + b_hh
            let mut gates = x_t.matmul(w_ih_t.clone()).add(h.matmul(w_hh_t.clone()));
            if let Some(ref bi) = b_ih {
                gates = gates.add_broadcast(bi.clone());
            }
            if let Some(ref bh) = b_hh {
                gates = gates.add_broadcast(bh.clone());
            }

            // Split en 4 portes (input, forget, cell, output)
            let d = self.hidden_size;
            let i_gate = gates.clone().slice_cols(0, d).sigmoid();
            let f_gate = gates.clone().slice_cols(d, d).sigmoid();
            let g_gate = gates.clone().slice_cols(2 * d, d).tanh();
            let o_gate = gates.slice_cols(3 * d, d).sigmoid();

            // c = f ⊙ c + i ⊙ g
            c = f_gate.hadamard(c).add(i_gate.hadamard(g_gate));
            h = o_gate.hadamard(c.clone().tanh());

            outputs.push(h.clone());
        }

        concat_rows(tape, &outputs)
    }

    pub fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::new();
        if let Some(i) = self.last_w_ih {
            v.push(i);
        }
        if let Some(i) = self.last_w_hh {
            v.push(i);
        }
        if let Some(i) = self.last_b_ih {
            v.push(i);
        }
        if let Some(i) = self.last_b_hh {
            v.push(i);
        }
        v
    }

    pub fn sync(&mut self, tape: &Tape) {
        if let Some(i) = self.last_w_ih {
            self.w_ih = tape.value(i);
        }
        if let Some(i) = self.last_w_hh {
            self.w_hh = tape.value(i);
        }
        if let Some(i) = self.last_b_ih {
            self.b_ih = Some(tape.value(i));
        }
        if let Some(i) = self.last_b_hh {
            self.b_hh = Some(tape.value(i));
        }
    }
}

impl Clone for LSTM {
    fn clone(&self) -> Self {
        Self {
            input_size: self.input_size,
            hidden_size: self.hidden_size,
            w_ih: self.w_ih.clone(),
            w_hh: self.w_hh.clone(),
            b_ih: self.b_ih.clone(),
            b_hh: self.b_hh.clone(),
            has_bias: self.has_bias,
            last_w_ih: None,
            last_w_hh: None,
            last_b_ih: None,
            last_b_hh: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::rng::PcgEngine;

    #[test]
    fn lstm_forward_shape() {
        let mut rng = PcgEngine::new(42);
        let input_size = 10;
        let hidden_size = 16;
        let seq_len = 5;
        let batch = 4;

        let mut lstm = LSTM::new(input_size, hidden_size, true, &mut rng);
        let tape = Tape::new();
        let input = Tensor::zeros(seq_len * batch, input_size);
        let x = tape.input(input);

        let out = lstm.forward_sequence(&tape, x, seq_len, batch);
        assert_eq!(out.shape(), (seq_len * batch, hidden_size));
    }

    #[test]
    fn lstm_no_bias_forward() {
        let mut rng = PcgEngine::new(42);
        let mut lstm = LSTM::new(8, 12, false, &mut rng);
        assert!(!lstm.has_bias);
        assert!(lstm.b_ih.is_none());
        assert!(lstm.b_hh.is_none());

        let tape = Tape::new();
        let input = Tensor::zeros(6, 8); // seq_len=3, batch=2
        let x = tape.input(input);
        let out = lstm.forward_sequence(&tape, x, 3, 2);
        assert_eq!(out.shape(), (6, 12));
    }

    #[test]
    fn lstm_parameter_indices_after_forward() {
        let mut rng = PcgEngine::new(42);
        let mut lstm = LSTM::new(10, 16, true, &mut rng);
        let tape = Tape::new();
        let x = tape.input(Tensor::zeros(8, 10));
        let _ = lstm.forward_sequence(&tape, x, 4, 2);

        let idxs = lstm.parameter_indices();
        assert_eq!(idxs.len(), 4, "bias LSTM should have 4 params");
    }

    #[test]
    fn lstm_sync_restores_weights() {
        let mut rng = PcgEngine::new(42);
        let mut lstm = LSTM::new(10, 16, true, &mut rng);
        let tape = Tape::new();
        let x = tape.input(Tensor::zeros(8, 10));
        let _ = lstm.forward_sequence(&tape, x, 4, 2);

        let w_ih_before = lstm.w_ih.clone();
        // Simulate an update: modify w_ih on tape
        let new_w = Tensor::zeros(64, 10);
        let w_tape = &tape;
        // Overwrite w_ih's tape value
        if let Some(idx) = lstm.last_w_ih {
            w_tape.values.borrow_mut()[idx] =
                crate::autodiff::reverse::DeviceTensor::cpu(new_w.clone());
        }
        lstm.sync(&tape);
        // After sync, w_ih should have been updated
        assert_eq!(lstm.w_ih.data, new_w.data);
        assert_ne!(lstm.w_ih.data, w_ih_before.data);
    }

    #[test]
    fn lstm_forward_deterministic() {
        let mut rng_a = PcgEngine::new(42);
        let mut rng_b = PcgEngine::new(42);
        let mut lstm_a = LSTM::new(10, 16, true, &mut rng_a);
        let mut lstm_b = LSTM::new(10, 16, true, &mut rng_b);

        let tape_a = Tape::new();
        let tape_b = Tape::new();
        let x_a = tape_a.input(Tensor::zeros(6, 10));
        let x_b = tape_b.input(Tensor::zeros(6, 10));

        let out_a = lstm_a.forward_sequence(&tape_a, x_a, 3, 2);
        let out_b = lstm_b.forward_sequence(&tape_b, x_b, 3, 2);

        let val_a = tape_a.value(out_a.idx());
        let val_b = tape_b.value(out_b.idx());
        assert_eq!(val_a.data, val_b.data, "deterministic output mismatch");
    }

    #[test]
    fn lstm_zeros_in_zeros_out() {
        let mut rng = PcgEngine::new(1);
        let mut lstm = LSTM::new(5, 8, true, &mut rng);
        let tape = Tape::new();
        let x = tape.input(Tensor::zeros(4, 5));
        let out = lstm.forward_sequence(&tape, x, 2, 2);
        let val = tape.value(out.idx());
        // All-zero input with freshly initialized weights should produce non-zero output
        let max_abs: f32 = val.data.iter().map(|x| x.abs()).fold(0.0, f32::max);
        assert!(max_abs > 0.0, "expected non-zero output from non-zero init");
    }
}
