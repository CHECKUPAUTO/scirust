// scirust-core/src/nn/positional_encoding.rs
//
// Positional Encoding sinusoidale — ajoute l'information de position
// aux embeddings de tokens. Pas de paramètres entraînables.
//
// Formule (Vaswani et al.) :
//   PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
//   PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))

use crate::autodiff::reverse::{Tape, Tensor, Var};
use std::collections::HashMap;

pub struct PositionalEncoding {
    pub d_model: usize,
    pub max_seq_len: usize,
    /// Table pré-calculée (max_seq_len, d_model)
    pub pe: Tensor,
}

impl PositionalEncoding {
    pub fn new(d_model: usize, max_seq_len: usize) -> Self {
        let mut pe = Tensor::zeros(max_seq_len, d_model);
        for pos in 0..max_seq_len
        {
            for i in 0..d_model
            {
                let div = 10000_f32.powf(2.0 * (i / 2) as f32 / d_model as f32);
                let val = if i % 2 == 0
                {
                    (pos as f32 / div).sin()
                }
                else
                {
                    (pos as f32 / div).cos()
                };
                pe.data[pos * d_model + i] = val;
            }
        }
        Self {
            d_model,
            max_seq_len,
            pe,
        }
    }

    /// Ajoute le PE à `input` de shape (batch * seq_len, d_model).
    /// `seq_len` doit être <= max_seq_len.
    pub fn forward<'t>(&self, tape: &'t Tape, input: Var<'t>, seq_len: usize) -> Var<'t> {
        assert!(
            seq_len <= self.max_seq_len,
            "PositionalEncoding: seq_len {seq_len} > max_seq_len {}",
            self.max_seq_len
        );
        let (total_rows, d) = input.shape();
        assert_eq!(d, self.d_model, "d_model mismatch");
        assert_eq!(
            total_rows % seq_len,
            0,
            "input rows {total_rows} not divisible by seq_len {seq_len}"
        );
        let batch = total_rows / seq_len;

        // Construire le PE broadcasté (batch*seq_len, d_model)
        let mut broadcasted = vec![0.0f32; total_rows * d];
        for b in 0..batch
        {
            for pos in 0..seq_len
            {
                for i in 0..d
                {
                    broadcasted[(b * seq_len + pos) * d + i] = self.pe.data[pos * d + i];
                }
            }
        }
        let pe_var = tape.input(Tensor::from_vec(broadcasted, total_rows, d));
        input.try_add(pe_var).unwrap()
    }

    /// Variante pour input 3D (B, T, D) via Var3D.
    pub fn forward_3d<'t>(
        &self,
        tape: &'t Tape,
        x_3d: crate::tensor::tensor3d::Var3D<'t>,
    ) -> crate::tensor::tensor3d::Var3D<'t> {
        let (batch, seq_len, d_model) = x_3d.shape();
        assert_eq!(d_model, self.d_model);
        let out = self.forward(tape, x_3d.as_var(), seq_len);
        crate::tensor::tensor3d::Var3D::from_var(out, batch, seq_len, d_model)
    }
}

impl Clone for PositionalEncoding {
    fn clone(&self) -> Self {
        Self {
            d_model: self.d_model,
            max_seq_len: self.max_seq_len,
            pe: self.pe.clone(),
        }
    }
}

// Pas de paramètres entraînables → pas de Module impl nécessaire,
// mais on fournit state_dict vide pour compatibilité.
impl crate::nn::module::Module for PositionalEncoding {
    fn forward<'t>(&mut self, _tape: &'t Tape, _input: Var<'t>) -> Var<'t> {
        // Module::forward ne passe pas seq_len ; utiliser forward() direct
        // avec seq_len explicite, ou forward_3d().
        panic!(
            "PositionalEncoding::forward() requires seq_len — use forward(tape, input, seq_len) or forward_3d()"
        )
    }
    fn parameter_indices(&self) -> Vec<usize> {
        Vec::new()
    }
    fn sync(&mut self, _tape: &Tape) {}
    fn state_dict(&self) -> HashMap<String, Tensor> {
        HashMap::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pe_has_correct_shape() {
        let pe = PositionalEncoding::new(8, 16);
        assert_eq!(pe.pe.shape(), (16, 8));
    }

    #[test]
    fn pe_values_in_range() {
        let pe = PositionalEncoding::new(16, 32);
        for &v in &pe.pe.data
        {
            assert!((-1.0..=1.0).contains(&v), "PE value out of [-1, 1]: {}", v);
        }
    }

    #[test]
    fn forward_adds_pe_correctly() {
        let pe = PositionalEncoding::new(4, 8);
        let tape = Tape::new();
        // batch=2, seq_len=3, d_model=4  →  flat (6, 4)
        let x = tape.input(Tensor::zeros(6, 4));
        let y = pe.forward(&tape, x, 3);
        let yt = tape.value(y.idx());

        assert_eq!(yt.shape(), (6, 4));
        // Les 2 premières lignes (batch 0, pos 0 et 1) doivent matcher pe[0..2]
        for i in 0..4
        {
            assert!(
                (yt.data[i] - pe.pe.data[i]).abs() < 1e-6,
                "batch0 pos0 mismatch at dim {i}"
            );
        }
        for i in 0..4
        {
            assert!(
                (yt.data[4 + i] - pe.pe.data[4 + i]).abs() < 1e-6,
                "batch0 pos1 mismatch at dim {i}"
            );
        }
        // batch 1 doit avoir les mêmes PE que batch 0
        for i in 0..4
        {
            assert!(
                (yt.data[12 + i] - pe.pe.data[i]).abs() < 1e-6,
                "batch1 pos0 mismatch at dim {i}"
            );
        }
    }

    #[test]
    fn forward_3d_preserves_shape() {
        let pe = PositionalEncoding::new(8, 16);
        let tape = Tape::new();
        let x = tape.input(Tensor::zeros(12, 8)); // batch=3, seq=4
        let x_3d = crate::tensor::tensor3d::Var3D::from_var(x, 3, 4, 8);
        let y_3d = pe.forward_3d(&tape, x_3d);
        assert_eq!(y_3d.shape(), (3, 4, 8));
    }

    #[test]
    fn gradient_flows_through_pe() {
        let pe = PositionalEncoding::new(4, 8);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0; 8], 2, 4));
        let y = pe.forward(&tape, x, 2);
        let loss = y.sum();
        loss.backward();
        let g = tape.grad(x.idx());
        // grad = 1 partout car loss = sum(y) et y = x + pe(const)
        for &v in &g.data
        {
            assert!((v - 1.0).abs() < 1e-6, "gradient should be 1, got {}", v);
        }
    }
}
