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
}

/// Sinusoidal encoding value at (pos, i) for a given `d_model` (Vaswani et al.).
/// Computed on demand — see [`PositionalEncoding`] for why the table is not
/// materialised.
#[inline]
fn pe_value(d_model: usize, pos: usize, i: usize) -> f32 {
    let div = 10000_f32.powf(2.0 * (i / 2) as f32 / d_model as f32);
    if i.is_multiple_of(2)
    {
        (pos as f32 / div).sin()
    }
    else
    {
        (pos as f32 / div).cos()
    }
}

impl PositionalEncoding {
    pub fn new(d_model: usize, max_seq_len: usize) -> Self {
        // The encoding is a closed-form function of (pos, i), so nothing is
        // precomputed here: materialising a full (max_seq_len, d_model) table
        // eagerly allocated ~128 MB for the MiniLLM default (max_seq_len=125_000)
        // even though a forward only ever touches `seq_len` rows. Rows are now
        // computed on the fly (bit-for-bit identical to the old table).
        Self {
            d_model,
            max_seq_len,
        }
    }

    /// The positional encoding row for a single absolute position (length
    /// `d_model`). Used for incremental (KV-cache) decoding, where one token is
    /// processed at a time and must receive the encoding for *its* position.
    pub fn encoding_at(&self, pos: usize) -> Vec<f32> {
        assert!(
            pos < self.max_seq_len,
            "encoding_at: pos {pos} out of range"
        );
        (0..self.d_model)
            .map(|i| pe_value(self.d_model, pos, i))
            .collect()
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

        // Construire le PE broadcasté (batch*seq_len, d_model), calculé à la
        // volée (aucune table pré-allouée).
        let mut broadcasted = vec![0.0f32; total_rows * d];
        for b in 0..batch
        {
            for pos in 0..seq_len
            {
                for i in 0..d
                {
                    broadcasted[(b * seq_len + pos) * d + i] = pe_value(d, pos, i);
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
    fn encoding_at_length_and_known_values() {
        let pe = PositionalEncoding::new(8, 16);
        let row0 = pe.encoding_at(0);
        assert_eq!(row0.len(), 8);
        // pos 0: sin(0)=0 at even dims, cos(0)=1 at odd dims.
        for (i, &v) in row0.iter().enumerate()
        {
            let expected = if i % 2 == 0 { 0.0 } else { 1.0 };
            assert!(
                (v - expected).abs() < 1e-6,
                "row0[{i}] = {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn pe_values_in_range() {
        let pe = PositionalEncoding::new(16, 32);
        for pos in 0..pe.max_seq_len
        {
            for &v in &pe.encoding_at(pos)
            {
                assert!((-1.0..=1.0).contains(&v), "PE value out of [-1, 1]: {v}");
            }
        }
    }

    // Constructing with a huge max_seq_len must be O(1) now — it previously
    // eagerly allocated ~128 MB and ran a 125k*d_model build loop.
    #[test]
    fn new_with_huge_max_seq_len_is_cheap() {
        let pe = PositionalEncoding::new(256, 125_000);
        assert_eq!(pe.d_model, 256);
        assert_eq!(pe.max_seq_len, 125_000);
        // Only the requested row is materialised, on demand.
        assert_eq!(pe.encoding_at(124_999).len(), 256);
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
        let row0 = pe.encoding_at(0);
        let row1 = pe.encoding_at(1);
        for i in 0..4
        {
            // batch 0, pos 0 / pos 1
            assert!((yt.data[i] - row0[i]).abs() < 1e-6, "batch0 pos0 dim {i}");
            assert!(
                (yt.data[4 + i] - row1[i]).abs() < 1e-6,
                "batch0 pos1 dim {i}"
            );
            // batch 1 has the same PE as batch 0
            assert!(
                (yt.data[12 + i] - row0[i]).abs() < 1e-6,
                "batch1 pos0 dim {i}"
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
