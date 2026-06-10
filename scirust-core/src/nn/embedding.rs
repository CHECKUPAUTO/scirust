// scirust-core/src/nn/embedding.rs
//
// Embedding layer — lookup table de tokens vers vecteurs denses.
//
// Input  : (batch, seq_len) contenant des indices f32 (0..vocab_size-1)
// Output : (batch * seq_len, embedding_dim)

use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::init::Initializer;
use crate::nn::module::Module;
use crate::nn::rng::PcgEngine;
use std::collections::HashMap;

pub struct Embedding {
    pub weight: Tensor, // (vocab_size, embedding_dim)
    pub vocab_size: usize,
    pub embedding_dim: usize,
    last_w_idx: Option<usize>,
    pub name: String,
}

impl Embedding {
    pub fn new<I: Initializer>(
        vocab_size: usize,
        embedding_dim: usize,
        init: &I,
        rng: &mut PcgEngine,
    ) -> Self {
        let mut weight = Tensor::zeros(vocab_size, embedding_dim);
        init.fill(&mut weight, vocab_size, embedding_dim, rng);
        Self {
            weight,
            vocab_size,
            embedding_dim,
            last_w_idx: None,
            name: format!("emb_{vocab_size}_{embedding_dim}"),
        }
    }

    #[must_use]
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.into();
        self
    }
}

impl Clone for Embedding {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            vocab_size: self.vocab_size,
            embedding_dim: self.embedding_dim,
            last_w_idx: None,
            name: self.name.clone(),
        }
    }
}

impl Module for Embedding {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        let input_t = tape.value(input.idx());
        let n = input_t.rows * input_t.cols;
        let mut indices = Vec::with_capacity(n);
        for i in 0..n
        {
            let idx = input_t.data[i] as u32;
            assert!(
                (idx as usize) < self.vocab_size,
                "Embedding: index {} >= vocab_size {}",
                idx,
                self.vocab_size
            );
            indices.push(idx);
        }

        let table_var = tape.input(self.weight.clone());
        self.last_w_idx = Some(table_var.idx());
        table_var.embedding(indices)
    }

    fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::new();
        if let Some(i) = self.last_w_idx
        {
            v.push(i);
        }
        v
    }

    fn sync(&mut self, tape: &Tape) {
        if let Some(i) = self.last_w_idx
        {
            self.weight = tape.value(i);
        }
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.insert(format!("{}.weight", self.name), self.weight.clone());
        map
    }

    fn load_state_dict(&mut self, sd: &HashMap<String, Tensor>) -> crate::error::Result<()> {
        let w = sd
            .get(&format!("{}.weight", self.name))
            .ok_or_else(|| format!("missing key: {}.weight", self.name))?;
        if w.shape() != (self.vocab_size, self.embedding_dim)
        {
            return Err(format!(
                "weight shape mismatch: expected {:?}, got {:?}",
                (self.vocab_size, self.embedding_dim),
                w.shape()
            )
            .into());
        }
        self.weight = w.clone();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::init::Zeros;

    #[test]
    fn embedding_lookup_correct() {
        let mut rng = PcgEngine::new(0);
        let mut emb = Embedding::new(4, 3, &Zeros, &mut rng);
        // weight est zero-init, on le remplace manuellement
        emb.weight = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0, // token 0
                4.0, 5.0, 6.0, // token 1
                7.0, 8.0, 9.0, // token 2
                10.0, 11.0, 12.0, // token 3
            ],
            4,
            3,
        );

        let tape = Tape::new();
        // batch=2, seq_len=2 : tokens [0, 2] et [1, 3]
        let x = tape.input(Tensor::from_vec(vec![0.0, 2.0, 1.0, 3.0], 2, 2));
        let y = emb.forward(&tape, x);
        let yt = tape.value(y.idx());

        // output shape = (batch*seq_len, emb_dim) = (4, 3)
        assert_eq!(yt.shape(), (4, 3));

        // row 0 = token 0
        assert_eq!(yt.data[0..3], vec![1.0, 2.0, 3.0]);
        // row 1 = token 2
        assert_eq!(yt.data[3..6], vec![7.0, 8.0, 9.0]);
        // row 2 = token 1
        assert_eq!(yt.data[6..9], vec![4.0, 5.0, 6.0]);
        // row 3 = token 3
        assert_eq!(yt.data[9..12], vec![10.0, 11.0, 12.0]);
    }

    #[test]
    fn embedding_gradient_flows_to_weight() {
        let mut rng = PcgEngine::new(0);
        let mut emb = Embedding::new(3, 2, &Zeros, &mut rng);
        emb.weight = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);

        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![0.0, 1.0, 2.0], 1, 3));
        let y = emb.forward(&tape, x);
        let loss = y.sum();
        loss.backward();

        let w_idx = emb.parameter_indices()[0];
        let g_w = tape.grad(w_idx);

        // token 0 apparaît 1 fois → grad = [1, 1]
        assert_eq!(g_w.data[0], 1.0);
        assert_eq!(g_w.data[1], 1.0);
        // token 1 apparaît 1 fois → grad = [1, 1]
        assert_eq!(g_w.data[2], 1.0);
        assert_eq!(g_w.data[3], 1.0);
        // token 2 apparaît 1 fois → grad = [1, 1]
        assert_eq!(g_w.data[4], 1.0);
        assert_eq!(g_w.data[5], 1.0);
    }

    #[test]
    fn embedding_state_dict_round_trip() {
        let mut rng = PcgEngine::new(0);
        let emb1 = Embedding::new(4, 3, &Zeros, &mut rng);
        let sd = emb1.state_dict();

        let mut rng2 = PcgEngine::new(99);
        let mut emb2 = Embedding::new(4, 3, &Zeros, &mut rng2);
        emb2.load_state_dict(&sd).unwrap();

        assert_eq!(emb2.weight.data, emb1.weight.data);
    }
}
