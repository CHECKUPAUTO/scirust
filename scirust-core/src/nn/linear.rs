// scirust-core/src/nn/linear.rs
//
// Linear layer : y = x @ W + b
//
// Shapes :
//   - input  : (batch, in_features)
//   - weight : (in_features, out_features)
//   - bias   : (1, out_features)  — broadcast row-wise sur le batch
//   - output : (batch, out_features)
//
// Architecture :
//   - weight et bias sont stockés comme Tensor dans la struct (persistent
//     entre les époques).
//   - À chaque forward(), on push weight et bias comme inputs sur la
//     nouvelle tape, on fait matmul + add_bias, et on garde leurs idx
//     pour parameter_indices et sync.

use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::init::Initializer;
use crate::nn::module::Module;
use crate::nn::rng::PcgEngine;

pub struct Linear {
    pub weight: Tensor, // (in_features, out_features)
    pub bias: Tensor,   // (1, out_features)
    pub in_features: usize,
    pub out_features: usize,
    last_w_idx: Option<usize>,
    last_b_idx: Option<usize>,
}

impl Linear {
    pub fn new<W: Initializer, B: Initializer>(
        in_features: usize,
        out_features: usize,
        w_init: &W,
        b_init: &B,
        rng: &mut PcgEngine,
    ) -> Self {
        let mut weight = Tensor::zeros(in_features, out_features);
        w_init.fill(&mut weight, in_features, out_features, rng);
        let mut bias = Tensor::zeros(1, out_features);
        b_init.fill(&mut bias, in_features, out_features, rng);
        Self {
            weight,
            bias,
            in_features,
            out_features,
            last_w_idx: None,
            last_b_idx: None,
        }
    }
}

impl Clone for Linear {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            in_features: self.in_features,
            out_features: self.out_features,
            last_w_idx: None,
            last_b_idx: None,
        }
    }
}

impl Module for Linear {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        let w = tape.input(self.weight.clone());
        let b = tape.input(self.bias.clone());
        self.last_w_idx = Some(w.idx());
        self.last_b_idx = Some(b.idx());
        input.try_matmul(w).and_then(|x| x.try_add_bias(b)).unwrap()
    }

    fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::new();
        if let Some(i) = self.last_w_idx
        {
            v.push(i);
        }
        if let Some(i) = self.last_b_idx
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
        if let Some(i) = self.last_b_idx
        {
            self.bias = tape.value(i);
        }
    }

    fn state_dict(&self) -> std::collections::HashMap<String, Tensor> {
        let mut map = std::collections::HashMap::new();
        map.insert("weight".to_string(), self.weight.clone());
        map.insert("bias".to_string(), self.bias.clone());
        map
    }

    fn load_state_dict(
        &mut self,
        sd: &std::collections::HashMap<String, Tensor>,
    ) -> crate::error::Result<()> {
        let w = sd
            .get("weight")
            .ok_or_else(|| "missing key: weight".to_string())?;
        let b = sd
            .get("bias")
            .ok_or_else(|| "missing key: bias".to_string())?;
        if w.shape() != (self.in_features, self.out_features)
        {
            return Err(crate::error::SciRustError::InvalidConfig(format!(
                "weight shape mismatch: expected {:?}, got {:?}",
                (self.in_features, self.out_features),
                w.shape()
            )));
        }
        if b.shape() != (1, self.out_features)
        {
            return Err(crate::error::SciRustError::InvalidConfig(format!(
                "bias shape mismatch: expected {:?}, got {:?}",
                (1, self.out_features),
                b.shape()
            )));
        }
        self.weight = w.clone();
        self.bias = b.clone();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::init::{KaimingNormal, Zeros};

    #[test]
    fn linear_construction() {
        let mut rng = PcgEngine::new(42);
        let lin = Linear::new(4, 8, &KaimingNormal, &Zeros, &mut rng);
        assert_eq!(lin.weight.shape(), (4, 8));
        assert_eq!(lin.bias.shape(), (1, 8));
    }

    #[test]
    fn linear_forward_shape_correct() {
        let mut rng = PcgEngine::new(42);
        let mut lin = Linear::new(3, 5, &KaimingNormal, &Zeros, &mut rng);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0; 6], 2, 3)); // batch=2, in=3
        let y = lin.forward(&tape, x);
        assert_eq!(y.shape(), (2, 5));
    }

    #[test]
    fn linear_with_zero_weights_produces_bias() {
        // Si W = 0 et b = [1, 2, 3], alors y[i] = b pour tout i.
        let mut rng = PcgEngine::new(0);
        let mut lin = Linear::new(2, 3, &Zeros, &Zeros, &mut rng);
        lin.bias = Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3);

        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![5.0, 7.0, 9.0, 11.0], 2, 2));
        let y = lin.forward(&tape, x);
        let v = tape.value(y.idx());
        assert_eq!(v.data, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn linear_gradient_flows_to_weight_and_input() {
        let mut rng = PcgEngine::new(42);
        let mut lin = Linear::new(2, 1, &KaimingNormal, &Zeros, &mut rng);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![3.0, 5.0], 1, 2));
        let y = lin.forward(&tape, x);
        let loss = y.sum();
        tape.backward(loss.idx());

        // Gradient sur input doit être W^T (broadcastée sur batch=1)
        let g_x = tape.grad(x.idx());
        assert_eq!(g_x.shape(), (1, 2));
        let max_abs: f32 = g_x.data.iter().map(|v| v.abs()).fold(0.0, f32::max);
        assert!(max_abs > 1e-6, "gradient on input is zero");

        // Gradient sur weight doit être x^T @ grad_out (∝ x.T)
        let w_idx = lin.parameter_indices()[0];
        let g_w = tape.grad(w_idx);
        assert_eq!(g_w.shape(), (2, 1));
        // grad_out = 1 (scalar sum), donc g_w = x.T = [3, 5].T
        assert!((g_w.data[0] - 3.0).abs() < 1e-5);
        assert!((g_w.data[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn linear_sync_persists_updated_weights() {
        let mut rng = PcgEngine::new(42);
        let mut lin = Linear::new(2, 1, &Zeros, &Zeros, &mut rng);
        let original_weight = lin.weight.clone();

        let tape = Tape::new();
        let _y = lin.forward(&tape, tape.input(Tensor::from_vec(vec![1.0, 1.0], 1, 2)));

        // Simule une mise à jour des poids sur la tape
        let w_idx = lin.parameter_indices()[0];
        let new_w = Tensor::from_vec(vec![42.0, 43.0], 2, 1);
        tape.set_value(w_idx, new_w.clone());

        lin.sync(&tape);
        assert_eq!(lin.weight.data, new_w.data);
        assert_ne!(lin.weight.data, original_weight.data);
    }

    #[test]
    fn linear_parameter_indices_count() {
        let mut rng = PcgEngine::new(42);
        let mut lin = Linear::new(3, 5, &KaimingNormal, &Zeros, &mut rng);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0; 6], 2, 3));
        let _y = lin.forward(&tape, x);
        // Linear a 2 paramètres : weight et bias
        assert_eq!(lin.parameter_indices().len(), 2);
    }

    #[test]
    fn linear_state_dict_contains_weight_and_bias() {
        let mut rng = PcgEngine::new(42);
        let lin = Linear::new(3, 5, &KaimingNormal, &Zeros, &mut rng);
        let sd = lin.state_dict();
        assert_eq!(sd.len(), 2);

        let w = sd.get("weight").unwrap();
        assert_eq!(w.shape(), (3, 5));

        let b = sd.get("bias").unwrap();
        assert_eq!(b.shape(), (1, 5));
    }

    #[test]
    fn linear_state_dict_round_trip() {
        let mut rng = PcgEngine::new(42);
        let lin1 = Linear::new(2, 3, &KaimingNormal, &Zeros, &mut rng);
        let sd = lin1.state_dict();

        // Create a second Linear with different weights and load state_dict from first
        let mut rng2 = PcgEngine::new(99);
        let mut lin2 = Linear::new(2, 3, &Zeros, &Zeros, &mut rng2);
        // Before load, lin2 has all zeros
        assert!(lin2.weight.data.iter().all(|&x| x == 0.0));

        lin2.load_state_dict(&sd).unwrap();

        // After load, lin2 should match lin1
        assert_eq!(lin2.weight.data, lin1.weight.data);
        assert_eq!(lin2.bias.data, lin1.bias.data);
    }

    #[test]
    fn linear_load_missing_keys_returns_error() {
        let mut rng = PcgEngine::new(42);
        let mut lin = Linear::new(2, 3, &KaimingNormal, &Zeros, &mut rng);
        let mut sd = std::collections::HashMap::new();
        sd.insert("weight".to_string(), Tensor::zeros(2, 3));
        // missing bias
        let res = lin.load_state_dict(&sd);
        assert!(res.is_err(), "expected error for missing bias");
        assert!(
            format!("{}", res.unwrap_err()).contains("bias"),
            "error should mention missing key"
        );
    }
}
