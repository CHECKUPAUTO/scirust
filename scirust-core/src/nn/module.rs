// scirust-core/src/nn/module.rs
//
// Trait Module + couches concrètes.
// Construit dessus de Initializer pour l'init des poids et de Loss pour
// l'entraînement.
//
// Pattern d'usage :
//
//   let mut rng = PcgEngine::new(42);
//   let mut model = Sequential::new()
//       .push(Linear::new(784, 128, &KaimingNormal, &Zeros, &mut rng).with_name("fc1"))
//       .push(ReLU)
//       .push(Linear::new(128, 10, &XavierUniform, &Zeros, &mut rng).with_name("fc2"));
//
//   for epoch in 0..n_epochs {
//       let tape = Tape::new();
//       let x = tape.input(batch_x);
//       let y = tape.input(batch_y);
//       let pred = model.forward(&tape, x);
//       let loss = MseLoss.forward(pred, y);
//       loss.backward();
//       optimizer.step(&model.parameter_indices(), &tape);
//       model.sync(&tape);
//   }

use std::collections::HashMap;
use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::init::Initializer;
use crate::nn::rng::PcgEngine;

// ================================================================== //
//  Trait Module                                                       //
// ================================================================== //

pub trait Module {
    /// Forward pass. Crée des Var pour les paramètres sur le tape, mémorise
    /// leurs indices pour le sync.
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t>;

    /// Indices des paramètres entraînables sur le tape, valides après forward().
    fn parameter_indices(&self) -> Vec<usize>;

    /// Récupère les valeurs mises à jour depuis le tape (post-step).
    fn sync(&mut self, tape: &Tape);

    /// (nom, tensor) pour la sérialisation.
    fn state_dict(&self) -> Vec<(String, Tensor)>;

    /// Charge depuis un state_dict. Renvoie le nombre de tensors chargés.
    fn load_state_dict(&mut self, dict: &HashMap<String, Tensor>) -> usize;

    /// Mode entraînement (pertinent pour Dropout, BatchNorm…).
    fn train(&mut self, _mode: bool) { /* default: no-op */ }
}

// ================================================================== //
//  Linear — y = x @ W + b                                             //
// ================================================================== //

pub struct Linear {
    pub weight:   Tensor,            // (in_features, out_features)
    pub bias:     Tensor,            // (1, out_features)
    last_w_idx:   Option<usize>,
    last_b_idx:   Option<usize>,
    pub name:     String,
}

impl Linear {
    /// Construit une couche linéaire avec deux Initializers séparés
    /// (un pour les poids, un pour les biais).
    pub fn new<W: Initializer, B: Initializer>(
        in_features:  usize,
        out_features: usize,
        weight_init:  &W,
        bias_init:    &B,
        rng:          &mut PcgEngine,
    ) -> Self {
        let mut weight = Tensor::zeros(in_features, out_features);
        let mut bias   = Tensor::zeros(1, out_features);
        weight_init.fill(&mut weight, rng);
        bias_init  .fill(&mut bias,   rng);
        Self {
            weight, bias,
            last_w_idx: None, last_b_idx: None,
            name: format!("linear_{}_{}", in_features, out_features),
        }
    }

    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.into();
        self
    }
}

impl Module for Linear {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        let w = tape.input(self.weight.clone());
        let b = tape.input(self.bias.clone());
        self.last_w_idx = Some(w.idx());
        self.last_b_idx = Some(b.idx());
        // x:(N, in) @ w:(in, out) → (N, out), broadcast b:(1, out)
        input.matmul(w).add_broadcast(b)
    }

    fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::with_capacity(2);
        if let Some(i) = self.last_w_idx { v.push(i); }
        if let Some(i) = self.last_b_idx { v.push(i); }
        v
    }

    fn sync(&mut self, tape: &Tape) {
        if let Some(i) = self.last_w_idx { self.weight = tape.value(i); }
        if let Some(i) = self.last_b_idx { self.bias   = tape.value(i); }
    }

    fn state_dict(&self) -> Vec<(String, Tensor)> {
        vec![
            (format!("{}.weight", self.name), self.weight.clone()),
            (format!("{}.bias",   self.name), self.bias.clone()),
        ]
    }

    fn load_state_dict(&mut self, dict: &HashMap<String, Tensor>) -> usize {
        let mut loaded = 0;
        if let Some(t) = dict.get(&format!("{}.weight", self.name)) {
            self.weight = t.clone(); loaded += 1;
        }
        if let Some(t) = dict.get(&format!("{}.bias", self.name)) {
            self.bias = t.clone(); loaded += 1;
        }
        loaded
    }
}

// ================================================================== //
//  Activations — sans paramètres                                      //
// ================================================================== //

macro_rules! impl_stateless_activation {
    ($name:ident, $method:ident) => {
        pub struct $name;
        impl Module for $name {
            fn forward<'t>(&mut self, _tape: &'t Tape, input: Var<'t>) -> Var<'t> {
                input.$method()
            }
            fn parameter_indices(&self) -> Vec<usize> { vec![] }
            fn sync(&mut self, _: &Tape) {}
            fn state_dict(&self) -> Vec<(String, Tensor)> { vec![] }
            fn load_state_dict(&mut self, _: &HashMap<String, Tensor>) -> usize { 0 }
        }
    };
}

impl_stateless_activation!(ReLU, relu);
impl_stateless_activation!(Sigmoid, sigmoid);

// ================================================================== //
//  Dropout                                                            //
// ================================================================== //

pub struct Dropout {
    pub p:        f32,
    pub training: bool,
    rng:          PcgEngine,
}

impl Dropout {
    pub fn new(p: f32, seed: u64) -> Self {
        assert!(p >= 0.0 && p < 1.0, "Dropout p must be in [0, 1)");
        Self { p, training: true, rng: PcgEngine::new(seed) }
    }
}

impl Module for Dropout {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        if !self.training || self.p == 0.0 { return input; }
        let (rows, cols) = input.shape();
        let scale = 1.0 / (1.0 - self.p);
        let mask: Vec<f32> = (0..rows * cols)
            .map(|_| if self.rng.float() < self.p { 0.0 } else { scale })
            .collect();
        let mask_v = tape.input(Tensor::from_vec(mask, rows, cols));
        input.hadamard(mask_v)
    }

    fn parameter_indices(&self) -> Vec<usize> { vec![] }
    fn sync(&mut self, _tape: &Tape) {}
    fn state_dict(&self) -> Vec<(String, Tensor)> { vec![] }
    fn load_state_dict(&mut self, _: &HashMap<String, Tensor>) -> usize { 0 }
    fn train(&mut self, mode: bool) { self.training = mode; }
}

// ================================================================== //
//  Sequential                                                         //
// ================================================================== //

pub struct Sequential {
    pub layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new() -> Self { Self { layers: Vec::new() } }

    pub fn push<M: Module + 'static>(mut self, m: M) -> Self {
        self.layers.push(Box::new(m));
        self
    }
}

impl Module for Sequential {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        let mut x = input;
        for layer in &mut self.layers { x = layer.forward(tape, x); }
        x
    }

    fn parameter_indices(&self) -> Vec<usize> {
        self.layers.iter().flat_map(|l| l.parameter_indices()).collect()
    }

    fn sync(&mut self, tape: &Tape) {
        for layer in &mut self.layers { layer.sync(tape); }
    }

    fn state_dict(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            for (name, t) in layer.state_dict() {
                out.push((format!("layers.{i}.{name}"), t));
            }
        }
        out
    }

    fn load_state_dict(&mut self, dict: &HashMap<String, Tensor>) -> usize {
        let mut total = 0;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("layers.{i}.");
            let stripped: HashMap<String, Tensor> = dict.iter()
                .filter_map(|(k, v)| k.strip_prefix(&prefix).map(|s| (s.to_string(), v.clone())))
                .collect();
            total += layer.load_state_dict(&stripped);
        }
        total
    }

    fn train(&mut self, mode: bool) {
        for layer in &mut self.layers { layer.train(mode); }
    }
}

// ================================================================== //
//  Tests                                                              //
// ================================================================== //
#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::init::{KaimingNormal, XavierUniform, Zeros};

    #[test]
    fn linear_forward_shape() {
        let mut rng = PcgEngine::new(1);
        let mut layer = Linear::new(3, 5, &KaimingNormal, &Zeros, &mut rng);
        let tape = Tape::new();
        let input = tape.input(Tensor::from_vec(vec![1.0; 6], 2, 3));
        let output = layer.forward(&tape, input);
        assert_eq!(output.shape(), (2, 5));
    }

    #[test]
    fn linear_bias_init_zero() {
        let mut rng = PcgEngine::new(1);
        let layer = Linear::new(4, 8, &KaimingNormal, &Zeros, &mut rng);
        assert!(layer.bias.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn sequential_state_dict_count() {
        let mut rng = PcgEngine::new(1);
        let model = Sequential::new()
            .push(Linear::new(2, 4, &XavierUniform, &Zeros, &mut rng).with_name("fc1"))
            .push(ReLU)
            .push(Linear::new(4, 1, &XavierUniform, &Zeros, &mut rng).with_name("fc2"));

        let dict = model.state_dict();
        assert_eq!(dict.len(), 4); // fc1.w, fc1.b, fc2.w, fc2.b
        assert!(dict.iter().any(|(k, _)| k == "layers.0.fc1.weight"));
        assert!(dict.iter().any(|(k, _)| k == "layers.2.fc2.bias"));
    }

    #[test]
    fn dropout_eval_passthrough() {
        let mut d = Dropout::new(0.5, 42);
        d.train(false);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0; 10], 1, 10));
        let y = d.forward(&tape, x);
        assert!(tape.value(y.idx()).data.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn parameter_indices_after_forward() {
        let mut rng = PcgEngine::new(1);
        let mut model = Sequential::new()
            .push(Linear::new(2, 3, &KaimingNormal, &Zeros, &mut rng))
            .push(ReLU)
            .push(Linear::new(3, 1, &KaimingNormal, &Zeros, &mut rng));

        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0], 1, 2));
        let _ = model.forward(&tape, x);

        // 2 Linear * 2 paramètres chacun = 4
        assert_eq!(model.parameter_indices().len(), 4);
    }
}
