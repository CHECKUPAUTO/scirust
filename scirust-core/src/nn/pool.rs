// scirust-core/src/nn/pool.rs
//
// MaxPool2d via Op::MaxPool2d sur la tape (ajouté par le patch v6.1).
//
// Convention : input shape (B, C·H·W), output shape (B, C·H_out·W_out).
// Chaque fenêtre K×K par canal est réduite à son max.
//
// Backward : recompute du mask (Option B validée v6.1) — chaque cellule
// max d'une fenêtre reçoit le gradient correspondant. Tie-breaking :
// toutes les cellules égales au max reçoivent le gradient.

use std::collections::HashMap;
use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::module::Module;

pub struct MaxPool2d {
    pub kernel: usize,
    pub stride: usize,
    cached_c: Option<usize>,
    cached_h: Option<usize>,
    cached_w: Option<usize>,
}

impl MaxPool2d {
    pub fn new(kernel: usize, stride: usize) -> Self {
        Self { kernel, stride, cached_c: None, cached_h: None, cached_w: None }
    }

    /// Configure les dimensions C, H, W de l'input.
    pub fn input_shape(mut self, c: usize, h: usize, w: usize) -> Self {
        self.cached_c = Some(c);
        self.cached_h = Some(h);
        self.cached_w = Some(w);
        self
    }
}

impl Module for MaxPool2d {
    fn forward<'t>(&mut self, _tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        let (c, h, w) = match (self.cached_c, self.cached_h, self.cached_w) {
            (Some(c), Some(h), Some(w)) => (c, h, w),
            _ => panic!("MaxPool2d: utiliser .input_shape(c, h, w) avant le forward"),
        };
        // Délègue à la méthode Var::max_pool2d ajoutée dans le patch reverse.rs
        input.max_pool2d(c, h, w, self.kernel, self.stride)
    }

    fn parameter_indices(&self) -> Vec<usize> { vec![] }
    fn sync(&mut self, _tape: &Tape) {}
    fn state_dict(&self) -> Vec<(String, Tensor)> { vec![] }
    fn load_state_dict(&mut self, _: &HashMap<String, Tensor>) -> usize { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maxpool_2x2_stride_2() {
        let mut pool = MaxPool2d::new(2, 2).input_shape(1, 4, 4);
        let tape = Tape::new();
        // Image 4×4 :
        //   1  2  3  4
        //   5  6  7  8
        //   9 10 11 12
        //  13 14 15 16
        let x = tape.input(Tensor::from_vec(
            (1..=16).map(|x| x as f32).collect(), 1, 16));
        let y = pool.forward(&tape, x);
        let yt = tape.value(y.idx());
        // Output 2×2 : max de chaque bloc 2×2
        //   max(1,2,5,6)=6, max(3,4,7,8)=8
        //   max(9,10,13,14)=14, max(11,12,15,16)=16
        assert_eq!(yt.shape(), (1, 4));
        assert_eq!(yt.data, vec![6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn maxpool_grad_routes_to_max() {
        let mut pool = MaxPool2d::new(2, 2).input_shape(1, 2, 2);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0, 5.0, 3.0, 2.0], 1, 4));
        let y = pool.forward(&tape, x).sum();
        y.backward();
        let g = tape.grad(x.idx());
        // Seul le 5.0 (index 1) reçoit grad = 1
        assert_eq!(g.data, vec![0.0, 1.0, 0.0, 0.0]);
    }
}
