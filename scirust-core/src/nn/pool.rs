// scirust-core/src/nn/pool.rs
//
// MaxPool2d — réduction spatiale par fenêtre glissante.
//
// Convention : input shape (B, C·H·W), output shape (B, C·H_out·W_out).

use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::module::Module;
use std::collections::HashMap;

#[derive(Clone)]
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
        input.max_pool2d(c, h, w, self.kernel, self.stride)
    }

    fn parameter_indices(&self) -> Vec<usize> { vec![] }
    fn sync(&mut self, _tape: &Tape) {}

    fn state_dict(&self) -> HashMap<String, Tensor> { HashMap::new() }
    fn load_state_dict(&mut self, _sd: &HashMap<String, Tensor>) -> std::result::Result<(), String> { Ok(()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maxpool_2x2_stride_2() {
        let mut pool = MaxPool2d::new(2, 2).input_shape(1, 4, 4);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(
            (1..=16).map(|x| x as f32).collect(), 1, 16));
        let y = pool.forward(&tape, x);
        let yt = tape.value(y.idx());
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
        assert_eq!(g.data, vec![0.0, 1.0, 0.0, 0.0]);
    }
}
