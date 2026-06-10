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
        Self {
            kernel,
            stride,
            cached_c: None,
            cached_h: None,
            cached_w: None,
        }
    }

    #[must_use]
    pub fn input_shape(mut self, c: usize, h: usize, w: usize) -> Self {
        self.cached_c = Some(c);
        self.cached_h = Some(h);
        self.cached_w = Some(w);
        self
    }
}

impl Module for MaxPool2d {
    fn forward<'t>(&mut self, _tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        let (c, h, w) = match (self.cached_c, self.cached_h, self.cached_w)
        {
            (Some(c), Some(h), Some(w)) => (c, h, w),
            _ => panic!("MaxPool2d: utiliser .input_shape(c, h, w) avant le forward"),
        };
        input.max_pool2d(c, h, w, self.kernel, self.stride)
    }

    fn parameter_indices(&self) -> Vec<usize> {
        vec![]
    }
    fn sync(&mut self, _tape: &Tape) {}

    fn state_dict(&self) -> HashMap<String, Tensor> {
        HashMap::new()
    }
    fn load_state_dict(&mut self, _sd: &HashMap<String, Tensor>) -> crate::error::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maxpool_2x2_stride_2() {
        let mut pool = MaxPool2d::new(2, 2).input_shape(1, 4, 4);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(
            (1..=16).map(|x| x as f32).collect(),
            1,
            16,
        ));
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

    #[test]
    fn maxpool_multi_channel() {
        let mut pool = MaxPool2d::new(2, 2).input_shape(2, 4, 4);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(
            (1..=32).map(|x| x as f32).collect(),
            1,
            32,
        ));
        let y = pool.forward(&tape, x);
        let yt = tape.value(y.idx());
        assert_eq!(yt.shape(), (1, 8));
        // Each 2x2 window max for each channel
        // Channel 1: [6,8,14,16], Channel 2: [22,24,30,32]
        assert_eq!(yt.data, vec![6.0, 8.0, 14.0, 16.0, 22.0, 24.0, 30.0, 32.0]);
    }

    #[test]
    fn maxpool_stride_smaller_than_kernel() {
        let mut pool = MaxPool2d::new(3, 1).input_shape(1, 4, 4);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(
            (1..=16).map(|x| x as f32).collect(),
            1,
            16,
        ));
        let y = pool.forward(&tape, x);
        let yt = tape.value(y.idx());
        // 4x4 with 3x3 kernel and stride 1 → 2x2 output
        assert_eq!(yt.shape(), (1, 4));
        // Max of each 3x3 window
        assert_eq!(yt.data, vec![11.0, 12.0, 15.0, 16.0]);
    }
}
