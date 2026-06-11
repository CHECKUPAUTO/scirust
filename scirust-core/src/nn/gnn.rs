use crate::autodiff::reverse::{Tape, Var};
use crate::nn::init::Initializer;
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::nn::rng::PcgEngine;

/// Graph Convolutional Network (GCN) layer.
pub struct GCNLayer {
    pub linear: Linear,
}

impl GCNLayer {
    pub fn new<W: Initializer, B: Initializer>(
        in_features: usize,
        out_features: usize,
        w_init: &W,
        b_init: &B,
        rng: &mut PcgEngine,
    ) -> Self {
        Self {
            linear: Linear::new(in_features, out_features, w_init, b_init, rng),
        }
    }

    /// Forward pass with adjacency matrix A (sparse or dense).
    /// H' = ReLU(A @ H @ W)
    pub fn forward_with_adj<'t>(&mut self, tape: &'t Tape, h: Var<'t>, adj: Var<'t>) -> Var<'t> {
        let wh = self.linear.forward(tape, h);
        adj.try_matmul(wh).unwrap().relu()
    }
}

pub struct GCN {
    pub layers: Vec<GCNLayer>,
}

impl GCN {
    pub fn new<W: Initializer, B: Initializer>(
        dims: &[usize],
        w_init: &W,
        b_init: &B,
        rng: &mut PcgEngine,
    ) -> Self {
        let mut layers = Vec::new();
        for i in 0..dims.len() - 1
        {
            layers.push(GCNLayer::new(dims[i], dims[i + 1], w_init, b_init, rng));
        }
        Self { layers }
    }

    pub fn forward<'t>(&mut self, tape: &'t Tape, x: Var<'t>, adj: Var<'t>) -> Var<'t> {
        let mut h = x;
        for layer in &mut self.layers
        {
            h = layer.forward_with_adj(tape, h, adj);
        }
        h
    }
}
