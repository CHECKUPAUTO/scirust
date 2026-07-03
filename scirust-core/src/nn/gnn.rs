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

    /// Forward pass with adjacency A, applying the ReLU activation iff
    /// `apply_activation`. The final layer of a classifier must stay LINEAR so
    /// its logits can be negative — clamping them to ≥0 biases softmax /
    /// cross-entropy and kills the gradient for negative pre-activations.
    pub fn forward_with_adj_act<'t>(
        &mut self,
        tape: &'t Tape,
        h: Var<'t>,
        adj: Var<'t>,
        apply_activation: bool,
    ) -> Var<'t> {
        let wh = self.linear.forward(tape, h);
        let ah = adj.try_matmul(wh).unwrap();
        if apply_activation { ah.relu() } else { ah }
    }

    /// Forward pass with adjacency matrix A (sparse or dense).
    /// H' = ReLU(A @ H @ W). For a hidden layer; the output layer should use
    /// [`forward_with_adj_act`] with `apply_activation = false`.
    pub fn forward_with_adj<'t>(&mut self, tape: &'t Tape, h: Var<'t>, adj: Var<'t>) -> Var<'t> {
        self.forward_with_adj_act(tape, h, adj, true)
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
        let last = self.layers.len().saturating_sub(1);
        // Hidden layers use ReLU; the final (output) layer stays linear so the
        // classification logits are unclamped.
        for (i, layer) in self.layers.iter_mut().enumerate()
        {
            h = layer.forward_with_adj_act(tape, h, adj, i < last);
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::reverse::Tensor;
    use crate::nn::init::{KaimingNormal, Zeros};

    // The output layer must be linear: with a suitable weight/adjacency the
    // final logits should be able to go negative. Under the old code (ReLU on
    // every layer, including the last) every output was clamped to >= 0.
    #[test]
    fn gcn_final_layer_can_produce_negative_logits() {
        let mut rng = PcgEngine::new(1);
        let mut model = GCN::new(&[3, 4, 2], &KaimingNormal, &Zeros, &mut rng);
        // Deterministic weights: hidden layer all +1 (so the ReLU'd hidden is
        // strictly positive), output layer all -1 (so linear logits are < 0).
        // A trailing ReLU on the output would clamp them to 0.
        model.layers[0].linear.weight = Tensor::from_vec(vec![1.0; 3 * 4], 3, 4);
        model.layers[1].linear.weight = Tensor::from_vec(vec![-1.0; 4 * 2], 4, 2);

        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0; 6], 2, 3));
        // Identity adjacency keeps each node's features.
        let adj = tape.input(Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], 2, 2));
        let out = model.forward(&tape, x, adj);
        let ov = tape.value(out.idx());
        assert_eq!(ov.shape(), (2, 2));
        assert!(
            ov.data.iter().any(|&v| v < -1e-6),
            "final GCN logits were clamped to >= 0 (final-layer ReLU not removed): {:?}",
            ov.data
        );
    }
}
