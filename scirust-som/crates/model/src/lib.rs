//! SOM Model Architecture (<= 300M parameters).
//! Implements a Graph Transformer architecture with Encoder and Fusion layers.

use scirust_core::autodiff::reverse::{Tape, Var};
use scirust_core::nn::linear::Linear;
use scirust_core::nn::Module;
use scirust_core::nn::rng::PcgEngine;
use scirust_core::nn::init::{KaimingNormal, Zeros};

pub struct GraphEncoder {
    pub node_emb: Linear,
    pub edge_emb: Linear,
}

impl GraphEncoder {
    pub fn new(d_model: usize, rng: &mut PcgEngine) -> Self {
        Self {
            node_emb: Linear::new(d_model, d_model, &KaimingNormal, &Zeros, rng),
            edge_emb: Linear::new(d_model, d_model, &KaimingNormal, &Zeros, rng),
        }
    }

    pub fn forward<'t>(&mut self, tape: &'t Tape, nodes: Var<'t>, edges: Var<'t>) -> Var<'t> {
        let n = self.node_emb.forward(tape, nodes);
        let e = self.edge_emb.forward(tape, edges);
        n.add(e) // Simple fusion for encoder
    }
}

pub struct GraphTransformerLayer {
    pub qkv: Linear,
    pub output: Linear,
}

impl GraphTransformerLayer {
    pub fn new(d_model: usize, rng: &mut PcgEngine) -> Self {
        Self {
            qkv: Linear::new(d_model, d_model * 3, &KaimingNormal, &Zeros, rng),
            output: Linear::new(d_model, d_model, &KaimingNormal, &Zeros, rng),
        }
    }

    pub fn forward<'t>(&mut self, tape: &'t Tape, x: Var<'t>) -> Var<'t> {
        let qkv = self.qkv.forward(tape, x);
        // Simplified self-attention logic for Graph Transformer
        let d = qkv.shape().1 / 3;
        let q = qkv.slice_cols(0, d);
        let k = qkv.slice_cols(d, d);
        let v = qkv.slice_cols(2 * d, d);

        let scores = q.matmul(k.transpose()).scale(1.0 / (d as f32).sqrt());
        let attn = scores.softmax(1);
        let context = attn.matmul(v);

        self.output.forward(tape, context).add(x).relu() // Residual + Relu
    }
}

pub struct SomModel {
    pub encoder: GraphEncoder,
    pub transformer_layers: Vec<GraphTransformerLayer>,
    pub fusion_layer: Linear,
    pub ownership_head: Linear,
    pub borrow_head: Linear,
    pub lifetime_head: Linear,
    pub alias_head: Linear,
    pub escape_head: Linear,
    pub mutability_head: Linear,
    pub unsafe_head: Linear,
    pub confidence_head: Linear,
}

impl SomModel {
    pub fn new(d_model: usize, n_layers: usize, rng: &mut PcgEngine) -> Self {
        let encoder = GraphEncoder::new(d_model, rng);
        let mut layers = Vec::new();
        for _ in 0..n_layers {
            layers.push(GraphTransformerLayer::new(d_model, rng));
        }
        let fusion_layer = Linear::new(d_model, d_model, &KaimingNormal, &Zeros, rng);

        Self {
            encoder,
            transformer_layers: layers,
            fusion_layer,
            ownership_head: Linear::new(d_model, 4, &KaimingNormal, &Zeros, rng),
            borrow_head: Linear::new(d_model, 3, &KaimingNormal, &Zeros, rng),
            lifetime_head: Linear::new(d_model, 1, &KaimingNormal, &Zeros, rng),
            alias_head: Linear::new(d_model, 1, &KaimingNormal, &Zeros, rng),
            escape_head: Linear::new(d_model, 1, &KaimingNormal, &Zeros, rng),
            mutability_head: Linear::new(d_model, 1, &KaimingNormal, &Zeros, rng),
            unsafe_head: Linear::new(d_model, 1, &KaimingNormal, &Zeros, rng),
            confidence_head: Linear::new(d_model, 1, &KaimingNormal, &Zeros, rng),
        }
    }

    pub fn forward<'t>(&mut self, tape: &'t Tape, nodes: Var<'t>, edges: Var<'t>) -> SomOutput<'t> {
        let mut h = self.encoder.forward(tape, nodes, edges);

        for layer in &mut self.transformer_layers {
            h = layer.forward(tape, h);
        }

        h = self.fusion_layer.forward(tape, h).relu();

        SomOutput {
            ownership: self.ownership_head.forward(tape, h),
            borrow: self.borrow_head.forward(tape, h),
            lifetime: self.lifetime_head.forward(tape, h),
            alias: self.alias_head.forward(tape, h),
            escape: self.escape_head.forward(tape, h),
            mutability: self.mutability_head.forward(tape, h),
            unsafe_prob: self.unsafe_head.forward(tape, h),
            confidence: self.confidence_head.forward(tape, h),
        }
    }
}

pub struct SomOutput<'t> {
    pub ownership: Var<'t>,
    pub borrow: Var<'t>,
    pub lifetime: Var<'t>,
    pub alias: Var<'t>,
    pub escape: Var<'t>,
    pub mutability: Var<'t>,
    pub unsafe_prob: Var<'t>,
    pub confidence: Var<'t>,
}
