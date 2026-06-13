//! Reusable neural-network layers over the **N-D autograd tape**
//! ([`crate::autodiff::nd`]).
//!
//! This is the step from "the N-D tape can *express* a layer" to "here is the
//! layer": [`NdLinear`] holds its own parameters, runs a forward on an
//! [`NdTape`], and applies an SGD update from the gradients — so a real N-D
//! network can be built and trained. Parameter init is seeded ([`PcgEngine`]),
//! preserving determinism.

use crate::autodiff::nd::{NdTape, NdVar};
use crate::nn::rng::PcgEngine;
use crate::tensor::tensor_nd::TensorND;

/// A dense layer `y = x · W + b` acting on the **last axis** of an N-D input:
/// `(…, in) → (…, out)`. The leading axes (batch, heads, sequence, …) are
/// flattened for the matmul and restored afterwards.
pub struct NdLinear {
    weight: TensorND, // (in, out)
    bias: TensorND,   // (1, out)
    in_features: usize,
    out_features: usize,
    w_idx: Option<usize>,
    b_idx: Option<usize>,
}

impl NdLinear {
    /// New layer with seeded init: `W ~ U(-s, s)` with `s = 1/√in`, `b = 0`.
    pub fn new(in_features: usize, out_features: usize, rng: &mut PcgEngine) -> Self {
        let scale = (1.0 / in_features as f32).sqrt();
        let w: Vec<f32> = (0..in_features * out_features)
            .map(|_| rng.float_signed() * scale)
            .collect();
        Self {
            weight: TensorND::new(w, vec![in_features, out_features]),
            bias: TensorND::zeros(&[1, out_features]),
            in_features,
            out_features,
            w_idx: None,
            b_idx: None,
        }
    }

    /// Forward pass. Records the parameter node indices so [`Self::sgd_step`]
    /// can read their gradients after `backward`.
    pub fn forward<'t>(&mut self, tape: &'t NdTape, x: NdVar<'t>) -> NdVar<'t> {
        let xs = x.shape();
        let in_f = *xs.last().expect("NdLinear: input has no axes");
        assert_eq!(in_f, self.in_features, "NdLinear: input feature mismatch");
        let m: usize = xs[..xs.len() - 1].iter().product();

        let w = tape.input(self.weight.clone());
        let b = tape.input(self.bias.clone());
        self.w_idx = Some(w.idx());
        self.b_idx = Some(b.idx());

        let y2 = x.reshape(&[m, in_f]).matmul(w).add(b); // (m, out)
        let mut out_shape = xs[..xs.len() - 1].to_vec();
        out_shape.push(self.out_features);
        y2.reshape(&out_shape)
    }

    /// Apply `param -= lr · grad` for the weight and bias using the gradients
    /// returned by [`NdTape::backward`] (must follow a `forward` on that tape).
    pub fn sgd_step(&mut self, grads: &[TensorND], lr: f32) {
        if let Some(i) = self.w_idx
        {
            for (p, &g) in self.weight.data.iter_mut().zip(&grads[i].data)
            {
                *p -= lr * g;
            }
        }
        if let Some(i) = self.b_idx
        {
            for (p, &g) in self.bias.data.iter_mut().zip(&grads[i].data)
            {
                *p -= lr * g;
            }
        }
    }

    /// The weight tensor `(in, out)`.
    pub fn weight(&self) -> &TensorND {
        &self.weight
    }
}

/// Multi-head self-attention over the N-D tape, built from [`NdLinear`]
/// projections and the N-D attention math (`bmm` / `transpose_last2` /
/// `softmax`). Input and output are `(seq, d_model)`.
pub struct NdMultiHeadAttention {
    w_q: NdLinear,
    w_k: NdLinear,
    w_v: NdLinear,
    w_o: NdLinear,
    n_heads: usize,
    d_model: usize,
    d_head: usize,
}

impl NdMultiHeadAttention {
    /// New layer; `d_model` must be divisible by `n_heads`. Seeded init.
    pub fn new(d_model: usize, n_heads: usize, rng: &mut PcgEngine) -> Self {
        assert!(d_model % n_heads == 0, "d_model must divide n_heads");
        Self {
            w_q: NdLinear::new(d_model, d_model, rng),
            w_k: NdLinear::new(d_model, d_model, rng),
            w_v: NdLinear::new(d_model, d_model, rng),
            w_o: NdLinear::new(d_model, d_model, rng),
            n_heads,
            d_model,
            d_head: d_model / n_heads,
        }
    }

    /// `(seq, d_model) → (n_heads, seq, d_head)`.
    fn split_heads<'t>(&self, x: NdVar<'t>, seq: usize) -> NdVar<'t> {
        x.reshape(&[seq, self.n_heads, self.d_head])
            .permute(&[1, 0, 2])
    }

    /// Self-attention `softmax(Q·Kᵀ/√d)·V`, then the output projection.
    pub fn forward<'t>(&mut self, tape: &'t NdTape, x: NdVar<'t>) -> NdVar<'t> {
        let seq = x.shape()[0];

        let q = self.w_q.forward(tape, x);
        let q = self.split_heads(q, seq);
        let k = self.w_k.forward(tape, x);
        let k = self.split_heads(k, seq);
        let v = self.w_v.forward(tape, x);
        let v = self.split_heads(v, seq);

        let scale = tape.input(TensorND::new(
            vec![1.0 / (self.d_head as f32).sqrt()],
            vec![1],
        ));
        let scores = q.bmm(k.transpose_last2()).mul(scale).softmax(); // (h, seq, seq)
        let ctx = scores.bmm(v); // (h, seq, d_head)

        // Merge heads: (h, seq, d_head) → (seq, h, d_head) → (seq, d_model).
        let merged = ctx.permute(&[1, 0, 2]).reshape(&[seq, self.d_model]);
        self.w_o.forward(tape, merged)
    }

    /// SGD-update every projection from a `backward` result.
    pub fn sgd_step(&mut self, grads: &[TensorND], lr: f32) {
        self.w_q.sgd_step(grads, lr);
        self.w_k.sgd_step(grads, lr);
        self.w_v.sgd_step(grads, lr);
        self.w_o.sgd_step(grads, lr);
    }
}

/// Layer normalisation over the last axis with a learnable affine
/// (`y = γ·layernorm(x) + β`).
pub struct NdLayerNorm {
    gamma: TensorND, // (d,)
    beta: TensorND,  // (d,)
    eps: f32,
    g_idx: Option<usize>,
    b_idx: Option<usize>,
}

impl NdLayerNorm {
    /// New layer over the last axis of width `d`. `γ = 1`, `β = 0`.
    pub fn new(d: usize, eps: f32) -> Self {
        Self {
            gamma: TensorND::ones(&[d]),
            beta: TensorND::zeros(&[d]),
            eps,
            g_idx: None,
            b_idx: None,
        }
    }

    /// Forward: normalise the last axis then apply the broadcast affine.
    pub fn forward<'t>(&mut self, tape: &'t NdTape, x: NdVar<'t>) -> NdVar<'t> {
        let g = tape.input(self.gamma.clone());
        let b = tape.input(self.beta.clone());
        self.g_idx = Some(g.idx());
        self.b_idx = Some(b.idx());
        x.layernorm(self.eps).mul(g).add(b)
    }

    /// SGD-update `γ` and `β`.
    pub fn sgd_step(&mut self, grads: &[TensorND], lr: f32) {
        if let Some(i) = self.g_idx
        {
            for (p, &gv) in self.gamma.data.iter_mut().zip(&grads[i].data)
            {
                *p -= lr * gv;
            }
        }
        if let Some(i) = self.b_idx
        {
            for (p, &gv) in self.beta.data.iter_mut().zip(&grads[i].data)
            {
                *p -= lr * gv;
            }
        }
    }
}

/// A full **Pre-LN transformer block** over the N-D tape:
/// `x₁ = x + Attn(LN₁(x))`, `y = x₁ + FFN(LN₂(x₁))`. Input/output `(seq, d_model)`.
pub struct NdTransformerBlock {
    ln1: NdLayerNorm,
    attn: NdMultiHeadAttention,
    ln2: NdLayerNorm,
    ffn1: NdLinear,
    ffn2: NdLinear,
}

impl NdTransformerBlock {
    /// New block. Seeded init.
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize, rng: &mut PcgEngine) -> Self {
        Self {
            ln1: NdLayerNorm::new(d_model, 1e-5),
            attn: NdMultiHeadAttention::new(d_model, n_heads, rng),
            ln2: NdLayerNorm::new(d_model, 1e-5),
            ffn1: NdLinear::new(d_model, d_ff, rng),
            ffn2: NdLinear::new(d_ff, d_model, rng),
        }
    }

    /// Pre-LN forward with residual connections.
    pub fn forward<'t>(&mut self, tape: &'t NdTape, x: NdVar<'t>) -> NdVar<'t> {
        let a = self.ln1.forward(tape, x);
        let a = self.attn.forward(tape, a);
        let x1 = x.add(a); // residual 1
        let f = self.ln2.forward(tape, x1);
        let f = self.ffn1.forward(tape, f).relu();
        let f = self.ffn2.forward(tape, f);
        x1.add(f) // residual 2
    }

    /// SGD-update every parameter (both LayerNorms, attention, both FFN linears).
    pub fn sgd_step(&mut self, grads: &[TensorND], lr: f32) {
        self.ln1.sgd_step(grads, lr);
        self.attn.sgd_step(grads, lr);
        self.ln2.sgd_step(grads, lr);
        self.ffn1.sgd_step(grads, lr);
        self.ffn2.sgd_step(grads, lr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mse<'t>(pred: NdVar<'t>, target: NdVar<'t>) -> NdVar<'t> {
        let diff = pred.sub(target);
        diff.mul(diff).sum()
    }

    /// Gradient of the loss w.r.t. the layer **input** matches finite
    /// differences (exercises reshape + matmul + bias-broadcast backward).
    #[test]
    fn nd_linear_input_gradient_check() {
        let mut rng = PcgEngine::new(1);
        let mut lin = NdLinear::new(3, 2, &mut rng);
        // A 3-D input (batch=2, seq=2, in=3) to exercise the flatten/restore.
        let x: Vec<f32> = (0..12).map(|i| (i as f32 * 0.2 - 1.0).sin()).collect();
        let target: Vec<f32> = (0..8).map(|i| (i as f32 * 0.1).cos()).collect();

        let loss_of = |xd: &[f32], lin: &mut NdLinear| -> f32 {
            let t = NdTape::new();
            let xv = t.input(TensorND::new(xd.to_vec(), vec![2, 2, 3]));
            let tv = t.input(TensorND::new(target.clone(), vec![2, 2, 2]));
            let y = lin.forward(&t, xv);
            t.value(mse(y, tv)).data[0]
        };

        let t = NdTape::new();
        let xv = t.input(TensorND::new(x.clone(), vec![2, 2, 3]));
        let tv = t.input(TensorND::new(target.clone(), vec![2, 2, 2]));
        let y = lin.forward(&t, xv);
        let grads = t.backward(mse(y, tv));
        let gx = grads[xv.idx()].clone();

        let eps = 1e-3f32;
        for k in 0..x.len()
        {
            let mut up = x.clone();
            let mut dn = x.clone();
            up[k] += eps;
            dn[k] -= eps;
            let num = (loss_of(&up, &mut lin) - loss_of(&dn, &mut lin)) / (2.0 * eps);
            assert!(
                (num - gx.data[k]).abs() < 2e-2,
                "input grad {k}: numeric {num}, analytic {}",
                gx.data[k]
            );
        }
    }

    /// A real training loop on a 2-layer N-D MLP: the regression loss strictly
    /// decreases — proving the weight gradients and the SGD step are correct.
    #[test]
    fn nd_mlp_trains_and_loss_decreases() {
        let mut rng = PcgEngine::new(7);
        let mut l1 = NdLinear::new(4, 8, &mut rng);
        let mut l2 = NdLinear::new(8, 3, &mut rng);

        // Deterministic synthetic data X (6×4) and a fixed teacher target.
        let xs: Vec<f32> = (0..24).map(|i| (i as f32 * 0.17 - 1.0).sin()).collect();
        let ts: Vec<f32> = (0..18).map(|i| (i as f32 * 0.09).cos() * 0.5).collect();

        let mut first = f32::NAN;
        let mut last = f32::NAN;
        for step in 0..40
        {
            let t = NdTape::new();
            let x = t.input(TensorND::new(xs.clone(), vec![6, 4]));
            let target = t.input(TensorND::new(ts.clone(), vec![6, 3]));
            let h = l1.forward(&t, x).relu();
            let y = l2.forward(&t, h);
            let loss_v = mse(y, target);
            let loss = t.value(loss_v).data[0];
            if step == 0
            {
                first = loss;
            }
            last = loss;
            let grads = t.backward(loss_v);
            l1.sgd_step(&grads, 0.02);
            l2.sgd_step(&grads, 0.02);
        }
        assert!(
            last < first * 0.7,
            "MLP did not learn: first {first}, last {last}"
        );
    }

    /// The full multi-head attention **layer** (q/k/v/o projections + the
    /// attention block) is correct: its gradient w.r.t. the input matches finite
    /// differences, and the output keeps the `(seq, d_model)` shape.
    #[test]
    fn nd_attention_layer_gradient_check() {
        let (d_model, n_heads, seq) = (8usize, 2, 3);
        let mut rng = PcgEngine::new(2);
        let mut attn = NdMultiHeadAttention::new(d_model, n_heads, &mut rng);
        let x: Vec<f32> = (0..seq * d_model)
            .map(|i| (i as f32 * 0.13 - 0.4).sin())
            .collect();

        let loss_of = |xd: &[f32], attn: &mut NdMultiHeadAttention| -> f32 {
            let t = NdTape::new();
            let xv = t.input(TensorND::new(xd.to_vec(), vec![seq, d_model]));
            let out = attn.forward(&t, xv);
            t.value(out.sum()).data[0]
        };

        let t = NdTape::new();
        let xv = t.input(TensorND::new(x.clone(), vec![seq, d_model]));
        let out = attn.forward(&t, xv);
        assert_eq!(out.shape(), vec![seq, d_model]);
        let grads = t.backward(out.sum());
        let gx = grads[xv.idx()].clone();

        let eps = 1e-3f32;
        for k in 0..x.len()
        {
            let mut up = x.clone();
            let mut dn = x.clone();
            up[k] += eps;
            dn[k] -= eps;
            let num = (loss_of(&up, &mut attn) - loss_of(&dn, &mut attn)) / (2.0 * eps);
            assert!(
                (num - gx.data[k]).abs() < 2e-2,
                "attention layer grad {k}: numeric {num}, analytic {}",
                gx.data[k]
            );
        }
    }

    /// A **full Pre-LN transformer block** (LayerNorm + attention + residual +
    /// FFN + residual) trains end to end on the N-D tape: the regression loss
    /// drops well below its initial value. The milestone: "here is the
    /// transformer block, and it learns".
    #[test]
    fn nd_transformer_block_trains() {
        let (d_model, n_heads, d_ff, seq) = (8usize, 2, 16, 4);
        let mut rng = PcgEngine::new(11);
        let mut block = NdTransformerBlock::new(d_model, n_heads, d_ff, &mut rng);

        let xs: Vec<f32> = (0..seq * d_model)
            .map(|i| (i as f32 * 0.13 - 0.5).sin())
            .collect();
        let ts: Vec<f32> = (0..seq * d_model)
            .map(|i| (i as f32 * 0.07).cos() * 0.3)
            .collect();

        let mut first = f32::NAN;
        let mut last = f32::NAN;
        for step in 0..80
        {
            let t = NdTape::new();
            let x = t.input(TensorND::new(xs.clone(), vec![seq, d_model]));
            let target = t.input(TensorND::new(ts.clone(), vec![seq, d_model]));
            let y = block.forward(&t, x);
            let loss_v = mse(y, target);
            let loss = t.value(loss_v).data[0];
            if step == 0
            {
                first = loss;
            }
            last = loss;
            let grads = t.backward(loss_v);
            block.sgd_step(&grads, 0.01);
        }
        assert!(
            last < first * 0.7,
            "transformer block did not learn: first {first}, last {last}"
        );
    }
}
