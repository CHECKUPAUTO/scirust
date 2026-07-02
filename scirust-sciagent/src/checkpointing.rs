//! Gradient (activation) checkpointing — the technique, validated on
//! `scirust-core`'s tape.
//!
//! Reverse-mode autodiff stores every intermediate activation so the backward
//! pass can use it. For a deep model at long sequence length that is the
//! dominant memory cost (see [`crate::planning`]). **Gradient checkpointing**
//! trades compute for memory: during the forward pass, keep only per-segment
//! *boundary* activations; during backward, recompute each segment's forward on
//! a fresh tape and backprop through just that segment. Activation memory drops
//! from `O(depth)` to `O(one segment)`.
//!
//! The one non-obvious piece is backpropagating an **upstream gradient**
//! `∂L/∂output` into a segment when the tape only offers scalar `backward`.
//! The trick ([`seed_upstream`]): form the surrogate scalar
//! `s = Σ(output ⊙ ḡ)` with `ḡ = ∂L/∂output` treated as a constant, then
//! `backward(s)`. Because `∂s/∂output = ḡ`, the chain rule makes every
//! resulting parameter/input gradient exactly `∂L/∂·`. This module proves that
//! recomputed, segment-wise gradients are numerically identical to a single
//! end-to-end tape — the correctness contract a checkpointed training loop
//! (the memory enabler for 350M on a Thor) must satisfy.

use scirust_core::autodiff::reverse::Var;

/// Surrogate scalar whose scalar-`backward` seeds the tape with the upstream
/// gradient `g_output` (`∂L/∂output`) at `output`. `g_output` must be a tape
/// input carrying the upstream gradient *values* and is treated as a constant
/// (its own gradient is ignored).
pub fn seed_upstream<'t>(output: Var<'t>, g_output: Var<'t>) -> Var<'t> {
    output.hadamard(g_output).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_core::autodiff::reverse::{Tape, Tensor};
    use scirust_core::nn::init::{KaimingNormal, Zeros};
    use scirust_core::nn::linear::Linear;
    use scirust_core::nn::module::Module;
    use scirust_core::nn::rng::PcgEngine;

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    // A two-layer Linear stack. Checkpointed segment-recompute backward must
    // produce the SAME parameter gradients as a monolithic end-to-end tape.
    #[test]
    fn checkpointed_gradients_equal_full_tape_gradients() {
        let (din, dhid, dout, n) = (5usize, 7usize, 4usize, 3usize);
        let mut rng = PcgEngine::new(42);
        let l0 = Linear::new(din, dhid, &KaimingNormal, &Zeros, &mut rng);
        let l1 = Linear::new(dhid, dout, &KaimingNormal, &Zeros, &mut rng);
        let x_data: Vec<f32> = (0..n * din).map(|i| (i as f32 * 0.11).sin()).collect();
        // Fixed output weighting so the loss is L = Σ(out ⊙ wl); then ∂L/∂out = wl.
        let wl: Vec<f32> = (0..n * dout).map(|i| (i as f32 * 0.07).cos()).collect();

        // --- Reference: one tape, end to end. ---
        let (ref_g0, ref_g1) = {
            let mut l0 = l0.clone();
            let mut l1 = l1.clone();
            let tape = Tape::new();
            let x = tape.input(Tensor::from_vec(x_data.clone(), n, din));
            let h = l0.forward(&tape, x);
            let o = l1.forward(&tape, h);
            let wlv = tape.input(Tensor::from_vec(wl.clone(), n, dout));
            let loss = o.hadamard(wlv).sum();
            tape.backward(loss.idx());
            let g0 = tape.grad(l0.parameter_indices()[0]).data;
            let g1 = tape.grad(l1.parameter_indices()[0]).data;
            (g0, g1)
        };

        // --- Checkpointed: boundaries + per-segment recompute. ---
        // Forward for boundary values (a real impl uses a no-record forward;
        // here a scratch tape we discard gives the same values).
        let h_val = {
            let mut l0 = l0.clone();
            let tape = Tape::new();
            let x = tape.input(Tensor::from_vec(x_data.clone(), n, din));
            let h = l0.forward(&tape, x);
            tape.value(h.idx())
        };

        // Segment L1: upstream ∂L/∂o = wl. Recompute L1 from h, seed, backprop.
        let (g1, g_h) = {
            let mut l1 = l1.clone();
            let tape = Tape::new();
            let h_in = tape.input(h_val.clone());
            let o = l1.forward(&tape, h_in);
            let g_o = tape.input(Tensor::from_vec(wl.clone(), n, dout));
            let s = seed_upstream(o, g_o);
            tape.backward(s.idx());
            let g1 = tape.grad(l1.parameter_indices()[0]).data;
            let g_h = tape.grad(h_in.idx());
            (g1, g_h)
        };

        // Segment L0: upstream is ∂L/∂h from the previous segment.
        let g0 = {
            let mut l0 = l0.clone();
            let tape = Tape::new();
            let x = tape.input(Tensor::from_vec(x_data.clone(), n, din));
            let h = l0.forward(&tape, x);
            let g_h_in = tape.input(g_h.clone());
            let s = seed_upstream(h, g_h_in);
            tape.backward(s.idx());
            tape.grad(l0.parameter_indices()[0]).data
        };

        assert!(
            max_abs_diff(&g1, &ref_g1) < 1e-5,
            "L1 checkpointed grad must match full-tape grad"
        );
        assert!(
            max_abs_diff(&g0, &ref_g0) < 1e-5,
            "L0 checkpointed grad must match full-tape grad"
        );
    }

    // The surrogate seeds exactly the upstream gradient: with ḡ = ones, the
    // input gradient equals the plain sum's gradient.
    #[test]
    fn seed_upstream_with_ones_equals_sum_backward() {
        let n = 4usize;
        let data: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();

        let direct = {
            let tape = Tape::new();
            let x = tape.input(Tensor::from_vec(data.clone(), 1, n));
            let y = x.hadamard(x); // some op
            let loss = y.sum();
            tape.backward(loss.idx());
            tape.grad(x.idx()).data
        };
        let via_surrogate = {
            let tape = Tape::new();
            let x = tape.input(Tensor::from_vec(data.clone(), 1, n));
            let y = x.hadamard(x);
            let ones = tape.input(Tensor::from_vec(vec![1.0; n], 1, n));
            let s = seed_upstream(y, ones);
            tape.backward(s.idx());
            tape.grad(x.idx()).data
        };
        assert_eq!(direct, via_surrogate);
    }
}
