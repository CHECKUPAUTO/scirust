//! Property-based gradient checks: for a range of ops, verify the analytic
//! (reverse-mode) gradient matches central finite differences across many
//! random shapes and seeds — not just one fixed input. Uses only the public
//! `Tape`/`Var` API and the crate's own deterministic PRNG.

use scirust_core::autodiff::reverse::{Tape, Tensor, Var, concat_rows};
use scirust_core::nn::loss::{CrossEntropyLoss, Loss, MseLoss, PoissonNllLoss};
use scirust_core::nn::rng::PcgEngine;

/// Seeds shared by every sweep in this file.
const SEEDS: [u64; 5] = [1, 7, 42, 123, 2024];

/// Build a scalar loss from one input `Var` and check `d loss / d input`
/// against central differences over every input element.
fn check_grad<F>(op: &str, seed: u64, rows: usize, cols: usize, build: F)
where
    F: for<'t> Fn(Var<'t>) -> Var<'t>,
{
    let n = rows * cols;
    let mut rng = PcgEngine::new(seed);
    // Inputs in [-1, 1] keep exp/softmax well-conditioned.
    let x0: Vec<f32> = (0..n).map(|_| rng.float() * 2.0 - 1.0).collect();

    // Analytic gradient.
    let tape = Tape::new();
    let xv = tape.input(Tensor::from_vec(x0.clone(), rows, cols));
    let loss = build(xv);
    loss.backward();
    let analytic = tape.grad(xv.idx()).data;
    assert_eq!(analytic.len(), n, "{op}: grad shape");

    // Scalar loss at a perturbed input.
    let loss_at = |x: &[f32]| -> f32 {
        let tape = Tape::new();
        let xv = tape.input(Tensor::from_vec(x.to_vec(), rows, cols));
        let l = build(xv);
        tape.value(l.idx()).data[0]
    };

    let eps = 1e-3f32;
    for i in 0..n
    {
        let mut xp = x0.clone();
        xp[i] += eps;
        let mut xm = x0.clone();
        xm[i] -= eps;
        let num = (loss_at(&xp) - loss_at(&xm)) / (2.0 * eps);
        let a = analytic[i];
        let tol = 3e-2 * (1.0 + a.abs().max(num.abs()));
        assert!(
            (a - num).abs() < tol,
            "{op} seed={seed} {rows}x{cols} elem {i}: analytic {a} vs finite-diff {num}"
        );
    }
}

/// Run `build` over a spread of seeds and (non-square, thin, wide) shapes.
fn sweep<F>(op: &str, build: F)
where
    F: for<'t> Fn(Var<'t>) -> Var<'t> + Copy,
{
    for &seed in &SEEDS
    {
        for &(r, c) in &[(1usize, 4usize), (3, 3), (2, 5), (4, 1), (5, 2)]
        {
            check_grad(op, seed, r, c, build);
        }
    }
}

/// Multi-input generalization of [`check_grad`] for the structured ops
/// (matmul/bmm2d/layer_norm/conv2d/losses/…): build a graph from
/// `shapes.len()` input `Var`s, then check `d loss / d input[v]` for every
/// input and every element against central finite differences.
///
/// The op output is weighted elementwise by a fixed random tensor before the
/// final `sum()`, so structurally-uniform gradients (e.g. `sum(A·B)` making
/// `dL/dA` identical across rows) cannot mask an indexing/transpose bug in the
/// backward.
fn check_grad_multi<F>(op: &str, seed: u64, shapes: &[(usize, usize)], build: F)
where
    F: for<'t> Fn(&'t Tape, &[Var<'t>]) -> Var<'t>,
{
    let mut rng = PcgEngine::new(seed);
    let xs: Vec<Vec<f32>> = shapes
        .iter()
        .map(|&(r, c)| (0..r * c).map(|_| rng.float() * 2.0 - 1.0).collect())
        .collect();

    // Full evaluation: loss = sum(build(inputs) ⊙ W) with W deterministic in
    // [0.5, 1.5), plus the analytic gradients of every input.
    let run = |xs: &[Vec<f32>]| -> (f32, Vec<Vec<f32>>) {
        let tape = Tape::new();
        let vars: Vec<Var<'_>> = shapes
            .iter()
            .zip(xs)
            .map(|(&(r, c), x)| tape.input(Tensor::from_vec(x.clone(), r, c)))
            .collect();
        let out = build(&tape, &vars);
        let (orows, ocols) = out.shape();
        let mut wrng = PcgEngine::new(seed ^ 0x5EED_CAFE);
        let w: Vec<f32> = (0..orows * ocols).map(|_| wrng.float() + 0.5).collect();
        let wv = tape.input(Tensor::from_vec(w, orows, ocols));
        let loss = out.hadamard(wv).sum();
        loss.backward();
        let grads = vars.iter().map(|v| tape.grad(v.idx()).data).collect();
        (tape.value(loss.idx()).data[0], grads)
    };

    let (_, analytic) = run(&xs);
    let eps = 1e-3f32;
    for (vi, &(r, c)) in shapes.iter().enumerate()
    {
        assert_eq!(analytic[vi].len(), r * c, "{op}: grad shape of input {vi}");
        for i in 0..r * c
        {
            let mut xp = xs.clone();
            xp[vi][i] += eps;
            let mut xm = xs.clone();
            xm[vi][i] -= eps;
            let num = (run(&xp).0 - run(&xm).0) / (2.0 * eps);
            let a = analytic[vi][i];
            let tol = 3e-2 * (1.0 + a.abs().max(num.abs()));
            assert!(
                (a - num).abs() < tol,
                "{op} seed={seed} input {vi} ({r}x{c}) elem {i}: analytic {a} vs finite-diff {num}"
            );
        }
    }
}

#[test]
fn grad_relu() {
    sweep("relu", |x| x.relu().sum());
}

#[test]
fn grad_tanh() {
    sweep("tanh", |x| x.tanh().sum());
}

#[test]
fn grad_sigmoid() {
    sweep("sigmoid", |x| x.sigmoid().sum());
}

#[test]
fn grad_exp() {
    sweep("exp", |x| x.exp().sum());
}

#[test]
fn grad_sin() {
    sweep("sin", |x| x.sin().sum());
}

#[test]
fn grad_scale() {
    sweep("scale", |x| x.scale(0.7).sum());
}

#[test]
fn grad_square_via_hadamard() {
    // Same operand used twice — exercises Mul(a, a) accumulation.
    sweep("square", |x| x.hadamard(x).sum());
}

#[test]
fn grad_softmax_weighted() {
    // Plain sum(softmax) is constant (grad 0); square it for a live gradient.
    sweep("softmax", |x| {
        let s = x.softmax(1);
        s.hadamard(s).sum()
    });
}

#[test]
fn grad_add_self() {
    // x + x = 2x, grad 2 — exercises Add with a shared operand.
    sweep("add_self", |x| x.add(x).sum());
}

#[test]
fn grad_sub_scaled() {
    // x - 0.5·x, exercises Sub with a shared operand.
    sweep("sub_scaled", |x| x.sub(x.scale(0.5)).sum());
}

// ================================================================== //
//  Structured ops: matmul family                                     //
// ================================================================== //

#[test]
fn grad_matmul() {
    // C = A·B — checks both dL/dA and dL/dB through the transposed backward
    // GEMMs (g·Bᵀ and Aᵀ·g).
    fn build<'t>(_tape: &'t Tape, v: &[Var<'t>]) -> Var<'t> {
        v[0].matmul(v[1])
    }
    for &seed in &SEEDS
    {
        for shapes in &[
            [(2usize, 3usize), (3usize, 4usize)],
            [(1, 4), (4, 3)],
            [(3, 2), (2, 5)],
            [(4, 4), (4, 1)],
        ]
        {
            check_grad_multi("matmul", seed, shapes, build);
        }
    }
}

#[test]
fn grad_matmul_bt() {
    // C = A·Bᵀ with A (m×k), B (n×k) — B is read transposed via strides, and
    // the backward must produce dB in B's *stored* (n×k) layout.
    fn build<'t>(_tape: &'t Tape, v: &[Var<'t>]) -> Var<'t> {
        v[0].matmul_bt(v[1])
    }
    for &seed in &SEEDS
    {
        for shapes in &[
            [(2usize, 3usize), (4usize, 3usize)],
            [(1, 5), (2, 5)],
            [(3, 2), (3, 2)],
        ]
        {
            check_grad_multi("matmul_bt", seed, shapes, build);
        }
    }
}

#[test]
fn grad_bmm2d_batch2_transpose_b() {
    // Batched A[i]·B[i]ᵀ, batch = 2: A is (2m × k), B is (2n × k), out (2m × n).
    fn build<'t>(_tape: &'t Tape, v: &[Var<'t>]) -> Var<'t> {
        v[0].bmm2d(v[1], 2, true)
    }
    for &seed in &SEEDS
    {
        for shapes in &[
            [(4usize, 3usize), (8usize, 3usize)], // m=2 k=3 n=4
            [(2, 2), (6, 2)],                     // m=1 k=2 n=3
            [(6, 4), (4, 4)],                     // m=3 k=4 n=2
        ]
        {
            check_grad_multi("bmm2d_bt", seed, shapes, build);
        }
    }
}

#[test]
fn grad_bmm2d_batch2_no_transpose() {
    // Batched A[i]·B[i], batch = 2: A is (2m × k), B is (2k × n), out (2m × n).
    fn build<'t>(_tape: &'t Tape, v: &[Var<'t>]) -> Var<'t> {
        v[0].bmm2d(v[1], 2, false)
    }
    for &seed in &SEEDS
    {
        for shapes in &[
            [(4usize, 3usize), (6usize, 4usize)], // m=2 k=3 n=4
            [(6, 2), (4, 2)],                     // m=3 k=2 n=2
            [(2, 4), (8, 1)],                     // m=1 k=4 n=1
        ]
        {
            check_grad_multi("bmm2d", seed, shapes, build);
        }
    }
}

// ================================================================== //
//  Normalization & cross-axis softmax                                //
// ================================================================== //

#[test]
fn grad_layer_norm_input_gamma_beta() {
    // layer_norm(x, γ, β): all three gradients — dL/dx couples every element
    // of a row through the mean/variance, dL/dγ and dL/dβ are column sums.
    fn build<'t>(_tape: &'t Tape, v: &[Var<'t>]) -> Var<'t> {
        v[0].layer_norm(v[1], v[2], 1e-5)
    }
    for &seed in &SEEDS
    {
        for &(r, c) in &[(3usize, 5usize), (2, 4), (4, 3), (1, 6)]
        {
            check_grad_multi("layer_norm", seed, &[(r, c), (1, c), (1, c)], build);
        }
    }
}

#[test]
fn grad_softmax_axis0() {
    // Column-wise softmax (axis 0) — axis 1 is covered by `grad_softmax_weighted`.
    // Plain sum is constant per column, so square for a live gradient.
    sweep("softmax_axis0", |x| {
        let s = x.softmax(0);
        s.hadamard(s).sum()
    });
}

// ================================================================== //
//  Gradient-routing ops: causal mask, slices, concat                 //
// ================================================================== //

#[test]
fn grad_causal_mask_softmax() {
    // causal_mask → softmax, the attention composition. Masked scores hold
    // the constant -1e9, so their probability — and both the analytic and the
    // finite-difference gradient — must be exactly 0; unmasked entries get the
    // usual softmax Jacobian. Shapes are (batch·seq × seq) stacked blocks.
    for &seed in &SEEDS
    {
        for &(batch, seq) in &[(1usize, 3usize), (2, 3), (1, 4)]
        {
            check_grad("causal_mask_softmax", seed, batch * seq, seq, move |x| {
                let s = x.causal_mask(seq).softmax(1);
                s.hadamard(s).sum()
            });
        }
    }
}

#[test]
fn grad_slice_rows() {
    // Rows outside the slice must get exactly zero gradient; rows inside get
    // the weighted-sum gradient routed back to the right offset.
    fn build<'t>(_tape: &'t Tape, v: &[Var<'t>]) -> Var<'t> {
        v[0].slice_rows(1, 2)
    }
    for &seed in &SEEDS
    {
        for &(r, c) in &[(4usize, 3usize), (5, 2), (3, 4)]
        {
            check_grad_multi("slice_rows", seed, &[(r, c)], build);
        }
    }
}

#[test]
fn grad_slice_cols() {
    fn build<'t>(_tape: &'t Tape, v: &[Var<'t>]) -> Var<'t> {
        v[0].slice_cols(1, 2)
    }
    for &seed in &SEEDS
    {
        for &(r, c) in &[(3usize, 4usize), (2, 5), (4, 3)]
        {
            check_grad_multi("slice_cols", seed, &[(r, c)], build);
        }
    }
}

#[test]
fn grad_concat_rows_two_and_four() {
    // concat_rows routes the output gradient back to each piece's row range.
    // Four pieces exercises the recursive chunks-of-3 grouping path.
    fn build<'t>(tape: &'t Tape, v: &[Var<'t>]) -> Var<'t> {
        concat_rows(tape, v)
    }
    for &seed in &SEEDS
    {
        check_grad_multi("concat_rows_2", seed, &[(2, 3), (3, 3)], build);
        check_grad_multi(
            "concat_rows_4",
            seed,
            &[(1, 2), (2, 2), (1, 2), (3, 2)],
            build,
        );
    }
}

// ================================================================== //
//  Convolution (public Var API, same op the nn::Conv2d module drives)//
// ================================================================== //

#[test]
fn grad_conv2d_input_weight_bias() {
    // conv2d_forward: batch=2, in_c=2, 4×4, out_c=3, k=3, stride=1, pad=1 —
    // all three gradients (input via col2im, weight via g·colᵀ, bias via sums).
    fn build<'t>(_tape: &'t Tape, v: &[Var<'t>]) -> Var<'t> {
        v[0].conv2d_forward(v[1], Some(v[2]), 2, 2, 4, 4, 3, 3, 1, 1)
    }
    for &seed in &[1u64, 7, 42]
    {
        check_grad_multi("conv2d", seed, &[(2, 32), (3, 18), (1, 3)], build);
    }
}

#[test]
fn grad_conv2d_stride2_no_bias() {
    // Strided, unpadded, bias-free variant: batch=1, in_c=1, 5×5, out_c=2,
    // k=3, stride=2, pad=0 → 2×2 output per channel.
    fn build<'t>(_tape: &'t Tape, v: &[Var<'t>]) -> Var<'t> {
        v[0].conv2d_forward(v[1], None, 1, 1, 5, 5, 2, 3, 2, 0)
    }
    for &seed in &[1u64, 7, 42]
    {
        check_grad_multi("conv2d_s2", seed, &[(1, 25), (2, 9)], build);
    }
}

// ================================================================== //
//  Losses end-to-end through the Loss trait                          //
// ================================================================== //

#[test]
fn grad_mse_loss_end_to_end() {
    // MseLoss::forward(pred, target) — both gradients (dL/dtarget = -dL/dpred).
    fn build<'t>(tape: &'t Tape, v: &[Var<'t>]) -> Var<'t> {
        MseLoss::new().forward(tape, v[0], v[1])
    }
    for &seed in &SEEDS
    {
        for &(r, c) in &[(2usize, 3usize), (3, 4), (1, 5)]
        {
            check_grad_multi("mse_loss", seed, &[(r, c), (r, c)], build);
        }
    }
}

#[test]
fn grad_cross_entropy_loss_one_hot() {
    // CrossEntropyLoss on raw logits with a deterministic one-hot target
    // (class (2b+1) mod n_classes for row b). Only the logits are perturbed;
    // dL/dlogits must equal (softmax − onehot)/batch, here validated purely
    // numerically.
    fn build<'t>(tape: &'t Tape, v: &[Var<'t>]) -> Var<'t> {
        let (batch, n_classes) = v[0].shape();
        let mut onehot = vec![0.0f32; batch * n_classes];
        for b in 0..batch
        {
            onehot[b * n_classes + (2 * b + 1) % n_classes] = 1.0;
        }
        let target = tape.input(Tensor::from_vec(onehot, batch, n_classes));
        CrossEntropyLoss::new().forward(tape, v[0], target)
    }
    for &seed in &SEEDS
    {
        for &(batch, n_classes) in &[(3usize, 4usize), (2, 5), (4, 3), (1, 4)]
        {
            check_grad_multi("cross_entropy", seed, &[(batch, n_classes)], build);
        }
    }
}

#[test]
fn grad_poisson_nll_loss_log_input() {
    // PoissonNllLoss (log_input = true, mean): dL/dpred = (exp(pred) − k)/N,
    // with deterministic integer counts k = i mod 4 as the target.
    fn build<'t>(tape: &'t Tape, v: &[Var<'t>]) -> Var<'t> {
        let (r, c) = v[0].shape();
        let counts: Vec<f32> = (0..r * c).map(|i| (i % 4) as f32).collect();
        let target = tape.input(Tensor::from_vec(counts, r, c));
        PoissonNllLoss::new().forward(tape, v[0], target)
    }
    for &seed in &SEEDS
    {
        for &(r, c) in &[(2usize, 3usize), (3, 4), (1, 5)]
        {
            check_grad_multi("poisson_nll_log", seed, &[(r, c)], build);
        }
    }
}

#[test]
fn grad_poisson_nll_loss_direct_rate() {
    // log_input = false needs λ > 0: map the harness input x ∈ [-1, 1] to
    // λ = 0.4·x + 1.5 ∈ [1.1, 1.9] inside the graph, so the finite difference
    // still perturbs the *pre-transform* input and the chain rule is checked.
    fn build<'t>(tape: &'t Tape, v: &[Var<'t>]) -> Var<'t> {
        let (r, c) = v[0].shape();
        let shift = tape.input(Tensor::from_vec(vec![1.5f32; r * c], r, c));
        let lambda = v[0].scale(0.4).add(shift);
        let counts: Vec<f32> = (0..r * c).map(|i| (i % 3) as f32).collect();
        let target = tape.input(Tensor::from_vec(counts, r, c));
        PoissonNllLoss::with(false, false).forward(tape, lambda, target)
    }
    for &seed in &SEEDS
    {
        for &(r, c) in &[(2usize, 3usize), (3, 2)]
        {
            check_grad_multi("poisson_nll_rate", seed, &[(r, c)], build);
        }
    }
}
