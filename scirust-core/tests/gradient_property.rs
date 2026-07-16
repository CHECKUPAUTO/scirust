//! Property-based gradient checks: for a range of ops, verify the analytic
//! (reverse-mode) gradient matches central finite differences across many
//! random shapes and seeds — not just one fixed input. Uses only the public
//! `Tape`/`Var` API and the crate's own deterministic PRNG.

use scirust_core::autodiff::reverse::{Tape, Tensor, Var};
use scirust_core::nn::rng::PcgEngine;

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
    for &seed in &[1u64, 7, 42, 123, 2024]
    {
        for &(r, c) in &[(1usize, 4usize), (3, 3), (2, 5), (4, 1), (5, 2)]
        {
            check_grad(op, seed, r, c, build);
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
