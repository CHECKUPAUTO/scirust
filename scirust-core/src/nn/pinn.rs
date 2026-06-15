//! **Physics-Informed Neural Networks** (Raissi, Perdikaris & Karniadakis,
//! *J. Comp. Phys.* 2019) over the N-D autograd tape.
//!
//! A PINN trains a network `u_θ(x)` so that a **PDE residual** vanishes at
//! collocation points, plus boundary/initial conditions — the physics is baked
//! into the loss rather than learned from labelled solutions. Here we solve the
//! 1-D harmonic boundary-value problem
//!
//! ```text
//! u''(x) = −u(x)   on [0, L],   u(0) = u0,  u(L) = uL
//! ```
//!
//! whose closed-form solution (for `L = π/2, u0 = 0, uL = 1`) is `u(x) = sin x` —
//! giving an exact oracle. The interior derivative `u''` is taken by **central
//! finite differences in the input** (`u(x±h)` evaluated through the *same*
//! parameters in one forward graph), so the loss is differentiable w.r.t. the
//! parameters by ordinary reverse-mode autodiff and the whole thing trains
//! deterministically.

use crate::autodiff::nd::{NdTape, NdVar};
use crate::nn::nd_layers::NdLinear;
use crate::nn::nd_optim::{NdAdam, NdParam};
use crate::nn::rng::PcgEngine;
use crate::tensor::tensor_nd::TensorND;

/// A small smooth MLP `1 → H → H → 1` with logistic activations — `C∞`, so its
/// finite-difference derivatives are well behaved. Maps a `(batch, 1)` column of
/// inputs to a `(batch, 1)` column of outputs.
pub struct Pinn1D {
    l1: NdLinear,
    l2: NdLinear,
    l3: NdLinear,
}

impl Pinn1D {
    /// New network with seeded init and `hidden` units per layer.
    pub fn new(hidden: usize, rng: &mut PcgEngine) -> Self {
        Self {
            l1: NdLinear::new(1, hidden, rng),
            l2: NdLinear::new(hidden, hidden, rng),
            l3: NdLinear::new(hidden, 1, rng),
        }
    }

    /// Forward `u_θ(x)` over a `(batch, 1)` input column.
    pub fn forward<'t>(&mut self, tape: &'t NdTape, x: NdVar<'t>) -> NdVar<'t> {
        let h = self.l1.forward(tape, x).sigmoid();
        let h = self.l2.forward(tape, h).sigmoid();
        self.l3.forward(tape, h)
    }

    /// Trainable parameters in a fixed order.
    pub fn parameters(&mut self) -> Vec<NdParam<'_>> {
        let mut p = self.l1.parameters();
        p.extend(self.l2.parameters());
        p.extend(self.l3.parameters());
        p
    }

    /// Evaluate `u_θ(x)` at a single point on a throwaway tape (no grad).
    pub fn eval(&mut self, x: f32) -> f32 {
        let tape = NdTape::new();
        let xv = tape.input(TensorND::new(vec![x], vec![1, 1]));
        tape.value(self.forward(&tape, xv)).data[0]
    }
}

/// Outcome of [`solve_harmonic`].
pub struct PinnSolution {
    /// The trained network.
    pub net: Pinn1D,
    /// Total loss (PDE residual + boundary) at the first step.
    pub first_loss: f32,
    /// Total loss at the last step.
    pub last_loss: f32,
    /// Max `|u_θ(x) − sin x|` over a uniform test grid (the analytic oracle).
    pub max_error: f32,
}

/// Train a PINN to solve `u'' = −u` on `[0, π/2]` with `u(0) = 0, u(π/2) = 1`
/// (solution `sin x`). `hidden`/`steps`/`lr` configure the network and Adam;
/// deterministic in `seed`. The PDE residual uses central differences with step
/// `h`; boundary conditions are weighted by `bc_weight`.
pub fn solve_harmonic(hidden: usize, steps: usize, lr: f32, seed: u64) -> PinnSolution {
    let l = std::f32::consts::FRAC_PI_2;
    let (u0, ul) = (0.0f32, 1.0f32);
    let h = 1e-2f32;
    let k = 24usize; // interior collocation points

    // Collocation points strictly inside (0, L).
    let xs: Vec<f32> = (0..k)
        .map(|i| l * (i as f32 + 1.0) / (k as f32 + 1.0))
        .collect();
    // One forward batch holds center, +h, −h evaluations, then the two BC points.
    let mut batch = Vec::with_capacity(3 * k + 2);
    batch.extend(xs.iter().copied()); // center   [0, k)
    batch.extend(xs.iter().map(|&x| x + h)); // plus [k, 2k)
    batch.extend(xs.iter().map(|&x| x - h)); // minus [2k, 3k)
    batch.push(0.0); // BC x=0   index 3k
    batch.push(l); // BC x=L     index 3k+1
    let m = batch.len();

    let center_idx: Vec<usize> = (0..k).collect();
    let plus_idx: Vec<usize> = (k..2 * k).collect();
    let minus_idx: Vec<usize> = (2 * k..3 * k).collect();
    let bc_idx: Vec<usize> = vec![3 * k, 3 * k + 1];

    let mut rng = PcgEngine::new(seed);
    let mut net = Pinn1D::new(hidden, &mut rng);
    let mut opt = NdAdam::with_lr(lr);
    let inv_h2 = 1.0 / (h * h);

    let (mut first, mut last) = (f32::NAN, f32::NAN);
    for step in 0..steps
    {
        let tape = NdTape::new();
        let xv = tape.input(TensorND::new(batch.clone(), vec![m, 1]));
        let u = net.forward(&tape, xv); // (m, 1)

        let center = u.gather(&center_idx);
        let plus = u.gather(&plus_idx);
        let minus = u.gather(&minus_idx);
        let two = tape.input(TensorND::new(vec![2.0f32], vec![1, 1]));
        let inv = tape.input(TensorND::new(vec![inv_h2], vec![1, 1]));
        // u'' ≈ (u(x+h) − 2u(x) + u(x−h)) / h²
        let u_xx = plus.add(minus).sub(center.mul(two)).mul(inv);
        // residual r = u'' + u
        let r = u_xx.add(center);
        let physics = r.mul(r).sum();

        // Boundary: (u(0) − u0)² + (u(L) − uL)².
        let bc = u.gather(&bc_idx);
        let bc_target = tape.input(TensorND::new(vec![u0, ul], vec![2, 1]));
        let bc_diff = bc.sub(bc_target);
        let bc_w = tape.input(TensorND::new(vec![10.0f32], vec![1, 1]));
        let bc_loss = bc_diff.mul(bc_diff).sum().mul(bc_w);

        let loss = physics.add(bc_loss);
        let lval = tape.value(loss).data[0];
        if step == 0
        {
            first = lval;
        }
        last = lval;
        let grads = tape.backward(loss);
        opt.step(&mut net.parameters(), &grads);
    }

    // Verify against the analytic solution sin(x) on a uniform grid.
    let mut max_error = 0.0f32;
    for i in 0..=20
    {
        let x = l * i as f32 / 20.0;
        let err = (net.eval(x) - x.sin()).abs();
        if err > max_error
        {
            max_error = err;
        }
    }

    PinnSolution {
        net,
        first_loss: first,
        last_loss: last,
        max_error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The PINN drives the PDE+boundary loss down and recovers the analytic
    /// solution `sin x` to within a small tolerance — physics in the loss, no
    /// labelled solution used.
    #[test]
    fn pinn_solves_harmonic_bvp() {
        let sol = solve_harmonic(16, 4000, 0.01, 1);
        assert!(
            sol.last_loss < sol.first_loss * 0.05,
            "PINN loss did not drop: {} -> {}",
            sol.first_loss,
            sol.last_loss
        );
        assert!(
            sol.max_error < 0.05,
            "PINN solution off from sin(x): max error {}",
            sol.max_error
        );
    }

    /// Deterministic: same seed ⇒ bit-identical final loss and solution.
    #[test]
    fn pinn_is_deterministic() {
        let a = solve_harmonic(12, 300, 0.01, 7);
        let b = solve_harmonic(12, 300, 0.01, 7);
        assert_eq!(a.last_loss.to_bits(), b.last_loss.to_bits());
        assert_eq!(a.max_error.to_bits(), b.max_error.to_bits());
    }
}
