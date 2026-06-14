//! **Neural ODEs** (Chen et al., NeurIPS 2018): a continuous-depth model
//! `dy/dt = f_θ(y)` integrated by a numerical solver, differentiated by
//! **backprop through the solver** on the N-D tape ([`crate::autodiff::nd`]).
//!
//! Where the rest of the workspace keeps solvers and autodiff separate, this
//! fuses them: the RK4 steps are ordinary tape ops, so the *same* reverse-mode
//! pass that trains a transformer also yields `∂loss/∂θ` and `∂loss/∂y₀` through
//! the integrator. The integration is deterministic (fixed step, fixed order).
//!
//! `f_θ` here is a small 2-layer ReLU MLP `(dim → hidden → dim)`; the dynamics
//! are autonomous (`f` does not depend on `t`).

use crate::autodiff::nd::{NdTape, NdVar};
use crate::nn::nd_optim::NdParam;
use crate::nn::rng::PcgEngine;
use crate::tensor::tensor_nd::TensorND;

/// Fixed-step classical **RK4** integration of `dy/dt = f(y)` on the tape, from
/// the current state for `steps` steps of size `dt`. `f` and all the
/// combinations are tape ops, so the result is differentiable w.r.t. `y0` and
/// anything `f` closes over.
pub fn rk4_integrate<'t, F>(
    tape: &'t NdTape,
    f: F,
    y0: NdVar<'t>,
    steps: usize,
    dt: f32,
) -> NdVar<'t>
where
    F: Fn(NdVar<'t>) -> NdVar<'t>,
{
    // Scalar constants live on the tape (their gradient is simply ignored).
    let sc = |c: f32| tape.input(TensorND::new(vec![c], vec![1]));
    let mut y = y0;
    for _ in 0..steps
    {
        let k1 = f(y);
        let k2 = f(y.add(k1.mul(sc(dt * 0.5))));
        let k3 = f(y.add(k2.mul(sc(dt * 0.5))));
        let k4 = f(y.add(k3.mul(sc(dt))));
        // y += dt/6 · (k1 + 2k2 + 2k3 + k4)
        let sum = k1.add(k2.mul(sc(2.0))).add(k3.mul(sc(2.0))).add(k4);
        y = y.add(sum.mul(sc(dt / 6.0)));
    }
    y
}

/// A neural ODE whose dynamics `f_θ(y) = W₂·relu(W₁·y + b₁) + b₂` map a state of
/// width `dim` to its time-derivative. Trainable on the N-D tape.
pub struct NeuralOde {
    w1: TensorND, // (dim, hidden)
    b1: TensorND, // (1, hidden)
    w2: TensorND, // (hidden, dim)
    b2: TensorND, // (1, dim)
    idx: [Option<usize>; 4],
}

impl NeuralOde {
    /// New dynamics MLP with seeded init (`W ~ U(-s, s)`, `s = 1/√fan_in`).
    pub fn new(dim: usize, hidden: usize, rng: &mut PcgEngine) -> Self {
        let init = |fan_in: usize, n: usize, rng: &mut PcgEngine| {
            let s = (1.0 / fan_in as f32).sqrt();
            (0..n).map(|_| rng.float_signed() * s).collect::<Vec<f32>>()
        };
        Self {
            w1: TensorND::new(init(dim, dim * hidden, rng), vec![dim, hidden]),
            b1: TensorND::zeros(&[1, hidden]),
            w2: TensorND::new(init(hidden, hidden * dim, rng), vec![hidden, dim]),
            b2: TensorND::zeros(&[1, dim]),
            idx: [None; 4],
        }
    }

    /// Integrate `y0` (shape `(1, dim)`) for `steps` RK4 steps of size `dt`,
    /// returning the final state `(1, dim)`. Records the parameter nodes so
    /// [`Self::parameters`] can read their gradients after `backward`.
    pub fn integrate<'t>(
        &mut self,
        tape: &'t NdTape,
        y0: NdVar<'t>,
        steps: usize,
        dt: f32,
    ) -> NdVar<'t> {
        // Upload each parameter **once** so its gradient accumulates across all
        // RK4 evaluations of f (4 per step) — not once per use.
        let w1 = tape.input(self.w1.clone());
        let b1 = tape.input(self.b1.clone());
        let w2 = tape.input(self.w2.clone());
        let b2 = tape.input(self.b2.clone());
        self.idx = [
            Some(w1.idx()),
            Some(b1.idx()),
            Some(w2.idx()),
            Some(b2.idx()),
        ];

        let f = move |y: NdVar<'t>| y.matmul(w1).add(b1).relu().matmul(w2).add(b2);
        rk4_integrate(tape, f, y0, steps, dt)
    }

    /// SGD-update every parameter from a `backward` result.
    pub fn sgd_step(&mut self, grads: &[TensorND], lr: f32) {
        for (param, idx) in [
            (&mut self.w1, self.idx[0]),
            (&mut self.b1, self.idx[1]),
            (&mut self.w2, self.idx[2]),
            (&mut self.b2, self.idx[3]),
        ]
        {
            if let Some(i) = idx
            {
                for (p, &g) in param.data.iter_mut().zip(&grads[i].data)
                {
                    *p -= lr * g;
                }
            }
        }
    }

    /// Trainable parameters (w1, b1, w2, b2) for an optimizer.
    pub fn parameters(&mut self) -> Vec<NdParam<'_>> {
        let mut out = Vec::new();
        // Disjoint field borrows, pushed in a fixed order.
        if let Some(i) = self.idx[0]
        {
            out.push(NdParam {
                value: &mut self.w1,
                grad_idx: i,
            });
        }
        if let Some(i) = self.idx[1]
        {
            out.push(NdParam {
                value: &mut self.b1,
                grad_idx: i,
            });
        }
        if let Some(i) = self.idx[2]
        {
            out.push(NdParam {
                value: &mut self.w2,
                grad_idx: i,
            });
        }
        if let Some(i) = self.idx[3]
        {
            out.push(NdParam {
                value: &mut self.b2,
                grad_idx: i,
            });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::nd_optim::NdAdam;

    fn mse<'t>(pred: NdVar<'t>, target: NdVar<'t>) -> NdVar<'t> {
        let d = pred.sub(target);
        d.mul(d).sum()
    }

    /// The RK4 integrator is numerically correct: `dy/dt = y` from `y(0)=1`
    /// integrates to `y(1) = e`.
    #[test]
    fn rk4_matches_exponential() {
        let t = NdTape::new();
        let y0 = t.input(TensorND::new(vec![1.0], vec![1, 1]));
        let y1 = rk4_integrate(&t, |y| y, y0, 100, 0.01);
        let got = t.value(y1).data[0];
        assert!(
            (got - std::f32::consts::E).abs() < 1e-3,
            "RK4 y(1) = {got}, expected e"
        );
    }

    /// Backprop **through the solver**: the gradient of the final state w.r.t.
    /// the initial condition matches finite differences.
    #[test]
    fn neural_ode_gradient_check() {
        let mut rng = PcgEngine::new(3);
        let mut ode = NeuralOde::new(2, 4, &mut rng);
        let y0 = [0.5f32, -0.3];
        let target = [0.2f32, 0.4];
        let (steps, dt) = (4usize, 0.1);

        let loss_of = |yd: &[f32], ode: &mut NeuralOde| -> f32 {
            let t = NdTape::new();
            let yv = t.input(TensorND::new(yd.to_vec(), vec![1, 2]));
            let tv = t.input(TensorND::new(target.to_vec(), vec![1, 2]));
            let yf = ode.integrate(&t, yv, steps, dt);
            t.value(mse(yf, tv)).data[0]
        };

        let t = NdTape::new();
        let yv = t.input(TensorND::new(y0.to_vec(), vec![1, 2]));
        let tv = t.input(TensorND::new(target.to_vec(), vec![1, 2]));
        let yf = ode.integrate(&t, yv, steps, dt);
        let grads = t.backward(mse(yf, tv));
        let gy = grads[yv.idx()].clone();

        let eps = 1e-3f32;
        for k in 0..y0.len()
        {
            let mut up = y0;
            let mut dn = y0;
            up[k] += eps;
            dn[k] -= eps;
            let num = (loss_of(&up, &mut ode) - loss_of(&dn, &mut ode)) / (2.0 * eps);
            assert!(
                (num - gy.data[k]).abs() < 2e-2,
                "neural-ode dy0 grad {k}: numeric {num}, analytic {}",
                gy.data[k]
            );
        }
    }

    /// End to end: training the dynamics (Adam through the solver) drives the
    /// integrated final state toward a target — the model learns a flow.
    #[test]
    fn neural_ode_trains() {
        let mut rng = PcgEngine::new(21);
        let mut ode = NeuralOde::new(2, 8, &mut rng);
        let y0 = [1.0f32, 0.0];
        let target = [0.0f32, 1.0];
        let (steps, dt) = (5usize, 0.1);
        let mut opt = NdAdam::with_lr(0.02);

        let mut first = f32::NAN;
        let mut last = f32::NAN;
        for step in 0..200
        {
            let t = NdTape::new();
            let yv = t.input(TensorND::new(y0.to_vec(), vec![1, 2]));
            let tv = t.input(TensorND::new(target.to_vec(), vec![1, 2]));
            let yf = ode.integrate(&t, yv, steps, dt);
            let loss_v = mse(yf, tv);
            let loss = t.value(loss_v).data[0];
            if step == 0
            {
                first = loss;
            }
            last = loss;
            let grads = t.backward(loss_v);
            let mut params = ode.parameters();
            opt.step(&mut params, &grads);
        }
        assert!(
            last < first * 0.1,
            "neural ODE did not learn the flow: first {first}, last {last}"
        );
    }
}
