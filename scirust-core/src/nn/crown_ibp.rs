//! **IBP certified (verified) training** — the interval-bound-propagation end of
//! CROWN-IBP (Zhang et al., *Towards Stable and Efficient Training of Verifiably
//! Robust Neural Networks*, ICLR 2020, arXiv:1906.06316).
//!
//! ⚠️ **Scope / honest labeling.** This implements pure **IBP** (differentiable
//! interval-bound propagation), which is the `β = 0` / IBP end of the CROWN-IBP
//! schedule. The bound it produces is **sound** (a valid over-approximation of
//! the worst-case logits, so certificates it issues are never false), but it is
//! *looser* than the full method: the **CROWN** component — per-neuron linear
//! lower/upper relaxations with backward substitution — is **not** implemented
//! here. Expect looser certified radii than a true CROWN-IBP; the linear
//! relaxation is future work. (Type names are kept for API stability.)
//!
//! Ordinary training minimises the loss at the *concrete* inputs; a network can fit
//! them perfectly yet flip its prediction under a tiny perturbation. CROWN-IBP
//! instead trains on a **certified bound of the worst-case loss** over an ℓ∞ ball
//! around each input, so the network becomes **provably** robust (its certified
//! radius grows).
//!
//! The key enabler is that **interval-bound propagation is differentiable**. For an
//! affine layer `y = x·W + b` the IBP box transforms as
//!
//! ```text
//! centre' = centre·W + b ,   radius' = radius·|W|
//! ```
//!
//! and `|W| = relu(W) + relu(−W)` — so the bound (including the `|W|` that used to
//! look like it needed a dedicated `abs` op) runs entirely on the N-D autograd tape.
//! ReLU on an interval `[l,u]` becomes `[relu(l), relu(u)]`, again pure `relu`. The
//! **robust logits** put the true class at its lower bound and every other class at
//! its upper bound (`zₜ = cₜ − rₜ`, `z_j = c_j + r_j`); a small cross-entropy on
//! those means the true class wins *even in the worst case* — i.e. the point is
//! certified. Deterministic; trained through the tape, measured with the plain-`f32`
//! [`IbpMlp`] verifier.

use crate::autodiff::nd::{NdTape, NdVar};
use crate::nn::ibp::{IbpLinear, IbpMlp};
use crate::nn::nd_optim::NdParam;
use crate::nn::rng::PcgEngine;
use crate::tensor::tensor_nd::TensorND;

/// A small ReLU MLP whose parameters are trained against the **certified** IBP loss.
/// Holds raw weights so the differentiable IBP forward can read `|W|` directly.
pub struct CrownIbpMlp {
    weights: Vec<TensorND>, // (din, dout) each
    biases: Vec<TensorND>,  // (1, dout) each
    w_idx: Vec<Option<usize>>,
    b_idx: Vec<Option<usize>>,
}

impl CrownIbpMlp {
    /// New MLP for the given layer sizes `dims = [in, h₁, …, out]` (ReLU between
    /// hidden layers), seeded with `1/√fan_in` weights and zero biases.
    pub fn new(dims: &[usize], rng: &mut PcgEngine) -> Self {
        assert!(dims.len() >= 2, "CrownIbpMlp: need at least in/out dims");
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        for l in 0..dims.len() - 1
        {
            let (din, dout) = (dims[l], dims[l + 1]);
            let scale = (1.0 / din as f32).sqrt();
            let w: Vec<f32> = (0..din * dout)
                .map(|_| rng.float_signed() * scale)
                .collect();
            weights.push(TensorND::new(w, vec![din, dout]));
            biases.push(TensorND::zeros(&[1, dout]));
        }
        let n = weights.len();
        Self {
            weights,
            biases,
            w_idx: vec![None; n],
            b_idx: vec![None; n],
        }
    }

    /// Input every parameter onto the tape **once** (so a parameter used by several
    /// paths keeps a single gradient node), recording its index.
    fn input_params<'t>(&mut self, tape: &'t NdTape) -> (Vec<NdVar<'t>>, Vec<NdVar<'t>>) {
        let mut ws = Vec::with_capacity(self.weights.len());
        let mut bs = Vec::with_capacity(self.biases.len());
        for l in 0..self.weights.len()
        {
            let w = tape.input(self.weights[l].clone());
            self.w_idx[l] = Some(w.idx());
            ws.push(w);
            let b = tape.input(self.biases[l].clone());
            self.b_idx[l] = Some(b.idx());
            bs.push(b);
        }
        (ws, bs)
    }

    /// Differentiable **IBP propagation** of the ℓ∞ box (`centre = x`, `radius = eps`)
    /// through the network, returning the certified output `(centre, radius)`. The
    /// `|W|` uses `relu(W)+relu(−W)` and the ReLU-interval uses `relu`, so the whole
    /// bound runs on the tape.
    fn ibp_propagate<'t>(
        &mut self,
        tape: &'t NdTape,
        x: NdVar<'t>,
        eps: f32,
    ) -> (NdVar<'t>, NdVar<'t>) {
        let (ws, bs) = self.input_params(tape);
        let (batch, din0) = (x.shape()[0], x.shape()[1]);
        let neg1 = tape.input(TensorND::new(vec![-1.0f32], vec![1, 1]));
        let half = tape.input(TensorND::new(vec![0.5f32], vec![1, 1]));
        let mut center = x;
        let mut radius = tape.input(TensorND::new(vec![eps; batch * din0], vec![batch, din0]));
        let nl = ws.len();
        for l in 0..nl
        {
            let absw = ws[l].relu().add(ws[l].mul(neg1).relu()); // |W| = relu(W)+relu(−W)
            let nc = center.matmul(ws[l]).add(bs[l]); // centre·W + b
            let nr = radius.matmul(absw); // radius·|W|
            if l + 1 < nl
            {
                // ReLU on the interval [c−r, c+r] → [relu(lo), relu(hi)].
                let lo = nc.sub(nr).relu();
                let hi = nc.add(nr).relu();
                center = lo.add(hi).mul(half);
                radius = hi.sub(lo).mul(half);
            }
            else
            {
                center = nc;
                radius = nr;
            }
        }
        (center, radius)
    }

    /// **Certified (robust) loss** over an ℓ∞ ball of radius `eps` around each row of
    /// `x` (`batch × in`). Propagates the IBP box through the network on the tape,
    /// builds the worst-case ("robust") logits, and returns their cross-entropy with
    /// `targets`. Minimising it makes the points **certifiably** classified.
    /// (`eps = 0` recovers ordinary cross-entropy training.)
    pub fn robust_loss<'t>(
        &mut self,
        tape: &'t NdTape,
        x: NdVar<'t>,
        eps: f32,
        targets: &[usize],
    ) -> NdVar<'t> {
        let (center, radius) = self.ibp_propagate(tape, x, eps);
        // Robust logits: zₜ = cₜ − rₜ (true class at its lower bound), z_j = c_j + r_j.
        let (batch, k) = (center.shape()[0], center.shape()[1]);
        let mut s = vec![1.0f32; batch * k];
        for (i, &t) in targets.iter().enumerate()
        {
            s[i * k + t] = -1.0;
        }
        let smask = tape.input(TensorND::new(s, vec![batch, k]));
        let z = center.add(radius.mul(smask));
        z.cross_entropy(targets)
    }

    /// The certified output box `(lower, upper)` for a **single** input `x` from the
    /// **tape** IBP forward — used to validate the differentiable propagation against
    /// the plain [`IbpMlp`] verifier.
    pub fn certified_box(&mut self, x: &[f32], eps: f32) -> (Vec<f32>, Vec<f32>) {
        let tape = NdTape::new();
        let xv = tape.input(TensorND::new(x.to_vec(), vec![1, x.len()]));
        let (c, r) = self.ibp_propagate(&tape, xv, eps);
        let (cv, rv) = (tape.value(c), tape.value(r));
        let lo = cv
            .data
            .iter()
            .zip(rv.data.iter())
            .map(|(&c, &r)| c - r)
            .collect();
        let hi = cv
            .data
            .iter()
            .zip(rv.data.iter())
            .map(|(&c, &r)| c + r)
            .collect();
        (lo, hi)
    }

    /// Trainable parameters (weights and biases, in layer order).
    pub fn parameters(&mut self) -> Vec<NdParam<'_>> {
        let mut params = Vec::new();
        for (((w, b), wi), bi) in self
            .weights
            .iter_mut()
            .zip(self.biases.iter_mut())
            .zip(&self.w_idx)
            .zip(&self.b_idx)
        {
            if let Some(i) = wi
            {
                params.push(NdParam {
                    value: w,
                    grad_idx: *i,
                });
            }
            if let Some(i) = bi
            {
                params.push(NdParam {
                    value: b,
                    grad_idx: *i,
                });
            }
        }
        params
    }

    /// Build the plain-`f32` [`IbpMlp`] verifier from the current weights — used to
    /// **measure** the certified radius independently of the tape.
    pub fn to_ibp_mlp(&self) -> IbpMlp {
        let layers: Vec<IbpLinear> = self
            .weights
            .iter()
            .zip(&self.biases)
            .map(|(w, b)| {
                let (din, dout) = (w.shape[0], w.shape[1]);
                IbpLinear::new(w.data.to_vec(), b.data.to_vec(), din, dout)
            })
            .collect();
        IbpMlp::new(layers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::ibp::{Interval, certified_robust};
    use crate::nn::nd_optim::NdAdam;

    /// The differentiable (tape) IBP forward **matches** the reference [`IbpMlp`]
    /// verifier and is **sound**: every concrete forward of a point in the input box
    /// lands inside the tape-certified output box `[c−r, c+r]`.
    #[test]
    fn ibp_forward_is_sound() {
        let mut rng = PcgEngine::new(1);
        let mut net = CrownIbpMlp::new(&[2, 5, 3], &mut rng);
        let x = [0.4f32, -0.3];
        let eps = 0.2f32;
        // Box from the tape IBP forward, and from the plain reference verifier.
        let (lo, hi) = net.certified_box(&x, eps);
        let ibp = net.to_ibp_mlp();
        let reference = ibp.certify(&Interval::around(&x, eps));
        for k in 0..lo.len()
        {
            assert!(
                (lo[k] - reference.lo[k]).abs() < 1e-4,
                "tape lo ≠ reference"
            );
            assert!(
                (hi[k] - reference.hi[k]).abs() < 1e-4,
                "tape hi ≠ reference"
            );
        }
        // Soundness: sample the input box, concrete-forward, must be within the box.
        let mut rng2 = PcgEngine::new(7);
        for _ in 0..2000
        {
            let p = [
                x[0] + eps * rng2.float_signed(),
                x[1] + eps * rng2.float_signed(),
            ];
            let y = ibp.forward(&p);
            for (k, &yk) in y.iter().enumerate()
            {
                assert!(
                    yk >= lo[k] - 1e-4 && yk <= hi[k] + 1e-4,
                    "unsound: y[{k}]={yk} not in [{}, {}]",
                    lo[k],
                    hi[k]
                );
            }
        }
    }

    /// Average certified ℓ∞ radius over a dataset, via the plain IBP verifier
    /// (binary search for the largest `eps` that certifies the correct class).
    fn avg_certified_radius(
        net: &CrownIbpMlp,
        data: &[[f32; 2]],
        labels: &[usize],
    ) -> (f32, usize) {
        let ibp = net.to_ibp_mlp();
        let mut total = 0.0f32;
        let mut correct = 0usize;
        for (x, &y) in data.iter().zip(labels)
        {
            let pred = ibp
                .forward(x)
                .iter()
                .enumerate()
                .fold(
                    (0usize, f32::MIN),
                    |b, (i, &v)| if v > b.1 { (i, v) } else { b },
                )
                .0;
            if pred == y
            {
                correct += 1;
            }
            // Binary search the certified radius.
            let (mut lo, mut hi) = (0.0f32, 2.0f32);
            for _ in 0..20
            {
                let mid = 0.5 * (lo + hi);
                if certified_robust(&ibp.certify(&Interval::around(x, mid)), y)
                {
                    lo = mid;
                }
                else
                {
                    hi = mid;
                }
            }
            total += lo;
        }
        (total / data.len() as f32, correct)
    }

    /// **Certified training works**: a net trained against the IBP robust loss
    /// certifies a substantially **larger** ℓ∞ radius than a net trained for plain
    /// accuracy, while both still classify the (separable) data correctly.
    #[test]
    fn crown_ibp_training_grows_certified_radius() {
        // Two well-separated 2-D clusters.
        let mut data = Vec::new();
        let mut labels = Vec::new();
        let mut rng = PcgEngine::new(3);
        for _ in 0..32
        {
            data.push([-1.5 + 0.25 * rng.float_signed(), 0.25 * rng.float_signed()]);
            labels.push(0usize);
            data.push([1.5 + 0.25 * rng.float_signed(), 0.25 * rng.float_signed()]);
            labels.push(1usize);
        }
        let flat: Vec<f32> = data.iter().flat_map(|p| [p[0], p[1]]).collect();
        let batch = data.len();

        let train = |eps: f32| -> CrownIbpMlp {
            let mut rng = PcgEngine::new(11);
            let mut net = CrownIbpMlp::new(&[2, 8, 2], &mut rng);
            let mut opt = NdAdam::with_lr(0.05);
            for _ in 0..200
            {
                let tape = NdTape::new();
                let xv = tape.input(TensorND::new(flat.to_vec(), vec![batch, 2]));
                let loss = net.robust_loss(&tape, xv, eps, &labels);
                let grads = tape.backward(loss);
                opt.step(&mut net.parameters(), &grads);
            }
            net
        };

        let clean_net = train(0.0); // plain cross-entropy
        let robust_net = train(0.4); // certified training at radius 0.4

        let (clean_r, clean_acc) = avg_certified_radius(&clean_net, &data, &labels);
        let (robust_r, robust_acc) = avg_certified_radius(&robust_net, &data, &labels);

        assert_eq!(clean_acc, batch, "clean net misclassifies");
        assert_eq!(robust_acc, batch, "robust net misclassifies");
        assert!(
            robust_r > clean_r + 0.2,
            "certified training did not grow the radius: robust {robust_r} vs clean {clean_r}"
        );

        // Determinism: identical certified radius on a repeat.
        let robust_net2 = train(0.4);
        let (robust_r2, _) = avg_certified_radius(&robust_net2, &data, &labels);
        assert_eq!(robust_r.to_bits(), robust_r2.to_bits());
    }
}
