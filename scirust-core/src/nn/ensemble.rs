//! **Deep Ensembles** (Lakshminarayanan, Pritzel & Blundell, NeurIPS 2017) —
//! predictive uncertainty by training several models from different seeds. The
//! ensemble **mean** is a better point predictor (its MSE is `≤` the average
//! member MSE, by Jensen), and the **disagreement** (standard deviation) across
//! members is an **epistemic uncertainty** signal: small where the data pinned the
//! function down, large out-of-distribution where the members are free to
//! extrapolate apart. Each member is a small ReLU MLP trained on the N-D tape with
//! [`NdAdam`]; with a seeded [`PcgEngine`] per member the whole ensemble is
//! **bit-for-bit deterministic**.

use crate::autodiff::nd::{NdTape, NdVar};
use crate::nn::PcgEngine;
use crate::nn::nd_layers::NdLinear;
use crate::nn::nd_optim::{NdAdam, NdParam};
use crate::tensor::tensor_nd::TensorND;

/// One ensemble member: a `1 → hidden → 1` ReLU MLP.
struct MlpMember {
    l1: NdLinear,
    l2: NdLinear,
}

impl MlpMember {
    fn new(hidden: usize, rng: &mut PcgEngine) -> Self {
        Self {
            l1: NdLinear::new(1, hidden, rng),
            l2: NdLinear::new(hidden, 1, rng),
        }
    }

    fn forward<'t>(&mut self, tape: &'t NdTape, x: NdVar<'t>) -> NdVar<'t> {
        let h = self.l1.forward(tape, x).relu();
        self.l2.forward(tape, h)
    }

    fn parameters(&mut self) -> Vec<NdParam<'_>> {
        let mut p = self.l1.parameters();
        p.extend(self.l2.parameters());
        p
    }

    fn predict(&mut self, x: f32) -> f32 {
        let tape = NdTape::new();
        let xv = tape.input(TensorND::new(vec![x], vec![1, 1]));
        tape.value(self.forward(&tape, xv)).data[0]
    }
}

/// A **deep ensemble** of small ReLU-MLP regressors.
pub struct DeepEnsemble {
    members: Vec<MlpMember>,
}

impl DeepEnsemble {
    /// Train `n_members` MLPs (hidden width `hidden`) on the 1-D dataset
    /// `(xs, ys)`, each seeded from `base_seed + member_index`, for `steps` Adam
    /// steps at learning rate `lr`.
    pub fn train(
        xs: &[f32],
        ys: &[f32],
        n_members: usize,
        hidden: usize,
        steps: usize,
        lr: f32,
        base_seed: u64,
    ) -> Self {
        assert_eq!(xs.len(), ys.len(), "DeepEnsemble: xs/ys length mismatch");
        let n = xs.len();
        let xt = TensorND::new(xs.to_vec(), vec![n, 1]);
        let yt = TensorND::new(ys.to_vec(), vec![n, 1]);
        let members = (0..n_members)
            .map(|m| {
                let mut rng = PcgEngine::new(base_seed + m as u64);
                let mut mlp = MlpMember::new(hidden, &mut rng);
                let mut opt = NdAdam::with_lr(lr);
                for _ in 0..steps
                {
                    let tape = NdTape::new();
                    let xv = tape.input(xt.clone());
                    let tv = tape.input(yt.clone());
                    let out = mlp.forward(&tape, xv);
                    let diff = out.sub(tv);
                    let loss = diff.mul(diff).sum();
                    let grads = tape.backward(loss);
                    opt.step(&mut mlp.parameters(), &grads);
                }
                mlp
            })
            .collect();
        Self { members }
    }

    /// Predict at `x`: `(mean, std)` over the members — the point estimate and its
    /// **epistemic uncertainty** (member disagreement).
    pub fn predict(&mut self, x: f32) -> (f32, f32) {
        let preds: Vec<f32> = self.members.iter_mut().map(|m| m.predict(x)).collect();
        let k = preds.len() as f32;
        let mean = preds.iter().sum::<f32>() / k;
        let var = preds.iter().map(|&p| (p - mean) * (p - mean)).sum::<f32>() / k;
        (mean, var.sqrt())
    }

    /// Number of members.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Whether the ensemble has no members.
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Deep ensembles, tested.** The ensemble mean's MSE is `≤` the average
    /// member MSE (Jensen — and here strictly lower, from member diversity), and
    /// the disagreement (std) is **far larger out-of-distribution** (well outside
    /// the training range) than in-distribution.
    #[test]
    fn deep_ensemble_reduces_error_and_flags_ood() {
        let n = 48usize;
        let xs: Vec<f32> = (0..n)
            .map(|i| -1.0 + 2.0 * i as f32 / (n as f32 - 1.0))
            .collect();
        let ys: Vec<f32> = xs.iter().map(|&x| (2.0 * x).sin()).collect();
        let mut ens = DeepEnsemble::train(&xs, &ys, 6, 24, 500, 0.02, 100);

        // (1) Ensemble MSE ≤ average member MSE (variance reduction).
        let k = ens.len() as f32;
        let mut avg_member_mse = 0.0f32;
        for m in ens.members.iter_mut()
        {
            let mse: f32 = xs
                .iter()
                .zip(&ys)
                .map(|(&x, &y)| (m.predict(x) - y).powi(2))
                .sum::<f32>()
                / n as f32;
            avg_member_mse += mse / k;
        }
        let ens_mse: f32 = xs
            .iter()
            .zip(&ys)
            .map(|(&x, &y)| (ens.predict(x).0 - y).powi(2))
            .sum::<f32>()
            / n as f32;
        assert!(
            ens_mse <= avg_member_mse + 1e-6,
            "ensemble {ens_mse} > avg member {avg_member_mse}"
        );

        // (2) OOD uncertainty: std at x=4 (far outside [-1,1]) ≫ std at x=0.
        let (_, u_ood) = ens.predict(4.0);
        let (_, u_in) = ens.predict(0.0);
        assert!(
            u_ood > 2.0 * u_in + 0.05,
            "OOD std {u_ood} not ≫ in-dist std {u_in}"
        );
    }

    /// The seeded ensemble is bit-for-bit deterministic.
    #[test]
    fn deep_ensemble_is_deterministic() {
        let xs: Vec<f32> = (0..20).map(|i| i as f32 * 0.1 - 1.0).collect();
        let ys: Vec<f32> = xs.iter().map(|&x| x * x).collect();
        let run = || {
            let mut e = DeepEnsemble::train(&xs, &ys, 3, 8, 100, 0.03, 7);
            let (m, s) = e.predict(0.5);
            (m.to_bits(), s.to_bits())
        };
        assert_eq!(run(), run());
    }
}
