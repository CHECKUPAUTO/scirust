// scirust-core/src/nn/loss/poisson.rs
//
// Poisson negative log-likelihood — the loss for count regression, where the
// target is a non-negative integer count `k` and the model predicts a Poisson
// rate `λ` (or its log). Core previously had only cross-entropy / MSE / NLL,
// none of which model counts.
//
// With `log_input = true` (the numerically stable default) the prediction is
// `log λ` and the per-element loss is
//
//   ℓ = exp(pred) − target · pred            (+ ln(k!) if `full`)
//
// With `log_input = false` the prediction is `λ` directly and
//
//   ℓ = pred − target · ln(pred + eps)        (+ ln(k!) if `full`)
//
// The `ln(k!)` normalizer is constant in `pred`, so it changes the reported
// loss but not the gradient; when `full` is set it is computed exactly as
// `ln Γ(k+1)` via `scirust_special::ln_gamma`. Reduction is mean by default.
//
// The backward is left entirely to autograd: `exp`, `hadamard`, `sub`, `log`,
// `sum` and `scale` push the right Ops, so `∂ℓ/∂pred` (e.g. `exp(pred) − target`
// for `log_input`) follows from the chain rule.

use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::loss::Loss;

pub struct PoissonNllLoss {
    /// If `true`, `pred` is `log λ` (stable, no `eps`); if `false`, `pred` is `λ`.
    pub log_input: bool,
    /// Include the `ln(k!) = ln Γ(k+1)` Stirling term (via `scirust_special`).
    pub full: bool,
    /// Small constant added inside the log when `log_input = false`.
    pub eps: f32,
    /// Mean-reduce (`true`) or sum-reduce (`false`) over all elements.
    pub mean: bool,
}

impl PoissonNllLoss {
    /// Default: log-input, no Stirling term, mean reduction — matches the common
    /// `PoissonNLLLoss(log_input=True, full=False)` configuration.
    pub fn new() -> Self {
        Self {
            log_input: true,
            full: false,
            eps: 1e-8,
            mean: true,
        }
    }

    /// Builder: choose the input parameterization and whether to add `ln(k!)`.
    pub fn with(log_input: bool, full: bool) -> Self {
        Self {
            log_input,
            full,
            ..Self::new()
        }
    }
}

impl Default for PoissonNllLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl Loss for PoissonNllLoss {
    fn forward<'t>(&self, tape: &'t Tape, pred: Var<'t>, target: Var<'t>) -> Var<'t> {
        let (rows, cols) = pred.shape();
        let n = (rows * cols) as f32;

        // Per-element base loss.
        let base = if self.log_input
        {
            // exp(pred) − target · pred
            pred.exp().sub(target.hadamard(pred))
        }
        else
        {
            // pred − target · ln(pred + eps)
            let eps_v = tape.input(Tensor::from_vec(vec![self.eps; rows * cols], rows, cols));
            pred.sub(target.hadamard(pred.add(eps_v).log()))
        };

        // Optional exact Stirling term ln Γ(k+1): constant in `pred`, added as a
        // leaf so it shows in the loss value without touching pred's gradient.
        let base = if self.full
        {
            let tv = tape.value(target.idx());
            let full_data: Vec<f32> = tv
                .data
                .iter()
                .map(|&k| scirust_special::ln_gamma((k as f64) + 1.0) as f32)
                .collect();
            base.add(tape.input(Tensor::from_vec(full_data, rows, cols)))
        }
        else
        {
            base
        };

        let s = base.sum();
        if self.mean { s.scale(1.0 / n) } else { s }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poisson_log_input_value_and_gradient() {
        // log_input: ℓ = mean(exp(pred) − target·pred);
        // ∂ℓ/∂pred = (exp(pred) − target) / N.
        let tape = Tape::new();
        let pred = tape.input(Tensor::from_vec(vec![0.0, 1.0], 1, 2)); // λ = [1, e]
        let target = tape.input(Tensor::from_vec(vec![2.0, 3.0], 1, 2));
        let loss = PoissonNllLoss::new().forward(&tape, pred, target);

        // value: mean( (e^0 − 2·0), (e^1 − 3·1) ) = mean(1, e−3) = (1 + e − 3)/2.
        let expected = (1.0 + std::f32::consts::E - 3.0) / 2.0;
        assert!((tape.value(loss.idx()).data[0] - expected).abs() < 1e-5);

        tape.backward(loss.idx());
        let g = tape.grad(pred.idx());
        // (e^0 − 2)/2 = −0.5 ; (e^1 − 3)/2 = (e−3)/2 ≈ −0.14086.
        assert!((g.data[0] - (-0.5)).abs() < 1e-5, "g0 = {}", g.data[0]);
        assert!(
            (g.data[1] - (std::f32::consts::E - 3.0) / 2.0).abs() < 1e-5,
            "g1 = {}",
            g.data[1]
        );
    }

    #[test]
    fn poisson_full_adds_exact_log_factorial() {
        // With `full`, the loss gains Σ ln(k!) / N but the gradient is unchanged.
        let tape = Tape::new();
        let pred = tape.input(Tensor::from_vec(vec![0.0, 1.0], 1, 2));
        let target = tape.input(Tensor::from_vec(vec![2.0, 3.0], 1, 2));
        let plain = PoissonNllLoss::new().forward(&tape, pred, target);
        let full = PoissonNllLoss::with(true, true).forward(&tape, pred, target);
        // ln(2!) + ln(3!) = ln2 + ln6, meaned over N=2.
        let delta = (2f32.ln() + 6f32.ln()) / 2.0;
        let got = tape.value(full.idx()).data[0] - tape.value(plain.idx()).data[0];
        assert!(
            (got - delta).abs() < 1e-4,
            "full − plain = {got}, expected {delta}"
        );
    }

    #[test]
    fn poisson_direct_rate_matches_formula() {
        // log_input = false: ℓ = mean(λ − k·ln(λ+eps)).
        let tape = Tape::new();
        let pred = tape.input(Tensor::from_vec(vec![1.0, 4.0], 1, 2)); // λ
        let target = tape.input(Tensor::from_vec(vec![1.0, 2.0], 1, 2));
        let loss = PoissonNllLoss::with(false, false).forward(&tape, pred, target);
        let eps = 1e-8f32;
        let expected = ((1.0 - 1.0 * (1.0 + eps).ln()) + (4.0 - 2.0 * (4.0 + eps).ln())) / 2.0;
        assert!((tape.value(loss.idx()).data[0] - expected).abs() < 1e-4);
    }
}
