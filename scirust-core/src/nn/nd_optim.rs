//! A reusable **Adam** optimizer (Kingma & Ba, 2014) for the N-D layers
//! ([`crate::nn::nd_layers`], [`crate::nn::nd_decoder`]).
//!
//! The 2-D path keys its moments by tape-node index ([`crate::autodiff::optim`]);
//! here the parameters live *inside* the layers, so a layer exposes them as an
//! ordered list of [`NdParam`] (a mutable view of the values plus the index of
//! their gradient in a [`backward`](crate::autodiff::nd::NdTape::backward)
//! result). The optimizer holds the moment buffers aligned to that list. All
//! arithmetic is plain `f32` in a fixed order, so a run is **bit-for-bit
//! deterministic**.

use crate::tensor::tensor_nd::TensorND;

/// A handle to one trainable parameter for an optimizer: a mutable view of its
/// values, and the index of its gradient in an `NdTape::backward` result.
pub struct NdParam<'a> {
    /// The parameter's values (updated in place).
    pub value: &'a mut TensorND,
    /// Index of this parameter's gradient in the `backward` result.
    pub grad_idx: usize,
}

/// Adam hyper-parameters. [`Default`] is the canonical
/// `lr = 1e-3, β1 = 0.9, β2 = 0.999, eps = 1e-8`.
#[derive(Clone, Copy, Debug)]
pub struct AdamConfig {
    /// Learning rate.
    pub lr: f32,
    /// First-moment (mean) decay.
    pub beta1: f32,
    /// Second-moment (variance) decay.
    pub beta2: f32,
    /// Numerical-stability term in the denominator.
    pub eps: f32,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }
}

/// Adam over a fixed, ordered set of N-D parameters. The first/second moment
/// buffers are lazily sized on the first [`step`](Self::step) and aligned to the
/// parameter list positionally, so the same parameters must be passed in the
/// same order on every call.
pub struct NdAdam {
    cfg: AdamConfig,
    t: u64,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl NdAdam {
    /// New optimizer with the given config (no steps taken yet).
    pub fn new(cfg: AdamConfig) -> Self {
        Self {
            cfg,
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    /// Adam with default betas/eps at learning rate `lr`.
    pub fn with_lr(lr: f32) -> Self {
        Self::new(AdamConfig {
            lr,
            ..AdamConfig::default()
        })
    }

    /// Number of steps taken so far (drives the bias correction).
    pub fn step_count(&self) -> u64 {
        self.t
    }

    /// One Adam update over `params`, reading each gradient from `grads` by the
    /// parameter's `grad_idx`. `params` must be in the same order on every call.
    pub fn step(&mut self, params: &mut [NdParam], grads: &[TensorND]) {
        if self.m.is_empty() && !params.is_empty()
        {
            self.m = params
                .iter()
                .map(|p| vec![0.0f32; p.value.data.len()])
                .collect();
            self.v = params
                .iter()
                .map(|p| vec![0.0f32; p.value.data.len()])
                .collect();
        }
        assert_eq!(
            self.m.len(),
            params.len(),
            "NdAdam: parameter count changed between steps"
        );
        self.t += 1;
        let AdamConfig {
            lr,
            beta1,
            beta2,
            eps,
        } = self.cfg;
        let bc1 = 1.0 - beta1.powi(self.t as i32);
        let bc2 = 1.0 - beta2.powi(self.t as i32);

        for (k, p) in params.iter_mut().enumerate()
        {
            let g = &grads[p.grad_idx].data;
            assert_eq!(
                g.len(),
                p.value.data.len(),
                "NdAdam: grad/param size mismatch at parameter {k}"
            );
            let mk = &mut self.m[k];
            let vk = &mut self.v[k];
            for j in 0..p.value.data.len()
            {
                let gj = g[j];
                mk[j] = beta1 * mk[j] + (1.0 - beta1) * gj;
                vk[j] = beta2 * vk[j] + (1.0 - beta2) * gj * gj;
                let mhat = mk[j] / bc1;
                let vhat = vk[j] / bc2;
                p.value.data[j] -= lr * mhat / (vhat.sqrt() + eps);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Adam minimises a quadratic `Σ(xᵢ − targetᵢ)²` (gradient `2(x − target)`),
    /// driving `x` to the target — the standard optimizer oracle.
    #[test]
    fn nd_adam_converges_on_quadratic() {
        let target = [3.0f32, -2.0, 0.5];
        let mut x = TensorND::new(vec![0.0, 0.0, 0.0], vec![3]);
        let mut opt = NdAdam::with_lr(0.1);

        for _ in 0..500
        {
            let grad_data: Vec<f32> = x
                .data
                .iter()
                .zip(&target)
                .map(|(&xi, &ti)| 2.0 * (xi - ti))
                .collect();
            let grads = vec![TensorND::new(grad_data, vec![3])];
            let mut params = vec![NdParam {
                value: &mut x,
                grad_idx: 0,
            }];
            opt.step(&mut params, &grads);
        }

        for (&xi, &ti) in x.data.iter().zip(&target)
        {
            assert!(
                (xi - ti).abs() < 1e-3,
                "x={:?}, target={:?}",
                x.data,
                target
            );
        }
        assert_eq!(opt.step_count(), 500);
    }

    /// Determinism: two independent Adam runs on the same problem produce
    /// bit-for-bit identical parameters.
    #[test]
    fn nd_adam_is_deterministic() {
        let run = || -> Vec<f32> {
            let target = [1.0f32, -1.0];
            let mut x = TensorND::new(vec![0.5, 0.5], vec![2]);
            let mut opt = NdAdam::with_lr(0.05);
            for _ in 0..120
            {
                let gd: Vec<f32> = x
                    .data
                    .iter()
                    .zip(&target)
                    .map(|(&xi, &ti)| 2.0 * (xi - ti))
                    .collect();
                let grads = vec![TensorND::new(gd, vec![2])];
                let mut params = vec![NdParam {
                    value: &mut x,
                    grad_idx: 0,
                }];
                opt.step(&mut params, &grads);
            }
            x.data
        };
        assert_eq!(run(), run());
    }
}
