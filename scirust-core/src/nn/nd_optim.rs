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

/// Adam / **AdamW** hyper-parameters. [`Default`] is the canonical
/// `lr = 1e-3, β1 = 0.9, β2 = 0.999, eps = 1e-8` with no weight decay (plain
/// Adam). Set `weight_decay > 0` for **AdamW** (decoupled weight decay,
/// Loshchilov & Hutter 2017): the decay is applied directly to the parameter,
/// not through the gradient/moments.
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
    /// Decoupled weight decay (0 = plain Adam).
    pub weight_decay: f32,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
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

    /// Adam with default betas/eps at learning rate `lr` (no weight decay).
    pub fn with_lr(lr: f32) -> Self {
        Self::new(AdamConfig {
            lr,
            ..AdamConfig::default()
        })
    }

    /// **AdamW** at learning rate `lr` with decoupled `weight_decay`.
    pub fn with_lr_wd(lr: f32, weight_decay: f32) -> Self {
        Self::new(AdamConfig {
            lr,
            weight_decay,
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
            weight_decay,
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
                // AdamW: decoupled weight decay on the pre-update parameter.
                let theta = p.value.data[j];
                p.value.data[j] = theta - lr * (mhat / (vhat.sqrt() + eps) + weight_decay * theta);
            }
        }
    }
}

/// Lion hyper-parameters (Chen et al. 2023). [`Default`] is `lr = 1e-4,
/// β1 = 0.9, β2 = 0.99` with no weight decay. Lion typically wants a learning
/// rate ~3–10× smaller than Adam.
#[derive(Clone, Copy, Debug)]
pub struct LionConfig {
    /// Learning rate.
    pub lr: f32,
    /// Decay for the update-direction interpolation.
    pub beta1: f32,
    /// Decay for the momentum state.
    pub beta2: f32,
    /// Decoupled weight decay (0 = none).
    pub weight_decay: f32,
}

impl Default for LionConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.0,
        }
    }
}

/// **Lion** optimizer (*EvoLved Sign Momentum*, Chen et al. 2023): the update is
/// the **sign** of an interpolated momentum, so every parameter moves by exactly
/// `±lr` (plus decoupled weight decay). One state buffer (half of Adam's), and
/// — being pure `f32` in a fixed order — **bit-for-bit deterministic**.
pub struct NdLion {
    cfg: LionConfig,
    m: Vec<Vec<f32>>,
}

impl NdLion {
    /// New optimizer with the given config.
    pub fn new(cfg: LionConfig) -> Self {
        Self { cfg, m: Vec::new() }
    }

    /// Lion with default betas at learning rate `lr`.
    pub fn with_lr(lr: f32) -> Self {
        Self::new(LionConfig {
            lr,
            ..LionConfig::default()
        })
    }

    /// One Lion update over `params` (same ordering contract as [`NdAdam`]).
    pub fn step(&mut self, params: &mut [NdParam], grads: &[TensorND]) {
        if self.m.is_empty() && !params.is_empty()
        {
            self.m = params
                .iter()
                .map(|p| vec![0.0f32; p.value.data.len()])
                .collect();
        }
        assert_eq!(
            self.m.len(),
            params.len(),
            "NdLion: parameter count changed between steps"
        );
        let LionConfig {
            lr,
            beta1,
            beta2,
            weight_decay,
        } = self.cfg;

        for (k, p) in params.iter_mut().enumerate()
        {
            let g = &grads[p.grad_idx].data;
            assert_eq!(
                g.len(),
                p.value.data.len(),
                "NdLion: grad/param size mismatch at parameter {k}"
            );
            let mk = &mut self.m[k];
            for j in 0..p.value.data.len()
            {
                let gj = g[j];
                // Update direction = sign(β1·m + (1−β1)·g).
                let u = beta1 * mk[j] + (1.0 - beta1) * gj;
                let step = if u > 0.0
                {
                    1.0
                }
                else if u < 0.0
                {
                    -1.0
                }
                else
                {
                    0.0
                };
                let theta = p.value.data[j];
                p.value.data[j] = theta - lr * (step + weight_decay * theta);
                // Momentum update uses β2.
                mk[j] = beta2 * mk[j] + (1.0 - beta2) * gj;
            }
        }
    }
}

// --- Muon (matrix-aware optimizer) ---------------------------------------

/// Frobenius norm of a flat matrix.
fn frob_norm(a: &[f32]) -> f32 {
    a.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// Transpose an `r×c` row-major matrix into `c×r`.
fn transpose(a: &[f32], r: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; r * c];
    for i in 0..r
    {
        for j in 0..c
        {
            out[j * r + i] = a[i * c + j];
        }
    }
    out
}

/// Matrix product `(ar×ac) · (ac×bc) → (ar×bc)`, row-major.
fn matmul(a: &[f32], ar: usize, ac: usize, b: &[f32], bc: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; ar * bc];
    for i in 0..ar
    {
        for p in 0..ac
        {
            let aip = a[i * ac + p];
            if aip == 0.0
            {
                continue;
            }
            for j in 0..bc
            {
                out[i * bc + j] += aip * b[p * bc + j];
            }
        }
    }
    out
}

/// **Newton–Schulz** orthogonalisation (the quintic iteration from Muon): pushes
/// a matrix's singular values into a band near 1 (approximately orthogonal)
/// using only matmuls, so it is matrix-free and deterministic. Five steps is the
/// usual choice; the iteration holds the values in the band rather than
/// converging to exactly `I`. Returns the `rows×cols` result.
pub fn newton_schulz_orthogonalize(g: &[f32], rows: usize, cols: usize, steps: usize) -> Vec<f32> {
    assert_eq!(g.len(), rows * cols, "newton_schulz: size mismatch");
    let (a, b, c) = (3.4445f32, -4.7750f32, 2.0315f32);
    let norm = frob_norm(g) + 1e-7;
    let mut x: Vec<f32> = g.iter().map(|&v| v / norm).collect();
    // Work in the "wide" orientation (rows ≤ cols).
    let transposed = rows > cols;
    let (mut r, mut cc) = (rows, cols);
    if transposed
    {
        x = transpose(&x, rows, cols);
        std::mem::swap(&mut r, &mut cc);
    }
    for _ in 0..steps
    {
        let xt = transpose(&x, r, cc); // cc×r
        let aa = matmul(&x, r, cc, &xt, r); // A = X·Xᵀ  (r×r)
        let a2 = matmul(&aa, r, r, &aa, r); // A²       (r×r)
        let bmat: Vec<f32> = aa.iter().zip(&a2).map(|(&u, &w)| b * u + c * w).collect();
        let bx = matmul(&bmat, r, r, &x, cc); // B·X     (r×cc)
        for (xi, &bxi) in x.iter_mut().zip(&bx)
        {
            *xi = a * *xi + bxi;
        }
    }
    if transposed
    {
        x = transpose(&x, r, cc);
    }
    x
}

/// Hyper-parameters for [`NdMuon`].
#[derive(Clone, Copy, Debug)]
pub struct MuonConfig {
    /// Learning rate.
    pub lr: f32,
    /// Momentum coefficient (β).
    pub momentum: f32,
    /// Newton–Schulz iteration count.
    pub ns_steps: usize,
    /// Decoupled weight decay.
    pub weight_decay: f32,
}

impl Default for MuonConfig {
    fn default() -> Self {
        Self {
            lr: 0.02,
            momentum: 0.95,
            ns_steps: 5,
            weight_decay: 0.0,
        }
    }
}

/// **Muon** (Jordan et al. 2024): momentum, then **orthogonalise** the update of
/// each 2-D weight matrix with Newton–Schulz before the step. Matrices with both
/// dims ≥ 2 use the orthogonalised update (scaled by `√(rows/cols)`); other
/// parameters (biases, `(1,n)` scales) fall back to momentum SGD — matching the
/// paper's "Muon for hidden matrices, plain update for the rest". Pure `f32` in a
/// fixed order ⇒ **bit-for-bit deterministic**.
pub struct NdMuon {
    cfg: MuonConfig,
    m: Vec<Vec<f32>>,
}

impl NdMuon {
    /// New optimizer with the given config.
    pub fn new(cfg: MuonConfig) -> Self {
        Self { cfg, m: Vec::new() }
    }

    /// Muon with default momentum/steps at learning rate `lr`.
    pub fn with_lr(lr: f32) -> Self {
        Self::new(MuonConfig {
            lr,
            ..MuonConfig::default()
        })
    }

    /// One Muon update over `params` (same ordering contract as [`NdAdam`]).
    pub fn step(&mut self, params: &mut [NdParam], grads: &[TensorND]) {
        if self.m.is_empty() && !params.is_empty()
        {
            self.m = params
                .iter()
                .map(|p| vec![0.0f32; p.value.data.len()])
                .collect();
        }
        assert_eq!(
            self.m.len(),
            params.len(),
            "NdMuon: parameter count changed between steps"
        );
        let MuonConfig {
            lr,
            momentum,
            ns_steps,
            weight_decay,
        } = self.cfg;

        for (k, p) in params.iter_mut().enumerate()
        {
            let g = &grads[p.grad_idx].data;
            let mk = &mut self.m[k];
            for (mj, &gj) in mk.iter_mut().zip(g)
            {
                *mj = momentum * *mj + (1.0 - momentum) * gj;
            }

            let shape = &p.value.shape;
            if shape.len() == 2 && shape[0] >= 2 && shape[1] >= 2
            {
                let (r, c) = (shape[0], shape[1]);
                let o = newton_schulz_orthogonalize(mk, r, c, ns_steps);
                let scale = (r as f32 / c as f32).max(1.0).sqrt();
                for (pv, &ov) in p.value.data.iter_mut().zip(&o)
                {
                    *pv -= lr * (scale * ov + weight_decay * *pv);
                }
            }
            else
            {
                // Non-matrix parameter: momentum SGD.
                for (pv, &mv) in p.value.data.iter_mut().zip(mk.iter())
                {
                    *pv -= lr * (mv + weight_decay * *pv);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::PcgEngine;

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

    /// AdamW's weight decay is **decoupled**: with a zero gradient the adaptive
    /// term vanishes, so `weight_decay > 0` shrinks the parameter toward 0 while
    /// plain Adam (`wd = 0`) leaves it unchanged.
    #[test]
    fn nd_adamw_decouples_weight_decay() {
        let zero_grad = vec![TensorND::new(vec![0.0, 0.0], vec![2])];

        let mut x_wd = TensorND::new(vec![1.0, -2.0], vec![2]);
        let mut opt_wd = NdAdam::with_lr_wd(0.1, 0.5);
        let mut x_plain = TensorND::new(vec![1.0, -2.0], vec![2]);
        let mut opt_plain = NdAdam::with_lr(0.1);
        for _ in 0..20
        {
            opt_wd.step(
                &mut [NdParam {
                    value: &mut x_wd,
                    grad_idx: 0,
                }],
                &zero_grad,
            );
            opt_plain.step(
                &mut [NdParam {
                    value: &mut x_plain,
                    grad_idx: 0,
                }],
                &zero_grad,
            );
        }
        // Plain Adam: untouched by a zero gradient.
        assert_eq!(x_plain.data, vec![1.0, -2.0]);
        // AdamW: each |value| strictly shrank toward 0.
        assert!(x_wd.data[0].abs() < 1.0 && x_wd.data[1].abs() < 2.0);
    }

    /// Lion minimises the same quadratic oracle as Adam, converging to within a
    /// small band of the target (the sign update settles around the optimum).
    #[test]
    fn nd_lion_converges_on_quadratic() {
        let target = [3.0f32, -2.0, 0.5];
        let mut x = TensorND::new(vec![0.0, 0.0, 0.0], vec![3]);
        let mut opt = NdLion::with_lr(0.01);

        for _ in 0..800
        {
            let gd: Vec<f32> = x
                .data
                .iter()
                .zip(&target)
                .map(|(&xi, &ti)| 2.0 * (xi - ti))
                .collect();
            let grads = vec![TensorND::new(gd, vec![3])];
            opt.step(
                &mut [NdParam {
                    value: &mut x,
                    grad_idx: 0,
                }],
                &grads,
            );
        }
        for (&xi, &ti) in x.data.iter().zip(&target)
        {
            assert!(
                (xi - ti).abs() < 0.05,
                "x={:?}, target={:?}",
                x.data,
                target
            );
        }
    }

    /// Lion is bit-for-bit deterministic across runs.
    #[test]
    fn nd_lion_is_deterministic() {
        let run = || -> Vec<f32> {
            let target = [1.0f32, -1.0];
            let mut x = TensorND::new(vec![0.5, 0.5], vec![2]);
            let mut opt = NdLion::with_lr(0.02);
            for _ in 0..100
            {
                let gd: Vec<f32> = x
                    .data
                    .iter()
                    .zip(&target)
                    .map(|(&xi, &ti)| 2.0 * (xi - ti))
                    .collect();
                let grads = vec![TensorND::new(gd, vec![2])];
                opt.step(
                    &mut [NdParam {
                        value: &mut x,
                        grad_idx: 0,
                    }],
                    &grads,
                );
            }
            x.data
        };
        assert_eq!(run(), run());
    }

    /// Newton–Schulz drives the singular values toward 1, so the matrix becomes
    /// **approximately orthogonal**: for a wide `m×n` matrix the deviation
    /// `‖A·Aᵀ − I_m‖_F` collapses versus the (Frobenius-normalised) input.
    #[test]
    fn newton_schulz_orthogonalizes() {
        let (m, n) = (3usize, 5usize);
        let mut rng = PcgEngine::new(2);
        let g: Vec<f32> = (0..m * n).map(|_| rng.float_signed()).collect();

        // Deviation from orthogonality of an m×n matrix: ‖(A·Aᵀ) − I_m‖_F.
        let dev = |a: &[f32]| -> f32 {
            let at = transpose(a, m, n);
            let aat = matmul(a, m, n, &at, m);
            let mut s = 0.0f32;
            for i in 0..m
            {
                for j in 0..m
                {
                    let want = if i == j { 1.0 } else { 0.0 };
                    s += (aat[i * m + j] - want).powi(2);
                }
            }
            s.sqrt()
        };

        let nrm = frob_norm(&g) + 1e-7;
        let normalized: Vec<f32> = g.iter().map(|&v| v / nrm).collect();
        let din = dev(&normalized);
        let dout = dev(&newton_schulz_orthogonalize(&g, m, n, 5));

        // NS pushes singular values into a band near 1 — much more orthogonal
        // than the raw (normalised) matrix, though not exactly `I`.
        assert!(dout < 0.7, "NS output not ~orthogonal: deviation {dout}");
        assert!(
            dout < 0.6 * din,
            "NS did not improve orthogonality: {din} -> {dout}"
        );
    }

    /// Muon reduces a matrix regression loss `‖W − T‖²` (grad `2(W − T)`): the
    /// orthogonalised update is a descent direction, so the loss collapses.
    #[test]
    fn nd_muon_reduces_matrix_loss() {
        let (r, c) = (4usize, 6usize);
        let target: Vec<f32> = (0..r * c).map(|i| (i as f32 * 0.2 - 0.5).sin()).collect();
        let mut w = TensorND::new(vec![0.0f32; r * c], vec![r, c]);
        let mut opt = NdMuon::with_lr(0.1);

        let loss = |w: &TensorND| -> f32 {
            w.data
                .iter()
                .zip(&target)
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum()
        };
        let first = loss(&w);
        for _ in 0..300
        {
            let gd: Vec<f32> = w
                .data
                .iter()
                .zip(&target)
                .map(|(&a, &b)| 2.0 * (a - b))
                .collect();
            let grads = vec![TensorND::new(gd, vec![r, c])];
            let mut params = vec![NdParam {
                value: &mut w,
                grad_idx: 0,
            }];
            opt.step(&mut params, &grads);
        }
        let last = loss(&w);
        assert!(
            last < first * 0.1,
            "Muon did not reduce loss: {first} -> {last}"
        );
    }

    /// Muon is bit-for-bit deterministic across runs.
    #[test]
    fn nd_muon_is_deterministic() {
        let run = || -> Vec<f32> {
            let (r, c) = (3usize, 4usize);
            let target: Vec<f32> = (0..r * c).map(|i| (i as f32 * 0.3).cos()).collect();
            let mut w = TensorND::new(vec![0.1f32; r * c], vec![r, c]);
            let mut opt = NdMuon::with_lr(0.05);
            for _ in 0..50
            {
                let gd: Vec<f32> = w
                    .data
                    .iter()
                    .zip(&target)
                    .map(|(&a, &b)| 2.0 * (a - b))
                    .collect();
                let grads = vec![TensorND::new(gd, vec![r, c])];
                opt.step(
                    &mut [NdParam {
                        value: &mut w,
                        grad_idx: 0,
                    }],
                    &grads,
                );
            }
            w.data
        };
        assert_eq!(run(), run());
    }
}
