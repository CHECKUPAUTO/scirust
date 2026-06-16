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

/// Hyper-parameters for [`NdScheduleFree`]. [`Default`] is `lr = 1.0,
/// beta = 0.9` with no weight decay.
#[derive(Clone, Copy, Debug)]
pub struct ScheduleFreeConfig {
    /// Learning rate (Schedule-Free tolerates a constant, large LR).
    pub lr: f32,
    /// Interpolation between the base sequence `z` and the average `x`.
    pub beta: f32,
    /// Decoupled weight decay (applied at the evaluation point).
    pub weight_decay: f32,
}

impl Default for ScheduleFreeConfig {
    fn default() -> Self {
        Self {
            lr: 1.0,
            beta: 0.9,
            weight_decay: 0.0,
        }
    }
}

/// **Schedule-Free** optimization (Defazio et al., *The Road Less Scheduled*,
/// 2024; AlgoPerf self-tuning winner): no learning-rate schedule. Three points
/// per parameter — a base sequence `z` (gradient descent), a Polyak average `x`
/// (the **evaluation point**), and an interpolation `y = (1−β)z + βx` where the
/// gradient is taken. With a constant LR the averaging weight is `1/(t+1)`
/// (uniform Polyak–Ruppert averaging).
///
/// Contract: the parameter tensors hold `y` (so the forward/backward computes
/// the gradient at `y`); [`Self::step`] consumes that gradient and writes the
/// next `y` back. Call [`Self::write_eval_point`] to load `x` for
/// evaluation/deployment. Pure `f32` in a fixed order ⇒ **deterministic**.
pub struct NdScheduleFree {
    cfg: ScheduleFreeConfig,
    t: u64,
    z: Vec<Vec<f32>>,
    x: Vec<Vec<f32>>,
}

impl NdScheduleFree {
    /// New optimizer with the given config.
    pub fn new(cfg: ScheduleFreeConfig) -> Self {
        Self {
            cfg,
            t: 0,
            z: Vec::new(),
            x: Vec::new(),
        }
    }

    /// Schedule-Free at learning rate `lr` (default `beta`, no weight decay).
    pub fn with_lr(lr: f32) -> Self {
        Self::new(ScheduleFreeConfig {
            lr,
            ..ScheduleFreeConfig::default()
        })
    }

    /// One update: read the gradient (taken at `y`), advance `z`/`x`, and write
    /// the next `y` into the parameter tensors.
    pub fn step(&mut self, params: &mut [NdParam], grads: &[TensorND]) {
        if self.z.is_empty() && !params.is_empty()
        {
            // z₁ = x₁ = θ₀ (the current parameter values, which equal y₁).
            self.z = params.iter().map(|p| p.value.data.clone()).collect();
            self.x = self.z.clone();
        }
        assert_eq!(
            self.z.len(),
            params.len(),
            "NdScheduleFree: parameter count changed between steps"
        );
        self.t += 1;
        let ScheduleFreeConfig {
            lr,
            beta,
            weight_decay,
        } = self.cfg;
        let c = 1.0 / (self.t as f32 + 1.0); // averaging weight for constant LR

        for (k, p) in params.iter_mut().enumerate()
        {
            let g = &grads[p.grad_idx].data;
            let zk = &mut self.z[k];
            let xk = &mut self.x[k];
            for j in 0..p.value.data.len()
            {
                let yj = p.value.data[j]; // gradient was taken here
                let geff = g[j] + weight_decay * yj;
                zk[j] -= lr * geff;
                xk[j] = (1.0 - c) * xk[j] + c * zk[j];
                p.value.data[j] = (1.0 - beta) * zk[j] + beta * xk[j];
            }
        }
    }

    /// Load the evaluation point `x` (the Polyak average) into the parameter
    /// tensors — call before measuring/deploying the model.
    pub fn write_eval_point(&self, params: &mut [NdParam]) {
        for (k, p) in params.iter_mut().enumerate()
        {
            p.value.data.copy_from_slice(&self.x[k]);
        }
    }
}

/// Hyper-parameters for [`NdAdEMAMix`]. [`Default`] is `lr = 1e-3, β1 = 0.9,
/// β2 = 0.999, β3 = 0.9999, α = 5.0, eps = 1e-8`.
#[derive(Clone, Copy, Debug)]
pub struct AdEMAMixConfig {
    /// Learning rate.
    pub lr: f32,
    /// Fast first-moment decay (Adam-like).
    pub beta1: f32,
    /// Second-moment decay.
    pub beta2: f32,
    /// **Slow** first-moment decay (long gradient memory, e.g. 0.9999).
    pub beta3: f32,
    /// Mixing weight of the slow EMA.
    pub alpha: f32,
    /// Numerical-stability term.
    pub eps: f32,
    /// Decoupled weight decay.
    pub weight_decay: f32,
}

impl Default for AdEMAMixConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            beta3: 0.9999,
            alpha: 5.0,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}

/// **AdEMAMix** (Pagliardini et al., Apple, 2024): Adam with **two** first-moment
/// EMAs — a fast one (`β1`) and a slow one (`β3`, long gradient memory) — mixed
/// by `α`. The update is `(m̂₁ + α·m₂) / (√v̂ + eps)`, with bias correction on the
/// fast moment and the second moment. Pure `f32` in a fixed order ⇒
/// **bit-for-bit deterministic**. (The paper warms `α`/`β3` up over training;
/// here they are constant — supply pre-warmed values if desired.)
pub struct NdAdEMAMix {
    cfg: AdEMAMixConfig,
    t: u64,
    m1: Vec<Vec<f32>>,
    m2: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl NdAdEMAMix {
    /// New optimizer with the given config.
    pub fn new(cfg: AdEMAMixConfig) -> Self {
        Self {
            cfg,
            t: 0,
            m1: Vec::new(),
            m2: Vec::new(),
            v: Vec::new(),
        }
    }

    /// AdEMAMix with default betas/alpha at learning rate `lr`.
    pub fn with_lr(lr: f32) -> Self {
        Self::new(AdEMAMixConfig {
            lr,
            ..AdEMAMixConfig::default()
        })
    }

    /// One AdEMAMix update over `params` (same ordering contract as [`NdAdam`]).
    pub fn step(&mut self, params: &mut [NdParam], grads: &[TensorND]) {
        if self.m1.is_empty() && !params.is_empty()
        {
            let zeros: Vec<Vec<f32>> = params
                .iter()
                .map(|p| vec![0.0f32; p.value.data.len()])
                .collect();
            self.m1 = zeros.clone();
            self.m2 = zeros.clone();
            self.v = zeros;
        }
        assert_eq!(
            self.m1.len(),
            params.len(),
            "NdAdEMAMix: parameter count changed between steps"
        );
        self.t += 1;
        let AdEMAMixConfig {
            lr,
            beta1,
            beta2,
            beta3,
            alpha,
            eps,
            weight_decay,
        } = self.cfg;
        let bc1 = 1.0 - beta1.powi(self.t as i32);
        let bc2 = 1.0 - beta2.powi(self.t as i32);

        for (k, p) in params.iter_mut().enumerate()
        {
            let g = &grads[p.grad_idx].data;
            let m1k = &mut self.m1[k];
            let m2k = &mut self.m2[k];
            let vk = &mut self.v[k];
            for j in 0..p.value.data.len()
            {
                let gj = g[j];
                m1k[j] = beta1 * m1k[j] + (1.0 - beta1) * gj;
                m2k[j] = beta3 * m2k[j] + (1.0 - beta3) * gj;
                vk[j] = beta2 * vk[j] + (1.0 - beta2) * gj * gj;
                let m1hat = m1k[j] / bc1;
                let vhat = vk[j] / bc2;
                let theta = p.value.data[j];
                p.value.data[j] = theta
                    - lr * ((m1hat + alpha * m2k[j]) / (vhat.sqrt() + eps) + weight_decay * theta);
            }
        }
    }
}

/// Eigenvectors of a symmetric `n×n` matrix `a` (row-major) by **cyclic Jacobi
/// rotations**. Deterministic: fixed sweep order, fixed convergence threshold and
/// max sweeps. Returns the orthogonal `Q` (`n×n`, row-major) whose **columns** are
/// eigenvectors, so `Qᵀ A Q` is (numerically) diagonal. Used by SOAP for the
/// Shampoo eigenbasis; `a` is assumed symmetric (only the symmetric part matters).
pub fn jacobi_eigenvectors(a: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(a.len(), n * n, "jacobi: size mismatch");
    let mut m = a.to_vec(); // diagonalised in place
    let mut v = vec![0f32; n * n];
    for i in 0..n
    {
        v[i * n + i] = 1.0;
    }
    if n == 1
    {
        return v;
    }
    for _sweep in 0..60
    {
        let mut off = 0f32;
        for p in 0..n
        {
            for q in (p + 1)..n
            {
                off += m[p * n + q].abs();
            }
        }
        if off <= 1e-12
        {
            break;
        }
        for p in 0..n
        {
            for q in (p + 1)..n
            {
                let apq = m[p * n + q];
                if apq.abs() <= 1e-20
                {
                    continue;
                }
                let (app, aqq) = (m[p * n + p], m[q * n + q]);
                let tau = (aqq - app) / (2.0 * apq);
                // tan of the rotation (numerically stable branch by sign of tau).
                let t = if tau >= 0.0
                {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                }
                else
                {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                // A ← Jᵀ A J : rotate columns p,q, then rows p,q.
                for k in 0..n
                {
                    let (akp, akq) = (m[k * n + p], m[k * n + q]);
                    m[k * n + p] = c * akp - s * akq;
                    m[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n
                {
                    let (apk, aqk) = (m[p * n + k], m[q * n + k]);
                    m[p * n + k] = c * apk - s * aqk;
                    m[q * n + k] = s * apk + c * aqk;
                }
                // V ← V J (accumulate eigenvectors).
                for k in 0..n
                {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    v
}

/// Hyper-parameters for [`NdSoap`]. [`Default`] mirrors the paper: Adam
/// `β1 = 0.9, β2 = 0.999`, preconditioner decay `shampoo_beta = 0.95`, eigenbasis
/// refreshed every `precond_freq = 10` steps.
#[derive(Clone, Copy, Debug)]
pub struct SoapConfig {
    /// Learning rate.
    pub lr: f32,
    /// Adam first-moment decay.
    pub beta1: f32,
    /// Adam second-moment decay.
    pub beta2: f32,
    /// Running-average decay of the Shampoo factors `L = E[GGᵀ]`, `R = E[GᵀG]`.
    pub shampoo_beta: f32,
    /// Numerical-stability term.
    pub eps: f32,
    /// Decoupled weight decay.
    pub weight_decay: f32,
    /// Eigenbasis refresh interval (in steps).
    pub precond_freq: usize,
}

impl Default for SoapConfig {
    fn default() -> Self {
        Self {
            lr: 3e-3,
            beta1: 0.9,
            beta2: 0.999,
            shampoo_beta: 0.95,
            eps: 1e-8,
            weight_decay: 0.0,
            precond_freq: 10,
        }
    }
}

/// Per-parameter SOAP state. Matrix parameters carry the Shampoo factors and
/// eigenbases; other parameters fall back to plain Adam (moments only).
struct SoapState {
    is_matrix: bool,
    rows: usize,
    cols: usize,
    l: Vec<f32>,
    r: Vec<f32>,
    ql: Vec<f32>,
    qr: Vec<f32>,
    mom1: Vec<f32>,
    mom2: Vec<f32>,
}

/// **SOAP** (Vyas et al. 2024): *Improving and Stabilizing Shampoo using Adam*.
/// For each 2-D weight matrix it maintains Shampoo's preconditioner factors
/// `L = E[GGᵀ]` and `R = E[GᵀG]`, rotates the gradient into their **eigenbasis**
/// (`Ĝ = Q_Lᵀ G Q_R`), runs **Adam** there, and rotates the update back
/// (`U = Q_L Û Q_Rᵀ`). The eigenbasis is refreshed every `precond_freq` steps; the
/// Adam moments are rotated into the new basis on refresh (the second-moment
/// rotation is kept non-negative, as the moment is a variance proxy).
/// Non-matrix parameters use plain Adam. Pure `f32`, fixed order ⇒
/// **bit-for-bit deterministic**.
pub struct NdSoap {
    cfg: SoapConfig,
    t: u64,
    state: Vec<SoapState>,
}

impl NdSoap {
    /// New optimizer with the given config (no steps taken yet).
    pub fn new(cfg: SoapConfig) -> Self {
        Self {
            cfg,
            t: 0,
            state: Vec::new(),
        }
    }

    /// SOAP with default hyper-parameters at learning rate `lr`.
    pub fn with_lr(lr: f32) -> Self {
        Self::new(SoapConfig {
            lr,
            ..SoapConfig::default()
        })
    }

    /// Steps taken so far.
    pub fn step_count(&self) -> u64 {
        self.t
    }

    /// One SOAP update over `params` (same ordering contract as [`NdAdam`]).
    pub fn step(&mut self, params: &mut [NdParam], grads: &[TensorND]) {
        if self.state.is_empty() && !params.is_empty()
        {
            self.state = params
                .iter()
                .map(|p| {
                    let shape = &p.value.shape;
                    if shape.len() == 2 && shape[0] >= 2 && shape[1] >= 2
                    {
                        let (m, n) = (shape[0], shape[1]);
                        SoapState {
                            is_matrix: true,
                            rows: m,
                            cols: n,
                            l: vec![0.0; m * m],
                            r: vec![0.0; n * n],
                            ql: Vec::new(),
                            qr: Vec::new(),
                            mom1: vec![0.0; m * n],
                            mom2: vec![0.0; m * n],
                        }
                    }
                    else
                    {
                        let len = p.value.data.len();
                        SoapState {
                            is_matrix: false,
                            rows: 0,
                            cols: 0,
                            l: Vec::new(),
                            r: Vec::new(),
                            ql: Vec::new(),
                            qr: Vec::new(),
                            mom1: vec![0.0; len],
                            mom2: vec![0.0; len],
                        }
                    }
                })
                .collect();
        }
        assert_eq!(
            self.state.len(),
            params.len(),
            "NdSoap: parameter count changed between steps"
        );
        self.t += 1;
        let t_now = self.t;
        let cfg = self.cfg;
        let bc1 = 1.0 - cfg.beta1.powi(t_now as i32);
        let bc2 = 1.0 - cfg.beta2.powi(t_now as i32);

        for (k, p) in params.iter_mut().enumerate()
        {
            let g = &grads[p.grad_idx].data;
            let st = &mut self.state[k];
            if !st.is_matrix
            {
                for (i, &gi) in g.iter().enumerate()
                {
                    st.mom1[i] = cfg.beta1 * st.mom1[i] + (1.0 - cfg.beta1) * gi;
                    st.mom2[i] = cfg.beta2 * st.mom2[i] + (1.0 - cfg.beta2) * gi * gi;
                    let mhat = st.mom1[i] / bc1;
                    let vhat = st.mom2[i] / bc2;
                    let theta = p.value.data[i];
                    p.value.data[i] = theta
                        - cfg.lr * (mhat / (vhat.sqrt() + cfg.eps) + cfg.weight_decay * theta);
                }
                continue;
            }

            let (m, n) = (st.rows, st.cols);
            let gt = transpose(g, m, n); // n×m
            // First touch: seed factors and eigenbases from this gradient.
            if st.ql.is_empty()
            {
                st.l = matmul(g, m, n, &gt, m); // GGᵀ  (m×m)
                st.r = matmul(&gt, n, m, g, n); // GᵀG  (n×n)
                st.ql = jacobi_eigenvectors(&st.l, m);
                st.qr = jacobi_eigenvectors(&st.r, n);
            }

            // Rotate gradient into the eigenbasis: Ĝ = Q_Lᵀ G Q_R.
            let qlt = transpose(&st.ql, m, m);
            let gl = matmul(&qlt, m, m, g, n); // m×n
            let ghat = matmul(&gl, m, n, &st.qr, n); // m×n

            // Adam in the eigenbasis.
            let mut upd = vec![0f32; m * n];
            for i in 0..m * n
            {
                st.mom1[i] = cfg.beta1 * st.mom1[i] + (1.0 - cfg.beta1) * ghat[i];
                st.mom2[i] = cfg.beta2 * st.mom2[i] + (1.0 - cfg.beta2) * ghat[i] * ghat[i];
                let mhat = st.mom1[i] / bc1;
                let vhat = st.mom2[i] / bc2;
                upd[i] = mhat / (vhat.sqrt() + cfg.eps);
            }

            // Rotate the update back: U = Q_L Û Q_Rᵀ.
            let qrt = transpose(&st.qr, n, n);
            let ul = matmul(&st.ql, m, m, &upd, n); // m×n
            let u = matmul(&ul, m, n, &qrt, n); // m×n
            for (pv, &uv) in p.value.data.iter_mut().zip(&u)
            {
                *pv -= cfg.lr * (uv + cfg.weight_decay * *pv);
            }

            // Update Shampoo factors (running average).
            let ggt = matmul(g, m, n, &gt, m);
            let gtg = matmul(&gt, n, m, g, n);
            let sb = cfg.shampoo_beta;
            for (li, &gi) in st.l.iter_mut().zip(&ggt)
            {
                *li = sb * *li + (1.0 - sb) * gi;
            }
            for (ri, &gi) in st.r.iter_mut().zip(&gtg)
            {
                *ri = sb * *ri + (1.0 - sb) * gi;
            }

            // Periodically refresh the eigenbasis and rotate the moments into it.
            if t_now as usize % cfg.precond_freq == 0
            {
                let ql_new = jacobi_eigenvectors(&st.l, m);
                let qr_new = jacobi_eigenvectors(&st.r, n);
                let rot_l = matmul(&transpose(&ql_new, m, m), m, m, &st.ql, m); // m×m
                let rot_r = matmul(&transpose(&st.qr, n, n), n, n, &qr_new, n); // n×n
                let rotate = |x: &[f32]| -> Vec<f32> {
                    let t1 = matmul(&rot_l, m, m, x, n); // m×n
                    matmul(&t1, m, n, &rot_r, n) // m×n
                };
                st.mom1 = rotate(&st.mom1);
                st.mom2 = rotate(&st.mom2).iter().map(|x| x.abs()).collect();
                st.ql = ql_new;
                st.qr = qr_new;
            }
        }
    }
}

/// Hyper-parameters for [`NdLookahead`]. [`Default`] is the paper's `k = 5`
/// (sync interval) and `alpha = 0.5` (slow-weight step).
#[derive(Clone, Copy, Debug)]
pub struct LookaheadConfig {
    /// Number of fast (inner-optimizer) steps between slow-weight syncs.
    pub k: usize,
    /// Slow-weight interpolation factor in `(0, 1]`.
    pub alpha: f32,
}

impl Default for LookaheadConfig {
    fn default() -> Self {
        Self { k: 5, alpha: 0.5 }
    }
}

/// **Lookahead** (Zhang et al., NeurIPS 2019): wrap a fast inner optimizer
/// (here [`NdAdam`]) with a set of **slow weights**. The inner optimizer takes
/// `k` ordinary steps; then the slow weights move a fraction `alpha` toward the
/// fast weights and the fast weights are reset to the slow ones
/// (`φ ← φ + α(θ − φ); θ ← φ`). This "k steps forward, 1 step back" reduces
/// variance and is robust to the inner learning rate. Pure `f32` in a fixed
/// order ⇒ **bit-for-bit deterministic**.
pub struct NdLookahead {
    base: NdAdam,
    cfg: LookaheadConfig,
    t: u64,
    slow: Vec<Vec<f32>>,
}

impl NdLookahead {
    /// Wrap an existing [`NdAdam`] with the given Lookahead config.
    pub fn new(base: NdAdam, cfg: LookaheadConfig) -> Self {
        Self {
            base,
            cfg,
            t: 0,
            slow: Vec::new(),
        }
    }

    /// Lookahead over Adam at learning rate `lr` (default `k = 5, alpha = 0.5`).
    pub fn with_lr(lr: f32) -> Self {
        Self::new(NdAdam::with_lr(lr), LookaheadConfig::default())
    }

    /// Steps taken so far.
    pub fn step_count(&self) -> u64 {
        self.t
    }

    /// One Lookahead update over `params` (same ordering contract as [`NdAdam`]).
    pub fn step(&mut self, params: &mut [NdParam], grads: &[TensorND]) {
        if self.slow.is_empty() && !params.is_empty()
        {
            // Slow weights start at the current parameters.
            self.slow = params.iter().map(|p| p.value.data.clone()).collect();
        }
        assert_eq!(
            self.slow.len(),
            params.len(),
            "NdLookahead: parameter count changed between steps"
        );
        // Fast (inner) update in place.
        self.base.step(params, grads);
        self.t += 1;
        // Every k steps: pull the slow weights toward fast, then reset fast to slow.
        if self.t as usize % self.cfg.k == 0
        {
            let alpha = self.cfg.alpha;
            for (k, p) in params.iter_mut().enumerate()
            {
                let slow = &mut self.slow[k];
                for (s, pv) in slow.iter_mut().zip(p.value.data.iter_mut())
                {
                    *s += alpha * (*pv - *s);
                    *pv = *s;
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

    /// Schedule-Free minimises the quadratic oracle: the **evaluation point**
    /// `x` converges to the target. The gradient is taken at `y` (the parameter
    /// the optimizer keeps live), as the contract requires.
    #[test]
    fn nd_schedule_free_converges_on_quadratic() {
        let target = [3.0f32, -2.0, 0.5];
        let mut p = TensorND::new(vec![0.0, 0.0, 0.0], vec![3]);
        let mut opt = NdScheduleFree::with_lr(0.2);

        for _ in 0..1000
        {
            // gradient of Σ(y−t)² at the current y (= p).
            let gd: Vec<f32> = p
                .data
                .iter()
                .zip(&target)
                .map(|(&yi, &ti)| 2.0 * (yi - ti))
                .collect();
            let grads = vec![TensorND::new(gd, vec![3])];
            opt.step(
                &mut [NdParam {
                    value: &mut p,
                    grad_idx: 0,
                }],
                &grads,
            );
        }
        // Load the averaged evaluation point and check it reached the target.
        opt.write_eval_point(&mut [NdParam {
            value: &mut p,
            grad_idx: 0,
        }]);
        for (&xi, &ti) in p.data.iter().zip(&target)
        {
            assert!(
                (xi - ti).abs() < 0.02,
                "x={:?}, target={:?}",
                p.data,
                target
            );
        }
    }

    /// Schedule-Free is bit-for-bit deterministic across runs.
    #[test]
    fn nd_schedule_free_is_deterministic() {
        let run = || -> Vec<f32> {
            let target = [1.0f32, -1.0];
            let mut p = TensorND::new(vec![0.5, 0.5], vec![2]);
            let mut opt = NdScheduleFree::with_lr(0.1);
            for _ in 0..200
            {
                let gd: Vec<f32> = p
                    .data
                    .iter()
                    .zip(&target)
                    .map(|(&yi, &ti)| 2.0 * (yi - ti))
                    .collect();
                let grads = vec![TensorND::new(gd, vec![2])];
                opt.step(
                    &mut [NdParam {
                        value: &mut p,
                        grad_idx: 0,
                    }],
                    &grads,
                );
            }
            opt.write_eval_point(&mut [NdParam {
                value: &mut p,
                grad_idx: 0,
            }]);
            p.data
        };
        assert_eq!(run(), run());
    }

    /// AdEMAMix minimises the quadratic oracle, converging to within a small
    /// band of the target (the long-memory slow EMA leaves a little residual
    /// oscillation on a deterministic problem — it is built for stochastic
    /// training).
    #[test]
    fn nd_ademamix_converges_on_quadratic() {
        let target = [3.0f32, -2.0, 0.5];
        let mut x = TensorND::new(vec![0.0, 0.0, 0.0], vec![3]);
        let mut opt = NdAdEMAMix::with_lr(0.02);
        for _ in 0..2000
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
            assert!((xi - ti).abs() < 0.1, "x={:?}, target={:?}", x.data, target);
        }
    }

    /// AdEMAMix is bit-for-bit deterministic across runs.
    #[test]
    fn nd_ademamix_is_deterministic() {
        let run = || -> Vec<f32> {
            let target = [1.0f32, -1.0];
            let mut x = TensorND::new(vec![0.5, 0.5], vec![2]);
            let mut opt = NdAdEMAMix::with_lr(0.02);
            for _ in 0..200
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

    /// The Jacobi eigensolver returns an **orthogonal** `Q` that **diagonalises**
    /// a symmetric matrix: `QᵀQ ≈ I`, `QᵀAQ` is diagonal, and `Q·diag·Qᵀ ≈ A`.
    #[test]
    fn jacobi_diagonalizes_symmetric() {
        let n = 5;
        let mut rng = PcgEngine::new(3);
        // A = MᵀM is symmetric PSD with non-trivial off-diagonal structure.
        let m: Vec<f32> = (0..n * n).map(|_| rng.float_signed()).collect();
        let mt = transpose(&m, n, n);
        let a = matmul(&mt, n, n, &m, n);

        let q = jacobi_eigenvectors(&a, n);
        let qt = transpose(&q, n, n);

        // Orthogonality: QᵀQ ≈ I.
        let qtq = matmul(&qt, n, n, &q, n);
        for i in 0..n
        {
            for j in 0..n
            {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qtq[i * n + j] - want).abs() < 1e-4,
                    "QᵀQ not identity at ({i},{j}): {}",
                    qtq[i * n + j]
                );
            }
        }
        // Diagonalisation: off-diagonal of QᵀAQ ≈ 0.
        let aq = matmul(&a, n, n, &q, n);
        let d = matmul(&qt, n, n, &aq, n);
        for i in 0..n
        {
            for j in 0..n
            {
                if i != j
                {
                    assert!(
                        d[i * n + j].abs() < 1e-3,
                        "off-diag ({i},{j}) = {}",
                        d[i * n + j]
                    );
                }
            }
        }
        // Reconstruction: Q·(QᵀAQ)·Qᵀ ≈ A.
        let qd = matmul(&q, n, n, &d, n);
        let recon = matmul(&qd, n, n, &qt, n);
        for (r, a0) in recon.iter().zip(&a)
        {
            assert!((r - a0).abs() < 1e-3, "reconstruction off: {r} vs {a0}");
        }
    }

    /// SOAP (Adam in the Shampoo eigenbasis) minimises a convex matrix quadratic
    /// `½‖W − T‖²` (gradient `W − T`), driving every entry to the target. Exercises
    /// the eigenbasis refresh + moment rotation (`precond_freq = 2`).
    #[test]
    fn nd_soap_converges_on_matrix_quadratic() {
        let (rows, cols) = (4usize, 3usize);
        let target: Vec<f32> = (0..rows * cols)
            .map(|k| (k as f32 * 0.5 - 2.0).sin())
            .collect();
        let mut w = TensorND::new(vec![0.0; rows * cols], vec![rows, cols]);
        let mut opt = NdSoap::new(SoapConfig {
            lr: 0.05,
            precond_freq: 2,
            ..SoapConfig::default()
        });
        for _ in 0..1000
        {
            let gd: Vec<f32> = w
                .data
                .iter()
                .zip(&target)
                .map(|(&wi, &ti)| wi - ti)
                .collect();
            let grads = vec![TensorND::new(gd, vec![rows, cols])];
            opt.step(
                &mut [NdParam {
                    value: &mut w,
                    grad_idx: 0,
                }],
                &grads,
            );
        }
        for (wi, ti) in w.data.iter().zip(&target)
        {
            assert!((wi - ti).abs() < 0.1, "SOAP did not converge: {wi} vs {ti}");
        }
    }

    /// SOAP is deterministic: two independent runs are bit-for-bit identical.
    #[test]
    fn nd_soap_is_deterministic() {
        let run = || -> Vec<f32> {
            let (rows, cols) = (3usize, 3usize);
            let target: Vec<f32> = (0..rows * cols).map(|k| k as f32 * 0.1).collect();
            let mut w = TensorND::new(vec![0.2; rows * cols], vec![rows, cols]);
            let mut opt = NdSoap::with_lr(0.03);
            for _ in 0..120
            {
                let gd: Vec<f32> = w
                    .data
                    .iter()
                    .zip(&target)
                    .map(|(&wi, &ti)| wi - ti)
                    .collect();
                let grads = vec![TensorND::new(gd, vec![rows, cols])];
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

    /// Lookahead-over-Adam minimises the same quadratic oracle, converging to the
    /// target (the slow weights track the fast trajectory).
    #[test]
    fn nd_lookahead_converges_on_quadratic() {
        let target = [3.0f32, -2.0, 0.5];
        let mut x = TensorND::new(vec![0.0, 0.0, 0.0], vec![3]);
        let mut opt = NdLookahead::with_lr(0.05);
        for _ in 0..400
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
        for (xi, ti) in x.data.iter().zip(&target)
        {
            assert!((xi - ti).abs() < 0.05, "Lookahead off: {xi} vs {ti}");
        }
        assert_eq!(opt.step_count(), 400);
    }

    /// Lookahead is deterministic: two identical runs are bit-for-bit equal.
    #[test]
    fn nd_lookahead_is_deterministic() {
        let run = || -> Vec<f32> {
            let target = [1.0f32, -1.0];
            let mut x = TensorND::new(vec![0.5, 0.5], vec![2]);
            let mut opt =
                NdLookahead::new(NdAdam::with_lr(0.05), LookaheadConfig { k: 3, alpha: 0.5 });
            for _ in 0..120
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
}
