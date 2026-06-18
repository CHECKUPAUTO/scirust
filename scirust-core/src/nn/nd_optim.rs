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

/// Hyper-parameters for [`NdLamb`]. [`Default`] is `lr = 1e-3, β1 = 0.9,
/// β2 = 0.999, eps = 1e-6` with no weight decay.
#[derive(Clone, Copy, Debug)]
pub struct LambConfig {
    /// Learning rate.
    pub lr: f32,
    /// First-moment decay.
    pub beta1: f32,
    /// Second-moment decay.
    pub beta2: f32,
    /// Numerical-stability term.
    pub eps: f32,
    /// Decoupled weight decay.
    pub weight_decay: f32,
}

impl Default for LambConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-6,
            weight_decay: 0.0,
        }
    }
}

/// **LAMB** (You et al., ICLR 2020): Adam with a **per-layer trust ratio**. The
/// Adam update direction `r` is rescaled by `‖θ‖ / ‖r‖` (computed per parameter
/// tensor — the "layer"), which lets LAMB use very large batches/learning rates
/// stably. Pure `f32`, fixed order ⇒ **bit-for-bit deterministic**.
pub struct NdLamb {
    cfg: LambConfig,
    t: u64,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl NdLamb {
    /// New optimizer with the given config.
    pub fn new(cfg: LambConfig) -> Self {
        Self {
            cfg,
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    /// LAMB with defaults at learning rate `lr`.
    pub fn with_lr(lr: f32) -> Self {
        Self::new(LambConfig {
            lr,
            ..LambConfig::default()
        })
    }

    /// Steps taken so far.
    pub fn step_count(&self) -> u64 {
        self.t
    }

    /// One LAMB update over `params` (same ordering contract as [`NdAdam`]).
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
            "NdLamb: parameter count changed"
        );
        self.t += 1;
        let LambConfig {
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
            let mk = &mut self.m[k];
            let vk = &mut self.v[k];
            // Adam update direction r, plus the running norms.
            let mut r = vec![0f32; p.value.data.len()];
            let mut w_norm2 = 0f32;
            let mut r_norm2 = 0f32;
            for j in 0..p.value.data.len()
            {
                let gj = g[j];
                mk[j] = beta1 * mk[j] + (1.0 - beta1) * gj;
                vk[j] = beta2 * vk[j] + (1.0 - beta2) * gj * gj;
                let mhat = mk[j] / bc1;
                let vhat = vk[j] / bc2;
                let theta = p.value.data[j];
                r[j] = mhat / (vhat.sqrt() + eps) + weight_decay * theta;
                w_norm2 += theta * theta;
                r_norm2 += r[j] * r[j];
            }
            let (w_norm, r_norm) = (w_norm2.sqrt(), r_norm2.sqrt());
            // Trust ratio (1.0 when a norm vanishes, matching the reference).
            let trust = if w_norm > 0.0 && r_norm > 0.0
            {
                w_norm / r_norm
            }
            else
            {
                1.0
            };
            for (pv, &rj) in p.value.data.iter_mut().zip(&r)
            {
                *pv -= lr * trust * rj;
            }
        }
    }
}

/// Hyper-parameters for [`NdAdan`]. [`Default`] follows the paper:
/// `β1 = 0.02, β2 = 0.08, β3 = 0.01` (these are the *complements* of the usual
/// momentum, i.e. EMAs `x ← (1−β)x + β·new`), `lr = 1e-3, eps = 1e-8`.
#[derive(Clone, Copy, Debug)]
pub struct AdanConfig {
    /// Learning rate.
    pub lr: f32,
    /// Gradient EMA rate.
    pub beta1: f32,
    /// Gradient-difference EMA rate.
    pub beta2: f32,
    /// Squared-term EMA rate.
    pub beta3: f32,
    /// Numerical-stability term.
    pub eps: f32,
    /// Decoupled weight decay (applied multiplicatively).
    pub weight_decay: f32,
}

impl Default for AdanConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.02,
            beta2: 0.08,
            beta3: 0.01,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}

/// **Adan** (Xie et al. 2022): *Adaptive Nesterov Momentum*. Tracks three EMAs —
/// of the gradient `m`, of consecutive gradient differences `v`, and of the
/// squared "look-ahead" gradient `n` — and steps with
/// `θ ← (θ − η⊙(m + (1−β2)v)) / (1 + lr·wd)`, `η = lr/(√n + eps)`. Pure `f32`,
/// fixed order ⇒ **bit-for-bit deterministic**.
pub struct NdAdan {
    cfg: AdanConfig,
    t: u64,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    n: Vec<Vec<f32>>,
    g_prev: Vec<Vec<f32>>,
}

impl NdAdan {
    /// New optimizer with the given config.
    pub fn new(cfg: AdanConfig) -> Self {
        Self {
            cfg,
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
            n: Vec::new(),
            g_prev: Vec::new(),
        }
    }

    /// Adan with defaults at learning rate `lr`.
    pub fn with_lr(lr: f32) -> Self {
        Self::new(AdanConfig {
            lr,
            ..AdanConfig::default()
        })
    }

    /// Steps taken so far.
    pub fn step_count(&self) -> u64 {
        self.t
    }

    /// One Adan update over `params` (same ordering contract as [`NdAdam`]).
    pub fn step(&mut self, params: &mut [NdParam], grads: &[TensorND]) {
        if self.m.is_empty() && !params.is_empty()
        {
            let z = || {
                params
                    .iter()
                    .map(|p| vec![0.0f32; p.value.data.len()])
                    .collect()
            };
            self.m = z();
            self.v = z();
            self.n = z();
            self.g_prev = z();
        }
        assert_eq!(
            self.m.len(),
            params.len(),
            "NdAdan: parameter count changed"
        );
        self.t += 1;
        let AdanConfig {
            lr,
            beta1,
            beta2,
            beta3,
            eps,
            weight_decay,
        } = self.cfg;
        let first = self.t == 1;
        let decay = 1.0 / (1.0 + lr * weight_decay);

        for (k, p) in params.iter_mut().enumerate()
        {
            let g = &grads[p.grad_idx].data;
            let (mk, vk, nk, gp) = (
                &mut self.m[k],
                &mut self.v[k],
                &mut self.n[k],
                &mut self.g_prev[k],
            );
            for j in 0..p.value.data.len()
            {
                let gj = g[j];
                let diff = if first { 0.0 } else { gj - gp[j] };
                mk[j] = (1.0 - beta1) * mk[j] + beta1 * gj;
                vk[j] = (1.0 - beta2) * vk[j] + beta2 * diff;
                let gn = gj + (1.0 - beta2) * diff;
                nk[j] = (1.0 - beta3) * nk[j] + beta3 * gn * gn;
                let eta = lr / (nk[j].sqrt() + eps);
                let upd = eta * (mk[j] + (1.0 - beta2) * vk[j]);
                p.value.data[j] = (p.value.data[j] - upd) * decay;
                gp[j] = gj;
            }
        }
    }
}

/// Reconstruct Adafactor's factored second-moment estimate from its row/column
/// accumulators: `V[i,j] = R[i]·C[j] / ΣR`. This is the rank-1 matrix that
/// preserves the row and column sums of the true squared-gradient matrix — the
/// memory-saving core of the method. For a `rows×cols` weight it needs only
/// `rows + cols` numbers of state instead of `rows·cols`. The reconstruction is
/// **exact** whenever the squared-gradient matrix is itself rank-1.
fn adafactor_factored_v(r: &[f32], c: &[f32]) -> Vec<f32> {
    let cols = c.len();
    let sum_r: f32 = r.iter().sum();
    let inv = if sum_r > 0.0 { 1.0 / sum_r } else { 0.0 };
    let mut v = vec![0.0f32; r.len() * cols];
    for (i, &ri) in r.iter().enumerate()
    {
        for (j, &cj) in c.iter().enumerate()
        {
            v[i * cols + j] = ri * cj * inv;
        }
    }
    v
}

/// Hyper-parameters for [`NdAdafactor`]. Defaults follow Shazeer & Stern (2018):
/// the β2 schedule `β2ₜ = 1 − t^(−decay_rate)` with `decay_rate = 0.8`,
/// `eps1 = 1e-30` (added to the squared gradient), update-RMS clip threshold
/// `d = 1.0`, **no** first moment (`beta1 = 0`), no weight decay. `lr` is an
/// **absolute** step size (equivalent to `relative_step = false,
/// scale_parameter = false` in common implementations).
#[derive(Clone, Copy, Debug)]
pub struct AdafactorConfig {
    /// Learning rate (absolute step size).
    pub lr: f32,
    /// Exponent of the β2 schedule `β2ₜ = 1 − t^(−decay_rate)`.
    pub decay_rate: f32,
    /// Regulariser added to the squared gradient.
    pub eps1: f32,
    /// Update-RMS clipping threshold `d` (the update is scaled so `RMS ≤ d`).
    pub clip_threshold: f32,
    /// Optional first-moment decay (`0` = canonical Adafactor, no momentum).
    pub beta1: f32,
    /// Decoupled weight decay (0 = none).
    pub weight_decay: f32,
}

impl Default for AdafactorConfig {
    fn default() -> Self {
        Self {
            lr: 1e-2,
            decay_rate: 0.8,
            eps1: 1e-30,
            clip_threshold: 1.0,
            beta1: 0.0,
            weight_decay: 0.0,
        }
    }
}

/// Per-parameter Adafactor state. Matrix parameters carry the factored row/column
/// accumulators (`r`, `c`); other parameters keep a full second moment (`v`).
struct AdafactorState {
    is_matrix: bool,
    rows: usize,
    cols: usize,
    r: Vec<f32>,
    c: Vec<f32>,
    v: Vec<f32>,
    m: Vec<f32>,
}

/// **Adafactor** (Shazeer & Stern, ICML 2018): *Adaptive Learning Rates with
/// Sublinear Memory Cost*. For a 2-D weight matrix it does **not** store the full
/// second-moment matrix; instead it keeps per-row and per-column running sums of
/// the squared gradient (`rows + cols` numbers) and reconstructs the second
/// moment as the rank-1 `V[i,j] = R[i]·C[j]/ΣR`. The raw update `G/√V` is then
/// **RMS-clipped** to a fixed threshold `d`. Non-matrix parameters keep the full
/// second moment (like RMSProp). A β2 schedule `β2ₜ = 1 − t^(−decay_rate)`
/// increases the averaging over training. An optional first moment (`beta1`) is
/// supported (off by default). Pure `f32` in a fixed order ⇒ **bit-for-bit
/// deterministic**.
pub struct NdAdafactor {
    cfg: AdafactorConfig,
    t: u64,
    state: Vec<AdafactorState>,
}

impl NdAdafactor {
    /// New optimizer with the given config (no steps taken yet).
    pub fn new(cfg: AdafactorConfig) -> Self {
        Self {
            cfg,
            t: 0,
            state: Vec::new(),
        }
    }

    /// Adafactor with default schedule/clipping at absolute learning rate `lr`.
    pub fn with_lr(lr: f32) -> Self {
        Self::new(AdafactorConfig {
            lr,
            ..AdafactorConfig::default()
        })
    }

    /// Steps taken so far (drives the β2 schedule).
    pub fn step_count(&self) -> u64 {
        self.t
    }

    /// One Adafactor update over `params` (same ordering contract as [`NdAdam`]).
    pub fn step(&mut self, params: &mut [NdParam], grads: &[TensorND]) {
        if self.state.is_empty() && !params.is_empty()
        {
            let beta1 = self.cfg.beta1;
            self.state = params
                .iter()
                .map(|p| {
                    let shape = &p.value.shape;
                    let len = p.value.data.len();
                    let m = if beta1 > 0.0
                    {
                        vec![0.0; len]
                    }
                    else
                    {
                        Vec::new()
                    };
                    if shape.len() == 2 && shape[0] >= 2 && shape[1] >= 2
                    {
                        AdafactorState {
                            is_matrix: true,
                            rows: shape[0],
                            cols: shape[1],
                            r: vec![0.0; shape[0]],
                            c: vec![0.0; shape[1]],
                            v: Vec::new(),
                            m,
                        }
                    }
                    else
                    {
                        AdafactorState {
                            is_matrix: false,
                            rows: 0,
                            cols: 0,
                            r: Vec::new(),
                            c: Vec::new(),
                            v: vec![0.0; len],
                            m,
                        }
                    }
                })
                .collect();
        }
        assert_eq!(
            self.state.len(),
            params.len(),
            "NdAdafactor: parameter count changed between steps"
        );
        self.t += 1;
        let cfg = self.cfg;
        let beta2 = 1.0 - (self.t as f32).powf(-cfg.decay_rate);

        for (k, p) in params.iter_mut().enumerate()
        {
            let g = &grads[p.grad_idx].data;
            assert_eq!(
                g.len(),
                p.value.data.len(),
                "NdAdafactor: grad/param size mismatch at parameter {k}"
            );
            let st = &mut self.state[k];

            // 1) Second-moment estimate V, then the raw update U = G/√V.
            let mut u = vec![0.0f32; g.len()];
            if st.is_matrix
            {
                let cols = st.cols;
                let mut row_sum = vec![0.0f32; st.rows];
                let mut col_sum = vec![0.0f32; cols];
                for (i, row) in g.chunks_exact(cols).enumerate()
                {
                    for (j, &gij) in row.iter().enumerate()
                    {
                        let g2 = gij * gij + cfg.eps1;
                        row_sum[i] += g2;
                        col_sum[j] += g2;
                    }
                }
                for (ri, &rs) in st.r.iter_mut().zip(&row_sum)
                {
                    *ri = beta2 * *ri + (1.0 - beta2) * rs;
                }
                for (ci, &cs) in st.c.iter_mut().zip(&col_sum)
                {
                    *ci = beta2 * *ci + (1.0 - beta2) * cs;
                }
                let v = adafactor_factored_v(&st.r, &st.c);
                for (ui, (&gi, &vi)) in u.iter_mut().zip(g.iter().zip(&v))
                {
                    *ui = gi / vi.sqrt();
                }
            }
            else
            {
                for (vi, &gi) in st.v.iter_mut().zip(g)
                {
                    *vi = beta2 * *vi + (1.0 - beta2) * (gi * gi + cfg.eps1);
                }
                for (ui, (&gi, &vi)) in u.iter_mut().zip(g.iter().zip(&st.v))
                {
                    *ui = gi / vi.sqrt();
                }
            }

            // 2) RMS-clip the update so RMS(U) ≤ clip_threshold.
            let rms = (u.iter().map(|&x| x * x).sum::<f32>() / u.len() as f32).sqrt();
            let denom = (rms / cfg.clip_threshold).max(1.0);
            if denom > 1.0
            {
                for ui in u.iter_mut()
                {
                    *ui /= denom;
                }
            }

            // 3) Optional first-moment smoothing.
            if cfg.beta1 > 0.0
            {
                for (mi, ui) in st.m.iter_mut().zip(u.iter_mut())
                {
                    *mi = cfg.beta1 * *mi + (1.0 - cfg.beta1) * *ui;
                    *ui = *mi;
                }
            }

            // 4) Step with decoupled weight decay.
            for (pv, &ui) in p.value.data.iter_mut().zip(&u)
            {
                *pv -= cfg.lr * (ui + cfg.weight_decay * *pv);
            }
        }
    }
}

/// **Inverse `p`-th root** of a symmetric PSD matrix `a` (`n×n`, row-major) via its
/// Jacobi eigendecomposition: with `A = Q diag(λ) Qᵀ`, returns
/// `(A + eps·I)^(−1/p) = Q diag((λ + eps)^(−1/p)) Qᵀ`. Eigenvalues are read from the
/// diagonal of `Qᵀ A Q` and clamped to `≥ 0` (numerical PSD safety). Used by
/// Shampoo for the `L^(−1/4)` / `R^(−1/4)` preconditioner factors. Deterministic
/// (Jacobi is deterministic; pure `f32` in a fixed order).
fn inverse_pth_root(a: &[f32], n: usize, p: f32, eps: f32) -> Vec<f32> {
    let q = jacobi_eigenvectors(a, n); // columns are eigenvectors
    let qt = transpose(&q, n, n);
    let aq = matmul(a, n, n, &q, n); // A Q
    let d = matmul(&qt, n, n, &aq, n); // Qᵀ A Q (diagonal ≈ eigenvalues)
    // scaled = Q · diag((λ + eps)^(−1/p))  (scale each eigenvector column).
    let mut scaled = vec![0.0f32; n * n];
    for i in 0..n
    {
        let lam = d[i * n + i].max(0.0) + eps;
        let s = lam.powf(-1.0 / p);
        for r in 0..n
        {
            scaled[r * n + i] = q[r * n + i] * s;
        }
    }
    matmul(&scaled, n, n, &qt, n) // (Q diag) Qᵀ
}

/// Hyper-parameters for [`NdShampoo`]. [`Default`] uses `lr = 0.1`, preconditioner
/// EMA decay `beta = 0.95`, root regularisation `eps = 1e-6`, no weight decay, and
/// refreshes the inverse roots every step (`precond_freq = 1`).
#[derive(Clone, Copy, Debug)]
pub struct ShampooConfig {
    /// Learning rate.
    pub lr: f32,
    /// Running-average decay of the preconditioner factors `L = E[GGᵀ]`,
    /// `R = E[GᵀG]` (EMA, as in practical Shampoo / SOAP; the 2018 original
    /// accumulates a plain running sum).
    pub beta: f32,
    /// Root regularisation: the inverse roots use `(L + eps·I)^(−1/4)`.
    pub eps: f32,
    /// Decoupled weight decay (0 = none).
    pub weight_decay: f32,
    /// Refresh interval (in steps) for recomputing the inverse roots.
    pub precond_freq: usize,
}

impl Default for ShampooConfig {
    fn default() -> Self {
        Self {
            lr: 0.1,
            beta: 0.95,
            eps: 1e-6,
            weight_decay: 0.0,
            precond_freq: 1,
        }
    }
}

/// Per-parameter Shampoo state. Matrix parameters carry the Kronecker factors
/// `l`, `r` and their cached inverse fourth roots `il`, `ir`; other parameters
/// fall back to diagonal Adagrad (`g2`).
struct ShampooState {
    is_matrix: bool,
    rows: usize,
    cols: usize,
    l: Vec<f32>,
    r: Vec<f32>,
    il: Vec<f32>,
    ir: Vec<f32>,
    g2: Vec<f32>,
}

/// **Shampoo** (Gupta, Koren & Singer, ICML 2018): a structure-aware
/// preconditioner. For a 2-D weight matrix it maintains the two Kronecker factors
/// `L = E[GGᵀ]` (`m×m`) and `R = E[GᵀG]` (`n×n`) and steps with the preconditioned
/// update `W ← W − lr · L^(−1/4) G R^(−1/4)`. The matrix inverse fourth roots come
/// from `inverse_pth_root` (Jacobi eigendecomposition), and are cached and
/// refreshed every `precond_freq` steps. Non-matrix parameters fall back to
/// diagonal **Adagrad** (Shampoo on a 1-D tensor). Pure `f32` in a fixed order ⇒
/// **bit-for-bit deterministic**.
pub struct NdShampoo {
    cfg: ShampooConfig,
    t: u64,
    state: Vec<ShampooState>,
}

impl NdShampoo {
    /// New optimizer with the given config (no steps taken yet).
    pub fn new(cfg: ShampooConfig) -> Self {
        Self {
            cfg,
            t: 0,
            state: Vec::new(),
        }
    }

    /// Shampoo with default preconditioner settings at learning rate `lr`.
    pub fn with_lr(lr: f32) -> Self {
        Self::new(ShampooConfig {
            lr,
            ..ShampooConfig::default()
        })
    }

    /// Steps taken so far.
    pub fn step_count(&self) -> u64 {
        self.t
    }

    /// One Shampoo update over `params` (same ordering contract as [`NdAdam`]).
    pub fn step(&mut self, params: &mut [NdParam], grads: &[TensorND]) {
        if self.state.is_empty() && !params.is_empty()
        {
            self.state = params
                .iter()
                .map(|p| {
                    let shape = &p.value.shape;
                    let len = p.value.data.len();
                    if shape.len() == 2 && shape[0] >= 2 && shape[1] >= 2
                    {
                        let (m, n) = (shape[0], shape[1]);
                        ShampooState {
                            is_matrix: true,
                            rows: m,
                            cols: n,
                            l: vec![0.0; m * m],
                            r: vec![0.0; n * n],
                            il: Vec::new(),
                            ir: Vec::new(),
                            g2: Vec::new(),
                        }
                    }
                    else
                    {
                        ShampooState {
                            is_matrix: false,
                            rows: 0,
                            cols: 0,
                            l: Vec::new(),
                            r: Vec::new(),
                            il: Vec::new(),
                            ir: Vec::new(),
                            g2: vec![0.0; len],
                        }
                    }
                })
                .collect();
        }
        assert_eq!(
            self.state.len(),
            params.len(),
            "NdShampoo: parameter count changed between steps"
        );
        self.t += 1;
        let t = self.t as usize;
        let cfg = self.cfg;

        for (k, p) in params.iter_mut().enumerate()
        {
            let g = &grads[p.grad_idx].data;
            assert_eq!(
                g.len(),
                p.value.data.len(),
                "NdShampoo: grad/param size mismatch at parameter {k}"
            );
            let st = &mut self.state[k];
            if st.is_matrix
            {
                let (m, n) = (st.rows, st.cols);
                let gt = transpose(g, m, n); // n×m
                let ggt = matmul(g, m, n, &gt, m); // m×m
                let gtg = matmul(&gt, n, m, g, n); // n×n
                let b = cfg.beta;
                for (li, &v) in st.l.iter_mut().zip(&ggt)
                {
                    *li = b * *li + (1.0 - b) * v;
                }
                for (ri, &v) in st.r.iter_mut().zip(&gtg)
                {
                    *ri = b * *ri + (1.0 - b) * v;
                }
                if st.il.is_empty() || t % cfg.precond_freq == 0
                {
                    st.il = inverse_pth_root(&st.l, m, 4.0, cfg.eps);
                    st.ir = inverse_pth_root(&st.r, n, 4.0, cfg.eps);
                }
                // Preconditioned update U = L^(−1/4) G R^(−1/4).
                let ilg = matmul(&st.il, m, m, g, n); // m×n
                let u = matmul(&ilg, m, n, &st.ir, n); // m×n
                for (pv, &uv) in p.value.data.iter_mut().zip(&u)
                {
                    *pv -= cfg.lr * (uv + cfg.weight_decay * *pv);
                }
            }
            else
            {
                // Diagonal Adagrad fallback.
                for (a, &gi) in st.g2.iter_mut().zip(g)
                {
                    *a += gi * gi;
                }
                for (pv, (&gi, &a)) in p.value.data.iter_mut().zip(g.iter().zip(&st.g2))
                {
                    *pv -= cfg.lr * (gi / (a.sqrt() + cfg.eps) + cfg.weight_decay * *pv);
                }
            }
        }
    }
}

/// Hyper-parameters for [`NdSam`]. [`Default`] uses the paper's neighbourhood size
/// `rho = 0.05`, inner SGD step `lr = 0.1`, `eps = 1e-12` for the gradient-norm
/// denominator, and no weight decay.
#[derive(Clone, Copy, Debug)]
pub struct SamConfig {
    /// Neighbourhood radius ρ of the sharpness perturbation.
    pub rho: f32,
    /// Inner (SGD) learning rate applied in the descent phase.
    pub lr: f32,
    /// Numerical-stability term for the global gradient norm.
    pub eps: f32,
    /// Decoupled weight decay applied in the descent phase (0 = none).
    pub weight_decay: f32,
}

impl Default for SamConfig {
    fn default() -> Self {
        Self {
            rho: 0.05,
            lr: 0.1,
            eps: 1e-12,
            weight_decay: 0.0,
        }
    }
}

/// **SAM** — Sharpness-Aware Minimization (Foret et al., ICLR 2021): instead of
/// minimising the loss at `θ`, SAM minimises the **worst-case loss in a ρ-ball**
/// around `θ`, biasing training toward flat minima that tend to generalise better.
/// Each update has two phases:
///
/// 1. [`ascent`](Self::ascent): perturb the parameters to the local worst case
///    `θ + ε`, where `ε = ρ · g / ‖g‖` and `g`, `‖g‖` are the gradient and its
///    **global** L2 norm (over all parameters) at `θ`.
/// 2. [`descent`](Self::descent): restore `θ` and take an SGD step using the
///    gradient evaluated at the perturbed point `θ + ε`.
///
/// Because the two phases need gradients at **two** different points, SAM does not
/// fit the single-gradient [`NdAdam::step`] contract: call `ascent`, recompute the
/// gradient at the perturbed parameters, then `descent`. Pure `f32` in a fixed
/// order ⇒ **bit-for-bit deterministic**.
pub struct NdSam {
    cfg: SamConfig,
    t: u64,
    eps_store: Vec<Vec<f32>>,
}

impl NdSam {
    /// New optimizer with the given config.
    pub fn new(cfg: SamConfig) -> Self {
        Self {
            cfg,
            t: 0,
            eps_store: Vec::new(),
        }
    }

    /// SAM with the given perturbation radius `rho` and inner SGD rate `lr`.
    pub fn with_rho_lr(rho: f32, lr: f32) -> Self {
        Self::new(SamConfig {
            rho,
            lr,
            ..SamConfig::default()
        })
    }

    /// Steps (ascent/descent pairs) completed so far.
    pub fn step_count(&self) -> u64 {
        self.t
    }

    /// Phase 1 — perturb `params` to `θ + ρ·g/‖g‖` (the local worst case) and
    /// remember the perturbation so [`descent`](Self::descent) can undo it. `grads`
    /// is the gradient at the current `θ`.
    pub fn ascent(&mut self, params: &mut [NdParam], grads: &[TensorND]) {
        if self.eps_store.len() != params.len()
        {
            self.eps_store = params
                .iter()
                .map(|p| vec![0.0f32; p.value.data.len()])
                .collect();
        }
        // Global L2 norm of the full gradient (over all parameters).
        let mut sumsq = 0.0f32;
        for p in params.iter()
        {
            for &gi in &grads[p.grad_idx].data
            {
                sumsq += gi * gi;
            }
        }
        let scale = self.cfg.rho / (sumsq.sqrt() + self.cfg.eps);
        for (k, p) in params.iter_mut().enumerate()
        {
            let g = &grads[p.grad_idx].data;
            let store = &mut self.eps_store[k];
            for ((pv, &gi), e_slot) in p.value.data.iter_mut().zip(g.iter()).zip(store.iter_mut())
            {
                let e = scale * gi;
                *e_slot = e;
                *pv += e;
            }
        }
    }

    /// Phase 2 — restore `params` to `θ` (undo the ascent perturbation) and take
    /// the inner SGD step using `grads_perturbed`, the gradient evaluated at the
    /// perturbed parameters `θ + ε`.
    pub fn descent(&mut self, params: &mut [NdParam], grads_perturbed: &[TensorND]) {
        assert_eq!(
            self.eps_store.len(),
            params.len(),
            "NdSam: descent before ascent / parameter count changed"
        );
        let SamConfig {
            lr, weight_decay, ..
        } = self.cfg;
        for (k, p) in params.iter_mut().enumerate()
        {
            let g = &grads_perturbed[p.grad_idx].data;
            let store = &self.eps_store[k];
            for ((pv, &gi), &e) in p.value.data.iter_mut().zip(g.iter()).zip(store.iter())
            {
                *pv -= e; // restore θ
                *pv -= lr * (gi + weight_decay * *pv); // SGD step at the perturbed gradient
            }
        }
        self.t += 1;
    }
}

/// Hyper-parameters for [`NdProdigy`]. [`Default`] follows the paper: base step
/// `γ = 1.0` (Prodigy adapts the effective rate, so 1.0 needs no tuning),
/// `β1 = 0.9`, `β2 = 0.999`, `eps = 1e-8`, and a tiny initial distance estimate
/// `d0 = 1e-6`.
#[derive(Clone, Copy, Debug)]
pub struct ProdigyConfig {
    /// Base step `γ` (Prodigy adapts the effective rate `γ·d`).
    pub lr: f32,
    /// First-moment decay.
    pub beta1: f32,
    /// Second-moment decay.
    pub beta2: f32,
    /// Numerical-stability term.
    pub eps: f32,
    /// Initial distance estimate `d₀` (grown toward `‖x₀ − x*‖`).
    pub d0: f32,
}

impl Default for ProdigyConfig {
    fn default() -> Self {
        Self {
            lr: 1.0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            d0: 1e-6,
        }
    }
}

/// **Prodigy** (Mishchenko & Defazio, 2023): a **parameter-free** Adam — it
/// estimates the distance `d ≈ ‖x₀ − x*‖` to the solution online and uses it as
/// the effective learning rate, so no manual lr tuning is needed. `d` grows from a
/// tiny `d₀` via the running correlation `⟨g, x₀ − x⟩` until it matches the problem
/// scale, then Adam proceeds at the right rate. `d`, the numerator `r`, and the
/// denominator L1-norm are **global** scalars across all parameters. Pure `f32`,
/// fixed order ⇒ **bit-for-bit deterministic**.
pub struct NdProdigy {
    cfg: ProdigyConfig,
    t: u64,
    d: f32,
    r: f32,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    s: Vec<Vec<f32>>,
    x0: Vec<Vec<f32>>,
}

impl NdProdigy {
    /// New optimizer with the given config (no steps taken yet).
    pub fn new(cfg: ProdigyConfig) -> Self {
        Self {
            cfg,
            t: 0,
            d: cfg.d0,
            r: 0.0,
            m: Vec::new(),
            v: Vec::new(),
            s: Vec::new(),
            x0: Vec::new(),
        }
    }

    /// Prodigy with default betas/eps; `lr` is the base step `γ` (default 1.0).
    pub fn with_lr(lr: f32) -> Self {
        Self::new(ProdigyConfig {
            lr,
            ..ProdigyConfig::default()
        })
    }

    /// Steps taken so far.
    pub fn step_count(&self) -> u64 {
        self.t
    }

    /// The current distance estimate `d` (the adapted learning-rate scale).
    pub fn d(&self) -> f32 {
        self.d
    }

    /// One Prodigy update over `params` (same ordering contract as [`NdAdam`]).
    pub fn step(&mut self, params: &mut [NdParam], grads: &[TensorND]) {
        if self.m.is_empty() && !params.is_empty()
        {
            let z = || -> Vec<Vec<f32>> {
                params
                    .iter()
                    .map(|p| vec![0.0f32; p.value.data.len()])
                    .collect()
            };
            self.m = z();
            self.v = z();
            self.s = z();
            self.x0 = params.iter().map(|p| p.value.data.clone()).collect();
            self.d = self.cfg.d0;
        }
        assert_eq!(
            self.m.len(),
            params.len(),
            "NdProdigy: parameter count changed between steps"
        );
        self.t += 1;
        let ProdigyConfig {
            lr,
            beta1,
            beta2,
            eps,
            ..
        } = self.cfg;
        let sb2 = beta2.sqrt();
        let d = self.d; // d_t — used for every update this step
        let d2 = d * d;

        // Pass 1: update m, v, s; accumulate the global numerator and ‖s‖₁.
        let mut r_num = 0.0f32;
        let mut s_l1 = 0.0f32;
        for (k, p) in params.iter().enumerate()
        {
            let g = &grads[p.grad_idx].data;
            let (mk, vk, sk, x0k) = (&mut self.m[k], &mut self.v[k], &mut self.s[k], &self.x0[k]);
            for j in 0..p.value.data.len()
            {
                let gj = g[j];
                mk[j] = beta1 * mk[j] + (1.0 - beta1) * d * gj;
                vk[j] = beta2 * vk[j] + (1.0 - beta2) * d2 * gj * gj;
                sk[j] = sb2 * sk[j] + (1.0 - sb2) * lr * d2 * gj;
                r_num += (1.0 - sb2) * lr * d2 * gj * (x0k[j] - p.value.data[j]);
                s_l1 += sk[j].abs();
            }
        }
        // Global distance update: d_{t+1} = max(d_t, r_{t+1} / ‖s_{t+1}‖₁).
        self.r = sb2 * self.r + r_num;
        let d_next = if s_l1 > 0.0
        {
            (self.r / s_l1).max(d)
        }
        else
        {
            d
        };

        // Pass 2: Adam step at the current d_t.
        for (k, p) in params.iter_mut().enumerate()
        {
            let (mk, vk) = (&self.m[k], &self.v[k]);
            for j in 0..p.value.data.len()
            {
                p.value.data[j] -= lr * d * mk[j] / (vk[j].sqrt() + d * eps);
            }
        }
        self.d = d_next;
    }
}

// ===== GaLore — Gradient Low-Rank Projection (Zhao et al., ICML 2024) ==========

/// Top-`rank` eigenvectors (by eigenvalue, **descending**) of a symmetric
/// `dim×dim` Gram matrix `gram` (row-major), returned as a `dim×rank` row-major
/// matrix whose **columns** form an orthonormal basis of the dominant subspace.
/// Reuses [`jacobi_eigenvectors`] (which returns an unsorted eigenbasis) and ranks
/// the columns by their Rayleigh quotient. This is GaLore's gradient-subspace
/// projector `P`: for a gradient `G`, the rank-`r` projection `P Pᵀ G` is the best
/// rank-`r` approximation of `G` (truncated SVD) when `gram = G Gᵀ`.
pub fn galore_subspace(gram: &[f32], dim: usize, rank: usize) -> Vec<f32> {
    assert_eq!(gram.len(), dim * dim, "galore_subspace: size mismatch");
    let r = rank.min(dim).max(1);
    let q = jacobi_eigenvectors(gram, dim); // columns = eigenvectors (unsorted)
    // Eigenvalue of column j via Rayleigh quotient λ_j = q_jᵀ (gram q_j).
    let mut eig: Vec<(f32, usize)> = (0..dim)
        .map(|j| {
            let mut lam = 0.0f32;
            for a in 0..dim
            {
                let mut gqa = 0.0f32;
                for b in 0..dim
                {
                    gqa += gram[a * dim + b] * q[b * dim + j];
                }
                lam += q[a * dim + j] * gqa;
            }
            (lam, j)
        })
        .collect();
    // Descending eigenvalue; index tie-break keeps it deterministic.
    eig.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut p = vec![0.0f32; dim * r];
    for (col, &(_, j)) in eig.iter().take(r).enumerate()
    {
        for a in 0..dim
        {
            p[a * r + col] = q[a * dim + j];
        }
    }
    p
}

/// `(rows×cols)ᵀ → (cols×rows)`, row-major.
fn galore_transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; rows * cols];
    for i in 0..rows
    {
        for j in 0..cols
        {
            t[j * rows + i] = a[i * cols + j];
        }
    }
    t
}

/// `A(m×k) · B(k×n) → (m×n)`, row-major.
fn galore_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m
    {
        for p in 0..k
        {
            let aip = a[i * k + p];
            if aip == 0.0
            {
                continue;
            }
            for j in 0..n
            {
                c[i * n + j] += aip * b[p * n + j];
            }
        }
    }
    c
}

/// **Gram matrix** `G Gᵀ` (`rows×rows`) of a row-major `rows×cols` matrix.
fn galore_gram(g: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut gram = vec![0.0f32; rows * rows];
    for i in 0..rows
    {
        for j in 0..rows
        {
            let mut s = 0.0f32;
            for c in 0..cols
            {
                s += g[i * cols + c] * g[j * cols + c];
            }
            gram[i * rows + j] = s;
        }
    }
    gram
}

/// GaLore hyper-parameters (Zhao et al., ICML 2024, arXiv:2403.03507). Adam runs
/// inside a low-rank projection of the gradient, so its moment buffers shrink from
/// `m×n` to `rank×max(m,n)`. [`Default`] is `lr = 1e-3`, standard Adam betas,
/// `rank = 4`, projector refresh every `update_gap = 50` steps, `scale = 1`.
#[derive(Clone, Copy, Debug)]
pub struct GaloreConfig {
    /// Learning rate.
    pub lr: f32,
    /// First-moment decay.
    pub beta1: f32,
    /// Second-moment decay.
    pub beta2: f32,
    /// Numerical epsilon.
    pub eps: f32,
    /// Projection rank `r` (subspace dimension).
    pub rank: usize,
    /// Steps a projector is reused before being recomputed from the gradient.
    pub update_gap: usize,
    /// GaLore α: rescales the projected-back update.
    pub scale: f32,
}

impl Default for GaloreConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            rank: 4,
            update_gap: 50,
            scale: 1.0,
        }
    }
}

/// Per-parameter GaLore state: dense (plain Adam) for vectors, low-rank for
/// matrices.
enum GaloreState {
    /// Non-matrix parameter: full-size Adam moments.
    Dense { m: Vec<f32>, v: Vec<f32> },
    /// Matrix parameter: projector `P` (`a×r`) plus Adam moments in the projected
    /// `r×b` space, where `(a, b) = (min, max)` of the parameter's two dims and
    /// `transpose` records whether we operate on `Gᵀ` (when `rows > cols`).
    LowRank {
        transpose: bool,
        a: usize,
        b: usize,
        r: usize,
        proj: Vec<f32>,
        m: Vec<f32>,
        v: Vec<f32>,
    },
}

/// **GaLore** — *Gradient Low-Rank Projection* (Zhao et al., ICML 2024). For a
/// matrix parameter the gradient `G` is projected onto its own dominant rank-`r`
/// subspace `P` (top-`r` left singular vectors, recomputed every `update_gap`
/// steps), Adam runs on the small projected gradient `Pᵀ G`, and the resulting
/// update is mapped back with `P`. The optimizer state thus shrinks from `m×n` to
/// `rank×max(m,n)` while the update direction stays in the gradient's most
/// informative subspace. Vector parameters fall back to plain Adam. Pure `f32` in
/// fixed order ⇒ bit-for-bit deterministic.
pub struct NdGalore {
    cfg: GaloreConfig,
    t: u64,
    state: Vec<GaloreState>,
}

impl NdGalore {
    /// New optimizer with the given config (no steps taken yet).
    pub fn new(cfg: GaloreConfig) -> Self {
        Self {
            cfg,
            t: 0,
            state: Vec::new(),
        }
    }

    /// GaLore at learning rate `lr` and projection `rank` (other fields default).
    pub fn with_lr_rank(lr: f32, rank: usize) -> Self {
        Self::new(GaloreConfig {
            lr,
            rank,
            ..GaloreConfig::default()
        })
    }

    /// Number of steps taken so far.
    pub fn step_count(&self) -> u64 {
        self.t
    }

    /// The projected-space moment-buffer length for parameter `i` (`rank×max(m,n)`
    /// for a matrix), exposed so tests can confirm the memory reduction.
    pub fn state_len(&self, i: usize) -> usize {
        match &self.state[i]
        {
            GaloreState::Dense { m, .. } => m.len(),
            GaloreState::LowRank { m, .. } => m.len(),
        }
    }

    /// One GaLore update over `params` (same order on every call).
    pub fn step(&mut self, params: &mut [NdParam], grads: &[TensorND]) {
        if self.state.is_empty() && !params.is_empty()
        {
            self.state = params
                .iter()
                .map(|p| {
                    if p.value.shape.len() == 2
                    {
                        let (rows, cols) = (p.value.shape[0], p.value.shape[1]);
                        let transpose = rows > cols;
                        let (a, b) = if transpose
                        {
                            (cols, rows)
                        }
                        else
                        {
                            (rows, cols)
                        };
                        let r = self.cfg.rank.min(a).max(1);
                        GaloreState::LowRank {
                            transpose,
                            a,
                            b,
                            r,
                            proj: Vec::new(),
                            m: vec![0.0f32; r * b],
                            v: vec![0.0f32; r * b],
                        }
                    }
                    else
                    {
                        GaloreState::Dense {
                            m: vec![0.0f32; p.value.data.len()],
                            v: vec![0.0f32; p.value.data.len()],
                        }
                    }
                })
                .collect();
        }
        assert_eq!(
            self.state.len(),
            params.len(),
            "NdGalore: parameter count changed between steps"
        );
        self.t += 1;
        let GaloreConfig {
            lr,
            beta1,
            beta2,
            eps,
            update_gap,
            scale,
            ..
        } = self.cfg;
        let bc1 = 1.0 - beta1.powi(self.t as i32);
        let bc2 = 1.0 - beta2.powi(self.t as i32);
        let refresh = (self.t - 1) % update_gap.max(1) as u64 == 0;

        for (k, p) in params.iter_mut().enumerate()
        {
            let g_full = &grads[p.grad_idx].data;
            match &mut self.state[k]
            {
                GaloreState::Dense { m, v } =>
                {
                    for j in 0..p.value.data.len()
                    {
                        let gj = g_full[j];
                        m[j] = beta1 * m[j] + (1.0 - beta1) * gj;
                        v[j] = beta2 * v[j] + (1.0 - beta2) * gj * gj;
                        let mhat = m[j] / bc1;
                        let vhat = v[j] / bc2;
                        p.value.data[j] -= lr * mhat / (vhat.sqrt() + eps);
                    }
                },
                GaloreState::LowRank {
                    transpose,
                    a,
                    b,
                    r,
                    proj,
                    m,
                    v,
                } =>
                {
                    let (a, b, r) = (*a, *b, *r);
                    // Orient the gradient so it is a×b (a = min dim).
                    let g = if *transpose
                    {
                        galore_transpose(g_full, b, a) // stored rows×cols = b×a
                    }
                    else
                    {
                        g_full.clone()
                    };
                    // (Re)compute the projector P (a×r) from the gradient subspace.
                    if refresh || proj.is_empty()
                    {
                        let gram = galore_gram(&g, a, b);
                        *proj = galore_subspace(&gram, a, r);
                    }
                    // Project: R = Pᵀ G  (r×b).
                    let pt = galore_transpose(proj, a, r); // r×a
                    let rgrad = galore_matmul(&pt, &g, r, a, b);
                    // Adam in the projected space → per-element update U (r×b).
                    let mut u = vec![0.0f32; r * b];
                    for j in 0..r * b
                    {
                        let gj = rgrad[j];
                        m[j] = beta1 * m[j] + (1.0 - beta1) * gj;
                        v[j] = beta2 * v[j] + (1.0 - beta2) * gj * gj;
                        let mhat = m[j] / bc1;
                        let vhat = v[j] / bc2;
                        u[j] = mhat / (vhat.sqrt() + eps);
                    }
                    // Project back: ΔW = scale · P U  (a×b), re-orient, subtract.
                    let dw = galore_matmul(proj, &u, a, r, b);
                    let dw = if *transpose
                    {
                        galore_transpose(&dw, a, b) // a×b → b×a = rows×cols
                    }
                    else
                    {
                        dw
                    };
                    for (pj, &dj) in p.value.data.iter_mut().zip(&dw)
                    {
                        *pj -= lr * scale * dj;
                    }
                },
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

    /// Run `steps` of an optimizer on the quadratic `Σ(x−target)²` and return `x`.
    fn quad_run<F: FnMut(&mut [NdParam], &[TensorND])>(
        target: &[f32],
        steps: usize,
        mut step: F,
    ) -> Vec<f32> {
        let mut x = TensorND::new(vec![0.0; target.len()], vec![target.len()]);
        for _ in 0..steps
        {
            let gd: Vec<f32> = x
                .data
                .iter()
                .zip(target)
                .map(|(&xi, &ti)| 2.0 * (xi - ti))
                .collect();
            let grads = vec![TensorND::new(gd, vec![target.len()])];
            step(
                &mut [NdParam {
                    value: &mut x,
                    grad_idx: 0,
                }],
                &grads,
            );
        }
        x.data
    }

    /// LAMB minimises the quadratic oracle, and is bit-for-bit deterministic.
    /// LAMB's per-layer trust ratio makes the update norm ≈ `lr·‖θ‖` independent
    /// of the residual, so — like sign-based methods — it settles in a small
    /// band (∝ `lr`) around the optimum rather than exactly; we use a small `lr`
    /// to keep that band tight.
    #[test]
    fn nd_lamb_converges_and_is_deterministic() {
        let target = [1.0f32, -0.8, 0.5];
        let run = || {
            let mut opt = NdLamb::with_lr(0.01);
            quad_run(&target, 1500, |p, g| opt.step(p, g))
        };
        let x = run();
        for (xi, ti) in x.iter().zip(&target)
        {
            assert!((xi - ti).abs() < 0.05, "LAMB off: {xi} vs {ti}");
        }
        assert_eq!(run(), x);
    }

    /// Adan minimises the quadratic oracle and is bit-for-bit deterministic.
    #[test]
    fn nd_adan_converges_and_is_deterministic() {
        let target = [3.0f32, -2.0, 0.5];
        let run = || {
            let mut opt = NdAdan::with_lr(0.1);
            quad_run(&target, 800, |p, g| opt.step(p, g))
        };
        let x = run();
        for (xi, ti) in x.iter().zip(&target)
        {
            assert!((xi - ti).abs() < 0.1, "Adan off: {xi} vs {ti}");
        }
        assert_eq!(run(), x);
    }

    /// Adafactor's factored reconstruction `V[i,j] = R[i]·C[j]/ΣR` is **exact**
    /// when the squared-gradient matrix is itself rank-1 (the best case for the
    /// sublinear-memory trick): build `G²` as an outer product `a⊗b`, feed its
    /// row and column sums in, and recover `G²` to floating-point tolerance.
    #[test]
    fn adafactor_factored_v_reconstructs_rank1() {
        let (rows, cols) = (3usize, 4usize);
        let a = [0.5f32, 1.5, 2.0];
        let b = [0.3f32, 0.7, 1.1, 0.2];
        let mut g2 = vec![0.0f32; rows * cols];
        let mut r = vec![0.0f32; rows];
        let mut c = vec![0.0f32; cols];
        for (i, &ai) in a.iter().enumerate()
        {
            for (j, &bj) in b.iter().enumerate()
            {
                let val = ai * bj;
                g2[i * cols + j] = val;
                r[i] += val;
                c[j] += val;
            }
        }
        let v = adafactor_factored_v(&r, &c);
        for (vi, gi) in v.iter().zip(&g2)
        {
            assert!((vi - gi).abs() < 1e-5, "factored V off: {vi} vs {gi}");
        }
    }

    /// Adafactor (vector path = RMSProp-with-schedule + update clipping) minimises
    /// the quadratic oracle to within a small band of the target, and is
    /// bit-for-bit deterministic. Like other RMS-/sign-scaled methods the clipped
    /// update settles in a band ∝ `lr`, so we use a small `lr`.
    #[test]
    fn nd_adafactor_converges_and_is_deterministic() {
        let target = [1.0f32, -0.8, 0.5];
        let run = || {
            let mut opt = NdAdafactor::with_lr(0.02);
            quad_run(&target, 2000, |p, g| opt.step(p, g))
        };
        let x = run();
        for (xi, ti) in x.iter().zip(&target)
        {
            assert!((xi - ti).abs() < 0.05, "Adafactor off: {xi} vs {ti}");
        }
        assert_eq!(run(), x);
        assert_eq!(
            {
                let mut o = NdAdafactor::with_lr(0.02);
                let _ = quad_run(&target, 3, |p, g| o.step(p, g));
                o.step_count()
            },
            3
        );
    }

    /// Adafactor's **factored** matrix path reduces a convex matrix quadratic
    /// `½‖W − T‖²` (gradient `W − T`): the rank-1 second-moment reconstruction
    /// still yields a descent direction, so the loss collapses. Exercises the
    /// row/column accumulators on a 2-D parameter.
    #[test]
    fn nd_adafactor_matrix_path_reduces_loss() {
        let (rows, cols) = (4usize, 3usize);
        let target: Vec<f32> = (0..rows * cols)
            .map(|k| (k as f32 * 0.5 - 2.0).sin())
            .collect();
        let mut w = TensorND::new(vec![0.0; rows * cols], vec![rows, cols]);
        let mut opt = NdAdafactor::with_lr(0.05);
        let loss = |w: &TensorND| -> f32 {
            w.data
                .iter()
                .zip(&target)
                .map(|(&a, &b)| 0.5 * (a - b) * (a - b))
                .sum()
        };
        let first = loss(&w);
        for _ in 0..1500
        {
            let gd: Vec<f32> = w.data.iter().zip(&target).map(|(&a, &b)| a - b).collect();
            let grads = vec![TensorND::new(gd, vec![rows, cols])];
            opt.step(
                &mut [NdParam {
                    value: &mut w,
                    grad_idx: 0,
                }],
                &grads,
            );
        }
        let last = loss(&w);
        assert!(
            last < first * 0.05,
            "Adafactor matrix path did not reduce loss: {first} -> {last}"
        );
    }

    /// `inverse_pth_root` matches the eigendecomposition definition: for a
    /// symmetric positive-definite `A`, `A^(−1/2) · A^(−1/2) · A ≈ I`.
    #[test]
    fn inverse_pth_root_matches_eigen_definition() {
        let n = 4;
        let mut rng = PcgEngine::new(7);
        // A = MᵀM + I : symmetric positive-definite.
        let m: Vec<f32> = (0..n * n).map(|_| rng.float_signed()).collect();
        let mt = transpose(&m, n, n);
        let mut a = matmul(&mt, n, n, &m, n);
        for i in 0..n
        {
            a[i * n + i] += 1.0;
        }
        let inv_half = inverse_pth_root(&a, n, 2.0, 0.0);
        let half_sq = matmul(&inv_half, n, n, &inv_half, n); // A^(−1)
        let prod = matmul(&half_sq, n, n, &a, n); // A^(−1) · A ≈ I
        for i in 0..n
        {
            for j in 0..n
            {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (prod[i * n + j] - want).abs() < 1e-3,
                    "A^(-1/2)² · A off at ({i},{j}): {}",
                    prod[i * n + j]
                );
            }
        }
    }

    /// Shampoo (Kronecker-preconditioned update `L^(−1/4) G R^(−1/4)`) minimises a
    /// convex matrix quadratic `½‖W − T‖²` (gradient `W − T`), driving every entry
    /// to the target, and is bit-for-bit deterministic across runs.
    #[test]
    fn nd_shampoo_converges_and_is_deterministic() {
        let (rows, cols) = (4usize, 3usize);
        let target: Vec<f32> = (0..rows * cols)
            .map(|k| (k as f32 * 0.4 - 1.0).sin())
            .collect();
        let run = || {
            let mut w = TensorND::new(vec![0.0; rows * cols], vec![rows, cols]);
            let mut opt = NdShampoo::with_lr(0.2);
            for _ in 0..1500
            {
                let gd: Vec<f32> = w.data.iter().zip(&target).map(|(&a, &b)| a - b).collect();
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
        let w = run();
        for (wi, ti) in w.iter().zip(&target)
        {
            assert!(
                (wi - ti).abs() < 0.1,
                "Shampoo did not converge: {wi} vs {ti}"
            );
        }
        assert_eq!(run(), w);
    }

    /// Shampoo's diagonal (Adagrad) fallback for a non-matrix parameter also
    /// minimises the quadratic oracle to within a small band.
    #[test]
    fn nd_shampoo_vector_fallback_converges() {
        let target = [1.0f32, -0.8, 0.5];
        let mut opt = NdShampoo::with_lr(0.3);
        let x = quad_run(&target, 4000, |p, g| opt.step(p, g));
        for (xi, ti) in x.iter().zip(&target)
        {
            assert!(
                (xi - ti).abs() < 0.05,
                "Shampoo Adagrad fallback off: {xi} vs {ti}"
            );
        }
    }

    /// SAM's ascent perturbs the parameters by exactly `ε = ρ·g/‖g‖`: the
    /// displacement is parallel to the gradient and has L2 norm `ρ`.
    #[test]
    fn nd_sam_ascent_perturbs_by_rho_along_gradient() {
        let mut x = TensorND::new(vec![1.0, 2.0, -2.0], vec![3]);
        let theta0 = x.data.clone();
        let g = vec![TensorND::new(vec![3.0, 0.0, 4.0], vec![3])]; // ‖g‖ = 5
        let rho = 0.1;
        let mut opt = NdSam::with_rho_lr(rho, 0.1);
        opt.ascent(
            &mut [NdParam {
                value: &mut x,
                grad_idx: 0,
            }],
            &g,
        );
        // ε = ρ·g/‖g‖ = 0.1·[3,0,4]/5 = [0.06, 0, 0.08].
        let expected = [0.06f32, 0.0, 0.08];
        let mut dnorm2 = 0.0f32;
        for ((xi, t0), e) in x.data.iter().zip(&theta0).zip(&expected)
        {
            let disp = xi - t0;
            assert!((disp - e).abs() < 1e-6, "perturbation off: {disp} vs {e}");
            dnorm2 += disp * disp;
        }
        assert!(
            (dnorm2.sqrt() - rho).abs() < 1e-6,
            "‖ε‖ = {} ≠ ρ = {rho}",
            dnorm2.sqrt()
        );
    }

    /// SAM (perturb, then SGD at the perturbed point) minimises the quadratic
    /// oracle to within a small band of the target, and is bit-for-bit
    /// deterministic. The fixed-size normalised perturbation leaves a residual
    /// band ∝ `lr·ρ`, so we use a small `rho`.
    #[test]
    fn nd_sam_converges_and_is_deterministic() {
        let target = [1.0f32, -0.8, 0.5];
        let run = || {
            let mut x = TensorND::new(vec![0.0; 3], vec![3]);
            let mut opt = NdSam::with_rho_lr(0.02, 0.1);
            for _ in 0..600
            {
                // Gradient of ½‖x−t‖² at θ.
                let g0: Vec<f32> = x
                    .data
                    .iter()
                    .zip(&target)
                    .map(|(&xi, &ti)| xi - ti)
                    .collect();
                opt.ascent(
                    &mut [NdParam {
                        value: &mut x,
                        grad_idx: 0,
                    }],
                    &[TensorND::new(g0, vec![3])],
                );
                // Gradient at the perturbed point θ+ε.
                let g1: Vec<f32> = x
                    .data
                    .iter()
                    .zip(&target)
                    .map(|(&xi, &ti)| xi - ti)
                    .collect();
                opt.descent(
                    &mut [NdParam {
                        value: &mut x,
                        grad_idx: 0,
                    }],
                    &[TensorND::new(g1, vec![3])],
                );
            }
            x.data
        };
        let x = run();
        for (xi, ti) in x.iter().zip(&target)
        {
            assert!((xi - ti).abs() < 0.05, "SAM off: {xi} vs {ti}");
        }
        assert_eq!(run(), x);
    }

    /// **Prodigy is parameter-free.** From a tiny `d₀ = 1e-6` it grows its distance
    /// estimate `d` to the problem scale `‖x₀ − x*‖` and substantially reduces the
    /// quadratic loss — no learning-rate tuning. (On a deterministic quadratic the
    /// Adam-style step settles in a band ∝ `γ·d`, so we use `γ = 0.1`.) Bit-for-bit
    /// deterministic.
    #[test]
    fn nd_prodigy_adapts_distance_and_reduces_loss() {
        let target = [1.0f32, -0.8, 0.5];
        let dist = (1.0f32 + 0.64 + 0.25).sqrt(); // ‖x₀ − x*‖ ≈ 1.375 (x₀ = 0)
        let run = || -> (Vec<f32>, f32) {
            let mut x = TensorND::new(vec![0.0; 3], vec![3]);
            let mut opt = NdProdigy::new(ProdigyConfig {
                lr: 0.1,
                ..ProdigyConfig::default()
            });
            for _ in 0..3000
            {
                let gd: Vec<f32> = x
                    .data
                    .iter()
                    .zip(&target)
                    .map(|(&xi, &ti)| 2.0 * (xi - ti))
                    .collect();
                opt.step(
                    &mut [NdParam {
                        value: &mut x,
                        grad_idx: 0,
                    }],
                    &[TensorND::new(gd, vec![3])],
                );
            }
            (x.data.clone(), opt.d())
        };
        let (x, d) = run();
        let loss: f32 = x
            .iter()
            .zip(&target)
            .map(|(&xi, &ti)| (xi - ti) * (xi - ti))
            .sum();
        let init_loss: f32 = target.iter().map(|&t| t * t).sum();
        assert!(
            loss < 0.2 * init_loss,
            "Prodigy loss {loss} not << initial {init_loss}"
        );
        assert!(
            d > 0.2 * dist && d < 5.0 * dist,
            "Prodigy d={d} not near the distance {dist}"
        );
        let (x2, d2) = run();
        assert_eq!(x, x2);
        assert_eq!(d.to_bits(), d2.to_bits());
    }

    // ----- GaLore (#48) -----------------------------------------------------

    fn frob2(a: &[f32]) -> f32 {
        a.iter().map(|&x| x * x).sum()
    }

    /// GaLore's projector is **orthonormal** (`PᵀP = I`) and `P Pᵀ` is the
    /// **orthogonal** projector onto the top-`r` subspace: the reconstruction error
    /// obeys the Pythagorean identity `‖G − PPᵀG‖² = ‖G‖² − ‖PᵀG‖²`, it **shrinks**
    /// as the rank grows, and it vanishes once the rank reaches `rank(G)`.
    #[test]
    fn galore_subspace_orthonormal_and_projection_optimal() {
        let mut rng = PcgEngine::new(11);
        let (rows, cols) = (5usize, 3usize); // rank(G) = 3
        let g: Vec<f32> = (0..rows * cols).map(|_| rng.float_signed()).collect();
        let gram = galore_gram(&g, rows, cols);
        let total = frob2(&g);
        let mut prev_err = f32::INFINITY;
        for r in 1..=3
        {
            let p = galore_subspace(&gram, rows, r);
            // Orthonormality: PᵀP = I_r.
            let pt = galore_transpose(&p, rows, r);
            let gram_p = galore_matmul(&pt, &p, r, rows, r);
            for i in 0..r
            {
                for j in 0..r
                {
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (gram_p[i * r + j] - want).abs() < 1e-3,
                        "PᵀP not identity at ({i},{j})"
                    );
                }
            }
            // Pythagoras: ‖G − PPᵀG‖² = ‖G‖² − ‖PᵀG‖².
            let ptg = galore_matmul(&pt, &g, r, rows, cols);
            let recon = galore_matmul(&p, &ptg, rows, r, cols);
            let err2: f32 = g.iter().zip(&recon).map(|(&a, &b)| (a - b) * (a - b)).sum();
            assert!(
                (err2 - (total - frob2(&ptg))).abs() < 1e-2,
                "rank {r}: projection not orthogonal ({err2} vs {})",
                total - frob2(&ptg)
            );
            assert!(err2 < prev_err + 1e-4, "error did not shrink at rank {r}");
            prev_err = err2;
        }
        assert!(
            prev_err < 1e-3,
            "full-rank reconstruction not exact: {prev_err}"
        );
    }

    /// A genuinely low-rank gradient is reconstructed **exactly** at the matching
    /// rank, and under-ranking leaves a residual — so the rank is doing real work.
    #[test]
    fn galore_preserves_low_rank_gradient() {
        let mut rng = PcgEngine::new(5);
        let (rows, cols, rho) = (6usize, 4usize, 2usize);
        let a: Vec<f32> = (0..rows * rho).map(|_| rng.float_signed()).collect();
        let b: Vec<f32> = (0..rho * cols).map(|_| rng.float_signed()).collect();
        let g = galore_matmul(&a, &b, rows, rho, cols); // exactly rank ρ
        let gram = galore_gram(&g, rows, cols);
        let recon_err = |r: usize| -> f32 {
            let p = galore_subspace(&gram, rows, r);
            let pt = galore_transpose(&p, rows, r);
            let ptg = galore_matmul(&pt, &g, r, rows, cols);
            let recon = galore_matmul(&p, &ptg, rows, r, cols);
            g.iter().zip(&recon).map(|(&x, &y)| (x - y) * (x - y)).sum()
        };
        assert!(recon_err(rho) < 1e-3, "rank-ρ projection not exact");
        assert!(recon_err(1) > 1e-2, "rank-1 should leave a residual");
    }

    /// Run GaLore on the matrix objective `½‖W − W*‖²` and return `(W, state_len)`.
    fn galore_matrix_run(
        target: &[f32],
        rows: usize,
        cols: usize,
        rank: usize,
        gap: usize,
        steps: usize,
    ) -> (Vec<f32>, usize) {
        let mut w = TensorND::new(vec![0.0; rows * cols], vec![rows, cols]);
        let mut opt = NdGalore::new(GaloreConfig {
            lr: 0.1,
            rank,
            update_gap: gap,
            ..GaloreConfig::default()
        });
        for _ in 0..steps
        {
            let gd: Vec<f32> = w
                .data
                .iter()
                .zip(target)
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
        let slen = opt.state_len(0);
        (w.data, slen)
    }

    /// **GaLore converges on a low-rank target with a compressed optimizer state.**
    /// For a rank-2 target `W*`, GaLore with `rank = 2` reaches it (the gradient
    /// stays in `W*`'s subspace), its moment buffer is `2×4` not `4×4` (the memory
    /// win), under-ranking (`rank = 1`) cannot reach it, and the whole run is
    /// bit-for-bit deterministic.
    #[test]
    fn nd_galore_converges_on_low_rank_target() {
        let mut rng = PcgEngine::new(3);
        let (n, rho) = (4usize, 2usize);
        let u: Vec<f32> = (0..n * rho).map(|_| rng.float_signed()).collect();
        let v: Vec<f32> = (0..rho * n).map(|_| rng.float_signed()).collect();
        let target = galore_matmul(&u, &v, n, rho, n); // rank-2 W*

        // Fixed projector (huge gap) isolates the convergence claim.
        let (w, slen) = galore_matrix_run(&target, n, n, rho, 100_000, 2000);
        for (wi, ti) in w.iter().zip(&target)
        {
            assert!((wi - ti).abs() < 0.05, "GaLore off: {wi} vs {ti}");
        }
        assert_eq!(slen, rho * n, "state not compressed to rank×n");
        assert!(slen < n * n, "no memory reduction");

        // Under-ranking leaves a residual.
        let (w1, _) = galore_matrix_run(&target, n, n, 1, 100_000, 2000);
        let res: f32 = w1
            .iter()
            .zip(&target)
            .map(|(&wi, &ti)| (wi - ti) * (wi - ti))
            .sum();
        assert!(res > 1e-2, "rank-1 GaLore should not reach a rank-2 target");

        // Determinism (including the refresh path at the default gap).
        let again = galore_matrix_run(&target, n, n, rho, 50, 300);
        let once = galore_matrix_run(&target, n, n, rho, 50, 300);
        assert_eq!(again.0, once.0);
    }

    /// Vector parameters fall back to plain Adam and still minimise the quadratic.
    #[test]
    fn nd_galore_vector_param_uses_adam() {
        let target = [1.0f32, -0.5, 0.3];
        let run = || {
            let mut opt = NdGalore::with_lr_rank(0.1, 2);
            quad_run(&target, 800, |p, g| opt.step(p, g))
        };
        let x = run();
        for (xi, ti) in x.iter().zip(&target)
        {
            assert!((xi - ti).abs() < 0.05, "GaLore-Adam off: {xi} vs {ti}");
        }
        assert_eq!(run(), x);
    }
}
