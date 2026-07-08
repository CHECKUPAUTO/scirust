//! Real wgpu compute path (feature `wgpu`).
//!
//! Provides a general `f32` GEMM as a WGSL compute shader executed through wgpu
//! (Vulkan/Metal/DX12/GL):
//!
//! ```text
//! C = alpha * op(A) * op(B) + beta * C
//! ```
//!
//! with optional transposes — the exact contract of
//! [`scirust_core::autodiff::reverse::GpuEngine`], so the same kernel powers
//! both the standalone [`crate::WgpuBackend`] and the autograd-tape engine
//! ([`crate::WgpuEngine`], see `engine.rs`).
//!
//! Results are validated against the deterministic [`crate::CpuBackend`] oracle
//! within a documented floating-point tolerance (GPU accumulation order is not
//! bit-identical to the scalar CPU path). The path is exercised in CI on a
//! software Vulkan adapter (Mesa *lavapipe*), satisfying "no claim without a
//! test" without physical GPU hardware. See `docs/GPU.md` (P2.2).

use std::borrow::Cow;
use std::sync::mpsc;

use wgpu::util::DeviceExt;

use crate::{BackendError, BackendResult};

/// Max workgroups per dispatch dimension under `downlevel_defaults` (and the
/// Vulkan spec's `maxComputeWorkGroupCount`): **65535**. lavapipe does not
/// enforce it, but real hardware (e.g. the Jetson Thor) rejects any dispatch
/// dimension above it. Flat-indexed kernels grid-stride (`i += num_workgroups.x
/// * 64`), so capping the launch here still covers tensors of any size — the
/// loop picks up the remainder. Needed at 350M scale, where e.g. the tied
/// embedding (`32768×1024` = 33.5M elements) would otherwise want 524288
/// workgroups and the logits (`128×32768`) 65536.
const MAX_WORKGROUPS_PER_DIM: u32 = 65535;

/// Workgroup count for a flat, grid-stride 1-D kernel over `threads` elements at
/// workgroup size 64, capped to [`MAX_WORKGROUPS_PER_DIM`].
#[inline]
fn flat_workgroups(threads: u32) -> u32 {
    threads.div_ceil(64).min(MAX_WORKGROUPS_PER_DIM)
}

/// General WGSL GEMM: `C = alpha·op(A)·op(B) + beta·C`, row-major, one
/// invocation per output cell. `op(A)` is `m×k`, `op(B)` is `k×n`, `C` is `m×n`;
/// `ta`/`tb` flag whether the *stored* `a`/`b` is the transpose of `op`.
const GEMM_WGSL: &str = r#"
struct P { m: u32, k: u32, n: u32, ta: u32, tb: u32, alpha: f32, beta: f32, _pad: u32, };

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             p: P;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    if (i >= p.m || j >= p.n) { return; }
    var acc: f32 = 0.0;
    for (var q: u32 = 0u; q < p.k; q = q + 1u) {
        var av: f32;
        var bv: f32;
        if (p.ta == 1u) { av = a[q * p.m + i]; } else { av = a[i * p.k + q]; }
        if (p.tb == 1u) { bv = b[j * p.k + q]; } else { bv = b[q * p.n + j]; }
        acc = acc + av * bv;
    }
    let idx = i * p.n + j;
    c[idx] = p.alpha * acc + p.beta * c[idx];
}
"#;

/// Elementwise kernel: `op` selects `0=add`, `1=mul` (binary, `a` and `b`),
/// `2=relu` (unary, `b` ignored), or `3=swiglu` (binary: `silu(a)·b`, the
/// SwiGLU gate — `silu(x) = x·σ(x)`). One invocation per element.
const EW_WGSL: &str = r#"
struct P { n: u32, op: u32, _p0: u32, _p1: u32, };

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let stride = nwg.x * 64u;
    var i = gid.x;
    while (i < p.n) {
        if (p.op == 0u) { c[i] = a[i] + b[i]; }
        else if (p.op == 1u) { c[i] = a[i] * b[i]; }
        else if (p.op == 2u) { c[i] = max(a[i], 0.0); }
        else { c[i] = (a[i] / (1.0 + exp(-a[i]))) * b[i]; }
        i = i + stride;
    }
}
"#;

/// Token embedding gather: row `i` of the output is row `tokens[i]` of the
/// `vocab × d` `table`. One invocation per output element. Out-of-range tokens
/// are clamped to the last row (defensive; callers pass valid ids). The CPU
/// contract is [`crate::ops::cpu_embed`].
const EMBED_WGSL: &str = r#"
struct P { rows: u32, d: u32, vocab: u32, _p0: u32, };

@group(0) @binding(0) var<storage, read>       tokens: array<u32>;
@group(0) @binding(1) var<storage, read>       table: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let n = p.rows * p.d;
    let stride = nwg.x * 64u;
    var i = gid.x;
    while (i < n) {
        let row = i / p.d;
        let col = i % p.d;
        let tok = min(tokens[row], p.vocab - 1u);
        out[i] = table[tok * p.d + col];
        i = i + stride;
    }
}
"#;

/// Row-wise RMSNorm: `x / sqrt(mean(x²) + eps) · weight`, one invocation per
/// row over the row's `cols` elements; `weight` is a `cols`-length gain vector.
/// `eps` rides through the `u32` uniform as raw bits. The CPU contract is
/// [`crate::ops::cpu_rms_norm`].
const RMSNORM_WGSL: &str = r#"
struct P { rows: u32, cols: u32, eps_bits: u32, _p0: u32, };

@group(0) @binding(0) var<storage, read>       inp: array<f32>;
@group(0) @binding(1) var<storage, read>       weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= p.rows) { return; }
    if (p.cols == 0u) { return; }
    let base = row * p.cols;
    var ss = 0.0;
    for (var j: u32 = 0u; j < p.cols; j = j + 1u) { let x = inp[base + j]; ss = ss + x * x; }
    let rms = sqrt(ss / f32(p.cols) + bitcast<f32>(p.eps_bits));
    for (var j: u32 = 0u; j < p.cols; j = j + 1u) { out[base + j] = inp[base + j] / rms * weight[j]; }
}
"#;

/// Backward of row-wise softmax. Given the forward output `y = softmax(x)` and
/// the upstream grad `dy`, one invocation per row computes the Jacobian-vector
/// product `dx = y ⊙ (dy − Σⱼ dyⱼ·yⱼ)`. The CPU contract is
/// [`crate::ops::cpu_softmax_backward`].
const SOFTMAX_BWD_WGSL: &str = r#"
struct P { rows: u32, cols: u32, _p0: u32, _p1: u32, };

@group(0) @binding(0) var<storage, read>       y:  array<f32>;
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read_write> dx: array<f32>;
@group(0) @binding(3) var<uniform>             p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= p.rows) { return; }
    if (p.cols == 0u) { return; }
    let base = row * p.cols;
    var s = 0.0;
    for (var j: u32 = 0u; j < p.cols; j = j + 1u) { s = s + dy[base + j] * y[base + j]; }
    for (var j: u32 = 0u; j < p.cols; j = j + 1u) { dx[base + j] = y[base + j] * (dy[base + j] - s); }
}
"#;

/// Backward of the SwiGLU gate `c = silu(a) ⊙ b` (`silu(x)=x·σ(x)`). One
/// invocation per element writes both gradients into a single `2n` output:
/// `dab[i] = da = dc · silu'(a) · b` and `dab[n+i] = db = dc · silu(a)`, where
/// `silu'(a) = σ(a)·(1 + a·(1−σ(a)))`. Packing both into one buffer keeps this
/// at **4 storage buffers** (a, b, dc, dab) — within the portable
/// `downlevel_defaults` limit that a five-buffer layout would exceed on real
/// hardware. The CPU contract is [`crate::ops::cpu_swiglu_backward`].
const SWIGLU_BWD_WGSL: &str = r#"
struct P { n: u32, _p0: u32, _p1: u32, _p2: u32, };

@group(0) @binding(0) var<storage, read>       a:   array<f32>;
@group(0) @binding(1) var<storage, read>       b:   array<f32>;
@group(0) @binding(2) var<storage, read>       dc:  array<f32>;
@group(0) @binding(3) var<storage, read_write> dab: array<f32>;
@group(0) @binding(4) var<uniform>             p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let stride = nwg.x * 64u;
    var i = gid.x;
    while (i < p.n) {
        let sig = 1.0 / (1.0 + exp(-a[i]));
        let silu = a[i] * sig;
        let dsilu = sig * (1.0 + a[i] * (1.0 - sig));
        dab[i] = dc[i] * dsilu * b[i];       // da
        dab[p.n + i] = dc[i] * silu;         // db
        i = i + stride;
    }
}
"#;

/// Backward of row-wise RMSNorm `y = x/rms · w` (`rms = √(mean(x²)+eps)`), input
/// gradient only. One invocation per row computes
/// `dx_j = (dy_j·w_j)/rms − x_j · (Σₖ dyₖ·wₖ·xₖ)/(d·rms³)` — the normalisation
/// jacobian, whose second term couples all elements through `rms`. The CPU
/// contract is [`crate::ops::cpu_rms_norm_backward`].
const RMSNORM_BWD_WGSL: &str = r#"
struct P { rows: u32, cols: u32, eps_bits: u32, _p0: u32, };

@group(0) @binding(0) var<storage, read>       x:      array<f32>;
@group(0) @binding(1) var<storage, read>       weight: array<f32>;
@group(0) @binding(2) var<storage, read>       dy:     array<f32>;
@group(0) @binding(3) var<storage, read_write> dx:     array<f32>;
@group(0) @binding(4) var<uniform>             p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= p.rows) { return; }
    if (p.cols == 0u) { return; }
    let base = row * p.cols;
    var ss = 0.0;
    for (var j: u32 = 0u; j < p.cols; j = j + 1u) { let v = x[base + j]; ss = ss + v * v; }
    let ms = ss / f32(p.cols) + bitcast<f32>(p.eps_bits);
    let rms = sqrt(ms);
    var dot = 0.0;
    for (var j: u32 = 0u; j < p.cols; j = j + 1u) { dot = dot + dy[base + j] * weight[j] * x[base + j]; }
    let inv = 1.0 / rms;
    let coef = dot / (f32(p.cols) * ms * rms);
    for (var j: u32 = 0u; j < p.cols; j = j + 1u) {
        dx[base + j] = dy[base + j] * weight[j] * inv - x[base + j] * coef;
    }
}
"#;

/// Backward of the scale + causal mask. `din = scale·dout` at kept positions
/// and `0` above the diagonal (masked keys carry no gradient — the `-1e30`
/// sentinel was a constant). Same 2D dispatch/convention as the forward mask.
const MASK_BWD_WGSL: &str = r#"
struct P { rows: u32, cols: u32, causal: u32, scale_bits: u32, };

@group(0) @binding(0) var<storage, read>       inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             p: P;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y;
    let j = gid.x;
    if (i >= p.rows || j >= p.cols) { return; }
    let idx = i * p.cols + j;
    if (p.causal == 1u && j > i) {
        out[idx] = 0.0;
    } else {
        out[idx] = inp[idx] * bitcast<f32>(p.scale_bits);
    }
}
"#;

/// Backward of the token embedding gather: accumulate the upstream grad `dout`
/// (`rows × d`) into the `vocab × d` table gradient — row `v` of `dtable` is the
/// sum of `dout` rows whose token id is `v`. Deterministic, **no atomics**: one
/// invocation per `(v, c)` output cell scans the `rows` tokens and sums the
/// matching contributions. The CPU contract is [`crate::ops::cpu_embed_backward`].
const EMBED_BWD_WGSL: &str = r#"
struct P { rows: u32, d: u32, vocab: u32, _p0: u32, };

@group(0) @binding(0) var<storage, read>       tokens: array<u32>;
@group(0) @binding(1) var<storage, read>       dout:   array<f32>;
@group(0) @binding(2) var<storage, read_write> dtable: array<f32>;
@group(0) @binding(3) var<uniform>             p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let n = p.vocab * p.d;
    let stride = nwg.x * 64u;
    var idx = gid.x;
    while (idx < n) {
        let v = idx / p.d;
        let c = idx % p.d;
        var acc = 0.0;
        for (var i: u32 = 0u; i < p.rows; i = i + 1u) {
            if (min(tokens[i], p.vocab - 1u) == v) { acc = acc + dout[i * p.d + c]; }
        }
        dtable[idx] = acc;
        idx = idx + stride;
    }
}
"#;

/// Gradient of the mean cross-entropy loss w.r.t. the logits. Given the softmax
/// probabilities `prob` (`rows × cols`) and the per-row `targets`,
/// `dlogits = (prob − onehot(target)) / rows`. One invocation per logit. The CPU
/// contract is [`crate::ops::cpu_cross_entropy_grad`].
const XENT_GRAD_WGSL: &str = r#"
struct P { rows: u32, cols: u32, inv_n_bits: u32, _p0: u32, };

@group(0) @binding(0) var<storage, read>       prob:    array<f32>;
@group(0) @binding(1) var<storage, read>       targets: array<u32>;
@group(0) @binding(2) var<storage, read_write> dlogits: array<f32>;
@group(0) @binding(3) var<uniform>             p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let n = p.rows * p.cols;
    let stride = nwg.x * 64u;
    var idx = gid.x;
    while (idx < n) {
        let i = idx / p.cols;
        let j = idx % p.cols;
        var v = prob[idx];
        if (j == targets[i]) { v = v - 1.0; }
        dlogits[idx] = v * bitcast<f32>(p.inv_n_bits);
        idx = idx + stride;
    }
}
"#;

/// One SGD parameter update: `out = param − lr·grad`, elementwise. The resident
/// optimizer step that closes the training loop. The CPU contract is
/// [`crate::ops::cpu_sgd_step`].
const SGD_WGSL: &str = r#"
struct P { n: u32, lr_bits: u32, _p0: u32, _p1: u32, };

@group(0) @binding(0) var<storage, read>       param: array<f32>;
@group(0) @binding(1) var<storage, read>       grad:  array<f32>;
@group(0) @binding(2) var<storage, read_write> out:   array<f32>;
@group(0) @binding(3) var<uniform>             p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let stride = nwg.x * 64u;
    var i = gid.x;
    while (i < p.n) {
        out[i] = param[i] - bitcast<f32>(p.lr_bits) * grad[i];
        i = i + stride;
    }
}
"#;

/// One AdamW step, updating `param`, `m`, `v` **in place** (bias-corrected Adam
/// with decoupled weight decay). Stays at 4 storage buffers (`param`, `grad`,
/// `m`, `v`) by updating in place. Hyper-parameters ride the uniform as f32
/// bits; `bc1`/`bc2` are the precomputed bias corrections `1 − βᵗ`. The CPU
/// contract is [`crate::ops::cpu_adamw_step`].
const ADAMW_WGSL: &str = r#"
struct P {
    n: u32, lr: u32, b1: u32, b2: u32,
    eps: u32, wd: u32, bc1: u32, bc2: u32,
};

@group(0) @binding(0) var<storage, read_write> param: array<f32>;
@group(0) @binding(1) var<storage, read>       grad:  array<f32>;
@group(0) @binding(2) var<storage, read_write> m:     array<f32>;
@group(0) @binding(3) var<storage, read_write> v:     array<f32>;
@group(0) @binding(4) var<uniform>             p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let lr = bitcast<f32>(p.lr);
    let b1 = bitcast<f32>(p.b1);
    let b2 = bitcast<f32>(p.b2);
    let eps = bitcast<f32>(p.eps);
    let wd = bitcast<f32>(p.wd);
    let stride = nwg.x * 64u;
    var i = gid.x;
    while (i < p.n) {
        let g = grad[i];
        let mi = b1 * m[i] + (1.0 - b1) * g;
        let vi = b2 * v[i] + (1.0 - b2) * g * g;
        m[i] = mi;
        v[i] = vi;
        let mhat = mi / bitcast<f32>(p.bc1);
        let vhat = vi / bitcast<f32>(p.bc2);
        param[i] = param[i] - lr * (mhat / (sqrt(vhat) + eps) + wd * param[i]);
        i = i + stride;
    }
}
"#;

/// Row-wise softmax: one invocation per row computes
/// `exp(x - rowmax) / sum(exp(x - rowmax))` over the row's `cols` elements
/// (max-subtracted for stability). The missing transformer-attention primitive;
/// the CPU contract is [`crate::ops::cpu_softmax`].
const SOFTMAX_WGSL: &str = r#"
struct P { rows: u32, cols: u32, _p0: u32, _p1: u32, };

@group(0) @binding(0) var<storage, read>       inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= p.rows) { return; }
    if (p.cols == 0u) { return; }
    let base = row * p.cols;
    var m = inp[base];
    for (var j: u32 = 1u; j < p.cols; j = j + 1u) { m = max(m, inp[base + j]); }
    var s = 0.0;
    for (var j: u32 = 0u; j < p.cols; j = j + 1u) { s = s + exp(inp[base + j] - m); }
    for (var j: u32 = 0u; j < p.cols; j = j + 1u) { out[base + j] = exp(inp[base + j] - m) / s; }
}
"#;

/// Pre-softmax attention step: scale a `rows × cols` score matrix by `scale`,
/// and — when `causal == 1` — overwrite every above-diagonal entry (key
/// `j > i` for query `i`) with the `-1e30` mask sentinel. 2D dispatch: `gid.x`
/// is the key column `j`, `gid.y` the query row `i`. `scale` is smuggled
/// through the `u32` uniform as raw bits and reconstructed with `bitcast`. The
/// CPU contract is [`crate::ops::cpu_scale_causal_mask`].
const MASK_WGSL: &str = r#"
struct P { rows: u32, cols: u32, causal: u32, scale_bits: u32, };

@group(0) @binding(0) var<storage, read>       inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             p: P;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y;
    let j = gid.x;
    if (i >= p.rows || j >= p.cols) { return; }
    let idx = i * p.cols + j;
    if (p.causal == 1u && j > i) {
        out[idx] = -1.0e30;
    } else {
        out[idx] = inp[idx] * bitcast<f32>(p.scale_bits);
    }
}
"#;

/// Rotary position embedding (RoPE), forward. One invocation per `(row, pair)`
/// rotates the interleaved lane pair `(2j, 2j+1)` of row `r` by angle
/// `θ_pos · freqⱼ`, with `pos = (r mod seq_len) + offset` and
/// `freqⱼ = theta^(-2j/dim)`:
/// `y[2j] = e·cos − o·sin`, `y[2j+1] = e·sin + o·cos` (`e=x[2j], o=x[2j+1]`).
/// This is exactly the model's interleaved rotation (`GQAAttention::rope_apply`
/// / `rope_on_tape`); the CPU contract is [`crate::ops::cpu_rope`]. `dim` must be
/// even. 2 storage buffers (well within the portable 4-buffer limit).
const ROPE_WGSL: &str = r#"
struct P { rows: u32, dim: u32, seq_len: u32, offset: u32, theta_bits: u32, _p0: u32, _p1: u32, _p2: u32, };

@group(0) @binding(0) var<storage, read>       inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             p: P;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;              // pair index in 0..dim/2
    let row = gid.y;
    let half = p.dim / 2u;
    if (row >= p.rows || j >= half) { return; }
    let theta = bitcast<f32>(p.theta_bits);
    let base = row * p.dim;
    let pos = f32((row % p.seq_len) + p.offset);
    let freq = pow(theta, -2.0 * f32(j) / f32(p.dim));
    let angle = pos * freq;
    let c = cos(angle);
    let s = sin(angle);
    let e = inp[base + 2u * j];
    let o = inp[base + 2u * j + 1u];
    out[base + 2u * j]      = e * c - o * s;
    out[base + 2u * j + 1u] = e * s + o * c;
}
"#;

/// Backward of [`ROPE_WGSL`]. RoPE is a rotation, so its adjoint is the
/// transpose rotation: given the upstream grad `dy`, one invocation per
/// `(row, pair)` computes `dx[2j] = cos·dy[2j] + sin·dy[2j+1]`,
/// `dx[2j+1] = −sin·dy[2j] + cos·dy[2j+1]` with the same `pos`/`freq` as the
/// forward. The CPU contract is [`crate::ops::cpu_rope_backward`].
const ROPE_BWD_WGSL: &str = r#"
struct P { rows: u32, dim: u32, seq_len: u32, offset: u32, theta_bits: u32, _p0: u32, _p1: u32, _p2: u32, };

@group(0) @binding(0) var<storage, read>       dy:  array<f32>;
@group(0) @binding(1) var<storage, read_write> dx:  array<f32>;
@group(0) @binding(2) var<uniform>             p: P;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let row = gid.y;
    let half = p.dim / 2u;
    if (row >= p.rows || j >= half) { return; }
    let theta = bitcast<f32>(p.theta_bits);
    let base = row * p.dim;
    let pos = f32((row % p.seq_len) + p.offset);
    let freq = pow(theta, -2.0 * f32(j) / f32(p.dim));
    let angle = pos * freq;
    let c = cos(angle);
    let s = sin(angle);
    let ge = dy[base + 2u * j];
    let go = dy[base + 2u * j + 1u];
    dx[base + 2u * j]      =  c * ge + s * go;
    dx[base + 2u * j + 1u] = -s * ge + c * go;
}
"#;

/// Gather a contiguous column block: `out[r, c] = inp[r, col_start + c]` for
/// `c in 0..ncols`. `inp` is `rows × src_cols`, `out` is `rows × ncols`. Used to
/// extract a single head's `d_head` columns from a full-width projection. The
/// CPU contract is [`crate::ops::cpu_slice_cols`]; its adjoint is
/// [`PLACE_COLS_WGSL`]. 2 storage buffers.
const SLICE_COLS_WGSL: &str = r#"
struct P { rows: u32, ncols: u32, src_cols: u32, col_start: u32, };

@group(0) @binding(0) var<storage, read>       inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             p: P;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    let r = gid.y;
    if (r >= p.rows || c >= p.ncols) { return; }
    out[r * p.ncols + c] = inp[r * p.src_cols + p.col_start + c];
}
"#;

/// Scatter a narrow block into a zero-padded wide matrix:
/// `out[r, col_start + c] = inp[r, c]` for `c in 0..ncols`, and `0` everywhere
/// else. `inp` is `rows × ncols`, `out` is `rows × dst_cols`. Used to place a
/// head's context back into its `d_model` column slot before summing heads (the
/// efficient equivalent of the model's `build_pad` matmul). The CPU contract is
/// [`crate::ops::cpu_place_cols`]; it is the adjoint of [`SLICE_COLS_WGSL`].
/// One invocation per *output* element, so the zeros are written explicitly
/// (no uninitialised VRAM). 2 storage buffers.
const PLACE_COLS_WGSL: &str = r#"
struct P { rows: u32, ncols: u32, dst_cols: u32, col_start: u32, };

@group(0) @binding(0) var<storage, read>       inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             p: P;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    let r = gid.y;
    if (r >= p.rows || c >= p.dst_cols) { return; }
    let lo = p.col_start;
    let hi = p.col_start + p.ncols;
    if (c >= lo && c < hi) {
        out[r * p.dst_cols + c] = inp[r * p.ncols + (c - lo)];
    } else {
        out[r * p.dst_cols + c] = 0.0;
    }
}
"#;

/// A wgpu device + compiled compute pipelines, created once and reused across
/// calls (adapter/device acquisition and shader compilation are expensive).
///
/// A cheap-to-[`Clone`] handle: the device, queue and pipelines live behind one
/// [`Arc`] (none of `wgpu::Device`/`Queue`/`ComputePipeline` are themselves
/// `Clone` in this wgpu version), so a clone shares the *same* underlying device.
/// The shared test context (see [`WgpuContext::new`]) relies on this. [`Deref`]
/// exposes the inner fields transparently, so call sites read `self.device`,
/// `self.pipeline`, … unchanged.
///
/// [`Deref`]: std::ops::Deref
#[derive(Clone)]
pub struct WgpuContext {
    inner: std::sync::Arc<WgpuContextInner>,
}

impl std::ops::Deref for WgpuContext {
    type Target = WgpuContextInner;
    fn deref(&self) -> &WgpuContextInner {
        &self.inner
    }
}

/// The owned device + pipelines behind a [`WgpuContext`]'s `Arc`.
pub struct WgpuContextInner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    ew_pipeline: wgpu::ComputePipeline,
    softmax_pipeline: wgpu::ComputePipeline,
    mask_pipeline: wgpu::ComputePipeline,
    rmsnorm_pipeline: wgpu::ComputePipeline,
    embed_pipeline: wgpu::ComputePipeline,
    softmax_bwd_pipeline: wgpu::ComputePipeline,
    swiglu_bwd_pipeline: wgpu::ComputePipeline,
    rmsnorm_bwd_pipeline: wgpu::ComputePipeline,
    mask_bwd_pipeline: wgpu::ComputePipeline,
    embed_bwd_pipeline: wgpu::ComputePipeline,
    xent_grad_pipeline: wgpu::ComputePipeline,
    sgd_pipeline: wgpu::ComputePipeline,
    adamw_pipeline: wgpu::ComputePipeline,
    rope_pipeline: wgpu::ComputePipeline,
    rope_bwd_pipeline: wgpu::ComputePipeline,
    slice_cols_pipeline: wgpu::ComputePipeline,
    place_cols_pipeline: wgpu::ComputePipeline,
    adapter_name: String,
}

/// A row-major `f32` matrix resident in GPU memory (a storage buffer + shape).
///
/// Produced by [`crate::GpuChain`] (`upload` / `matmul`); an intermediate stays
/// in VRAM and feeds the next GEMM without a CPU round-trip.
pub struct GpuMatrix {
    buf: wgpu::Buffer,
    rows: usize,
    cols: usize,
}

impl GpuMatrix {
    /// Row count.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Column count.
    pub fn cols(&self) -> usize {
        self.cols
    }
}

impl WgpuContext {
    /// Acquire an adapter/device and compile the compute pipelines. Returns
    /// [`BackendError::Unavailable`] if no adapter is available (e.g. no Vulkan
    /// driver) — never a silent fake.
    ///
    /// In **test builds** this hands out cheap clones of a single process-wide
    /// context, created on first use and held in a never-dropped [`OnceLock`], so
    /// every unit test shares one device instead of creating its own. The static
    /// keeps a permanent reference, so the Vulkan device is never torn down —
    /// which sidesteps a teardown-time SIGSEGV seen in some drivers (e.g. the
    /// Jetson Thor's) when dozens of devices are created and dropped in a single
    /// process. Production builds always create a fresh context.
    #[cfg(test)]
    pub fn new() -> BackendResult<Self> {
        use std::sync::OnceLock;
        static SHARED: OnceLock<Option<WgpuContext>> = OnceLock::new();
        SHARED
            .get_or_init(|| WgpuContext::new_uncached().ok())
            .clone()
            .ok_or(BackendError::Unavailable("wgpu"))
    }

    /// See [`WgpuContext::new`]; production always builds a fresh context.
    #[cfg(not(test))]
    pub fn new() -> BackendResult<Self> {
        Self::new_uncached()
    }

    /// Build a fresh context: acquire an adapter/device and compile every compute
    /// pipeline. The uncached path behind [`WgpuContext::new`].
    fn new_uncached() -> BackendResult<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .ok_or(BackendError::Unavailable("wgpu"))?;
        let adapter_name = adapter.get_info().name;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("scirust-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))
        .map_err(|_| BackendError::Unavailable("wgpu"))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gemm"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(GEMM_WGSL)),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gemm"),
            layout: None,
            module: &shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let ew_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ew"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(EW_WGSL)),
        });
        let ew_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ew"),
            layout: None,
            module: &ew_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let softmax_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("softmax"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SOFTMAX_WGSL)),
        });
        let softmax_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("softmax"),
            layout: None,
            module: &softmax_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let mask_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scale_causal_mask"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(MASK_WGSL)),
        });
        let mask_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scale_causal_mask"),
            layout: None,
            module: &mask_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let rmsnorm_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rmsnorm"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(RMSNORM_WGSL)),
        });
        let rmsnorm_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rmsnorm"),
            layout: None,
            module: &rmsnorm_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let embed_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("embed"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(EMBED_WGSL)),
        });
        let embed_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("embed"),
            layout: None,
            module: &embed_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let softmax_bwd_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("softmax_bwd"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SOFTMAX_BWD_WGSL)),
        });
        let softmax_bwd_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("softmax_bwd"),
                layout: None,
                module: &softmax_bwd_shader,
                entry_point: "main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        let swiglu_bwd_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("swiglu_bwd"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SWIGLU_BWD_WGSL)),
        });
        let swiglu_bwd_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("swiglu_bwd"),
                layout: None,
                module: &swiglu_bwd_shader,
                entry_point: "main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        let rmsnorm_bwd_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rmsnorm_bwd"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(RMSNORM_BWD_WGSL)),
        });
        let rmsnorm_bwd_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("rmsnorm_bwd"),
                layout: None,
                module: &rmsnorm_bwd_shader,
                entry_point: "main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        let mask_bwd_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mask_bwd"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(MASK_BWD_WGSL)),
        });
        let mask_bwd_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mask_bwd"),
            layout: None,
            module: &mask_bwd_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let embed_bwd_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("embed_bwd"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(EMBED_BWD_WGSL)),
        });
        let embed_bwd_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("embed_bwd"),
            layout: None,
            module: &embed_bwd_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let xent_grad_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("xent_grad"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(XENT_GRAD_WGSL)),
        });
        let xent_grad_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("xent_grad"),
            layout: None,
            module: &xent_grad_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let sgd_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sgd"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SGD_WGSL)),
        });
        let sgd_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sgd"),
            layout: None,
            module: &sgd_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let adamw_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("adamw"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(ADAMW_WGSL)),
        });
        let adamw_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("adamw"),
            layout: None,
            module: &adamw_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let rope_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rope"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(ROPE_WGSL)),
        });
        let rope_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rope"),
            layout: None,
            module: &rope_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let rope_bwd_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rope_bwd"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(ROPE_BWD_WGSL)),
        });
        let rope_bwd_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rope_bwd"),
            layout: None,
            module: &rope_bwd_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let slice_cols_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("slice_cols"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SLICE_COLS_WGSL)),
        });
        let slice_cols_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("slice_cols"),
                layout: None,
                module: &slice_cols_shader,
                entry_point: "main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        let place_cols_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("place_cols"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(PLACE_COLS_WGSL)),
        });
        let place_cols_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("place_cols"),
                layout: None,
                module: &place_cols_shader,
                entry_point: "main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        Ok(Self {
            inner: std::sync::Arc::new(WgpuContextInner {
                device,
                queue,
                pipeline,
                ew_pipeline,
                softmax_pipeline,
                mask_pipeline,
                rmsnorm_pipeline,
                embed_pipeline,
                softmax_bwd_pipeline,
                swiglu_bwd_pipeline,
                rmsnorm_bwd_pipeline,
                mask_bwd_pipeline,
                embed_bwd_pipeline,
                xent_grad_pipeline,
                sgd_pipeline,
                adamw_pipeline,
                rope_pipeline,
                rope_bwd_pipeline,
                slice_cols_pipeline,
                place_cols_pipeline,
                adapter_name,
            }),
        })
    }

    /// Row-wise softmax of a row-major `rows × cols` matrix, matching
    /// [`crate::ops::cpu_softmax`]. Uploads the input, dispatches one thread per
    /// row, downloads the result.
    pub fn softmax_rows(&self, data: &[f32], rows: usize, cols: usize) -> BackendResult<Vec<f32>> {
        if data.len() != rows * cols
        {
            return Err(BackendError::ShapeMismatch(format!(
                "softmax: {} elems != {rows}×{cols}",
                data.len()
            )));
        }
        if data.is_empty()
        {
            return Ok(Vec::new());
        }
        let bytes = std::mem::size_of_val(data) as u64;
        let in_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("softmax-in"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("softmax-out"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self._encode_softmax(&in_buf, &out_buf, rows, cols);
        self.download_buffer(&out_buf, data.len(), bytes)
    }

    /// Row-wise softmax of a **resident** `rows × cols` matrix, result kept in
    /// VRAM (no download) — the resident counterpart of [`Self::softmax_rows`]
    /// used to keep attention weights on-device across the score → weight →
    /// context chain.
    pub fn softmax_resident(&self, x: &GpuMatrix) -> BackendResult<GpuMatrix> {
        let elems = x.rows * x.cols;
        let bytes = (elems.max(1) * std::mem::size_of::<f32>()) as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("softmax-res"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if elems > 0 && x.cols > 0
        {
            self._encode_softmax(&x.buf, &out_buf, x.rows, x.cols);
        }
        Ok(GpuMatrix {
            buf: out_buf,
            rows: x.rows,
            cols: x.cols,
        })
    }

    /// Encode + submit one row-wise-softmax dispatch reading `in_buf`
    /// (`rows × cols`, row-major) and writing `out_buf`. Shared by the
    /// upload/download and resident entry points.
    fn _encode_softmax(
        &self,
        in_buf: &wgpu::Buffer,
        out_buf: &wgpu::Buffer,
        rows: usize,
        cols: usize,
    ) {
        let params: [u32; 4] = [rows as u32, cols as u32, 0, 0];
        let p_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("softmax-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softmax"),
            layout: &self.softmax_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("softmax"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("softmax"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.softmax_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((rows as u32).div_ceil(64), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// Pre-softmax attention step on a row-major `rows × cols` score matrix:
    /// multiply by `scale`, and — when `causal` — replace every entry above the
    /// diagonal (key `j > i`) with the `-1e30` mask sentinel. Matches
    /// [`crate::ops::cpu_scale_causal_mask`]. One thread per score cell.
    pub fn scale_causal_mask(
        &self,
        scores: &[f32],
        rows: usize,
        cols: usize,
        scale: f32,
        causal: bool,
    ) -> BackendResult<Vec<f32>> {
        if scores.len() != rows * cols
        {
            return Err(BackendError::ShapeMismatch(format!(
                "scale_causal_mask: {} elems != {rows}×{cols}",
                scores.len()
            )));
        }
        if scores.is_empty()
        {
            return Ok(Vec::new());
        }
        let bytes = std::mem::size_of_val(scores) as u64;
        let in_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mask-in"),
                contents: bytemuck::cast_slice(scores),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mask-out"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self._encode_scale_causal_mask(&in_buf, &out_buf, rows, cols, scale, causal);
        self.download_buffer(&out_buf, scores.len(), bytes)
    }

    /// Scale + causal mask of a **resident** `rows × cols` score matrix, result
    /// kept in VRAM (no download) — the resident counterpart of
    /// [`Self::scale_causal_mask`]. `rows`/`cols` are `x.rows`/`x.cols`; the
    /// masking convention (query = row, key = column) is the same.
    pub fn scale_causal_mask_resident(
        &self,
        x: &GpuMatrix,
        scale: f32,
        causal: bool,
    ) -> BackendResult<GpuMatrix> {
        let elems = x.rows * x.cols;
        let bytes = (elems.max(1) * std::mem::size_of::<f32>()) as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mask-res"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if elems > 0
        {
            self._encode_scale_causal_mask(&x.buf, &out_buf, x.rows, x.cols, scale, causal);
        }
        Ok(GpuMatrix {
            buf: out_buf,
            rows: x.rows,
            cols: x.cols,
        })
    }

    /// Encode + submit one scale + causal-mask dispatch reading `in_buf`
    /// (`rows × cols`, row-major) and writing `out_buf`. Shared by the
    /// upload/download and resident entry points.
    fn _encode_scale_causal_mask(
        &self,
        in_buf: &wgpu::Buffer,
        out_buf: &wgpu::Buffer,
        rows: usize,
        cols: usize,
        scale: f32,
        causal: bool,
    ) {
        let params: [u32; 4] = [rows as u32, cols as u32, causal as u32, scale.to_bits()];
        let p_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mask-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mask"),
            layout: &self.mask_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mask"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mask"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.mask_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((cols as u32).div_ceil(8), (rows as u32).div_ceil(8), 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// Rotary position embedding of a **resident** `rows × dim` matrix (rows are
    /// token positions, `dim` the per-head dimension), result kept in VRAM.
    /// Applies the interleaved-pair rotation with `pos = (row mod seq_len) +
    /// offset` and `freqⱼ = theta^(-2j/dim)` — matching [`crate::ops::cpu_rope`]
    /// and the sciagent model's RoPE. `dim` must be even and `seq_len ≥ 1`.
    pub fn rope_resident(
        &self,
        x: &GpuMatrix,
        seq_len: usize,
        offset: usize,
        theta: f32,
    ) -> BackendResult<GpuMatrix> {
        if !x.cols.is_multiple_of(2)
        {
            return Err(BackendError::ShapeMismatch(format!(
                "rope: dim (cols) must be even, got {}",
                x.cols
            )));
        }
        if seq_len == 0
        {
            return Err(BackendError::ShapeMismatch(
                "rope: seq_len must be ≥ 1".into(),
            ));
        }
        let elems = x.rows * x.cols;
        let bytes = (elems.max(1) * std::mem::size_of::<f32>()) as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rope-res"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if elems > 0
        {
            self._encode_rope(
                &self.rope_pipeline,
                &x.buf,
                &out_buf,
                x.rows,
                x.cols,
                seq_len,
                offset,
                theta,
            );
        }
        Ok(GpuMatrix {
            buf: out_buf,
            rows: x.rows,
            cols: x.cols,
        })
    }

    /// Backward of [`Self::rope_resident`]: given the upstream grad `dy`
    /// (`rows × dim`, resident), returns the resident `dx` via the transpose
    /// rotation. Same `seq_len`/`offset`/`theta` as the forward. The CPU contract
    /// is [`crate::ops::cpu_rope_backward`].
    pub fn rope_backward_resident(
        &self,
        dy: &GpuMatrix,
        seq_len: usize,
        offset: usize,
        theta: f32,
    ) -> BackendResult<GpuMatrix> {
        if !dy.cols.is_multiple_of(2)
        {
            return Err(BackendError::ShapeMismatch(format!(
                "rope_backward: dim (cols) must be even, got {}",
                dy.cols
            )));
        }
        if seq_len == 0
        {
            return Err(BackendError::ShapeMismatch(
                "rope_backward: seq_len must be ≥ 1".into(),
            ));
        }
        let elems = dy.rows * dy.cols;
        let bytes = (elems.max(1) * std::mem::size_of::<f32>()) as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rope-bwd-res"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if elems > 0
        {
            self._encode_rope(
                &self.rope_bwd_pipeline,
                &dy.buf,
                &out_buf,
                dy.rows,
                dy.cols,
                seq_len,
                offset,
                theta,
            );
        }
        Ok(GpuMatrix {
            buf: out_buf,
            rows: dy.rows,
            cols: dy.cols,
        })
    }

    /// Encode + submit one RoPE (or RoPE-backward) dispatch: `in_buf`
    /// (`rows × dim`, row-major) rotated pairwise into `out_buf`. The forward and
    /// backward kernels share this bind-group layout (in, out, params); the
    /// caller picks the pipeline.
    #[allow(clippy::too_many_arguments)]
    fn _encode_rope(
        &self,
        pipeline: &wgpu::ComputePipeline,
        in_buf: &wgpu::Buffer,
        out_buf: &wgpu::Buffer,
        rows: usize,
        dim: usize,
        seq_len: usize,
        offset: usize,
        theta: f32,
    ) {
        let params: [u32; 8] = [
            rows as u32,
            dim as u32,
            seq_len as u32,
            offset as u32,
            theta.to_bits(),
            0,
            0,
            0,
        ];
        let p_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rope-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rope"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rope"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rope"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(((dim / 2) as u32).div_ceil(8), (rows as u32).div_ceil(8), 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// Gather a contiguous column block of a **resident** matrix:
    /// `out[r, c] = x[r, col_start + c]` for `c in 0..ncols`, result resident.
    /// Extracts one head's `d_head` columns from a full-width projection. The CPU
    /// contract is [`crate::ops::cpu_slice_cols`].
    pub fn slice_cols_resident(
        &self,
        x: &GpuMatrix,
        col_start: usize,
        ncols: usize,
    ) -> BackendResult<GpuMatrix> {
        if col_start + ncols > x.cols
        {
            return Err(BackendError::ShapeMismatch(format!(
                "slice_cols: [{col_start}, {}) out of {} columns",
                col_start + ncols,
                x.cols
            )));
        }
        let elems = x.rows * ncols;
        let bytes = (elems.max(1) * std::mem::size_of::<f32>()) as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("slice-cols-res"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if elems > 0
        {
            self._encode_cols(
                &self.slice_cols_pipeline,
                &x.buf,
                &out_buf,
                x.rows,
                ncols,
                x.cols,
                col_start,
                ncols,
            );
        }
        Ok(GpuMatrix {
            buf: out_buf,
            rows: x.rows,
            cols: ncols,
        })
    }

    /// Scatter a **resident** narrow block into a zero-padded wide matrix:
    /// `out[r, col_start + c] = x[r, c]`, `0` elsewhere; result resident and
    /// `rows × dst_cols`. Places a head's context back into its `d_model` slot.
    /// This is the adjoint of [`Self::slice_cols_resident`]; the CPU contract is
    /// [`crate::ops::cpu_place_cols`].
    pub fn place_cols_resident(
        &self,
        x: &GpuMatrix,
        col_start: usize,
        dst_cols: usize,
    ) -> BackendResult<GpuMatrix> {
        if col_start + x.cols > dst_cols
        {
            return Err(BackendError::ShapeMismatch(format!(
                "place_cols: block [{col_start}, {}) does not fit in {dst_cols} columns",
                col_start + x.cols
            )));
        }
        let elems = x.rows * dst_cols;
        let bytes = (elems.max(1) * std::mem::size_of::<f32>()) as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("place-cols-res"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if elems > 0
        {
            self._encode_cols(
                &self.place_cols_pipeline,
                &x.buf,
                &out_buf,
                x.rows,
                x.cols,
                dst_cols,
                col_start,
                dst_cols,
            );
        }
        Ok(GpuMatrix {
            buf: out_buf,
            rows: x.rows,
            cols: dst_cols,
        })
    }

    /// Encode + submit one column slice/place dispatch. `p1`/`p2` fill the
    /// kernel's `ncols` and `src_cols`/`dst_cols` params (position 1/2);
    /// `dispatch_cols` is the output width driving the x-dimension. Shared by
    /// [`Self::slice_cols_resident`] and [`Self::place_cols_resident`], which
    /// differ only by pipeline and which width is the output.
    #[allow(clippy::too_many_arguments)]
    fn _encode_cols(
        &self,
        pipeline: &wgpu::ComputePipeline,
        in_buf: &wgpu::Buffer,
        out_buf: &wgpu::Buffer,
        rows: usize,
        p1: usize,
        p2: usize,
        col_start: usize,
        dispatch_cols: usize,
    ) {
        let params: [u32; 4] = [rows as u32, p1 as u32, p2 as u32, col_start as u32];
        let p_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cols-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cols"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cols"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cols"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                (dispatch_cols as u32).div_ceil(8),
                (rows as u32).div_ceil(8),
                1,
            );
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// Row-wise RMSNorm of a **resident** `rows × cols` matrix, result kept in
    /// VRAM — `x / sqrt(mean(x²) + eps) · weight`, matching
    /// [`crate::ops::cpu_rms_norm`]. `weight` is a resident `cols`-length gain
    /// vector (any shape whose element count is `x.cols`).
    pub fn rms_norm_resident(
        &self,
        x: &GpuMatrix,
        weight: &GpuMatrix,
        eps: f32,
    ) -> BackendResult<GpuMatrix> {
        let weight_len = weight.rows * weight.cols;
        if weight_len != x.cols
        {
            return Err(BackendError::ShapeMismatch(format!(
                "rms_norm: weight has {weight_len} elems, expected cols = {}",
                x.cols
            )));
        }
        let elems = x.rows * x.cols;
        let bytes = (elems.max(1) * std::mem::size_of::<f32>()) as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rmsnorm-res"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if elems > 0
        {
            self._encode_rms_norm(&x.buf, &weight.buf, &out_buf, x.rows, x.cols, eps);
        }
        Ok(GpuMatrix {
            buf: out_buf,
            rows: x.rows,
            cols: x.cols,
        })
    }

    /// Encode + submit one RMSNorm dispatch: `in_buf` (`rows × cols`) normalised
    /// row-wise and scaled by the `cols`-length `weight_buf`, written to
    /// `out_buf`. `eps` is passed as raw bits and reconstructed with `bitcast`.
    fn _encode_rms_norm(
        &self,
        in_buf: &wgpu::Buffer,
        weight_buf: &wgpu::Buffer,
        out_buf: &wgpu::Buffer,
        rows: usize,
        cols: usize,
        eps: f32,
    ) {
        let params: [u32; 4] = [rows as u32, cols as u32, eps.to_bits(), 0];
        let p_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rmsnorm-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rmsnorm"),
            layout: &self.rmsnorm_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rmsnorm"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rmsnorm"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.rmsnorm_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((rows as u32).div_ceil(64), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// Token embedding gather: build a resident `tokens.len() × d` matrix whose
    /// row `i` is row `tokens[i]` of the resident `vocab × d` `table`. Matches
    /// [`crate::ops::cpu_embed`]. The token ids are uploaded as a `u32` buffer;
    /// the gathered rows stay in VRAM to feed the transformer stack.
    pub fn embed_resident(&self, tokens: &[u32], table: &GpuMatrix) -> BackendResult<GpuMatrix> {
        let rows = tokens.len();
        let d = table.cols;
        let vocab = table.rows;
        let elems = rows * d;
        let bytes = (elems.max(1) * std::mem::size_of::<f32>()) as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("embed-out"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if elems > 0
        {
            let tok_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("embed-tokens"),
                    contents: bytemuck::cast_slice(tokens),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let params: [u32; 4] = [rows as u32, d as u32, vocab as u32, 0];
            let p_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("embed-params"),
                    contents: bytemuck::cast_slice(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("embed"),
                layout: &self.embed_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: tok_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: table.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: p_buf.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("embed"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("embed"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.embed_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(flat_workgroups(elems as u32), 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
        }
        Ok(GpuMatrix {
            buf: out_buf,
            rows,
            cols: d,
        })
    }

    /// Backward of row-wise softmax: given the forward output `y` and upstream
    /// grad `dy` (same `rows × cols`), returns `dx = y ⊙ (dy − Σⱼ dyⱼyⱼ)`,
    /// resident. Matches [`crate::ops::cpu_softmax_backward`].
    pub fn softmax_backward_resident(
        &self,
        y: &GpuMatrix,
        dy: &GpuMatrix,
    ) -> BackendResult<GpuMatrix> {
        if y.rows != dy.rows || y.cols != dy.cols
        {
            return Err(BackendError::ShapeMismatch(format!(
                "softmax_backward: {}×{} vs {}×{}",
                y.rows, y.cols, dy.rows, dy.cols
            )));
        }
        let elems = y.rows * y.cols;
        let bytes = (elems.max(1) * std::mem::size_of::<f32>()) as u64;
        let dx = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("softmax-bwd-dx"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if elems > 0 && y.cols > 0
        {
            let params: [u32; 4] = [y.rows as u32, y.cols as u32, 0, 0];
            let p_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("softmax-bwd-params"),
                    contents: bytemuck::cast_slice(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("softmax-bwd"),
                layout: &self.softmax_bwd_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: y.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dy.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: dx.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: p_buf.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("softmax-bwd"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("softmax-bwd"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.softmax_bwd_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((y.rows as u32).div_ceil(64), 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
        }
        Ok(GpuMatrix {
            buf: dx,
            rows: y.rows,
            cols: y.cols,
        })
    }

    /// Backward of the SwiGLU gate `c = silu(a) ⊙ b`: given `a`, `b` and the
    /// upstream grad `dc` (same shape), returns `(da, db)` resident, where
    /// `da = dc·silu'(a)·b` and `db = dc·silu(a)`. Matches
    /// [`crate::ops::cpu_swiglu_backward`].
    pub fn swiglu_backward_resident(
        &self,
        a: &GpuMatrix,
        b: &GpuMatrix,
        dc: &GpuMatrix,
    ) -> BackendResult<(GpuMatrix, GpuMatrix)> {
        if a.rows != b.rows || a.cols != b.cols || a.rows != dc.rows || a.cols != dc.cols
        {
            return Err(BackendError::ShapeMismatch(format!(
                "swiglu_backward: a {}×{}, b {}×{}, dc {}×{}",
                a.rows, a.cols, b.rows, b.cols, dc.rows, dc.cols
            )));
        }
        let n = a.rows * a.cols;
        let bytes = (n.max(1) * std::mem::size_of::<f32>()) as u64;
        let mk = |label| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let da = mk("swiglu-bwd-da");
        let db = mk("swiglu-bwd-db");
        if n > 0
        {
            // The kernel writes da into [0, n) and db into [n, 2n) of one buffer,
            // keeping the layout at 4 storage buffers (a, b, dc, dab) — within the
            // portable `downlevel_defaults` limit. We then split it into the two
            // result buffers with GPU-side copies (no CPU round-trip).
            let dab = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("swiglu-bwd-dab"),
                size: 2 * bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let params: [u32; 4] = [n as u32, 0, 0, 0];
            let p_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("swiglu-bwd-params"),
                    contents: bytemuck::cast_slice(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("swiglu-bwd"),
                layout: &self.swiglu_bwd_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: dc.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: dab.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: p_buf.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("swiglu-bwd"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("swiglu-bwd"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.swiglu_bwd_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(flat_workgroups(n as u32), 1, 1);
            }
            // Split the packed [da | db] into the two resident result buffers.
            encoder.copy_buffer_to_buffer(&dab, 0, &da, 0, bytes);
            encoder.copy_buffer_to_buffer(&dab, bytes, &db, 0, bytes);
            self.queue.submit(Some(encoder.finish()));
        }
        Ok((
            GpuMatrix {
                buf: da,
                rows: a.rows,
                cols: a.cols,
            },
            GpuMatrix {
                buf: db,
                rows: a.rows,
                cols: a.cols,
            },
        ))
    }

    /// Backward of row-wise RMSNorm (input gradient): given `x`, the `cols`
    /// `weight`, upstream grad `dy` and `eps`, returns `dx` resident. Matches
    /// [`crate::ops::cpu_rms_norm_backward`].
    pub fn rms_norm_backward_resident(
        &self,
        x: &GpuMatrix,
        weight: &GpuMatrix,
        dy: &GpuMatrix,
        eps: f32,
    ) -> BackendResult<GpuMatrix> {
        if x.rows != dy.rows || x.cols != dy.cols || weight.rows * weight.cols != x.cols
        {
            return Err(BackendError::ShapeMismatch(format!(
                "rms_norm_backward: x {}×{}, dy {}×{}, weight {}",
                x.rows,
                x.cols,
                dy.rows,
                dy.cols,
                weight.rows * weight.cols
            )));
        }
        let elems = x.rows * x.cols;
        let bytes = (elems.max(1) * std::mem::size_of::<f32>()) as u64;
        let dx = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rmsnorm-bwd-dx"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if elems > 0 && x.cols > 0
        {
            let params: [u32; 4] = [x.rows as u32, x.cols as u32, eps.to_bits(), 0];
            let p_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("rmsnorm-bwd-params"),
                    contents: bytemuck::cast_slice(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rmsnorm-bwd"),
                layout: &self.rmsnorm_bwd_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: x.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: weight.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: dy.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: dx.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: p_buf.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rmsnorm-bwd"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rmsnorm-bwd"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.rmsnorm_bwd_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((x.rows as u32).div_ceil(64), 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
        }
        Ok(GpuMatrix {
            buf: dx,
            rows: x.rows,
            cols: x.cols,
        })
    }

    /// One SGD step, resident: `out = param − lr·grad` (same shape). Matches
    /// [`crate::ops::cpu_sgd_step`]. Returns a fresh resident matrix; feed it
    /// back as the next iteration's parameter.
    pub fn sgd_step_resident(
        &self,
        param: &GpuMatrix,
        grad: &GpuMatrix,
        lr: f32,
    ) -> BackendResult<GpuMatrix> {
        if param.rows != grad.rows || param.cols != grad.cols
        {
            return Err(BackendError::ShapeMismatch(format!(
                "sgd_step: param {}×{} vs grad {}×{}",
                param.rows, param.cols, grad.rows, grad.cols
            )));
        }
        let n = param.rows * param.cols;
        let bytes = (n.max(1) * std::mem::size_of::<f32>()) as u64;
        let out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sgd-out"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if n > 0
        {
            let params: [u32; 4] = [n as u32, lr.to_bits(), 0, 0];
            let p_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("sgd-params"),
                    contents: bytemuck::cast_slice(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sgd"),
                layout: &self.sgd_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: param.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: grad.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: p_buf.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sgd") });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("sgd"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.sgd_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(flat_workgroups(n as u32), 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
        }
        Ok(GpuMatrix {
            buf: out,
            rows: param.rows,
            cols: param.cols,
        })
    }

    /// One AdamW step at optimizer step `step` (1-based), updating `param`, `m`
    /// and `v` **in place** — bias-corrected Adam with decoupled weight decay.
    /// `m`/`v` are the resident first/second-moment buffers (start them at zero
    /// and reuse them across steps). Matches [`crate::ops::cpu_adamw_step`].
    #[allow(clippy::too_many_arguments)]
    pub fn adamw_step_resident(
        &self,
        param: &GpuMatrix,
        grad: &GpuMatrix,
        m: &GpuMatrix,
        v: &GpuMatrix,
        lr: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
        step: u32,
    ) -> BackendResult<()> {
        let n = param.rows * param.cols;
        if grad.rows * grad.cols != n || m.rows * m.cols != n || v.rows * v.cols != n
        {
            return Err(BackendError::ShapeMismatch(format!(
                "adamw_step: param {n} vs grad {} m {} v {}",
                grad.rows * grad.cols,
                m.rows * m.cols,
                v.rows * v.cols
            )));
        }
        if n == 0
        {
            return Ok(());
        }
        let (b1, b2) = betas;
        let bc1 = 1.0 - b1.powi(step as i32);
        let bc2 = 1.0 - b2.powi(step as i32);
        let params: [u32; 8] = [
            n as u32,
            lr.to_bits(),
            b1.to_bits(),
            b2.to_bits(),
            eps.to_bits(),
            weight_decay.to_bits(),
            bc1.to_bits(),
            bc2.to_bits(),
        ];
        let p_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("adamw-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("adamw"),
            layout: &self.adamw_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: param.buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grad.buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: m.buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: v.buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("adamw"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("adamw"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.adamw_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(flat_workgroups(n as u32), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Gradient of the mean cross-entropy loss w.r.t. the `logits` (`rows ×
    /// vocab`) for the per-row `targets`: softmaxes each row, then
    /// `dlogits = (softmax(logits) − onehot(target)) / rows`, resident. This is
    /// the seed of the whole training backward. Matches
    /// [`crate::ops::cpu_cross_entropy_grad`].
    pub fn cross_entropy_grad_resident(
        &self,
        logits: &GpuMatrix,
        targets: &[u32],
    ) -> BackendResult<GpuMatrix> {
        let rows = logits.rows;
        let cols = logits.cols;
        if targets.len() != rows
        {
            return Err(BackendError::ShapeMismatch(format!(
                "cross_entropy_grad: {} targets != {rows} rows",
                targets.len()
            )));
        }
        let prob = self.softmax_resident(logits)?;
        let elems = rows * cols;
        let bytes = (elems.max(1) * std::mem::size_of::<f32>()) as u64;
        let dlogits = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("xent-dlogits"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if elems > 0
        {
            let tgt_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("xent-targets"),
                    contents: bytemuck::cast_slice(targets),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let inv_n = 1.0f32 / rows as f32;
            let params: [u32; 4] = [rows as u32, cols as u32, inv_n.to_bits(), 0];
            let p_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("xent-params"),
                    contents: bytemuck::cast_slice(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("xent-grad"),
                layout: &self.xent_grad_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: prob.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: tgt_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: dlogits.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: p_buf.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("xent-grad"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("xent-grad"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.xent_grad_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(flat_workgroups(elems as u32), 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
        }
        Ok(GpuMatrix {
            buf: dlogits,
            rows,
            cols,
        })
    }

    /// Backward of the embedding gather: accumulate the upstream grad `dout`
    /// (`tokens.len() × d`) into a resident `vocab × d` table gradient — row `v`
    /// is the sum of `dout` rows whose token is `v`. Deterministic, no atomics.
    /// Matches [`crate::ops::cpu_embed_backward`].
    pub fn embed_backward_resident(
        &self,
        tokens: &[u32],
        dout: &GpuMatrix,
        vocab: usize,
    ) -> BackendResult<GpuMatrix> {
        let rows = tokens.len();
        let d = dout.cols;
        if dout.rows != rows
        {
            return Err(BackendError::ShapeMismatch(format!(
                "embed_backward: dout has {} rows, expected {rows} tokens",
                dout.rows
            )));
        }
        let elems = vocab * d;
        let bytes = (elems.max(1) * std::mem::size_of::<f32>()) as u64;
        let dtable = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("embed-bwd-dtable"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if elems > 0 && rows > 0
        {
            let tok_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("embed-bwd-tokens"),
                    contents: bytemuck::cast_slice(tokens),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let params: [u32; 4] = [rows as u32, d as u32, vocab as u32, 0];
            let p_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("embed-bwd-params"),
                    contents: bytemuck::cast_slice(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("embed-bwd"),
                layout: &self.embed_bwd_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: tok_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dout.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: dtable.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: p_buf.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("embed-bwd"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("embed-bwd"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.embed_bwd_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(flat_workgroups(elems as u32), 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
        }
        Ok(GpuMatrix {
            buf: dtable,
            rows: vocab,
            cols: d,
        })
    }

    /// Backward of scale + causal mask: `din = scale·dout` at kept positions,
    /// `0` above the diagonal. Matches [`crate::ops::cpu_scale_causal_mask_backward`].
    pub fn scale_causal_mask_backward_resident(
        &self,
        dout: &GpuMatrix,
        scale: f32,
        causal: bool,
    ) -> BackendResult<GpuMatrix> {
        let elems = dout.rows * dout.cols;
        let bytes = (elems.max(1) * std::mem::size_of::<f32>()) as u64;
        let din = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mask-bwd-din"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if elems > 0
        {
            let params: [u32; 4] = [
                dout.rows as u32,
                dout.cols as u32,
                causal as u32,
                scale.to_bits(),
            ];
            let p_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mask-bwd-params"),
                    contents: bytemuck::cast_slice(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mask-bwd"),
                layout: &self.mask_bwd_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: dout.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: din.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: p_buf.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("mask-bwd"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("mask-bwd"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.mask_bwd_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(
                    (dout.cols as u32).div_ceil(8),
                    (dout.rows as u32).div_ceil(8),
                    1,
                );
            }
            self.queue.submit(Some(encoder.finish()));
        }
        Ok(GpuMatrix {
            buf: din,
            rows: dout.rows,
            cols: dout.cols,
        })
    }

    /// Resident elementwise op: `op` is `0=add`, `1=mul` (binary), `2=relu`
    /// (unary), `3=swiglu` (binary: `silu(a)·b`). For binary ops `a` and `b`
    /// must share a shape; the result stays in VRAM. For relu, pass `b = a`
    /// (it is ignored).
    pub fn ew_resident(&self, a: &GpuMatrix, b: &GpuMatrix, op: u32) -> BackendResult<GpuMatrix> {
        if op != 2 && (a.rows != b.rows || a.cols != b.cols)
        {
            return Err(BackendError::ShapeMismatch(format!(
                "elementwise: {}×{} vs {}×{}",
                a.rows, a.cols, b.rows, b.cols
            )));
        }
        let n = a.rows * a.cols;
        let bytes = (n.max(1) * std::mem::size_of::<f32>()) as u64;
        let c_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ew-c"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if n > 0
        {
            let params: [u32; 4] = [n as u32, op, 0, 0];
            let p_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("ew-params"),
                    contents: bytemuck::cast_slice(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ew"),
                layout: &self.ew_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: c_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: p_buf.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ew") });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ew"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.ew_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(flat_workgroups(n as u32), 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
        }
        Ok(GpuMatrix {
            buf: c_buf,
            rows: a.rows,
            cols: a.cols,
        })
    }

    /// Access to the raw wgpu Device (for tensor, fusion, conv_gpu modules).
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Access to the raw wgpu Queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Name of the underlying adapter (e.g. `"llvmpipe (LLVM 20, 256 bits)"`).
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// Download a storage buffer back to CPU.
    pub fn download_buffer(
        &self,
        buf: &wgpu::Buffer,
        elems: usize,
        bytes: u64,
    ) -> BackendResult<Vec<f32>> {
        if elems == 0
        {
            return Ok(Vec::new());
        }
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("download"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("download"),
            });
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| BackendError::Unavailable("wgpu"))?
            .map_err(|_| BackendError::Unavailable("wgpu"))?;
        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(out)
    }

    /// Encode + submit one GEMM dispatch (public for tensor.rs / fusion.rs).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_gemm(
        &self,
        a_buf: &wgpu::Buffer,
        b_buf: &wgpu::Buffer,
        c_buf: &wgpu::Buffer,
        m: usize,
        k: usize,
        n: usize,
        ta: bool,
        tb: bool,
        alpha: f32,
        beta: f32,
    ) {
        // Reuse the private encode_gemm logic
        self._encode_gemm(a_buf, b_buf, c_buf, m, k, n, ta, tb, alpha, beta)
    }

    /// Elementwise resident op that takes GpuTensor-compatible buffers.
    pub fn ew_resident_tensor(
        &self,
        a: &crate::tensor::GpuTensor,
        b: &crate::tensor::GpuTensor,
        op: u32,
    ) -> BackendResult<crate::tensor::GpuTensor> {
        if op != 2 && (a.rows != b.rows || a.cols != b.cols)
        {
            return Err(BackendError::ShapeMismatch(format!(
                "elementwise shape mismatch: {}×{} vs {}×{}",
                a.rows, a.cols, b.rows, b.cols
            )));
        }
        let n = a.elems;
        let bytes = (n.max(1) * std::mem::size_of::<f32>()) as u64;
        let c_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ew-c"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if n > 0
        {
            let params: [u32; 4] = [n as u32, op, 0, 0];
            let p_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("ew-params"),
                    contents: bytemuck::cast_slice(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ew"),
                layout: &self.ew_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: c_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: p_buf.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ew") });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ew"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.ew_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(flat_workgroups(n as u32), 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
        }
        Ok(crate::tensor::GpuTensor {
            buf: std::sync::Arc::new(c_buf),
            rows: a.rows,
            cols: a.cols,
            elems: n,
        })
    }

    /// `C = alpha·op(A)·op(B) + beta·C`, writing the result back into `c`.
    ///
    /// `op(A)` is `m×k`, `op(B)` is `k×n`, `C` is `m×n`. When `ta`/`tb` is set,
    /// the stored `a`/`b` buffer is the transpose of the corresponding operand.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &self,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
        ta: bool,
        tb: bool,
    ) -> BackendResult<()> {
        if m == 0 || n == 0
        {
            return Ok(());
        }
        if k == 0
        {
            // No contraction: C = beta·C (handled on the host, no GPU work).
            for v in c.iter_mut()
            {
                *v *= beta;
            }
            return Ok(());
        }

        let bytes = (m * n * std::mem::size_of::<f32>()) as u64;
        let a_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("a"),
                contents: bytemuck::cast_slice(a),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let b_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("b"),
                contents: bytemuck::cast_slice(b),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let c_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("c"),
                contents: bytemuck::cast_slice(c),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let params: [u32; 8] = [
            m as u32,
            k as u32,
            n as u32,
            ta as u32,
            tb as u32,
            alpha.to_bits(),
            beta.to_bits(),
            0,
        ];
        let p_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gemm"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gemm"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gemm"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((m as u32).div_ceil(8), (n as u32).div_ceil(8), 1);
        }
        encoder.copy_buffer_to_buffer(&c_buf, 0, &staging, 0, bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| BackendError::Unavailable("wgpu"))?
            .map_err(|_| BackendError::Unavailable("wgpu"))?;

        let data = slice.get_mapped_range();
        c.copy_from_slice(bytemuck::cast_slice(&data));
        drop(data);
        staging.unmap();
        Ok(())
    }

    /// Upload a row-major `rows×cols` matrix to a resident GPU storage buffer.
    pub fn upload(&self, data: &[f32], rows: usize, cols: usize) -> GpuMatrix {
        // wgpu rejects zero-sized buffers; back an empty matrix with a 4-byte
        // placeholder so the handle stays valid (`download` short-circuits empties).
        let buf = if data.is_empty()
        {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("resident-empty"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        }
        else
        {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("resident"),
                    contents: bytemuck::cast_slice(data),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                })
        };
        GpuMatrix { buf, rows, cols }
    }

    /// `C = op(A)·op(B)` with both operands already resident; the result **stays
    /// in VRAM** (no download). `ta`/`tb` request a transpose of `a`/`b`. This
    /// is what keeps activations device-resident across a chain of GEMMs.
    pub fn gemm_resident(
        &self,
        a: &GpuMatrix,
        b: &GpuMatrix,
        ta: bool,
        tb: bool,
    ) -> BackendResult<GpuMatrix> {
        let m = if ta { a.cols } else { a.rows };
        let k = if ta { a.rows } else { a.cols };
        let n = if tb { b.rows } else { b.cols };
        let kb = if tb { b.cols } else { b.rows };
        if k != kb
        {
            return Err(BackendError::ShapeMismatch(format!(
                "inner dims disagree: op(A) is {m}×{k}, op(B) is {kb}×{n}"
            )));
        }
        // Never create a zero-sized buffer (wgpu rejects it). For a degenerate
        // result (`m`/`n`/`k == 0`) the zero-initialised buffer already holds
        // the correct empty/all-zeros matrix, so we skip the dispatch entirely.
        let elems = m * n;
        let bytes = (elems.max(1) * std::mem::size_of::<f32>()) as u64;
        let c_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resident-c"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Fresh result: alpha=1, beta=0. wgpu zero-initialises c_buf, so the
        // `beta·C` term reads valid zeros.
        if m != 0 && n != 0 && k != 0
        {
            self._encode_gemm(&a.buf, &b.buf, &c_buf, m, k, n, ta, tb, 1.0, 0.0);
        }
        Ok(GpuMatrix {
            buf: c_buf,
            rows: m,
            cols: n,
        })
    }

    /// Download a resident matrix back to a CPU `Vec<f32>` (row-major).
    pub fn download(&self, mat: &GpuMatrix) -> BackendResult<Vec<f32>> {
        let elems = mat.rows * mat.cols;
        if elems == 0
        {
            return Ok(Vec::new());
        }
        let bytes = (elems * std::mem::size_of::<f32>()) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("download"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("download"),
            });
        encoder.copy_buffer_to_buffer(&mat.buf, 0, &staging, 0, bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| BackendError::Unavailable("wgpu"))?
            .map_err(|_| BackendError::Unavailable("wgpu"))?;
        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(out)
    }

    /// Encode + submit one GEMM dispatch into the given buffers (no download).
    #[allow(clippy::too_many_arguments)]
    fn _encode_gemm(
        &self,
        a_buf: &wgpu::Buffer,
        b_buf: &wgpu::Buffer,
        c_buf: &wgpu::Buffer,
        m: usize,
        k: usize,
        n: usize,
        ta: bool,
        tb: bool,
        alpha: f32,
        beta: f32,
    ) {
        let params: [u32; 8] = [
            m as u32,
            k as u32,
            n as u32,
            ta as u32,
            tb as u32,
            alpha.to_bits(),
            beta.to_bits(),
            0,
        ];
        let p_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gemm"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gemm"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gemm"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((m as u32).div_ceil(8), (n as u32).div_ceil(8), 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }
}

/// One-shot row-major `C = A·B`. Acquires a fresh [`WgpuContext`]; for repeated
/// calls (e.g. an autograd backward pass) prefer a cached [`crate::WgpuEngine`].
pub fn wgpu_gemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> BackendResult<Vec<f32>> {
    if m == 0 || n == 0
    {
        return Ok(Vec::new());
    }
    let mut c = vec![0.0f32; m * n];
    if k == 0
    {
        return Ok(c);
    }
    WgpuContext::new()?.gemm(1.0, a, b, 0.0, &mut c, m, k, n, false, false)?;
    Ok(c)
}

/// One-shot row-wise softmax over a `rows × cols` matrix. Acquires a fresh
/// [`WgpuContext`]; for repeated calls prefer holding a context.
pub fn wgpu_softmax(data: &[f32], rows: usize, cols: usize) -> BackendResult<Vec<f32>> {
    WgpuContext::new()?.softmax_rows(data, rows, cols)
}

/// One-shot scale + causal mask over a `rows × cols` score matrix. Acquires a
/// fresh [`WgpuContext`]; for repeated calls prefer holding a context.
pub fn wgpu_scale_causal_mask(
    scores: &[f32],
    rows: usize,
    cols: usize,
    scale: f32,
    causal: bool,
) -> BackendResult<Vec<f32>> {
    WgpuContext::new()?.scale_causal_mask(scores, rows, cols, scale, causal)
}

#[cfg(test)]
mod tests {
    use crate::{CpuBackend, GpuAccelerator, RawComputeBackend, WgpuBackend};

    /// Maximum |gpu - cpu| relative to the CPU Frobenius norm. GPU accumulation
    /// is not bit-identical to the scalar oracle, so we assert a tolerance.
    fn rel_err(gpu: &[f32], cpu: &[f32]) -> f32 {
        let num: f32 = gpu
            .iter()
            .zip(cpu)
            .map(|(g, c)| (g - c) * (g - c))
            .sum::<f32>()
            .sqrt();
        let den: f32 = cpu.iter().map(|c| c * c).sum::<f32>().sqrt().max(1e-30);
        num / den
    }

    /// If no adapter is available in this environment, skip rather than fail —
    /// CI provides a software Vulkan adapter (lavapipe) so the assertion path
    /// is actually exercised there.
    fn run_or_skip(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Option<Vec<f32>> {
        match super::wgpu_gemm(a, b, m, k, n)
        {
            Ok(v) => Some(v),
            Err(crate::BackendError::Unavailable(_)) =>
            {
                eprintln!("wgpu: no adapter available, skipping");
                None
            },
            Err(e) => panic!("unexpected error: {e:?}"),
        }
    }

    #[test]
    fn wgpu_gemm_matches_cpu_oracle() {
        // A (3×4) · B (4×2), values chosen to be non-trivial.
        let a: Vec<f32> = (0..12).map(|i| (i as f32 * 0.5 - 2.0).sin()).collect();
        let b: Vec<f32> = (0..8).map(|i| (i as f32 * 0.3 + 1.0).cos()).collect();
        if let Some(gpu) = run_or_skip(&a, &b, 3, 4, 2)
        {
            let cpu = CpuBackend.gemm_f32(&a, &b, 3, 4, 2).unwrap();
            assert_eq!(gpu.len(), cpu.len());
            assert!(rel_err(&gpu, &cpu) < 1e-4, "gpu={gpu:?} cpu={cpu:?}");
        }
    }

    #[test]
    fn wgpu_gemm_identity_roundtrip() {
        let a = [1.0f32, 2.0, 3.0, 4.0]; // 2×2
        let id = [1.0f32, 0.0, 0.0, 1.0];
        if let Some(gpu) = run_or_skip(&a, &id, 2, 2, 2)
        {
            assert!(rel_err(&gpu, &a) < 1e-5);
        }
    }

    #[test]
    fn wgpu_backend_wired_under_feature() {
        // With the feature on, the WgpuBackend dispatches to the real path:
        // either a correct result (adapter present) or an honest Unavailable.
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let id = [1.0f32, 0.0, 0.0, 1.0];
        match WgpuBackend.gemm_f32(&a, &id, 2, 2, 2)
        {
            Ok(v) =>
            {
                let cpu = CpuBackend.gemm_f32(&a, &id, 2, 2, 2).unwrap();
                assert!(rel_err(&v, &cpu) < 1e-5);
                assert_eq!(GpuAccelerator::Wgpu(WgpuBackend).device_name(), "wgpu");
            },
            Err(crate::BackendError::Unavailable("wgpu")) =>
            {},
            Err(e) => panic!("unexpected: {e:?}"),
        }
    }

    /// GPU row-wise softmax must match the CPU oracle. Exercised against
    /// lavapipe in CI and the real GPU on-device; skipped where no adapter is
    /// present (this dev container has no Vulkan ICD).
    #[test]
    fn wgpu_softmax_matches_cpu_oracle() {
        let (rows, cols) = (4usize, 7usize);
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.37 - 3.0).sin() * 2.0)
            .collect();
        match super::wgpu_softmax(&data, rows, cols)
        {
            Ok(gpu) =>
            {
                let cpu = crate::ops::cpu_softmax(&data, rows, cols);
                assert_eq!(gpu.len(), cpu.len());
                assert!(rel_err(&gpu, &cpu) < 1e-4, "gpu={gpu:?} cpu={cpu:?}");
                // Each row is a probability distribution.
                for r in 0..rows
                {
                    let s: f32 = gpu[r * cols..r * cols + cols].iter().sum();
                    assert!((s - 1.0).abs() < 1e-3, "row {r} sums to {s}");
                }
            },
            Err(crate::BackendError::Unavailable(_)) =>
            {
                eprintln!("wgpu: no adapter available, skipping softmax parity");
            },
            Err(e) => panic!("unexpected error: {e:?}"),
        }
    }

    /// GPU scale + causal mask must match the CPU oracle, including the exact
    /// `-1e30` sentinel written above the diagonal. Exercised on lavapipe in CI
    /// and the real GPU on-device; skipped where no adapter is present.
    #[test]
    fn wgpu_scale_causal_mask_matches_cpu_oracle() {
        let (rows, cols) = (5usize, 5usize);
        let scores: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.21 - 2.0).cos() * 3.0)
            .collect();
        let scale = 0.125_f32; // 1/sqrt(64), a realistic 1/sqrt(head_dim).
        match super::wgpu_scale_causal_mask(&scores, rows, cols, scale, true)
        {
            Ok(gpu) =>
            {
                let cpu = crate::ops::cpu_scale_causal_mask(&scores, rows, cols, scale, true);
                assert_eq!(gpu.len(), cpu.len());
                // Above-diagonal entries are the exact sentinel on both paths.
                for i in 0..rows
                {
                    for j in 0..cols
                    {
                        let idx = i * cols + j;
                        if j > i
                        {
                            assert_eq!(gpu[idx], crate::ops::MASK_NEG, "masked ({i},{j})");
                        }
                        else
                        {
                            assert!(
                                (gpu[idx] - cpu[idx]).abs() < 1e-5,
                                "kept ({i},{j}): gpu={} cpu={}",
                                gpu[idx],
                                cpu[idx]
                            );
                        }
                    }
                }
                // Non-causal path is a pure scale.
                let gpu_ns =
                    super::wgpu_scale_causal_mask(&scores, rows, cols, scale, false).unwrap();
                let cpu_ns = crate::ops::cpu_scale_causal_mask(&scores, rows, cols, scale, false);
                assert!(rel_err(&gpu_ns, &cpu_ns) < 1e-5);
            },
            Err(crate::BackendError::Unavailable(_)) =>
            {
                eprintln!("wgpu: no adapter available, skipping mask parity");
            },
            Err(e) => panic!("unexpected error: {e:?}"),
        }
    }

    /// The general kernel must honour transpose + alpha/beta, matching a hand
    /// computation. C = 2·Aᵀ·B + 0.5·C0.
    #[test]
    fn wgpu_gemm_transpose_alpha_beta() {
        let ctx = match super::WgpuContext::new()
        {
            Ok(c) => c,
            Err(_) =>
            {
                eprintln!("wgpu: no adapter, skipping");
                return;
            },
        };
        // op(A) = Aᵀ where stored A is k×m. Take A stored as 2×2 → op(A) 2×2.
        // stored a (k×m = 2×2) = [[1,2],[3,4]] → op(A)=Aᵀ=[[1,3],[2,4]]
        // b (k×n = 2×2) = [[1,0],[0,1]] (identity) → op(A)·op(B) = op(A)
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [1.0f32, 0.0, 0.0, 1.0];
        let mut c = [10.0f32, 20.0, 30.0, 40.0];
        ctx.gemm(2.0, &a, &b, 0.5, &mut c, 2, 2, 2, true, false)
            .unwrap();
        // 2·[[1,3],[2,4]] + 0.5·[[10,20],[30,40]] = [[7,16],[19,28]]
        let expected = [7.0f32, 16.0, 19.0, 28.0];
        let err: f32 = c
            .iter()
            .zip(expected.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        assert!(err < 1e-3, "got {c:?}");
    }
}
