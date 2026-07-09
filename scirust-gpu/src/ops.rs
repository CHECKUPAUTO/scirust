//! Extended GPU operations: activations, reductions, normalisations.

use crate::kernels::EwOp;

/// Apply an activation function elementwise (CPU reference, deterministic).
pub fn cpu_activation(data: &[f32], op: EwOp) -> Vec<f32> {
    data.iter()
        .map(|&x| match op
        {
            EwOp::Relu => x.max(0.0),
            EwOp::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            EwOp::Tanh => x.tanh(),
            EwOp::Gelu =>
            {
                let c = (2.0 / std::f32::consts::PI).sqrt();
                0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
            },
            EwOp::Silu => x / (1.0 + (-x).exp()),
            EwOp::LeakyRelu =>
            {
                if x >= 0.0
                {
                    x
                }
                else
                {
                    0.01 * x
                }
            },
            EwOp::Elu =>
            {
                if x >= 0.0
                {
                    x
                }
                else
                {
                    1.0 * (x.exp() - 1.0)
                }
            },
            EwOp::Softplus => (1.0 + x.exp()).ln(),
            EwOp::Sqrt => x.max(0.0).sqrt(),
            EwOp::Exp => x.exp(),
        })
        .collect()
}

/// CPU reference for deterministic reduction along the last axis.
#[allow(clippy::needless_range_loop)]
pub fn cpu_reduce_sum(data: &[f32], outer: usize, axis_size: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; outer];
    for i in 0..outer
    {
        let start = i * axis_size;
        out[i] = data[start..start + axis_size].iter().sum();
    }
    out
}

/// CPU reference for mean reduction along the last axis.
pub fn cpu_reduce_mean(data: &[f32], outer: usize, axis_size: usize) -> Vec<f32> {
    if axis_size == 0
    {
        return vec![0.0; outer];
    }
    let sums = cpu_reduce_sum(data, outer, axis_size);
    sums.iter().map(|&s| s / axis_size as f32).collect()
}

/// CPU reference for max reduction along the last axis.
#[allow(clippy::needless_range_loop)]
pub fn cpu_reduce_max(data: &[f32], outer: usize, axis_size: usize) -> Vec<f32> {
    if axis_size == 0
    {
        return vec![f32::NEG_INFINITY; outer];
    }
    let mut out = vec![f32::NEG_INFINITY; outer];
    for i in 0..outer
    {
        let start = i * axis_size;
        for k in 0..axis_size
        {
            out[i] = out[i].max(data[start + k]);
        }
    }
    out
}

/// CPU reference for L2 norm reduction along the last axis.
#[allow(clippy::needless_range_loop)]
pub fn cpu_reduce_norm(data: &[f32], outer: usize, axis_size: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; outer];
    for i in 0..outer
    {
        let start = i * axis_size;
        out[i] = data[start..start + axis_size]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
    }
    out
}

/// CPU reference for LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta.
pub fn cpu_layer_norm(
    data: &[f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for r in 0..rows
    {
        let start = r * cols;
        let slice = &data[start..start + cols];
        let mean: f32 = slice.iter().sum::<f32>() / cols as f32;
        let var: f32 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / cols as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for c in 0..cols
        {
            out[start + c] = (data[start + c] - mean) * inv_std * gamma[c] + beta[c];
        }
    }
    out
}

/// CPU reference for RMSNorm: x / sqrt(mean(x^2) + eps) * weight.
pub fn cpu_rms_norm(data: &[f32], weight: &[f32], eps: f32, rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for r in 0..rows
    {
        let start = r * cols;
        let slice = &data[start..start + cols];
        let rms: f32 = (slice.iter().map(|x| x * x).sum::<f32>() / cols as f32 + eps).sqrt();
        for c in 0..cols
        {
            out[start + c] = (data[start + c] / rms) * weight[c];
        }
    }
    out
}

/// CPU reference for row-wise softmax: `exp(x - rowmax) / sum(exp(x - rowmax))`,
/// max-subtracted for numerical stability. The correctness contract for the GPU
/// `softmax_rows` kernel and the missing transformer-attention primitive.
pub fn cpu_softmax(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for r in 0..rows
    {
        let start = r * cols;
        let row = &data[start..start + cols];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for &x in row
        {
            sum += (x - m).exp();
        }
        for c in 0..cols
        {
            out[start + c] = (data[start + c] - m).exp() / sum;
        }
    }
    out
}

/// Large negative sentinel written into causally-masked score entries. Big
/// enough that `exp(MASK_NEG - rowmax)` underflows to 0 in the following
/// softmax, but finite (not `-inf`) so parity checks stay numeric.
pub const MASK_NEG: f32 = -1.0e30;

/// CPU reference for the pre-softmax attention step: scale the `rows × cols`
/// score matrix by `scale`, and — when `causal` — replace every entry above the
/// diagonal (key `j > i` for query `i`) with [`MASK_NEG`]. The GPU
/// `scale_causal_mask` kernel's correctness contract.
pub fn cpu_scale_causal_mask(
    scores: &[f32],
    rows: usize,
    cols: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let mut out = vec![0.0f32; scores.len()];
    for i in 0..rows
    {
        for j in 0..cols
        {
            out[i * cols + j] = if causal && j > i
            {
                MASK_NEG
            }
            else
            {
                scores[i * cols + j] * scale
            };
        }
    }
    out
}

/// CPU reference for the softmax backward: given the forward output `y` and
/// upstream grad `dy`, `dx = y ⊙ (dy − Σⱼ dyⱼyⱼ)` per row. The GPU
/// `softmax_backward_resident` kernel's correctness contract.
pub fn cpu_softmax_backward(y: &[f32], dy: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut dx = vec![0.0f32; y.len()];
    for r in 0..rows
    {
        let base = r * cols;
        let s: f32 = (0..cols).map(|j| dy[base + j] * y[base + j]).sum();
        for j in 0..cols
        {
            dx[base + j] = y[base + j] * (dy[base + j] - s);
        }
    }
    dx
}

/// CPU reference for the SwiGLU-gate backward of `c = silu(a) ⊙ b`: returns
/// `(da, db)` with `da = dc·silu'(a)·b`, `db = dc·silu(a)`, where
/// `silu'(x) = σ(x)·(1 + x·(1−σ(x)))`. The GPU `swiglu_backward_resident`
/// kernel's correctness contract.
pub fn cpu_swiglu_backward(a: &[f32], b: &[f32], dc: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut da = vec![0.0f32; a.len()];
    let mut db = vec![0.0f32; a.len()];
    for i in 0..a.len()
    {
        let sig = 1.0 / (1.0 + (-a[i]).exp());
        let silu = a[i] * sig;
        let dsilu = sig * (1.0 + a[i] * (1.0 - sig));
        da[i] = dc[i] * dsilu * b[i];
        db[i] = dc[i] * silu;
    }
    (da, db)
}

/// CPU reference for the RMSNorm input-gradient backward. Given `x`, the `cols`
/// gain `weight`, upstream grad `dy` and `eps`,
/// `dx_j = (dy_j·w_j)/rms − x_j·(Σₖ dyₖwₖxₖ)/(d·rms³)` per row, where
/// `rms = √(mean(x²)+eps)`. The GPU `rms_norm_backward_resident` contract.
#[allow(clippy::needless_range_loop)]
pub fn cpu_rms_norm_backward(
    x: &[f32],
    weight: &[f32],
    dy: &[f32],
    eps: f32,
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let mut dx = vec![0.0f32; x.len()];
    for r in 0..rows
    {
        let base = r * cols;
        let ms = x[base..base + cols].iter().map(|v| v * v).sum::<f32>() / cols as f32 + eps;
        let rms = ms.sqrt();
        let dot: f32 = (0..cols)
            .map(|j| dy[base + j] * weight[j] * x[base + j])
            .sum();
        let coef = dot / (cols as f32 * ms * rms);
        for j in 0..cols
        {
            dx[base + j] = dy[base + j] * weight[j] / rms - x[base + j] * coef;
        }
    }
    dx
}

/// CPU reference for the RMSNorm **gain** gradient — the oracle for
/// [`crate::GpuChain::rms_norm_gain_backward`]. `dweight[j] = Σ_r dy[r,j] ·
/// x[r,j] / rms[r]` (`rms[r] = sqrt(mean_j x[r,j]² + eps)`); result length `cols`.
pub fn cpu_rms_norm_gain_backward(
    x: &[f32],
    dy: &[f32],
    eps: f32,
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let mut dweight = vec![0.0f32; cols];
    for r in 0..rows
    {
        let base = r * cols;
        let ms = x[base..base + cols].iter().map(|v| v * v).sum::<f32>() / cols as f32 + eps;
        let rms = ms.sqrt();
        for j in 0..cols
        {
            dweight[j] += dy[base + j] * x[base + j] / rms;
        }
    }
    dweight
}

/// CPU reference for a **LoRA-adapted linear** forward — the oracle for
/// [`crate::GpuChain::lora_linear_forward`]. `x` is `m×in`, base `w` is `in×out`
/// (row-major, frozen), `a` is `in×r`, `b` is `r×out`; returns the `m×out`
/// `y = x·W + scaling·(x·A)·B`.
#[allow(clippy::too_many_arguments)]
pub fn cpu_lora_linear(
    x: &[f32],
    w: &[f32],
    a: &[f32],
    b: &[f32],
    scaling: f32,
    m: usize,
    in_f: usize,
    r: usize,
    out: usize,
) -> Vec<f32> {
    // xa = x·A  (m×r)
    let mut xa = vec![0.0f32; m * r];
    for i in 0..m
    {
        for kk in 0..r
        {
            let mut s = 0.0f32;
            for p in 0..in_f
            {
                s += x[i * in_f + p] * a[p * r + kk];
            }
            xa[i * r + kk] = s;
        }
    }
    let mut y = vec![0.0f32; m * out];
    for i in 0..m
    {
        for j in 0..out
        {
            let mut base = 0.0f32;
            for p in 0..in_f
            {
                base += x[i * in_f + p] * w[p * out + j];
            }
            let mut xab = 0.0f32;
            for kk in 0..r
            {
                xab += xa[i * r + kk] * b[kk * out + j];
            }
            y[i * out + j] = base + scaling * xab;
        }
    }
    y
}

/// CPU reference for the **LoRA-adapted linear backward** — the oracle for
/// [`crate::GpuChain::lora_linear_backward`]. Returns `(dx, dA, dB)` with shapes
/// `m×in`, `in×r`, `r×out`; the base `W` is frozen (no gradient). For
/// `delta = scaling·(x·A·B)`: `dB = (x·A)ᵀ·(scaling·dy)`,
/// `dA = xᵀ·((scaling·dy)·Bᵀ)`, `dx = dy·Wᵀ + ((scaling·dy)·Bᵀ)·Aᵀ`.
#[allow(clippy::too_many_arguments)]
pub fn cpu_lora_linear_backward(
    x: &[f32],
    w: &[f32],
    a: &[f32],
    b: &[f32],
    dy: &[f32],
    scaling: f32,
    m: usize,
    in_f: usize,
    r: usize,
    out: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // xa = x·A  (m×r)
    let mut xa = vec![0.0f32; m * r];
    for i in 0..m
    {
        for kk in 0..r
        {
            let mut s = 0.0f32;
            for p in 0..in_f
            {
                s += x[i * in_f + p] * a[p * r + kk];
            }
            xa[i * r + kk] = s;
        }
    }
    // d_xa = (scaling·dy)·Bᵀ  (m×r)
    let mut d_xa = vec![0.0f32; m * r];
    for i in 0..m
    {
        for kk in 0..r
        {
            let mut s = 0.0f32;
            for j in 0..out
            {
                s += scaling * dy[i * out + j] * b[kk * out + j];
            }
            d_xa[i * r + kk] = s;
        }
    }
    // dB = (x·A)ᵀ·(scaling·dy)  (r×out)
    let mut db = vec![0.0f32; r * out];
    for kk in 0..r
    {
        for j in 0..out
        {
            let mut s = 0.0f32;
            for i in 0..m
            {
                s += xa[i * r + kk] * scaling * dy[i * out + j];
            }
            db[kk * out + j] = s;
        }
    }
    // dA = xᵀ·d_xa  (in×r)
    let mut da = vec![0.0f32; in_f * r];
    for p in 0..in_f
    {
        for kk in 0..r
        {
            let mut s = 0.0f32;
            for i in 0..m
            {
                s += x[i * in_f + p] * d_xa[i * r + kk];
            }
            da[p * r + kk] = s;
        }
    }
    // dx = dy·Wᵀ + d_xa·Aᵀ  (m×in)
    let mut dx = vec![0.0f32; m * in_f];
    for i in 0..m
    {
        for p in 0..in_f
        {
            let mut s = 0.0f32;
            for j in 0..out
            {
                s += dy[i * out + j] * w[p * out + j];
            }
            for kk in 0..r
            {
                s += d_xa[i * r + kk] * a[p * r + kk];
            }
            dx[i * in_f + p] = s;
        }
    }
    (dx, da, db)
}

/// CPU reference for a **DoRA-adapted linear** forward — the oracle for
/// [`crate::GpuChain::dora_linear_forward`]. `x` is `m×in`, base `w0` is `in×out`
/// (row-major, frozen), `a` is `in×r`, `b` is `r×out`, `mag` is length `in`
/// (per-row magnitude). Returns `m×out` `y = x·W'` with
/// `W' = mag ⊙ V/‖V‖_row`, `V = w0 + a·b` (per-row norm over the `out` columns,
/// guarded by `1/√max(·, 1e-12)` to mirror the `rsqrt` kernel).
#[allow(clippy::too_many_arguments)]
pub fn cpu_dora_linear(
    x: &[f32],
    w0: &[f32],
    a: &[f32],
    b: &[f32],
    mag: &[f32],
    m: usize,
    in_f: usize,
    r: usize,
    out: usize,
) -> Vec<f32> {
    let (v, rnr) = dora_v_rnr(w0, a, b, in_f, r, out);
    // W' = mag ⊙ V/‖V‖_row.
    let mut wp = vec![0.0f32; in_f * out];
    for p in 0..in_f
    {
        for o in 0..out
        {
            wp[p * out + o] = mag[p] * v[p * out + o] * rnr[p];
        }
    }
    // y = x·W'.
    let mut y = vec![0.0f32; m * out];
    for i in 0..m
    {
        for o in 0..out
        {
            let mut s = 0.0f32;
            for p in 0..in_f
            {
                s += x[i * in_f + p] * wp[p * out + o];
            }
            y[i * out + o] = s;
        }
    }
    y
}

/// `V = w0 + a·b` (`in×out`) and the guarded reciprocal per-row norms `rnr`
/// (`1/√max(Σ_o V², 1e-12)`, length `in`). Shared by the DoRA oracles.
fn dora_v_rnr(
    w0: &[f32],
    a: &[f32],
    b: &[f32],
    in_f: usize,
    r: usize,
    out: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut v = vec![0.0f32; in_f * out];
    for p in 0..in_f
    {
        for o in 0..out
        {
            let mut s = w0[p * out + o];
            for k in 0..r
            {
                s += a[p * r + k] * b[k * out + o];
            }
            v[p * out + o] = s;
        }
    }
    let mut rnr = vec![0.0f32; in_f];
    for p in 0..in_f
    {
        let mut ss = 0.0f32;
        for o in 0..out
        {
            ss += v[p * out + o] * v[p * out + o];
        }
        rnr[p] = 1.0 / ss.max(1e-12).sqrt();
    }
    (v, rnr)
}

/// CPU reference for the **DoRA-adapted linear backward** — the oracle for
/// [`crate::GpuChain::dora_linear_backward`]. Returns `(dx, dA, dB, dm)` with
/// shapes `m×in`, `in×r`, `r×out`, `in`; `w0` is frozen. With `dW' = xᵀ·dy`,
/// `u = V/‖V‖_row`, `s = Σ_o dW'·u` (per row): `dm = s`,
/// `dV = (mag/‖V‖_row)·(dW' − u·s)`, `dA = dV·Bᵀ`, `dB = Aᵀ·dV`, `dx = dy·W'ᵀ`.
#[allow(clippy::too_many_arguments)]
pub fn cpu_dora_linear_backward(
    x: &[f32],
    w0: &[f32],
    a: &[f32],
    b: &[f32],
    mag: &[f32],
    dy: &[f32],
    m: usize,
    in_f: usize,
    r: usize,
    out: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let (v, rnr) = dora_v_rnr(w0, a, b, in_f, r, out);
    let mut wp = vec![0.0f32; in_f * out];
    for p in 0..in_f
    {
        for o in 0..out
        {
            wp[p * out + o] = mag[p] * v[p * out + o] * rnr[p];
        }
    }
    // dW' = xᵀ·dy  (in×out).
    let mut dwp = vec![0.0f32; in_f * out];
    for p in 0..in_f
    {
        for o in 0..out
        {
            let mut s = 0.0f32;
            for i in 0..m
            {
                s += x[i * in_f + p] * dy[i * out + o];
            }
            dwp[p * out + o] = s;
        }
    }
    // dx = dy·W'ᵀ  (m×in).
    let mut dx = vec![0.0f32; m * in_f];
    for i in 0..m
    {
        for p in 0..in_f
        {
            let mut s = 0.0f32;
            for o in 0..out
            {
                s += dy[i * out + o] * wp[p * out + o];
            }
            dx[i * in_f + p] = s;
        }
    }
    // dm[p] = Σ_o dW'[p,o]·u[p,o], u = V/‖V‖_row.
    let mut dm = vec![0.0f32; in_f];
    for p in 0..in_f
    {
        let mut sp = 0.0f32;
        for o in 0..out
        {
            sp += dwp[p * out + o] * (v[p * out + o] * rnr[p]);
        }
        dm[p] = sp;
    }
    // dV[p,o] = (mag/‖V‖_row)·(dW' − u·s).
    let mut dv = vec![0.0f32; in_f * out];
    for p in 0..in_f
    {
        for o in 0..out
        {
            let u = v[p * out + o] * rnr[p];
            dv[p * out + o] = (mag[p] * rnr[p]) * (dwp[p * out + o] - u * dm[p]);
        }
    }
    // dA = dV·Bᵀ  (in×r).
    let mut da = vec![0.0f32; in_f * r];
    for p in 0..in_f
    {
        for k in 0..r
        {
            let mut s = 0.0f32;
            for o in 0..out
            {
                s += dv[p * out + o] * b[k * out + o];
            }
            da[p * r + k] = s;
        }
    }
    // dB = Aᵀ·dV  (r×out).
    let mut db = vec![0.0f32; r * out];
    for k in 0..r
    {
        for o in 0..out
        {
            let mut s = 0.0f32;
            for p in 0..in_f
            {
                s += a[p * r + k] * dv[p * out + o];
            }
            db[k * out + o] = s;
        }
    }
    (dx, da, db, dm)
}

/// CPU reference for rotary position embedding — the bit-exact oracle for
/// [`crate::WgpuContext::rope_resident`]. Rotates the interleaved lane pair
/// `(2j, 2j+1)` of each row by `pos·freqⱼ`, with `pos = (row mod seq_len) +
/// offset` and `freqⱼ = theta^(-2j/dim)`:
/// `y[2j] = e·cos − o·sin`, `y[2j+1] = e·sin + o·cos` (`e=x[2j], o=x[2j+1]`).
/// This is exactly the sciagent model's RoPE (`GQAAttention::rope_apply`).
/// `dim` must be even.
pub fn cpu_rope(
    x: &[f32],
    rows: usize,
    dim: usize,
    seq_len: usize,
    offset: usize,
    theta: f32,
) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0.0f32; rows * dim];
    for r in 0..rows
    {
        let base = r * dim;
        let pos = ((r % seq_len.max(1)) + offset) as f32;
        for j in 0..half
        {
            let freq = theta.powf(-2.0 * j as f32 / dim as f32);
            let angle = pos * freq;
            let (s, c) = angle.sin_cos();
            let e = x[base + 2 * j];
            let o = x[base + 2 * j + 1];
            out[base + 2 * j] = e * c - o * s;
            out[base + 2 * j + 1] = e * s + o * c;
        }
    }
    out
}

/// CPU reference for the RoPE backward — the adjoint of the rotation is the
/// transpose rotation: `dx[2j] = cos·dy[2j] + sin·dy[2j+1]`,
/// `dx[2j+1] = −sin·dy[2j] + cos·dy[2j+1]`, same `pos`/`freq` as [`cpu_rope`].
/// The GPU `rope_backward_resident` contract.
pub fn cpu_rope_backward(
    dy: &[f32],
    rows: usize,
    dim: usize,
    seq_len: usize,
    offset: usize,
    theta: f32,
) -> Vec<f32> {
    let half = dim / 2;
    let mut dx = vec![0.0f32; rows * dim];
    for r in 0..rows
    {
        let base = r * dim;
        let pos = ((r % seq_len.max(1)) + offset) as f32;
        for j in 0..half
        {
            let freq = theta.powf(-2.0 * j as f32 / dim as f32);
            let angle = pos * freq;
            let (s, c) = angle.sin_cos();
            let ge = dy[base + 2 * j];
            let go = dy[base + 2 * j + 1];
            dx[base + 2 * j] = c * ge + s * go;
            dx[base + 2 * j + 1] = -s * ge + c * go;
        }
    }
    dx
}

/// CPU reference for a contiguous column-block gather — the oracle for
/// [`crate::WgpuContext::slice_cols_resident`]. `out[r, c] = x[r, col_start + c]`
/// for `c in 0..ncols`; `x` is `rows × src_cols`, `out` is `rows × ncols`.
pub fn cpu_slice_cols(
    x: &[f32],
    rows: usize,
    src_cols: usize,
    col_start: usize,
    ncols: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * ncols];
    for r in 0..rows
    {
        for c in 0..ncols
        {
            out[r * ncols + c] = x[r * src_cols + col_start + c];
        }
    }
    out
}

/// CPU reference for the column scatter — the oracle for
/// [`crate::WgpuContext::place_cols_resident`] and the adjoint of
/// [`cpu_slice_cols`]. `out[r, col_start + c] = x[r, c]`, `0` elsewhere; `x` is
/// `rows × ncols`, `out` is `rows × dst_cols`.
pub fn cpu_place_cols(
    x: &[f32],
    rows: usize,
    ncols: usize,
    col_start: usize,
    dst_cols: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * dst_cols];
    for r in 0..rows
    {
        for c in 0..ncols
        {
            out[r * dst_cols + col_start + c] = x[r * ncols + c];
        }
    }
    out
}

/// CPU reference for the vertical row-stack — the oracle for
/// [`crate::WgpuContext::concat_rows_resident`]. `a` is `a_rows × cols`, `b` is
/// `b_rows × cols`; returns `(a_rows + b_rows) × cols` with `a`'s rows first.
pub fn cpu_concat_rows(a: &[f32], b: &[f32], cols: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    out.extend_from_slice(a);
    out.extend_from_slice(b);
    debug_assert!(cols == 0 || (a.len() + b.len()).is_multiple_of(cols));
    out
}

/// CPU reference for the row-range gather — the oracle for
/// [`crate::WgpuContext::slice_rows_resident`]. `x` is `rows × cols`; returns the
/// contiguous `count × cols` block starting at row `start`.
pub fn cpu_slice_rows(x: &[f32], cols: usize, start: usize, count: usize) -> Vec<f32> {
    x[start * cols..(start + count) * cols].to_vec()
}

/// CPU reference for resident multi-head grouped-query attention, single
/// sequence (`rows = seq_len`) — the oracle for
/// [`crate::GpuChain::gqa_attention`]. Matches the sciagent model's attention
/// math exactly: full-width RoPE on `q` and `k` (each using **its own width** in
/// the frequency, as the model's `rope_on_tape` does — `d_model` for `q`,
/// `kv_dim` for `k`), then per head `kv = head / (n_heads/n_kv_heads)`,
/// `softmax((qs·ksᵀ)·(1/√dh) [+ causal]) · vs`, placed into the head's `d_model`
/// slot and summed. `q` is `[rows, n_heads·dh]`; `k`/`v` are `[rows, n_kv_heads·dh]`;
/// returns `[rows, n_heads·dh]` (the concatenated context, before `w_o`).
#[allow(clippy::too_many_arguments)]
pub fn cpu_gqa_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    rows: usize,
    n_heads: usize,
    n_kv_heads: usize,
    dh: usize,
    seq_len: usize,
    theta: f32,
    causal: bool,
) -> Vec<f32> {
    let d_model = n_heads * dh;
    let kv_dim = n_kv_heads * dh;
    let qr = cpu_rope(q, rows, d_model, seq_len, 0, theta);
    let kr = cpu_rope(k, rows, kv_dim, seq_len, 0, theta);
    let repeat = n_heads / n_kv_heads;
    let scale = 1.0 / (dh as f32).sqrt();
    let mm = |a: &[f32], b: &[f32], m: usize, kk: usize, n: usize| -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m
        {
            for j in 0..n
            {
                let mut acc = 0.0f32;
                for p in 0..kk
                {
                    acc += a[i * kk + p] * b[p * n + j];
                }
                c[i * n + j] = acc;
            }
        }
        c
    };
    let mut out = vec![0.0f32; rows * d_model];
    for head in 0..n_heads
    {
        let kv = head / repeat;
        let qs = cpu_slice_cols(&qr, rows, d_model, head * dh, dh); // [rows, dh]
        let ks = cpu_slice_cols(&kr, rows, kv_dim, kv * dh, dh);
        let vs = cpu_slice_cols(v, rows, kv_dim, kv * dh, dh);
        // scores = qs · ksᵀ  → transpose ks to [dh, rows] first.
        let mut kt = vec![0.0f32; dh * rows];
        for r in 0..rows
        {
            for c in 0..dh
            {
                kt[c * rows + r] = ks[r * dh + c];
            }
        }
        let s = mm(&qs, &kt, rows, dh, rows); // [rows, rows]
        let s = cpu_scale_causal_mask(&s, rows, rows, scale, causal);
        let w = cpu_softmax(&s, rows, rows);
        let ctx = mm(&w, &vs, rows, rows, dh); // [rows, dh]
        let placed = cpu_place_cols(&ctx, rows, dh, head * dh, d_model);
        for i in 0..rows * d_model
        {
            out[i] += placed[i];
        }
    }
    out
}

/// CPU reference for one **GQA transformer block** — the oracle for
/// [`crate::GpuChain::gqa_transformer_block`], matching the sciagent
/// `SciAgentBlock`: pre-norm multi-head grouped-query attention (with RoPE) +
/// residual, then pre-norm SwiGLU MLP + residual. `x` is `[t, d]`
/// (`d = n_heads·dh`); `wk`/`wv` are `[d, kv_dim]` (`kv_dim = n_kv_heads·dh`);
/// `wg`/`wu` are `[d, ff]`, `wd` is `[ff, d]`.
#[allow(clippy::too_many_arguments)]
pub fn cpu_gqa_transformer_block(
    x: &[f32],
    norm1: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
    norm2: &[f32],
    wg: &[f32],
    wu: &[f32],
    wd: &[f32],
    t: usize,
    d: usize,
    kv_dim: usize,
    ff: usize,
    n_heads: usize,
    n_kv_heads: usize,
    dh: usize,
    theta: f32,
    eps: f32,
    causal: bool,
) -> Vec<f32> {
    let mm = |a: &[f32], b: &[f32], m: usize, kk: usize, n: usize| -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m
        {
            for j in 0..n
            {
                let mut acc = 0.0f32;
                for p in 0..kk
                {
                    acc += a[i * kk + p] * b[p * n + j];
                }
                c[i * n + j] = acc;
            }
        }
        c
    };
    // Attention sub-block.
    let xn = cpu_rms_norm(x, norm1, eps, t, d);
    let q = mm(&xn, wq, t, d, d);
    let k = mm(&xn, wk, t, d, kv_dim);
    let v = mm(&xn, wv, t, d, kv_dim);
    let ctx = cpu_gqa_attention(&q, &k, &v, t, n_heads, n_kv_heads, dh, t, theta, causal);
    let attn_out = mm(&ctx, wo, t, d, d);
    let h: Vec<f32> = x.iter().zip(&attn_out).map(|(a, b)| a + b).collect();
    // MLP sub-block.
    let hn = cpu_rms_norm(&h, norm2, eps, t, d);
    let gate = mm(&hn, wg, t, d, ff);
    let up = mm(&hn, wu, t, d, ff);
    let act: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
        .collect();
    let mlp = mm(&act, wd, t, ff, d);
    h.iter().zip(&mlp).map(|(a, b)| a + b).collect()
}

/// CPU reference for the scale + causal-mask backward: `din = scale·dout` at
/// kept positions, `0` above the diagonal (masked keys carry no gradient). The
/// GPU `scale_causal_mask_backward_resident` contract.
#[allow(clippy::needless_range_loop)]
pub fn cpu_scale_causal_mask_backward(
    dout: &[f32],
    rows: usize,
    cols: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let mut din = vec![0.0f32; dout.len()];
    for i in 0..rows
    {
        for j in 0..cols
        {
            din[i * cols + j] = if causal && j > i
            {
                0.0
            }
            else
            {
                dout[i * cols + j] * scale
            };
        }
    }
    din
}

/// CPU reference for the embedding-gather backward: accumulate upstream grad
/// `dout` (`tokens.len() × d`) into a `vocab × d` table gradient — row `v` sums
/// the `dout` rows whose token id is `v`. The GPU `embed_backward_resident`
/// kernel's correctness contract.
pub fn cpu_embed_backward(tokens: &[u32], dout: &[f32], d: usize, vocab: usize) -> Vec<f32> {
    let mut dtable = vec![0.0f32; vocab * d];
    for (i, &tok) in tokens.iter().enumerate()
    {
        let v = (tok as usize).min(vocab.saturating_sub(1));
        for c in 0..d
        {
            dtable[v * d + c] += dout[i * d + c];
        }
    }
    dtable
}

/// CPU reference for one SGD step: `param − lr·grad`, elementwise. The GPU
/// `sgd_step_resident` kernel's correctness contract.
pub fn cpu_sgd_step(param: &[f32], grad: &[f32], lr: f32) -> Vec<f32> {
    param.iter().zip(grad).map(|(p, g)| p - lr * g).collect()
}

/// CPU reference for one in-place AdamW step at `step` (1-based): bias-corrected
/// Adam with decoupled weight decay. Updates `param`, `m`, `v` in place. The GPU
/// `adamw_step_resident` kernel's correctness contract.
#[allow(clippy::too_many_arguments)]
pub fn cpu_adamw_step(
    param: &mut [f32],
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
    step: u32,
) {
    let (b1, b2) = betas;
    let bc1 = 1.0 - b1.powi(step as i32);
    let bc2 = 1.0 - b2.powi(step as i32);
    for i in 0..param.len()
    {
        let g = grad[i];
        m[i] = b1 * m[i] + (1.0 - b1) * g;
        v[i] = b2 * v[i] + (1.0 - b2) * g * g;
        let mhat = m[i] / bc1;
        let vhat = v[i] / bc2;
        param[i] -= lr * (mhat / (vhat.sqrt() + eps) + weight_decay * param[i]);
    }
}

/// CPU reference for the mean cross-entropy loss: `−(1/rows)·Σᵢ log P[i,tgtᵢ]`
/// where `P = softmax(logits)` row-wise (`rows × cols` logits, `rows` targets).
pub fn cpu_cross_entropy(logits: &[f32], targets: &[u32], rows: usize, cols: usize) -> f32 {
    let p = cpu_softmax(logits, rows, cols);
    let mut loss = 0.0f32;
    for (i, &t) in targets.iter().enumerate()
    {
        loss -= p[i * cols + (t as usize).min(cols - 1)].max(1e-30).ln();
    }
    loss / rows as f32
}

/// CPU reference for the cross-entropy gradient w.r.t. the logits:
/// `dlogits = (softmax(logits) − onehot(target)) / rows`. The GPU
/// `cross_entropy_grad_resident` kernel's correctness contract.
pub fn cpu_cross_entropy_grad(
    logits: &[f32],
    targets: &[u32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let mut d = cpu_softmax(logits, rows, cols);
    let inv = 1.0f32 / rows as f32;
    for (i, &t) in targets.iter().enumerate()
    {
        d[i * cols + (t as usize).min(cols - 1)] -= 1.0;
    }
    for v in d.iter_mut()
    {
        *v *= inv;
    }
    d
}

/// CPU reference for token embedding gather: output row `i` is row `tokens[i]`
/// of the `vocab × d` row-major `table` (token ids clamped to `vocab-1`). The
/// GPU `embed_resident` kernel's correctness contract.
pub fn cpu_embed(tokens: &[u32], table: &[f32], d: usize, vocab: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; tokens.len() * d];
    for (i, &tok) in tokens.iter().enumerate()
    {
        let row = (tok as usize).min(vocab.saturating_sub(1));
        out[i * d..i * d + d].copy_from_slice(&table[row * d..row * d + d]);
    }
    out
}

/// Relative Frobenius error.
pub fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt();
    let den: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-30);
    num / den
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_activation_relu() {
        let data = vec![-1.0, 0.0, 2.0, -0.5];
        let out = cpu_activation(&data, EwOp::Relu);
        assert_eq!(out, vec![0.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn test_cpu_softmax_rows_sum_to_one_and_match_hand() {
        // Two rows; each softmax row must sum to 1 and be order-preserving.
        let data = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
        let out = cpu_softmax(&data, 2, 3);
        for r in 0..2
        {
            let s: f32 = out[r * 3..r * 3 + 3].iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "row {r} sums to {s}");
        }
        // Uniform row → uniform distribution.
        assert!(out[3..6].iter().all(|&x| (x - 1.0 / 3.0).abs() < 1e-6));
        // Monotonic row preserves order.
        assert!(out[0] < out[1] && out[1] < out[2]);
    }

    #[test]
    fn test_cpu_scale_causal_mask() {
        // 3×3 scores, scale 0.5, causal: upper triangle → MASK_NEG.
        let s: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let out = cpu_scale_causal_mask(&s, 3, 3, 0.5, true);
        // Row 0: keep [0], mask [1],[2].
        assert_eq!(out[0], 0.5); // 1*0.5
        assert_eq!(out[1], MASK_NEG);
        assert_eq!(out[2], MASK_NEG);
        // Row 1: keep [0],[1], mask [2].
        assert_eq!(out[3], 2.0); // 4*0.5
        assert_eq!(out[4], 2.5); // 5*0.5
        assert_eq!(out[5], MASK_NEG);
        // Row 2: all kept.
        assert!(out[6..9].iter().all(|&x| x > 0.0));
        // Non-causal just scales.
        let ns = cpu_scale_causal_mask(&s, 3, 3, 2.0, false);
        assert_eq!(ns, s.iter().map(|x| x * 2.0).collect::<Vec<_>>());
    }

    #[test]
    fn test_cpu_softmax_is_shift_invariant() {
        // softmax(x) == softmax(x + c): the max-subtraction guarantees it.
        let a = vec![-2.0, 0.5, 3.0, 1.0];
        let b: Vec<f32> = a.iter().map(|x| x + 100.0).collect();
        let sa = cpu_softmax(&a, 1, 4);
        let sb = cpu_softmax(&b, 1, 4);
        assert!(rel_err(&sa, &sb) < 1e-6);
    }

    #[test]
    fn test_cpu_activation_sigmoid_range() {
        let data = vec![-10.0, 0.0, 10.0];
        let out = cpu_activation(&data, EwOp::Sigmoid);
        assert!(out[0] < 0.001);
        assert!((out[1] - 0.5).abs() < 1e-6);
        assert!(out[2] > 0.999);
    }

    #[test]
    fn test_cpu_reduce_sum() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 rows of 3
        let out = cpu_reduce_sum(&data, 2, 3);
        assert_eq!(out, vec![6.0, 15.0]);
    }

    #[test]
    fn test_cpu_reduce_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = cpu_reduce_mean(&data, 2, 3);
        assert_eq!(out, vec![2.0, 5.0]);
    }

    #[test]
    fn test_cpu_reduce_max() {
        let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let out = cpu_reduce_max(&data, 2, 3);
        assert_eq!(out, vec![5.0, 6.0]);
    }

    #[test]
    fn test_cpu_layer_norm() {
        // 2 rows, 2 cols, gamma=[1,1], beta=[0,0], eps=0
        let data = vec![1.0, 3.0, 5.0, 7.0];
        let gamma = vec![1.0, 1.0];
        let beta = vec![0.0, 0.0];
        let out = cpu_layer_norm(&data, &gamma, &beta, 1e-5, 2, 2);
        // Row 0: mean=2, var=1, out = (x-2)/1 = [-1, 1]
        // Row 1: mean=6, var=1, out = (x-6)/1 = [-1, 1]
        assert!((out[0] + 1.0).abs() < 1e-5);
        assert!((out[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_rms_norm() {
        let data = vec![2.0, 2.0, 4.0, 4.0]; // 2 rows of 2
        let weight = vec![1.0, 1.0];
        let out = cpu_rms_norm(&data, &weight, 1e-5, 2, 2);
        // Row 0: rms = sqrt((4+4)/2 + eps) ≈ 2.0, normalized: [1, 1]
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - 1.0).abs() < 1e-5);
    }
}
