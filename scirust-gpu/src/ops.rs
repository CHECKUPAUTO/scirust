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
