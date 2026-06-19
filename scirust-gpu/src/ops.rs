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
