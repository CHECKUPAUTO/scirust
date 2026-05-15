use crate::autodiff::reverse::Tensor;

/// Rotary Position Embedding (RoPE)
/// Applique la rotation aux paires de dimensions de x.
/// x : tenseur de forme (seq_len, dim)
/// offset : décalage de position (pour le cache KV)
pub fn rope_apply(x: &Tensor, offset: usize, theta: f32) -> Tensor {
    let seq_len = x.rows;
    let dim = x.cols;
    let half_dim = dim / 2;

    // 1. Construire les tenseurs cos/sin
    let mut cos_data = vec![0.0; seq_len * half_dim];
    let mut sin_data = vec![0.0; seq_len * half_dim];
    for pos_idx in 0..seq_len {
        let pos = (pos_idx + offset) as f32;
        for j in 0..half_dim {
            let freq = theta.powf(-2.0 * j as f32 / dim as f32);
            let angle = pos * freq;
            cos_data[pos_idx * half_dim + j] = angle.cos();
            sin_data[pos_idx * half_dim + j] = angle.sin();
        }
    }
    let cos_t = Tensor::from_vec(cos_data, seq_len, half_dim);
    let sin_t = Tensor::from_vec(sin_data, seq_len, half_dim);

    // 2. Extraire paires/impaires
    let mut even_data = vec![0.0; seq_len * half_dim];
    let mut odd_data = vec![0.0; seq_len * half_dim];
    for i in 0..seq_len {
        for j in 0..half_dim {
            even_data[i * half_dim + j] = x.data[i * dim + 2 * j];
            odd_data[i * half_dim + j] = x.data[i * dim + 2 * j + 1];
        }
    }
    let even = Tensor::from_vec(even_data, seq_len, half_dim);
    let odd = Tensor::from_vec(odd_data, seq_len, half_dim);

    // 3. Rotation: even' = even*cos - odd*sin, odd' = even*sin + odd*cos (element-wise)
    let even_cos = even.mul(&cos_t);
    let odd_sin = odd.mul(&sin_t);
    let even_rot = even_cos.sub(&odd_sin);

    let even_sin = even.mul(&sin_t);
    let odd_cos = odd.mul(&cos_t);
    let odd_rot = even_sin.add(&odd_cos);

    // 4. Réassembler
    let mut rotated = vec![0.0; seq_len * dim];
    for i in 0..seq_len {
        for j in 0..half_dim {
            rotated[i * dim + 2 * j] = even_rot.data[i * half_dim + j];
            rotated[i * dim + 2 * j + 1] = odd_rot.data[i * half_dim + j];
        }
    }
    Tensor::from_vec(rotated, seq_len, dim)
}
