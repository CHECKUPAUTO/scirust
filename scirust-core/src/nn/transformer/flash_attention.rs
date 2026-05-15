use crate::autodiff::reverse::{Tape, Tensor};

/// Flash Attention simplifiée (tiled softmax attention).
/// Implémente le calcul de l'attention par blocs pour réduire l'empreinte mémoire O(N²).
/// Version basique: softmax + matmul, mais la structure permet l'extension vers
/// le tiling réel avec accumulation online.
pub struct FlashAttention;

impl FlashAttention {
    /// Calcule l'attention avec softmax tildé (un seul bloc pour l'instant).
    /// Q: (seq_len, n_heads * d_k)
    /// K: (seq_len, n_heads * d_k)
    /// V: (seq_len, n_heads * d_v)
    /// Retourne: (seq_len, n_heads * d_v)
    pub fn forward(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, _tape: &Tape) -> Tensor {
        // Score: Q @ K^T / scale
        let scores = q.matmul(&k.transpose()).scale(1.0 / scale);

        // Softmax stable (en un bloc pour l'instant)
        let max_row = scores.max_axis(1);
        let exp = scores.sub(&max_row).exp();
        let sum_exp = exp.sum_axis(1);
        let attn = exp.div(&sum_exp);

        // Output: attn @ V
        attn.matmul(v)
    }

    /// Version avec causal mask pour l'inférence séquentielle
    pub fn forward_causal(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, _tape: &Tape) -> Tensor {
        let seq_len = q.rows;
        let scores = q.matmul(&k.transpose()).scale(1.0 / scale);

        // Mask causal: -inf pour les positions futures
        let mut mask_data = vec![0.0; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i+1)..seq_len {
                mask_data[i * seq_len + j] = -1e9;
            }
        }
        let mask = Tensor::from_vec(mask_data, seq_len, seq_len);
        let masked = scores.add(&mask);

        let max_row = masked.max_axis(1);
        let exp = masked.sub(&max_row).exp();
        let sum_exp = exp.sum_axis(1);
        let attn = exp.div(&sum_exp);
        attn.matmul(v)
    }
}
