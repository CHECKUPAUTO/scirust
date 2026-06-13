// scirust-core/src/nn/transformer/flash_attention.rs
// Flash Attention avec tiling et online softmax.
//
// Algorithme : pour chaque tête, découper Q en blocs de block_size lignes,
// et pour chaque bloc de Q, itérer sur les blocs de K/V.
// Running max et running sum pour le softmax online, normalisation en fin de forward.

use crate::autodiff::reverse::{Op, SavedData, Tape, Tensor, Var};

/// Flash Attention avec tiling et online softmax.
///
/// # Arguments
///
/// * `q_var` - Requêtes : forme (batch * n_heads * seq_len, d_head) en row-major
/// * `k_var` - Clés :    forme (batch * n_heads * seq_len, d_head)
/// * `v_var` - Valeurs : forme (batch * n_heads * seq_len, d_v) où d_v = d_head pour self-attention
/// * `batch` - Batch size
/// * `n_heads` - Nombre de têtes
/// * `seq_len` - Longueur de séquence (requêtes et clés)
/// * `d_head` - Dimension par tête
/// * `scale` - Facteur d'échelle (typ. 1.0 / sqrt(d_head))
/// * `block_size` - Taille de bloc pour le tiling (typ. 32 ou 64)
/// * `causal` - Si true, applique un masque causal
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_forward<'t>(
    tape: &'t Tape,
    q_var: Var<'t>,
    k_var: Var<'t>,
    v_var: Var<'t>,
    batch: usize,
    n_heads: usize,
    seq_len: usize,
    d_head: usize,
    scale: f32,
    block_size: usize,
    causal: bool,
) -> Var<'t> {
    let q_idx = q_var.idx();
    let k_idx = k_var.idx();
    let v_idx = v_var.idx();

    let q = tape.value(q_idx);
    let k = tape.value(k_idx);
    let v = tape.value(v_idx);

    let total_heads = batch * n_heads;
    let s_len = seq_len; // self-attention : S = L

    // Dimensions de V
    let dv = if v.rows > 0 { v.cols } else { d_head };

    // O : (total_heads * seq_len, dv)
    let mut o_data = vec![0.0f32; total_heads * seq_len * dv];

    // saved_m : running max par ligne
    // saved_l : running sum par ligne
    let mut saved_m = vec![-f32::INFINITY; total_heads * seq_len];
    let mut saved_l = vec![0.0f32; total_heads * seq_len];

    // Pour chaque tête
    for h in 0..total_heads
    {
        let q_base = h * seq_len * d_head;
        let k_base = h * s_len * d_head;
        let v_base = h * s_len * dv;
        let o_base = h * seq_len * dv;
        let m_base = h * seq_len;

        // Boucle sur les blocs de Q (outer loop)
        for qi in (0..seq_len).step_by(block_size)
        {
            let br = (seq_len - qi).min(block_size);

            // O_i, m_i, l_i pour ce bloc de Q
            let mut o_i = vec![0.0f32; br * dv];
            let mut m_i = vec![-f32::INFINITY; br];
            let mut l_i = vec![0.0f32; br];

            // Boucle sur les blocs de K/V (inner loop)
            for kj in (0..s_len).step_by(block_size)
            {
                let bc = (s_len - kj).min(block_size);

                // S_ij = Q_i @ K_j^T
                // Q_i : (br, d_head), K_j : (bc, d_head), S_ij : (br, bc)
                let mut s_ij = vec![0.0f32; br * bc];
                for r in 0..br
                {
                    for c in 0..bc
                    {
                        let mut sum = 0.0f32;
                        for d in 0..d_head
                        {
                            sum += q.data[q_base + (qi + r) * d_head + d]
                                * k.data[k_base + (kj + c) * d_head + d];
                        }
                        s_ij[r * bc + c] = sum * scale;
                    }
                }

                // Masque causal si demandé
                if causal
                {
                    for r in 0..br
                    {
                        let qpos = qi + r;
                        for c in 0..bc
                        {
                            let kpos = kj + c;
                            if kpos > qpos
                            {
                                s_ij[r * bc + c] = -f32::INFINITY;
                            }
                        }
                    }
                }

                // Online softmax :
                // m_new = max(m_i, rowmax(S_ij))
                // P_ij = exp(S_ij - m_new)
                // l_i' = exp(m_i - m_new) * l_i + rowsum(P_ij)
                // O_i' = exp(m_i - m_new) * O_i + P_ij @ V_j

                // 1. Rowmax de S_ij
                let mut row_max = vec![-f32::INFINITY; br];
                for r in 0..br
                {
                    for c in 0..bc
                    {
                        row_max[r] = row_max[r].max(s_ij[r * bc + c]);
                    }
                }

                // 2. m_new
                let mut m_new = vec![-f32::INFINITY; br];
                for r in 0..br
                {
                    m_new[r] = m_i[r].max(row_max[r]);
                }

                // 3. P_ij = exp(S_ij - m_new)
                let mut p_ij = vec![0.0f32; br * bc];
                let mut row_sum_p = vec![0.0f32; br];
                for r in 0..br
                {
                    for c in 0..bc
                    {
                        p_ij[r * bc + c] = (s_ij[r * bc + c] - m_new[r]).exp();
                        row_sum_p[r] += p_ij[r * bc + c];
                    }
                }

                // 4. Rescale O_i et l_i
                let mut rescale = vec![0.0f32; br];
                for r in 0..br
                {
                    rescale[r] = (m_i[r] - m_new[r]).exp();
                }

                // 5. l_i' = exp(m_i - m_new) * l_i + rowsum(P_ij)
                for r in 0..br
                {
                    l_i[r] = rescale[r] * l_i[r] + row_sum_p[r];
                }

                // 6. O_i' = exp(m_i - m_new) * O_i + P_ij @ V_j
                let mut pv = vec![0.0f32; br * dv];
                for r in 0..br
                {
                    for c in 0..bc
                    {
                        let p = p_ij[r * bc + c];
                        if p == 0.0
                        {
                            continue;
                        }
                        for d in 0..dv
                        {
                            pv[r * dv + d] += p * v.data[v_base + (kj + c) * dv + d];
                        }
                    }
                }

                for r in 0..br
                {
                    for d in 0..dv
                    {
                        o_i[r * dv + d] = rescale[r] * o_i[r * dv + d] + pv[r * dv + d];
                    }
                }

                // 7. Update m_i
                m_i[..br].copy_from_slice(&m_new[..br]);
            }

            // Normalisation : O_i = O_i / l_i
            for r in 0..br
            {
                let inv_l = 1.0 / l_i[r];
                for d in 0..dv
                {
                    o_i[r * dv + d] *= inv_l;
                }
            }

            // Copier les résultats dans les tableaux finaux
            for r in 0..br
            {
                saved_m[m_base + qi + r] = m_i[r];
                saved_l[m_base + qi + r] = l_i[r];
                for d in 0..dv
                {
                    o_data[o_base + (qi + r) * dv + d] = o_i[r * dv + d];
                }
            }
        }
    }

    let o_tensor = Tensor::from_vec(o_data, total_heads * seq_len, dv);

    // Sauvegarder m et l dans SavedData
    let m_tensor = Tensor::from_vec(saved_m, total_heads * seq_len, 1);
    let l_tensor = Tensor::from_vec(saved_l, total_heads * seq_len, 1);

    let mask_idx = if causal { Some(0usize) } else { None };

    let new_idx = tape.push_with_saved(
        Op::FlashAttention {
            q: q_idx,
            k: k_idx,
            v: v_idx,
            mask: mask_idx,
            batch,
            n_heads,
            seq_len,
            d_head,
            scale,
            block_size,
        },
        crate::autodiff::reverse::DeviceTensor::cpu(o_tensor.clone()),
        SavedData::FlashAttentionState {
            m: m_tensor,
            l: l_tensor,
        },
    );

    Var { tape, idx: new_idx }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference dense attention for one head: O = softmax(scale·QKᵀ)·V,
    /// with an optional causal mask. This is the oracle `flash_attention_forward`
    /// is validated against.
    #[allow(clippy::too_many_arguments)]
    fn reference_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq: usize,
        d_head: usize,
        dv: usize,
        scale: f32,
        causal: bool,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; seq * dv];
        for i in 0..seq
        {
            // scores
            let mut scores = vec![f32::NEG_INFINITY; seq];
            for (j, sc) in scores.iter_mut().enumerate()
            {
                if causal && j > i
                {
                    continue;
                }
                let mut dot = 0.0f32;
                for d in 0..d_head
                {
                    dot += q[i * d_head + d] * k[j * d_head + d];
                }
                *sc = dot * scale;
            }
            // stable softmax
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0f32;
            let mut p = vec![0.0f32; seq];
            for j in 0..seq
            {
                if scores[j].is_finite()
                {
                    p[j] = (scores[j] - m).exp();
                    denom += p[j];
                }
            }
            for j in 0..seq
            {
                let w = p[j] / denom;
                for d in 0..dv
                {
                    out[i * dv + d] += w * v[j * dv + d];
                }
            }
        }
        out
    }

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    // A small, fixed self-attention case (1 head, seq=4, d=4). block_size=2
    // forces the tiling/online-softmax path to run more than one inner block.
    fn fixture() -> (Vec<f32>, Vec<f32>, Vec<f32>, usize, usize, f32) {
        let seq = 4;
        let d = 4;
        let scale = 1.0 / (d as f32).sqrt();
        let q: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.1) - 0.7).collect();
        let k: Vec<f32> = (0..seq * d).map(|i| (i as f32 * -0.05) + 0.3).collect();
        let v: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.07) - 0.2).collect();
        (q, k, v, seq, d, scale)
    }

    #[test]
    fn flash_matches_standard_attention() {
        let (q, k, v, seq, d, scale) = fixture();
        let tape = Tape::new();
        let qv = tape.input(Tensor::from_vec(q.clone(), seq, d));
        let kv = tape.input(Tensor::from_vec(k.clone(), seq, d));
        let vv = tape.input(Tensor::from_vec(v.clone(), seq, d));

        let out = flash_attention_forward(&tape, qv, kv, vv, 1, 1, seq, d, scale, 2, false);
        let got = tape.value(out.idx()).data.clone();
        let want = reference_attention(&q, &k, &v, seq, d, d, scale, false);

        assert!(
            approx_eq(&got, &want, 1e-5),
            "flash != reference attention\n got={got:?}\nwant={want:?}"
        );
    }

    #[test]
    fn flash_causal_matches_masked_reference() {
        let (q, k, v, seq, d, scale) = fixture();
        let tape = Tape::new();
        let qv = tape.input(Tensor::from_vec(q.clone(), seq, d));
        let kv = tape.input(Tensor::from_vec(k.clone(), seq, d));
        let vv = tape.input(Tensor::from_vec(v.clone(), seq, d));

        let out = flash_attention_forward(&tape, qv, kv, vv, 1, 1, seq, d, scale, 2, true);
        let got = tape.value(out.idx()).data.clone();
        let want = reference_attention(&q, &k, &v, seq, d, d, scale, true);
        assert!(
            approx_eq(&got, &want, 1e-5),
            "causal flash != masked reference"
        );

        // Row 0 attends only to position 0 ⇒ its output equals V row 0.
        assert!(
            approx_eq(&got[0..d], &v[0..d], 1e-6),
            "causal row 0 must equal V[0]"
        );
    }

    #[test]
    fn flash_is_deterministic() {
        let (q, k, v, seq, d, scale) = fixture();
        let run = || {
            let tape = Tape::new();
            let qv = tape.input(Tensor::from_vec(q.clone(), seq, d));
            let kv = tape.input(Tensor::from_vec(k.clone(), seq, d));
            let vv = tape.input(Tensor::from_vec(v.clone(), seq, d));
            let out = flash_attention_forward(&tape, qv, kv, vv, 1, 1, seq, d, scale, 2, false);
            tape.value(out.idx())
                .data
                .iter()
                .map(|f| f.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run(), "same inputs ⇒ bit-identical output");
    }

    #[test]
    fn flash_backward_produces_finite_gradients() {
        let (q, k, v, seq, d, scale) = fixture();
        let tape = Tape::new();
        let qv = tape.input(Tensor::from_vec(q, seq, d));
        let kv = tape.input(Tensor::from_vec(k, seq, d));
        let vv = tape.input(Tensor::from_vec(v, seq, d));
        let (qi, ki, vi) = (qv.idx(), kv.idx(), vv.idx());

        let out = flash_attention_forward(&tape, qv, kv, vv, 1, 1, seq, d, scale, 2, false);
        out.sum().backward();

        for idx in [qi, ki, vi]
        {
            let g = tape.grad(idx);
            assert_eq!(g.data.len(), seq * d, "gradient shape must match input");
            assert!(
                g.data.iter().all(|x| x.is_finite()),
                "gradients must be finite"
            );
            assert!(
                g.data.iter().any(|&x| x != 0.0),
                "at least some gradient must be non-zero"
            );
        }
    }
}
