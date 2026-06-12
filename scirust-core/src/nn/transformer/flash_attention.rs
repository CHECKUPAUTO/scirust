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
