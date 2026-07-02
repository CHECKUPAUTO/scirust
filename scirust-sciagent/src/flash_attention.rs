//! Block-streaming ("flash") attention — CPU reference and numerical oracle.
//!
//! The dense attention in [`crate::attention`] computes, per (head, batch):
//! `softmax(causal_mask(Q·Kᵀ · scale)) · V`, materializing the full
//! `S×S` score matrix. That matrix is the memory wall for long sequences
//! (see [`crate::planning`]). Flash attention computes the **identical**
//! result with an online-softmax recurrence that only ever holds a
//! `block × block` tile, so its working set is `O(block)` instead of `O(S²)`.
//!
//! This module is the backend-agnostic **correctness contract**: a GPU flash
//! kernel is correct iff it matches [`flash_attention`] here, which the tests
//! pin against the naive [`dense_attention`] to tight tolerance. Validating the
//! algorithm on CPU first is phase 1 of the Jetson Thor roadmap — no kernel is
//! written blind.
//!
//! All inputs are row-major `[n, dh]` slices; `scale` is applied to the scores
//! (typically `1/sqrt(dh)`); `causal` masks key `j > i` for query `i`.

/// Naive dense attention for one (head, batch): the reference the model's
/// tape path also computes. Materializes the `n_q × n_k` score row.
#[allow(clippy::too_many_arguments)] // explicit q/k/v/shape/scale/causal signature
pub fn dense_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_q: usize,
    n_k: usize,
    dh: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    assert_eq!(q.len(), n_q * dh);
    assert_eq!(k.len(), n_k * dh);
    assert_eq!(v.len(), n_k * dh);
    let mut out = vec![0.0f32; n_q * dh];

    for i in 0..n_q
    {
        let qi = &q[i * dh..i * dh + dh];
        let j_max = if causal { i + 1 } else { n_k };

        // scores + row max
        let mut scores = vec![f32::NEG_INFINITY; j_max];
        let mut m = f32::NEG_INFINITY;
        for (j, s) in scores.iter_mut().enumerate()
        {
            let kj = &k[j * dh..j * dh + dh];
            let mut dot = 0.0f32;
            for t in 0..dh
            {
                dot += qi[t] * kj[t];
            }
            *s = dot * scale;
            if *s > m
            {
                m = *s;
            }
        }
        // softmax + weighted sum of V
        let mut denom = 0.0f32;
        for s in &scores
        {
            denom += (s - m).exp();
        }
        let oi = &mut out[i * dh..i * dh + dh];
        for (j, s) in scores.iter().enumerate()
        {
            let p = (s - m).exp() / denom;
            let vj = &v[j * dh..j * dh + dh];
            for t in 0..dh
            {
                oi[t] += p * vj[t];
            }
        }
    }
    out
}

/// Flash attention: identical output to [`dense_attention`], computed with an
/// online-softmax recurrence over key blocks of `block` rows. Never allocates
/// an `n_q × n_k` buffer — the working set is `O(block + dh)` per query.
#[allow(clippy::too_many_arguments)] // explicit q/k/v/shape/scale/causal/block signature
pub fn flash_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_q: usize,
    n_k: usize,
    dh: usize,
    scale: f32,
    causal: bool,
    block: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), n_q * dh);
    assert_eq!(k.len(), n_k * dh);
    assert_eq!(v.len(), n_k * dh);
    assert!(block >= 1, "block size must be >= 1");
    let mut out = vec![0.0f32; n_q * dh];

    for i in 0..n_q
    {
        let qi = &q[i * dh..i * dh + dh];
        let j_max = if causal { i + 1 } else { n_k };

        // Online-softmax running state: max `m`, normalizer `l`, accumulator.
        let mut m = f32::NEG_INFINITY;
        let mut l = 0.0f32;
        let mut acc = vec![0.0f32; dh];

        let mut j0 = 0;
        while j0 < j_max
        {
            let j1 = (j0 + block).min(j_max);
            // Block scores and block max.
            let mut block_scores = vec![0.0f32; j1 - j0];
            let mut block_m = f32::NEG_INFINITY;
            for (bj, sc) in block_scores.iter_mut().enumerate()
            {
                let kj = &k[(j0 + bj) * dh..(j0 + bj) * dh + dh];
                let mut dot = 0.0f32;
                for t in 0..dh
                {
                    dot += qi[t] * kj[t];
                }
                *sc = dot * scale;
                if *sc > block_m
                {
                    block_m = *sc;
                }
            }
            // Merge this block into the running softmax.
            let new_m = m.max(block_m);
            let correction = (m - new_m).exp(); // rescale prior mass
            let mut block_l = 0.0f32;
            // Rescale the accumulator to the new max.
            for a in acc.iter_mut()
            {
                *a *= correction;
            }
            for (bj, &sc) in block_scores.iter().enumerate()
            {
                let p = (sc - new_m).exp();
                block_l += p;
                let vj = &v[(j0 + bj) * dh..(j0 + bj) * dh + dh];
                for t in 0..dh
                {
                    acc[t] += p * vj[t];
                }
            }
            l = l * correction + block_l;
            m = new_m;
            j0 = j1;
        }

        let oi = &mut out[i * dh..i * dh + dh];
        if l > 0.0
        {
            for t in 0..dh
            {
                oi[t] = acc[t] / l;
            }
        }
    }
    out
}

/// Peak score-matrix elements held at once, dense vs flash — the memory
/// argument in [`crate::planning`], made concrete for one (head, batch).
pub fn peak_score_elements(n_q: usize, n_k: usize, block: usize, flash: bool) -> usize {
    if flash
    {
        block.min(n_k) // one key tile per query
    }
    else
    {
        n_q * n_k // the full score matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded(n: usize, seed: u64) -> Vec<f32> {
        // Deterministic pseudo-random in [-1, 1); no rng dependency.
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn flash_matches_dense_across_shapes_blocks_and_causality() {
        for &(n_q, n_k, dh) in &[
            (1usize, 1usize, 4usize),
            (8, 8, 16),
            (13, 13, 8),
            (5, 9, 12),
        ]
        {
            let q = seeded(n_q * dh, 1);
            let k = seeded(n_k * dh, 2);
            let v = seeded(n_k * dh, 3);
            let scale = 1.0 / (dh as f32).sqrt();
            for &causal in &[false, true]
            {
                // Causal requires square-ish alignment (query i attends keys<=i).
                if causal && n_k < n_q
                {
                    continue;
                }
                let dense = dense_attention(&q, &k, &v, n_q, n_k, dh, scale, causal);
                for &block in &[1usize, 2, 4, 7, 64]
                {
                    let flash = flash_attention(&q, &k, &v, n_q, n_k, dh, scale, causal, block);
                    let d = max_abs_diff(&dense, &flash);
                    assert!(
                        d < 1e-5,
                        "n_q={n_q} n_k={n_k} dh={dh} causal={causal} block={block}: max diff {d}"
                    );
                }
            }
        }
    }

    #[test]
    fn block_size_does_not_change_the_result() {
        // Flash must be invariant to tiling — the online-softmax correction is
        // what guarantees a GPU kernel can pick any block size.
        let (n, dh) = (32usize, 16usize);
        let q = seeded(n * dh, 10);
        let k = seeded(n * dh, 20);
        let v = seeded(n * dh, 30);
        let scale = 1.0 / (dh as f32).sqrt();
        let ref_out = flash_attention(&q, &k, &v, n, n, dh, scale, true, 1);
        for &block in &[2usize, 3, 8, 16, 32, 100]
        {
            let out = flash_attention(&q, &k, &v, n, n, dh, scale, true, block);
            assert!(
                max_abs_diff(&ref_out, &out) < 1e-6,
                "block {block} diverged"
            );
        }
    }

    #[test]
    fn causal_mask_hides_the_future() {
        // Query 0 attends only key 0, so its output must equal V row 0 exactly,
        // regardless of later keys.
        let dh = 4;
        let q = seeded(2 * dh, 1);
        let k = seeded(2 * dh, 2);
        let mut v = seeded(2 * dh, 3);
        let out = flash_attention(&q, &k, &v, 2, 2, dh, 1.0, true, 4);
        assert!(max_abs_diff(&out[0..dh], &v[0..dh]) < 1e-6);
        // Changing key/value row 1 must not affect query 0's output.
        for x in &mut v[dh..]
        {
            *x += 5.0;
        }
        let out2 = flash_attention(&q, &k, &v, 2, 2, dh, 1.0, true, 4);
        assert!(max_abs_diff(&out[0..dh], &out2[0..dh]) < 1e-6);
    }

    #[test]
    fn flash_working_set_is_bounded() {
        assert_eq!(peak_score_elements(8192, 8192, 128, false), 8192 * 8192);
        assert_eq!(peak_score_elements(8192, 8192, 128, true), 128);
    }

    #[test]
    fn dense_reference_matches_the_models_tape_ops() {
        // Closes the loop: the model computes one head as
        // `matmul(Kᵀ).scale.causal_mask.softmax(1).matmul(V)` on the tape.
        // Proving `dense_attention` equals that chain — and `flash` equals
        // `dense` (above) — means a GPU flash kernel validated against this
        // module is validated against the real model math.
        use scirust_core::autodiff::reverse::{Tape, Tensor};

        let (n, dh) = (6usize, 8usize);
        let q = seeded(n * dh, 7);
        let k = seeded(n * dh, 8);
        let v = seeded(n * dh, 9);
        let scale = 1.0 / (dh as f32).sqrt();

        let tape = Tape::new();
        let qv = tape.input(Tensor::from_vec(q.clone(), n, dh));
        let kv = tape.input(Tensor::from_vec(k.clone(), n, dh));
        let vv = tape.input(Tensor::from_vec(v.clone(), n, dh));
        let o = qv
            .matmul(kv.transpose_2d())
            .scale(scale)
            .causal_mask(n)
            .softmax(1)
            .matmul(vv);
        let model_out = tape.value(o.idx()).data;

        let dense = dense_attention(&q, &k, &v, n, n, dh, scale, true);
        let flash = flash_attention(&q, &k, &v, n, n, dh, scale, true, 3);
        assert!(
            max_abs_diff(&dense, &model_out) < 1e-5,
            "dense reference must equal the model's tape ops"
        );
        assert!(
            max_abs_diff(&flash, &model_out) < 1e-5,
            "flash must equal the model's tape ops"
        );
    }
}
