//! Route B parity (feature `cuda`).
//!
//! Builds one SCIAGENT model and checks that the **CUDA + Tensor-core** resident
//! forward ([`CudaModel`]) matches the CPU reference forward within a bf16
//! tolerance. This is B3 of `ROUTE_B.md`: the whole decoder — tied embeddings,
//! RoPE, GQA attention, SwiGLU, tied LM head — running on Blackwell Tensor cores.
//!
//! bf16 rounds inputs and the GEMMs accumulate in fp32, so results are **not**
//! bit-identical (unlike Route A's ~3e-3 fp32 tolerance); a correct composition
//! lands at a few percent, while any wiring bug is `O(1)`. CUDA-only to build, so
//! this whole file is `#[cfg(feature = "cuda")]` and runs on the Thor.
#![cfg(feature = "cuda")]

use scirust_core::autodiff::reverse::Tape;
use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::cuda_model::CudaModel;
use scirust_sciagent::model::SciAgentModel;
use scirust_sciagent::train::cross_entropy_loss;

fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt();
    let den: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-30);
    num / den
}

/// A small tied config exercising every op (GQA `n_heads != n_kv_heads`, RoPE,
/// SwiGLU, tied head over a non-zero table).
fn tiny_tied() -> SciAgentConfig {
    SciAgentConfig {
        vocab_size: 48,
        d_model: 32,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        d_ff: 64,
        max_seq_len: 16,
        rope_theta: 10_000.0,
        tie_embeddings: true,
        use_bias: false,
        eps: 1e-5,
    }
}

/// The CUDA (bf16, Tensor-core) forward matches the CPU `SciAgentModel` forward
/// within bf16 tolerance — the whole decoder on Route B. Skips with no device.
#[test]
fn cuda_forward_matches_cpu_model() {
    let config = tiny_tied();
    let mut model = SciAgentModel::new(&config);
    let seq_len = 8usize;
    let ids: Vec<usize> = (0..seq_len)
        .map(|i| (i * 7 + 3) % config.vocab_size)
        .collect();

    // CPU reference forward.
    let tape = Tape::new();
    let logits_v = model.forward(&tape, &ids, seq_len);
    let cpu_logits = tape.value(logits_v.idx()).data;

    // CUDA forward from the same weights.
    let Some(cm) = CudaModel::from_model(&model)
    else
    {
        eprintln!("cuda: no device, skipping CUDA forward parity");
        return;
    };
    let tokens: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
    let got = cm.forward(&tokens);

    assert_eq!(got.len(), cpu_logits.len(), "logit shape mismatch");
    let e = rel_err(&got, &cpu_logits);
    // bf16 through a whole decoder: a correct composition is a few percent; a
    // wiring bug is O(1). 12% ceiling cleanly separates the two.
    assert!(
        e < 1.2e-1,
        "CUDA bf16 forward rel_err {e} too large (wiring bug?)"
    );
    eprintln!("CUDA bf16 Tensor-core forward vs CPU model: rel_err {e:.3e} — PASS");
}

/// The CUDA (bf16, Tensor-core) **backward** matches the CPU `SciAgentModel`'s
/// analytic tied-embedding gradient within bf16 tolerance — B4e of `ROUTE_B.md`.
///
/// The tied-embedding grad is the strongest single check: it sums the LM-head
/// gradient (`dlogitsᵀ·normed`) with the gradient backpropagated through every
/// block, RoPE, GQA attention, SwiGLU and both RMSNorms into the input gather — so
/// if it matches, the whole backward composition (the matmul VJP and every adjoint
/// kernel that feeds AdamW) is validated end-to-end. Both sides are analytic (finite
/// differences are too coarse in bf16), differing only by bf16 rounding that
/// compounds through the depth to a few percent; a wiring bug is `O(1)`. Skips with
/// no device.
#[test]
fn cuda_backward_matches_cpu_embedding_grad() {
    let config = tiny_tied();
    let mut model = SciAgentModel::new(&config);
    let seq_len = 8usize;
    let ids: Vec<usize> = (0..seq_len)
        .map(|i| (i * 7 + 3) % config.vocab_size)
        .collect();
    // Next-token-style targets (any consistent targets work for a grad check).
    let targets: Vec<usize> = (0..seq_len)
        .map(|i| (ids[i] + 1) % config.vocab_size)
        .collect();

    // CPU analytic tied-embedding grad via the reverse-mode tape.
    let tape = Tape::new();
    let logits_v = model.forward(&tape, &ids, seq_len);
    let loss = cross_entropy_loss(&tape, logits_v, &targets);
    tape.backward(loss.idx());
    let tied_idx = model.parameter_indices()[0]; // tied path pushes the embedding first
    let cpu_dembed = tape.grad(tied_idx).data;

    // CUDA backward from the same weights + targets.
    let Some(cm) = CudaModel::from_model(&model)
    else
    {
        eprintln!("cuda: no device, skipping CUDA backward parity");
        return;
    };
    let tokens: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
    let tgt_u32: Vec<u32> = targets.iter().map(|&t| t as u32).collect();
    let got = cm.embedding_grad(&tokens, &tgt_u32);

    assert_eq!(got.len(), cpu_dembed.len(), "embedding-grad shape mismatch");
    let e = rel_err(&got, &cpu_dembed);
    // bf16 backprop through a 2-layer decoder compounds more than the forward; a
    // correct composition is still a few percent, a wiring bug is O(1).
    assert!(
        e < 2.5e-1,
        "CUDA bf16 backward rel_err {e} too large (wiring bug?)"
    );
    eprintln!("CUDA bf16 Tensor-core backward vs CPU tied-embedding grad: rel_err {e:.3e} — PASS");
}
