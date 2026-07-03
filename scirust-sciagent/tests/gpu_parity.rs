//! Whole-model GPU parity (feature `gpu`).
//!
//! Builds one SCIAGENT model, then runs the *same* forward + backward on a
//! CPU-only tape and on a tape with the wgpu GEMM engine attached
//! (`gpu::attach_gpu`), and checks that the logits and **every** parameter
//! gradient agree within a small relative tolerance. This is the end-to-end
//! statement of direction C1: the real model — tied embeddings, RoPE, GQA
//! attention, SwiGLU, tied LM head — trains through the GPU engine and matches
//! the CPU reference it was validated against, brick by brick.
//!
//! Skips cleanly when no GPU adapter is present (this CI container has none);
//! it runs on Mesa lavapipe in the `GPU (wgpu / lavapipe)` job and on the
//! Jetson Thor's Blackwell via `examples/gpu_forward_parity.rs`.
#![cfg(feature = "gpu")]

use scirust_core::autodiff::reverse::Tape;
use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::gpu::attach_gpu;
use scirust_sciagent::model::SciAgentModel;

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

/// A small tied config that still exercises every GEMM in the model (GQA with
/// n_heads != n_kv_heads, RoPE, SwiGLU, the tied head over a non-zero table).
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

#[test]
fn full_model_forward_and_backward_match_cpu_on_gpu() {
    let config = tiny_tied();
    let mut model = SciAgentModel::new(&config);
    let seq_len = 8usize;
    let ids: Vec<usize> = (0..seq_len)
        .map(|i| (i * 7 + 3) % config.vocab_size)
        .collect();

    // CPU reference: forward -> scalar loss -> backward.
    let cpu_tape = Tape::new();
    let cpu_logits = model.forward(&cpu_tape, &ids, seq_len);
    cpu_tape.backward(cpu_logits.sum().idx());
    let cpu_out = cpu_tape.value(cpu_logits.idx()).data;
    let cpu_params = model.parameter_indices();
    let cpu_grads: Vec<Vec<f32>> = cpu_params.iter().map(|&i| cpu_tape.grad(i).data).collect();

    // GPU: identical model, identical inputs, GEMMs routed to the device.
    let gpu_tape = Tape::new();
    let Some(name) = attach_gpu(&gpu_tape)
    else
    {
        eprintln!("wgpu: no adapter, skipping full-model parity");
        return;
    };
    eprintln!("full-model GPU parity on: {name}");
    let gpu_logits = model.forward(&gpu_tape, &ids, seq_len);
    gpu_tape.backward(gpu_logits.sum().idx());
    let gpu_out = gpu_tape.value(gpu_logits.idx()).data;
    let gpu_params = model.parameter_indices();
    let gpu_grads: Vec<Vec<f32>> = gpu_params.iter().map(|&i| gpu_tape.grad(i).data).collect();

    // Forward logits: GPU GEMM accumulates in a different order, so match within
    // tolerance rather than bit-exactly. A routing bug gives rel_err ~O(1).
    let fwd = rel_err(&gpu_out, &cpu_out);
    assert!(fwd < 3e-3, "forward logits mismatch: rel_err {fwd}");

    // Every parameter gradient (embedding/tied head, all projections, norms).
    assert_eq!(cpu_params.len(), gpu_params.len(), "param count changed");
    let mut worst = 0.0f32;
    for (k, (cg, gg)) in cpu_grads.iter().zip(&gpu_grads).enumerate()
    {
        let e = rel_err(gg, cg);
        worst = worst.max(e);
        assert!(e < 2e-2, "param {k} grad mismatch: rel_err {e}");
    }
    eprintln!("forward rel_err {fwd:.2e}, worst grad rel_err {worst:.2e} — PASS");
}
