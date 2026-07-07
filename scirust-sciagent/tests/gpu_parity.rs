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

/// The **fully-resident** path (`ResidentModel`, `scirust-gpu`'s `GpuChain`)
/// reproduces the real model's forward. Uploads every `SciAgentModel` weight to
/// VRAM, runs `gqa_model_forward` on the device, and checks the logits against
/// the model's own CPU forward. This is the bridge that lets the whole decoder
/// run on the resident path (the one that beats the per-op tape ~4× on the Thor),
/// not just its GEMMs. Skips cleanly with no adapter.
#[test]
fn resident_model_forward_matches_cpu_model() {
    use scirust_sciagent::gpu::ResidentModel;

    let config = tiny_tied();
    let mut model = SciAgentModel::new(&config);
    let seq_len = 8usize;
    let ids: Vec<usize> = (0..seq_len)
        .map(|i| (i * 7 + 3) % config.vocab_size)
        .collect();

    // The model's own CPU forward logits.
    let tape = Tape::new();
    let logits_v = model.forward(&tape, &ids, seq_len);
    let cpu_logits = tape.value(logits_v.idx()).data;

    // The resident path, from the same weights.
    let Some(rm) = ResidentModel::from_model(&model)
    else
    {
        eprintln!("wgpu: no adapter, skipping resident-model parity");
        return;
    };
    eprintln!("resident model on: {}", rm.adapter_name());
    let tokens: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
    let gpu_logits = rm.forward(&tokens);

    assert_eq!(gpu_logits.len(), cpu_logits.len(), "logit shape mismatch");
    let e = rel_err(&gpu_logits, &cpu_logits);
    assert!(e < 3e-3, "resident vs CPU model logits: rel_err {e}");
    eprintln!("resident-model forward rel_err {e:.2e} — PASS");
}

/// A **resident AdamW training step** on the real model reduces the loss: forward
/// → cross-entropy grad → full backward → AdamW on every trainable weight, all in
/// VRAM, iterated on a fixed `(tokens, targets)` pair. Proves the whole resident
/// training loop works end-to-end on the actual `SciAgentModel`. Skips with no
/// adapter.
#[test]
fn resident_train_step_reduces_loss() {
    use scirust_sciagent::gpu::ResidentModel;

    let config = tiny_tied();
    let model = SciAgentModel::new(&config);
    let Some(mut rm) = ResidentModel::from_model(&model)
    else
    {
        eprintln!("wgpu: no adapter, skipping resident training");
        return;
    };
    let seq_len = 8usize;
    let tokens: Vec<u32> = (0..seq_len)
        .map(|i| ((i * 7 + 3) % config.vocab_size) as u32)
        .collect();
    let targets: Vec<u32> = (0..seq_len)
        .map(|i| ((i * 5 + 1) % config.vocab_size) as u32)
        .collect();
    let betas = (0.9, 0.999);

    let first = rm.train_step(&tokens, &targets, 0.05, betas, 1e-8, 0.0);
    let mut last = first;
    for _ in 0..25
    {
        last = rm.train_step(&tokens, &targets, 0.05, betas, 1e-8, 0.0);
    }
    eprintln!("resident training: loss {first:.4} -> {last:.4}");
    assert!(
        last < first * 0.7,
        "resident training did not reduce the loss: {first} -> {last}"
    );
}

/// `sync_to_model` round-trips: after a few resident training steps, writing the
/// resident weights back into the `SciAgentModel` makes its own CPU forward match
/// the resident forward (they now hold the same weights). Skips with no adapter.
#[test]
fn resident_sync_roundtrips_into_model() {
    use scirust_sciagent::gpu::ResidentModel;

    let config = tiny_tied();
    let mut model = SciAgentModel::new(&config);
    let Some(mut rm) = ResidentModel::from_model(&model)
    else
    {
        eprintln!("wgpu: no adapter, skipping resident sync");
        return;
    };
    let seq_len = 8usize;
    let tokens: Vec<u32> = (0..seq_len)
        .map(|i| ((i * 7 + 3) % config.vocab_size) as u32)
        .collect();
    let targets: Vec<u32> = (0..seq_len)
        .map(|i| ((i * 5 + 1) % config.vocab_size) as u32)
        .collect();
    for _ in 0..5
    {
        rm.train_step(&tokens, &targets, 0.05, (0.9, 0.999), 1e-8, 0.0);
    }
    // Write the trained weights back, then compare the model's own CPU forward.
    rm.sync_to_model(&mut model);
    let ids: Vec<usize> = tokens.iter().map(|&t| t as usize).collect();
    let tape = Tape::new();
    let lv = model.forward(&tape, &ids, seq_len);
    let cpu_logits = tape.value(lv.idx()).data;
    let gpu_logits = rm.forward(&tokens);
    let e = rel_err(&gpu_logits, &cpu_logits);
    assert!(e < 3e-3, "post-sync model vs resident logits: rel_err {e}");
    eprintln!("post-sync round-trip rel_err {e:.2e} — PASS");
}
