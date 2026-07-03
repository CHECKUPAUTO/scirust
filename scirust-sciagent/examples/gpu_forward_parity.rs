//! **Does the real SCIAGENT model run correctly on the GPU?** Builds one model,
//! then runs the identical forward + backward on a CPU-only tape and on a tape
//! with the wgpu GEMM engine attached, and reports whether the logits and every
//! parameter gradient agree. This is direction C1 on real silicon: the whole
//! transformer — tied embeddings, RoPE, GQA attention, SwiGLU, tied LM head —
//! driven through `scirust-gpu`'s validated tape engine.
//!
//!   cargo run -p scirust-sciagent --features gpu --release --example gpu_forward_parity
//!
//! On the Jetson Thor (Blackwell) this is the honest on-device answer. Exit code
//! 2 means no GPU adapter was found (install a Vulkan ICD, or run on the Thor).
//!
//! Note on timing: the tape engine routes each GEMM individually, so every op
//! pays an upload/dispatch/download round-trip — the per-op path is about
//! *correctness*, not speed. Residency (the `GpuChain` path in `scirust-gpu`) is
//! where the 8–60× speedups live (`scirust-gpu`'s `gpu_bench`). What this example
//! proves is that the model's math is bit-tolerantly identical on the device.

use std::time::Instant;

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

fn main() {
    // A modest but real config: 4 layers, GQA 8:2, RoPE, SwiGLU, tied head.
    let config = SciAgentConfig {
        vocab_size: 2048,
        d_model: 256,
        n_layers: 4,
        n_heads: 8,
        n_kv_heads: 2,
        d_ff: 512,
        max_seq_len: 256,
        rope_theta: 10_000.0,
        tie_embeddings: true,
        use_bias: false,
        eps: 1e-5,
    };
    let seq_len = 128usize;
    let ids: Vec<usize> = (0..seq_len)
        .map(|i| (i * 13 + 7) % config.vocab_size)
        .collect();

    let mut model = SciAgentModel::new(&config);
    println!(
        "SCIAGENT parity: d_model {}, {} layers, {}h/{}kv, d_ff {}, vocab {}, seq {}\n",
        config.d_model,
        config.n_layers,
        config.n_heads,
        config.n_kv_heads,
        config.d_ff,
        config.vocab_size,
        seq_len
    );

    // CPU reference.
    let t = Instant::now();
    let cpu_tape = Tape::new();
    let cpu_logits = model.forward(&cpu_tape, &ids, seq_len);
    cpu_tape.backward(cpu_logits.sum().idx());
    let cpu_ms = t.elapsed().as_secs_f64() * 1e3;
    let cpu_out = cpu_tape.value(cpu_logits.idx()).data;
    let cpu_params = model.parameter_indices();
    let cpu_grads: Vec<Vec<f32>> = cpu_params.iter().map(|&i| cpu_tape.grad(i).data).collect();

    // GPU.
    let gpu_tape = Tape::new();
    let Some(name) = attach_gpu(&gpu_tape)
    else
    {
        eprintln!("no GPU adapter available. Install a Vulkan ICD or run on the Jetson Thor.");
        std::process::exit(2);
    };
    println!("GPU adapter: {name}\n");
    let t = Instant::now();
    let gpu_logits = model.forward(&gpu_tape, &ids, seq_len);
    gpu_tape.backward(gpu_logits.sum().idx());
    let gpu_ms = t.elapsed().as_secs_f64() * 1e3;
    let gpu_out = gpu_tape.value(gpu_logits.idx()).data;
    let gpu_params = model.parameter_indices();
    let gpu_grads: Vec<Vec<f32>> = gpu_params.iter().map(|&i| gpu_tape.grad(i).data).collect();

    // Report.
    let fwd = rel_err(&gpu_out, &cpu_out);
    let mut worst = 0.0f32;
    let mut worst_k = 0usize;
    for (k, (cg, gg)) in cpu_grads.iter().zip(&gpu_grads).enumerate()
    {
        let e = rel_err(gg, cg);
        if e > worst
        {
            worst = e;
            worst_k = k;
        }
    }

    println!("forward logits    rel_err {fwd:.3e}");
    println!(
        "{} param grads     worst rel_err {worst:.3e} (param #{worst_k})",
        cpu_params.len()
    );

    let fwd_ok = fwd < 3e-3;
    let bwd_ok = worst < 2e-2;
    println!(
        "\nforward {}   backward {}",
        if fwd_ok { "PASS" } else { "FAIL" },
        if bwd_ok { "PASS" } else { "FAIL" }
    );
    println!(
        "\ntiming (forward+backward, per-op dispatch incl. host round-trips):\n  \
         CPU {cpu_ms:8.1} ms     GPU {gpu_ms:8.1} ms"
    );
    println!(
        "  (per-op GPU is not the speed path — residency/GpuChain is; this proves correctness.)"
    );

    if !(fwd_ok && bwd_ok)
    {
        std::process::exit(1);
    }
}
