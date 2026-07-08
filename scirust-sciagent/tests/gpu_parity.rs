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

/// The **resident next-token pretraining loop** (`train_tokens`) learns a
/// repeating token pattern: sliding a window over the stream and training each
/// window on the GPU drops the loss. End-to-end proof of a resident pretraining
/// run on the real model. Skips with no adapter.
#[test]
fn resident_train_tokens_reduces_loss() {
    use scirust_sciagent::gpu::{ResidentModel, ResidentTrainConfig};

    let config = tiny_tied();
    let model = SciAgentModel::new(&config);
    let Some(mut rm) = ResidentModel::from_model(&model)
    else
    {
        eprintln!("wgpu: no adapter, skipping resident pretraining");
        return;
    };
    // A short, learnable repeating pattern of valid token ids.
    let pattern = [1u32, 5, 9, 2, 7, 3];
    let tokens: Vec<u32> = (0..8 * 40).map(|i| pattern[i % pattern.len()]).collect();
    let cfg = ResidentTrainConfig {
        lr: 0.03,
        seq_len: 8,
        weight_decay: 0.0,
        ..Default::default()
    };

    let losses = rm.train_tokens(&tokens, &cfg);
    assert!(
        losses.len() >= 20,
        "expected many windows, got {}",
        losses.len()
    );
    let first: f32 = losses[..5].iter().sum::<f32>() / 5.0;
    let last: f32 = losses[losses.len() - 5..].iter().sum::<f32>() / 5.0;
    eprintln!("resident pretraining: loss {first:.4} (first 5) -> {last:.4} (last 5)");
    assert!(
        last < first * 0.8,
        "resident pretraining did not reduce the loss: {first} -> {last}"
    );
}

/// The **production resident pretraining harness** (`pretrain`): a warmup + cosine
/// LR schedule and periodic checkpointing over a token stream, all in VRAM.
/// Checks the loss descends, exactly `total_steps` windows run, a checkpoint is
/// written at `save_interval` with the right `meta.step`, and it reloads into a
/// fresh `SciAgentModel` producing finite, correctly-shaped logits. Skips with no
/// adapter.
#[test]
fn resident_pretrain_schedules_and_checkpoints() {
    use scirust_sciagent::gpu::{ResidentModel, ResidentPretrainConfig};
    use scirust_sciagent::train::checkpoint::{latest_checkpoint, load_checkpoint};

    let config = tiny_tied();
    let mut model = SciAgentModel::new(&config);
    let Some(mut rm) = ResidentModel::from_model(&model)
    else
    {
        eprintln!("wgpu: no adapter, skipping resident pretrain");
        return;
    };
    let pattern = [1u32, 5, 9, 2, 7, 3];
    let tokens: Vec<u32> = (0..8 * 30).map(|i| pattern[i % pattern.len()]).collect();

    let ckpt_dir = std::env::temp_dir().join("scirust_resident_pretrain_ckpt");
    let _ = std::fs::remove_dir_all(&ckpt_dir);
    let cfg = ResidentPretrainConfig {
        base_lr: 0.03,
        min_lr: 0.003,
        warmup_steps: 5,
        total_steps: 40,
        start_step: 0,
        seq_len: 8,
        weight_decay: 0.0,
        log_interval: 0,
        save_interval: 20,
        checkpoint_dir: ckpt_dir.to_string_lossy().into_owned(),
        ..Default::default()
    };

    let losses = rm.pretrain(&tokens, &mut model, &config, &cfg);
    assert_eq!(losses.len(), 40, "should run exactly total_steps windows");
    let first: f32 = losses[..5].iter().sum::<f32>() / 5.0;
    let last: f32 = losses[losses.len() - 5..].iter().sum::<f32>() / 5.0;
    eprintln!("resident pretrain: loss {first:.4} -> {last:.4}");
    assert!(
        last < first * 0.8,
        "pretrain must descend: {first} -> {last}"
    );

    // A checkpoint was written and reloads into a fresh model.
    let latest = latest_checkpoint(&ckpt_dir).expect("a checkpoint should exist");
    let mut reloaded = SciAgentModel::new(&config);
    let meta = load_checkpoint(&mut reloaded, &latest).expect("checkpoint should load");
    assert_eq!(
        meta.step, 40,
        "latest checkpoint should be the last save step"
    );
    let ids: Vec<usize> = (0..8usize)
        .map(|i| pattern[i % pattern.len()] as usize)
        .collect();
    let tape = Tape::new();
    let lv = reloaded.forward(&tape, &ids, 8);
    let logits = tape.value(lv.idx()).data;
    assert_eq!(logits.len(), 8 * config.vocab_size);
    assert!(
        logits.iter().all(|x| x.is_finite()),
        "reloaded logits must be finite"
    );
    let _ = std::fs::remove_dir_all(&ckpt_dir);
    eprintln!(
        "resident pretrain: checkpoint at {} reloaded — PASS",
        latest.display()
    );
}

/// **Resident LoRA fine-tuning** (`ResidentLoraModel`): the base model is frozen
/// and only q/k/v/o LoRA adapters train. Checks (1) at init the adapters are a
/// no-op — the LoRA forward equals the base model's forward; (2) fine-tuning the
/// adapters reduces the loss; (3) `sync_to_model` merges the adapters into the
/// base so the model's own CPU forward matches the LoRA forward. Skips with no
/// adapter.
#[test]
fn resident_lora_finetune_reduces_loss_and_syncs() {
    use scirust_sciagent::gpu::{LoraConfig, ResidentLoraModel};

    let config = tiny_tied();
    let mut model = SciAgentModel::new(&config);
    let seq_len = 8usize;
    let ids: Vec<usize> = (0..seq_len)
        .map(|i| (i * 7 + 3) % config.vocab_size)
        .collect();

    // Base model's own CPU forward (reference for "adapters are a no-op at init").
    let tape = Tape::new();
    let lv = model.forward(&tape, &ids, seq_len);
    let cpu_base = tape.value(lv.idx()).data;

    let Some(mut rm) = ResidentLoraModel::from_model(
        &model,
        LoraConfig {
            rank: 4,
            alpha: 8.0,
        },
    )
    else
    {
        eprintln!("wgpu: no adapter, skipping resident LoRA");
        return;
    };
    eprintln!("resident LoRA on: {}", rm.adapter_name());

    // At init (B = 0) the adapters are identity ⇒ LoRA forward == base forward.
    let tokens: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
    let lora0 = rm.forward(&tokens);
    let e0 = rel_err(&lora0, &cpu_base);
    assert!(e0 < 3e-3, "init LoRA forward must equal base: rel_err {e0}");

    // Fine-tune only the adapters on a fixed (tokens, targets) pair.
    let targets: Vec<u32> = (0..seq_len)
        .map(|i| ((i * 5 + 1) % config.vocab_size) as u32)
        .collect();
    let betas = (0.9, 0.999);
    let first = rm.train_step(&tokens, &targets, 0.05, betas, 1e-8, 0.0);
    let mut last = first;
    for _ in 0..30
    {
        last = rm.train_step(&tokens, &targets, 0.05, betas, 1e-8, 0.0);
    }
    eprintln!("resident LoRA fine-tune: loss {first:.4} -> {last:.4}");
    assert!(
        last < first * 0.8,
        "LoRA fine-tune must reduce the loss: {first} -> {last}"
    );

    // sync merges the adapters into the base: model CPU forward == LoRA forward.
    rm.sync_to_model(&mut model);
    let tape = Tape::new();
    let lv = model.forward(&tape, &ids, seq_len);
    let cpu_merged = tape.value(lv.idx()).data;
    let lora_now = rm.forward(&tokens);
    let e = rel_err(&cpu_merged, &lora_now);
    assert!(e < 3e-3, "post-sync merge mismatch: rel_err {e}");
    eprintln!("resident LoRA fine-tune + merge — PASS");
}

/// **Resident DoRA fine-tuning** (`ResidentDoraModel`): the base is frozen and
/// only q/k/v/o DoRA adapters (direction `A`/`B` + per-row magnitude `m`) train.
/// Checks (1) at init the adapters are a no-op — the DoRA forward equals the base
/// model's forward (`B = 0`, `m = ‖W₀‖_row`); (2) fine-tuning reduces the loss;
/// (3) `sync_to_model` merges the effective weights so the model's own CPU forward
/// matches the DoRA forward. Skips with no adapter.
#[test]
fn resident_dora_finetune_reduces_loss_and_syncs() {
    use scirust_sciagent::gpu::{DoraConfig, ResidentDoraModel};

    let config = tiny_tied();
    let mut model = SciAgentModel::new(&config);
    let seq_len = 8usize;
    let ids: Vec<usize> = (0..seq_len)
        .map(|i| (i * 7 + 3) % config.vocab_size)
        .collect();

    let tape = Tape::new();
    let lv = model.forward(&tape, &ids, seq_len);
    let cpu_base = tape.value(lv.idx()).data;

    let Some(mut rm) = ResidentDoraModel::from_model(&model, DoraConfig { rank: 4 })
    else
    {
        eprintln!("wgpu: no adapter, skipping resident DoRA");
        return;
    };
    eprintln!("resident DoRA on: {}", rm.adapter_name());

    // At init (B = 0, m = ‖W₀‖_row) the effective weight is exactly W₀.
    let tokens: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
    let dora0 = rm.forward(&tokens);
    let e0 = rel_err(&dora0, &cpu_base);
    assert!(e0 < 3e-3, "init DoRA forward must equal base: rel_err {e0}");

    // Fine-tune only the DoRA adapters.
    let targets: Vec<u32> = (0..seq_len)
        .map(|i| ((i * 5 + 1) % config.vocab_size) as u32)
        .collect();
    let betas = (0.9, 0.999);
    let first = rm.train_step(&tokens, &targets, 0.05, betas, 1e-8, 0.0);
    let mut last = first;
    for _ in 0..30
    {
        last = rm.train_step(&tokens, &targets, 0.05, betas, 1e-8, 0.0);
    }
    eprintln!("resident DoRA fine-tune: loss {first:.4} -> {last:.4}");
    assert!(
        last < first * 0.8,
        "DoRA fine-tune must reduce the loss: {first} -> {last}"
    );

    // sync merges the effective weights: model CPU forward == DoRA forward.
    rm.sync_to_model(&mut model);
    let tape = Tape::new();
    let lv = model.forward(&tape, &ids, seq_len);
    let cpu_merged = tape.value(lv.idx()).data;
    let dora_now = rm.forward(&tokens);
    let e = rel_err(&cpu_merged, &dora_now);
    assert!(e < 3e-3, "post-sync merge mismatch: rel_err {e}");
    eprintln!("resident DoRA fine-tune + merge — PASS");
}

/// **Resident greedy generation** (`ResidentModel::generate`) matches a CPU greedy
/// decode of the same weights: at each step the resident argmax next-token equals
/// the CPU model's, and `generate` reproduces the step-by-step continuation. The
/// on-device inference side of the loop (pretrain/fine-tune → merge → generate).
/// Skips with no adapter.
#[test]
fn resident_generate_matches_cpu_greedy() {
    use scirust_sciagent::gpu::ResidentModel;

    fn argmax(xs: &[f32]) -> usize {
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &v) in xs.iter().enumerate()
        {
            if v > bv
            {
                bv = v;
                best = i;
            }
        }
        best
    }

    let config = tiny_tied();
    let mut model = SciAgentModel::new(&config);
    let prompt: Vec<u32> = vec![3, 7, 1, 4];
    let Some(rm) = ResidentModel::from_model(&model)
    else
    {
        eprintln!("wgpu: no adapter, skipping resident generate");
        return;
    };
    eprintln!("resident generate on: {}", rm.adapter_name());
    let vocab = config.vocab_size;

    // Step-by-step greedy: the resident and CPU argmax next-token must agree.
    let mut toks = prompt.clone();
    let steps = 5usize;
    for _ in 0..steps
    {
        let seq = toks.len();
        let ids: Vec<usize> = toks.iter().map(|&t| t as usize).collect();
        let tape = Tape::new();
        let lv = model.forward(&tape, &ids, seq);
        let cpu_logits = tape.value(lv.idx()).data;
        let cpu_next = argmax(&cpu_logits[(seq - 1) * vocab..seq * vocab]);

        let res_logits = rm.forward(&toks);
        let res_next = argmax(&res_logits[(seq - 1) * vocab..seq * vocab]);
        assert_eq!(
            cpu_next, res_next,
            "greedy next-token mismatch at seq {seq}"
        );
        toks.push(cpu_next as u32);
    }

    // The one-shot generate() reproduces the same continuation.
    let gen = rm.generate(&prompt, steps);
    assert_eq!(gen, toks, "generate() must match step-by-step greedy");
    eprintln!("resident greedy generation matches CPU — PASS ({gen:?})");
}
