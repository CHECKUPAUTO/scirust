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
use scirust_sciagent::cuda_model::{CudaModel, CudaTrainer};
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

/// The mixed-precision [`CudaTrainer`] actually **learns**: repeated AdamW steps on
/// a fixed batch drive the cross-entropy loss down — B4f of `ROUTE_B.md`, the closed
/// bf16 training loop (forward → CE grad → backward → fp32-master AdamW → refreshed
/// bf16 views, all on Tensor cores). A memorization check: overfitting one batch is
/// the minimal proof the whole loop's signs and scales are right. Skips with no
/// device.
#[test]
fn cuda_trainer_reduces_loss() {
    let config = tiny_tied();
    let model = SciAgentModel::new(&config);
    let seq_len = 8usize;
    let tokens: Vec<u32> = (0..seq_len)
        .map(|i| ((i * 7 + 3) % config.vocab_size) as u32)
        .collect();
    let targets: Vec<u32> = (0..seq_len)
        .map(|i| ((i * 5 + 1) % config.vocab_size) as u32)
        .collect();

    let Some(mut trainer) = CudaTrainer::from_model(&model)
    else
    {
        eprintln!("cuda: no device, skipping CUDA trainer loss-decrease");
        return;
    };

    let (lr, betas, eps, wd) = (3e-3f32, (0.9f32, 0.999f32), 1e-8f32, 0.0f32);
    let first = trainer.train_step(&tokens, &targets, lr, betas, eps, wd);
    let mut last = first;
    for _ in 0..40
    {
        last = trainer.train_step(&tokens, &targets, lr, betas, eps, wd);
    }
    eprintln!("CUDA bf16 trainer: loss {first:.4} → {last:.4} over 41 steps");
    assert!(
        last < first * 0.7,
        "CUDA bf16 training did not reduce loss: {first:.4} → {last:.4}"
    );
    eprintln!("CUDA bf16 Tensor-core training loop reduces loss — PASS");
}

/// B31: CUDA KV-cached greedy decoding must reproduce the original non-cached
/// CUDA decoder token-for-token. This pins prompt prefill, absolute-position RoPE,
/// per-layer K/V cache growth and incremental GQA as one end-to-end contract.
#[test]
fn cuda_cached_generation_matches_naive_greedy() {
    let config = tiny_tied();
    let model = SciAgentModel::new(&config);
    let Some(cm) = CudaModel::from_model(&model)
    else
    {
        eprintln!("cuda: no device, skipping cached-generation parity");
        return;
    };
    let prompt = vec![3u32, 5, 7, 11];
    let params = scirust_sciagent::generate::SamplingParams::default();
    let naive = cm.generate(&prompt, 6, &params, 0xB31);
    let cached = cm.generate_cached(&prompt, 6, &params, 0xB31);
    assert_eq!(cached, naive, "CUDA KV cache changed greedy decode tokens");
}

/// Empty-prompt behavior is part of the existing CUDA generation API: generation
/// starts from token 0. The cached path must preserve that behavior exactly.
#[test]
fn cuda_cached_generation_preserves_empty_prompt_semantics() {
    let config = tiny_tied();
    let model = SciAgentModel::new(&config);
    let Some(cm) = CudaModel::from_model(&model)
    else
    {
        eprintln!("cuda: no device, skipping cached empty-prompt parity");
        return;
    };
    let params = scirust_sciagent::generate::SamplingParams::default();
    assert_eq!(
        cm.generate_cached(&[], 3, &params, 7),
        cm.generate(&[], 3, &params, 7)
    );
}

/// B32: an interrupted CUDA run restored from model + optimizer sidecars must take
/// the same next AdamW step as the uninterrupted trainer. This is the regression
/// test for moment/bias-correction loss on resume.
#[test]
fn cuda_optimizer_resume_matches_uninterrupted_next_step() {
    use scirust_sciagent::cuda_model::CudaPretrainConfig;
    use scirust_sciagent::train::checkpoint::{CheckpointMeta, load_checkpoint, save_checkpoint};
    use std::path::PathBuf;

    let config = tiny_tied();
    let mut model = SciAgentModel::new(&config);
    let Some(mut continuous) = CudaTrainer::from_model(&model)
    else
    {
        eprintln!("cuda: no device, skipping optimizer-resume parity");
        return;
    };
    let tokens: Vec<u32> = (0..8)
        .map(|i| ((i * 7 + 3) % config.vocab_size) as u32)
        .collect();
    let targets: Vec<u32> = (0..8)
        .map(|i| ((i * 5 + 1) % config.vocab_size) as u32)
        .collect();
    let (lr, betas, eps, wd) = (3e-3f32, (0.9f32, 0.95f32), 1e-5f32, 0.0f32);
    for _ in 0..3
    {
        continuous.train_step(&tokens, &targets, lr, betas, eps, wd);
    }
    continuous.sync_to_model(&mut model);

    let dir = PathBuf::from("/tmp/scirust_cuda_optimizer_resume");
    let _ = std::fs::remove_dir_all(&dir);
    let meta = CheckpointMeta {
        step: 3,
        loss: 0.0,
        lr,
        config: config.clone(),
    };
    save_checkpoint(&model, &meta, &dir).expect("save model checkpoint");
    let cfg = CudaPretrainConfig {
        base_lr: lr,
        min_lr: lr * 0.1,
        warmup_steps: 1,
        total_steps: 10,
        betas,
        adam_eps: eps,
        weight_decay: wd,
        ..Default::default()
    };
    continuous
        .save_optimizer_state(&cfg, &dir)
        .expect("save optimizer state");

    let mut resumed_model = SciAgentModel::new(&config);
    load_checkpoint(&mut resumed_model, &dir).expect("reload model checkpoint");
    let mut resumed = CudaTrainer::from_model(&resumed_model).expect("CUDA trainer");
    let resume_meta = resumed
        .load_optimizer_state(&dir)
        .expect("load optimizer state")
        .expect("B32 optimizer sidecar");
    assert_eq!(resume_meta.step, 3);

    let loss_a = continuous.train_step(&tokens, &targets, lr, betas, eps, wd);
    let loss_b = resumed.train_step(&tokens, &targets, lr, betas, eps, wd);
    assert!(
        (loss_a - loss_b).abs() < 1e-4,
        "resume loss mismatch: {loss_a} vs {loss_b}"
    );

    let mut a = SciAgentModel::new(&config);
    let mut b = SciAgentModel::new(&config);
    continuous.sync_to_model(&mut a);
    resumed.sync_to_model(&mut b);
    let logits_a = CudaModel::from_model(&a)
        .expect("CUDA model A")
        .forward(&tokens);
    let logits_b = CudaModel::from_model(&b)
        .expect("CUDA model B")
        .forward(&tokens);
    let e = rel_err(&logits_a, &logits_b);
    assert!(
        e < 1e-4,
        "resumed optimizer diverged from uninterrupted step: rel_err {e}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
