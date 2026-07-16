//! Vertical certification: end-to-end integration tests over the recently
//! reformed stack. Where the unit tests exercise each piece in isolation,
//! every test here drives the FULL pipeline — seeded data → forward → loss →
//! backward → optimizer (through the scheduler) → sync → checkpoint → restore —
//! so a regression in any seam between the pieces (train/eval mode switching,
//! `try_forward` error propagation, ND state dicts + safetensors, the
//! `HasLearningRate`/`NdOptimizer` traits, `LrSchedule::drive`, tape `AdamW`,
//! bit-for-bit reproducibility) fails one of these certifications.

use scirust_core::autodiff::nd::NdTape;
use scirust_core::autodiff::optim::{AdamW, Optimizer};
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::autodiff::scheduler::{CosineAnnealing, LrSchedule, WarmupCosine};
use scirust_core::error::SciRustError;
use scirust_core::io::safetensors::{
    load_state_dict, load_state_dict_nd, save_state_dict, save_state_dict_nd,
};
use scirust_core::nn::nd_decoder::{NdDecoderConfig, NdDecoderLM};
use scirust_core::nn::nd_optim::{NdAdam, NdOptimizer};
use scirust_core::nn::rng::PcgEngine;
use scirust_core::nn::{
    BatchNorm1d, Dropout, KaimingNormal, Linear, Loss, Module, MseLoss, ReLU, Sequential, Zeros,
};
use scirust_core::tensor::tensor_nd::TensorND;
use scirust_stats::describe::{mean, std_dev};
use scirust_stats::htest::{Tail, t_test_one_sample};
use std::collections::HashMap;

// ================================================================== //
//  Shared fixtures: a seeded synthetic regression task               //
// ================================================================== //

const N_IN: usize = 8;
const N_OUT: usize = 4;
const HIDDEN: usize = 16;

/// Fixed random teacher weights (seeded): the ground-truth mapping the MLP
/// has to learn is `y_k = tanh(Σ_j teacher[k·N_IN + j] · x_j)`.
fn teacher_weights() -> Vec<f32> {
    let mut rng = PcgEngine::new(0x7EAC);
    (0..N_IN * N_OUT).map(|_| rng.float_signed()).collect()
}

/// One seeded batch of the synthetic task: inputs in [-1, 1), targets from
/// the (non-linear) teacher. Fully deterministic given the caller's RNG state.
fn synth_batch(rng: &mut PcgEngine, teacher: &[f32], batch: usize) -> (Tensor, Tensor) {
    let mut xs = Vec::with_capacity(batch * N_IN);
    let mut ys = Vec::with_capacity(batch * N_OUT);
    for _ in 0..batch
    {
        let x: Vec<f32> = (0..N_IN).map(|_| rng.float_signed()).collect();
        for k in 0..N_OUT
        {
            let z: f32 = (0..N_IN).map(|j| teacher[k * N_IN + j] * x[j]).sum();
            ys.push(z.tanh());
        }
        xs.extend_from_slice(&x);
    }
    (
        Tensor::from_vec(xs, batch, N_IN),
        Tensor::from_vec(ys, batch, N_OUT),
    )
}

/// The certification MLP: Linear → BatchNorm1d → ReLU → Dropout → Linear,
/// covering the three module kinds (parametric, mode-stateful with imperative
/// running stats, mode-stateful with an RNG) behind one `Sequential`.
fn build_mlp(init_seed: u64, dropout_seed: u64) -> Sequential {
    let mut rng = PcgEngine::new(init_seed);
    Sequential::new()
        .add(Linear::new(N_IN, HIDDEN, &KaimingNormal, &Zeros, &mut rng))
        .add(BatchNorm1d::new(HIDDEN))
        .add(ReLU::new())
        .add(Dropout::new(0.3, dropout_seed))
        .add(Linear::new(HIDDEN, N_OUT, &KaimingNormal, &Zeros, &mut rng))
}

/// The canonical 2-D training loop: fresh tape per step, `try_forward`, MSE
/// loss, backward, the NEW tape `AdamW` with its LR pushed each step through
/// `LrSchedule::drive` (warmup + cosine), then `sync` back into the modules.
fn train_mlp(model: &mut Sequential, steps: usize, data_seed: u64) {
    let teacher = teacher_weights();
    let sched = WarmupCosine::new(0.02, 1e-4, 10, steps);
    // lr starts at 0 on purpose: drive() must set it before every step.
    let mut opt = AdamW::new(0.0).with_weight_decay(0.01);
    let mut data_rng = PcgEngine::new(data_seed);
    for step in 0..steps
    {
        sched.drive(&mut opt, step);
        let (x, y) = synth_batch(&mut data_rng, &teacher, 32);
        let tape = Tape::new();
        let xv = tape.input(x);
        let yv = tape.input(y);
        let out = model
            .try_forward(&tape, xv)
            .expect("well-formed training forward must not error");
        let loss = MseLoss::new().forward(&tape, out, yv);
        tape.backward(loss.idx());
        opt.step(&model.parameter_indices(), &tape);
        model.sync(&tape);
    }
}

/// Forward `x` through the model on a throwaway tape and return the output.
fn forward_values(model: &mut Sequential, x: &Tensor) -> Vec<f32> {
    let tape = Tape::new();
    let xv = tape.input(x.clone());
    let y = model.forward(&tape, xv);
    tape.value(y.idx()).data
}

/// MSE of the model on a fixed (x, y) set, on a throwaway tape.
fn eval_mse(model: &mut Sequential, x: &Tensor, y: &Tensor) -> f32 {
    let tape = Tape::new();
    let xv = tape.input(x.clone());
    let yv = tape.input(y.clone());
    let out = model.forward(&tape, xv);
    let loss = MseLoss::new().forward(&tape, out, yv);
    tape.value(loss.idx()).data[0]
}

fn bits_equal_slice(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

fn bits_equal_nd(a: &TensorND, b: &TensorND) -> bool {
    a.shape == b.shape && bits_equal_slice(&a.data, &b.data)
}

// ================================================================== //
//  1. Full 2-D pipeline: train → eval mode → checkpoint → restore    //
// ================================================================== //

/// End-to-end 2-D certification: the `Sequential` MLP trains with the tape
/// `AdamW` driven by `LrSchedule::drive` (loss drops on a held-out set), then
/// `train(false)` makes inference deterministic (two eval forwards
/// bit-identical, dropout off; a train-mode forward differs), and the
/// state_dict — through the safetensors file round trip — restores into a
/// fresh differently-seeded model that produces bit-identical eval outputs.
#[test]
fn full_2d_training_pipeline_with_eval_and_checkpoint() {
    let teacher = teacher_weights();
    let mut model = build_mlp(42, 4242);

    // Held-out evaluation set, fixed for the before/after comparison.
    let mut eval_rng = PcgEngine::new(555);
    let (x_eval, y_eval) = synth_batch(&mut eval_rng, &teacher, 64);

    model.train(false);
    let loss_before = eval_mse(&mut model, &x_eval, &y_eval);
    model.train(true);

    train_mlp(&mut model, 200, 7);

    model.train(false);
    let loss_after = eval_mse(&mut model, &x_eval, &y_eval);
    assert!(
        loss_before.is_finite() && loss_after.is_finite(),
        "non-finite eval loss: before {loss_before}, after {loss_after}"
    );
    assert!(
        loss_before > 1e-3,
        "untrained model already fits the teacher (loss {loss_before}) — vacuous test"
    );
    assert!(
        loss_after < 0.5 * loss_before,
        "training did not reduce the held-out loss: before {loss_before}, after {loss_after}"
    );

    // Eval mode is deterministic: two identical forwards are bit-identical
    // (dropout must be the identity, BatchNorm must use frozen running stats).
    let eval_a = forward_values(&mut model, &x_eval);
    let eval_b = forward_values(&mut model, &x_eval);
    assert!(
        bits_equal_slice(&eval_a, &eval_b),
        "two eval-mode forwards are not bit-identical — train(false) leaks randomness"
    );

    // Back in train mode the dropout mask is live again: the output differs.
    model.train(true);
    let train_out = forward_values(&mut model, &x_eval);
    assert!(
        !bits_equal_slice(&eval_a, &train_out),
        "train(true) forward equals the eval forward — dropout never re-enabled"
    );
    model.train(false);

    // Checkpoint: state_dict → safetensors file → fresh model → load.
    // NOTE: the train-mode forward above advanced BatchNorm's running stats
    // (an imperative side effect of training forwards), so the checkpoint
    // baseline is a FRESH eval forward taken now — eval forwards leave the
    // stats untouched, making this baseline stable.
    let eval_ref = forward_values(&mut model, &x_eval);
    let sd = model.state_dict();
    let path = std::env::temp_dir().join("vertcert_2d_mlp.safetensors");
    save_state_dict(&path, &sd, None).unwrap();
    let (loaded, _meta) = load_state_dict(&path).unwrap();
    let _ = std::fs::remove_file(&path);

    let mut restored = build_mlp(999, 17); // different init AND dropout seeds
    let fresh_sd = restored.state_dict();
    assert!(
        !bits_equal_slice(&fresh_sd["0.weight"].data, &sd["0.weight"].data),
        "fresh model accidentally equals the trained one — restore check is vacuous"
    );
    restored.load_state_dict(&loaded).unwrap();
    restored.train(false);

    let eval_restored = forward_values(&mut restored, &x_eval);
    assert!(
        bits_equal_slice(&eval_ref, &eval_restored),
        "restored model's eval outputs are not bit-identical to the original's"
    );
}

// ================================================================== //
//  2. Full N-D pipeline: NdDecoderLM + dyn NdOptimizer + safetensors //
// ================================================================== //

/// End-to-end N-D certification: a tiny `NdDecoderLM` trains through the
/// `NdOptimizer` trait object (`Box<dyn NdOptimizer>` holding `NdAdam`, its LR
/// scheduled each step via `LrSchedule::drive` across the trait-upcast to
/// `HasLearningRate`), the next-token loss drops, and the ND state dict
/// round-trips through a safetensors file into a fresh model with every
/// parameter AND the next-token logits bit-identical.
#[test]
fn full_nd_training_pipeline_with_safetensors() {
    // Mirrors the tiny config used by the nd_decoder unit tests.
    let cfg = NdDecoderConfig {
        vocab: 6,
        d_model: 16,
        n_heads: 2,
        d_ff: 32,
        n_layers: 2,
        max_seq: 8,
    };
    let mut lm = NdDecoderLM::new(cfg, &mut PcgEngine::new(11));
    let seq = [1usize, 2, 3, 4, 2, 5];

    // The optimizer is held ONLY through the trait object, and scheduled
    // through the cross-family drive (dyn NdOptimizer → dyn HasLearningRate).
    let mut opt: Box<dyn NdOptimizer> = Box::new(NdAdam::with_lr(0.0));
    let sched = CosineAnnealing::new(0.02, 1e-3, 80);

    let steps = 80;
    let mut first = f32::NAN;
    let mut last = f32::NAN;
    for step in 0..steps
    {
        sched.drive(opt.as_mut(), step);
        let t = NdTape::new();
        let loss_v = lm.loss(&t, &seq);
        let loss = t.value(loss_v).data[0];
        if step == 0
        {
            first = loss;
        }
        last = loss;
        let grads = t.backward(loss_v);
        let mut params = lm.parameters();
        opt.step(&mut params, &grads);
    }
    assert!(
        first.is_finite() && last.is_finite(),
        "non-finite ND loss: first {first}, last {last}"
    );
    assert!(
        last < 0.5 * first,
        "scheduled dyn NdOptimizer did not train the decoder: first {first}, last {last}"
    );

    // Save the trained model, restore into a fresh differently-seeded one.
    let sd = lm.state_dict();
    let mut meta = HashMap::new();
    meta.insert("campaign".to_string(), "vertical_certification".to_string());
    let path = std::env::temp_dir().join("vertcert_nd_decoder.safetensors");
    save_state_dict_nd(&path, &sd, Some(meta)).unwrap();
    let (loaded, loaded_meta) = load_state_dict_nd(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(
        loaded_meta.get("campaign").map(String::as_str),
        Some("vertical_certification"),
        "metadata did not survive the safetensors round trip"
    );

    let mut restored = NdDecoderLM::new(cfg, &mut PcgEngine::new(9999));
    restored.load_state_dict(&loaded).unwrap();

    // Every parameter bit-identical after the file round trip.
    let sd2 = restored.state_dict();
    assert_eq!(sd2.len(), sd.len(), "state dict key count changed");
    for (k, v) in &sd
    {
        assert!(
            bits_equal_nd(&sd2[k], v),
            "param {k} not bit-identical after safetensors round trip"
        );
    }

    // And the restored model is functionally identical: bit-identical logits.
    let (ta, tb) = (NdTape::new(), NdTape::new());
    let la = lm.forward(&ta, &seq);
    let lb = restored.forward(&tb, &seq);
    assert!(
        bits_equal_nd(&ta.value(la), &tb.value(lb)),
        "next-token logits differ between the trained and the restored model"
    );
}

// ================================================================== //
//  3. Reproducibility: the same training run twice, bit for bit      //
// ================================================================== //

/// The end-to-end reproducibility promise: two full training runs with the
/// same seeds (init, dropout masks, data stream, scheduler, AdamW) must end
/// with bit-identical parameters — including BatchNorm's imperatively updated
/// running statistics, which the state dict also carries.
#[test]
fn training_is_deterministic_across_runs() {
    let run = || -> HashMap<String, Tensor> {
        let mut model = build_mlp(42, 4242);
        train_mlp(&mut model, 80, 7);
        model.state_dict()
    };
    let a = run();
    let b = run();

    assert_eq!(a.len(), b.len(), "state dict key count differs across runs");
    let mut keys: Vec<&String> = a.keys().collect();
    keys.sort();
    for k in keys
    {
        let (ta, tb) = (&a[k], &b[k]);
        assert_eq!(ta.shape(), tb.shape(), "shape of {k} differs across runs");
        assert!(
            bits_equal_slice(&ta.data, &tb.data),
            "param {k} not bit-identical across two identically-seeded runs"
        );
    }
}

// ================================================================== //
//  4. Gradient distribution sanity at init (scirust-stats)           //
// ================================================================== //

/// Statistical certification of the backward pass at a Kaiming init: pooled
/// over ~30 fresh seeded batches, the first Linear's weight gradients must be
/// finite, centred near zero (one-sample t-test at α = 0.01, with a generous
/// |mean| ≪ std fallback so the test flags gross breakage — all-zero,
/// one-sided, or exploding gradients — rather than seed luck), and their
/// dispersion must sit inside broad sanity bounds.
#[test]
fn gradient_distribution_is_sane() {
    let mut rng = PcgEngine::new(2024);
    let mut model = Sequential::new()
        .add(Linear::new(N_IN, HIDDEN, &KaimingNormal, &Zeros, &mut rng))
        .add(ReLU::new())
        .add(Linear::new(HIDDEN, N_OUT, &KaimingNormal, &Zeros, &mut rng));

    let teacher = teacher_weights();
    let mut data_rng = PcgEngine::new(31);
    let mut samples: Vec<f64> = Vec::new();
    for _ in 0..30
    {
        let (x, y) = synth_batch(&mut data_rng, &teacher, 32);
        let tape = Tape::new();
        let xv = tape.input(x);
        let yv = tape.input(y);
        let out = model.forward(&tape, xv);
        let loss = MseLoss::new().forward(&tape, out, yv);
        tape.backward(loss.idx());

        // First Linear's weight is the first registered parameter.
        let w_idx = model.parameter_indices()[0];
        let g = tape.grad(w_idx);
        assert_eq!(g.shape(), (N_IN, HIDDEN), "unexpected weight-grad shape");
        samples.extend(g.data.iter().map(|&v| f64::from(v)));
        // No optimizer step / sync: the model stays at its Kaiming init.
    }
    assert_eq!(samples.len(), 30 * N_IN * HIDDEN);

    // No NaN/Inf anywhere in the gradients.
    assert!(
        samples.iter().all(|v| v.is_finite()),
        "non-finite gradient at init"
    );

    let m = mean(&samples);
    let sd = std_dev(&samples);

    // Dispersion sanity: not collapsed to zero, not exploding.
    assert!(
        sd > 1e-6,
        "gradient dispersion collapsed (std {sd}) — all-zero backward?"
    );
    assert!(sd < 10.0, "gradient dispersion exploded (std {sd})");

    // Mean near zero: the t-test must not reject H0: mean = 0 at α = 0.01 —
    // or, failing that (the batches share one fixed init, so a small
    // systematic component is legitimate), the mean must still be small
    // against the gradient dispersion itself.
    let test =
        t_test_one_sample(&samples, 0.0, Tail::TwoSided).expect("enough samples for a t-test");
    assert!(
        test.p_value > 0.01 || m.abs() < 0.5 * sd,
        "gradients are not centred near 0: mean {m}, std {sd}, t {t}, p {p}",
        t = test.statistic,
        p = test.p_value
    );
}

// ================================================================== //
//  5. try_forward: structured error through a deep composite         //
// ================================================================== //

/// A mid-stack shape mismatch in a deep `Sequential` (including the
/// mode-stateful layers) must surface as a structured `Err` — the typed
/// `DimMismatch` from the matmul — through `try_forward`, not as a panic;
/// and the same stack with the mismatch fixed forwards cleanly.
#[test]
fn try_forward_propagates_through_full_stack() {
    // `mid_in` is the input width of the 4th Linear: 8 matches the stack,
    // 6 plants a mismatch four layers deep.
    let build_deep = |mid_in: usize| -> Sequential {
        let mut rng = PcgEngine::new(42);
        Sequential::new()
            .add(Linear::new(4, 8, &KaimingNormal, &Zeros, &mut rng))
            .add(BatchNorm1d::new(8))
            .add(ReLU::new())
            .add(Dropout::new(0.2, 77))
            .add(Linear::new(8, 8, &KaimingNormal, &Zeros, &mut rng))
            .add(ReLU::new())
            .add(Linear::new(mid_in, 3, &KaimingNormal, &Zeros, &mut rng))
            .add(ReLU::new())
            .add(Linear::new(3, 2, &KaimingNormal, &Zeros, &mut rng))
    };

    // Broken stack: Err (typed), no panic, even with layers before the fault
    // having already executed.
    let mut broken = build_deep(6);
    let tape = Tape::new();
    let x = tape.input(Tensor::from_vec(vec![0.5; 8], 2, 4));
    match broken.try_forward(&tape, x)
    {
        Err(SciRustError::DimMismatch { .. }) =>
        {},
        Err(other) => panic!("expected DimMismatch, got a different error: {other}"),
        Ok(_) => panic!("mid-stack shape mismatch must be an Err through try_forward"),
    }

    // Control: the identical stack with compatible widths forwards fine.
    let mut sound = build_deep(8);
    let tape2 = Tape::new();
    let x2 = tape2.input(Tensor::from_vec(vec![0.5; 8], 2, 4));
    let y = sound
        .try_forward(&tape2, x2)
        .expect("well-formed deep stack must forward");
    assert_eq!(y.shape(), (2, 2));
}
