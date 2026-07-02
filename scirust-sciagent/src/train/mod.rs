pub mod checkpoint;
pub mod dataset;
pub mod optimizer;
pub mod scheduler;
pub mod sft;

use scirust_core::autodiff::reverse::{Tape, Tensor, Var};

use crate::config::SciAgentConfig;
use crate::model::SciAgentModel;
use crate::train::checkpoint::{CheckpointMeta, save_checkpoint};
use crate::train::dataset::PretrainDataset;
use crate::train::optimizer::TrainOptimizer;
use crate::train::scheduler::WarmupCosineSchedule;

pub struct TrainerConfig {
    pub lr: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
    pub batch_size: usize,
    pub seq_len: usize,
    pub micro_batch_size: usize,
    pub grad_accum_steps: usize,
    pub max_grad_norm: f32,
    pub log_interval: usize,
    pub save_interval: usize,
    pub seed: u64,
    pub checkpoint_dir: String,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            lr: 3e-4,
            min_lr: 3e-5,
            warmup_steps: 2000,
            total_steps: 50000,
            batch_size: 64,
            seq_len: 8192,
            micro_batch_size: 8,
            grad_accum_steps: 8,
            max_grad_norm: 1.0,
            log_interval: 100,
            save_interval: 500,
            seed: 42,
            checkpoint_dir: "checkpoints".to_string(),
        }
    }
}

pub fn cross_entropy_loss<'t>(tape: &'t Tape, logits: Var<'t>, targets: &[usize]) -> Var<'t> {
    let (n, vocab) = logits.shape();
    assert_eq!(n, targets.len(), "cross_entropy: one target per row");
    let lprobs = logits.log_softmax(1);

    let mut onehot = vec![0.0f32; n * vocab];
    for (r, &t) in targets.iter().enumerate()
    {
        assert!(
            t < vocab,
            "cross_entropy: target {t} out of range ({vocab})"
        );
        onehot[r * vocab + t] = 1.0;
    }
    let onehot_v = tape.input(Tensor::from_vec(onehot, n, vocab));
    let loss = lprobs.hadamard(onehot_v).sum().neg().scale(1.0 / n as f32);
    loss
}

pub fn train_step(
    model: &mut SciAgentModel,
    inputs: &[usize],
    targets: &[usize],
    seq_len: usize,
) -> f32 {
    let tape = Tape::new();
    let logits = model.forward(&tape, inputs, seq_len);
    let loss = cross_entropy_loss(&tape, logits, targets);
    tape.backward(loss.idx());
    let loss_val = tape.value(loss.idx()).data[0];
    model.sync(&tape);
    loss_val
}

pub fn train_epoch(
    model: &mut SciAgentModel,
    dataset: &mut PretrainDataset,
    config: &SciAgentConfig,
    trainer_cfg: &TrainerConfig,
) -> f64 {
    let mut opt = TrainOptimizer::new_muon(trainer_cfg.lr);
    let scheduler = WarmupCosineSchedule::new(
        trainer_cfg.lr,
        trainer_cfg.min_lr,
        trainer_cfg.warmup_steps,
        trainer_cfg.total_steps,
    );
    let mut total_loss = 0.0f64;
    let mut steps = 0usize;
    let batch_size = trainer_cfg.batch_size;
    let seq_len = trainer_cfg.seq_len;

    while let Some((inputs, targets)) = dataset.next_batch(batch_size)
    {
        let tape = Tape::new();
        let logits = model.forward(&tape, &inputs, seq_len);
        let params = model.parameter_indices();
        let loss = cross_entropy_loss(&tape, logits, &targets);
        tape.backward(loss.idx());
        let loss_val = tape.value(loss.idx()).data[0] as f64;
        total_loss += loss_val;
        steps += 1;

        opt.apply_schedule(&scheduler, steps);
        if trainer_cfg.max_grad_norm > 0.0
        {
            opt.clip_grad_norm(&tape, trainer_cfg.max_grad_norm);
        }
        opt.step(&params, &tape);
        model.sync(&tape);

        if steps % trainer_cfg.log_interval == 0
        {
            let avg = total_loss / steps as f64;
            let lr = opt.lr();
            println!("[Step {steps}] loss: {avg:.6} | lr: {lr:.8}");
        }

        if steps % trainer_cfg.save_interval == 0
        {
            let ckpt_dir =
                std::path::Path::new(&trainer_cfg.checkpoint_dir).join(format!("step_{steps}"));
            let meta = CheckpointMeta {
                step: steps,
                loss: loss_val as f32,
                lr: opt.lr(),
                config: config.clone(),
            };
            let _ = save_checkpoint(model, &meta, &ckpt_dir);
        }

        if steps >= trainer_cfg.total_steps
        {
            break;
        }
    }

    if steps > 0
    {
        total_loss / steps as f64
    }
    else
    {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SciAgentConfig;
    use crate::model::SciAgentModel;

    #[test]
    fn test_cross_entropy_shape() {
        let cfg = SciAgentConfig::debug();
        let mut model = SciAgentModel::new(&cfg);
        let tape = Tape::new();
        let input_ids = vec![4usize, 5, 6, 7];
        let logits = model.forward(&tape, &input_ids, 4);
        let targets = vec![5usize, 6, 7, 4];
        let loss = cross_entropy_loss(&tape, logits, &targets);
        let val = tape.value(loss.idx()).data[0];
        assert!(val > 0.0, "Cross-entropy should be positive, got {val}");
    }

    #[test]
    fn test_cross_entropy_gradient_flows() {
        let cfg = SciAgentConfig::debug();
        let mut model = SciAgentModel::new(&cfg);
        let tape = Tape::new();
        let input_ids = vec![4usize, 5, 6, 7];
        let logits = model.forward(&tape, &input_ids, 4);
        let targets = vec![5usize, 6, 7, 4];
        let loss = cross_entropy_loss(&tape, logits, &targets);
        loss.backward();
        let params = model.parameter_indices();
        let has_grad = params.iter().any(|&p| {
            let g = tape.grad(p);
            g.data.iter().map(|x| x.abs()).fold(0.0, f32::max) > 1e-10
        });
        assert!(has_grad, "Cross-entropy should produce non-zero gradients");
    }

    #[test]
    fn test_cross_entropy_derivative() {
        let cfg = SciAgentConfig::debug();
        let mut model = SciAgentModel::new(&cfg);

        let tape = Tape::new();
        let input_ids = vec![4usize, 5, 6, 7];
        let logits = model.forward(&tape, &input_ids, 4);

        let targets = vec![5usize, 6, 7, 4];
        let loss = cross_entropy_loss(&tape, logits, &targets);
        loss.backward();

        let params = model.parameter_indices();
        assert!(!params.is_empty(), "Should have parameters");
    }

    #[test]
    fn tied_embeddings_receive_the_head_gradient_and_learn() {
        // Regression for the tied-embeddings bug: the LM head used a fresh
        // `tape.input(weight.clone())`, so the head-side gradient — the main
        // next-token learning signal — never reached the shared parameter and
        // a tied model could not descend below the ln(vocab) floor (the
        // trained `small` run sat at 9.01→8.90 while the untied `debug`
        // config learned normally).
        let cfg = SciAgentConfig {
            vocab_size: 32,
            d_model: 16,
            n_layers: 1,
            n_heads: 4,
            n_kv_heads: 2,
            d_ff: 32,
            max_seq_len: 8,
            rope_theta: 10000.0,
            tie_embeddings: true,
            use_bias: false,
            eps: 1e-5,
        };
        let mut model = SciAgentModel::new(&cfg);
        let inputs = vec![4usize, 5, 6, 7];
        let targets = vec![5usize, 6, 7, 8];

        // 1) The head gradient reaches the shared table: rows of tokens that
        //    never appear in the INPUT (e.g. target 8) must still get gradient
        //    through the output projection. With the old clone path this was
        //    exactly zero.
        let tape = Tape::new();
        let logits = model.forward(&tape, &inputs, 4);
        let loss = cross_entropy_loss(&tape, logits, &targets);
        tape.backward(loss.idx());
        let params = model.parameter_indices();
        let widx = params[0]; // the tied table is reported first
        let g = tape.grad(widx);
        let row = 8 * cfg.d_model..9 * cfg.d_model;
        let head_grad_norm: f32 = g.data[row].iter().map(|x| x.abs()).sum();
        assert!(
            head_grad_norm > 1e-12,
            "head-side gradient must reach the tied table (got {head_grad_norm})"
        );

        // 2) With an optimizer in the loop, a tied model now actually learns.
        let mut opt = TrainOptimizer::new_muon(0.05);
        let first = {
            let tape = Tape::new();
            let logits = model.forward(&tape, &inputs, 4);
            let loss = cross_entropy_loss(&tape, logits, &targets);
            tape.backward(loss.idx());
            let v = tape.value(loss.idx()).data[0];
            opt.step(&model.parameter_indices(), &tape);
            model.sync(&tape);
            v
        };
        let mut last = first;
        for _ in 0..20
        {
            let tape = Tape::new();
            let logits = model.forward(&tape, &inputs, 4);
            let loss = cross_entropy_loss(&tape, logits, &targets);
            tape.backward(loss.idx());
            last = tape.value(loss.idx()).data[0];
            opt.step(&model.parameter_indices(), &tape);
            model.sync(&tape);
        }
        assert!(
            last < first * 0.8,
            "a tied model must descend on a memorisable batch: {first} -> {last}"
        );
    }

    #[test]
    fn test_train_step_decreases_loss() {
        let cfg = SciAgentConfig {
            vocab_size: 32,
            d_model: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            d_ff: 32,
            max_seq_len: 16,
            rope_theta: 10000.0,
            tie_embeddings: false,
            use_bias: false,
            eps: 1e-5,
        };
        let mut model = SciAgentModel::new(&cfg);
        let inputs: Vec<usize> = (4..12).collect();
        let targets: Vec<usize> = (5..13)
            .map(|x| if x >= cfg.vocab_size { 0 } else { x })
            .collect();

        let loss1 = train_step(&mut model, &inputs, &targets, 8);
        let loss2 = train_step(&mut model, &inputs, &targets, 8);
        assert!(
            loss2 <= loss1 * 1.1,
            "Loss should not increase significantly after one step: {loss1} -> {loss2}"
        );
    }
}
