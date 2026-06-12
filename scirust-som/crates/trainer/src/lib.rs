//! SOM training loop.
//!
//! One tape per sample, Adam optimizer, multi-task loss: cross-entropy on
//! ownership classes, cross-entropy on borrow classes, and MSE on the
//! sigmoid of the fault head. Fully deterministic: the model seed fixes
//! the weights, the dataset seed fixes the data, and execution is
//! single-threaded — two identical runs produce bit-identical losses.

use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::{CrossEntropyLoss, Loss, MseLoss};
use scirust_som_dataset::TrainingSample;
use scirust_som_model::SomModel;

#[derive(Debug, Clone)]
pub struct TrainerConfig {
    pub epochs: usize,
    pub learning_rate: f32,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            epochs: 3,
            learning_rate: 0.01,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainReport {
    /// Mean loss per epoch, in order.
    pub epoch_losses: Vec<f32>,
}

impl TrainReport {
    pub fn first_loss(&self) -> f32 {
        *self.epoch_losses.first().expect("at least one epoch")
    }
    pub fn last_loss(&self) -> f32 {
        *self.epoch_losses.last().expect("at least one epoch")
    }
}

/// Train `model` in place on `samples`. Returns per-epoch mean losses.
pub fn train(model: &mut SomModel, samples: &[TrainingSample], cfg: &TrainerConfig) -> TrainReport {
    assert!(!samples.is_empty(), "empty training set");
    let ce = CrossEntropyLoss::new();
    let mse = MseLoss::new();
    let mut opt = Adam::new(cfg.learning_rate);
    let mut epoch_losses = Vec::with_capacity(cfg.epochs);

    for _ in 0..cfg.epochs
    {
        let mut total = 0.0f32;
        for sample in samples
        {
            let tape = Tape::new();
            let logits = model.forward(&tape, &sample.token_ids);
            let seq = sample.token_ids.len();

            let own_targets =
                Tensor::from_vec(sample.ownership.iter().map(|&c| c as f32).collect(), seq, 1);
            let bor_targets =
                Tensor::from_vec(sample.borrow.iter().map(|&c| c as f32).collect(), seq, 1);
            let inv_targets = tape.input(Tensor::from_vec(sample.invalid.clone(), seq, 1));

            let loss_own = ce.forward_with_indices(&tape, logits.ownership, &own_targets);
            let loss_bor = ce.forward_with_indices(&tape, logits.borrow, &bor_targets);
            let loss_inv = mse.forward(&tape, logits.invalid.sigmoid(), inv_targets);
            let loss = loss_own.add(loss_bor).add(loss_inv);

            tape.backward(loss.idx());
            opt.step(&model.parameter_indices(), &tape);
            model.sync(&tape);

            total += tape.value(loss.idx()).data[0];
        }
        epoch_losses.push(total / samples.len() as f32);
    }

    TrainReport { epoch_losses }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_som_dataset::build_training_set;
    use scirust_som_model::SomModelConfig;
    use scirust_som_tokenizer::SomVocab;

    fn tiny_model() -> SomModel {
        SomModel::new(SomModelConfig {
            vocab_size: SomVocab::vocab_size(),
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            d_ff: 32,
            max_seq_len: 64,
            seed: 42,
            ..SomModelConfig::default()
        })
    }

    #[test]
    fn loss_decreases_on_fixed_seed() {
        let samples = build_training_set(42, 24, 64);
        let mut model = tiny_model();
        let report = train(
            &mut model,
            &samples,
            &TrainerConfig {
                epochs: 3,
                learning_rate: 0.01,
            },
        );
        assert_eq!(report.epoch_losses.len(), 3);
        assert!(
            report.last_loss() < report.first_loss(),
            "loss must decrease: {:?}",
            report.epoch_losses
        );
        assert!(report.last_loss().is_finite());
    }

    #[test]
    fn training_is_bit_deterministic() {
        let run = || -> Vec<u32> {
            let samples = build_training_set(7, 12, 64);
            let mut model = tiny_model();
            let report = train(
                &mut model,
                &samples,
                &TrainerConfig {
                    epochs: 2,
                    learning_rate: 0.01,
                },
            );
            report.epoch_losses.iter().map(|f| f.to_bits()).collect()
        };
        assert_eq!(run(), run(), "two identical runs ⇒ bit-identical losses");
    }
}
