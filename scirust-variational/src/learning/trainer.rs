use scirust_core::autodiff::nd::{NdTape, NdVar};
use scirust_core::nn::nd_optim::{NdAdam, NdParam};
use scirust_core::nn::rng::PcgEngine;
use scirust_core::tensor::tensor_nd::TensorND;

use crate::error::Result;
use crate::learning::dataset::TrajectoryDataset;
use crate::learning::losses::LossValue;

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub learning_rate: f32,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub seed: u64,
    pub gradient_clip_norm: Option<f32>,
    pub checkpoint_dir: Option<String>,
    pub checkpoint_interval: usize,
    pub regularization_weight: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            num_epochs: 100,
            batch_size: 64,
            seed: 42,
            gradient_clip_norm: Some(10.0),
            checkpoint_dir: None,
            checkpoint_interval: 50,
            regularization_weight: 1e-5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub epoch: Vec<usize>,
    pub train_loss: Vec<f32>,
    pub val_loss: Vec<f32>,
    pub grad_norm: Vec<f32>,
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainingMetrics {
    pub fn new() -> Self {
        Self {
            epoch: Vec::new(),
            train_loss: Vec::new(),
            val_loss: Vec::new(),
            grad_norm: Vec::new(),
        }
    }

    pub fn record(&mut self, epoch: usize, train: f32, val: f32, grad_norm: f32) {
        self.epoch.push(epoch);
        self.train_loss.push(train);
        self.val_loss.push(val);
        self.grad_norm.push(grad_norm);
    }

    pub fn best_epoch(&self) -> usize {
        self.val_loss
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| self.epoch[i])
            .unwrap_or(0)
    }

    pub fn final_train_loss(&self) -> f32 {
        self.train_loss.last().copied().unwrap_or(f32::NAN)
    }

    pub fn final_val_loss(&self) -> f32 {
        self.val_loss.last().copied().unwrap_or(f32::NAN)
    }
}

pub struct PhysicsTrainer;

impl PhysicsTrainer {
    pub fn train_lnn<M, L>(
        model: &mut M,
        dataset: &TrajectoryDataset,
        config: &TrainingConfig,
        val_dataset: Option<&TrajectoryDataset>,
        compute_loss: L,
    ) -> Result<TrainingMetrics>
    where
        M: HasParameters,
        L: for<'a> Fn(
            &'a mut M,
            &'a NdTape,
            &'a [NdVar<'a>],
            &'a [NdVar<'a>],
            &'a [f32],
            &'a [f32],
        ) -> (NdVar<'a>, LossValue),
    {
        let mut opt = NdAdam::with_lr(config.learning_rate);
        let _rng = PcgEngine::new(config.seed);
        let mut metrics = TrainingMetrics::new();

        let ndim = dataset.ndim;
        let mut val_batches = 0usize;

        for epoch in 0..config.num_epochs
        {
            let mut total_train_loss = 0.0;
            let mut n_train = 0;

            for batch_start in (0..dataset.len()).step_by(config.batch_size)
            {
                let batch_end = (batch_start + config.batch_size).min(dataset.len());
                let batch_size = batch_end - batch_start;

                let mut q_batch = Vec::with_capacity(batch_size * ndim);
                let mut dq_batch = Vec::with_capacity(batch_size * ndim);
                let mut ddq_batch = Vec::with_capacity(batch_size * ndim);

                for i in batch_start..batch_end
                {
                    let s = dataset.get(i);
                    q_batch.extend_from_slice(&s.q);
                    dq_batch.extend_from_slice(&s.dq);
                    ddq_batch.extend_from_slice(&s.ddq);
                }

                let q_shape = vec![batch_size, ndim];
                let tape = NdTape::new();
                let qv = tape.input(TensorND::new(q_batch.clone(), q_shape));
                let dqv = tape.input(TensorND::new(dq_batch.clone(), vec![batch_size, ndim]));
                let q_arr = [qv];
                let dq_arr = [dqv];

                let grads = {
                    let (loss_var, _) =
                        compute_loss(model, &tape, &q_arr, &dq_arr, &ddq_batch, &q_batch);
                    let loss_val = tape.value(loss_var).data[0];
                    total_train_loss += loss_val * batch_size as f32;
                    tape.backward(loss_var)
                };

                let mut params = model.parameters();
                let total_grad_norm: f32 = grads
                    .iter()
                    .map(|g| g.data.iter().map(|x| x * x).sum::<f32>())
                    .sum::<f32>()
                    .sqrt();

                let mut clipped_grads = grads.clone();
                if let Some(clip_norm) = config.gradient_clip_norm
                {
                    if total_grad_norm > clip_norm
                    {
                        let scale = clip_norm / total_grad_norm;
                        for g in &mut clipped_grads
                        {
                            for x in g.data_mut().iter_mut()
                            {
                                *x *= scale;
                            }
                        }
                    }
                }

                opt.step(&mut params, &clipped_grads);

                n_train += batch_size;
            }

            let avg_train_loss = if n_train > 0
            {
                total_train_loss / n_train as f32
            }
            else
            {
                0.0
            };

            let avg_val_loss = if let Some(vd) = val_dataset
            {
                let mut val_loss = 0.0;
                for batch_start in (0..vd.len()).step_by(config.batch_size)
                {
                    let batch_end = (batch_start + config.batch_size).min(vd.len());
                    let batch_size = batch_end - batch_start;

                    let mut q_batch = Vec::with_capacity(batch_size * ndim);
                    let mut dq_batch = Vec::with_capacity(batch_size * ndim);
                    let mut ddq_batch = Vec::with_capacity(batch_size * ndim);

                    for i in batch_start..batch_end
                    {
                        let s = vd.get(i);
                        q_batch.extend_from_slice(&s.q);
                        dq_batch.extend_from_slice(&s.dq);
                        ddq_batch.extend_from_slice(&s.ddq);
                    }

                    let tape = NdTape::new();
                    let qv = tape.input(TensorND::new(q_batch.clone(), vec![batch_size, ndim]));
                    let dqv = tape.input(TensorND::new(dq_batch.clone(), vec![batch_size, ndim]));
                    let q_arr = [qv];
                    let dq_arr = [dqv];

                    let (loss_var, _) =
                        compute_loss(model, &tape, &q_arr, &dq_arr, &ddq_batch, &q_batch);

                    let lv = tape.value(loss_var).data[0];
                    val_loss += lv * batch_size as f32;
                    val_batches += batch_size;
                }
                val_loss / val_batches.max(1) as f32
            }
            else
            {
                f32::NAN
            };

            metrics.record(epoch, avg_train_loss, avg_val_loss, 0.0);

            if let Some(ref ckpt_dir) = config.checkpoint_dir
            {
                if epoch % config.checkpoint_interval == 0
                {
                    let dir = format!("{}/epoch_{}", ckpt_dir, epoch);
                    let _ = std::fs::create_dir_all(&dir);
                }
            }
        }

        Ok(metrics)
    }
}

pub trait HasParameters {
    fn parameters(&mut self) -> Vec<NdParam<'_>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    struct DummyModel {
        params: Vec<Vec<f32>>,
    }

    impl HasParameters for DummyModel {
        fn parameters(&mut self) -> Vec<NdParam<'_>> {
            Vec::new()
        }
    }

    #[test]
    fn test_training_config_default() {
        let cfg = TrainingConfig::default();
        assert_eq!(cfg.learning_rate, 1e-3);
        assert_eq!(cfg.num_epochs, 100);
        assert_eq!(cfg.seed, 42);
    }

    #[test]
    fn test_metrics_tracking() {
        let mut metrics = TrainingMetrics::new();
        metrics.record(0, 1.0, 2.0, 0.5);
        metrics.record(1, 0.5, 1.0, 0.3);
        assert_eq!(metrics.best_epoch(), 1);
    }
}
