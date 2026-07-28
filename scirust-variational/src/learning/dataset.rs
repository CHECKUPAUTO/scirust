use scirust_core::nn::rng::PcgEngine;

use crate::error::{Result, VariationalError};

#[derive(Debug, Clone)]
pub struct TrajectorySample {
    pub q: Vec<f32>,
    pub dq: Vec<f32>,
    pub ddq: Vec<f32>,
    pub t: f32,
}

#[derive(Debug, Clone)]
pub struct TrajectoryDataset {
    pub samples: Vec<TrajectorySample>,
    pub ndim: usize,
    pub indices: Vec<usize>,
}

impl TrajectoryDataset {
    pub fn new(samples: Vec<TrajectorySample>) -> Result<Self> {
        if samples.is_empty()
        {
            return Err(VariationalError::TrainingFailure {
                details: "empty trajectory dataset".into(),
            });
        }
        let ndim = samples[0].q.len();
        for sample in &samples
        {
            if sample.q.len() != ndim || sample.dq.len() != ndim || sample.ddq.len() != ndim
            {
                return Err(VariationalError::DimensionMismatch {
                    expected: ndim,
                    got: sample.q.len(),
                    context: "TrajectoryDataset::new".into(),
                });
            }
        }
        let indices: Vec<usize> = (0..samples.len()).collect();
        Ok(Self {
            samples,
            ndim,
            indices,
        })
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn shuffle(&mut self, rng: &mut PcgEngine) {
        let n = self.indices.len();
        for i in (1..n).rev()
        {
            let j = (rng.float() * (i as f32 + 1.0)) as usize;
            self.indices.swap(i, j.min(i));
        }
    }

    pub fn get(&self, index: usize) -> &TrajectorySample {
        &self.samples[self.indices[index]]
    }

    pub fn split(mut self, train_frac: f32, seed: u64) -> (Self, Self) {
        let mut rng = PcgEngine::new(seed);
        self.shuffle(&mut rng);
        let split_point = (self.samples.len() as f32 * train_frac) as usize;
        let train_indices = self.indices[..split_point].to_vec();
        let val_indices = self.indices[split_point..].to_vec();

        let train = Self {
            samples: self.samples.clone(),
            ndim: self.ndim,
            indices: train_indices,
        };
        let val = Self {
            samples: self.samples,
            ndim: self.ndim,
            indices: val_indices,
        };
        (train, val)
    }
}

pub fn trajectory_from_ode<F>(
    dynamics: F,
    q0: &[f32],
    dq0: &[f32],
    t_span: &[f32],
    ndim: usize,
) -> Result<TrajectoryDataset>
where
    F: Fn(f32, &[f32], &mut [f32]),
{
    if t_span.len() < 2
    {
        return Err(VariationalError::InvalidInterval {
            start: t_span.first().copied().unwrap_or(0.0),
            end: t_span.last().copied().unwrap_or(0.0),
        });
    }

    let mut samples = Vec::with_capacity(t_span.len());
    let mut state: Vec<f32> = Vec::with_capacity(2 * ndim);
    state.extend_from_slice(q0);
    state.extend_from_slice(dq0);

    let dt = t_span[1] - t_span[0];
    let mut deriv = vec![0.0; 2 * ndim];

    for &t in t_span
    {
        let q: Vec<f32> = state[..ndim].to_vec();
        let dq: Vec<f32> = state[ndim..].to_vec();

        dynamics(t, &state, &mut deriv);
        let ddq: Vec<f32> = deriv[ndim..].to_vec();

        samples.push(TrajectorySample { q, dq, ddq, t });

        if (t - t_span[t_span.len() - 1]).abs() < 1e-10
        {
            break;
        }

        let mut k1 = vec![0.0; 2 * ndim];
        let mut k2 = vec![0.0; 2 * ndim];
        let mut k3 = vec![0.0; 2 * ndim];
        let mut k4 = vec![0.0; 2 * ndim];
        let mut tmp = vec![0.0; 2 * ndim];

        dynamics(t, &state, &mut k1);
        for i in 0..2 * ndim
        {
            tmp[i] = state[i] + 0.5 * dt * k1[i];
        }
        dynamics(t + 0.5 * dt, &tmp, &mut k2);
        for i in 0..2 * ndim
        {
            tmp[i] = state[i] + 0.5 * dt * k2[i];
        }
        dynamics(t + 0.5 * dt, &tmp, &mut k3);
        for i in 0..2 * ndim
        {
            tmp[i] = state[i] + dt * k3[i];
        }
        dynamics(t + dt, &tmp, &mut k4);
        let h = dt;
        for i in 0..2 * ndim
        {
            state[i] += (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
    }

    TrajectoryDataset::new(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn harmonic_dynamics(_t: f32, state: &[f32], deriv: &mut [f32]) {
        deriv[0] = state[1];
        deriv[1] = -state[0];
    }

    #[test]
    fn test_trajectory_dataset() {
        let t_span: Vec<f32> = (0..10).map(|i| i as f32 * 0.1).collect();
        let dataset = trajectory_from_ode(harmonic_dynamics, &[1.0], &[0.0], &t_span, 1).unwrap();
        assert_eq!(dataset.len(), 10);
        assert_eq!(dataset.ndim, 1);
    }

    #[test]
    fn test_train_val_split() {
        let t_span: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let dataset = trajectory_from_ode(harmonic_dynamics, &[1.0], &[0.0], &t_span, 1).unwrap();
        let n = dataset.len();
        let (train, val) = dataset.split(0.8, 42);
        assert_eq!(train.len() + val.len(), n);
        assert!(!train.is_empty());
        assert!(!val.is_empty());
    }
}
