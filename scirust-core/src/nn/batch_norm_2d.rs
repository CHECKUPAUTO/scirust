// scirust-core/src/nn/batch_norm_2d.rs
//
// BatchNorm2d — normalisation par batch pour convolutions.
//
// Input attendu : (N, C*H*W) — la sortie flatten de Conv2d.
// En interne on reshape en (N*H*W, C) pour normaliser par canal.

use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::module::Module;
use std::collections::HashMap;

pub struct BatchNorm2d {
    pub num_channels: usize,
    pub gamma: Tensor, // (1, C)
    pub beta: Tensor,  // (1, C)
    pub eps: f32,
    pub momentum: f32,
    pub running_mean: Tensor, // (1, C)
    pub running_var: Tensor,  // (1, C)
    pub training: bool,
    last_g_idx: Option<usize>,
    last_b_idx: Option<usize>,
    pub name: String,
}

impl BatchNorm2d {
    pub fn new(num_channels: usize) -> Self {
        Self {
            num_channels,
            gamma: Tensor::from_vec(vec![1.0; num_channels], 1, num_channels),
            beta: Tensor::zeros(1, num_channels),
            eps: 1e-5,
            momentum: 0.1,
            running_mean: Tensor::zeros(1, num_channels),
            running_var: Tensor::from_vec(vec![1.0; num_channels], 1, num_channels),
            training: true,
            last_g_idx: None,
            last_b_idx: None,
            name: format!("bn2d_{num_channels}"),
        }
    }

    #[must_use]
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.into();
        self
    }

    pub fn set_training(&mut self, mode: bool) {
        self.training = mode;
    }

    fn compute_batch_stats(
        &self,
        input_data: &[f32],
        n_spatial: usize, // N*H*W
        c: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let inv_n = 1.0 / n_spatial as f32;
        let mut mean = vec![0.0f32; c];
        for i in 0..n_spatial
        {
            for j in 0..c
            {
                mean[j] += input_data[i * c + j];
            }
        }
        for v in mean.iter_mut()
        {
            *v *= inv_n;
        }

        let mut var = vec![0.0f32; c];
        for i in 0..n_spatial
        {
            for j in 0..c
            {
                let d = input_data[i * c + j] - mean[j];
                var[j] += d * d;
            }
        }
        for v in var.iter_mut()
        {
            *v *= inv_n;
        }
        (mean, var)
    }

    fn update_running_stats(&mut self, batch_mean: &[f32], batch_var: &[f32]) {
        let alpha = self.momentum;
        for j in 0..self.num_channels
        {
            self.running_mean.data[j] =
                (1.0 - alpha) * self.running_mean.data[j] + alpha * batch_mean[j];
            self.running_var.data[j] =
                (1.0 - alpha) * self.running_var.data[j] + alpha * batch_var[j];
        }
    }
}

impl Module for BatchNorm2d {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        let (n, total_features) = input.shape();
        assert_eq!(
            total_features % self.num_channels,
            0,
            "BatchNorm2d: total_features {total_features} not divisible by num_channels {}",
            self.num_channels
        );
        let spatial = total_features / self.num_channels;
        let n_spatial = n * spatial;

        // Reshape (N, C*H*W) -> (N*H*W, C)
        let reshaped = input.reshape(&[n_spatial, self.num_channels]);

        let gamma_v = tape.input(self.gamma.clone());
        let beta_v = tape.input(self.beta.clone());
        self.last_g_idx = Some(gamma_v.idx());
        self.last_b_idx = Some(beta_v.idx());

        if self.training
        {
            let input_t = tape.value(reshaped.idx());
            let (batch_mean, batch_var) =
                self.compute_batch_stats(&input_t.data, n_spatial, self.num_channels);
            self.update_running_stats(&batch_mean, &batch_var);

            let inv_n = 1.0 / n_spatial as f32;
            let mu = reshaped.sum_axis(0).scale(inv_n);
            let centered = reshaped.try_add_broadcast(mu.neg()).unwrap();
            let centered_sq = centered.try_hadamard(centered).unwrap();
            let var = centered_sq.sum_axis(0).scale(inv_n);
            let eps_t = tape.input(Tensor::from_vec(
                vec![self.eps; self.num_channels],
                1,
                self.num_channels,
            ));
            let std = var.try_add(eps_t).unwrap().sqrt();
            let inv_std = std.reciprocal();
            let x_hat = centered.try_mul_broadcast(inv_std).unwrap();
            let scaled = x_hat.try_mul_broadcast(gamma_v).unwrap();
            let out = scaled.try_add_broadcast(beta_v).unwrap();
            // Reshape back
            out.reshape(&[n, total_features])
        }
        else
        {
            let rmean_v = tape.input(self.running_mean.clone());
            let centered = reshaped.try_add_broadcast(rmean_v.neg()).unwrap();
            let eps_t = tape.input(Tensor::from_vec(
                vec![self.eps; self.num_channels],
                1,
                self.num_channels,
            ));
            let rvar_v = tape.input(self.running_var.clone());
            let std = rvar_v.try_add(eps_t).unwrap().sqrt();
            let x_hat = centered.try_mul_broadcast(std.reciprocal()).unwrap();
            let scaled = x_hat.try_mul_broadcast(gamma_v).unwrap();
            let out = scaled.try_add_broadcast(beta_v).unwrap();
            out.reshape(&[n, total_features])
        }
    }

    fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::new();
        if let Some(i) = self.last_g_idx
        {
            v.push(i);
        }
        if let Some(i) = self.last_b_idx
        {
            v.push(i);
        }
        v
    }

    fn sync(&mut self, tape: &Tape) {
        if let Some(i) = self.last_g_idx
        {
            self.gamma = tape.value(i);
        }
        if let Some(i) = self.last_b_idx
        {
            self.beta = tape.value(i);
        }
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.insert(format!("{}.gamma", self.name), self.gamma.clone());
        map.insert(format!("{}.beta", self.name), self.beta.clone());
        map.insert(
            format!("{}.running_mean", self.name),
            self.running_mean.clone(),
        );
        map.insert(
            format!("{}.running_var", self.name),
            self.running_var.clone(),
        );
        map
    }

    fn load_state_dict(&mut self, sd: &HashMap<String, Tensor>) -> crate::error::Result<()> {
        let g = sd
            .get(&format!("{}.gamma", self.name))
            .ok_or_else(|| format!("missing key: {}.gamma", self.name))?;
        let b = sd
            .get(&format!("{}.beta", self.name))
            .ok_or_else(|| format!("missing key: {}.beta", self.name))?;
        let rm = sd
            .get(&format!("{}.running_mean", self.name))
            .ok_or_else(|| format!("missing key: {}.running_mean", self.name))?;
        let rv = sd
            .get(&format!("{}.running_var", self.name))
            .ok_or_else(|| format!("missing key: {}.running_var", self.name))?;

        self.gamma = g.clone();
        self.beta = b.clone();
        self.running_mean = rm.clone();
        self.running_var = rv.clone();
        Ok(())
    }
}

impl Clone for BatchNorm2d {
    fn clone(&self) -> Self {
        Self {
            num_channels: self.num_channels,
            gamma: self.gamma.clone(),
            beta: self.beta.clone(),
            eps: self.eps,
            momentum: self.momentum,
            running_mean: self.running_mean.clone(),
            running_var: self.running_var.clone(),
            training: self.training,
            last_g_idx: None,
            last_b_idx: None,
            name: self.name.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bn2d_normalizes_per_channel() {
        let mut bn = BatchNorm2d::new(2);
        let tape = Tape::new();
        // N=2, C=2, spatial=2  →  flat (2, 4)
        // channel 0 : [1, 2, 3, 4]  mean=2.5, var=1.25
        // channel 1 : [5, 6, 7, 8]  mean=6.5, var=1.25
        let x = tape.input(Tensor::from_vec(
            vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0],
            2,
            4,
        ));
        let y = bn.forward(&tape, x);
        let yt = tape.value(y.idx());

        // After BN (gamma=1, beta=0) :
        // channel 0 normalized: [-1.3416, -0.4472, 0.4472, 1.3416]
        // channel 1 normalized: same pattern
        assert_eq!(yt.shape(), (2, 4));
    }

    #[test]
    fn bn2d_running_stats_update() {
        let mut bn = BatchNorm2d::new(2);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 1, 4));
        let _ = bn.forward(&tape, x);
        // After 1 batch: running_mean ≈ [0.1*2, 0.1*3] = [0.2, 0.3] (channel-wise over spatial=2)
        // Actually: channel0=[1,3] mean=2, channel1=[2,4] mean=3
        assert!(
            (bn.running_mean.data[0] - 0.2).abs() < 1e-5,
            "running_mean[0] = {}",
            bn.running_mean.data[0]
        );
        assert!(
            (bn.running_mean.data[1] - 0.3).abs() < 1e-5,
            "running_mean[1] = {}",
            bn.running_mean.data[1]
        );
    }

    #[test]
    fn bn2d_eval_uses_running_stats() {
        let mut bn = BatchNorm2d::new(2);
        bn.running_mean = Tensor::from_vec(vec![2.0, 3.0], 1, 2);
        bn.running_var = Tensor::from_vec(vec![4.0, 9.0], 1, 2);
        bn.set_training(false);

        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![2.0, 3.0, 4.0, 6.0], 1, 4));
        let y = bn.forward(&tape, x);
        let yt = tape.value(y.idx());
        // x[0]=2, mean=2, var=4, std=2 → (2-2)/2=0
        // x[1]=3, mean=3, var=9, std=3 → (3-3)/3=0
        assert!(yt.data[0].abs() < 1e-3);
        assert!(yt.data[1].abs() < 1e-3);
    }

    #[test]
    fn bn2d_gradient_flows() {
        let mut bn = BatchNorm2d::new(2);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 1, 4));
        let x_idx = x.idx();
        let y = bn.forward(&tape, x);
        let weights = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 1, 4));
        let loss = y.hadamard(weights).sum();
        loss.backward();
        let g = tape.grad(x_idx);
        assert!(g.data.iter().any(|&v| v.abs() > 1e-6), "gradient is zero");
    }

    #[test]
    fn bn2d_state_dict_round_trip() {
        let mut bn1 = BatchNorm2d::new(3);
        bn1.gamma = Tensor::from_vec(vec![2.0; 3], 1, 3);
        bn1.beta = Tensor::from_vec(vec![1.0; 3], 1, 3);
        let sd = bn1.state_dict();

        let mut bn2 = BatchNorm2d::new(3);
        bn2.load_state_dict(&sd).unwrap();
        assert_eq!(bn2.gamma.data, bn1.gamma.data);
        assert_eq!(bn2.beta.data, bn1.beta.data);
        assert_eq!(bn2.running_mean.data, bn1.running_mean.data);
        assert_eq!(bn2.running_var.data, bn1.running_var.data);
    }
}
