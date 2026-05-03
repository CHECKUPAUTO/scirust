// scirust-core/src/nn/batch_norm.rs
//
// BatchNorm1d — normalisation par batch de features.
//
// En training : calcule batch_mean/batch_var depuis le tape AD pour le
// gradient, ET met à jour running_mean/running_var (effet de bord
// impératif sur le module, hors du graphe AD).
// En eval : utilise running_mean/running_var avec graphe AD minimal.

use std::collections::HashMap;
use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::module::Module;

pub struct BatchNorm1d {
    pub gamma:        Tensor,
    pub beta:         Tensor,
    pub eps:          f32,
    pub momentum:     f32,
    pub running_mean: Tensor,
    pub running_var:  Tensor,
    pub training:     bool,
    last_g_idx:       Option<usize>,
    last_b_idx:       Option<usize>,
    pub name:         String,
}

impl BatchNorm1d {
    pub fn new(num_features: usize) -> Self {
        Self {
            gamma:        Tensor::from_vec(vec![1.0; num_features], 1, num_features),
            beta:         Tensor::zeros(1, num_features),
            eps:          1e-5,
            momentum:     0.1,
            running_mean: Tensor::zeros(1, num_features),
            running_var:  Tensor::from_vec(vec![1.0; num_features], 1, num_features),
            training:     true,
            last_g_idx:   None,
            last_b_idx:   None,
            name:         format!("bn_{num_features}"),
        }
    }

    pub fn with_name(mut self, name: &str) -> Self { self.name = name.into(); self }

    pub fn set_training(&mut self, mode: bool) { self.training = mode; }

    fn compute_batch_stats(&self, input_data: &[f32], n: usize, f: usize)
        -> (Vec<f32>, Vec<f32>)
    {
        let inv_n = 1.0 / n as f32;
        let mut mean = vec![0.0f32; f];
        for i in 0..n {
            for j in 0..f { mean[j] += input_data[i * f + j]; }
        }
        for v in mean.iter_mut() { *v *= inv_n; }

        let mut var = vec![0.0f32; f];
        for i in 0..n {
            for j in 0..f {
                let d = input_data[i * f + j] - mean[j];
                var[j] += d * d;
            }
        }
        for v in var.iter_mut() { *v *= inv_n; }
        (mean, var)
    }

    fn update_running_stats(&mut self, batch_mean: &[f32], batch_var: &[f32]) {
        let alpha = self.momentum;
        for j in 0..self.running_mean.cols {
            self.running_mean.data[j] =
                (1.0 - alpha) * self.running_mean.data[j] + alpha * batch_mean[j];
            self.running_var.data[j] =
                (1.0 - alpha) * self.running_var.data[j] + alpha * batch_var[j];
        }
    }
}

impl Module for BatchNorm1d {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        let (n, f) = input.shape();
        let inv_n = 1.0 / n as f32;

        let gamma_v = tape.input(self.gamma.clone());
        let beta_v  = tape.input(self.beta.clone());
        self.last_g_idx = Some(gamma_v.idx());
        self.last_b_idx = Some(beta_v.idx());

        if self.training {
            let input_t = tape.value(input.idx());
            let (batch_mean, batch_var) = self.compute_batch_stats(&input_t.data, n, f);
            self.update_running_stats(&batch_mean, &batch_var);

            let mu = input.clone().sum_axis(0).scale(inv_n);
            let mu_neg = mu.neg();
            let centered = input.add_broadcast(mu_neg);
            let centered_sq = centered.clone().hadamard(centered.clone());
            let var = centered_sq.sum_axis(0).scale(inv_n);
            let eps_t = tape.input(Tensor::from_vec(vec![self.eps; f], 1, f));
            let std = var.add(eps_t).sqrt();
            let inv_std = std.reciprocal();
            let x_hat = centered.mul_broadcast(inv_std);
            let scaled = x_hat.mul_broadcast(gamma_v);
            scaled.add_broadcast(beta_v)
        } else {
            let rmean_v = tape.input(self.running_mean.clone());
            let rvar_v  = tape.input(self.running_var.clone());
            let centered = input.add_broadcast(rmean_v.neg());
            let eps_t = tape.input(Tensor::from_vec(vec![self.eps; f], 1, f));
            let std = rvar_v.add(eps_t).sqrt();
            let x_hat = centered.mul_broadcast(std.reciprocal());
            let scaled = x_hat.mul_broadcast(gamma_v);
            scaled.add_broadcast(beta_v)
        }
    }

    fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::new();
        if let Some(i) = self.last_g_idx { v.push(i); }
        if let Some(i) = self.last_b_idx { v.push(i); }
        v
    }

    fn sync(&mut self, tape: &Tape) {
        if let Some(i) = self.last_g_idx { self.gamma = tape.value(i); }
        if let Some(i) = self.last_b_idx { self.beta  = tape.value(i); }
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.insert(format!("{}.gamma", self.name), self.gamma.clone());
        map.insert(format!("{}.beta", self.name), self.beta.clone());
        map.insert(format!("{}.running_mean", self.name), self.running_mean.clone());
        map.insert(format!("{}.running_var", self.name), self.running_var.clone());
        map
    }

    fn load_state_dict(&mut self, sd: &HashMap<String, Tensor>) -> std::result::Result<(), String> {
        let g = sd.get(&format!("{}.gamma", self.name))
            .ok_or_else(|| format!("missing key: {}.gamma", self.name))?;
        let b = sd.get(&format!("{}.beta", self.name))
            .ok_or_else(|| format!("missing key: {}.beta", self.name))?;
        let rm = sd.get(&format!("{}.running_mean", self.name))
            .ok_or_else(|| format!("missing key: {}.running_mean", self.name))?;
        let rv = sd.get(&format!("{}.running_var", self.name))
            .ok_or_else(|| format!("missing key: {}.running_var", self.name))?;

        if g.shape() != (1, self.gamma.cols) {
            return Err(format!("gamma shape mismatch"));
        }
        self.gamma = g.clone();
        self.beta = b.clone();
        self.running_mean = rm.clone();
        self.running_var = rv.clone();
        Ok(())
    }
}

impl Clone for BatchNorm1d {
    fn clone(&self) -> Self {
        Self {
            gamma: self.gamma.clone(), beta: self.beta.clone(),
            eps: self.eps, momentum: self.momentum,
            running_mean: self.running_mean.clone(),
            running_var:  self.running_var.clone(),
            training: self.training,
            last_g_idx: None, last_b_idx: None,
            name: self.name.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_to_zero_mean_unit_var() {
        let mut bn = BatchNorm1d::new(3);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(
            vec![1.0, 10.0, -5.0,  3.0, 12.0, -3.0,
                 5.0, 14.0, -1.0,  7.0, 16.0,  1.0], 4, 3));
        let y = bn.forward(&tape, x);
        let yt = tape.value(y.idx());
        for j in 0..3 {
            let mean: f32 = (0..4).map(|i| yt.data[i*3 + j]).sum::<f32>() / 4.0;
            assert!(mean.abs() < 1e-4, "col {j} mean = {mean}");
        }
    }

    #[test]
    fn running_stats_update_on_training() {
        let mut bn = BatchNorm1d::new(2);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![10.0, 20.0,  10.0, 20.0], 2, 2));
        let _ = bn.forward(&tape, x);
        assert!((bn.running_mean.data[0] - 1.0).abs() < 1e-5);
        assert!((bn.running_mean.data[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn eval_uses_running_stats() {
        let mut bn = BatchNorm1d::new(2);
        bn.running_mean = Tensor::from_vec(vec![5.0, 10.0], 1, 2);
        bn.running_var  = Tensor::from_vec(vec![4.0, 9.0],  1, 2);
        bn.set_training(false);

        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![5.0, 10.0,  7.0, 13.0], 2, 2));
        let y = bn.forward(&tape, x);
        let yt = tape.value(y.idx());
        assert!(yt.data[0].abs() < 1e-3);
        assert!(yt.data[1].abs() < 1e-3);
    }

    #[test]
    fn state_dict_has_4_entries() {
        let bn = BatchNorm1d::new(8);
        let sd = bn.state_dict();
        assert_eq!(sd.len(), 4);
    }
}
